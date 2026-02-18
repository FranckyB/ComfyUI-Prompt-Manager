"""
ComfyUI Prompt Extractor - Extract prompts and LoRAs from images, videos, and workflow files
Reads metadata from ComfyUI-generated files to extract workflow information
"""
import os
import json
import re
import folder_paths
import torch
import server

# Import PIL for image metadata reading
try:
    import numpy as np
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False
    print("[PromptExtractor] Warning: PIL/numpy not available, image metadata reading disabled")


# Cache for file metadata (read by JavaScript, used by Python)
_file_metadata_cache = {}
# Cache for video frames extracted by JavaScript
_video_frames_cache = {}

# LoRA Blacklist - LoRAs containing these keywords will be excluded from extraction
# Add keywords that identify non-style LoRAs (e.g., distillation, optimization LoRAs)
LORA_BLACKLIST = [
    'lightx2v',                 # Distillation LoRAs
    ['4steps', 'seko'],         # Fast sampling LoRAs - requires BOTH keywords
    ['4steps', 'lightning']]    # Fast sampling LoRAs - requires BOTH keywords

def is_lora_blacklisted(lora_name):
    """Check if a LoRA name contains any blacklisted keywords (case-insensitive)"""
    if not lora_name:
        return False
    lora_name_lower = lora_name.lower()
    for keyword in LORA_BLACKLIST:
        if isinstance(keyword, list):
            # We check if all keywords in the list are present in the lora_name for a match
            if all(k.lower() in lora_name_lower for k in keyword):
                return True
        else:
            if keyword.lower() in lora_name_lower:
                return True
    return False


def parse_a1111_parameters(parameters_text):
    """
    Parse A1111/Forge parameters format
    Returns dict with prompt, negative_prompt, and loras
    """
    if not parameters_text:
        return None

    result = {
        'prompt': '',
        'negative_prompt': '',
        'loras': []
    }

    # Split by "Negative prompt:" to separate positive and negative
    parts = re.split(r'Negative prompt:\s*', parameters_text, flags=re.IGNORECASE)
    positive_prompt = parts[0].strip()
    remainder = parts[1] if len(parts) > 1 else ''

    # Extract LoRAs using pattern: <lora:name:strength> or <lora:name:model_strength:clip_strength>
    lora_pattern = r'<lora:([^:>]+):([^:>]+)(?::([^:>]+))?>'
    loras = []

    for match in re.finditer(lora_pattern, positive_prompt):
        lora_name = match.group(1).strip()
        strength1 = float(match.group(2))
        strength2 = float(match.group(3)) if match.group(3) else strength1

        loras.append({
            'name': lora_name,
            'model_strength': strength1,
            'clip_strength': strength2,
            'active': True
        })

    # Remove LoRA tags from prompt
    positive_prompt = re.sub(lora_pattern, '', positive_prompt).strip()
    result['prompt'] = positive_prompt
    result['loras'] = loras

    # Extract negative prompt (before any "Steps:" line if present)
    settings_match = re.match(r'^(.*?)[\r\n]+Steps:', remainder, re.DOTALL)
    if settings_match:
        result['negative_prompt'] = settings_match.group(1).strip()
    else:
        result['negative_prompt'] = remainder.strip()

    return result


# API endpoint to cache file metadata (sent from JavaScript)
@server.PromptServer.instance.routes.post("/prompt-extractor/cache-file-metadata")
async def cache_file_metadata(request):
    """API endpoint to cache file metadata read by JavaScript"""
    try:
        data = await request.json()
        filename = data.get('filename')
        metadata = data.get('metadata')

        if not filename:
            return server.web.json_response({"success": False, "error": "Missing filename"}, status=400)

        if metadata:
            # Use filename as-is (with forward slashes) as cache key
            cache_key = filename.replace('\\', '/')
            _file_metadata_cache[cache_key] = metadata
            print(f"[PromptExtractor] Cached metadata for: {cache_key}")

        return server.web.json_response({"success": True})
    except Exception as e:
        print(f"[PromptExtractor] Error caching file metadata: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


# API endpoint to cache video frame (sent from JavaScript)
@server.PromptServer.instance.routes.post("/prompt-extractor/cache-video-frame")
async def cache_video_frame(request):
    """API endpoint to cache a single video frame extracted by JavaScript"""
    try:
        data = await request.json()
        filename = data.get('filename')
        frame = data.get('frame')  # Single base64 data URL
        frame_position = data.get('frame_position', 0.0)
        # Handle None or null from JavaScript
        if frame_position is None:
            frame_position = 0.0

        if not filename:
            return server.web.json_response({"success": False, "error": "Missing filename"}, status=400)

        if frame:
            # Normalize path separators and replace for cache key (matching JavaScript)
            # Note: We don't include frame_position in the cache key because the video
            # frame changes when the user adjusts the slider, and we only cache one frame per video
            path_key = filename.replace('\\', '/').replace('/', '_')
            _video_frames_cache[path_key] = frame

        return server.web.json_response({"success": True})
    except Exception as e:
        print(f"[PromptExtractor] Error caching video frame: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


# API endpoint to extract video metadata using ffprobe (fallback when JavaScript can't parse)
@server.PromptServer.instance.routes.post("/prompt-extractor/extract-video-metadata")
async def extract_video_metadata_api(request):
    """API endpoint to extract video metadata using ffprobe when JavaScript parser fails"""
    try:
        data = await request.json()
        filename = data.get('filename')

        print(f"[PromptExtractor] JavaScript requested ffprobe extraction for: {filename}")

        if not filename:
            return server.web.json_response({"success": False, "error": "Missing filename"}, status=400)

        # Build full path
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, filename.replace('/', os.sep))

        if not os.path.exists(file_path):
            print(f"[PromptExtractor] File not found: {file_path}")
            return server.web.json_response({"success": False, "error": "File not found"}, status=404)

        # Try extracting with ffprobe
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', file_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                ffprobe_data = json.loads(result.stdout)
                if 'format' in ffprobe_data and 'tags' in ffprobe_data['format']:
                    tags = ffprobe_data['format']['tags']

                    metadata = {}

                    # Extract prompt if present
                    if 'prompt' in tags:
                        try:
                            metadata['prompt'] = json.loads(tags['prompt'])
                        except:
                            metadata['prompt'] = tags['prompt']

                    # Extract workflow if present
                    if 'workflow' in tags:
                        try:
                            metadata['workflow'] = json.loads(tags['workflow'])
                        except:
                            metadata['workflow'] = tags['workflow']

                    if metadata:
                        # Sanitize metadata to replace NaN with null for valid JSON
                        def sanitize_json(obj):
                            if isinstance(obj, dict):
                                return {k: sanitize_json(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [sanitize_json(item) for item in obj]
                            elif isinstance(obj, float) and (obj != obj):  # NaN check
                                return None
                            elif obj == float('inf') or obj == float('-inf'):
                                return None
                            else:
                                return obj

                        metadata = sanitize_json(metadata)

                        # Cache it
                        cache_key = filename.replace('\\', '/')
                        _file_metadata_cache[cache_key] = metadata
                        return server.web.json_response({"success": True, "metadata": metadata})

            return server.web.json_response({"success": False, "error": "No metadata found"}, status=404)

        except FileNotFoundError:
            print("[PromptExtractor] WARNING: ffprobe not found. Video metadata fallback unavailable.")
            print("[PromptExtractor] Please install FFmpeg to enable video metadata extraction for all formats.")
            return server.web.json_response({"success": False, "error": "ffprobe_not_found", "warning": True}, status=200)
        except Exception as e:
            print(f"[PromptExtractor] ffprobe extraction error: {e}")
            return server.web.json_response({"success": False, "error": f"ffprobe error: {str(e)}"}, status=500)

    except Exception as e:
        print(f"[PromptExtractor] Error in extract-video-metadata API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


# API endpoint to list files in input directory
@server.PromptServer.instance.routes.get("/prompt-extractor/list-files")
async def list_input_files(request):
    """API endpoint to get list of supported files in input directory, including subfolders"""
    try:
        input_dir = folder_paths.get_input_directory()
        files = []
        supported_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.json', '.mp4', '.webm', '.mov', '.avi']

        if os.path.exists(input_dir):
            # Walk through directory recursively
            for root, dirs, filenames in os.walk(input_dir):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in supported_extensions:
                        # Get relative path from input directory
                        full_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(full_path, input_dir)
                        # Convert Windows backslashes to forward slashes
                        rel_path = rel_path.replace('\\', '/')
                        files.append(rel_path)

        files.sort()
        return server.web.json_response({"files": files})
    except Exception as e:
        print(f"[PromptExtractor] Error listing files: {e}")
        return server.web.json_response({"files": [], "error": str(e)}, status=500)


def get_available_loras():
    """Get all available LoRAs from ComfyUI's folder system"""
    return folder_paths.get_filename_list("loras")


def resolve_lora_path(lora_name):
    """
    Resolve a LoRA name to its full path using ComfyUI's folder system.
    Returns (full_path, found) tuple.
    """
    lora_files = get_available_loras()

    # Try exact match first (with extension)
    for lora_file in lora_files:
        if lora_file == lora_name:
            return folder_paths.get_full_path("loras", lora_file), True

    # Try matching by name without extension
    lora_name_lower = lora_name.lower()
    for lora_file in lora_files:
        file_name_no_ext = os.path.splitext(os.path.basename(lora_file))[0]
        if file_name_no_ext.lower() == lora_name_lower:
            return folder_paths.get_full_path("loras", lora_file), True

    # Try partial match (lora_name might be just the filename, lora_file might have subdirs)
    for lora_file in lora_files:
        if lora_name_lower in lora_file.lower():
            return folder_paths.get_full_path("loras", lora_file), True

    return None, False


def extract_metadata_from_png(file_path):
    """Extract workflow/prompt metadata from PNG file (cached from JavaScript)"""
    try:
        # Try to get relative path from input directory first (matches JavaScript cache keys)
        input_dir = folder_paths.get_input_directory()
        if file_path.startswith(input_dir):
            cache_key = os.path.relpath(file_path, input_dir).replace('\\', '/')
        else:
            cache_key = os.path.basename(file_path)

        # Check if metadata was cached by JavaScript
        if cache_key in _file_metadata_cache:
            metadata = _file_metadata_cache[cache_key]
            print(f"[PromptExtractor] Using cached PNG metadata for: {cache_key}")

            if isinstance(metadata, dict):
                prompt_data = metadata.get('prompt')
                workflow_data = metadata.get('workflow')

                # Check if we have parsed A1111 parameters
                if metadata.get('parsed_parameters'):
                    print("[PromptExtractor] Found parsed A1111 parameters")
                    return metadata.get('parsed_parameters'), None

                return prompt_data, workflow_data

        # Fallback to PIL if no cached data (backwards compatibility)
        if not IMAGE_SUPPORT:
            return None, None

        print(f"[PromptExtractor] Falling back to PIL for: {filename}")
        with Image.open(file_path) as img:
            metadata = img.info

            # Debug: Print all metadata keys found
            print(f"[PromptExtractor] PNG metadata keys: {list(metadata.keys())}")

            # ComfyUI stores data in 'prompt' and 'workflow' text chunks
            prompt_data = metadata.get('prompt')
            workflow_data = metadata.get('workflow')

            # Also check for alternative key names (some tools use different names)
            if not prompt_data:
                prompt_data = metadata.get('Prompt') or metadata.get('parameters') or metadata.get('Comment')
            if not workflow_data:
                workflow_data = metadata.get('Workflow')

            # Debug: Show if we found anything
            print(f"[PromptExtractor] prompt_data found: {prompt_data is not None}, workflow_data found: {workflow_data is not None}")
            if prompt_data:
                print(f"[PromptExtractor] prompt_data preview: {str(prompt_data)[:200]}...")
            if workflow_data:
                print(f"[PromptExtractor] workflow_data preview: {str(workflow_data)[:200]}...")

            # Parse JSON if present
            prompt_json = None
            workflow_json = None

            if prompt_data:
                try:
                    prompt_json = json.loads(prompt_data) if isinstance(prompt_data, str) else prompt_data
                except json.JSONDecodeError as e:
                    print(f"[PromptExtractor] Failed to parse prompt JSON: {e}")
                    # Check if it's A1111 parameters format
                    if isinstance(prompt_data, str) and ('Negative prompt:' in prompt_data or '<lora:' in prompt_data):
                        print("[PromptExtractor] Detected A1111 parameters format, parsing...")
                        parsed = parse_a1111_parameters(prompt_data)
                        if parsed:
                            prompt_json = parsed
                            print(f"[PromptExtractor] Parsed {len(parsed.get('loras', []))} LoRAs from A1111 parameters")
                    else:
                        # Plain text prompt (fallback)
                        prompt_json = {'positive': prompt_data}

            if workflow_data:
                try:
                    workflow_json = json.loads(workflow_data) if isinstance(workflow_data, str) else workflow_data
                except json.JSONDecodeError as e:
                    print(f"[PromptExtractor] Failed to parse workflow JSON: {e}")

            return prompt_json, workflow_json
    except Exception as e:
        print(f"[PromptExtractor] Error reading PNG metadata: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def extract_metadata_from_jpeg(file_path):
    """Extract workflow/prompt metadata from JPEG/WebP file (cached from JavaScript)"""
    try:
        # Try to get relative path from input directory first (matches JavaScript cache keys)
        input_dir = folder_paths.get_input_directory()
        if file_path.startswith(input_dir):
            cache_key = os.path.relpath(file_path, input_dir).replace('\\', '/')
        else:
            cache_key = os.path.basename(file_path)

        # Check if metadata was cached by JavaScript
        if cache_key in _file_metadata_cache:
            metadata = _file_metadata_cache[cache_key]
            print(f"[PromptExtractor] Using cached JPEG/WebP metadata for: {cache_key}")

            if isinstance(metadata, dict):
                # Check for prompt/workflow structure
                if 'prompt' in metadata and 'workflow' in metadata:
                    return metadata.get('prompt'), metadata.get('workflow')
                elif 'workflow' in metadata:
                    return None, metadata.get('workflow')
                else:
                    return metadata, None
        else:
            print(f"[PromptExtractor] No cached metadata found for JPEG/WebP: {filename}")
            print("[PromptExtractor] Note: Image metadata is read by JavaScript when file is selected")

        # Fallback to PIL if no cached data (backwards compatibility)
        if not IMAGE_SUPPORT:
            return None, None

        print(f"[PromptExtractor] Falling back to PIL for: {filename}")
        with Image.open(file_path) as img:
            # Try EXIF data
            exif = img.getexif()
            if exif:
                # UserComment field (0x9286)
                user_comment = exif.get(0x9286)
                if user_comment:
                    # Try to parse as JSON
                    try:
                        if isinstance(user_comment, bytes):
                            user_comment = user_comment.decode('utf-8', errors='ignore')
                        # Remove potential UNICODE prefix
                        if user_comment.startswith('UNICODE'):
                            user_comment = user_comment[7:].lstrip('\x00')
                        data = json.loads(user_comment)
                        return data.get('prompt'), data.get('workflow')
                    except:
                        pass

            # Try ImageDescription
            if hasattr(img, 'info'):
                for key in ['prompt', 'workflow', 'parameters', 'Comment']:
                    if key in img.info:
                        try:
                            data = json.loads(img.info[key])
                            if isinstance(data, dict):
                                return data, None
                        except:
                            pass

            return None, None
    except Exception as e:
        print(f"[PromptExtractor] Error reading JPEG/WebP metadata: {e}")
        return None, None


def extract_metadata_from_json(file_path):
    """Extract workflow data from JSON file (cached from JavaScript)"""
    try:
        # Try to get relative path from input directory first (matches JavaScript cache keys)
        input_dir = folder_paths.get_input_directory()
        if file_path.startswith(input_dir):
            cache_key = os.path.relpath(file_path, input_dir).replace('\\', '/')
        else:
            cache_key = os.path.basename(file_path)

        # Check if metadata was cached by JavaScript
        if cache_key in _file_metadata_cache:
            data = _file_metadata_cache[cache_key]
            print(f"[PromptExtractor] Using cached JSON metadata for: {cache_key}")
        else:
            print(f"[PromptExtractor] No cached metadata found for JSON: {cache_key}")
            print("[PromptExtractor] Falling back to file read")
            # Fallback to reading file (backwards compatibility)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        print(f"[PromptExtractor] JSON loaded, type: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")

        # Check if it's a workflow format (has nodes) or prompt format
        if isinstance(data, dict):
            # API format (prompt) - node_id: {class_type, inputs}
            if any(isinstance(v, dict) and 'class_type' in v for v in data.values()):
                print("[PromptExtractor] JSON detected as API/prompt format")
                return data, None
            # Workflow format - has 'nodes' array
            if 'nodes' in data:
                print(f"[PromptExtractor] JSON detected as workflow format with {len(data.get('nodes', []))} nodes")
                return None, data
            # Could be wrapped
            if 'prompt' in data:
                print("[PromptExtractor] JSON detected as wrapped format")
                return data.get('prompt'), data.get('workflow')

        print("[PromptExtractor] JSON format not recognized, returning as-is")
        return data, None
    except Exception as e:
        print(f"[PromptExtractor] Error reading JSON file: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def extract_metadata_from_video(file_path):
    """Extract workflow/prompt metadata from video file (cached from JavaScript)"""
    try:
        # Get the relative path from input directory to match JavaScript cache keys
        input_dir = folder_paths.get_input_directory()
        if file_path.startswith(input_dir):
            # Normalize path separators to forward slashes
            cache_key = os.path.relpath(file_path, input_dir).replace('\\', '/')
        else:
            # Fallback to basename for absolute paths outside input dir
            cache_key = os.path.basename(file_path)

        # Check if metadata was cached by JavaScript
        if cache_key in _file_metadata_cache:
            metadata = _file_metadata_cache[cache_key]

            # Parse the cached metadata
            if isinstance(metadata, dict):
                # Check various structures
                if 'prompt' in metadata and 'workflow' in metadata:
                    prompt_val = metadata.get('prompt')
                    workflow_val = metadata.get('workflow')
                    # Handle nested JSON strings
                    if isinstance(prompt_val, str):
                        try:
                            prompt_val = json.loads(prompt_val)
                        except:
                            pass
                    if isinstance(workflow_val, str):
                        try:
                            workflow_val = json.loads(workflow_val)
                        except:
                            pass
                    return prompt_val, workflow_val
                elif 'workflow' in metadata:
                    workflow_val = metadata.get('workflow')
                    if isinstance(workflow_val, str):
                        try:
                            workflow_val = json.loads(workflow_val)
                        except:
                            pass
                    return None, workflow_val
                elif 'positive' in metadata or 'negative' in metadata:
                    return metadata, None
                # Check if metadata itself is the workflow
                elif 'nodes' in metadata or 'last_node_id' in metadata:
                    return None, metadata
                else:
                    return metadata, None
        else:
            print(f"[PromptExtractor] No cached metadata found for video: {cache_key}")
            print("[PromptExtractor] Attempting ffprobe extraction as fallback...")

            # Try extracting with ffprobe as fallback
            try:
                import subprocess
                result = subprocess.run(
                    ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', file_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    ffprobe_data = json.loads(result.stdout)
                    if 'format' in ffprobe_data and 'tags' in ffprobe_data['format']:
                        tags = ffprobe_data['format']['tags']

                        prompt_val = None
                        workflow_val = None

                        # Extract prompt if present
                        if 'prompt' in tags:
                            try:
                                prompt_val = json.loads(tags['prompt'])
                            except:
                                prompt_val = tags['prompt']

                        # Extract workflow if present
                        if 'workflow' in tags:
                            try:
                                workflow_val = json.loads(tags['workflow'])
                            except:
                                workflow_val = tags['workflow']

                        if prompt_val or workflow_val:
                            print(f"[PromptExtractor] Successfully extracted metadata using ffprobe")
                            # Cache it for next time
                            _file_metadata_cache[cache_key] = {
                                'prompt': prompt_val,
                                'workflow': workflow_val
                            }
                            return prompt_val, workflow_val

                print(f"[PromptExtractor] No metadata found in video with ffprobe")
            except FileNotFoundError:
                print("[PromptExtractor] ffprobe not found - cannot extract video metadata")
            except Exception as e:
                print(f"[PromptExtractor] ffprobe extraction failed: {e}")

        return None, None
    except Exception as e:
        print(f"[PromptExtractor] Error reading video metadata: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def base64_to_tensor(base64_data):
    """
    Convert base64 data URL to ComfyUI tensor format.
    """
    try:
        import base64
        import io
        from PIL import Image

        # Remove data URL prefix (data:image/png;base64,)
        if ',' in base64_data:
            base64_data = base64_data.split(',', 1)[1]

        # Decode base64 to bytes
        img_bytes = base64.b64decode(base64_data)

        # Load as PIL Image
        img = Image.open(io.BytesIO(img_bytes))

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert to numpy array and normalize to 0-1
        img_array = np.array(img).astype(np.float32) / 255.0

        # Convert to torch tensor with batch dimension (B, H, W, C)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return img_tensor
    except Exception as e:
        print(f"[PromptExtractor] Error converting base64 to tensor: {e}")
        return None


def get_cached_video_frame(relative_path, frame_position):
    """
    Retrieve cached video frame extracted by JavaScript at the specified position.

    Args:
        relative_path: The video file path relative to input directory (e.g., "workflow/video.mp4")
        frame_position: float from 0.0 to 1.0 representing position in video (used for requesting new extraction)

    Returns:
        A single frame tensor or None if cache is missing.
    """
    import time

    # Handle None frame_position
    if frame_position is None:
        frame_position = 0.01

    # Normalize path separators and replace for cache key (matching JavaScript)
    # Note: Cache key does NOT include frame_position - we only cache one frame per video
    # When user adjusts the slider, JavaScript recaches with the new frame
    path_key = relative_path.replace('\\', '/').replace('/', '_')

    if path_key not in _video_frames_cache:
        print(f"[PromptExtractor] Cache missing for: {relative_path} at position {frame_position}, requesting extraction...")

        # Broadcast directly to JavaScript clients
        try:
            server.PromptServer.instance.send_sync("prompt-extractor-extract-frame", {
                "filename": relative_path,
                "frame_position": frame_position
            })

            # Wait briefly for JavaScript to extract and cache the frame
            max_wait = 5  # seconds
            start_time = time.time()
            while path_key not in _video_frames_cache and (time.time() - start_time) < max_wait:
                time.sleep(0.1)

            if path_key in _video_frames_cache:
                print(f"[PromptExtractor] Frame cached successfully for: {relative_path}")
            else:
                print(f"[PromptExtractor] Timeout waiting for frame extraction: {relative_path}")
                return None

        except Exception as e:
            print(f"[PromptExtractor] Error requesting frame extraction: {e}")
            return None

    frame_data = _video_frames_cache[path_key]

    # Convert base64 frame to tensor
    frame_tensor = base64_to_tensor(frame_data)

    return frame_tensor


def build_link_map(workflow_data):
    """Build a map from link_id to (source_node_id, source_slot, dest_node_id, dest_slot), including links from subgraphs"""
    link_map = {}

    # Add top-level links
    links = workflow_data.get('links', [])
    for link in links:
        # link format: [link_id, source_node_id, source_slot, dest_node_id, dest_slot, type]
        if len(link) >= 5:
            link_id = link[0]
            link_map[link_id] = {
                'source_node': link[1],
                'source_slot': link[2],
                'dest_node': link[3],
                'dest_slot': link[4]
            }

    # Add links from subgraph definitions
    if 'definitions' in workflow_data and 'subgraphs' in workflow_data['definitions']:
        for subgraph in workflow_data['definitions']['subgraphs']:
            if 'links' in subgraph:
                for link in subgraph['links']:
                    # Subgraph links can be either dict format or array format
                    if isinstance(link, dict):
                        link_id = link.get('id')
                        if link_id:
                            link_map[link_id] = {
                                'source_node': link.get('origin_id'),
                                'source_slot': link.get('origin_slot'),
                                'dest_node': link.get('target_id'),
                                'dest_slot': link.get('target_slot')
                            }
                    elif len(link) >= 5:
                        link_id = link[0]
                        link_map[link_id] = {
                            'source_node': link[1],
                            'source_slot': link[2],
                            'dest_node': link[3],
                            'dest_slot': link[4]
                        }

    return link_map


def build_node_map(workflow_data):
    """Build a map from node_id to node data, including nodes from subgraphs"""
    node_map = {}

    # Add top-level nodes
    nodes = workflow_data.get('nodes', [])
    for node in nodes:
        node_id = node.get('id')
        if node_id is not None:
            node_map[node_id] = node

    # Add nodes from subgraph definitions
    if 'definitions' in workflow_data and 'subgraphs' in workflow_data['definitions']:
        for subgraph in workflow_data['definitions']['subgraphs']:
            if 'nodes' in subgraph:
                for node in subgraph['nodes']:
                    node_id = node.get('id')
                    if node_id is not None:
                        node_map[node_id] = node

    return node_map


def determine_clip_text_encode_type(node_id, workflow_data, node_map):
    """
    Determine if a CLIPTextEncode node is positive or negative by checking
    what input it connects to in downstream nodes.
    Returns: 'positive', 'negative', or None if unclear
    """
    links = workflow_data.get('links', [])

    # Find all links where this node is the source
    for link in links:
        if len(link) >= 5:
            source_node_id = link[1]
            dest_node_id = link[3]
            dest_slot = link[4]

            if source_node_id == node_id:
                # This node is the source, check where it connects
                dest_node = node_map.get(dest_node_id)
                if dest_node:
                    dest_inputs = dest_node.get('inputs', [])
                    # Check the destination input name
                    if dest_slot < len(dest_inputs):
                        input_name = dest_inputs[dest_slot].get('name', '').lower()
                        if 'positive' in input_name:
                            return 'positive'
                        elif 'negative' in input_name:
                            return 'negative'

    return None


def traverse_to_find_text(node_id, input_slot, node_map, link_map, visited=None, max_depth=20):
    """
    Traverse backwards through node connections to find the actual prompt text.
    Follows through Concatenate, Find and Replace, and other string manipulation nodes.
    Returns the found text or empty string.
    """
    if visited is None:
        visited = set()

    if node_id in visited or max_depth <= 0:
        return ""
    visited.add(node_id)

    node = node_map.get(node_id)
    if not node:
        return ""

    node_type = node.get('type', '')
    widgets_values = node.get('widgets_values', [])
    inputs = node.get('inputs', [])

    # Check if this node has direct text in widgets_values
    if node_type in ['PrimitiveStringMultiline', 'PrimitiveString', 'String', 'Text']:
        # Return first string value
        for val in widgets_values:
            if isinstance(val, str) and val.strip():
                return val.strip()

    # CLIPTextEncode - check if text is in widgets or need to traverse
    if node_type in ['CLIPTextEncode', 'CLIPTextEncodeSDXL', 'CLIPTextEncodeFlux']:
        # Check widget values first
        for val in widgets_values:
            if isinstance(val, str) and len(val) > 10:  # Likely a prompt
                return val.strip()
        # Otherwise traverse the text input
        for inp in inputs:
            if inp.get('name') == 'text' and inp.get('link'):
                link_id = inp['link']
                link_info = link_map.get(link_id)
                if link_info:
                    return traverse_to_find_text(
                        link_info['source_node'],
                        link_info['source_slot'],
                        node_map, link_map, visited, max_depth - 1
                    )
        return ""

    # StringConcatenate - combine inputs
    if node_type in ['StringConcatenate', 'Text Concatenate', 'Concat String']:
        parts = []
        delimiter = " "
        # Get delimiter from widgets if present
        for val in widgets_values:
            if isinstance(val, str) and len(val) <= 3:
                delimiter = val
                break

        # Find string_a and string_b inputs
        for inp in inputs:
            name = inp.get('name', '')
            if name in ['string_a', 'string_b', 'text_a', 'text_b'] and inp.get('link'):
                link_id = inp['link']
                link_info = link_map.get(link_id)
                if link_info:
                    text = traverse_to_find_text(
                        link_info['source_node'],
                        link_info['source_slot'],
                        node_map, link_map, visited.copy(), max_depth - 1
                    )
                    if text:
                        parts.append(text)

        return delimiter.join(parts) if parts else ""

    # Text Find and Replace - traverse to first input
    if node_type in ['Text Find and Replace', 'FindReplace', 'String Replace']:
        # These just pass through modified text, traverse to input
        for inp in inputs:
            if inp.get('name') in ['text', 'string', 'input'] and inp.get('link'):
                link_id = inp['link']
                link_info = link_map.get(link_id)
                if link_info:
                    return traverse_to_find_text(
                        link_info['source_node'],
                        link_info['source_slot'],
                        node_map, link_map, visited, max_depth - 1
                    )

    # Florence2Run - has caption output
    if node_type in ['Florence2Run', 'Florence2']:
        # Can't traverse further, but check for cached caption in widgets
        for val in widgets_values:
            if isinstance(val, str) and len(val) > 20:
                return val.strip()
        return ""

    # easy showAnything - traverse input
    if node_type in ['easy showAnything', 'ShowText', 'Preview String']:
        for inp in inputs:
            if inp.get('link'):
                link_id = inp['link']
                link_info = link_map.get(link_id)
                if link_info:
                    return traverse_to_find_text(
                        link_info['source_node'],
                        link_info['source_slot'],
                        node_map, link_map, visited, max_depth - 1
                    )

    # Generic: if node has a text/string output, check widgets
    for val in widgets_values:
        if isinstance(val, str) and len(val) > 20:
            return val.strip()

    # Try traversing first text input
    for inp in inputs:
        name = inp.get('name', '').lower()
        if ('text' in name or 'string' in name or 'prompt' in name) and inp.get('link'):
            link_id = inp['link']
            link_info = link_map.get(link_id)
            if link_info:
                result = traverse_to_find_text(
                    link_info['source_node'],
                    link_info['source_slot'],
                    node_map, link_map, visited, max_depth - 1
                )
                if result:
                    return result

    return ""


def extract_power_lora_loader(node):
    """Extract ALL LoRAs from Power Lora Loader (rgthree) node (regardless of active state)"""
    loras = []
    widgets_values = node.get('widgets_values', [])

    for val in widgets_values:
        if isinstance(val, dict):
            # Format: {"on": true, "lora": "path/to/lora.safetensors", "strength": 1.0, "strengthTwo": null}
            # Extract ALL LoRAs, not just active ones
            if val.get('lora'):
                lora_path = val['lora']
                strength = float(val.get('strength', 1.0))
                strength_two = val.get('strengthTwo')
                clip_strength = float(strength_two) if strength_two is not None else strength
                is_active = val.get('on', True)

                loras.append({
                    'name': os.path.splitext(os.path.basename(lora_path))[0],
                    'path': lora_path,
                    'model_strength': strength,
                    'clip_strength': clip_strength,
                    'active': is_active
                })

    return loras


def extract_lora_manager_stacker(node):
    """Extract ALL LoRAs from Lora Stacker (LoraManager) node (regardless of active state)"""
    loras = []
    widgets_values = node.get('widgets_values', [])

    # Format: widgets_values[1] contains array of LoRA objects
    # [{"name":"lora_name","strength":0.33,"active":true,"expanded":false,"clipStrength":0.33}, ...]
    for val in widgets_values:
        if isinstance(val, list):
            for lora in val:
                if isinstance(lora, dict):
                    lora_name = lora.get('name', '')
                    # Extract ALL LoRAs, not just active ones
                    if lora_name:
                        # Handle strength as string or number
                        strength = lora.get('strength', 1.0)
                        if isinstance(strength, str):
                            strength = float(strength)
                        clip_strength = lora.get('clipStrength', strength)
                        if isinstance(clip_strength, str):
                            clip_strength = float(clip_strength)
                        is_active = lora.get('active', True)

                        loras.append({
                            'name': lora_name,
                            'path': '',
                            'model_strength': float(strength),
                            'clip_strength': float(clip_strength),
                            'active': is_active
                        })

    return loras


def extract_wan_video_lora_select_multi(node):
    """Extract ALL LoRAs from WanVideoLoraSelectMulti node (regardless of active state)"""
    loras = []
    widgets_values = node.get('widgets_values', [])

    # WanVideoLoraSelectMulti stores LoRAs as pairs in widgets_values:
    # [lora_name_1, strength_1, lora_name_2, strength_2, ..., none, strength, bool, bool]
    # Parse pairs of (lora_name, strength)

    i = 0
    while i < len(widgets_values) - 1:
        lora_name = widgets_values[i]

        # Check if this is a string (potential LoRA name)
        if isinstance(lora_name, str):
            # Skip if it's "none" or empty
            if lora_name.lower() == 'none' or not lora_name.strip():
                i += 2  # Skip the strength value too
                continue

            # Next value should be the strength
            if i + 1 < len(widgets_values):
                strength_val = widgets_values[i + 1]

                # If next value is numeric, it's the strength for this LoRA
                if isinstance(strength_val, (int, float)):
                    strength = float(strength_val)

                    loras.append({
                        'name': os.path.splitext(os.path.basename(lora_name))[0],
                        'path': lora_name,
                        'model_strength': strength,
                        'clip_strength': strength,  # WanVideo doesn't separate model/clip strength
                        'active': True  # All LoRAs in the list are considered active
                    })

                    i += 2  # Move to next pair
                    continue

        # If we didn't process a pair, just move to next item
        i += 1

    # Legacy fallback for other possible formats
    for val in widgets_values:
        # Format 1: List of LoRA dictionaries
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    # Extract LoRA info from dictionary format
                    lora_name = item.get('lora') or item.get('name') or item.get('lora_name', '')
                    if lora_name and lora_name != 'None':
                        strength = item.get('strength') or item.get('model_strength', 1.0)
                        if isinstance(strength, str):
                            strength = float(strength) if strength else 1.0
                        else:
                            strength = float(strength)

                        clip_strength = item.get('clip_strength') or item.get('strengthTwo')
                        if clip_strength is not None:
                            if isinstance(clip_strength, str):
                                clip_strength = float(clip_strength) if clip_strength else strength
                            else:
                                clip_strength = float(clip_strength)
                        else:
                            clip_strength = strength

                        is_active = item.get('on', item.get('active', item.get('enabled', True)))

                        loras.append({
                            'name': os.path.splitext(os.path.basename(lora_name))[0],
                            'path': lora_name,
                            'model_strength': strength,
                            'clip_strength': clip_strength,
                            'active': is_active
                        })
                elif isinstance(item, str) and item and item != 'None':
                    # Format 2: Simple string list of LoRA names
                    loras.append({
                        'name': os.path.splitext(os.path.basename(item))[0],
                        'path': item,
                        'model_strength': 1.0,
                        'clip_strength': 1.0,
                        'active': True
                    })
        # Format 3: Dictionary containing LoRA info
        elif isinstance(val, dict):
            lora_name = val.get('lora') or val.get('name') or val.get('lora_name', '')
            if lora_name and lora_name != 'None':
                strength = val.get('strength') or val.get('model_strength', 1.0)
                if isinstance(strength, str):
                    strength = float(strength) if strength else 1.0
                else:
                    strength = float(strength)

                clip_strength = val.get('clip_strength') or val.get('strengthTwo')
                if clip_strength is not None:
                    if isinstance(clip_strength, str):
                        clip_strength = float(clip_strength) if clip_strength else strength
                    else:
                        clip_strength = float(clip_strength)
                else:
                    clip_strength = strength

                is_active = val.get('on', val.get('active', val.get('enabled', True)))

                loras.append({
                    'name': os.path.splitext(os.path.basename(lora_name))[0],
                    'path': lora_name,
                    'model_strength': strength,
                    'clip_strength': clip_strength,
                    'active': is_active
                })

    return loras


def extract_lora_loader_stack_rgthree(node):
    """Extract LoRAs from Lora Loader Stack (rgthree) node - supports up to 4 LoRAs"""
    loras = []
    widgets_values = node.get('widgets_values', [])

    # Lora Loader Stack (rgthree) stores LoRAs as pairs in widgets_values:
    # [lora_01, strength_01, lora_02, strength_02, lora_03, strength_03, lora_04, strength_04]
    # Parse pairs of (lora_name, strength) - max 4 LoRAs

    i = 0
    while i < len(widgets_values) - 1 and i < 8:  # Max 4 LoRAs = 8 values
        lora_name = widgets_values[i]

        # Check if this is a string (potential LoRA name)
        if isinstance(lora_name, str):
            # Skip if it's "None" or empty
            if lora_name == 'None' or not lora_name.strip():
                i += 2  # Skip the strength value too
                continue

            # Next value should be the strength
            if i + 1 < len(widgets_values):
                strength_val = widgets_values[i + 1]

                # If next value is numeric, it's the strength for this LoRA
                if isinstance(strength_val, (int, float)):
                    strength = float(strength_val)

                    loras.append({
                        'name': os.path.splitext(os.path.basename(lora_name))[0],
                        'path': lora_name,
                        'model_strength': strength,
                        'clip_strength': strength,  # No separate model/clip strength
                        'active': True  # All LoRAs are active (no on/off toggle)
                    })

                    i += 2  # Move to next pair
                    continue

        # If we didn't process a pair, move to next item
        i += 1

    return loras


def extract_standard_lora_loader(node):
    """Extract LoRA from standard LoraLoader or LoraLoaderModelOnly node"""
    widgets_values = node.get('widgets_values', [])

    if len(widgets_values) < 1:
        return []

    lora_name = widgets_values[0] if isinstance(widgets_values[0], str) else ''
    if not lora_name:
        return []

    model_strength = 1.0
    clip_strength = 1.0

    if len(widgets_values) >= 2:
        model_strength = float(widgets_values[1]) if widgets_values[1] is not None else 1.0
    if len(widgets_values) >= 3:
        clip_strength = float(widgets_values[2]) if widgets_values[2] is not None else model_strength
    else:
        clip_strength = model_strength

    # Standard LoRA loaders are always active (no on/off toggle)
    return [{
        'name': os.path.splitext(os.path.basename(lora_name))[0],
        'path': lora_name,
        'model_strength': model_strength,
        'clip_strength': clip_strength,
        'active': True
    }]


def extract_loras_from_node(node):
    """
    Extract LoRAs from any supported LoRA loader node type.
    Returns a list of LoRA dicts: {name, path, model_strength, clip_strength}
    """
    node_type = node.get('type', '')

    # Power Lora Loader (rgthree) - multiple LoRAs
    if node_type == 'Power Lora Loader (rgthree)':
        return extract_power_lora_loader(node)

    # Lora Stacker (LoraManager) variants
    lora_stacker_types = [
        'Lora Stacker (LoraManager)',
        'LoRA Stacker',
        'LoraStacker',
        'LoRA Stacker (LoRA Manager)'
    ]
    if node_type in lora_stacker_types:
        return extract_lora_manager_stacker(node)

    # WanVideoLoraSelectMulti (video LoRA loader)
    if node_type == 'WanVideoLoraSelectMulti':
        return extract_wan_video_lora_select_multi(node)

    # Lora Loader Stack (rgthree) - up to 4 LoRAs
    if node_type == 'Lora Loader Stack (rgthree)':
        return extract_lora_loader_stack_rgthree(node)

    # Standard LoRA loaders
    standard_loader_types = [
        'LoraLoader',
        'LoraLoaderModelOnly',
        'LoRALoader',  # Alternative casing
        'LoraLoaderKJNodes',  # KJNodes variant
    ]
    if node_type in standard_loader_types:
        return extract_standard_lora_loader(node)

    return []


def is_lora_node(node_type):
    """Check if a node type is any kind of LoRA loader"""
    lora_node_types = [
        'Power Lora Loader (rgthree)',
        'Lora Loader Stack (rgthree)',
        'Lora Stacker (LoraManager)',
        'LoRA Stacker',
        'LoraStacker',
        'LoRA Stacker (LoRA Manager)',
        'LoraLoader',
        'LoraLoaderModelOnly',
        'LoRALoader',
        'LoraLoaderKJNodes',
        'WanVideoLoraSelectMulti',
    ]
    return node_type in lora_node_types


def collect_lora_model_chain(start_node_id, node_map, link_map, visited=None):
    """
    Traverse backwards through MODEL connections to collect all LoRAs in a chain.
    This works for any mix of LoRA loader types (Power Lora, standard LoraLoader, Stacker, etc.)

    Returns a tuple of (loras, titles) where:
      - loras: list of all LoRAs found in the chain
      - titles: list of all node titles in the chain (for determining high/low assignment)
    """
    if visited is None:
        visited = set()

    if start_node_id in visited:
        return [], []
    visited.add(start_node_id)

    node = node_map.get(start_node_id)
    if not node:
        return [], []

    node_type = node.get('type', '')
    all_loras = []
    all_titles = []

    # Extract LoRAs from this node if it's a LoRA loader AND it's actually connected
    if is_lora_node(node_type):
        # Check if this LoRA node's MODEL output is connected to something
        outputs = node.get('outputs', [])
        has_connected_output = False
        for output in outputs:
            output_type = output.get('type', '')
            # Check if this is a MODEL, LORA_STACK, or WANVIDLORA output with connections
            if output_type in ['MODEL', 'LORA_STACK', 'WANVIDLORA']:
                links = output.get('links')
                # links can be None, [], or a list with items
                if links is not None and len(links) > 0:
                    has_connected_output = True
                    break

        # Only extract LoRAs if this node's output is connected
        if has_connected_output:
            node_loras = extract_loras_from_node(node)
            all_loras.extend(node_loras)
            title = node.get('title', '')
            if title:
                all_titles.append(title)

    # Look for MODEL, lora_stack, or WANVIDLORA input connections and traverse backwards
    inputs = node.get('inputs', [])
    for inp in inputs:
        input_name = inp.get('name', '')
        input_type = inp.get('type', '')
        # Follow MODEL, model, lora_stack, or WANVIDLORA connections
        if (input_name in ['model', 'MODEL', 'lora_stack', 'lora'] or input_type == 'WANVIDLORA') and inp.get('link'):
            link_id = inp['link']
            link_info = link_map.get(link_id)
            if link_info:
                source_node_id = link_info['source_node']
                # Recursively collect LoRAs from the chain
                chain_loras, chain_titles = collect_lora_model_chain(source_node_id, node_map, link_map, visited)
                all_loras.extend(chain_loras)
                all_titles.extend(chain_titles)

    return all_loras, all_titles


def collect_lora_stack_chain(start_node_id, node_map, link_map, visited=None):
    """
    Traverse backwards through lora_stack connections to collect all LoRAs in a chain.
    This is specifically for LORA_STACK type connections (Lora Stacker nodes).
    Returns a tuple of (loras, titles) where:
      - loras: list of all LoRAs found in the chain
      - titles: list of all node titles in the chain (for determining high/low assignment)
    """
    if visited is None:
        visited = set()

    if start_node_id in visited:
        return [], []
    visited.add(start_node_id)

    node = node_map.get(start_node_id)
    if not node:
        return [], []

    node_type = node.get('type', '')
    all_loras = []
    all_titles = []

    # Check if this node is a LoRA Stacker type
    lora_stacker_types = [
        'Lora Stacker (LoraManager)',
        'LoRA Stacker',
        'LoraStacker',
        'LoRA Stacker (LoRA Manager)'
    ]

    if node_type in lora_stacker_types:
        # Extract LoRAs from this node
        node_loras = extract_lora_manager_stacker(node)
        all_loras.extend(node_loras)
        # Track the title for this node
        title = node.get('title', '')
        if title:
            all_titles.append(title)

    # Look for lora_stack input connection and traverse backwards
    inputs = node.get('inputs', [])
    for inp in inputs:
        if inp.get('name') == 'lora_stack' and inp.get('link'):
            link_id = inp['link']
            link_info = link_map.get(link_id)
            if link_info:
                source_node_id = link_info['source_node']
                # Recursively collect LoRAs from the chain
                chain_loras, chain_titles = collect_lora_stack_chain(source_node_id, node_map, link_map, visited)
                all_loras.extend(chain_loras)
                all_titles.extend(chain_titles)

    return all_loras, all_titles


def find_lora_chain_terminals(workflow_data, node_map, link_map):
    """
    Find terminal nodes for LoRA chains - nodes that receive MODEL input from LoRA loaders
    but are NOT LoRA loaders themselves (e.g., KSampler, other processing nodes).

    Returns a list of terminal node IDs that have LoRA chains feeding into them.
    """
    terminals = []

    # List of input names that receive MODEL type connections
    model_input_names = [
        'model', 'MODEL',
        'model_high_noise', 'model_low_noise',  # WanMoeKSamplerAdvanced
        'base_model', 'refiner_model',  # SDXL workflows
        'unet',  # Some custom nodes
    ]

    # Collect all nodes including those in subgraphs
    all_nodes = []
    if 'nodes' in workflow_data:
        all_nodes.extend(workflow_data.get('nodes', []))

    # Add nodes from subgraph definitions
    if 'definitions' in workflow_data and 'subgraphs' in workflow_data['definitions']:
        subgraph_count = len(workflow_data['definitions']['subgraphs'])
        subgraph_node_count = 0
        for subgraph in workflow_data['definitions']['subgraphs']:
            if 'nodes' in subgraph:
                subgraph_nodes = subgraph['nodes']
                subgraph_node_count += len(subgraph_nodes)
                all_nodes.extend(subgraph_nodes)
        print(f"[PromptExtractor] find_lora_chain_terminals: {len(workflow_data.get('nodes', []))} top-level nodes + {subgraph_node_count} subgraph nodes from {subgraph_count} subgraphs = {len(all_nodes)} total")

    for node in all_nodes:
        node_id = node.get('id')
        node_type = node.get('type', '')

        # Skip LoRA loader nodes - we want non-LoRA nodes that receive model input
        if is_lora_node(node_type):
            continue

        # Check if this node receives MODEL input
        # NOTE: Some nodes have multiple MODEL inputs (e.g., model, model_1 for high/low)
        inputs = node.get('inputs', [])
        for inp in inputs:
            inp_name = inp.get('name', '')
            inp_type = inp.get('type', '')

            # Check if this is a MODEL or WANVIDLORA type input (by type or by name matching)
            is_model_input = (inp_type in ['MODEL', 'WANVIDLORA'] or inp_name in model_input_names)

            if is_model_input and inp.get('link'):
                link_id = inp['link']
                link_info = link_map.get(link_id)
                if link_info:
                    source_id = link_info['source_node']

                    # Trace back through the MODEL chain to find a LoRA loader
                    # (might go through intermediate nodes like ModelSamplingSD3)
                    lora_source_id = trace_to_lora_loader(source_id, node_map, link_map, set())

                    if lora_source_id:
                        # Get the label to better identify high/low
                        inp_label = inp.get('label', '').lower()

                        # This node receives MODEL from a LoRA loader (possibly through intermediate nodes)
                        terminals.append({
                            'terminal_id': node_id,
                            'terminal_type': node_type,
                            'terminal_title': node.get('title', ''),
                            'lora_source_id': lora_source_id,
                            'input_name': inp_name,  # Track which input (model, model_1, etc)
                            'input_label': inp_label  # Track the label (model H, model L)
                        })

    return terminals


def trace_to_lora_loader(node_id, node_map, link_map, visited, max_depth=10):
    """
    Trace backwards through MODEL connections to find the first LoRA loader node.
    Returns the LoRA loader node ID, or None if no LoRA loader is found.
    """
    if max_depth <= 0 or node_id in visited:
        return None

    visited.add(node_id)

    node = node_map.get(node_id)
    if not node:
        return None

    # Check if this node is a LoRA loader
    if is_lora_node(node.get('type', '')):
        return node_id

    # Otherwise, trace back through MODEL or WANVIDLORA input
    inputs = node.get('inputs', [])
    for inp in inputs:
        inp_name = inp.get('name', '')
        inp_type = inp.get('type', '')
        # Follow MODEL, model, lora, or WANVIDLORA connections
        if (inp_name in ['model', 'MODEL', 'lora'] or inp_type in ['MODEL', 'WANVIDLORA']) and inp.get('link'):
            link_id = inp['link']
            link_info = link_map.get(link_id)
            if link_info:
                source_node_id = link_info['source_node']
                result = trace_to_lora_loader(source_node_id, node_map, link_map, visited, max_depth - 1)
                if result:
                    return result

    return None


def parse_workflow_for_prompts(prompt_data, workflow_data=None):
    """
    Parse workflow/prompt data to extract positive/negative prompts and LoRAs

    Returns dict with:
        - positive_prompt: str
        - negative_prompt: str
        - loras_a: list of {name, model_strength, clip_strength} - first lora loader (High noise)
        - loras_b: list of {name, model_strength, clip_strength} - second lora loader (Low noise)
    """
    result = {
        'positive_prompt': '',
        'negative_prompt': '',
        'loras_a': [],
        'loras_b': []
    }

    if not prompt_data and not workflow_data:
        return result

    # Check if prompt_data is parsed A1111 parameters (from JavaScript)
    if isinstance(prompt_data, dict) and 'prompt' in prompt_data and 'loras' in prompt_data:
        print("[PromptExtractor] Processing A1111 parsed parameters")
        result['positive_prompt'] = prompt_data.get('prompt', '')
        result['negative_prompt'] = prompt_data.get('negative_prompt', '')

        # Add all LoRAs to stack_a (A1111 doesn't have dual stacks)
        for lora in prompt_data.get('loras', []):
            # Skip blacklisted LoRAs
            if is_lora_blacklisted(lora['name']):
                continue
            result['loras_a'].append({
                'name': lora['name'],
                'model_strength': lora['model_strength'],
                'clip_strength': lora['clip_strength'],
                'active': True
            })

        print(f"[PromptExtractor] Extracted A1111: {len(result['loras_a'])} LoRAs, prompt length: {len(result['positive_prompt'])}")
        return result

    # Build maps for traversal if workflow_data is available
    node_map = {}
    link_map = {}
    if workflow_data and isinstance(workflow_data, dict) and 'nodes' in workflow_data:
        node_map = build_node_map(workflow_data)
        link_map = build_link_map(workflow_data)

    # Use prompt_data (API format) as primary source
    data = prompt_data if prompt_data else {}

    # If workflow_data exists but prompt_data doesn't, try to extract from workflow
    if not prompt_data and workflow_data:
        # Workflow format has 'nodes' array with widget values
        data = convert_workflow_to_prompt_format(workflow_data)

    if not isinstance(data, dict):
        return result

    # Track found prompts and LoRAs
    positive_prompts = []
    negative_prompts = []
    loras_a = []
    loras_b = []
    lora_names_seen_a = set()
    lora_names_seen_b = set()

    # Iterate through all nodes (workflow format) - including subgraphs
    all_workflow_nodes = []
    if workflow_data:
        # Add top-level nodes
        if 'nodes' in workflow_data:
            all_workflow_nodes.extend(workflow_data.get('nodes', []))

        # Add nodes from subgraph definitions
        if 'definitions' in workflow_data and 'subgraphs' in workflow_data['definitions']:
            for subgraph in workflow_data['definitions']['subgraphs']:
                if 'nodes' in subgraph:
                    all_workflow_nodes.extend(subgraph['nodes'])

    if all_workflow_nodes:
        for node in all_workflow_nodes:
            if not isinstance(node, dict):
                continue

            node_type = node.get('type', '')
            node_id = node.get('id')
            title = node.get('title', '')
            widgets_values = node.get('widgets_values', [])
            inputs = node.get('inputs', [])

            # Extract prompts - with traversal if needed
            if node_type in ['CLIPTextEncode', 'CLIPTextEncodeSDXL', 'CLIPTextEncodeFlux']:
                # Determine positive/negative by checking output connections (most reliable)
                connection_type = determine_clip_text_encode_type(node_id, workflow_data, node_map)

                # Fallback to title checking if connections don't give us an answer
                if not connection_type:
                    title_lower = title.lower() if title else ""
                    if 'negative' in title_lower:
                        connection_type = 'negative'
                    elif 'positive' in title_lower:
                        connection_type = 'positive'
                    else:
                        # Default to positive if no clear indicator
                        connection_type = 'positive'

                # First try to get text directly from widgets
                text_found = ""
                for val in widgets_values:
                    if isinstance(val, str) and len(val) > 10:
                        text_found = val.strip()
                        break

                # If text input is connected, traverse to find actual prompt
                for inp in inputs:
                    if inp.get('name') == 'text' and inp.get('link'):
                        link_id = inp['link']
                        link_info = link_map.get(link_id)
                        if link_info:
                            traversed_text = traverse_to_find_text(
                                link_info['source_node'],
                                link_info['source_slot'],
                                node_map, link_map, set(), 20
                            )
                            if traversed_text:
                                text_found = traversed_text

                if text_found:
                    if connection_type == 'negative':
                        negative_prompts.append(text_found)
                    else:
                        positive_prompts.append(text_found)

            # PromptManager nodes
            elif node_type in ['PromptManager', 'PromptManagerAdvanced']:
                for val in widgets_values:
                    if isinstance(val, str) and len(val) > 20:
                        positive_prompts.append(val.strip())
                        break

            # PrimitiveStringMultiline - direct prompt text (only add if connected to something)
            elif node_type == 'PrimitiveStringMultiline':
                # Check title for hints (must be explicit)
                title_lower = title.lower() if title else ""
                is_negative = 'negative' in title_lower
                is_positive = 'positive' in title_lower or not is_negative  # Default to positive if not explicitly negative

                for val in widgets_values:
                    if isinstance(val, str) and len(val) > 20:
                        if is_negative:
                            negative_prompts.append(val.strip())
                        else:
                            positive_prompts.append(val.strip())
                        break

    # ========================================
    # UNIFIED LORA CHAIN EXTRACTION
    # ========================================
    # Find all LoRA chains by starting from terminal nodes (non-LoRA nodes that receive MODEL input)
    # and traversing backwards through MODEL connections

    lora_chains = []
    processed_terminals = set()  # Track by (terminal_id, input_name) tuple to allow multiple inputs per node

    if workflow_data:
        # Method 1: Find chains ending at non-LoRA nodes (MODEL input chains)
        terminals = find_lora_chain_terminals(workflow_data, node_map, link_map)
        print(f"[PromptExtractor] Found {len(terminals)} terminal nodes for LoRA chains")

        for terminal_info in terminals:
            terminal_id = terminal_info['terminal_id']
            input_name = terminal_info.get('input_name', '')
            source_id = terminal_info['lora_source_id']

            # Track by (terminal_id, input_name) to allow same terminal with multiple model inputs
            terminal_key = (terminal_id, input_name)
            if terminal_key in processed_terminals:
                continue

            # Collect all LoRAs in this chain
            chain_loras, chain_titles = collect_lora_model_chain(source_id, node_map, link_map)

            if chain_loras:
                active_count = sum(1 for lora in chain_loras if lora.get('active', True))
                inactive_count = len(chain_loras) - active_count
                lora_names_in_chain = [lora.get('name', 'unknown') for lora in chain_loras if lora.get('active', True)]
                print(f"[PromptExtractor] Chain from terminal {terminal_id} ({terminal_info.get('terminal_title', '')}), input '{input_name}': {active_count} active, {inactive_count} inactive LoRAs")
                print(f"[PromptExtractor]   Active LoRAs: {lora_names_in_chain}")
                print(f"[PromptExtractor]   Chain titles: {chain_titles}")
                print(f"[PromptExtractor]   Input label: {terminal_info.get('input_label', '')}")

            # Mark this terminal input as processed
            processed_terminals.add(terminal_key)

            if chain_loras:
                lora_chains.append({
                    'titles': chain_titles,
                    'loras': chain_loras,
                    'terminal_title': terminal_info.get('terminal_title', ''),
                    'terminal_id': terminal_id,
                    'source_id': source_id,
                    'input_name': terminal_info.get('input_name', ''),
                    'input_label': terminal_info.get('input_label', '')
                })

        # Method 2: Find LORA_STACK chains (for Lora Stacker nodes)
        lora_stacker_types = [
            'Lora Stacker (LoraManager)',
            'LoRA Stacker',
            'LoraStacker',
            'LoRA Stacker (LoRA Manager)'
        ]

        stacker_nodes = {}
        for node in all_workflow_nodes:
            if node.get('type') in lora_stacker_types:
                stacker_nodes[node.get('id')] = node

        # Find stackers feeding other stackers
        stackers_feeding_stackers = set()
        for node in all_workflow_nodes:
            if node.get('type') in lora_stacker_types:
                for inp in node.get('inputs', []):
                    if inp.get('name') == 'lora_stack' and inp.get('link'):
                        link_info = link_map.get(inp['link'])
                        if link_info and link_info['source_node'] in stacker_nodes:
                            stackers_feeding_stackers.add(link_info['source_node'])

        # Terminal stackers - but only include them if their output is actually connected
        terminal_stackers = [nid for nid in stacker_nodes.keys() if nid not in stackers_feeding_stackers]

        for terminal_id in terminal_stackers:
            node = stacker_nodes[terminal_id]

            # Check if this stacker's output is actually connected to something
            # A disconnected stacker should not be included
            outputs = node.get('outputs', [])
            has_connected_output = False
            for output in outputs:
                # Check if any link exists from this output
                if output.get('links') and len(output.get('links', [])) > 0:
                    has_connected_output = True
                    break

            # Skip this stacker if it's not connected to anything
            if not has_connected_output:
                continue

            chain_loras, chain_titles = collect_lora_stack_chain(terminal_id, node_map, link_map)

            if chain_loras:
                lora_chains.append({
                    'titles': chain_titles,
                    'loras': chain_loras,
                    'terminal_title': node.get('title', ''),
                    'source_id': terminal_id
                })

    # ========================================
    # ASSIGN LORA CHAINS TO STACKS A AND B
    # ========================================
    # Based on title hints (high/low) or position

    lora_chains.sort(key=lambda x: x.get('source_id', 0))

    print(f"[PromptExtractor] Processing {len(lora_chains)} chains for stack assignment")
    for i, chain in enumerate(lora_chains):
        # Check ALL titles in the chain for high/low hints
        all_titles = chain.get('titles', []) + [chain.get('terminal_title', '')]
        all_titles_lower = ' '.join(all_titles).lower()

        # ALSO check the input_name and input_label which are the most reliable indicators
        input_name = chain.get('input_name', '').lower()
        input_label = chain.get('input_label', '').lower()

        # IMPORTANT: Only check ACTIVE loras for the chain assignment
        active_loras = [lora for lora in chain.get('loras', []) if lora.get('active', True)]

        print(f"[PromptExtractor] Chain {i}: {len(chain.get('loras', []))} total LoRAs, {len(active_loras)} active")
        print(f"  Titles: {all_titles}")
        print(f"  Active LoRA names: {[lora.get('name', '') for lora in active_loras]}")

        # PRIORITY 1: Check CHAIN STRUCTURE (most reliable)
        # Use word boundaries to match complete words in titles and input names
        chain_has_high = (
            re.search(r'\bhigh\b', all_titles_lower) or
            re.search(r'\bhigh\b', input_name) or
            re.search(r'\bhigh\b', input_label) or
            'highnoise' in all_titles_lower.replace('_', '').replace('-', '').replace(' ', '')
        )
        chain_has_low = (
            re.search(r'\blow\b', all_titles_lower) or
            re.search(r'\blow\b', input_name) or
            re.search(r'\blow\b', input_label) or
            'lownoise' in all_titles_lower.replace('_', '').replace('-', '').replace(' ', '')
        )

        # PRIORITY 2: Check LoRA filenames with MAJORITY VOTING (fallback when chain structure unclear)
        # Count how many LoRAs have high/low indicators (excluding blacklisted LoRAs)
        high_count = 0
        low_count = 0
        for lora in active_loras:
            lora_name = lora.get('name', '')
            # Skip blacklisted LoRAs from voting
            if is_lora_blacklisted(lora_name):
                continue
            lora_name_lower = lora_name.lower()
            has_high_pattern = (
                '_high' in lora_name_lower or
                '-high' in lora_name_lower or
                'high_' in lora_name_lower or
                '_h.' in lora_name_lower or
                '_h_' in lora_name_lower
            )
            has_low_pattern = (
                '_low' in lora_name_lower or
                '-low' in lora_name_lower or
                'low_' in lora_name_lower or
                '_l.' in lora_name_lower or
                '_l_' in lora_name_lower
            )
            if has_high_pattern:
                high_count += 1
            if has_low_pattern:
                low_count += 1
                low_count += 1

        # Combine: Chain structure takes priority, filenames used as tiebreaker
        has_high = chain_has_high or (not chain_has_low and high_count > low_count)
        has_low = chain_has_low or (not chain_has_high and low_count > high_count)

        print(f"  Chain structure: high={chain_has_high}, low={chain_has_low}")
        print(f"  LoRA filename voting: {high_count} high, {low_count} low")
        print(f"  Final decision: has_high={has_high}, has_low={has_low}")

        # Determine which stack based on title hints
        if has_high and not has_low:
            print(f"[PromptExtractor] Chain {i} → STACK A (high detected in chain structure)")
            for lora in chain['loras']:
                # Only add active LoRAs
                if not lora.get('active', True):
                    continue
                # Skip blacklisted LoRAs
                if is_lora_blacklisted(lora['name']):
                    print(f"  Skipping blacklisted LoRA: {lora['name']}")
                    continue
                if lora['name'] not in lora_names_seen_a:
                    lora_names_seen_a.add(lora['name'])
                    loras_a.append(lora)
        elif has_low and not has_high:
            print(f"[PromptExtractor] Chain {i} → STACK B (low detected in chain structure)")
            for lora in chain['loras']:
                # Only add active LoRAs
                if not lora.get('active', True):
                    continue
                # Skip blacklisted LoRAs
                if is_lora_blacklisted(lora['name']):
                    print(f"  Skipping blacklisted LoRA: {lora['name']}")
                    continue
                if lora['name'] not in lora_names_seen_b:
                    lora_names_seen_b.add(lora['name'])
                    loras_b.append(lora)
        elif i == 0:
            # First chain defaults to A
            print(f"[PromptExtractor] Chain {i} → STACK A (first chain default, has_high={has_high}, has_low={has_low})")
            for lora in chain['loras']:
                # Only add active LoRAs
                if not lora.get('active', True):
                    continue
                # Skip blacklisted LoRAs
                if is_lora_blacklisted(lora['name']):
                    print(f"  Skipping blacklisted LoRA: {lora['name']}")
                    continue
                if lora['name'] not in lora_names_seen_a:
                    lora_names_seen_a.add(lora['name'])
                    loras_a.append(lora)
        elif i == 1:
            # Second chain defaults to B
            print(f"[PromptExtractor] Chain {i} → STACK B (second chain default, has_high={has_high}, has_low={has_low})")
            for lora in chain['loras']:
                # Only add active LoRAs
                if not lora.get('active', True):
                    continue
                # Skip blacklisted LoRAs
                if is_lora_blacklisted(lora['name']):
                    print(f"  Skipping blacklisted LoRA: {lora['name']}")
                    continue
                if lora['name'] not in lora_names_seen_b:
                    lora_names_seen_b.add(lora['name'])
                    loras_b.append(lora)
        else:
            # Additional chains go to A
            print(f"[PromptExtractor] Chain {i} → STACK A (additional chain default, has_high={has_high}, has_low={has_low})")
            for lora in chain['loras']:
                # Only add active LoRAs
                if not lora.get('active', True):
                    continue
                # Skip blacklisted LoRAs
                if is_lora_blacklisted(lora['name']):
                    print(f"  Skipping blacklisted LoRA: {lora['name']}")
                    continue
                if lora['name'] not in lora_names_seen_a:
                    lora_names_seen_a.add(lora['name'])
                    loras_a.append(lora)

    # Also iterate through prompt_data format (API format)
    # BUT: Only extract LoRAs from API format if we didn't already get them from workflow chains
    skip_api_lora_extraction = len(lora_chains) > 0

    for node_id, node_data in data.items():
        if not isinstance(node_data, dict):
            continue

        class_type = node_data.get('class_type', '')
        inputs = node_data.get('inputs', {})

        # Extract prompts from various node types (API format has direct text values)
        if class_type in ['CLIPTextEncode', 'CLIPTextEncodeSDXL', 'CLIPTextEncodeFlux']:
            text = inputs.get('text', '')
            if text and isinstance(text, str):
                positive_prompts.append(text)

        # PromptManager nodes
        elif class_type in ['PromptManager', 'PromptManagerAdvanced']:
            text = inputs.get('text', '')
            if text and isinstance(text, str):
                positive_prompts.append(text)

        # Standard LoRA loaders (API format)
        # Skip this if we already extracted LoRAs from workflow chains
        elif class_type in ['LoraLoader', 'LoraLoaderModelOnly'] and not skip_api_lora_extraction:
            # Check if this node's MODEL output is connected
            if node_map:
                node = node_map.get(int(node_id) if str(node_id).isdigit() else node_id)
                if node:
                    outputs = node.get('outputs', [])
                    has_connected_output = False
                    for output in outputs:
                        if output.get('type') == 'MODEL':
                            links = output.get('links')
                            if links and isinstance(links, list) and len(links) > 0:
                                has_connected_output = True
                                break
                    if not has_connected_output:
                        continue  # Skip disconnected nodes

            lora_name = inputs.get('lora_name', '')
            if lora_name and lora_name not in lora_names_seen_a:
                # Skip blacklisted LoRAs
                lora_basename = os.path.splitext(os.path.basename(lora_name))[0]
                if is_lora_blacklisted(lora_basename):
                    continue
                lora_names_seen_a.add(lora_name)
                model_strength = float(inputs.get('strength_model', inputs.get('strength', 1.0)))
                clip_strength = float(inputs.get('strength_clip', model_strength))
                loras_a.append({
                    'name': lora_basename,
                    'path': lora_name,
                    'model_strength': model_strength,
                    'clip_strength': clip_strength
                })

    # Also check for LoRA syntax in prompts: <lora:name:strength>
    # Skip this if we already extracted LoRAs from workflow chains
    if not skip_api_lora_extraction:
        all_prompts = ' '.join(positive_prompts + negative_prompts)
        lora_pattern = r'<lora:([^:>]+):([^:>]+)(?::([^>]+))?>'
        for match in re.finditer(lora_pattern, all_prompts):
            lora_name = match.group(1).strip()
            # Skip blacklisted LoRAs
            if is_lora_blacklisted(lora_name):
                continue
            if lora_name not in lora_names_seen_a:
                lora_names_seen_a.add(lora_name)
                model_strength = float(match.group(2)) if match.group(2) else 1.0
                clip_strength = float(match.group(3)) if match.group(3) else model_strength
                loras_a.append({
                    'name': lora_name,
                    'path': '',
                    'model_strength': model_strength,
                    'clip_strength': clip_strength
                })

    # Clean LoRA syntax from prompts (even if we skipped extraction, we still clean the syntax)
    lora_pattern = r'<lora:([^:>]+):([^:>]+)(?::([^>]+))?>'
    clean_positive = []
    for p in positive_prompts:
        cleaned = re.sub(lora_pattern, '', p).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Collapse multiple spaces
        if cleaned:
            clean_positive.append(cleaned)

    clean_negative = []
    for p in negative_prompts:
        cleaned = re.sub(lora_pattern, '', p).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        if cleaned:
            clean_negative.append(cleaned)

    # Use the first prompt from each list (our connection logic already determined which is which)
    # If multiple prompts exist, concatenate them with commas
    result['positive_prompt'] = ', '.join(clean_positive) if clean_positive else ''
    result['negative_prompt'] = ', '.join(clean_negative) if clean_negative else ''
    result['loras_a'] = loras_a
    result['loras_b'] = loras_b

    return result


def convert_workflow_to_prompt_format(workflow_data):
    """Convert workflow format (nodes array) to prompt format (node_id: data dict), including subgraph nodes"""
    if not isinstance(workflow_data, dict):
        return {}

    result = {}

    # Collect all nodes (top-level + subgraphs)
    all_nodes = []

    # Add top-level nodes
    if 'nodes' in workflow_data:
        all_nodes.extend(workflow_data.get('nodes', []))

    # Add nodes from subgraph definitions
    if 'definitions' in workflow_data and 'subgraphs' in workflow_data['definitions']:
        for subgraph in workflow_data['definitions']['subgraphs']:
            if 'nodes' in subgraph:
                all_nodes.extend(subgraph['nodes'])

    for node in all_nodes:
        if not isinstance(node, dict):
            continue

        node_id = str(node.get('id', ''))
        if not node_id:
            continue

        class_type = node.get('type', '')
        widgets_values = node.get('widgets_values', [])

        # Map widgets_values to inputs based on node type
        inputs = {}

        if class_type in ['CLIPTextEncode', 'CLIPTextEncodeSDXL']:
            if widgets_values:
                inputs['text'] = widgets_values[0] if widgets_values else ''

        elif class_type in ['LoraLoader', 'LoraLoaderModelOnly']:
            if len(widgets_values) >= 1:
                inputs['lora_name'] = widgets_values[0]
            if len(widgets_values) >= 2:
                inputs['strength_model'] = widgets_values[1]
            if len(widgets_values) >= 3:
                inputs['strength_clip'] = widgets_values[2]

        elif class_type in ['PromptManager', 'PromptManagerAdvanced']:
            # Find text widget value
            for val in widgets_values:
                if isinstance(val, str) and len(val) > 20:  # Likely the prompt text
                    inputs['text'] = val
                    break

        result[node_id] = {
            'class_type': class_type,
            'inputs': inputs
        }

    return result


def load_image_as_tensor(file_path):
    """Load an image file and convert to ComfyUI tensor format (B, H, W, C) as torch tensor"""
    if not IMAGE_SUPPORT:
        return None

    try:
        img = Image.open(file_path)
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert to numpy array and normalize to 0-1
        img_array = np.array(img).astype(np.float32) / 255.0

        # Convert to torch tensor with batch dimension (B, H, W, C)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return img_tensor
    except Exception as e:
        print(f"[PromptExtractor] Error loading image: {e}")
        return None


def get_placeholder_image_tensor():
    """Load the placeholder PNG as a tensor for display when no image is available"""
    if not IMAGE_SUPPORT:
        return torch.zeros((1, 128, 128, 3), dtype=torch.float32)

    try:
        # Get the path to placeholder.png relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        png_path = os.path.join(current_dir, 'js', 'placeholder.png')

        if os.path.exists(png_path):
            return load_image_as_tensor(png_path)
    except Exception as e:
        print(f"[PromptExtractor] Could not load placeholder PNG: {e}")

    # Fallback: create a simple gray placeholder
    img_array = np.full((128, 128, 3), 42 / 255.0, dtype=np.float32)
    return torch.from_numpy(img_array).unsqueeze(0)


class PromptExtractor:
    """
    Extract prompts and LoRA configurations from images, videos, and workflow files.
    Reads ComfyUI metadata embedded in files.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of supported files from input directory, including subfolders
        input_dir = folder_paths.get_input_directory()
        files = ["(none)"]  # Add empty option at the top for passthrough
        supported_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.json', '.mp4', '.webm', '.mov', '.avi']

        if os.path.exists(input_dir):
            # Walk through directory recursively
            for root, dirs, filenames in os.walk(input_dir):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in supported_extensions:
                        # Get relative path from input directory
                        full_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(full_path, input_dir)
                        # Convert Windows backslashes to forward slashes
                        rel_path = rel_path.replace('\\', '/')
                        files.append(rel_path)

        # Sort files alphabetically (except first entry which is "(none)")
        files_to_sort = files[1:]
        files_to_sort.sort()
        files = ["(none)"] + files_to_sort

        return {
            "required": {
                "image": (files, {"image_upload": True}),
                "frame_position": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            },
            "optional": {
                "lora_stack_a": ("LORA_STACK",),
                "lora_stack_b": ("LORA_STACK",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    CATEGORY = "Prompt Manager"
    RETURN_TYPES = ("STRING", "STRING", "LORA_STACK", "LORA_STACK", "IMAGE")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "lora_stack_a", "lora_stack_b", "image")
    FUNCTION = "extract"
    OUTPUT_NODE = True  # Enable preview display

    def extract(self, image="", frame_position=0.0, lora_stack_a=None, lora_stack_b=None, unique_id=None):
        """Extract prompts and LoRAs from the specified file."""

        # Handle None for frame_position (workflow compatibility)
        if frame_position is None:
            frame_position = 0.0

        # Always initialize with empty strings, never None
        positive_prompt = ""
        negative_prompt = ""
        extracted_lora_stack_a = []
        extracted_lora_stack_b = []
        image_tensor = None

        # Handle None or missing image parameter
        if image is None:
            image = ""

        # Skip metadata extraction if "(none)" is selected (passthrough mode)
        if image == "(none)":
            image = ""

        # Normalize file path
        resolved_path = None
        file_path = ""  # Initialize file_path to avoid UnboundLocalError
        if image and image.strip():
            file_path = image.strip()

            # Handle relative paths (check input directory first, then temp as fallback)
            if not os.path.isabs(file_path):
                # Check input directory first (this is where files should be)
                input_dir = folder_paths.get_input_directory()
                potential_path = os.path.join(input_dir, file_path)
                if os.path.exists(potential_path):
                    resolved_path = potential_path
                else:
                    # Check temp directory as fallback (for backwards compatibility)
                    temp_dir = folder_paths.get_temp_directory()
                    potential_path = os.path.join(temp_dir, file_path)
                    if os.path.exists(potential_path):
                        resolved_path = potential_path
                    else:
                        print(f"[PromptExtractor] File not found in input or temp directories: {file_path}")
            else:
                if os.path.exists(file_path):
                    resolved_path = file_path
                else:
                    print(f"[PromptExtractor] Absolute path does not exist: {file_path}")

        if resolved_path:
            print(f"[PromptExtractor] Processing file: {resolved_path}")
            ext = os.path.splitext(resolved_path)[1].lower()

            prompt_data = None
            workflow_data = None
            loras_a = []
            loras_b = []

            # Extract based on file type
            if ext == '.png':
                prompt_data, workflow_data = extract_metadata_from_png(resolved_path)
                image_tensor = load_image_as_tensor(resolved_path)

            elif ext in ['.jpg', '.jpeg', '.webp']:
                prompt_data, workflow_data = extract_metadata_from_jpeg(resolved_path)
                image_tensor = load_image_as_tensor(resolved_path)

            elif ext == '.json':
                prompt_data, workflow_data = extract_metadata_from_json(resolved_path)
                print(f"[PromptExtractor] JSON extraction: prompt_data={prompt_data is not None}, workflow_data={workflow_data is not None}")
                # No image for JSON files

            elif ext in ['.mp4', '.webm', '.mov', '.avi']:
                prompt_data, workflow_data = extract_metadata_from_video(resolved_path)
                # Get frame cached by JavaScript at the specified position
                # Get relative path from input directory to match JavaScript cache keys
                input_dir = folder_paths.get_input_directory()
                if resolved_path.startswith(input_dir):
                    relative_path = os.path.relpath(resolved_path, input_dir)
                    relative_path = relative_path.replace('\\', '/')  # Normalize to forward slashes
                else:
                    relative_path = os.path.basename(resolved_path)
                image_tensor = get_cached_video_frame(relative_path, frame_position)

                # If no cached frame, use placeholder (JavaScript will cache before execution)
                if image_tensor is None:
                    image_tensor = get_placeholder_image_tensor()

            # Parse the extracted data
            if prompt_data or workflow_data:
                print("[PromptExtractor] Successfully extracted metadata")
                parsed = parse_workflow_for_prompts(prompt_data, workflow_data)
                positive_prompt = parsed['positive_prompt'] or ""
                negative_prompt = parsed['negative_prompt'] or ""
                loras_a = parsed['loras_a']
                loras_b = parsed['loras_b']

            # Process loras_a into stack A (only active LoRAs)
            for lora in loras_a:
                # Skip inactive LoRAs
                if not lora.get('active', True):
                    continue

                # Normalize path separators based on OS
                lora_path = lora['name']
                if os.name == 'nt':  # Windows
                    lora_path = lora_path.replace('/', '\\')
                else:  # Linux/Mac
                    lora_path = lora_path.replace('\\', '/')
                lora_name = os.path.basename(lora_path)
                model_strength = lora['model_strength']
                clip_strength = lora['clip_strength']

                # PromptManagerAdvanced handles matching to actual files
                extracted_lora_stack_a.append((lora_path, model_strength, clip_strength))

            # Process loras_b into stack B (only active LoRAs)
            for lora in loras_b:
                # Skip inactive LoRAs
                if not lora.get('active', True):
                    continue

                # Normalize path separators based on OS
                lora_path = lora['name']
                if os.name == 'nt':  # Windows
                    lora_path = lora_path.replace('/', '\\')
                else:  # Linux/Mac
                    lora_path = lora_path.replace('\\', '/')
                lora_name = os.path.basename(lora_path)
                model_strength = lora['model_strength']
                clip_strength = lora['clip_strength']

                # PromptManagerAdvanced handles matching to actual files
                extracted_lora_stack_b.append((lora_path, model_strength, clip_strength))
        else:
            if file_path:
                print(f"[PromptExtractor] File not found: {file_path}")

        # Combine input lora stacks with extracted loras
        # Extracted loras come first (to preserve workflow strengths), then input loras are appended (avoiding duplicates)
        final_lora_stack_a = []
        final_lora_stack_b = []

        # Start with extracted loras (these have the correct strengths from the workflow)
        final_lora_stack_a.extend(extracted_lora_stack_a)
        final_lora_stack_b.extend(extracted_lora_stack_b)

        # Build set of existing lora names (compare by base name only, without path or extension)
        existing_names_a = {os.path.splitext(os.path.basename(lora[0]))[0].lower() for lora in final_lora_stack_a}
        existing_names_b = {os.path.splitext(os.path.basename(lora[0]))[0].lower() for lora in final_lora_stack_b}

        # Add input loras (skip duplicates)
        if lora_stack_a is not None and isinstance(lora_stack_a, list):
            added_count = 0
            skipped_count = 0
            for lora in lora_stack_a:
                lora_basename = os.path.splitext(os.path.basename(lora[0]))[0].lower()
                if lora_basename not in existing_names_a:
                    final_lora_stack_a.append(lora)
                    added_count += 1
                else:
                    skipped_count += 1

        if lora_stack_b is not None and isinstance(lora_stack_b, list):
            added_count = 0
            skipped_count = 0
            for lora in lora_stack_b:
                lora_basename = os.path.splitext(os.path.basename(lora[0]))[0].lower()
                if lora_basename not in existing_names_b:
                    final_lora_stack_b.append(lora)
                    added_count += 1
                else:
                    skipped_count += 1

        # Provide placeholder image tensor if none loaded (e.g., for JSON files)
        if image_tensor is None:
            image_tensor = get_placeholder_image_tensor()

        # Save preview image for display
        preview_images = self.save_preview_images(image_tensor)

        # Ensure strings are never empty - ComfyUI treats "" as "no connection"
        # Use single space " " to indicate "connected but no content found"
        if not positive_prompt:
            positive_prompt = " "
        if not negative_prompt:
            negative_prompt = " "

        return {
            "ui": {"images": preview_images},
            "result": (positive_prompt, negative_prompt, final_lora_stack_a, final_lora_stack_b, image_tensor)
        }

    def save_preview_images(self, images):
        """Save images temporarily for preview display"""
        import random
        results = []

        output_dir = folder_paths.get_temp_directory()

        for i in range(images.shape[0]):
            img = images[i]
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)

            # Generate unique filename
            filename = f"prompt_extractor_preview_{random.randint(0, 0xFFFFFF):06x}.png"
            filepath = os.path.join(output_dir, filename)
            pil_img.save(filepath)

            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "temp"
            })

        return results

    @classmethod
    def IS_CHANGED(cls, image, frame_position, **kwargs):
        """
        Check if the file has changed or if frame position changed.
        Returns a tuple of (file_modification_time, frame_position) to track changes.
        """
        mtime = "no_file"
        if image:
            file_path = image.strip()
            # Handle relative paths
            if not os.path.isabs(file_path):
                input_dir = folder_paths.get_input_directory()
                potential_path = os.path.join(input_dir, file_path)
                if os.path.exists(potential_path):
                    mtime = os.path.getmtime(potential_path)
            elif os.path.exists(file_path):
                mtime = os.path.getmtime(file_path)

        return (mtime, frame_position)
