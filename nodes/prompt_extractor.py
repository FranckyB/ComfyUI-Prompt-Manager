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
import comfy.samplers

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
# Cache for last extracted info per node (keyed by unique_id) — used by WorkflowBuilder's
# "Update Workflow" button to pull data without re-executing PromptExtractor.
_last_extracted_info = {}


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


# ── A1111 → ComfyUI name mappings ─────────────────────────────────────────────
# A1111 uses human-readable names (e.g. "DPM++ 2M SDE"); ComfyUI uses
# k-diffusion internal names (e.g. "dpmpp_2m_sde").  Lookup is case-insensitive.
_A1111_SAMPLER_MAP = {
    "euler":                "euler",
    "euler a":              "euler_ancestral",
    "lms":                  "lms",
    "heun":                 "heun",
    "heun++":               "heunpp2",
    "dpm2":                 "dpm_2",
    "dpm2 a":               "dpm_2_ancestral",
    "dpm fast":             "dpm_fast",
    "dpm adaptive":         "dpm_adaptive",
    "dpm++ sde":            "dpmpp_sde",
    "dpm++ 2s a":           "dpmpp_2s_ancestral",
    "dpm++ 2m":             "dpmpp_2m",
    "dpm++ 2m sde":         "dpmpp_2m_sde",
    "dpm++ 2m sde heun":    "dpmpp_2m_sde",   # no exact ComfyUI equivalent
    "dpm++ 3m sde":         "dpmpp_3m_sde",
    "ddim":                 "ddim",
    "plms":                 "ipndm",           # closest equivalent
    "ddpm":                 "ddpm",
    "unipc":                "uni_pc",
    "uni_pc":               "uni_pc",
    "uni_pc_bh2":           "uni_pc_bh2",
    "lcm":                  "lcm",
    "deis":                 "deis",
}

_A1111_SCHEDULER_MAP = {
    "karras":               "karras",
    "exponential":          "exponential",
    "sgm uniform":          "sgm_uniform",
    "uniform":              "normal",
    "normal":               "normal",
    "simple":               "simple",
    "ddim":                 "ddim_uniform",
    "beta":                 "beta",
    "polyexponential":      "exponential",     # closest available
    "align your steps":     "normal",          # no ComfyUI equivalent
    "kl optimal":           "normal",          # no ComfyUI equivalent
    "automatic":            "normal",
}


def _map_a1111_sampler(name):
    """Convert an A1111 sampler name to its ComfyUI equivalent.
    Returns 'euler' if the mapped value isn't a valid ComfyUI sampler."""
    if not name:
        return 'euler'
    mapped = _A1111_SAMPLER_MAP.get(name.lower().strip(), name)
    if mapped not in comfy.samplers.KSampler.SAMPLERS:
        return 'euler'
    return mapped


def _map_a1111_scheduler(name):
    """Convert an A1111 scheduler / schedule-type to its ComfyUI equivalent.
    Returns 'simple' if the mapped value isn't a valid ComfyUI scheduler."""
    if not name:
        return 'simple'
    mapped = _A1111_SCHEDULER_MAP.get(name.lower().strip(), name)
    if mapped not in comfy.samplers.KSampler.SCHEDULERS:
        return 'simple'
    return mapped


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

    # Extract model name from settings line (e.g. "Model: modelName")
    model_match = re.search(r'\bModel:\s*([^,\n]+)', parameters_text)
    if model_match:
        result['model'] = model_match.group(1).strip()

    # ── Extract sampler / resolution / generation settings ────────────────
    # A1111 format: "Steps: 20, Sampler: DPM++ 2M SDE, Schedule type: Karras,
    #                CFG scale: 5, Seed: 2427658518, Size: 768x1152, ..."
    settings_line = ''
    steps_pos = parameters_text.find('Steps:')
    if steps_pos >= 0:
        settings_line = parameters_text[steps_pos:]

    if settings_line:
        def _a1111_val(key, text=settings_line):
            m = re.search(r'\b' + re.escape(key) + r':\s*([^,\n]+)', text)
            return m.group(1).strip() if m else None

        steps = _a1111_val('Steps')
        if steps:
            try:
                result['steps'] = int(steps)
            except ValueError:
                pass

        sampler = _a1111_val('Sampler')
        if sampler:
            result['sampler_name'] = _map_a1111_sampler(sampler)

        schedule = _a1111_val('Schedule type')
        if schedule:
            result['scheduler'] = _map_a1111_scheduler(schedule)

        cfg = _a1111_val('CFG scale')
        if cfg:
            try: 
                result['cfg'] = float(cfg)
            except ValueError:
                pass

        seed = _a1111_val('Seed')
        if seed:
            try:
                result['seed'] = int(seed)
            except ValueError:
                pass

        size = _a1111_val('Size')
        if size:
            m = re.match(r'(\d+)\s*x\s*(\d+)', size)
            if m:
                result['width'] = int(m.group(1))
                result['height'] = int(m.group(2))

        # ── Extract Forge/A1111 Module fields ────────────────────────────
        # Forge embeds CLIP/VAE module names as "Module 1: ae, Module 2: clip_l, Module 3: t5xxl_fp16"
        # These are reliable family indicators when the model name is unrecognised.
        modules = []
        for i in range(1, 5):
            mod = _a1111_val(f'Module {i}')
            if mod:
                modules.append(mod.lower())
        if modules:
            result['modules'] = modules

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
        source = data.get('source', 'input')

        print(f"[PromptExtractor] JavaScript requested ffprobe extraction for: {filename}")

        if not filename:
            return server.web.json_response({"success": False, "error": "Missing filename"}, status=400)

        # Build full path based on source folder
        if source == 'output':
            base_dir = folder_paths.get_output_directory()
        else:
            base_dir = folder_paths.get_input_directory()
        file_path = os.path.join(base_dir, filename.replace('/', os.sep))

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


# API endpoint to list files in input or output directory
@server.PromptServer.instance.routes.get("/prompt-extractor/list-files")
async def list_input_files(request):
    """API endpoint to get list of supported files in input or output directory, including subfolders"""
    try:
        source = request.rel_url.query.get('source', 'input')
        if source == 'output':
            base_dir = folder_paths.get_output_directory()
        else:
            base_dir = folder_paths.get_input_directory()

        files = []
        supported_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.json', '.mp4', '.webm', '.mov', '.avi']

        if os.path.exists(base_dir):
            # Walk through directory recursively
            for root, dirs, filenames in os.walk(base_dir):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in supported_extensions:
                        # Get relative path from base directory
                        full_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(full_path, base_dir)
                        # Convert Windows backslashes to forward slashes
                        rel_path = rel_path.replace('\\', '/')
                        files.append(rel_path)

        files.sort()
        return server.web.json_response({"files": files})
    except Exception as e:
        print(f"[PromptExtractor] Error listing files: {e}")
        return server.web.json_response({"files": [], "error": str(e)}, status=500)


# API endpoint to get cached extracted info for WorkflowBuilder's "Update Workflow" button.
# Accepts ?node_id=<id> for a specific PE node, or returns all cached entries.
@server.PromptServer.instance.routes.get("/prompt-extractor/get-extracted-data")
async def get_extracted_data(request):
    """Return the last extracted info cached by PromptExtractor after execution."""
    try:
        node_id = request.rel_url.query.get('node_id', '')
        if node_id:
            data = _last_extracted_info.get(str(node_id))
            if data:
                return server.web.json_response({"extracted": data, "node_id": node_id})
            else:
                return server.web.json_response({"extracted": None, "node_id": node_id,
                                                  "error": "No cached data for this node. Execute PromptExtractor first."})
        else:
            # Return all cached node IDs so JS can pick
            available = {nid: bool(d) for nid, d in _last_extracted_info.items()}
            return server.web.json_response({"available": available})
    except Exception as e:
        print(f"[PromptExtractor] Error in get-extracted-data: {e}")
        return server.web.json_response({"extracted": None, "error": str(e)}, status=500)


# API endpoint to extract metadata from a file on demand (no node execution required).
# Used by WorkflowBuilder's "Update Workflow" button and PE JS on file selection.
@server.PromptServer.instance.routes.get("/prompt-extractor/extract-preview")
async def extract_preview(request):
    """Extract metadata from a file and return structured info for WorkflowBuilder."""
    try:
        filename = request.rel_url.query.get('filename', '')
        source = request.rel_url.query.get('source', 'input')

        if not filename or filename == '(none)':
            return server.web.json_response({"extracted": None})

        # Resolve path with directory-traversal protection
        base_dir = folder_paths.get_output_directory() if source == 'output' else folder_paths.get_input_directory()
        file_path = os.path.join(base_dir, filename.replace('/', os.sep))
        real_base = os.path.realpath(base_dir)
        real_path = os.path.realpath(file_path)
        if not (real_path == real_base or real_path.startswith(real_base + os.sep)):
            return server.web.json_response({"extracted": None, "error": "Invalid path"}, status=403)
        if not os.path.exists(file_path):
            return server.web.json_response({"extracted": None, "error": "File not found"})

        ext = os.path.splitext(file_path)[1].lower()
        prompt_data = None
        wf_data = None

        if ext == '.png':
            prompt_data, wf_data = extract_metadata_from_png(file_path)
        elif ext in ['.jpg', '.jpeg', '.webp']:
            prompt_data, wf_data = extract_metadata_from_jpeg(file_path)
        elif ext == '.json':
            prompt_data, wf_data = extract_metadata_from_json(file_path)
        elif ext in ['.mp4', '.webm', '.mov', '.avi']:
            prompt_data, wf_data = extract_metadata_from_video(file_path)

        if not prompt_data and not wf_data:
            return server.web.json_response({"extracted": None, "error": "No metadata found in file"})

        parsed = parse_workflow_for_prompts(prompt_data, wf_data)
        pos = parsed['positive_prompt'] or ""
        neg = parsed['negative_prompt'] or ""
        loras_a = parsed['loras_a']
        loras_b = parsed['loras_b']

        model_a = ""
        model_b = ""
        raw_a = parsed.get('models_a', [])
        raw_b = parsed.get('models_b', [])
        if raw_a:
            model_a = os.path.basename(raw_a[0].replace('\\', '/'))
        if raw_b:
            model_b = os.path.basename(raw_b[0].replace('\\', '/'))

        from ..py.workflow_extraction_utils import (
            extract_sampler_params, extract_vae_info, extract_clip_info,
            extract_resolution, resolve_model_name, resolve_vae_name,
        )
        from ..py.workflow_families import get_model_family, get_family_label

        sampler = extract_sampler_params(prompt_data, wf_data)
        vae = extract_vae_info(prompt_data, wf_data)
        clip = extract_clip_info(prompt_data, wf_data)
        resolution = extract_resolution(prompt_data, wf_data)

        # Handle A1111 format
        is_a1111 = isinstance(prompt_data, dict) and 'prompt' in prompt_data and 'loras' in prompt_data
        if is_a1111:
            for key in ['sampler_name', 'scheduler', 'cfg']:
                if prompt_data.get(key) is not None:
                    sampler[key] = prompt_data[key]
            if prompt_data.get('steps') is not None:
                sampler['steps_a'] = prompt_data['steps']
            if prompt_data.get('seed') is not None:
                sampler['seed_a'] = prompt_data['seed']
            if prompt_data.get('width') and prompt_data.get('height'):
                resolution['width'] = prompt_data['width']
                resolution['height'] = prompt_data['height']
            # Forge modules → clip_type inference
            _modules = prompt_data.get('modules', [])
            if _modules and not clip.get('type'):
                _mod_str = ' '.join(_modules)
                if 'qwen_3_8b' in _mod_str:
                    clip['type'] = 'flux2'
                    clip['source'] = 'a1111_modules'
                elif 'qwen_3_4b' in _mod_str or 'qwen-4b' in _mod_str:
                    clip['type'] = 'lumina2'
                    clip['source'] = 'a1111_modules'
                elif 't5xxl' in _mod_str:
                    clip['type'] = 'flux'
                    clip['source'] = 'a1111_modules'
                elif 'umt5' in _mod_str:
                    clip['type'] = 'wan'
                    clip['source'] = 'a1111_modules'

        _model_a_path = raw_a[0] if raw_a else model_a
        family = get_model_family(_model_a_path)

        # Fallback: infer family from clip type (same logic as PE execute path)
        if not family:
            _clip_src  = clip.get('source', '')
            _clip_type = clip.get('type', '').lower()
            if _clip_src == 'checkpoint':
                family = 'sdxl'
            elif 'flux2' in _clip_type:
                family = 'flux2'
            elif 'flux' in _clip_type:
                family = 'flux1'
            elif 'sd3' in _clip_type:
                family = 'sd3'
            elif 'wan' in _clip_type:
                family = 'wan_video_t2v'
            elif 'qwen_image' in _clip_type:
                family = 'qwen_image'
            elif 'lumina2' in _clip_type:
                family = 'zimage'
        if not family:
            family = 'sdxl'

        # Check availability
        model_a_found = True
        model_b_found = True
        if model_a:
            _r, _ = resolve_model_name(model_a)
            model_a_found = _r is not None
        if model_b:
            _r, _ = resolve_model_name(model_b)
            model_b_found = _r is not None

        vae_found = True
        vae_name_str = vae.get('name', '') if isinstance(vae, dict) else (vae or '')
        if vae_name_str and not vae_name_str.startswith('('):
            vae_found = resolve_vae_name(vae_name_str) is not None

        lora_avail = {}
        for _l in loras_a + loras_b:
            _ln = _l.get('name', '')
            if _ln:
                _, _found = resolve_lora_path(_ln)
                lora_avail[_ln] = _found

        extracted = {
            'positive_prompt':    pos,
            'negative_prompt':    neg,
            'model_a':            model_a,
            'model_b':            model_b,
            'model_a_found':      model_a_found,
            'model_b_found':      model_b_found,
            'loras_a':            loras_a,
            'loras_b':            loras_b,
            'vae':                vae,
            'vae_found':          vae_found,
            'clip':               clip,
            'sampler':            sampler,
            'resolution':         resolution,
            'is_video':           ext in ['.mp4', '.webm', '.mov', '.avi'],
            'model_family':       family,
            'model_family_label': get_family_label(family),
            'lora_availability':  lora_avail,
        }

        print(f"[PromptExtractor] extract-preview: {filename} -> {family}, model_a={model_a}")
        return server.web.json_response({"extracted": extracted})
    except Exception as e:
        print(f"[PromptExtractor] extract-preview error: {e}")
        import traceback
        traceback.print_exc()
        return server.web.json_response({"extracted": None, "error": str(e)}, status=500)


# ── Shared LoRA utilities ─────────────────────────────────────────────────────
from ..py.lora_utils import (
    get_available_loras,
    resolve_lora_path,
)


# Known model file extensions
MODEL_EXTENSIONS = ['.safetensors', '.ckpt', '.pt', '.bin', '.pth', '.gguf']


def strip_model_extension(name):
    """Remove known model file extensions from a name."""
    name_lower = name.lower()
    for ext in MODEL_EXTENSIONS:
        if name_lower.endswith(ext):
            return name[:-len(ext)]
    return name


def get_available_models():
    """Get all available models from ComfyUI's checkpoints, diffusion_models and unet folders."""
    models = []
    seen = set()
    for folder_name in ['checkpoints', 'diffusion_models', 'unet', 'unet_gguf']:
        try:
            for m in folder_paths.get_filename_list(folder_name):
                if m not in seen:
                    seen.add(m)
                    models.append(m)
        except Exception:
            pass
    return models


def resolve_model_path(model_name):
    """
    Resolve a model name to its relative path in ComfyUI's folder system.
    Strips any path prefix from the extracted name and searches by basename.
    Returns (resolved_path, found) tuple.
    If not found, returns the sanitized basename (so the user gets a clear 'not found' error).
    """
    if not model_name:
        return "", False

    available_models = get_available_models()

    # Sanitize: extract just the filename (strip any directory path from the source workflow)
    sanitized_name = os.path.basename(model_name.replace('\\', '/'))
    sanitized_name_no_ext = strip_model_extension(sanitized_name).lower()

    # Try exact match first (full relative path as stored in metadata)
    for model_file in available_models:
        if model_file == model_name or model_file.replace('/', '\\') == model_name.replace('/', '\\'):
            return model_file, True

    # Try matching by basename without extension
    for model_file in available_models:
        file_basename_no_ext = strip_model_extension(os.path.basename(model_file)).lower()
        if file_basename_no_ext == sanitized_name_no_ext:
            return model_file, True

    # Not found - return sanitized name so ComfyUI shows a clear error
    print(f"[PromptExtractor] Model not found: {model_name} (searched as '{sanitized_name}')")
    return sanitized_name, False


def extract_metadata_from_png(file_path):
    """Extract workflow/prompt metadata from PNG file (cached from JavaScript)"""
    try:
        # Try to get relative path from input or output directory (matches JavaScript cache keys)
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()
        if file_path.startswith(input_dir):
            cache_key = os.path.relpath(file_path, input_dir).replace('\\', '/')
        elif file_path.startswith(output_dir):
            cache_key = os.path.relpath(file_path, output_dir).replace('\\', '/')
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
                # Return both parsed parameters AND workflow data (workflow needed for JSON export)
                if metadata.get('parsed_parameters'):
                    print("[PromptExtractor] Found parsed A1111 parameters")
                    parsed = metadata['parsed_parameters']
                    # JS parser doesn't extract sampler/resolution/modules — enrich
                    # from the raw parameters text via the Python parser.
                    raw_params = metadata.get('parameters', '')
                    if raw_params:
                        py_parsed = parse_a1111_parameters(raw_params)
                        if py_parsed:
                            for k in ('steps', 'sampler_name', 'scheduler', 'cfg',
                                      'seed', 'width', 'height', 'modules'):
                                if k in py_parsed and k not in parsed:
                                    parsed[k] = py_parsed[k]
                    return parsed, workflow_data

                return prompt_data, workflow_data

        # Fallback to PIL if no cached data (backwards compatibility)
        if not IMAGE_SUPPORT:
            return None, None

        print(f"[PromptExtractor] Falling back to PIL for: {file_path}")
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
        # Try to get relative path from input or output directory (matches JavaScript cache keys)
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()
        if file_path.startswith(input_dir):
            cache_key = os.path.relpath(file_path, input_dir).replace('\\', '/')
        elif file_path.startswith(output_dir):
            cache_key = os.path.relpath(file_path, output_dir).replace('\\', '/')
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
            print(f"[PromptExtractor] No cached metadata found for JPEG/WebP: {file_path}")
            print("[PromptExtractor] Note: Image metadata is read by JavaScript when file is selected")

        # Fallback to PIL if no cached data (backwards compatibility)
        if not IMAGE_SUPPORT:
            return None, None

        print(f"[PromptExtractor] Falling back to PIL for: {file_path}")
        with Image.open(file_path) as img:
            # Try EXIF data
            exif = img.getexif()
            if exif:
                prompt_data = None
                workflow_data = None

                # ComfyUI stores metadata in EXIF tags:
                # 0x010e (ImageDescription): "Workflow: {json}"
                # 0x010f (Make): "Prompt: {json}"
                for tag_id in (0x010e, 0x010f):
                    tag_val = exif.get(tag_id)
                    if tag_val:
                        if isinstance(tag_val, bytes):
                            tag_val = tag_val.decode('utf-8', errors='ignore')
                        tag_val = tag_val.strip().rstrip('\x00')
                        if tag_val.startswith('Workflow:'):
                            json_str = tag_val[len('Workflow:'):].strip()
                            try:
                                workflow_data = json.loads(json_str)
                            except:
                                pass
                        elif tag_val.startswith('Prompt:'):
                            json_str = tag_val[len('Prompt:'):].strip()
                            try:
                                prompt_data = json.loads(json_str)
                            except:
                                pass

                if prompt_data or workflow_data:
                    return prompt_data, workflow_data

                # Fallback: UserComment field (0x9286) - used by some tools
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
        # Try to get relative path from input or output directory (matches JavaScript cache keys)
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()
        if file_path.startswith(input_dir):
            cache_key = os.path.relpath(file_path, input_dir).replace('\\', '/')
        elif file_path.startswith(output_dir):
            cache_key = os.path.relpath(file_path, output_dir).replace('\\', '/')
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
        # Get the relative path from input or output directory to match JavaScript cache keys
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()
        if file_path.startswith(input_dir):
            cache_key = os.path.relpath(file_path, input_dir).replace('\\', '/')
        elif file_path.startswith(output_dir):
            cache_key = os.path.relpath(file_path, output_dir).replace('\\', '/')
        else:
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

                        # Extract prompt if present as direct tag
                        if 'prompt' in tags:
                            try:
                                prompt_val = json.loads(tags['prompt'])
                            except:
                                prompt_val = tags['prompt']

                        # Extract workflow if present as direct tag
                        if 'workflow' in tags:
                            try:
                                workflow_val = json.loads(tags['workflow'])
                            except:
                                workflow_val = tags['workflow']

                        # Fallback: check 'comment' tag which may contain JSON with prompt/workflow
                        if not prompt_val and not workflow_val and 'comment' in tags:
                            try:
                                comment_data = json.loads(tags['comment'])
                                if isinstance(comment_data, dict):
                                    if 'prompt' in comment_data:
                                        try:
                                            prompt_val = json.loads(comment_data['prompt']) if isinstance(comment_data['prompt'], str) else comment_data['prompt']
                                        except:
                                            prompt_val = comment_data['prompt']
                                    if 'workflow' in comment_data:
                                        try:
                                            workflow_val = json.loads(comment_data['workflow']) if isinstance(comment_data['workflow'], str) else comment_data['workflow']
                                        except:
                                            workflow_val = comment_data['workflow']
                            except (json.JSONDecodeError, TypeError):
                                pass

                        if prompt_val or workflow_val:
                            print("[PromptExtractor] Successfully extracted metadata using ffprobe")
                            # Cache it for next time
                            _file_metadata_cache[cache_key] = {
                                'prompt': prompt_val,
                                'workflow': workflow_val
                            }
                            return prompt_val, workflow_val

                print("[PromptExtractor] No metadata found in video with ffprobe")
            except FileNotFoundError:
                print("[PromptExtractor] ffprobe not found - cannot extract video metadata")
            except Exception as e:
                print("[PromptExtractor] ffprobe extraction failed: {e}")

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
    Falls back to server-side extraction using PyAV for H265/yuv444 videos.

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
                print("[PromptExtractor] Timeout waiting for JS frame extraction, trying PyAV...")
                return None

        except Exception as e:
            print(f"[PromptExtractor] Error requesting frame extraction: {e}")
            return None

    frame_data = _video_frames_cache[path_key]

    # Convert base64 frame to tensor
    frame_tensor = base64_to_tensor(frame_data)

    return frame_tensor


def extract_video_frame_av(file_path, frame_position=0.0):
    """
    Extract a video frame using PyAV. Works with H265, yuv444, and other codecs
    that browsers cannot decode.

    Seeks to the nearest keyframe before the target position, then decodes forward
    to the exact target frame for frame-accurate extraction.

    Args:
        file_path: Absolute path to the video file
        frame_position: float from 0.0 to 1.0 representing position in video

    Returns:
        PIL Image or None on failure
    """
    try:
        import av
    except ImportError:
        print("[PromptExtractor] PyAV not available, cannot extract frame server-side")
        return None

    try:
        container = av.open(file_path)
        stream = container.streams.video[0]

        # For position 0.0, just return the first frame
        if frame_position <= 0.0:
            for frame in container.decode(video=0):
                img = frame.to_image()
                container.close()
                return img
            container.close()
            return None

        # Calculate target timestamp in stream time_base units
        duration = stream.duration
        target_ts = None

        if duration and stream.time_base:
            target_ts = int(frame_position * duration)
        else:
            # Fallback: estimate from frame count and average rate
            total_frames = stream.frames
            if total_frames > 0 and stream.average_rate:
                target_frame = int(frame_position * total_frames)
                fps = float(stream.average_rate)
                if fps > 0 and stream.time_base:
                    target_sec = target_frame / fps
                    target_ts = int(target_sec / float(stream.time_base))

        if target_ts is not None:
            # Seek to nearest keyframe before target (backward seek)
            container.seek(target_ts, stream=stream, backward=True)

            # Decode forward until we reach or pass the target timestamp
            best_frame = None
            for frame in container.decode(video=0):
                best_frame = frame
                if frame.pts is not None and frame.pts >= target_ts:
                    break

            if best_frame is not None:
                img = best_frame.to_image()
                container.close()
                return img
        else:
            # No duration info - just decode the first frame
            for frame in container.decode(video=0):
                img = frame.to_image()
                container.close()
                return img

        container.close()
        return None
    except Exception as e:
        print(f"[PromptExtractor] PyAV frame extraction error: {e}")
        return None


def extract_video_frame_av_to_tensor(file_path, frame_position=0.0):
    """
    Extract a video frame using PyAV and return as a ComfyUI tensor.

    Args:
        file_path: Absolute path to the video file
        frame_position: float from 0.0 to 1.0

    Returns:
        Tensor (B, H, W, C) or None
    """
    img = extract_video_frame_av(file_path, frame_position)
    if img is None:
        return None

    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_array).unsqueeze(0)
    except Exception as e:
        print(f"[PromptExtractor] Error converting PyAV frame to tensor: {e}")
        return None


# API endpoint to extract a video frame server-side using PyAV (for H265/yuv444)
@server.PromptServer.instance.routes.get("/prompt-extractor/video-frame")
async def extract_video_frame_api(request):
    """Extract a video frame server-side for videos the browser can't decode (H265, yuv444)"""
    try:
        filename = request.rel_url.query.get('filename', '')
        source = request.rel_url.query.get('source', 'input')
        position = float(request.rel_url.query.get('position', '0.0'))

        if not filename:
            return server.web.json_response({"error": "Missing filename"}, status=400)

        # Build full path
        if source == 'output':
            base_dir = folder_paths.get_output_directory()
        else:
            base_dir = folder_paths.get_input_directory()

        file_path = os.path.join(base_dir, filename.replace('/', os.sep))

        if not os.path.exists(file_path):
            return server.web.json_response({"error": "File not found"}, status=404)

        # Validate path stays within base directory
        real_base = os.path.realpath(base_dir)
        real_path = os.path.realpath(file_path)
        if not real_path.startswith(real_base):
            return server.web.json_response({"error": "Invalid path"}, status=403)

        img = extract_video_frame_av(file_path, position)
        if img is None:
            return server.web.json_response({"error": "Failed to extract frame"}, status=500)

        # Return as JPEG image
        import io
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        buf.seek(0)

        return server.web.Response(
            body=buf.read(),
            content_type='image/jpeg'
        )
    except Exception as e:
        print(f"[PromptExtractor] Error in video-frame API: {e}")
        return server.web.json_response({"error": str(e)}, status=500)


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

    # PromptExtractor / WorkflowRenderer — text outputs are computed at runtime,
    # NOT stored in widgets_values (which contain file selector, toggles, etc.).
    # Without this guard the generic fallback below picks up the image filename.
    if node_type in ['PromptExtractor', 'WorkflowRenderer', 'WorkflowGenerator']:
        return ""

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


# Node types that load checkpoints or diffusion models (not LoRA, VAE, or CLIP loaders)
MODEL_LOADER_TYPES = [
    'CheckpointLoader',
    'CheckpointLoaderSimple',
    'CheckpointLoaderKJ',
    'CheckpointLoaderNF4',
    'UNETLoader',
    'UnetLoaderGGUF',
    'DiffusionModelLoader',
    'DiffusionModelLoaderKJ',
    'WanVideoModelLoader',
    'SeaArtUnetLoader',
    'CyberdyneModelHub',
    'PromptModelLoader',
]


def is_model_loader_node(node_type):
    """Check if a node type is a checkpoint or diffusion model loader"""
    return node_type in MODEL_LOADER_TYPES


def get_model_name_from_node(node, prompt_node=None):
    """
    Extract the model/checkpoint name from a model loader node.
    Checks both workflow widgets_values and API-format inputs.
    Returns the model name string or None.
    """
    node_type = node.get('type', '')

    # From API format (prompt_data) — most reliable for active nodes
    if prompt_node and isinstance(prompt_node, dict):
        inputs = prompt_node.get('inputs', {})
        for key in ['ckpt_name', 'unet_name', 'model_name', 'diffusion_model', 'model', 'model_path']:
            val = inputs.get(key)
            if val and isinstance(val, str):
                return val

    # From workflow widgets_values — fallback
    widgets = node.get('widgets_values', [])
    if widgets and isinstance(widgets[0], str):
        return widgets[0]

    return None


def trace_to_model_loader(node_id, node_map, link_map, visited=None, max_depth=20):
    """
    Trace backwards through MODEL connections to find the model loader node at the root.
    Passes through LoRA nodes, ModelSamplingSD3, etc.
    Returns the model loader node ID, or None if not found.
    """
    if visited is None:
        visited = set()
    if max_depth <= 0 or node_id in visited:
        return None

    visited.add(node_id)
    node = node_map.get(node_id)
    if not node:
        return None

    if is_model_loader_node(node.get('type', '')):
        return node_id

    # Trace backwards through MODEL input
    inputs = node.get('inputs', [])
    for inp in inputs:
        inp_name = inp.get('name', '')
        inp_type = inp.get('type', '')
        if (inp_name in ['model', 'MODEL'] or inp_type == 'MODEL') and inp.get('link'):
            link_info = link_map.get(inp['link'])
            if link_info:
                result = trace_to_model_loader(link_info['source_node'], node_map, link_map, visited, max_depth - 1)
                if result:
                    return result

    return None


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
        - models_a: list of model names - first/high model chain
        - models_b: list of model names - second/low model chain
    """
    result = {
        'positive_prompt': '',
        'negative_prompt': '',
        'loras_a': [],
        'loras_b': [],
        'models_a': [],
        'models_b': []
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

        # Extract model if present
        model_name = prompt_data.get('model', '')
        if model_name:
            result['models_a'].append(model_name)
            print(f"[PromptExtractor] A1111 Model: {model_name}")

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

    # Track models extracted from our own nodes (PromptExtractor embedded data)
    _pe_extracted_models = []

    # Initialize lora_chains early so embedded data extraction can append to it
    lora_chains = []

    # Collect embedded data candidates from PromptExtractor/WorkflowRenderer nodes
    # (resolved after the loop — WorkflowRenderer takes priority if both are present)
    _embedded_candidates = []

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
            elif node_type == 'PromptManager':
                for val in widgets_values:
                    if isinstance(val, str) and len(val) > 20:
                        positive_prompts.append(val.strip())
                        break

            elif node_type == 'PromptManagerAdvanced':
                # Widget order: [category, name, use_prompt_input, use_lora_input, text, swap_lora_outputs]
                pm_text = widgets_values[4] if len(widgets_values) > 4 else None
                if pm_text and isinstance(pm_text, str) and len(pm_text.strip()) > 0:
                    positive_prompts.append(pm_text.strip())
                else:
                    # Fallback: find first long string
                    for val in widgets_values:
                        if isinstance(val, str) and len(val) > 20:
                            positive_prompts.append(val.strip())
                            break

            # PromptExtractor / WorkflowRenderer nodes — collect embedded extracted_data.
            # If both are present, WorkflowRenderer takes priority (it generated the image).
            # Also accept legacy 'WorkflowGenerator' class_type from older images.
            elif node_type in ('PromptExtractor', 'WorkflowRenderer', 'WorkflowGenerator'):
                ext_data = node.get('extracted_data')
                if ext_data and isinstance(ext_data, dict):
                    _embedded_candidates.append((node_type, node_id, title, ext_data))

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
    # RESOLVE EMBEDDED DATA (PromptExtractor vs WorkflowRenderer)
    # ========================================
    # If both node types are present, use only WorkflowRenderer’s data
    # (it actually generated the image; PromptExtractor merely forwarded).
    if _embedded_candidates:
        has_wg = any(nt in ('WorkflowRenderer', 'WorkflowGenerator') for nt, *_ in _embedded_candidates)
        chosen = [
            c for c in _embedded_candidates
            if not has_wg or c[0] in ('WorkflowRenderer', 'WorkflowGenerator')
        ]
        if has_wg and len(_embedded_candidates) > len(chosen):
            print("[PromptExtractor] Both PromptExtractor and WorkflowRenderer found "
                  "— preferring WorkflowRenderer embedded data")

        for node_type, node_id, title, ext_data in chosen:
            ext_pos = ext_data.get('positive_prompt', '').strip()
            ext_neg = ext_data.get('negative_prompt', '').strip()
            if ext_pos:
                positive_prompts.append(ext_pos)
            if ext_neg:
                negative_prompts.append(ext_neg)

            # Extract LoRAs from embedded data (with active/available state)
            for stack_label, key in [('A', 'loras_a'), ('B', 'loras_b')]:
                ext_loras = ext_data.get(key, [])
                if not ext_loras:
                    continue
                chain_loras_list = []
                for lora_item in ext_loras:
                    if not isinstance(lora_item, dict):
                        continue
                    lora_name = lora_item.get('name', '')
                    if not lora_name or is_lora_blacklisted(lora_name):
                        continue
                    strength = float(lora_item.get('strength', lora_item.get('model_strength', 1.0)))
                    clip_strength = float(lora_item.get('clip_strength', strength))
                    chain_loras_list.append({
                        'name': lora_name,
                        'path': lora_item.get('path', lora_name),
                        'model_strength': strength,
                        'clip_strength': clip_strength,
                        'active': lora_item.get('active', True),
                        'available': lora_item.get('available', True),
                    })
                if chain_loras_list:
                    active_count = sum(1 for lr in chain_loras_list if lr.get('active', True))
                    avail_count = sum(1 for lr in chain_loras_list if lr.get('available', True))
                    print(f"[PromptExtractor] {node_type} embedded data stack {stack_label}: "
                          f"{len(chain_loras_list)} LoRAs ({active_count} active, {avail_count} available)")
                    lora_chains.append({
                        'titles': [title or node_type],
                        'loras': chain_loras_list,
                        'terminal_title': title or node_type,
                        'source_id': node_id,
                        '_pm_stack': stack_label
                    })

            # Extract model info
            ext_model_a = ext_data.get('model_a', '').strip()
            ext_model_b = ext_data.get('model_b', '').strip()
            if ext_model_a:
                _pe_extracted_models.append(('a', ext_model_a, title or node_type))
            if ext_model_b:
                _pe_extracted_models.append(('b', ext_model_b, title or node_type))

    # ========================================
    # UNIFIED LORA CHAIN EXTRACTION
    # ========================================
    # Find all LoRA chains by starting from terminal nodes (non-LoRA nodes that receive MODEL input)
    # and traversing backwards through MODEL connections

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

        # Method 3: Extract LoRAs from PromptManagerAdvanced nodes
        # These store LoRA data in the saved prompt JSON file, keyed by category/name
        _pm_prompts_cache = None
        for node in all_workflow_nodes:
            if node.get('type') != 'PromptManagerAdvanced':
                continue

            wv = node.get('widgets_values', [])
            if len(wv) < 6:
                continue

            pm_category = wv[0] if isinstance(wv[0], str) else None
            pm_name = wv[1] if isinstance(wv[1], str) else None
            pm_swap = wv[5] if len(wv) > 5 else False

            if not pm_category or not pm_name:
                continue

            # Load prompt data from disk (cached)
            if _pm_prompts_cache is None:
                try:
                    pm_data_path = os.path.join(folder_paths.get_user_directory(), "default", "prompt_manager_data.json")
                    if os.path.exists(pm_data_path):
                        with open(pm_data_path, 'r', encoding='utf-8') as f:
                            _pm_prompts_cache = json.load(f)
                    else:
                        _pm_prompts_cache = {}
                except Exception as e:
                    print(f"[PromptExtractor] Could not load prompt manager data: {e}")
                    _pm_prompts_cache = {}

            prompt_entry = _pm_prompts_cache.get(pm_category, {}).get(pm_name, {})
            pm_loras_a = prompt_entry.get('loras_a', [])
            pm_loras_b = prompt_entry.get('loras_b', [])

            if pm_swap:
                pm_loras_a, pm_loras_b = pm_loras_b, pm_loras_a

            node_title = node.get('title', f'PromptManagerAdvanced ({pm_name})')

            for stack_label, pm_loras in [('A', pm_loras_a), ('B', pm_loras_b)]:
                if not pm_loras:
                    continue
                chain_loras_list = []
                for lora_item in pm_loras:
                    if not isinstance(lora_item, dict):
                        continue
                    lora_name = lora_item.get('name', '')
                    if not lora_name:
                        continue
                    if lora_item.get('active', True) is False:
                        continue
                    if is_lora_blacklisted(lora_name):
                        continue
                    strength = float(lora_item.get('strength', 1.0))
                    clip_strength = float(lora_item.get('clip_strength', strength))
                    chain_loras_list.append({
                        'name': lora_name,
                        'path': lora_name,
                        'model_strength': strength,
                        'clip_strength': clip_strength,
                        'active': True
                    })

                if chain_loras_list:
                    # Use explicit stack assignment marker in title
                    stack_title = f"{node_title} [stack_{stack_label.lower()}]"
                    print(f"[PromptExtractor] PromptManagerAdvanced '{pm_name}' stack {stack_label}: {len(chain_loras_list)} LoRAs")
                    lora_chains.append({
                        'titles': [stack_title],
                        'loras': chain_loras_list,
                        'terminal_title': stack_title,
                        'source_id': node.get('id', 0),
                        '_pm_stack': stack_label  # Direct stack assignment marker
                    })

    # ========================================
    # ASSIGN LORA CHAINS TO STACKS A AND B
    # ========================================
    # Based on title hints (high/low) or position

    lora_chains.sort(key=lambda x: x.get('source_id', 0))

    print(f"[PromptExtractor] Processing {len(lora_chains)} chains for stack assignment")
    for i, chain in enumerate(lora_chains):
        # Direct stack assignment from PromptManagerAdvanced nodes
        pm_stack = chain.get('_pm_stack')
        if pm_stack:
            target_stack = loras_a if pm_stack == 'A' else loras_b
            target_seen = lora_names_seen_a if pm_stack == 'A' else lora_names_seen_b
            print(f"[PromptExtractor] Chain {i} → STACK {pm_stack} (PromptManagerAdvanced direct assignment)")
            for lora in chain['loras']:
                if lora['name'] not in target_seen:
                    target_seen.add(lora['name'])
                    target_stack.append(lora)
            continue

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
                # Determine if this is positive or negative by checking connections
                connection_type = None
                if node_map:
                    actual_node_id = int(node_id) if str(node_id).isdigit() else node_id
                    connection_type = determine_clip_text_encode_type(actual_node_id, workflow_data, node_map)

                # Fallback: check node title if we have node_map
                if not connection_type and node_map:
                    node = node_map.get(actual_node_id)
                    if node:
                        title_lower = node.get('title', '').lower()
                        if 'negative' in title_lower:
                            connection_type = 'negative'
                        elif 'positive' in title_lower:
                            connection_type = 'positive'

                # Add to appropriate list (default to positive if unclear)
                if connection_type == 'negative':
                    negative_prompts.append(text)
                else:
                    positive_prompts.append(text)

        # PromptManager nodes (always positive)
        elif class_type == 'PromptManager':
            text = inputs.get('text', '')
            if text and isinstance(text, str):
                positive_prompts.append(text)

        elif class_type == 'PromptManagerAdvanced':
            text = inputs.get('text', '')
            if text and isinstance(text, str):
                positive_prompts.append(text)

            # Extract LoRAs from toggle data (API format has these as JSON strings)
            pm_swap = inputs.get('swap_lora_outputs', False)
            for stack_key, stack_label in [('loras_a_toggle', 'A'), ('loras_b_toggle', 'B')]:
                toggle_raw = inputs.get(stack_key, '')
                if not toggle_raw or not isinstance(toggle_raw, str):
                    continue
                try:
                    toggle_list = json.loads(toggle_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(toggle_list, list):
                    continue

                # Determine actual stack after swap
                actual_label = stack_label
                if pm_swap:
                    actual_label = 'B' if stack_label == 'A' else 'A'
                target_stack = loras_a if actual_label == 'A' else loras_b
                target_seen = lora_names_seen_a if actual_label == 'A' else lora_names_seen_b

                for lora_item in toggle_list:
                    if not isinstance(lora_item, dict):
                        continue
                    lora_name = lora_item.get('name', '')
                    if not lora_name or lora_item.get('active', True) is False:
                        continue
                    if is_lora_blacklisted(lora_name):
                        continue
                    if lora_name not in target_seen:
                        target_seen.add(lora_name)
                        strength = float(lora_item.get('strength', 1.0))
                        clip_strength = float(lora_item.get('clip_strength', strength))
                        target_stack.append({
                            'name': lora_name,
                            'path': lora_name,
                            'model_strength': strength,
                            'clip_strength': clip_strength
                        })

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

    # ========================================
    # MODEL / CHECKPOINT EXTRACTION
    # ========================================
    # Find model loaders (CheckpointLoader*, UNETLoader, etc.) and assign to A/B
    # using the same high/low chain logic as LoRAs.
    #
    # Strategy:
    # 1. Find model loaders that are at the root of LoRA chains (already traced)
    # 2. Find model loaders connected directly to KSamplers (no LoRAs in chain)
    # 3. Find standalone model loaders from API format
    # 4. Assign to A (high/first) or B (low/second) based on chain context

    models_a = []
    models_b = []
    model_names_seen = set()

    if workflow_data and node_map:
        # Approach: trace each KSampler/terminal MODEL input back to its model loader
        # and use the terminal's high/low context for assignment
        model_input_names = [
            'model', 'MODEL',
            'model_high_noise', 'model_low_noise',
            'base_model', 'refiner_model',
            'unet',
        ]

        for node in all_workflow_nodes:
            node_id = node.get('id')
            node_type = node.get('type', '')

            # Skip LoRA loaders and model loaders themselves
            if is_lora_node(node_type) or is_model_loader_node(node_type):
                continue

            inputs = node.get('inputs', [])
            for inp in inputs:
                inp_name = inp.get('name', '')
                inp_type = inp.get('type', '')
                is_model_input = (inp_type == 'MODEL' or inp_name in model_input_names)

                if is_model_input and inp.get('link'):
                    link_info = link_map.get(inp['link'])
                    if not link_info:
                        continue

                    # Trace back through the chain to find the model loader
                    loader_id = trace_to_model_loader(link_info['source_node'], node_map, link_map)
                    if not loader_id:
                        continue

                    loader_node = node_map.get(loader_id)
                    if not loader_node:
                        continue

                    loader_type = loader_node.get('type', '')
                    prompt_node = data.get(str(loader_id))

                    # Special handling for CyberdyneModelHub which outputs high/low from different slots
                    if loader_type == 'CyberdyneModelHub':
                        inputs_api = prompt_node.get('inputs', {}) if prompt_node else {}
                        high_name = inputs_api.get('model_high_name')
                        low_name = inputs_api.get('model_low_name')
                        if high_name and isinstance(high_name, str) and high_name not in model_names_seen:
                            model_names_seen.add(high_name)
                            print(f"[PromptExtractor] Model → A (high, CyberdyneModelHub): {high_name}")
                            models_a.append(high_name)
                        if low_name and isinstance(low_name, str) and low_name not in model_names_seen:
                            model_names_seen.add(low_name)
                            print(f"[PromptExtractor] Model → B (low, CyberdyneModelHub): {low_name}")
                            models_b.append(low_name)
                        continue

                    # Get model name - prefer API format (prompt_data) for active nodes
                    model_name = get_model_name_from_node(loader_node, prompt_node)
                    if not model_name or model_name in model_names_seen:
                        continue

                    model_names_seen.add(model_name)

                    # Determine high/low assignment from context
                    inp_label = inp.get('label', '').lower()
                    inp_name_lower = inp_name.lower()
                    terminal_title = node.get('title', '').lower()
                    loader_title = loader_node.get('title', '').lower()
                    model_name_lower = model_name.lower()

                    # Check all available context for high/low indicators
                    all_context = f"{inp_label} {inp_name_lower} {terminal_title} {loader_title} {model_name_lower}"
                    all_context_compact = all_context.replace('_', '').replace('-', '').replace(' ', '')

                    has_high = bool(
                        re.search(r'\bhigh\b', all_context) or
                        re.search(r'high(?:noise|_noise)', all_context_compact) or
                        re.search(r'i2v\s*high|t2v\s*high|_high|high_', all_context) or
                        re.search(r'high', model_name_lower)
                    )
                    has_low = bool(
                        re.search(r'\blow\b', all_context) or
                        re.search(r'low(?:noise|_noise)', all_context_compact) or
                        re.search(r'i2v\s*low|t2v\s*low|_low|low_', all_context) or
                        re.search(r'low', model_name_lower)
                    )

                    if has_low and not has_high:
                        print(f"[PromptExtractor] Model → B (low): {model_name}")
                        models_b.append(model_name)
                    else:
                        print(f"[PromptExtractor] Model → A (high/default): {model_name}")
                        models_a.append(model_name)

    # Fallback: extract from API format if workflow traversal found nothing
    if not models_a and not models_b and data:
        for node_id_str, node_data in data.items():
            if not isinstance(node_data, dict):
                continue
            class_type = node_data.get('class_type', '')
            if class_type not in MODEL_LOADER_TYPES:
                continue

            inputs = node_data.get('inputs', {})

            # Special handling for CyberdyneModelHub which has high/low in one node
            if class_type == 'CyberdyneModelHub':
                high_name = inputs.get('model_high_name')
                low_name = inputs.get('model_low_name')
                if high_name and isinstance(high_name, str) and high_name not in model_names_seen:
                    model_names_seen.add(high_name)
                    print(f"[PromptExtractor] Model → A (high, CyberdyneModelHub): {high_name}")
                    models_a.append(high_name)
                if low_name and isinstance(low_name, str) and low_name not in model_names_seen:
                    model_names_seen.add(low_name)
                    print(f"[PromptExtractor] Model → B (low, CyberdyneModelHub): {low_name}")
                    models_b.append(low_name)
                continue

            model_name = None
            for key in ['ckpt_name', 'unet_name', 'model_name', 'diffusion_model', 'model', 'model_path']:
                val = inputs.get(key)
                if val and isinstance(val, str):
                    model_name = val
                    break

            if not model_name or model_name in model_names_seen:
                continue

            model_names_seen.add(model_name)
            model_name_lower = model_name.lower()

            has_low = bool(
                'low_noise' in model_name_lower or '_low' in model_name_lower or
                re.search(r'i2v\s*low|t2v\s*low|low_', model_name_lower) or
                re.search(r'low', model_name_lower)
            )
            has_high = bool(
                'high_noise' in model_name_lower or '_high' in model_name_lower or
                re.search(r'i2v\s*high|t2v\s*high|high_', model_name_lower) or
                re.search(r'high', model_name_lower)
            )

            if has_low and not has_high:
                print(f"[PromptExtractor] Model → B (low, API fallback): {model_name}")
                models_b.append(model_name)
            else:
                print(f"[PromptExtractor] Model → A (high/default, API fallback): {model_name}")
                models_a.append(model_name)

    result['models_a'] = models_a
    result['models_b'] = models_b

    # Add models from PromptExtractor embedded data (if no models found from other sources)
    if _pe_extracted_models and not models_a and not models_b:
        for stack, model_path, source in _pe_extracted_models:
            if stack == 'a':
                print(f"[PromptExtractor] Model → A (from embedded PromptExtractor data): {model_path}")
                models_a.append(model_path)
            elif stack == 'b':
                print(f"[PromptExtractor] Model → B (from embedded PromptExtractor data): {model_path}")
                models_b.append(model_path)

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

        elif class_type in ['CheckpointLoaderSimple', 'CheckpointLoader', 'CheckpointLoaderKJ', 'CheckpointLoaderNF4']:
            if widgets_values:
                inputs['ckpt_name'] = widgets_values[0]

        elif class_type in ['UNETLoader', 'UnetLoaderGGUF', 'DiffusionModelLoader']:
            if widgets_values:
                inputs['unet_name'] = widgets_values[0]

        elif class_type == 'DiffusionModelLoaderKJ':
            if widgets_values:
                inputs['model_name'] = widgets_values[0]

        elif class_type == 'WanVideoModelLoader':
            if widgets_values:
                inputs['model'] = widgets_values[0]

        elif class_type == 'SeaArtUnetLoader':
            if widgets_values:
                inputs['unet_name'] = widgets_values[0]

        elif class_type == 'CyberdyneModelHub':
            if len(widgets_values) >= 1:
                inputs['model_high_name'] = widgets_values[0]
            if len(widgets_values) >= 3:
                inputs['model_low_name'] = widgets_values[2]

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
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        png_path = os.path.join(package_dir, 'js', 'placeholder.png')

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
                "source_folder": (["input", "output"], {
                    "default": "input",
                    "tooltip": "Browse files from the input or output folder"
                }),
                "image": (files, {}),
                "frame_position": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "use_lora_input_only": ("BOOLEAN", {
                    "default": False,
                    "label_on": "input only",
                    "label_off": "extract",
                    "tooltip": "When enabled, skip LoRA extraction and use only connected LoRA inputs"
                }),
            },
            "optional": {
                "lora_stack_a": ("LORA_STACK",),
                "lora_stack_b": ("LORA_STACK",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
            },
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Extract prompts, LoRA configurations, and model paths from images, videos, and workflow files."
    RETURN_TYPES = ("STRING", "STRING", "LORA_STACK", "LORA_STACK", "WORKFLOW_DATA", "IMAGE")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "lora_stack_a", "lora_stack_b", "workflow_data", "image")
    FUNCTION = "extract"
    OUTPUT_NODE = False

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def extract(self, image="", source_folder="input", frame_position=0.0, use_lora_input_only=False, lora_stack_a=None, lora_stack_b=None, unique_id=None, extra_pnginfo=None, prompt=None):
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
        workflow_data = ""
        model_a = ""
        model_b = ""

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

            # Handle relative paths (check selected source directory first, then temp as fallback)
            if not os.path.isabs(file_path):
                # Check selected source directory first
                if source_folder == "output":
                    base_dir = folder_paths.get_output_directory()
                else:
                    base_dir = folder_paths.get_input_directory()
                potential_path = os.path.join(base_dir, file_path)
                if os.path.exists(potential_path):
                    resolved_path = potential_path
                else:
                    # Check temp directory as fallback (for backwards compatibility)
                    temp_dir = folder_paths.get_temp_directory()
                    potential_path = os.path.join(temp_dir, file_path)
                    if os.path.exists(potential_path):
                        resolved_path = potential_path
                    else:
                        print(f"[PromptExtractor] File not found in {source_folder} or temp directories: {file_path}")
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
                # Get relative path from base directory to match JavaScript cache keys
                if source_folder == "output":
                    base_dir = folder_paths.get_output_directory()
                else:
                    base_dir = folder_paths.get_input_directory()
                if resolved_path.startswith(base_dir):
                    relative_path = os.path.relpath(resolved_path, base_dir)
                    relative_path = relative_path.replace('\\', '/')  # Normalize to forward slashes
                else:
                    relative_path = os.path.basename(resolved_path)

                # Prefer PyAV for frame extraction (accurate frame_position, handles H265/yuv444)
                image_tensor = extract_video_frame_av_to_tensor(resolved_path, frame_position)

                # Fall back to JS-cached frame if PyAV unavailable
                if image_tensor is None:
                    image_tensor = get_cached_video_frame(relative_path, frame_position)

                # Last resort: placeholder
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

                # Return model base names (without extension) — PromptModelLoader handles resolution
                raw_models_a = parsed.get('models_a', [])
                raw_models_b = parsed.get('models_b', [])
                if raw_models_a:
                    model_a = os.path.basename(raw_models_a[0].replace('\\', '/'))
                    print(f"[PromptExtractor] Model A: {model_a}")
                if raw_models_b:
                    model_b = os.path.basename(raw_models_b[0].replace('\\', '/'))
                    print(f"[PromptExtractor] Model B: {model_b}")

            # Build workflow_data in the same structured format as WorkflowBuilder
            # so both nodes output identical workflow_data that can feed PromptManagerAdvanced.
            if prompt_data or workflow_data:
                try:
                    # Late import to avoid circular dependency (py/ imports from nodes/)
                    from ..py.workflow_extraction_utils import (
                        extract_sampler_params,
                        extract_vae_info,
                        extract_clip_info,
                        extract_resolution,
                        build_simplified_workflow_data,
                        get_model_family,
                        get_family_label,
                    )
                    # Use raw workflow_data dict (before we overwrite the var below)
                    _raw_wf = workflow_data  # still a dict here
                    _is_a1111 = isinstance(prompt_data, dict) and 'prompt' in prompt_data and 'loras' in prompt_data

                    _sampler  = extract_sampler_params(prompt_data, _raw_wf)
                    _vae      = extract_vae_info(prompt_data, _raw_wf)
                    _clip     = extract_clip_info(prompt_data, _raw_wf)
                    _res      = extract_resolution(prompt_data, _raw_wf)

                    # If resolution came from node-refs (runtime values unknown at
                    # parse time), or is still the bare 512 default, use the actual
                    # image/video frame dimensions instead.
                    # image_tensor shape is [B, H, W, C].
                    if image_tensor is not None and hasattr(image_tensor, 'shape'):
                        src_h = int(image_tensor.shape[1])
                        src_w = int(image_tensor.shape[2])
                        if src_w > 0 and src_h > 0:
                            if _res.get('_width_from_node_ref') or _res['width'] == 512:
                                _res['width'] = src_w
                            if _res.get('_height_from_node_ref') or _res['height'] == 512:
                                _res['height'] = src_h

                    # A1111 images embed sampler/resolution in the parameters
                    # string — the ComfyUI extraction functions won't find them.
                    if _is_a1111:
                        if prompt_data.get('sampler_name'):
                            _sampler['sampler_name'] = prompt_data['sampler_name']
                        if prompt_data.get('scheduler'):
                            _sampler['scheduler'] = prompt_data['scheduler']
                        if prompt_data.get('steps'):
                            _sampler['steps_a'] = prompt_data['steps']
                        if prompt_data.get('cfg'):
                            _sampler['cfg'] = prompt_data['cfg']
                        if prompt_data.get('seed'):
                            _sampler['seed_a'] = prompt_data['seed']
                        if prompt_data.get('width') and prompt_data.get('height'):
                            _res['width'] = prompt_data['width']
                            _res['height'] = prompt_data['height']

                        # Forge embeds CLIP/VAE module names (e.g. "Module 2: clip_l, Module 3: t5xxl_fp16").
                        # Use these to infer clip_type when extract_clip_info found nothing.
                        _modules = prompt_data.get('modules', [])
                        if _modules and not _clip.get('type'):
                            _mod_str = ' '.join(_modules)
                            if 'qwen_3_8b' in _mod_str:
                                _clip['type'] = 'flux2'
                                _clip['source'] = 'a1111_modules'
                            elif 'qwen_3_4b' in _mod_str or 'qwen-4b' in _mod_str:
                                _clip['type'] = 'lumina2'
                                _clip['source'] = 'a1111_modules'
                            elif 't5xxl' in _mod_str:
                                _clip['type'] = 'flux'
                                _clip['source'] = 'a1111_modules'
                            elif 'umt5' in _mod_str:
                                _clip['type'] = 'wan'
                                _clip['source'] = 'a1111_modules'

                    # Resolve model family from model_a path
                    _raw_models_a = parsed.get('models_a', [])
                    _model_a_path = _raw_models_a[0] if _raw_models_a else model_a
                    _family = get_model_family(_model_a_path)

                    # Fallback: if model name doesn't match any family pattern,
                    # infer from clip source/type (checkpoint → sdxl, clip_type
                    # substring → specific family).
                    if not _family:
                        _clip_src  = _clip.get('source', '')
                        _clip_type = _clip.get('type', '').lower()
                        if _clip_src == 'checkpoint':
                            _family = 'sdxl'
                        elif 'flux2' in _clip_type:
                            _family = 'flux2'
                        elif 'flux' in _clip_type:
                            _family = 'flux1'
                        elif 'sd3' in _clip_type:
                            _family = 'sd3'
                        elif 'wan' in _clip_type:
                            _family = 'wan_video_t2v'
                        elif 'qwen_image' in _clip_type:
                            _family = 'qwen_image'
                        elif 'lumina2' in _clip_type:
                            _family = 'zimage'
                    if not _family:
                        _family = 'sdxl'

                    _extracted_for_wf = {
                        'positive_prompt': positive_prompt,
                        'negative_prompt': negative_prompt,
                        'loras_a':  loras_a,
                        'loras_b':  loras_b,
                        'model_a':  model_a,
                        'model_b':  model_b,
                        'vae':      _vae,
                        'clip':     _clip,
                        'sampler':  _sampler,
                        'resolution': _res,
                        'model_family':       _family,
                        'model_family_label': get_family_label(_family),
                    }
                    _simplified = build_simplified_workflow_data(
                        _extracted_for_wf,
                        overrides={'_source': 'PromptExtractor'},
                        sampler_params=_sampler,
                    )
                    # Add model / VAE availability so WorkflowBuilder can show red
                    from ..py.workflow_extraction_utils import resolve_model_name, resolve_vae_name
                    if model_a:
                        _res_a, _ = resolve_model_name(model_a)
                        _simplified['model_a_found'] = _res_a is not None
                    if model_b:
                        _res_b, _ = resolve_model_name(model_b)
                        _simplified['model_b_found'] = _res_b is not None
                    _vae_name = _vae.get('name', '') if isinstance(_vae, dict) else (_vae or '')
                    if _vae_name and not _vae_name.startswith('('):
                        _simplified['vae_found'] = resolve_vae_name(_vae_name) is not None
                    workflow_data = _simplified
                    print(f"[PromptExtractor] Output structured workflow_data (dict, {len(workflow_data)} keys)")

                    # Cache extracted info for WorkflowBuilder's "Update Workflow" button.
                    # Shape matches what WB's updateUI() expects (ui_info['extracted']).
                    if unique_id is not None:
                        from ..py.lora_utils import resolve_lora_path as _resolve_lora
                        _lora_avail = {}
                        for _l in loras_a + loras_b:
                            _ln = _l.get('name', '')
                            if _ln:
                                _, _found = _resolve_lora(_ln)
                                _lora_avail[_ln] = _found
                        _last_extracted_info[str(unique_id)] = {
                            '_source_file':       file_path,
                            '_source_folder':     source_folder,
                            'positive_prompt':    positive_prompt,
                            'negative_prompt':    negative_prompt,
                            'model_a':            model_a,
                            'model_b':            model_b,
                            'model_a_found':      _simplified.get('model_a_found', True),
                            'model_b_found':      _simplified.get('model_b_found', True),
                            'loras_a':            loras_a,
                            'loras_b':            loras_b,
                            'vae':                _vae,
                            'vae_found':          _simplified.get('vae_found', True),
                            'clip':               _clip,
                            'sampler':            _sampler,
                            'resolution':         _res,
                            'is_video':           ext in ['.mp4', '.webm', '.mov', '.avi'],
                            'model_family':       _family,
                            'model_family_label': get_family_label(_family),
                            'lora_availability':  _lora_avail,
                        }
                        print(f"[PromptExtractor] Cached extracted info for node {unique_id}")

                except Exception as e:
                    print(f"[PromptExtractor] Error building structured workflow_data: {e}")
                    import traceback
                    traceback.print_exc()
                    workflow_data = {}
            else:
                workflow_data = {}

            # Process loras only if use_lora_input_only is disabled (extract mode)
            if not use_lora_input_only:
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
        final_lora_stack_a = []
        final_lora_stack_b = []

        if use_lora_input_only:
            # Skip extraction, use only input loras
            if lora_stack_a is not None and isinstance(lora_stack_a, list):
                final_lora_stack_a = list(lora_stack_a)
            if lora_stack_b is not None and isinstance(lora_stack_b, list):
                final_lora_stack_b = list(lora_stack_b)
        else:
            # Extracted loras come first (to preserve workflow strengths), then input loras are appended (avoiding duplicates)
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

        # ── Merge final LoRA stacks back into workflow_data & cache ──────
        # Connected lora_stack inputs may have added LoRAs that weren't in
        # the extracted metadata.  Push the merged set into workflow_data
        # (the _simplified dict) and the _last_extracted_info cache so
        # WorkflowBuilder's "Update Workflow" button sees the full set.
        if isinstance(workflow_data, dict) and (final_lora_stack_a or final_lora_stack_b):
            def _tuples_to_lora_dicts(stack_tuples):
                """Convert (path, model_str, clip_str) tuples to LoRA dicts."""
                result = []
                for path, ms, cs in stack_tuples:
                    result.append({
                        'name': os.path.splitext(os.path.basename(path))[0],
                        'path': path,
                        'model_strength': ms,
                        'clip_strength': cs,
                        'active': True,
                    })
                return result

            merged_a = _tuples_to_lora_dicts(final_lora_stack_a)
            merged_b = _tuples_to_lora_dicts(final_lora_stack_b)
            workflow_data['loras_a'] = merged_a
            workflow_data['loras_b'] = merged_b

            # Update the _last_extracted_info cache too
            uid_str = str(unique_id) if unique_id is not None else None
            if uid_str and uid_str in _last_extracted_info:
                _last_extracted_info[uid_str]['loras_a'] = merged_a
                _last_extracted_info[uid_str]['loras_b'] = merged_b
                # Refresh availability for any newly-added LoRAs
                avail = _last_extracted_info[uid_str].get('lora_availability', {})
                for _l in merged_a + merged_b:
                    _ln = _l.get('name', '')
                    if _ln and _ln not in avail:
                        _, _found = resolve_lora_path(_ln)
                        avail[_ln] = _found
                _last_extracted_info[uid_str]['lora_availability'] = avail

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
        if not workflow_data:
            workflow_data = " "

        # Embed extracted_data into workflow so it's saved in the PNG metadata.
        # Uses the same schema as workflow_data (build_simplified_workflow_data)
        # with LoRA entries enriched with active/available state.
        if extra_pnginfo is not None and unique_id is not None:
            # Handle extra_pnginfo whether it's a dict or a list wrapper
            pnginfo = extra_pnginfo
            if isinstance(pnginfo, list) and len(pnginfo) > 0:
                pnginfo = pnginfo[0]
            if hasattr(pnginfo, 'get'):
                workflow = pnginfo.get('workflow', {})
            elif hasattr(pnginfo, 'workflow'):
                workflow = pnginfo.workflow
            else:
                workflow = {}

            # Build enriched LoRA lists with active + available state
            def _enrich_lora_stack(stack_tuples):
                enriched = []
                for lora_path, strength, clip_strength in stack_tuples:
                    lora_name = os.path.splitext(os.path.basename(lora_path))[0]
                    _, found = resolve_lora_path(lora_name)
                    enriched.append({
                        'name': lora_name,
                        'path': lora_path,
                        'strength': strength,
                        'clip_strength': clip_strength,
                        'active': True,
                        'available': found,
                    })
                return enriched

            loras_a_enriched = _enrich_lora_stack(final_lora_stack_a)
            loras_b_enriched = _enrich_lora_stack(final_lora_stack_b)

            # Reuse the already-built _simplified workflow_data dict if available,
            # otherwise build a minimal one.
            try:
                extracted_data = dict(workflow_data) if isinstance(workflow_data, dict) else {}
            except Exception:
                extracted_data = {}

            # Overwrite LoRA lists with enriched versions
            extracted_data['loras_a'] = loras_a_enriched
            extracted_data['loras_b'] = loras_b_enriched
            # Ensure core fields are present (even if workflow_data build failed)
            extracted_data.setdefault('positive_prompt', positive_prompt.strip())
            extracted_data.setdefault('negative_prompt', negative_prompt.strip())
            extracted_data.setdefault('model_a', model_a.strip())
            extracted_data.setdefault('model_b', model_b.strip())

            wf_nodes = workflow.get('nodes', []) if isinstance(workflow, dict) else []
            for wf_node in wf_nodes:
                if str(wf_node.get('id')) == str(unique_id):
                    wf_node['extracted_data'] = extracted_data
                    break

        return {
            "ui": {"images": preview_images},
            "result": (positive_prompt, negative_prompt, final_lora_stack_a, final_lora_stack_b, workflow_data, image_tensor)
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
    def IS_CHANGED(cls, image, source_folder="input", frame_position=0.0, use_lora_input_only=False, **kwargs):
        """
        Check if the file has changed or if frame position, source_folder, or use_lora_input_only changed.
        Returns a tuple to track changes.
        """
        mtime = "no_file"
        if image:
            file_path = image.strip()
            # Handle relative paths
            if not os.path.isabs(file_path):
                if source_folder == "output":
                    base_dir = folder_paths.get_output_directory()
                else:
                    base_dir = folder_paths.get_input_directory()
                potential_path = os.path.join(base_dir, file_path)
                if os.path.exists(potential_path):
                    mtime = os.path.getmtime(potential_path)
            elif os.path.exists(file_path):
                mtime = os.path.getmtime(file_path)

        return (mtime, source_folder, frame_position, use_lora_input_only)


class WorkflowExtractor:
    """
    Simplified extractor for WorkflowBuilder — outputs only workflow_data + image.
    No LoRA inputs/outputs, no prompt outputs. Same extraction logic as PromptExtractor.
    Uses composition (not inheritance) to avoid ComfyUI node-type confusion on reload.
    """

    def __init__(self):
        self._pe = PromptExtractor()

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = ["(none)"]
        supported_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.json', '.mp4', '.webm', '.mov', '.avi']

        if os.path.exists(input_dir):
            for root, dirs, filenames in os.walk(input_dir):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in supported_extensions:
                        full_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(full_path, input_dir).replace('\\', '/')
                        files.append(rel_path)

        files_to_sort = files[1:]
        files_to_sort.sort()
        files = ["(none)"] + files_to_sort

        return {
            "required": {
                "source_folder": (["input", "output"], {
                    "default": "input",
                    "tooltip": "Browse files from the input or output folder"
                }),
                "image": (files, {}),
                "frame_position": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
            },
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Extract workflow data from images, videos, and workflow files. Simplified version for WorkflowBuilder."
    RETURN_TYPES = ("WORKFLOW_DATA", "IMAGE")
    RETURN_NAMES = ("workflow_data", "image")
    FUNCTION = "extract_workflow"
    OUTPUT_NODE = False

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def extract_workflow(self, image="", source_folder="input", frame_position=0.0,
                         unique_id=None, extra_pnginfo=None, prompt=None):
        """Extract workflow_data and image only — delegates to PromptExtractor.extract()."""
        result = self._pe.extract(
            image=image, source_folder=source_folder, frame_position=frame_position,
            use_lora_input_only=True, lora_stack_a=None, lora_stack_b=None,
            unique_id=unique_id, extra_pnginfo=extra_pnginfo, prompt=prompt,
        )
        # Parent returns {"ui": ..., "result": (pos, neg, loras_a, loras_b, workflow_data, image)}
        full = result["result"]
        workflow_data = full[4]
        image_tensor = full[5]
        return {
            "ui": result["ui"],
            "result": (workflow_data, image_tensor),
        }

    @classmethod
    def IS_CHANGED(cls, image, source_folder="input", frame_position=0.0, **kwargs):
        mtime = "no_file"
        if image:
            file_path = image.strip()
            if not os.path.isabs(file_path):
                if source_folder == "output":
                    base_dir = folder_paths.get_output_directory()
                else:
                    base_dir = folder_paths.get_input_directory()
                potential_path = os.path.join(base_dir, file_path)
                if os.path.exists(potential_path):
                    mtime = os.path.getmtime(potential_path)
            elif os.path.exists(file_path):
                mtime = os.path.getmtime(file_path)
        return (mtime, source_folder, frame_position)
