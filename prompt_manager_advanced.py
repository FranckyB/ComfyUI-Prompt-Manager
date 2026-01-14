"""
ComfyUI Prompt Manager Advanced - Extended prompt management with LoRA stack support
Saves prompts along with associated LoRA configurations
"""
import os
import json
import base64
from io import BytesIO
import folder_paths
import server

# Import numpy and PIL for image processing (available in ComfyUI environment)
try:
    import numpy as np
    from PIL import Image
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False
    print("[PromptManagerAdvanced] Warning: PIL/numpy not available, thumbnail from image input disabled")


def image_to_base64_thumbnail(image_tensor, max_size=128):
    """
    Convert a ComfyUI image tensor to a base64 thumbnail string.

    Args:
        image_tensor: ComfyUI image tensor (B, H, W, C) in float32 0-1 range
        max_size: Maximum dimension for the thumbnail

    Returns:
        Base64 encoded JPEG string or None if conversion fails
    """
    if not IMAGE_SUPPORT or image_tensor is None:
        return None

    try:
        # Get first image from batch
        if len(image_tensor.shape) == 4:
            img_array = image_tensor[0]
        else:
            img_array = image_tensor

        # Convert to numpy and scale to 0-255
        if hasattr(img_array, 'cpu'):
            img_array = img_array.cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)

        # Create PIL Image
        img = Image.fromarray(img_array)

        # Resize maintaining aspect ratio
        width, height = img.size
        if width > height:
            if width > max_size:
                new_height = int((height * max_size) / width)
                new_width = max_size
                img = img.resize((new_width, new_height), Image.LANCZOS)
        else:
            if height > max_size:
                new_width = int((width * max_size) / height)
                new_height = max_size
                img = img.resize((new_width, new_height), Image.LANCZOS)

        # Convert to JPEG base64
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return f"data:image/jpeg;base64,{base64_str}"
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error converting image to thumbnail: {e}")
        return None


def get_available_loras():
    """Get all available LoRAs from ComfyUI's folder system"""
    lora_files = folder_paths.get_filename_list("loras")
    return lora_files


def normalize_path_separators(path):
    """Normalize path separators based on OS - for basename extraction only"""
    if os.name == 'nt':  # Windows
        return path.replace('/', '\\')
    else:  # Linux/Mac
        return path.replace('\\', '/')


def strip_lora_extension(name):
    """Remove only known LoRA file extensions, not arbitrary dots in names."""
    lora_extensions = ['.safetensors', '.ckpt', '.pt', '.bin', '.pth']
    name_lower = name.lower()
    for ext in lora_extensions:
        if name_lower.endswith(ext):
            return name[:-len(ext)]
    return name


def fuzzy_match_lora(lora_name, lora_files):
    """
    Attempt to find a matching LoRA using fuzzy matching.
    Handles renamed LoRAs by removing common WAN-related tokens.

    Example: "DR34LAY_HIGH_V2" can match "DR34LAY_I2V_14B_HIGH_V2"

    Returns (matched_file, True) if found, (None, False) otherwise.
    """
    import re

    # Tokens to remove for fuzzy matching (case-insensitive)
    # Order matters: remove longer tokens first to avoid partial matches
    wan_tokens = ['wan_2_2', 'wan22', 'wan2.2', '20epoc', 'a14b', '14b', 'i2v', 't2v']

    def normalize_name(name):
        """
        Remove WAN tokens from name, treating underscores and hyphens as separators.
        Also removes content in parentheses (e.g., "MyLora (1)" becomes "MyLora").
        Returns list of remaining non-empty parts in lowercase.
        """
        name_lower = name.lower()

        # Remove anything in parentheses (e.g., " (1)", "(copy)", etc.)
        name_lower = re.sub(r'\s*\([^)]*\)', '', name_lower)

        # Replace each token with a placeholder to preserve boundaries
        for token in wan_tokens:
            # Use word boundary-aware replacement with _ or - as delimiter
            # Replace [_-]token[_-], [_-]token, token[_-], or standalone token
            pattern = rf'(?:^|[_-]){re.escape(token)}(?:[_-]|$)'
            name_lower = re.sub(pattern, '_', name_lower)

        # Split by underscore or hyphen and filter out empty strings
        parts = [p for p in re.split(r'[_-]', name_lower) if p]
        return parts

    # Normalize the search name - use our extension stripper, not splitext
    search_parts = normalize_name(strip_lora_extension(lora_name))
    search_set = set(search_parts)

    # If search_set is empty after normalization, we can't do fuzzy matching
    if not search_set:
        return None, False

    candidates = []
    for lora_file in lora_files:
        file_name_no_ext = strip_lora_extension(os.path.basename(lora_file))
        file_parts = normalize_name(file_name_no_ext)
        file_set = set(file_parts)

        # Check if all search parts are present in the file parts
        if search_set.issubset(file_set):
            # Calculate how well this matches (prefer closer matches)
            extra_parts = len(file_set - search_set)
            candidates.append((lora_file, file_name_no_ext, extra_parts))

    if not candidates:
        return None, False

    # If only one match, return it
    if len(candidates) == 1:
        return candidates[0][0], True

    # Multiple matches - prefer ones that match i2v/t2v from original if present
    lora_name_lower = lora_name.lower()
    has_i2v = 'i2v' in lora_name_lower
    has_t2v = 't2v' in lora_name_lower

    if has_i2v:
        i2v_matches = [c for c in candidates if 'i2v' in c[1].lower()]
        if i2v_matches:
            candidates = i2v_matches
    elif has_t2v:
        t2v_matches = [c for c in candidates if 't2v' in c[1].lower()]
        if t2v_matches:
            candidates = t2v_matches

    # Return the match with fewest extra parts (closest match)
    candidates.sort(key=lambda x: x[2])
    return candidates[0][0], True


def get_lora_relative_path(lora_name):
    """
    Get the relative path for a LoRA that ComfyUI expects for loading.
    Returns (relative_path, found) tuple.
    Supports fuzzy matching for renamed LoRAs.
    """
    lora_files = get_available_loras()

    lora_name_lower = lora_name.lower()

    # Try exact match first
    for lora_file in lora_files:
        file_name_no_ext = strip_lora_extension(os.path.basename(lora_file))
        if file_name_no_ext.lower() == lora_name_lower:
            return lora_file, True

    # Try fuzzy match for renamed LoRAs
    fuzzy_match, found = fuzzy_match_lora(lora_name, lora_files)
    if found:
        return fuzzy_match, True

    # Not found
    return lora_name, False


class PromptManagerAdvanced:
    """
    Advanced Prompt Manager with LoRA stack integration.
    Supports saving/loading prompts with associated LoRA configurations.
    Features two LoRA stack inputs/outputs for dual LoRA workflows (e.g., Wan video).
    """

    @classmethod
    def INPUT_TYPES(s):
        prompts_data = PromptManagerAdvanced.load_prompts()
        categories = list(prompts_data.keys()) if prompts_data else ["Default"]

        all_prompts = set()
        for category_prompts in prompts_data.values():
            all_prompts.update(category_prompts.keys())

        all_prompts.add("")
        all_prompts_list = sorted(list(all_prompts))

        first_category = categories[0] if categories else "Default"

        # Get first prompt from first category
        first_prompt = ""
        first_prompt_text = ""
        if prompts_data and first_category in prompts_data and prompts_data[first_category]:
            first_category_prompts = list(prompts_data[first_category].keys())
            first_prompt = sorted(first_category_prompts, key=str.lower)[0] if first_category_prompts else ""
            if first_prompt:
                first_prompt_text = prompts_data[first_category][first_prompt].get("prompt", "")

        return {
            "required": {
                "category": (categories, {"default": first_category}),
                "name": (all_prompts_list, {"default": first_prompt}),
                "use_prompt_input": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off", "tooltip": "Toggle to use connected prompt input instead of internal text"}),
                "use_lora_input": ("BOOLEAN", {
                    "default": False,
                    "label_on": "on",
                    "label_off": "off",
                    "tooltip": "When enabled, use LoRAs from connected inputs. When off, use only prompt LoRAs."
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": first_prompt_text,
                    "placeholder": "Enter prompt text",
                    "dynamicPrompts": False,
                    "tooltip": "Enter prompt text directly"
                }),
            },
            "optional": {
                "prompt_input": ("STRING", {"multiline": True, "forceInput": True, "lazy": True, "tooltip": "Connect prompt text input here"}),
                "lora_stack_a": ("LORA_STACK", {"tooltip": "First LoRA stack input (e.g., for base model)"}),
                "lora_stack_b": ("LORA_STACK", {"tooltip": "Second LoRA stack input (e.g., for video model)"}),
                "trigger_words": ("STRING", {"forceInput": True, "tooltip": "Comma-separated trigger words to append to prompt"}),
                "thumbnail_image": ("IMAGE", {"tooltip": "Connect an image to use as thumbnail when saving the prompt"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "loras_a_toggle": "STRING",
                "loras_b_toggle": "STRING",
                "trigger_words_toggle": "STRING",
            }
        }

    CATEGORY = "Prompt Manager"
    RETURN_TYPES = ("STRING", "LORA_STACK", "LORA_STACK")
    RETURN_NAMES = ("prompt", "lora_stack_a", "lora_stack_b")
    FUNCTION = "get_prompt"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, category, name, text, use_prompt_input, use_lora_input, **kwargs):
        """
        Track changes to the node's inputs to determine if re-execution is needed.
        Returns a tuple of relevant values that should trigger re-execution when changed.
        """
        # Get optional inputs
        prompt_input = kwargs.get('prompt_input', None)
        lora_stack_a = kwargs.get('lora_stack_a', None)
        lora_stack_b = kwargs.get('lora_stack_b', None)
        trigger_words = kwargs.get('trigger_words', None)

        # Return tuple of all values that should trigger re-execution when changed
        # Convert lists/objects to strings for hashable comparison
        return (
            category,
            name,
            text,
            use_prompt_input,
            use_lora_input,
            str(prompt_input) if prompt_input else None,
            str(lora_stack_a) if lora_stack_a else None,
            str(lora_stack_b) if lora_stack_b else None,
            trigger_words
        )

    @classmethod
    def VALIDATE_INPUTS(cls, name, **kwargs):
        """Allow any name value, including temporary unsaved prompts"""
        # Return True to accept any value - this allows new prompts to be tested before saving
        return True

    @staticmethod
    def get_prompts_path():
        """Get the path to the prompts JSON file in user/default folder (shared with standard PromptManager)"""
        return os.path.join(folder_paths.get_user_directory(), "default", "prompt_manager_data.json")

    @staticmethod
    def get_default_prompts_path():
        """Get the path to the default prompts JSON file"""
        return os.path.join(os.path.dirname(__file__), "default_prompts.json")

    @classmethod
    def load_prompts(cls):
        """Load prompts from user folder or default (shared with standard PromptManager)"""
        user_path = cls.get_prompts_path()
        default_path = cls.get_default_prompts_path()

        if os.path.exists(user_path):
            try:
                with open(user_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[PromptManagerAdvanced] Error loading user prompts: {e}")

        if os.path.exists(default_path):
            try:
                with open(default_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cls.save_prompts(data)
                    return data
            except Exception as e:
                print(f"[PromptManagerAdvanced] Error loading default prompts: {e}")

        # Return empty structure if all else fails
        print("[PromptManagerAdvanced] Warning: No prompt files found, starting with empty data")
        return {}

    @staticmethod
    def sort_prompts_data(data):
        """Sort categories and prompts alphabetically (case-insensitive)"""
        sorted_data = {}
        for category in sorted(data.keys(), key=str.lower):
            sorted_data[category] = dict(sorted(data[category].items(), key=lambda item: item[0].lower()))
        return sorted_data

    @classmethod
    def save_prompts(cls, data):
        """Save prompts to user folder"""
        user_path = cls.get_prompts_path()
        sorted_data = cls.sort_prompts_data(data)

        try:
            os.makedirs(os.path.dirname(user_path), exist_ok=True)
            with open(user_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[PromptManagerAdvanced] Error saving prompts: {e}")

    def _merge_lora_stacks(self, preset_stack, connected_stack):
        """
        Merge preset loras with connected loras, avoiding duplicates.
        Preset loras come first, then connected loras that aren't already in preset.

        Args:
            preset_stack: List of tuples from saved preset
            connected_stack: List of tuples from connected input

        Returns:
            Combined lora stack with no duplicates (by lora name, case-insensitive)
        """
        if not connected_stack:
            return preset_stack if preset_stack else []
        if not preset_stack:
            return list(connected_stack)

        # Build set of lora names from preset (case-insensitive)
        preset_names = set()
        for lora_path, _, _ in preset_stack:
            # Normalize path separators for OS-aware basename extraction
            normalized_path = normalize_path_separators(lora_path)
            lora_name = strip_lora_extension(os.path.basename(normalized_path)).lower()
            preset_names.add(lora_name)

        # Start with preset, add non-duplicate connected loras
        merged = list(preset_stack)
        for lora_path, model_strength, clip_strength in connected_stack:
            # Normalize path separators for OS-aware basename extraction
            normalized_path = normalize_path_separators(lora_path)
            lora_name = strip_lora_extension(os.path.basename(normalized_path)).lower()
            if lora_name not in preset_names:
                merged.append((lora_path, model_strength, clip_strength))
                preset_names.add(lora_name)  # Prevent duplicates within connected too

        return merged

    def _process_lora_toggle(self, lora_stack, toggle_data):
        """
        Process lora stack with toggle data to filter active loras and apply strength adjustments.
        Also attempts to resolve LoRA paths that may not exist locally by searching by name.

        Args:
            lora_stack: List of tuples (lora_path, model_strength, clip_strength)
            toggle_data: JSON string or list of toggle states [{name, active, strength}, ...]

        Returns:
            Filtered lora stack with adjusted strengths and resolved paths
        """
        if not lora_stack:
            return []

        if not toggle_data:
            # Even without toggle data, still try to resolve paths (skip unfound)
            resolved_stack = []
            lora_files = get_available_loras()
            for lora_path, model_strength, clip_strength in lora_stack:
                resolved_path, found = self._resolve_lora_path_with_status(lora_path, lora_files)
                if found:
                    resolved_stack.append((resolved_path, model_strength, clip_strength))
                else:
                    print(f"[PromptManagerAdvanced] Skipping unfound LoRA: {lora_path}")
            return resolved_stack

        # Parse toggle data if it's a string
        try:
            if isinstance(toggle_data, str):
                toggle_data = json.loads(toggle_data)
        except (json.JSONDecodeError, TypeError):
            return list(lora_stack)

        if not isinstance(toggle_data, list):
            return list(lora_stack)

        # Create a map of toggle states by lora name (case-insensitive)
        toggle_map = {}
        for item in toggle_data:
            if isinstance(item, dict):
                name = item.get('name', '')
                toggle_map[name.lower()] = {
                    'active': item.get('active', True),
                    'strength': item.get('strength'),
                    'original_name': name  # Keep original for debugging
                }

        # Filter and adjust lora stack
        filtered_stack = []
        lora_files = get_available_loras()

        for lora_path, model_strength, clip_strength in lora_stack:
            # Extract lora name from path - normalize separators for OS-aware basename extraction
            normalized_path = normalize_path_separators(lora_path)
            lora_name = strip_lora_extension(os.path.basename(normalized_path))

            # Check toggle state (case-insensitive lookup)
            toggle_state = toggle_map.get(lora_name.lower(), {'active': True, 'strength': None})

            # Debug logging
            print(f"[PromptManagerAdvanced] Processing LoRA: {lora_name}, original_strength={model_strength}, toggle_strength={toggle_state.get('strength')}, active={toggle_state.get('active')}")

            if not toggle_state['active']:
                continue  # Skip inactive loras

            # Apply strength adjustment if provided
            if toggle_state['strength'] is not None:
                adjusted_strength = float(toggle_state['strength'])
                # Scale both model and clip strength proportionally
                ratio = adjusted_strength / model_strength if model_strength != 0 else 1.0
                model_strength = adjusted_strength
                clip_strength = clip_strength * ratio

            # Try to resolve the path to a local LoRA - skip if not found
            resolved_path, found = self._resolve_lora_path_with_status(lora_path, lora_files)
            if found:
                filtered_stack.append((resolved_path, model_strength, clip_strength))
            else:
                print(f"[PromptManagerAdvanced] Skipping unfound LoRA: {lora_name}")

        return filtered_stack

    def _resolve_lora_path_with_status(self, lora_path, lora_files):
        """
        Try to resolve a LoRA path to an available local LoRA.
        First checks if the exact path exists, then searches by name.

        Args:
            lora_path: The original path from the stack
            lora_files: List of available LoRA files

        Returns:
            Tuple of (resolved_path, found) - path and whether it was found
        """
        # Check if exact path exists
        if lora_path in lora_files:
            return lora_path, True

        # Try to find by name
        lora_name = strip_lora_extension(os.path.basename(lora_path))
        found_path, found = get_lora_relative_path(lora_name)
        if found:
            return found_path, True

        # Not found
        return lora_path, False

    def get_prompt(self, category, name, use_prompt_input, text="", use_lora_input=True, prompt_input="",
                   lora_stack_a=None, lora_stack_b=None, trigger_words=None, thumbnail_image=None,
                   unique_id=None, loras_a_toggle=None, loras_b_toggle=None, trigger_words_toggle=None):
        """Return the prompt text and filtered lora stacks based on toggle states"""

        # Choose which text to use based on the toggle
        if use_prompt_input and prompt_input:
            output_text = prompt_input
        else:
            output_text = text

        # Capture the original input loras BEFORE any processing
        # These are used by the frontend to detect when inputs change
        # and clear toggle states accordingly
        input_loras_a = self._format_loras_for_display(lora_stack_a) if lora_stack_a else []
        input_loras_b = self._format_loras_for_display(lora_stack_b) if lora_stack_b else []

        # Process trigger words
        trigger_words_display = self._process_trigger_words(
            trigger_words, trigger_words_toggle
        )
        active_trigger_words = self._get_active_trigger_words(trigger_words_toggle, trigger_words)

        # Build preset stacks from saved toggle data (these are what we display/manage)
        preset_stack_a = self._build_stack_from_toggle(loras_a_toggle) if loras_a_toggle else []
        preset_stack_b = self._build_stack_from_toggle(loras_b_toggle) if loras_b_toggle else []

        # Get ALL preset loras including unavailable ones (for display purposes)
        all_preset_loras_a = self._get_all_loras_from_toggle(loras_a_toggle) if loras_a_toggle else []
        all_preset_loras_b = self._get_all_loras_from_toggle(loras_b_toggle) if loras_b_toggle else []

        # When use_lora_input is disabled, ignore connected stacks and use only saved loras
        if not use_lora_input:
            lora_stack_a = preset_stack_a
            lora_stack_b = preset_stack_b
        else:
            # Merge: preset loras + connected loras (avoiding duplicates by lora name)
            # Handle case where connected stack might be None or empty
            if lora_stack_a:
                lora_stack_a = self._merge_lora_stacks(preset_stack_a, lora_stack_a)
            else:
                lora_stack_a = preset_stack_a if preset_stack_a else []

            if lora_stack_b:
                lora_stack_b = self._merge_lora_stacks(preset_stack_b, lora_stack_b)
            else:
                lora_stack_b = preset_stack_b if preset_stack_b else []

        # Process lora stacks with toggle data (filter inactive, apply strength)
        processed_stack_a = self._process_lora_toggle(lora_stack_a, loras_a_toggle)
        processed_stack_b = self._process_lora_toggle(lora_stack_b, loras_b_toggle)

        # Display logic: Include ALL loras (available + unavailable) so UI shows complete picture
        # Build display from available loras first, then add unavailable ones
        if not use_lora_input:
            loras_a_display = self._format_loras_for_display_with_unavailable(preset_stack_a, all_preset_loras_a)
            loras_b_display = self._format_loras_for_display_with_unavailable(preset_stack_b, all_preset_loras_b)
        else:
            loras_a_display = self._format_loras_for_display_with_unavailable(lora_stack_a, all_preset_loras_a)
            loras_b_display = self._format_loras_for_display_with_unavailable(lora_stack_b, all_preset_loras_b)

        # Convert thumbnail image to base64 if provided
        thumbnail_base64 = None
        if thumbnail_image is not None and IMAGE_SUPPORT:
            try:
                thumbnail_base64 = image_to_base64_thumbnail(thumbnail_image)
            except Exception as e:
                print(f"[PromptManagerAdvanced] Failed to convert thumbnail image: {e}")

        # Build explicit list of unavailable lora names for frontend
        unavailable_loras_a = [lora['name'] for lora in all_preset_loras_a if not lora.get('available', False)]
        unavailable_loras_b = [lora['name'] for lora in all_preset_loras_b if not lora.get('available', False)]

        # Broadcast update to frontend
        if unique_id is not None:
            server.PromptServer.instance.send_sync("prompt-manager-advanced-update", {
                "node_id": unique_id,
                "prompt": output_text,
                "use_prompt_input": use_prompt_input,
                "prompt_input": prompt_input,
                "loras_a": loras_a_display,
                "loras_b": loras_b_display,
                "input_loras_a": input_loras_a,  # Original input loras for change detection
                "input_loras_b": input_loras_b,  # Original input loras for change detection
                "unavailable_loras_a": unavailable_loras_a,  # Explicit list of unavailable lora names
                "unavailable_loras_b": unavailable_loras_b,  # Explicit list of unavailable lora names
                "trigger_words": trigger_words_display,
                "connected_thumbnail": thumbnail_base64
            })

        # Append active trigger words to output
        final_output = self._append_trigger_words_to_prompt(output_text, active_trigger_words)

        return (final_output, processed_stack_a if processed_stack_a else [], processed_stack_b if processed_stack_b else [])

    def check_lazy_status(self, category, name, use_prompt_input, text, use_lora_input=True, prompt_input=None,
                          lora_stack_a=None, lora_stack_b=None, trigger_words=None, thumbnail_image=None,
                          unique_id=None, loras_a_toggle=None, loras_b_toggle=None, trigger_words_toggle=None):
        """Tell ComfyUI which lazy inputs are needed based on current settings.

        Note: We don't mark prompt_input as required even when use_prompt_input is on.
        This allows the node to gracefully fall back to internal text if nothing is connected,
        rather than throwing a ComfyUI error about missing input.
        """
        # Return empty list - all optional inputs are truly optional
        # The get_prompt method handles fallback logic gracefully
        return []

    def _build_stack_from_toggle(self, toggle_data):
        """
        Build a lora stack from toggle data by resolving paths at runtime.
        This is used when no input stack is connected but we have saved lora data.
        """
        if not toggle_data:
            return []

        # Parse toggle data if it's a string
        try:
            if isinstance(toggle_data, str):
                toggle_data = json.loads(toggle_data)
        except (json.JSONDecodeError, TypeError):
            return []

        if not isinstance(toggle_data, list):
            return []

        stack = []
        missing_loras = []

        for item in toggle_data:
            if not isinstance(item, dict):
                continue

            lora_name = item.get('name', '')
            if not lora_name:
                continue

            # Skip inactive loras
            if item.get('active', True) is False:
                continue

            # Resolve the path at runtime
            relative_path, found = get_lora_relative_path(lora_name)

            if not found:
                missing_loras.append(lora_name)
                continue  # Skip missing loras

            strength = float(item.get('strength', 1.0))
            clip_strength = float(item.get('clip_strength', strength))

            stack.append((relative_path, strength, clip_strength))

        # Log warning for missing loras
        if missing_loras:
            print(f"[PromptManagerAdvanced] Warning: Could not find LoRAs: {', '.join(missing_loras)}")

        return stack

    def _format_loras_for_display(self, lora_stack):
        """Format lora stack for frontend display, checking availability and deduplicating by name"""
        if not lora_stack:
            return []

        display_list = []
        seen_names = set()  # Track seen LoRA names (case-insensitive) to avoid duplicates
        lora_files = get_available_loras()

        for lora_path, model_strength, clip_strength in lora_stack:
            # Normalize path separators for OS-aware basename extraction
            normalized_path = normalize_path_separators(lora_path)
            lora_name = strip_lora_extension(os.path.basename(normalized_path))
            lora_name_lower = lora_name.lower()

            # Skip if we've already added this LoRA (case-insensitive)
            if lora_name_lower in seen_names:
                continue

            # Check if LoRA is available locally
            # First try exact path match, then try name-based search
            available = False
            resolved_path = lora_path

            # Check if exact path exists in available loras
            if lora_path in lora_files:
                available = True
                resolved_path = lora_path
            else:
                # Try to find by name
                found_path, found = get_lora_relative_path(lora_name)
                if found:
                    available = True
                    resolved_path = found_path

            display_list.append({
                "name": lora_name,
                "path": resolved_path,
                "model_strength": model_strength,
                "clip_strength": clip_strength,
                "active": True,
                "strength": model_strength,
                "available": available
            })
            seen_names.add(lora_name_lower)

        return display_list

    def _get_all_loras_from_toggle(self, toggle_data):
        """
        Get ALL loras from toggle data including unavailable ones.
        Returns list of dicts with name, strength, active, and available status.
        Used for building complete display list.
        """
        if not toggle_data:
            return []

        try:
            if isinstance(toggle_data, str):
                toggle_data = json.loads(toggle_data)
        except (json.JSONDecodeError, TypeError):
            return []

        if not isinstance(toggle_data, list):
            return []

        all_loras = []
        for item in toggle_data:
            if not isinstance(item, dict):
                continue

            lora_name = item.get('name', '')
            if not lora_name:
                continue

            # Check availability using same fuzzy matching as actual loading
            # UI should match what Python will actually do
            _, found = get_lora_relative_path(lora_name)

            all_loras.append({
                "name": lora_name,
                "strength": float(item.get('strength', 1.0)),
                "clip_strength": float(item.get('clip_strength', item.get('strength', 1.0))),
                "active": item.get('active', True),
                "available": found
            })

        return all_loras

    def _format_loras_for_display_with_unavailable(self, lora_stack, all_preset_loras):
        """
        Format lora stack for frontend display, including unavailable preset loras.
        This ensures the UI shows all saved loras even if they're not found locally.
        """
        # Start with the available loras from the stack
        display_list = self._format_loras_for_display(lora_stack) if lora_stack else []
        seen_names = set(item['name'].lower() for item in display_list)

        # Add any unavailable preset loras that aren't already in the list
        for lora in all_preset_loras:
            lora_name_lower = lora['name'].lower()
            if lora_name_lower not in seen_names:
                display_list.append({
                    "name": lora['name'],
                    "path": lora['name'],  # Just use name as path for unavailable
                    "model_strength": lora['strength'],
                    "clip_strength": lora['clip_strength'],
                    "active": lora['active'],
                    "strength": lora['strength'],
                    "available": lora['available']
                })
                seen_names.add(lora_name_lower)

        return display_list

    def _process_trigger_words(self, connected_trigger_words, toggle_data):
        """
        Process trigger words from connected input and saved toggle states.
        Returns list of trigger word objects for display.
        """
        # Parse saved toggle data
        saved_words = {}
        if toggle_data:
            try:
                if isinstance(toggle_data, str):
                    toggle_data = json.loads(toggle_data)
                if isinstance(toggle_data, list):
                    for item in toggle_data:
                        if isinstance(item, dict) and item.get('text'):
                            word = item['text'].strip()
                            saved_words[word.lower()] = {
                                'text': word,
                                'active': item.get('active', True)
                            }
            except (json.JSONDecodeError, TypeError):
                pass

        # Parse connected trigger words (comma-separated)
        connected_words = []
        if connected_trigger_words and isinstance(connected_trigger_words, str):
            connected_words = [w.strip() for w in connected_trigger_words.split(',') if w.strip()]

        # Merge: saved words + new connected words (no duplicates)
        result = []
        seen = set()

        # First, add all saved words (preserving their state)
        for word_lower, data in saved_words.items():
            if word_lower not in seen:
                result.append({
                    'text': data['text'],
                    'active': data['active'],
                    'source': 'saved'
                })
                seen.add(word_lower)

        # Then add connected words that aren't already saved
        for word in connected_words:
            word_lower = word.lower()
            if word_lower not in seen:
                result.append({
                    'text': word,
                    'active': True,
                    'source': 'connected'
                })
                seen.add(word_lower)

        return result

    def _get_active_trigger_words(self, toggle_data, connected_trigger_words=None):
        """
        Get list of active trigger word strings from toggle data and connected input.
        On first execution, toggle_data may be empty, so we also include connected words.
        """
        active_words = []
        seen = set()

        # First, get active words from saved toggle data
        if toggle_data:
            try:
                if isinstance(toggle_data, str):
                    toggle_data = json.loads(toggle_data)
                if isinstance(toggle_data, list):
                    for item in toggle_data:
                        if isinstance(item, dict) and item.get('active', True):
                            word = item.get('text', '').strip()
                            if word:
                                active_words.append(word)
                                seen.add(word.lower())
            except (json.JSONDecodeError, TypeError):
                pass

        # Also include connected trigger words that aren't already in the saved data
        # This ensures they work on first execution before the UI syncs
        if connected_trigger_words and isinstance(connected_trigger_words, str):
            for word in connected_trigger_words.split(','):
                word = word.strip()
                if word and word.lower() not in seen:
                    active_words.append(word)
                    seen.add(word.lower())

        return active_words

    def _append_trigger_words_to_prompt(self, prompt, trigger_words):
        """
        Append trigger words to the end of the prompt.
        Adds a period before trigger words if prompt doesn't end with comma or period.
        """
        if not trigger_words:
            return prompt

        prompt = prompt.rstrip()
        if not prompt:
            return ', '.join(trigger_words)

        # Check if prompt ends with punctuation
        if not prompt.endswith((',', '.')):
            prompt += '.'

        # Add space if needed
        if not prompt.endswith(' '):
            prompt += ' '

        return prompt + ', '.join(trigger_words)


# API Routes for Advanced Prompt Manager

@server.PromptServer.instance.routes.get("/prompt-manager-advanced/get-prompts")
async def get_prompts_advanced(request):
    """API endpoint to get all advanced prompts"""
    try:
        prompts = PromptManagerAdvanced.load_prompts()
        return server.web.json_response(prompts)
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in get_prompts API: {e}")
        return server.web.json_response({"error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager-advanced/save-category")
async def save_category_advanced(request):
    """API endpoint to create a new category"""
    try:
        data = await request.json()
        category_name = data.get("category_name", "").strip()

        if not category_name:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        prompts = PromptManagerAdvanced.load_prompts()

        # Case-insensitive check for existing category
        existing_categories_lower = {k.lower(): k for k in prompts.keys()}
        if category_name.lower() in existing_categories_lower:
            return server.web.json_response({
                "success": False,
                "error": f"Category already exists as '{existing_categories_lower[category_name.lower()]}'"
            })

        prompts[category_name] = {}
        PromptManagerAdvanced.save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in save_category API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager-advanced/save-prompt")
async def save_prompt_advanced(request):
    """API endpoint to save a prompt with lora configurations"""
    try:
        data = await request.json()
        category = data.get("category", "").strip()
        name = data.get("name", "").strip()
        text = data.get("text", "").strip()
        loras_a = data.get("loras_a", [])
        loras_b = data.get("loras_b", [])
        trigger_words = data.get("trigger_words", [])

        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name are required"})

        prompts = PromptManagerAdvanced.load_prompts()

        if category not in prompts:
            prompts[category] = {}

        # Case-insensitive check for existing prompt
        existing_prompts_lower = {k.lower(): k for k in prompts[category].keys()}
        if name.lower() in existing_prompts_lower:
            old_name = existing_prompts_lower[name.lower()]
            if old_name != name:
                # Delete the old casing version
                print(f"[PromptManagerAdvanced] Removing old casing '{old_name}' before saving as '{name}'")
                del prompts[category][old_name]

        # Normalize lora data - save name, strengths, and active state
        # Paths are resolved at runtime for portability
        def normalize_lora_data(loras):
            normalized = []
            for lora in loras:
                if isinstance(lora, dict) and lora.get('name'):
                    normalized.append({
                        "name": lora.get('name'),
                        "strength": lora.get('strength', lora.get('model_strength', 1.0)),
                        "clip_strength": lora.get('clip_strength', lora.get('strength', 1.0)),
                        "active": lora.get('active', True)
                    })
            return normalized

        # Normalize trigger words data
        def normalize_trigger_words(words):
            normalized = []
            seen = set()
            for word in words:
                if isinstance(word, dict) and word.get('text'):
                    text = word['text'].strip()
                    text_lower = text.lower()
                    if text and text_lower not in seen:
                        normalized.append({
                            "text": text,
                            "active": word.get('active', True)
                        })
                        seen.add(text_lower)
                elif isinstance(word, str) and word.strip():
                    text = word.strip()
                    text_lower = text.lower()
                    if text_lower not in seen:
                        normalized.append({"text": text, "active": True})
                        seen.add(text_lower)
            return normalized

        # Preserve existing thumbnail if not provided, or use new thumbnail
        thumbnail = data.get("thumbnail")
        existing_prompt = prompts.get(category, {}).get(name, {})
        if thumbnail is None:
            thumbnail = existing_prompt.get("thumbnail")

        # Save prompt with normalized lora data (no paths - they're resolved at runtime)
        prompt_data = {
            "prompt": text,
            "loras_a": normalize_lora_data(loras_a),
            "loras_b": normalize_lora_data(loras_b),
            "trigger_words": normalize_trigger_words(trigger_words)
        }
        if thumbnail:
            prompt_data["thumbnail"] = thumbnail

        prompts[category][name] = prompt_data
        PromptManagerAdvanced.save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in save_prompt API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager-advanced/delete-category")
async def delete_category_advanced(request):
    """API endpoint to delete a category"""
    try:
        data = await request.json()
        category = data.get("category", "").strip()

        if not category:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        prompts = PromptManagerAdvanced.load_prompts()

        if category not in prompts:
            return server.web.json_response({"success": False, "error": "Category not found"})

        del prompts[category]
        PromptManagerAdvanced.save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in delete_category API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager-advanced/delete-prompt")
async def delete_prompt_advanced(request):
    """API endpoint to delete a prompt"""
    try:
        data = await request.json()
        category = data.get("category", "").strip()
        name = data.get("name", "").strip()

        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and prompt name are required"})

        prompts = PromptManagerAdvanced.load_prompts()

        if category not in prompts or name not in prompts[category]:
            return server.web.json_response({"success": False, "error": "Prompt not found"})

        del prompts[category][name]
        PromptManagerAdvanced.save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in delete_prompt API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager-advanced/check-loras")
async def check_loras_advanced(request):
    """API endpoint to check which LoRAs are available in the system"""
    try:
        data = await request.json()
        lora_names = data.get("lora_names", [])

        results = {}
        for lora_name in lora_names:
            _, found = get_lora_relative_path(lora_name)
            results[lora_name] = found

        return server.web.json_response({"success": True, "results": results})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in check_loras API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.get("/prompt-manager-advanced/available-loras")
async def get_available_loras_api(request):
    """API endpoint to get list of all available LoRA names"""
    try:
        lora_files = get_available_loras()
        # Return just the names without extensions for easier matching
        lora_names = [strip_lora_extension(os.path.basename(f)) for f in lora_files]
        return server.web.json_response({"success": True, "loras": sorted(set(lora_names))})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in available_loras API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager-advanced/get-prompt-data")
async def get_prompt_data_advanced(request):
    """API endpoint to get specific prompt data including loras with availability check"""
    try:
        data = await request.json()
        category = data.get("category", "").strip()
        name = data.get("name", "").strip()

        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name are required"})

        prompts = PromptManagerAdvanced.load_prompts()

        if category not in prompts or name not in prompts[category]:
            return server.web.json_response({"success": False, "error": "Prompt not found"})

        prompt_data = prompts[category][name]

        # Add availability info to loras
        def add_availability(loras):
            result = []
            for lora in loras:
                lora_copy = dict(lora)
                _, found = get_lora_relative_path(lora.get('name', ''))
                lora_copy['available'] = found
                result.append(lora_copy)
            return result

        return server.web.json_response({
            "success": True,
            "data": {
                "prompt": prompt_data.get("prompt", ""),
                "loras_a": add_availability(prompt_data.get("loras_a", [])),
                "loras_b": add_availability(prompt_data.get("loras_b", [])),
                "trigger_words": prompt_data.get("trigger_words", [])
            }
        })
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in get_prompt_data API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager-advanced/import-prompts")
async def import_prompts_advanced(request):
    """API endpoint to import prompts from JSON"""
    try:
        data = await request.json()
        imported_data = data.get("data", {})
        mode = data.get("mode", "merge")  # "merge" or "replace"

        if not isinstance(imported_data, dict):
            return server.web.json_response({"success": False, "error": "Invalid data format"})

        if mode == "replace":
            # Replace all existing prompts
            prompts = {}
        else:
            # Merge with existing prompts
            prompts = PromptManagerAdvanced.load_prompts()

        # Process imported data
        for category, category_prompts in imported_data.items():
            if not isinstance(category_prompts, dict):
                continue

            if category not in prompts:
                prompts[category] = {}

            for prompt_name, prompt_data in category_prompts.items():
                if not isinstance(prompt_data, dict):
                    continue

                # Normalize the prompt data structure (include thumbnail if present)
                normalized = {
                    "prompt": prompt_data.get("prompt", ""),
                    "loras_a": prompt_data.get("loras_a", []),
                    "loras_b": prompt_data.get("loras_b", []),
                    "trigger_words": prompt_data.get("trigger_words", [])
                }

                # Include thumbnail if present in imported data
                if "thumbnail" in prompt_data and prompt_data["thumbnail"]:
                    normalized["thumbnail"] = prompt_data["thumbnail"]

                prompts[category][prompt_name] = normalized

        PromptManagerAdvanced.save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in import_prompts API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager-advanced/save-thumbnail")
async def save_thumbnail(request):
    """API endpoint to save or remove a thumbnail for a prompt"""
    try:
        data = await request.json()
        category = data.get("category", "").strip()
        name = data.get("name", "").strip()
        thumbnail = data.get("thumbnail")  # Can be None to remove thumbnail

        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name required"})

        prompts = PromptManagerAdvanced.load_prompts()

        if category not in prompts or name not in prompts[category]:
            return server.web.json_response({"success": False, "error": "Prompt not found"})

        # Update or remove thumbnail
        if thumbnail:
            prompts[category][name]["thumbnail"] = thumbnail
        elif "thumbnail" in prompts[category][name]:
            del prompts[category][name]["thumbnail"]

        PromptManagerAdvanced.save_prompts(prompts)

        return server.web.json_response({"success": True})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in save_thumbnail API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)
