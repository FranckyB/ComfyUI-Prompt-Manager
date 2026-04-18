"""
ComfyUI Prompt Manager Advanced - Extended prompt management with LoRA stack support
Saves prompts along with associated LoRA configurations
"""
import os
import json
import shutil
import base64
from io import BytesIO
import folder_paths
import server
from ..py.workflow_data_utils import strip_runtime_objects

# Import numpy and PIL for image processing (available in ComfyUI environment)
try:
    import numpy as np
    from PIL import Image
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False
    print("[PromptManagerAdvanced] Warning: PIL/numpy not available, thumbnail from image input disabled")


def image_to_base64_thumbnail(image_tensor, max_size=200):
    """
    Convert a ComfyUI image tensor to a base64 thumbnail string.

    Args:
        image_tensor: ComfyUI image tensor (B, H, W, C) in float32 0-1 range
        max_size: Minimum dimension for the thumbnail (default 200px for smallest side)

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

        # Resize maintaining aspect ratio, limiting smallest dimension to max_size
        width, height = img.size
        min_dim = min(width, height)

        if min_dim > max_size:
            scale = max_size / min_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)

        # Convert to JPEG base64
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return f"data:image/jpeg;base64,{base64_str}"
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error converting image to thumbnail: {e}")
        return None


# ── Shared LoRA utilities ─────────────────────────────────────────────────────
from ..py.lora_utils import (
    get_available_loras,
    normalize_path_separators,
    strip_lora_extension,
    fuzzy_match_lora,
    get_lora_relative_path,
)


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
            all_prompts.update(k for k in category_prompts.keys() if k != "__meta__")

        all_prompts.add("")
        all_prompts_list = sorted(list(all_prompts))

        first_category = categories[0] if categories else "Default"

        # Get first prompt from first category
        first_prompt = ""
        first_prompt_text = ""
        if prompts_data and first_category in prompts_data and prompts_data[first_category]:
            first_category_prompts = [k for k in prompts_data[first_category].keys() if k != "__meta__"]
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
                "use_workflow_data": ("BOOLEAN", {
                    "default": False,
                    "label_on": "on",
                    "label_off": "off",
                    "tooltip": "When enabled, use connected workflow_data prompt/LoRAs at execution."
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": first_prompt_text,
                    "placeholder": "Enter prompt text",
                    "dynamicPrompts": False,
                    "tooltip": "Enter prompt text directly"
                }),
                "swap_lora_outputs": ("BOOLEAN", {
                    "default": False,
                    "label_on": "swapped",
                    "label_off": "normal",
                    "tooltip": "Swap the lora_stack_a and lora_stack_b outputs."
                }),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "forceInput": True, "lazy": True, "tooltip": "Connect prompt text input here"}),
                "lora_stack_a": ("LORA_STACK", {"tooltip": "First LoRA stack input (e.g., for base model)"}),
                "lora_stack_b": ("LORA_STACK", {"tooltip": "Second LoRA stack input (e.g., for video model)"}),
                "trigger_words": ("STRING", {"forceInput": True, "tooltip": "Comma-separated trigger words to append to prompt"}),
                "thumbnail_image": ("IMAGE", {"tooltip": "Connect an image to use as thumbnail when saving the prompt"}),
                "workflow_data": ("WORKFLOW_DATA", {"forceInput": True, "tooltip": "Connect workflow_data from WorkflowBuilder or PromptExtractor"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "loras_a_toggle": "STRING",
                "loras_b_toggle": "STRING",
                "trigger_words_toggle": "STRING",
                "saved_workflow_data": "STRING",
            }
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Full-featured prompt manager with dual LoRA stack support, trigger words, and thumbnail browser."
    RETURN_TYPES = ("STRING", "LORA_STACK", "LORA_STACK", "WORKFLOW_DATA")
    RETURN_NAMES = ("prompt", "lora_stack_a", "lora_stack_b", "workflow_data")
    FUNCTION = "get_prompt"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, category, name, use_prompt_input, use_lora_input=True, use_workflow_data=False, text="", swap_lora_outputs=False,
                   **kwargs):
        """
        Track changes to the node's inputs to determine if re-execution is needed.
        Returns a tuple of relevant values that should trigger re-execution when changed.
        """
        # Get optional inputs
        prompt_input = kwargs.get('prompt', None)
        lora_stack_a = kwargs.get('lora_stack_a', None)
        lora_stack_b = kwargs.get('lora_stack_b', None)
        trigger_words = kwargs.get('trigger_words', None)
        workflow_data = kwargs.get('workflow_data', None)
        # Return tuple of all values that should trigger re-execution when changed
        # Convert lists/objects to strings for hashable comparison
        return (
            category,
            name,
            use_prompt_input,
            use_lora_input,
            use_workflow_data,
            text,
            swap_lora_outputs,
            str(prompt_input) if prompt_input else None,
            str(lora_stack_a) if lora_stack_a else None,
            str(lora_stack_b) if lora_stack_b else None,
            trigger_words,
            str(workflow_data) if workflow_data else None,
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
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "default_prompts.json")

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
                # Try backup variants before falling back to defaults.
                # Never overwrite a user file just because parsing failed.
                backup_candidates = [
                    user_path + ".bak",
                    user_path + ".backup",
                    user_path + ".tmp",
                ]
                for backup_path in backup_candidates:
                    if not os.path.exists(backup_path):
                        continue
                    try:
                        with open(backup_path, 'r', encoding='utf-8') as f:
                            backup_data = json.load(f)
                        print(f"[PromptManagerAdvanced] Recovered prompts from backup: {backup_path}")
                        return backup_data
                    except Exception as be:
                        print(f"[PromptManagerAdvanced] Backup load failed ({backup_path}): {be}")
                print("[PromptManagerAdvanced] User prompt file exists but could not be parsed; not overwriting with defaults.")
                return {}

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
        """Sort categories and prompts alphabetically (case-insensitive), preserving __meta__"""
        sorted_data = {}
        for category in sorted(data.keys(), key=str.lower):
            cat_data = data[category]
            # Preserve __meta__ key (not a prompt), sort the rest
            meta = cat_data.get("__meta__")
            sorted_prompts = dict(sorted(
                ((k, v) for k, v in cat_data.items() if k != "__meta__"),
                key=lambda item: item[0].lower()
            ))
            if meta is not None:
                sorted_prompts["__meta__"] = meta
            sorted_data[category] = sorted_prompts
        return sorted_data

    @classmethod
    def save_prompts(cls, data):
        """Save prompts to user folder"""
        user_path = cls.get_prompts_path()
        sorted_data = cls.sort_prompts_data(data)
        tmp_path = user_path + ".tmp"
        bak_path = user_path + ".bak"

        try:
            os.makedirs(os.path.dirname(user_path), exist_ok=True)
            # Keep a rolling backup of the previous file before replacing it.
            if os.path.exists(user_path):
                shutil.copy2(user_path, bak_path)

            # Atomic write: write temp then replace to avoid truncated/corrupt files.
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, user_path)
        except Exception as e:
            print(f"[PromptManagerAdvanced] Error saving prompts: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

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

    def get_prompt(self, category, name, use_prompt_input, use_lora_input=True, use_workflow_data=False, text="", swap_lora_outputs=False,
                   prompt=None, lora_stack_a=None, lora_stack_b=None,
                   trigger_words=None, thumbnail_image=None, workflow_data=None,
                   unique_id=None, loras_a_toggle=None, loras_b_toggle=None, trigger_words_toggle=None,
                   saved_workflow_data=None):
        """Return the prompt text and filtered lora stacks based on toggle states"""

        # ========================================
        # RESET LOGIC - Determine if we should clear toggles and start fresh
        # ========================================
        # Track state for this node instance (stored in class variable per node)
        if not hasattr(self, '_last_known_state'):
            self._last_known_state = {}

        node_state_key = str(unique_id) if unique_id else "default"
        if node_state_key not in self._last_known_state:
            self._last_known_state[node_state_key] = {
                'prompt_key': '',
                'input_loras_a': [],
                'input_loras_b': [],
                'prompt_input_text': '',
                'toggle_data_a': '',
                'toggle_data_b': ''
            }

        last_state = self._last_known_state[node_state_key]

        # Current state signatures (using LoRA file paths directly - NO splitext)
        current_prompt_key = f"{category}|{name}"
        current_input_loras_a = [lora_path for lora_path, _, _ in lora_stack_a] if lora_stack_a else []
        current_input_loras_b = [lora_path for lora_path, _, _ in lora_stack_b] if lora_stack_b else []
        current_prompt_input_text = prompt if prompt else ""
        current_toggle_data_a = loras_a_toggle if loras_a_toggle else ""
        current_toggle_data_b = loras_b_toggle if loras_b_toggle else ""

        # Determine if reset is needed
        prompt_changed = current_prompt_key != last_state['prompt_key']
        input_loras_changed = (current_input_loras_a != last_state['input_loras_a'] or current_input_loras_b != last_state['input_loras_b'])
        prompt_input_changed = current_prompt_input_text != last_state['prompt_input_text']
        toggle_data_changed = (current_toggle_data_a != last_state['toggle_data_a'] or current_toggle_data_b != last_state['toggle_data_b'])

        # Check if input loras were cleared (had loras before, now empty)
        had_loras = last_state['input_loras_a'] or last_state['input_loras_b']
        now_has_no_loras = not current_input_loras_a and not current_input_loras_b
        loras_were_cleared = had_loras and now_has_no_loras

        # Reset conditions:
        # 1. Prompt dropdown changed
        # 2. Input loras AND prompt text both changed (new extraction)
        # 3. Input loras were cleared (switching from image with workflow to one without)
        # 4. Don't reset if ONLY toggle data changed (user is adjusting toggles)
        is_new_extraction = input_loras_changed and prompt_input_changed
        should_reset = prompt_changed or is_new_extraction or loras_were_cleared

        # Update tracked state
        self._last_known_state[node_state_key] = {
            'prompt_key': current_prompt_key,
            'input_loras_a': current_input_loras_a,
            'input_loras_b': current_input_loras_b,
            'prompt_input_text': current_prompt_input_text,
            'toggle_data_a': current_toggle_data_a,
            'toggle_data_b': current_toggle_data_b
        }

        # ========================================
        # CONTINUE WITH NORMAL PROCESSING
        # ========================================

        # Resolve workflow payload candidates.
        # IMPORTANT: when use_workflow_data is OFF, do not pass through live
        # upstream workflow_data updates to this node's workflow_data output.
        prompts_data = self.load_prompts()
        prompt_entry = prompts_data.get(category, {}).get(name, {}) if isinstance(prompts_data, dict) else {}
        stored_prompt_wf = prompt_entry.get("workflow_data") if isinstance(prompt_entry, dict) else None

        live_workflow_data = None
        if isinstance(workflow_data, dict):
            live_workflow_data = workflow_data
        elif isinstance(workflow_data, str) and workflow_data.strip():
            try:
                parsed = json.loads(workflow_data)
                if isinstance(parsed, dict):
                    live_workflow_data = parsed
            except (json.JSONDecodeError, TypeError):
                live_workflow_data = None

        hidden_saved_wf = None
        if isinstance(saved_workflow_data, str) and saved_workflow_data.strip():
            try:
                parsed_saved = json.loads(saved_workflow_data)
                if isinstance(parsed_saved, dict):
                    hidden_saved_wf = parsed_saved
            except (json.JSONDecodeError, TypeError):
                hidden_saved_wf = None

        if use_workflow_data:
            # Workflow mode ON: prefer live upstream, then hidden saved widget,
            # then prompt-stored workflow_data.
            resolved_workflow_data = live_workflow_data or hidden_saved_wf
            if resolved_workflow_data is None and isinstance(stored_prompt_wf, dict):
                resolved_workflow_data = stored_prompt_wf
        else:
            # Workflow mode OFF: never source from connected/hidden live payload.
            resolved_workflow_data = stored_prompt_wf if isinstance(stored_prompt_wf, dict) else None

        # Choose which text to use based on the toggles
        # Priority: use_prompt_input > use_workflow_data > internal text
        if use_prompt_input and prompt:
            output_text = prompt
        elif use_workflow_data and resolved_workflow_data:
            wf = resolved_workflow_data if isinstance(resolved_workflow_data, dict) else {}
            output_text = wf.get('positive_prompt', '') or text or ""
        else:
            output_text = text if text else ""

        # Capture the original input loras BEFORE any processing
        # These are used by the frontend to detect when inputs change
        # and clear toggle states accordingly
        input_loras_a = self._format_loras_for_display(lora_stack_a) if lora_stack_a else []
        input_loras_b = self._format_loras_for_display(lora_stack_b) if lora_stack_b else []

        # Build map of ORIGINAL strengths (before any user adjustments via toggles)
        # This is used for "Reset Strength" functionality
        original_strengths_a = {}
        original_strengths_b = {}

        # Get original strengths from saved prompt dict
        prompts = self.load_prompts()
        if prompts and category and name:
            prompt_data = prompts.get(category, {}).get(name, {})
            if prompt_data:
                for lora in prompt_data.get('loras_a', []):
                    lora_name = lora.get('name', '')
                    if lora_name:
                        original_strengths_a[lora_name] = lora.get('strength', lora.get('model_strength', 1.0))
                for lora in prompt_data.get('loras_b', []):
                    lora_name = lora.get('name', '')
                    if lora_name:
                        original_strengths_b[lora_name] = lora.get('strength', lora.get('model_strength', 1.0))

        # Add original strengths from connected input loras
        for lora in input_loras_a:
            lora_name = lora.get('name', '')
            if lora_name and lora_name not in original_strengths_a:
                original_strengths_a[lora_name] = lora.get('strength', lora.get('model_strength', 1.0))
        for lora in input_loras_b:
            lora_name = lora.get('name', '')
            if lora_name and lora_name not in original_strengths_b:
                original_strengths_b[lora_name] = lora.get('strength', lora.get('model_strength', 1.0))

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

        wf_loras_a = []
        wf_loras_b = []
        if use_workflow_data and resolved_workflow_data and isinstance(resolved_workflow_data, dict):
            wf_loras_a = [
                (wf_lora['name'], wf_lora.get('model_strength', 1.0), wf_lora.get('clip_strength', 1.0))
                for wf_lora in resolved_workflow_data.get('loras_a', []) if isinstance(wf_lora, dict) and wf_lora.get('name')
            ]
            wf_loras_b = [
                (wf_lora['name'], wf_lora.get('model_strength', 1.0), wf_lora.get('clip_strength', 1.0))
                for wf_lora in resolved_workflow_data.get('loras_b', []) if isinstance(wf_lora, dict) and wf_lora.get('name')
            ]

        # When use_lora_input is disabled, ignore connected stacks and use only saved loras
        if not use_lora_input:
            # If workflow mode is enabled and workflow_data carries LoRAs, show/use those.
            # Otherwise, fall back to saved preset LoRAs.
            lora_stack_a = wf_loras_a if wf_loras_a else preset_stack_a
            lora_stack_b = wf_loras_b if wf_loras_b else preset_stack_b
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

            # Merge workflow_data loras (lowest priority — existing stacks take precedence)
            if wf_loras_a:
                lora_stack_a = self._merge_lora_stacks(lora_stack_a, wf_loras_a)
            if wf_loras_b:
                lora_stack_b = self._merge_lora_stacks(lora_stack_b, wf_loras_b)

        # Process lora stacks with toggle data (filter inactive, apply strength)
        processed_stack_a = self._process_lora_toggle(lora_stack_a, loras_a_toggle)
        processed_stack_b = self._process_lora_toggle(lora_stack_b, loras_b_toggle)

        # Display logic: Include ALL loras (available + unavailable) so UI shows complete picture
        # Build display from available loras first, then add unavailable ones
        if not use_lora_input:
            if use_workflow_data and (wf_loras_a or wf_loras_b):
                # Workflow mode: display workflow LoRAs directly.
                loras_a_display = self._format_loras_for_display(lora_stack_a)
                loras_b_display = self._format_loras_for_display(lora_stack_b)
            else:
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
                "use_workflow_data": use_workflow_data,
                "prompt_input": prompt,
                "workflow_data": strip_runtime_objects(resolved_workflow_data) if isinstance(resolved_workflow_data, dict) else None,
                "loras_a": loras_a_display,
                "loras_b": loras_b_display,
                "input_loras_a": input_loras_a,  # Original input loras for change detection
                "input_loras_b": input_loras_b,  # Original input loras for change detection
                "unavailable_loras_a": unavailable_loras_a,  # Explicit list of unavailable lora names
                "unavailable_loras_b": unavailable_loras_b,  # Explicit list of unavailable lora names
                "trigger_words": trigger_words_display,
                "connected_thumbnail": thumbnail_base64,
                "should_reset": should_reset,  # Python tells JavaScript when to reset toggles
                "original_strengths_a": original_strengths_a,  # For "Reset Strength" button
                "original_strengths_b": original_strengths_b   # For "Reset Strength" button
            })

        # Append active trigger words to output
        final_output = self._append_trigger_words_to_prompt(output_text, active_trigger_words)

        # Swap outputs if requested
        out_stack_a = processed_stack_a if processed_stack_a else []
        out_stack_b = processed_stack_b if processed_stack_b else []
        if swap_lora_outputs:
            out_stack_a, out_stack_b = out_stack_b, out_stack_a

        # Build workflow_data output — start from incoming or saved data, update with PMA state
        if isinstance(resolved_workflow_data, dict):
            out_workflow_data = dict(resolved_workflow_data)
        else:
            out_workflow_data = {}
        out_workflow_data['positive_prompt'] = final_output
        out_workflow_data['loras_a'] = [
            {
                'name': lora.get('name', ''),
                'model_strength': lora.get('model_strength', lora.get('strength', 1.0)),
                'clip_strength': lora.get('clip_strength', lora.get('strength', 1.0)),
                'active': lora.get('active', True),
                'available': lora.get('available', True),
                'found': lora.get('available', True),
            }
            for lora in loras_a_display
            if isinstance(lora, dict) and lora.get('name')
        ]
        out_workflow_data['loras_b'] = [
            {
                'name': lora.get('name', ''),
                'model_strength': lora.get('model_strength', lora.get('strength', 1.0)),
                'clip_strength': lora.get('clip_strength', lora.get('strength', 1.0)),
                'active': lora.get('active', True),
                'available': lora.get('available', True),
                'found': lora.get('available', True),
            }
            for lora in loras_b_display
            if isinstance(lora, dict) and lora.get('name')
        ]
        out_workflow_data['_source'] = 'PromptManagerAdvanced'

        return (final_output, out_stack_a, out_stack_b, out_workflow_data)

    def check_lazy_status(self, category, name, use_prompt_input, use_lora_input=True, use_workflow_data=False, text="", swap_lora_outputs=False,
                          prompt=None, lora_stack_a=None, lora_stack_b=None,
                          trigger_words=None, thumbnail_image=None, workflow_data=None,
                          unique_id=None, loras_a_toggle=None, loras_b_toggle=None,
                          trigger_words_toggle=None, saved_workflow_data=None):
        """Tell ComfyUI which lazy inputs are needed based on current settings.

        When use_prompt_input is enabled, we need to request the prompt_input
        so ComfyUI will evaluate any connected nodes.
        """
        needed = []
        if use_prompt_input:
            needed.append("prompt")
        return needed

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
                "available": available,
                "found": available
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
                "available": found,
                "found": found
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
                    "available": lora['available'],
                    "found": lora.get('found', lora['available'])
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

        nsfw = data.get("nsfw", False)
        cat_data = {}
        if nsfw:
            cat_data["__meta__"] = {"nsfw": True}
        prompts[category_name] = cat_data
        PromptManagerAdvanced.save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in save_category API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager-advanced/rename-category")
async def rename_category_advanced(request):
    """API endpoint to rename a category"""
    try:
        data = await request.json()
        old_category = data.get("old_category", "").strip()
        new_category = data.get("new_category", "").strip()

        if not old_category or not new_category:
            return server.web.json_response({"success": False, "error": "Both old and new category names are required"})

        prompts = PromptManagerAdvanced.load_prompts()

        # Check if old category exists
        if old_category not in prompts:
            return server.web.json_response({"success": False, "error": f"Category '{old_category}' not found"})

        # Case-insensitive check for new category name conflicts (excluding the old name)
        existing_categories_lower = {k.lower(): k for k in prompts.keys() if k.lower() != old_category.lower()}
        if new_category.lower() in existing_categories_lower:
            return server.web.json_response({
                "success": False,
                "error": f"Category already exists as '{existing_categories_lower[new_category.lower()]}'"
            })

        # Rename by copying data and deleting old
        prompts[new_category] = prompts[old_category]
        del prompts[old_category]
        PromptManagerAdvanced.save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts, "new_category": new_category})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in rename_category API: {e}")
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
        # Get existing prompt data BEFORE potentially deleting it (to preserve thumbnail)
        existing_prompts_lower = {k.lower(): k for k in prompts[category].keys()}
        existing_prompt = {}
        if name.lower() in existing_prompts_lower:
            old_name = existing_prompts_lower[name.lower()]
            existing_prompt = prompts[category].get(old_name, {})
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
                    available = lora.get('available', lora.get('found', True))
                    found = lora.get('found', available)
                    normalized.append({
                        "name": lora.get('name'),
                        "strength": lora.get('strength', lora.get('model_strength', 1.0)),
                        "clip_strength": lora.get('clip_strength', lora.get('strength', 1.0)),
                        "active": lora.get('active', True),
                        "available": bool(available),
                        "found": bool(found),
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
        if thumbnail is None:
            thumbnail = existing_prompt.get("thumbnail")

        # Save prompt with normalized lora data (no paths - they're resolved at runtime)
        prompt_data = {
            "prompt": text,
            "loras_a": normalize_lora_data(loras_a),
            "loras_b": normalize_lora_data(loras_b),
            "trigger_words": normalize_trigger_words(trigger_words)
        }

        # Save workflow_data if provided.
        # Always sanitize to remove runtime-only objects (MODEL_A/MODEL_B/CLIP/VAE)
        # and accept either dict or JSON string payloads from frontend.
        workflow_data = data.get("workflow_data")
        wf_to_save = None
        if isinstance(workflow_data, dict):
            wf_to_save = strip_runtime_objects(workflow_data)
        elif isinstance(workflow_data, str) and workflow_data.strip():
            try:
                parsed_wf = json.loads(workflow_data)
                if isinstance(parsed_wf, dict):
                    wf_to_save = strip_runtime_objects(parsed_wf)
            except (json.JSONDecodeError, TypeError):
                wf_to_save = None

        if wf_to_save:
            prompt_data["workflow_data"] = wf_to_save
        else:
            # Preserve previously saved workflow_data when user saves prompt/LoRA
            # edits without a currently connected workflow_data source.
            existing_wf = existing_prompt.get("workflow_data")
            if isinstance(existing_wf, dict):
                prompt_data["workflow_data"] = existing_wf

        if thumbnail:
            prompt_data["thumbnail"] = thumbnail

        # Preserve or set NSFW flag
        nsfw = data.get("nsfw")
        if nsfw is not None:
            prompt_data["nsfw"] = bool(nsfw)
        elif existing_prompt.get("nsfw"):
            prompt_data["nsfw"] = existing_prompt["nsfw"]

        # Preserve any extra fields from an existing prompt that this node does not manage.
        # This includes workflow_config (added by WorkflowManager) and any future extensions.
        # Known managed keys — everything else is preserved verbatim.
        _MANAGED_KEYS = {"prompt", "loras_a", "loras_b", "trigger_words", "thumbnail", "nsfw", "workflow_data"}
        for extra_key, extra_val in existing_prompt.items():
            if extra_key not in _MANAGED_KEYS and extra_key not in prompt_data:
                prompt_data[extra_key] = extra_val

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


@server.PromptServer.instance.routes.post("/prompt-manager-advanced/rename-prompt")
async def rename_prompt_advanced(request):
    """API endpoint to rename/move a prompt, optionally to a different category"""
    try:
        data = await request.json()
        category = data.get("category", "").strip()
        old_name = data.get("old_name", "").strip()
        new_name = data.get("new_name", "").strip()
        new_category = data.get("new_category", "").strip() or category

        if not category or not old_name or not new_name:
            return server.web.json_response({"success": False, "error": "Category, old name, and new name are required"})

        prompts = PromptManagerAdvanced.load_prompts()

        if category not in prompts or old_name not in prompts[category]:
            return server.web.json_response({"success": False, "error": "Prompt not found"})

        if new_category not in prompts:
            return server.web.json_response({"success": False, "error": f"Category '{new_category}' not found"})

        # Check for name conflicts in the target category
        # If same category, exclude the old name from conflict check
        target_prompts = prompts[new_category]
        if category == new_category:
            existing_lower = {k.lower(): k for k in target_prompts.keys() if k != "__meta__" and k.lower() != old_name.lower()}
        else:
            existing_lower = {k.lower(): k for k in target_prompts.keys() if k != "__meta__"}

        if new_name.lower() in existing_lower:
            return server.web.json_response({
                "success": False,
                "error": f"A prompt named '{existing_lower[new_name.lower()]}' already exists in '{new_category}'"
            })

        prompts[new_category][new_name] = prompts[category][old_name]
        del prompts[category][old_name]
        PromptManagerAdvanced.save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts, "new_name": new_name, "new_category": new_category})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in rename_prompt API: {e}")
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
                "trigger_words": prompt_data.get("trigger_words", []),
                "workflow_data": prompt_data.get("workflow_data")
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

                # Handle __meta__ key (category metadata, not a prompt)
                if prompt_name == "__meta__" and isinstance(prompt_data, dict):
                    prompts[category]["__meta__"] = prompt_data
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

                # Preserve NSFW flag if present
                if prompt_data.get("nsfw"):
                    normalized["nsfw"] = prompt_data["nsfw"]

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


@server.PromptServer.instance.routes.post("/prompt-manager-advanced/toggle-nsfw")
async def toggle_nsfw_advanced(request):
    """API endpoint to toggle NSFW flag on a category or prompt"""
    try:
        data = await request.json()
        toggle_type = data.get("type", "").strip()  # "category" or "prompt"
        category = data.get("category", "").strip()
        name = data.get("name", "").strip()

        if not category:
            return server.web.json_response({"success": False, "error": "Category is required"})

        prompts = PromptManagerAdvanced.load_prompts()

        if category not in prompts:
            return server.web.json_response({"success": False, "error": "Category not found"})

        if toggle_type == "category":
            meta = prompts[category].get("__meta__", {})
            meta["nsfw"] = not meta.get("nsfw", False)
            prompts[category]["__meta__"] = meta
        elif toggle_type == "prompt":
            if not name or name not in prompts[category]:
                return server.web.json_response({"success": False, "error": "Prompt not found"})
            prompts[category][name]["nsfw"] = not prompts[category][name].get("nsfw", False)
        else:
            return server.web.json_response({"success": False, "error": "Invalid type, must be 'category' or 'prompt'"})

        PromptManagerAdvanced.save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in toggle_nsfw API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.get("/prompt-manager-advanced/list-checkpoints")
async def list_checkpoints(request):
    """API endpoint to list available checkpoints for thumbnail generation"""
    try:
        checkpoints = folder_paths.get_filename_list("checkpoints")
        return server.web.json_response({"success": True, "checkpoints": checkpoints})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error listing checkpoints: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager-advanced/resolve-loras")
async def resolve_loras(request):
    """API endpoint to resolve LoRA names to paths for thumbnail generation workflow"""
    try:
        data = await request.json()
        lora_names = data.get("lora_names", [])

        resolved = []
        for lora in lora_names:
            name = lora.get("name", "")
            path, found = get_lora_relative_path(name)
            if found:
                resolved.append({
                    "path": path,
                    "strength": lora.get("strength", 1.0),
                    "clip_strength": lora.get("clip_strength", 1.0)
                })

        return server.web.json_response({"success": True, "loras": resolved})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error resolving LoRAs: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)
