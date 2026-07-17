"""
ComfyUI Prompt Manager Advanced - Extended prompt management with LoRA stack support
Saves prompts along with associated LoRA configurations
"""
import os
import json
import shutil
import time
import base64
from datetime import datetime
from io import BytesIO
import folder_paths
import server
from ..py.workflow_data_utils import ensure_v2_recipe_data, to_json_safe_workflow_data, build_v2_recipe_data_from_prompt

# Import numpy and PIL for image processing (available in ComfyUI environment)
try:
    import numpy as np
    from PIL import Image
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False
    print("[PromptManagerAdvanced] Warning: PIL/numpy not available, thumbnail from image input disabled")


def _get_workflow_node(extra_pnginfo, node_id: str):
    """Find workflow node by id, including nested subgraphs (id chains like a:b:c)."""
    if not isinstance(extra_pnginfo, dict):
        return None

    workflow = extra_pnginfo.get("workflow")
    if not isinstance(workflow, dict):
        return None

    nodes_list = workflow.get("nodes", []) if isinstance(workflow.get("nodes"), list) else []
    subgraphs = []
    definitions = workflow.get("definitions")
    if isinstance(definitions, dict):
        maybe_subgraphs = definitions.get("subgraphs")
        if isinstance(maybe_subgraphs, list):
            subgraphs = maybe_subgraphs

    found = None
    for part in str(node_id).split(":"):
        found = next((n for n in nodes_list if isinstance(n, dict) and str(n.get("id")) == part), None)
        if not isinstance(found, dict):
            return None

        subgraph = next((sg for sg in subgraphs if isinstance(sg, dict) and str(sg.get("id")) == str(found.get("type"))), None)
        if isinstance(subgraph, dict) and isinstance(subgraph.get("nodes"), list):
            nodes_list = subgraph.get("nodes", [])

    return found


def _patch_runtime_prompt_metadata(unique_id, output_text, extra_pnginfo=None, api_prompt=None):
    """Persist resolved prompt into workflow/api metadata for downstream save nodes."""
    node_id = str(unique_id) if unique_id is not None else ""
    if not node_id:
        return

    # Patch workflow metadata (EXTRA_PNGINFO).
    workflow_node = _get_workflow_node(extra_pnginfo, node_id)
    if isinstance(workflow_node, dict):
        widgets = workflow_node.get("widgets_values")
        if isinstance(widgets, list):
            # PromptManagerAdvanced widgets order:
            # [category, name, use_prompt_input, use_lora_input, text, swap_lora_outputs]
            if len(widgets) > 2:
                widgets[2] = False
            if len(widgets) > 4:
                widgets[4] = output_text

    # Patch API prompt metadata (PROMPT).
    if isinstance(api_prompt, dict):
        prompt_node = api_prompt.get(node_id)
        if isinstance(prompt_node, dict):
            inputs = prompt_node.get("inputs")
            if isinstance(inputs, dict):
                inputs["use_prompt_input"] = False
                inputs["text"] = output_text


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


def _has_meaningful_workflow_data(workflow_data):
    """Return True when workflow_data contains authored prompt/model/lora content."""
    if not isinstance(workflow_data, dict):
        return False

    if int(workflow_data.get("version", 0) or 0) >= 2 and isinstance(workflow_data.get("models"), dict):
        for mk in ("model_a", "model_b", "model_c", "model_d"):
            block = workflow_data.get("models", {}).get(mk)
            if not isinstance(block, dict):
                continue
            positive_prompt = str(block.get("positive_prompt", "")).strip()
            model_name = str(block.get("model", "")).strip()
            loras = block.get("loras")
            has_loras = isinstance(loras, list) and any(
                isinstance(lora_item, dict) and str(lora_item.get("name", "")).strip()
                for lora_item in loras
            )
            if positive_prompt or model_name or has_loras:
                return True
        return False

    positive_prompt = str(workflow_data.get("positive_prompt", "")).strip()
    model_a = str(workflow_data.get("model_a", "")).strip()
    model_b = str(workflow_data.get("model_b", "")).strip()

    loras_a = workflow_data.get("loras_a")
    has_loras_a = isinstance(loras_a, list) and any(
        isinstance(lora_item, dict) and str(lora_item.get("name", "")).strip()
        for lora_item in loras_a
    )

    loras_b = workflow_data.get("loras_b")
    has_loras_b = isinstance(loras_b, list) and any(
        isinstance(lora_item, dict) and str(lora_item.get("name", "")).strip()
        for lora_item in loras_b
    )

    return bool(positive_prompt or model_a or model_b or has_loras_a or has_loras_b)


def _normalize_workflow_loras_for_prompt(loras):
    """Normalize workflow_data LoRA list to prompt-manager LoRA schema."""
    normalized = []
    if not isinstance(loras, list):
        return normalized
    for lora in loras:
        if not isinstance(lora, dict):
            continue
        name = str(lora.get("name", "")).strip()
        if not name:
            continue
        strength = lora.get("strength", lora.get("model_strength", 1.0))
        clip_strength = lora.get("clip_strength", strength)
        normalized.append({
            "name": name,
            "path": str(lora.get("path", name) or name),
            "strength": strength,
            "clip_strength": clip_strength,
            "active": lora.get("active", True),
            "available": bool(lora.get("available", True)),
        })
    return normalized


def _derive_prompt_fields_from_workflow_data(workflow_data):
    """Derive prompt + LoRA compatibility fields from v2 workflow_data model blocks."""
    if not isinstance(workflow_data, dict):
        return {
            "prompt": "",
            "negative_prompt": "",
            "loras_a": [],
            "loras_b": [],
            "loras_c": [],
            "loras_d": [],
            "model_a": "",
            "model_b": "",
            "model_c": "",
            "model_d": "",
        }

    wf_v2 = ensure_v2_recipe_data(workflow_data, source="PromptManagerAdvanced")
    wf_models = wf_v2.get("models", {}) if isinstance(wf_v2.get("models"), dict) else {}
    model_a_block = wf_models.get("model_a") if isinstance(wf_models.get("model_a"), dict) else {}
    model_b_block = wf_models.get("model_b") if isinstance(wf_models.get("model_b"), dict) else {}
    model_c_block = wf_models.get("model_c") if isinstance(wf_models.get("model_c"), dict) else {}
    model_d_block = wf_models.get("model_d") if isinstance(wf_models.get("model_d"), dict) else {}

    return {
        "prompt": str(model_a_block.get("positive_prompt", "") or ""),
        "negative_prompt": str(model_a_block.get("negative_prompt", "") or ""),
        "loras_a": _normalize_workflow_loras_for_prompt(model_a_block.get("loras", [])),
        "loras_b": _normalize_workflow_loras_for_prompt(model_b_block.get("loras", [])),
        "loras_c": _normalize_workflow_loras_for_prompt(model_c_block.get("loras", [])),
        "loras_d": _normalize_workflow_loras_for_prompt(model_d_block.get("loras", [])),
        "model_a": str(model_a_block.get("model", "") or ""),
        "model_b": str(model_b_block.get("model", "") or ""),
        "model_c": str(model_c_block.get("model", "") or ""),
        "model_d": str(model_d_block.get("model", "") or ""),
    }


# ── Shared LoRA utilities ─────────────────────────────────────────────────────
from ..py.lora_utils import (
    get_available_loras,
    normalize_path_separators,
    strip_lora_extension,
    fuzzy_match_lora,
    get_lora_relative_path,
)


_MODEL_SLOT_KEYS = ("a", "b", "c", "d")
_LORA_INPUT_MODE_PROMPT_ONLY = "Prompt LoRAs Only"
_LORA_INPUT_MODE_COMBINE = "Combine LoRAs"
_LORA_INPUT_MODE_INPUT_ONLY = "Input LoRAs Only"
_LORA_INPUT_MODES = (
    _LORA_INPUT_MODE_PROMPT_ONLY,
    _LORA_INPUT_MODE_COMBINE,
    _LORA_INPUT_MODE_INPUT_ONLY,
)


def _coerce_lora_stack(raw_stack):
    """Normalize a LORA_STACK-like payload into list[(path, model, clip)]."""
    if raw_stack is None:
        return []

    if isinstance(raw_stack, dict) and "__value__" in raw_stack:
        raw_stack = raw_stack.get("__value__")

    if not isinstance(raw_stack, list):
        return []

    out = []
    for item in raw_stack:
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            path = str(item[0] or "").strip()
            if not path:
                continue
            try:
                model_strength = float(item[1]) if len(item) >= 2 else 1.0
            except Exception:
                model_strength = 1.0
            try:
                clip_strength = float(item[2]) if len(item) >= 3 else model_strength
            except Exception:
                clip_strength = model_strength
            out.append((path, model_strength, clip_strength))
            continue

        if isinstance(item, dict):
            path = str(item.get("path") or item.get("name") or "").strip()
            if not path:
                continue
            try:
                model_strength = float(item.get("model_strength", item.get("strength", 1.0)))
            except Exception:
                model_strength = 1.0
            try:
                clip_strength = float(item.get("clip_strength", model_strength))
            except Exception:
                clip_strength = model_strength
            out.append((path, model_strength, clip_strength))

    return out


def _normalize_input_multi_lora_payload(raw_input, primary_slot):
    """Normalize one PMA input socket into 4 slots (A/B/C/D)."""
    slots = {k: [] for k in _MODEL_SLOT_KEYS}

    plain = _coerce_lora_stack(raw_input)
    if plain:
        slots[primary_slot] = plain
        return slots

    if not isinstance(raw_input, dict):
        return slots

    for slot_key, suffix in (("model_a", "a"), ("model_b", "b"), ("model_c", "c"), ("model_d", "d")):
        candidate = None
        if slot_key in raw_input:
            candidate = raw_input.get(slot_key)
        elif suffix in raw_input:
            candidate = raw_input.get(suffix)
        slots[suffix] = _coerce_lora_stack(candidate)

    return slots


def _combine_multi_lora_inputs(input_a, input_b):
    """Combine PMA A/B sockets into a single multi-slot payload."""
    normalized_a = _normalize_input_multi_lora_payload(input_a, "a")
    normalized_b = _normalize_input_multi_lora_payload(input_b, "b")

    out = {}
    for slot in _MODEL_SLOT_KEYS:
        out[slot] = [*(normalized_a.get(slot) or []), *(normalized_b.get(slot) or [])]
    return out


class PromptManagerAdvanced:
    """
    Advanced Prompt Manager with LoRA stack integration.
    Supports saving/loading prompts with associated LoRA configurations.
    Features two LoRA stack inputs/outputs for dual LoRA workflows (e.g., Wan video).
    """

    @staticmethod
    def get_weekly_backup_path():
        """Get the path to the weekly rotating backup file."""
        return PromptManagerAdvanced.get_prompts_path() + "_bak"

    @staticmethod
    def _refresh_weekly_backup_if_due(source_path, backup_path, interval_days=7):
        """Refresh backup only when missing or older than interval_days."""
        if not os.path.exists(source_path):
            return

        should_refresh = not os.path.exists(backup_path)
        if not should_refresh:
            try:
                age_seconds = time.time() - os.path.getmtime(backup_path)
                should_refresh = age_seconds >= (interval_days * 24 * 60 * 60)
            except Exception:
                should_refresh = True

        if should_refresh:
            shutil.copy2(source_path, backup_path)

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
        # Keep new node instances empty until the user explicitly selects a prompt.
        first_prompt = ""
        first_prompt_text = ""

        return {
            "required": {
                "category": (categories, {"default": first_category}),
                "name": (all_prompts_list, {"default": first_prompt}),
                "use_prompt_input": ("BOOLEAN", {
                    "default": False,
                    "label_on": "on",
                    "label_off": "off",
                    "tooltip": "Toggle to use connected prompt input instead of internal text"}),
                "use_lora_input": (_LORA_INPUT_MODES, {
                    "default": _LORA_INPUT_MODE_PROMPT_ONLY,
                    "tooltip": (
                        "LoRA mode. Prompt LoRAs Only = use saved prompt stacks only. "
                        "Combine LoRAs = merge connected stacks with prompt stacks. "
                        "Input LoRAs Only = use only connected input stacks."
                    ),
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
                "lora_stack_a": ("LORA_STACK,MULTI_LORA_STACK", {"forceInput": True, "tooltip": "First LoRA stack input. Accepts LORA_STACK or MULTI_LORA_STACK."}),
                "lora_stack_b": ("LORA_STACK,MULTI_LORA_STACK", {"forceInput": True, "tooltip": "Second LoRA stack input. Accepts LORA_STACK or MULTI_LORA_STACK."}),
                "trigger_words": ("STRING", {"forceInput": True, "tooltip": "Comma-separated trigger words to append to prompt"}),
                "thumbnail_image": ("IMAGE", {"tooltip": "Connect an image to use as thumbnail when saving the prompt"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "loras_a_toggle": "STRING",
                "loras_b_toggle": "STRING",
                "loras_c_toggle": "STRING",
                "loras_d_toggle": "STRING",
                "trigger_words_toggle": "STRING",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "api_prompt": "PROMPT",
            }
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Full-featured prompt manager with multi-slot LoRA stack support, trigger words, and thumbnail browser."
    RETURN_TYPES = ("STRING", "LORA_STACK", "LORA_STACK", "RECIPE_DATA", "MULTI_LORA_STACK")
    RETURN_NAMES = ("prompt", "lora_stack_a", "lora_stack_b", "recipe_data", "multi_lora_stack")
    FUNCTION = "get_prompt"
    OUTPUT_NODE = True

    @staticmethod
    def _normalize_lora_input_mode(mode_value):
        """Normalize legacy bool/new string modes to canonical dropdown values."""
        if mode_value is False:
            return _LORA_INPUT_MODE_PROMPT_ONLY
        if mode_value is True:
            return _LORA_INPUT_MODE_COMBINE

        value = str(mode_value or "").strip()
        if value in _LORA_INPUT_MODES:
            return value

        lowered = value.lower()
        if lowered in ("off", "false", "0", "prompt", "prompt only", "prompt loras", "prompt loras only"):
            return _LORA_INPUT_MODE_PROMPT_ONLY
        if lowered in ("on", "true", "1", "combine", "combine loras"):
            return _LORA_INPUT_MODE_COMBINE
        if lowered in ("2", "input", "input only", "input loras", "input loras only"):
            return _LORA_INPUT_MODE_INPUT_ONLY

        return _LORA_INPUT_MODE_PROMPT_ONLY

    @staticmethod
    def _should_use_combined_loras(mode_value):
        return PromptManagerAdvanced._normalize_lora_input_mode(mode_value) == _LORA_INPUT_MODE_COMBINE

    @staticmethod
    def _should_use_input_only_loras(mode_value):
        return PromptManagerAdvanced._normalize_lora_input_mode(mode_value) == _LORA_INPUT_MODE_INPUT_ONLY

    @classmethod
    def IS_CHANGED(cls, category, name, use_prompt_input, use_lora_input=_LORA_INPUT_MODE_PROMPT_ONLY, text="", swap_lora_outputs=False,
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
        # Return tuple of all values that should trigger re-execution when changed
        # Convert lists/objects to strings for hashable comparison
        return (
            category,
            name,
            use_prompt_input,
            cls._normalize_lora_input_mode(use_lora_input),
            text,
            swap_lora_outputs,
            str(prompt_input) if prompt_input else None,
            str(lora_stack_a) if lora_stack_a else None,
            str(lora_stack_b) if lora_stack_b else None,
            trigger_words,
        )

    @classmethod
    def VALIDATE_INPUTS(cls, name, input_types=None, **kwargs):
        """Allow any name and validate optional LoRA input socket types."""
        for key in ("lora_stack_a", "lora_stack_b"):
            t = (input_types or {}).get(key) if isinstance(input_types, dict) else None
            if t is None:
                continue
            if t not in ("LORA_STACK", "MULTI_LORA_STACK"):
                return f"Unsupported type for {key}: {t}"
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
                    cls.get_weekly_backup_path(),
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
        bak_path = cls.get_weekly_backup_path()

        try:
            os.makedirs(os.path.dirname(user_path), exist_ok=True)
            # Weekly rotating backup: refresh at most once every 7 days.
            cls._refresh_weekly_backup_if_due(user_path, bak_path)

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

    def get_prompt(self, category, name, use_prompt_input, use_lora_input=_LORA_INPUT_MODE_PROMPT_ONLY, text="", swap_lora_outputs=False,
                   prompt=None, lora_stack_a=None, lora_stack_b=None,
                   trigger_words=None, thumbnail_image=None,
                   unique_id=None, loras_a_toggle=None, loras_b_toggle=None, loras_c_toggle=None, loras_d_toggle=None, trigger_words_toggle=None,
                   extra_pnginfo=None, api_prompt=None,
                   **kwargs):
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
                'input_loras_c': [],
                'input_loras_d': [],
                'prompt_input_text': '',
                'toggle_data_a': '',
                'toggle_data_b': '',
                'toggle_data_c': '',
                'toggle_data_d': '',
            }

        last_state = self._last_known_state[node_state_key]

        # Current state signatures (using LoRA file paths directly - NO splitext)
        current_prompt_key = f"{category}|{name}"
        combined_input_multi = _combine_multi_lora_inputs(lora_stack_a, lora_stack_b)
        current_input_loras_a = [lora_path for lora_path, _, _ in combined_input_multi.get('a', [])]
        current_input_loras_b = [lora_path for lora_path, _, _ in combined_input_multi.get('b', [])]
        current_input_loras_c = [lora_path for lora_path, _, _ in combined_input_multi.get('c', [])]
        current_input_loras_d = [lora_path for lora_path, _, _ in combined_input_multi.get('d', [])]
        current_prompt_input_text = prompt if prompt else ""
        current_toggle_data_a = loras_a_toggle if loras_a_toggle else ""
        current_toggle_data_b = loras_b_toggle if loras_b_toggle else ""
        current_toggle_data_c = loras_c_toggle if loras_c_toggle else ""
        current_toggle_data_d = loras_d_toggle if loras_d_toggle else ""

        # Determine if reset is needed
        prompt_changed = current_prompt_key != last_state['prompt_key']
        input_loras_changed = (
            current_input_loras_a != last_state['input_loras_a'] or
            current_input_loras_b != last_state['input_loras_b'] or
            current_input_loras_c != last_state['input_loras_c'] or
            current_input_loras_d != last_state['input_loras_d']
        )
        prompt_input_changed = current_prompt_input_text != last_state['prompt_input_text']
        toggle_data_changed = (
            current_toggle_data_a != last_state['toggle_data_a'] or
            current_toggle_data_b != last_state['toggle_data_b'] or
            current_toggle_data_c != last_state['toggle_data_c'] or
            current_toggle_data_d != last_state['toggle_data_d']
        )

        # Check if input loras were cleared (had loras before, now empty)
        had_loras = last_state['input_loras_a'] or last_state['input_loras_b'] or last_state['input_loras_c'] or last_state['input_loras_d']
        now_has_no_loras = not current_input_loras_a and not current_input_loras_b and not current_input_loras_c and not current_input_loras_d
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
            'input_loras_c': current_input_loras_c,
            'input_loras_d': current_input_loras_d,
            'prompt_input_text': current_prompt_input_text,
            'toggle_data_a': current_toggle_data_a,
            'toggle_data_b': current_toggle_data_b,
            'toggle_data_c': current_toggle_data_c,
            'toggle_data_d': current_toggle_data_d,
        }

        lora_input_mode = self._normalize_lora_input_mode(use_lora_input)
        use_combined_loras = self._should_use_combined_loras(lora_input_mode)
        use_input_only_loras = self._should_use_input_only_loras(lora_input_mode)

        # ========================================
        # CONTINUE WITH NORMAL PROCESSING
        # ========================================

        prompts_data = self.load_prompts()
        prompt_entry = prompts_data.get(category, {}).get(name, {}) if isinstance(prompts_data, dict) else {}
        stored_prompt_wf = prompt_entry.get("workflow_data") if isinstance(prompt_entry, dict) else None
        resolved_workflow_data = ensure_v2_recipe_data(stored_prompt_wf, source="PromptManagerAdvanced") if isinstance(stored_prompt_wf, dict) else None
        workflow_fields = _derive_prompt_fields_from_workflow_data(resolved_workflow_data)

        # Choose which text to use based on the toggles
        # Priority: use_prompt_input > internal text
        if use_prompt_input and prompt:
            output_text = prompt
        else:
            output_text = text if text else ""
            if not output_text.strip() and workflow_fields.get("prompt"):
                # Compatibility fallback: v2 prompt is authoritative in model_a block.
                output_text = workflow_fields.get("prompt", "")

        # Capture the original input loras BEFORE any processing
        # These are used by the frontend to detect when inputs change
        # and clear toggle states accordingly
        input_loras_a = self._format_loras_for_display(combined_input_multi.get('a')) if combined_input_multi.get('a') else []
        input_loras_b = self._format_loras_for_display(combined_input_multi.get('b')) if combined_input_multi.get('b') else []
        input_loras_c = self._format_loras_for_display(combined_input_multi.get('c')) if combined_input_multi.get('c') else []
        input_loras_d = self._format_loras_for_display(combined_input_multi.get('d')) if combined_input_multi.get('d') else []

        # Build map of ORIGINAL strengths (before any user adjustments via toggles)
        # This is used for "Reset Strength" functionality
        original_strengths_a = {}
        original_strengths_b = {}
        original_strengths_c = {}
        original_strengths_d = {}

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
                for lora in prompt_data.get('loras_c', []):
                    lora_name = lora.get('name', '')
                    if lora_name:
                        original_strengths_c[lora_name] = lora.get('strength', lora.get('model_strength', 1.0))
                for lora in prompt_data.get('loras_d', []):
                    lora_name = lora.get('name', '')
                    if lora_name:
                        original_strengths_d[lora_name] = lora.get('strength', lora.get('model_strength', 1.0))

        # Add original strengths from connected input loras
        for lora in input_loras_a:
            lora_name = lora.get('name', '')
            if lora_name and lora_name not in original_strengths_a:
                original_strengths_a[lora_name] = lora.get('strength', lora.get('model_strength', 1.0))
        for lora in input_loras_b:
            lora_name = lora.get('name', '')
            if lora_name and lora_name not in original_strengths_b:
                original_strengths_b[lora_name] = lora.get('strength', lora.get('model_strength', 1.0))
        for lora in input_loras_c:
            lora_name = lora.get('name', '')
            if lora_name and lora_name not in original_strengths_c:
                original_strengths_c[lora_name] = lora.get('strength', lora.get('model_strength', 1.0))
        for lora in input_loras_d:
            lora_name = lora.get('name', '')
            if lora_name and lora_name not in original_strengths_d:
                original_strengths_d[lora_name] = lora.get('strength', lora.get('model_strength', 1.0))

        # Process trigger words
        trigger_words_display = self._process_trigger_words(
            trigger_words, trigger_words_toggle
        )
        active_trigger_words = self._get_active_trigger_words(trigger_words_toggle, trigger_words)

        # Build preset stacks from saved toggle data (these are what we display/manage)
        preset_stack_a = self._build_stack_from_toggle(loras_a_toggle) if loras_a_toggle else []
        preset_stack_b = self._build_stack_from_toggle(loras_b_toggle) if loras_b_toggle else []
        preset_stack_c = self._build_stack_from_toggle(loras_c_toggle) if loras_c_toggle else []
        preset_stack_d = self._build_stack_from_toggle(loras_d_toggle) if loras_d_toggle else []

        # Compatibility fallback: when no saved toggle payload exists, seed
        # PMA stack defaults from v2 model_a/model_b LoRA blocks.
        if not preset_stack_a and not loras_a_toggle:
            preset_stack_a = [
                (
                    lora.get("name", ""),
                    lora.get("strength", lora.get("model_strength", 1.0)),
                    lora.get("clip_strength", lora.get("strength", lora.get("model_strength", 1.0))),
                )
                for lora in workflow_fields.get("loras_a", [])
                if isinstance(lora, dict) and lora.get("name")
            ]
        if not preset_stack_b and not loras_b_toggle:
            preset_stack_b = [
                (
                    lora.get("name", ""),
                    lora.get("strength", lora.get("model_strength", 1.0)),
                    lora.get("clip_strength", lora.get("strength", lora.get("model_strength", 1.0))),
                )
                for lora in workflow_fields.get("loras_b", [])
                if isinstance(lora, dict) and lora.get("name")
            ]
        if not preset_stack_c and not loras_c_toggle:
            preset_stack_c = [
                (
                    lora.get("name", ""),
                    lora.get("strength", lora.get("model_strength", 1.0)),
                    lora.get("clip_strength", lora.get("strength", lora.get("model_strength", 1.0))),
                )
                for lora in workflow_fields.get("loras_c", [])
                if isinstance(lora, dict) and lora.get("name")
            ]
        if not preset_stack_d and not loras_d_toggle:
            preset_stack_d = [
                (
                    lora.get("name", ""),
                    lora.get("strength", lora.get("model_strength", 1.0)),
                    lora.get("clip_strength", lora.get("strength", lora.get("model_strength", 1.0))),
                )
                for lora in workflow_fields.get("loras_d", [])
                if isinstance(lora, dict) and lora.get("name")
            ]

        # Get ALL preset loras including unavailable ones (for display purposes)
        all_preset_loras_a = self._get_all_loras_from_toggle(loras_a_toggle) if loras_a_toggle else []
        all_preset_loras_b = self._get_all_loras_from_toggle(loras_b_toggle) if loras_b_toggle else []
        all_preset_loras_c = self._get_all_loras_from_toggle(loras_c_toggle) if loras_c_toggle else []
        all_preset_loras_d = self._get_all_loras_from_toggle(loras_d_toggle) if loras_d_toggle else []

        # If no saved toggle payload exists yet, preserve active flags from
        # incoming workflow_data loras as the initial authoritative state.
        if not all_preset_loras_a and not loras_a_toggle:
            all_preset_loras_a = [
                {
                    "name": lora.get("name", ""),
                    "strength": float(lora.get("strength", lora.get("model_strength", 1.0))),
                    "clip_strength": float(lora.get("clip_strength", lora.get("strength", lora.get("model_strength", 1.0)))),
                    "active": lora.get("active", True),
                    "available": bool(lora.get("available", True)),
                }
                for lora in workflow_fields.get("loras_a", [])
                if isinstance(lora, dict) and lora.get("name")
            ]
        if not all_preset_loras_b and not loras_b_toggle:
            all_preset_loras_b = [
                {
                    "name": lora.get("name", ""),
                    "strength": float(lora.get("strength", lora.get("model_strength", 1.0))),
                    "clip_strength": float(lora.get("clip_strength", lora.get("strength", lora.get("model_strength", 1.0)))),
                    "active": lora.get("active", True),
                    "available": bool(lora.get("available", True)),
                }
                for lora in workflow_fields.get("loras_b", [])
                if isinstance(lora, dict) and lora.get("name")
            ]
        if not all_preset_loras_c and not loras_c_toggle:
            all_preset_loras_c = [
                {
                    "name": lora.get("name", ""),
                    "strength": float(lora.get("strength", lora.get("model_strength", 1.0))),
                    "clip_strength": float(lora.get("clip_strength", lora.get("strength", lora.get("model_strength", 1.0)))),
                    "active": lora.get("active", True),
                    "available": bool(lora.get("available", True)),
                }
                for lora in workflow_fields.get("loras_c", [])
                if isinstance(lora, dict) and lora.get("name")
            ]
        if not all_preset_loras_d and not loras_d_toggle:
            all_preset_loras_d = [
                {
                    "name": lora.get("name", ""),
                    "strength": float(lora.get("strength", lora.get("model_strength", 1.0))),
                    "clip_strength": float(lora.get("clip_strength", lora.get("strength", lora.get("model_strength", 1.0)))),
                    "active": lora.get("active", True),
                    "available": bool(lora.get("available", True)),
                }
                for lora in workflow_fields.get("loras_d", [])
                if isinstance(lora, dict) and lora.get("name")
            ]

        # When use_lora_input is disabled, ignore connected stacks and use only saved loras
        if use_input_only_loras:
            lora_stack_a = combined_input_multi.get('a') or []
            lora_stack_b = combined_input_multi.get('b') or []
            lora_stack_c = combined_input_multi.get('c') or []
            lora_stack_d = combined_input_multi.get('d') or []
        elif not use_combined_loras:
            lora_stack_a = preset_stack_a
            lora_stack_b = preset_stack_b
            lora_stack_c = preset_stack_c
            lora_stack_d = preset_stack_d
        else:
            # Merge: preset loras + connected loras (avoiding duplicates by lora name)
            # Handle case where connected stack might be None or empty
            input_stack_a = combined_input_multi.get('a')
            if input_stack_a:
                lora_stack_a = self._merge_lora_stacks(preset_stack_a, input_stack_a)
            else:
                lora_stack_a = preset_stack_a if preset_stack_a else []

            input_stack_b = combined_input_multi.get('b')
            if input_stack_b:
                lora_stack_b = self._merge_lora_stacks(preset_stack_b, input_stack_b)
            else:
                lora_stack_b = preset_stack_b if preset_stack_b else []

            input_stack_c = combined_input_multi.get('c')
            if input_stack_c:
                lora_stack_c = self._merge_lora_stacks(preset_stack_c, input_stack_c)
            else:
                lora_stack_c = preset_stack_c if preset_stack_c else []

            input_stack_d = combined_input_multi.get('d')
            if input_stack_d:
                lora_stack_d = self._merge_lora_stacks(preset_stack_d, input_stack_d)
            else:
                lora_stack_d = preset_stack_d if preset_stack_d else []

        # Process lora stacks with toggle data (filter inactive, apply strength)
        processed_stack_a = self._process_lora_toggle(lora_stack_a, None if use_input_only_loras else loras_a_toggle)
        processed_stack_b = self._process_lora_toggle(lora_stack_b, None if use_input_only_loras else loras_b_toggle)
        processed_stack_c = self._process_lora_toggle(lora_stack_c, None if use_input_only_loras else loras_c_toggle)
        processed_stack_d = self._process_lora_toggle(lora_stack_d, None if use_input_only_loras else loras_d_toggle)

        # Display logic: Include ALL loras (available + unavailable) so UI shows complete picture
        # Build display from available loras first, then add unavailable ones
        if use_input_only_loras:
            loras_a_display = self._format_loras_for_display(lora_stack_a)
            loras_b_display = self._format_loras_for_display(lora_stack_b)
            loras_c_display = self._format_loras_for_display(lora_stack_c)
            loras_d_display = self._format_loras_for_display(lora_stack_d)
        elif not use_combined_loras:
            loras_a_display = self._format_loras_for_display_with_unavailable(preset_stack_a, all_preset_loras_a)
            loras_b_display = self._format_loras_for_display_with_unavailable(preset_stack_b, all_preset_loras_b)
            loras_c_display = self._format_loras_for_display_with_unavailable(preset_stack_c, all_preset_loras_c)
            loras_d_display = self._format_loras_for_display_with_unavailable(preset_stack_d, all_preset_loras_d)
        else:
            loras_a_display = self._format_loras_for_display_with_unavailable(lora_stack_a, all_preset_loras_a)
            loras_b_display = self._format_loras_for_display_with_unavailable(lora_stack_b, all_preset_loras_b)
            loras_c_display = self._format_loras_for_display_with_unavailable(lora_stack_c, all_preset_loras_c)
            loras_d_display = self._format_loras_for_display_with_unavailable(lora_stack_d, all_preset_loras_d)

        # Convert thumbnail image to base64 if provided
        thumbnail_base64 = None
        if thumbnail_image is not None and IMAGE_SUPPORT:
            try:
                thumbnail_base64 = image_to_base64_thumbnail(thumbnail_image)
            except Exception as e:
                print(f"[PromptManagerAdvanced] Failed to convert thumbnail image: {e}")

        # Build explicit list of unavailable lora names for frontend
        if use_input_only_loras:
            unavailable_loras_a = [l.get('name') for l in loras_a_display if l.get('available') is False]
            unavailable_loras_b = [l.get('name') for l in loras_b_display if l.get('available') is False]
            unavailable_loras_c = [l.get('name') for l in loras_c_display if l.get('available') is False]
            unavailable_loras_d = [l.get('name') for l in loras_d_display if l.get('available') is False]
        else:
            unavailable_loras_a = [lora['name'] for lora in all_preset_loras_a if not lora.get('available', False)]
            unavailable_loras_b = [lora['name'] for lora in all_preset_loras_b if not lora.get('available', False)]
            unavailable_loras_c = [lora['name'] for lora in all_preset_loras_c if not lora.get('available', False)]
            unavailable_loras_d = [lora['name'] for lora in all_preset_loras_d if not lora.get('available', False)]

        # Broadcast update to frontend
        if unique_id is not None:
            server.PromptServer.instance.send_sync("prompt-manager-advanced-update", {
                "node_id": unique_id,
                "prompt": output_text,
                "use_prompt_input": use_prompt_input,
                "prompt_input": prompt,
                "loras_a": loras_a_display,
                "loras_b": loras_b_display,
                "loras_c": loras_c_display,
                "loras_d": loras_d_display,
                "input_loras_a": input_loras_a,  # Original input loras for change detection
                "input_loras_b": input_loras_b,  # Original input loras for change detection
                "input_loras_c": input_loras_c,
                "input_loras_d": input_loras_d,
                "unavailable_loras_a": unavailable_loras_a,  # Explicit list of unavailable lora names
                "unavailable_loras_b": unavailable_loras_b,  # Explicit list of unavailable lora names
                "unavailable_loras_c": unavailable_loras_c,
                "unavailable_loras_d": unavailable_loras_d,
                "trigger_words": trigger_words_display,
                "connected_thumbnail": thumbnail_base64,
                "should_reset": should_reset,  # Python tells JavaScript when to reset toggles
                "lora_input_mode": lora_input_mode,
                "use_lora_input": use_combined_loras,
                "use_input_only_loras": use_input_only_loras,
                "original_strengths_a": original_strengths_a,  # For "Reset Strength" button
                "original_strengths_b": original_strengths_b,  # For "Reset Strength" button
                "original_strengths_c": original_strengths_c,
                "original_strengths_d": original_strengths_d,
                "has_multi_lora_input": bool(combined_input_multi.get('c') or combined_input_multi.get('d')),
            })

        # Append active trigger words to output
        final_output = self._append_trigger_words_to_prompt(output_text, active_trigger_words)

        # Save resolved runtime prompt into execution metadata so saver nodes
        # embed the final prompt value, not the unresolved linked input state.
        _patch_runtime_prompt_metadata(
            unique_id=unique_id,
            output_text=final_output,
            extra_pnginfo=extra_pnginfo,
            api_prompt=api_prompt,
        )

        # Swap outputs if requested
        out_stack_a = processed_stack_a if processed_stack_a else []
        out_stack_b = processed_stack_b if processed_stack_b else []
        out_stack_c = processed_stack_c if processed_stack_c else []
        out_stack_d = processed_stack_d if processed_stack_d else []
        if swap_lora_outputs:
            out_stack_a, out_stack_b = out_stack_b, out_stack_a

        neg_prompt = str(workflow_fields.get('negative_prompt', '') or '')
        out_workflow_data = build_v2_recipe_data_from_prompt(
            prompt_text=final_output,
            negative_prompt=neg_prompt,
            loras_a=loras_a_display,
            loras_b=loras_b_display,
            loras_c=loras_c_display,
            loras_d=loras_d_display,
            source='PromptManagerAdvanced',
            base_recipe_data=resolved_workflow_data,
        )

        out_multi_stack = {
            "a": out_stack_a,
            "b": out_stack_b,
            "c": out_stack_c,
            "d": out_stack_d,
        }

        return (final_output, out_stack_a, out_stack_b, out_workflow_data, out_multi_stack)

    def check_lazy_status(self, category, name, use_prompt_input, use_lora_input=_LORA_INPUT_MODE_PROMPT_ONLY, text="", swap_lora_outputs=False,
                          prompt=None, lora_stack_a=None, lora_stack_b=None,
                          trigger_words=None, thumbnail_image=None,
                          unique_id=None, loras_a_toggle=None, loras_b_toggle=None, loras_c_toggle=None, loras_d_toggle=None,
                          trigger_words_toggle=None, extra_pnginfo=None, api_prompt=None,
                          **kwargs):
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
            })

        return all_loras

    def _format_loras_for_display_with_unavailable(self, lora_stack, all_preset_loras):
        """
        Format lora stack for frontend display, including unavailable preset loras.
        This ensures the UI shows all saved loras even if they're not found locally.
        """
        # Start with the available loras from the stack
        display_list = self._format_loras_for_display(lora_stack) if lora_stack else []
        preset_by_name = {
            str(item.get('name', '')).lower(): item
            for item in (all_preset_loras or [])
            if isinstance(item, dict) and item.get('name')
        }

        # Preserve active state from preset/toggle/workflow payload for rows
        # that are already present in the display list.
        for row in display_list:
            row_name_lower = str(row.get('name', '')).lower()
            preset = preset_by_name.get(row_name_lower)
            if preset is not None:
                row['active'] = preset.get('active', True)

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
        loras_c = data.get("loras_c", [])
        loras_d = data.get("loras_d", [])
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

        # Normalize lora data - keep path when present for lossless save/restore.
        def normalize_lora_data(loras):
            normalized = []
            for lora in loras:
                if isinstance(lora, dict) and lora.get('name'):
                    available = lora.get('available', lora.get('available', True))
                    normalized.append({
                        "name": lora.get('name'),
                        "path": lora.get('path', lora.get('name')),
                        "strength": lora.get('strength', lora.get('model_strength', 1.0)),
                        "clip_strength": lora.get('clip_strength', lora.get('strength', 1.0)),
                        "active": lora.get('active', True),
                        "available": bool(available),
                    })
            return normalized

        def normalize_workflow_lora_data(wf_loras):
            """Normalize LoRAs coming from workflow_data payload structure."""
            normalized = []
            if not isinstance(wf_loras, list):
                return normalized
            for lora in wf_loras:
                if not isinstance(lora, dict):
                    continue
                name = str(lora.get('name', '')).strip()
                if not name:
                    continue
                strength = lora.get('strength', lora.get('model_strength', 1.0))
                clip_strength = lora.get('clip_strength', strength)
                available = lora.get('available', lora.get('available', True))
                normalized.append({
                    "name": name,
                    "path": lora.get('path', name),
                    "strength": strength,
                    "clip_strength": clip_strength,
                    "active": lora.get('active', True),
                    "available": bool(available),
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

        # Save workflow_data if provided.
        workflow_data = data.get("workflow_data")
        wf_to_save = None
        if isinstance(workflow_data, dict):
            wf_to_save = to_json_safe_workflow_data(ensure_v2_recipe_data(workflow_data, source="PromptManagerAdvanced"))
        elif isinstance(workflow_data, str) and workflow_data.strip():
            try:
                parsed_wf = json.loads(workflow_data)
                if isinstance(parsed_wf, dict):
                    wf_to_save = to_json_safe_workflow_data(ensure_v2_recipe_data(parsed_wf, source="PromptManagerAdvanced"))
            except (json.JSONDecodeError, TypeError):
                wf_to_save = None

        # Save prompt with normalized lora data (no paths - they're resolved at runtime)
        normalized_loras_a = normalize_lora_data(loras_a)
        normalized_loras_b = normalize_lora_data(loras_b)
        normalized_loras_c = normalize_lora_data(loras_c)
        normalized_loras_d = normalize_lora_data(loras_d)

        workflow_has_meaningful_data = _has_meaningful_workflow_data(wf_to_save)

        # Workflow Saver parity: when meaningful workflow_data is present, treat
        # its LoRA lists as authoritative for persistence.
        wf_models = (wf_to_save or {}).get("models", {}) if isinstance((wf_to_save or {}).get("models"), dict) else {}
        wf_loras_a = normalize_workflow_lora_data((wf_models.get("model_a") or {}).get("loras", []))
        wf_loras_b = normalize_workflow_lora_data((wf_models.get("model_b") or {}).get("loras", []))
        wf_loras_c = normalize_workflow_lora_data((wf_models.get("model_c") or {}).get("loras", []))
        wf_loras_d = normalize_workflow_lora_data((wf_models.get("model_d") or {}).get("loras", []))
        if isinstance(wf_to_save, dict) and workflow_has_meaningful_data:
            if isinstance(wf_models.get("model_a"), dict) and wf_loras_a:
                normalized_loras_a = wf_loras_a
            elif not normalized_loras_a and wf_loras_a:
                normalized_loras_a = wf_loras_a

            if isinstance(wf_models.get("model_b"), dict) and wf_loras_b:
                normalized_loras_b = wf_loras_b
            elif not normalized_loras_b and wf_loras_b:
                normalized_loras_b = wf_loras_b

            if isinstance(wf_models.get("model_c"), dict) and wf_loras_c:
                normalized_loras_c = wf_loras_c
            elif not normalized_loras_c and wf_loras_c:
                normalized_loras_c = wf_loras_c

            if isinstance(wf_models.get("model_d"), dict) and wf_loras_d:
                normalized_loras_d = wf_loras_d
            elif not normalized_loras_d and wf_loras_d:
                normalized_loras_d = wf_loras_d

        # Guard against accidental empty-stack overwrites from transient UI state.
        # If the incoming payload has no LoRAs for a side, preserve existing saved
        # LoRAs for that side instead of silently erasing them.
        if not normalized_loras_a and isinstance(existing_prompt.get("loras_a"), list):
            normalized_loras_a = existing_prompt.get("loras_a", [])
        if not normalized_loras_b and isinstance(existing_prompt.get("loras_b"), list):
            normalized_loras_b = existing_prompt.get("loras_b", [])
        if not normalized_loras_c and isinstance(existing_prompt.get("loras_c"), list):
            normalized_loras_c = existing_prompt.get("loras_c", [])
        if not normalized_loras_d and isinstance(existing_prompt.get("loras_d"), list):
            normalized_loras_d = existing_prompt.get("loras_d", [])

        prompt_data = {
            "prompt": text,
            "loras_a": normalized_loras_a,
            "loras_b": normalized_loras_b,
            "loras_c": normalized_loras_c,
            "loras_d": normalized_loras_d,
            "trigger_words": normalize_trigger_words(trigger_words)
        }

        # Persist prompt fields from workflow_data snapshot when available.
        if isinstance(wf_to_save, dict):
            model_a_block = wf_models.get("model_a") if isinstance(wf_models.get("model_a"), dict) else {}
            if isinstance(model_a_block.get("positive_prompt"), str) and model_a_block.get("positive_prompt", "").strip():
                prompt_data["prompt"] = model_a_block.get("positive_prompt", "")
            if isinstance(model_a_block.get("negative_prompt"), str):
                prompt_data["negative_prompt"] = model_a_block.get("negative_prompt", "")
            prompt_data["saved_from"] = "RecipeManager"
            prompt_data["saved_at"] = datetime.utcnow().isoformat() + "Z"

        if wf_to_save:
            prompt_data["workflow_data"] = wf_to_save
        else:
            # Preserve previously saved workflow_data when user saves prompt/LoRA
            # edits without a currently connected workflow_data source.
            existing_wf = existing_prompt.get("workflow_data")
            if isinstance(existing_wf, dict):
                prompt_data["workflow_data"] = to_json_safe_workflow_data(ensure_v2_recipe_data(existing_wf, source="PromptManagerAdvanced"))

        if thumbnail:
            prompt_data["thumbnail"] = thumbnail

        # Preserve or set NSFW flag
        nsfw = data.get("nsfw")
        if nsfw is not None:
            prompt_data["nsfw"] = bool(nsfw)
        elif existing_prompt.get("nsfw"):
            prompt_data["nsfw"] = existing_prompt["nsfw"]

        # Preserve any extra fields from an existing prompt that this node does not manage.
        # This includes workflow_config (added by RecipeManager) and any future extensions.
        # Known managed keys — everything else is preserved verbatim.
        _MANAGED_KEYS = {"prompt", "negative_prompt", "loras_a", "loras_b", "loras_c", "loras_d", "trigger_words", "thumbnail", "nsfw", "workflow_data", "saved_from", "saved_at"}
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

        # Compatibility: if prompt/loras are missing at top-level, derive them
        # from v2 workflow_data model_a/model_b blocks.
        prompt_response = dict(prompt_data)
        workflow_data = prompt_response.get("workflow_data")
        derived = _derive_prompt_fields_from_workflow_data(workflow_data)

        if not str(prompt_response.get("prompt", "")).strip() and derived.get("prompt"):
            prompt_response["prompt"] = derived.get("prompt", "")

        if not isinstance(prompt_response.get("loras_a"), list) or len(prompt_response.get("loras_a") or []) == 0:
            prompt_response["loras_a"] = derived.get("loras_a", [])

        if not isinstance(prompt_response.get("loras_b"), list) or len(prompt_response.get("loras_b") or []) == 0:
            prompt_response["loras_b"] = derived.get("loras_b", [])
        if not isinstance(prompt_response.get("loras_c"), list) or len(prompt_response.get("loras_c") or []) == 0:
            prompt_response["loras_c"] = derived.get("loras_c", [])
        if not isinstance(prompt_response.get("loras_d"), list) or len(prompt_response.get("loras_d") or []) == 0:
            prompt_response["loras_d"] = derived.get("loras_d", [])

        # Add availability info to loras
        def add_availability(loras):
            result = []
            for lora in loras:
                lora_copy = dict(lora)
                _, available = get_lora_relative_path(lora.get('name', ''))
                lora_copy['available'] = available
                result.append(lora_copy)
            return result

        return server.web.json_response({
            "success": True,
            "data": {
                "prompt": prompt_response.get("prompt", ""),
                "loras_a": add_availability(prompt_response.get("loras_a", [])),
                "loras_b": add_availability(prompt_response.get("loras_b", [])),
                "loras_c": add_availability(prompt_response.get("loras_c", [])),
                "loras_d": add_availability(prompt_response.get("loras_d", [])),
                "trigger_words": prompt_response.get("trigger_words", []),
                "workflow_data": prompt_response.get("workflow_data")
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
                    "loras_c": prompt_data.get("loras_c", []),
                    "loras_d": prompt_data.get("loras_d", []),
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
