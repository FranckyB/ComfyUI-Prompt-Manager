"""
ComfyUI Prompt Manager Advanced - Extended prompt management with LoRA stack support
Saves prompts along with associated LoRA configurations
"""
import os
import json
from datetime import datetime
import server
from ..py.prompt_metadata import record_prompt
from ..py.prompt_store import is_hidden_key, load_prompts, save_prompts
from ..py.workflow_data_utils import ensure_v2_recipe_data, to_json_safe_workflow_data, build_v2_recipe_data_from_prompt


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
_PROMPT_COMPOSER_RECIPE_KEY = "prompt_composer"
# Marks manager_data produced by Prompt Manager, next to "prompt_composer" and "prompt_layout".
_PROMPT_MANAGER_RECIPE_KEY = "prompt_manager"
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


def _flatten_multi_lora_slots(multi_slots):
    flat = []
    if not isinstance(multi_slots, dict):
        return flat
    for slot in _MODEL_SLOT_KEYS:
        flat.extend(_coerce_lora_stack(multi_slots.get(slot)))
    return flat


def _normalize_prompt_composer_input_loras(raw_loras):
    normalized = []
    for path, model_strength, clip_strength in _coerce_lora_stack(raw_loras):
        normalized.append({
            "name": str(os.path.splitext(os.path.basename(normalize_path_separators(path)))[0] or "").strip(),
            "path": str(path or "").strip(),
            "model_strength": float(model_strength),
            "clip_strength": float(clip_strength),
            "active": True,
            "available": True,
            "source": "upstream",
        })
    return [entry for entry in normalized if entry.get("name")]


def _prompt_composer_lora_key(item):
    raw_path = str(item.get("path", item.get("name", "")) or "").replace("\\", "/").strip().lower()
    leaf = os.path.basename(raw_path)
    stem, _ext = os.path.splitext(leaf)
    return stem or leaf


def _merge_prompt_composer_input_data(recipe_data, input_prompt="", prompt_enabled=False, input_lora_stack=None):
    if not isinstance(recipe_data, dict):
        return recipe_data

    payload = recipe_data.get(_PROMPT_COMPOSER_RECIPE_KEY)
    if not isinstance(payload, dict):
        return recipe_data

    input_data = payload.get("input_data") if isinstance(payload.get("input_data"), dict) else {}
    prompt_data = payload.get("prompt_data") if isinstance(payload.get("prompt_data"), dict) else {}
    existing_prompt = str(input_data.get("prompt", payload.get("input_prompt", "")) or "").strip()
    incoming_prompt = str(input_prompt or "").strip()
    merged_prompt = existing_prompt if existing_prompt else (incoming_prompt if prompt_enabled and incoming_prompt else "")

    existing_loras = _normalize_prompt_composer_input_loras(
        input_data.get("lora_stack", payload.get("input_lora_stack", []))
    )
    added_loras = _normalize_prompt_composer_input_loras(input_lora_stack or []) if input_lora_stack is not None else []
    prompt_loras = _normalize_prompt_composer_input_loras(
        prompt_data.get("lora_stack", payload.get("prompt_lora_stack", []))
    )
    prompt_lora_keys = {
        _prompt_composer_lora_key(item)
        for item in prompt_loras
        if _prompt_composer_lora_key(item)
    }

    merged_loras = []
    seen = set()
    for item in [*existing_loras, *added_loras]:
        key = _prompt_composer_lora_key(item)
        if not key:
            continue
        if key in prompt_lora_keys:
            continue
        if key in seen:
            continue
        seen.add(key)
        merged_loras.append(item)

    payload["input_data"] = {
        **input_data,
        "source": "upstream",
        "prompt": merged_prompt,
        "lora_stack": merged_loras,
    }
    recipe_data[_PROMPT_COMPOSER_RECIPE_KEY] = payload
    return recipe_data


class PromptManagerAdvanced:
    """
    Advanced Prompt Manager with LoRA stack integration.
    Supports saving/loading prompts with associated LoRA configurations.
    Features two LoRA stack inputs/outputs for dual LoRA workflows (e.g., Wan video).
    """

    def __init__(self):
        # Per node: the last run's prompt, input LoRA paths and input prompt, to tell when the frontend resets.
        self._last_known_state = {}

    @classmethod
    def INPUT_TYPES(s):
        prompts_data = load_prompts()
        categories = list(prompts_data.keys()) if prompts_data else ["Default"]

        all_prompts = set()
        for category_prompts in prompts_data.values():
            all_prompts.update(k for k in category_prompts.keys() if not is_hidden_key(k))

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
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "forceInput": True, "lazy": True, "tooltip": "Connect prompt text input here"}),
                "manager_data": ("MANAGER_DATA", {"forceInput": True, "tooltip": "Optional saved recipe or composer payload."}),
                "lora_stack_a": ("LORA_STACK,MULTI_LORA_STACK", {"forceInput": True, "tooltip": "First LoRA stack input. Accepts LORA_STACK or MULTI_LORA_STACK."}),
                "lora_stack_b": ("LORA_STACK,MULTI_LORA_STACK", {"forceInput": True, "tooltip": "Second LoRA stack input. Accepts LORA_STACK or MULTI_LORA_STACK."}),
                "trigger_words": ("STRING", {"forceInput": True, "tooltip": "Comma-separated trigger words to append to prompt"}),
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
    RETURN_TYPES = ("STRING", "LORA_STACK", "LORA_STACK", "MULTI_LORA_STACK", "MANAGER_DATA")
    RETURN_NAMES = ("prompt", "lora_stack_a", "lora_stack_b", "multi_lora_stack", "manager_data")
    OUTPUT_TOOLTIPS = ("", "", "", "", "Prompt, LoRAs and saved recipe data; connect to Library Manager to save it.")
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
    def IS_CHANGED(cls, category, name, use_prompt_input, use_lora_input=_LORA_INPUT_MODE_PROMPT_ONLY, text="", **kwargs):
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

    def get_prompt(self, category, name, use_prompt_input, use_lora_input=_LORA_INPUT_MODE_PROMPT_ONLY,
                   text="", prompt=None, manager_data=None, lora_stack_a=None, lora_stack_b=None,
                   trigger_words=None,
                   unique_id=None, loras_a_toggle=None, loras_b_toggle=None, loras_c_toggle=None, loras_d_toggle=None, trigger_words_toggle=None,
                   extra_pnginfo=None, api_prompt=None):
        """Return the prompt text and filtered lora stacks based on toggle states"""
        toggles = {"a": loras_a_toggle, "b": loras_b_toggle, "c": loras_c_toggle, "d": loras_d_toggle}
        inputs = _combine_multi_lora_inputs(lora_stack_a, lora_stack_b)
        input_paths = {slot: [path for path, _, _ in inputs.get(slot, [])] for slot in _MODEL_SLOT_KEYS}
        prompt_input_text = prompt if prompt else ""

        # Tell the frontend to reset its toggles when the prompt changed, when the input LoRAs and the input prompt
        # changed together (a new extraction), or when the input LoRAs were cleared; not when only toggles changed.
        node_key = str(unique_id) if unique_id else "default"
        last = self._last_known_state.get(node_key, {"prompt_key": "", "input_paths": {slot: [] for slot in _MODEL_SLOT_KEYS}, "prompt_input_text": ""})
        prompt_key = f"{category}|{name}"
        input_loras_changed = input_paths != last["input_paths"]
        loras_were_cleared = any(last["input_paths"].values()) and not any(input_paths.values())
        should_reset = (prompt_key != last["prompt_key"]
                        or (input_loras_changed and prompt_input_text != last["prompt_input_text"])
                        or loras_were_cleared)
        self._last_known_state[node_key] = {"prompt_key": prompt_key, "input_paths": input_paths, "prompt_input_text": prompt_input_text}

        lora_input_mode = self._normalize_lora_input_mode(use_lora_input)
        use_combined_loras = self._should_use_combined_loras(lora_input_mode)
        use_input_only_loras = self._should_use_input_only_loras(lora_input_mode)

        prompt_entry = load_prompts().get(category, {}).get(name, {})
        if not isinstance(prompt_entry, dict):
            prompt_entry = {}
        stored_prompt_wf = prompt_entry.get("workflow_data")
        connected_recipe_data = ensure_v2_recipe_data(manager_data, source="PromptManagerAdvanced") if isinstance(manager_data, dict) else None
        resolved_workflow_data = connected_recipe_data or (ensure_v2_recipe_data(stored_prompt_wf, source="PromptManagerAdvanced") if isinstance(stored_prompt_wf, dict) else None)
        workflow_fields = _derive_prompt_fields_from_workflow_data(resolved_workflow_data)

        # The prompt input when it is on and has text, else the typed text, else the recipe's prompt.
        if use_prompt_input and prompt:
            output_text = prompt
        else:
            output_text = text if text else ""
            if not output_text.strip() and workflow_fields.get("prompt"):
                output_text = workflow_fields.get("prompt", "")

        trigger_words_display = self._process_trigger_words(trigger_words, trigger_words_toggle)
        active_trigger_words = self._get_active_trigger_words(trigger_words_toggle, trigger_words)

        update = {}
        out_stacks = {}
        for slot in _MODEL_SLOT_KEYS:
            toggle = toggles[slot]
            connected = inputs.get(slot)
            # The connected LoRAs as they came in, for the frontend to notice an input change.
            input_display = self._format_loras_for_display(connected) if connected else []

            # Strengths before any adjustment, for "Reset Strength": the saved prompt's, then the inputs'.
            original_strengths = {}
            for lora in prompt_entry.get(f"loras_{slot}", []) if category and name else []:
                if lora.get("name"):
                    original_strengths[lora["name"]] = lora.get("strength", lora.get("model_strength", 1.0))
            for lora in input_display:
                if lora.get("name") and lora["name"] not in original_strengths:
                    original_strengths[lora["name"]] = lora.get("strength", lora.get("model_strength", 1.0))

            # The prompt's own LoRAs: the saved toggle payload, else the recipe's LoRAs.
            recipe_loras = [lora for lora in workflow_fields.get(f"loras_{slot}", []) if isinstance(lora, dict) and lora.get("name")]
            preset_stack = self._build_stack_from_toggle(toggle) if toggle else []
            all_preset_loras = self._get_all_loras_from_toggle(toggle) if toggle else []
            if not toggle:
                preset_stack = [
                    (lora.get("name", ""), lora.get("strength", lora.get("model_strength", 1.0)),
                     lora.get("clip_strength", lora.get("strength", lora.get("model_strength", 1.0))))
                    for lora in recipe_loras
                ]
                all_preset_loras = [
                    {
                        "name": lora.get("name", ""),
                        "strength": float(lora.get("strength", lora.get("model_strength", 1.0))),
                        "clip_strength": float(lora.get("clip_strength", lora.get("strength", lora.get("model_strength", 1.0)))),
                        "active": lora.get("active", True),
                        "available": bool(lora.get("available", True)),
                    }
                    for lora in recipe_loras
                ]

            if use_input_only_loras:
                stack = connected or []
            elif use_combined_loras and connected:
                stack = self._merge_lora_stacks(preset_stack, connected)
            else:
                stack = preset_stack

            out_stacks[slot] = self._process_lora_toggle(stack, None if use_input_only_loras else toggle) or []
            if use_input_only_loras:
                display = self._format_loras_for_display(stack)
                unavailable = [lora.get("name") for lora in display if lora.get("available") is False]
            else:
                display = self._format_loras_for_display_with_unavailable(stack, all_preset_loras)
                unavailable = [lora["name"] for lora in all_preset_loras if not lora.get("available", False)]

            update[f"loras_{slot}"] = display
            update[f"input_loras_{slot}"] = input_display
            update[f"unavailable_loras_{slot}"] = unavailable
            update[f"original_strengths_{slot}"] = original_strengths

        if unique_id is not None:
            server.PromptServer.instance.send_sync("prompt-manager-advanced-update", {
                "node_id": unique_id,
                "prompt": output_text,
                "use_prompt_input": use_prompt_input,
                "prompt_input": prompt,
                **update,
                "trigger_words": trigger_words_display,
                "connected_thumbnail": None,
                "should_reset": should_reset,
                "lora_input_mode": lora_input_mode,
                "use_lora_input": use_combined_loras,
                "use_input_only_loras": use_input_only_loras,
                "has_multi_lora_input": bool(inputs.get("c") or inputs.get("d")),
            })

        final_output = self._append_trigger_words_to_prompt(output_text, active_trigger_words)

        # Saver nodes embed the final prompt as typed text, not the link that fed it.
        record_prompt(unique_id, final_output, extra_pnginfo, api_prompt, toggle="use_prompt_input", link_input="prompt", text_index=4)

        out_workflow_data = build_v2_recipe_data_from_prompt(
            prompt_text=final_output,
            negative_prompt=str(workflow_fields.get("negative_prompt", "") or ""),
            **{f"loras_{slot}": update[f"loras_{slot}"] for slot in _MODEL_SLOT_KEYS},
            source="PromptManagerAdvanced",
            base_recipe_data=resolved_workflow_data,
        )
        out_workflow_data = _merge_prompt_composer_input_data(
            out_workflow_data,
            input_prompt=prompt_input_text,
            prompt_enabled=bool(use_prompt_input and prompt_input_text),
            input_lora_stack=_flatten_multi_lora_slots(out_stacks),
        )
        out_workflow_data[_PROMPT_MANAGER_RECIPE_KEY] = {"category": category, "name": name}

        return (final_output, out_stacks["a"], out_stacks["b"], out_stacks, out_workflow_data)

    def check_lazy_status(self, category, name, use_prompt_input, use_lora_input=_LORA_INPUT_MODE_PROMPT_ONLY,
                          text="", prompt=None, lora_stack_a=None, lora_stack_b=None,
                          trigger_words=None,
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
        Connected-origin words are pruned when missing, while prompt-saved words remain.
        """
        connected_words = []
        connected_word_set = set()
        if connected_trigger_words and isinstance(connected_trigger_words, str):
            connected_words = [w.strip() for w in connected_trigger_words.split(',,') if w.strip()]
            connected_word_set = {w.lower() for w in connected_words}

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
                            word_lower = word.lower()
                            from_input = item.get('fromInput') is True
                            if from_input and word_lower not in connected_word_set:
                                continue
                            saved_words[word.lower()] = {
                                'text': word,
                                'active': item.get('active', True),
                                'fromInput': from_input,
                            }
            except (json.JSONDecodeError, TypeError):
                pass

        # Merge: saved words + new connected words (no duplicates)
        result = []
        seen = set()

        # First, add all saved words (preserving their state)
        for word_lower, data in saved_words.items():
            if word_lower not in seen:
                result.append({
                    'text': data['text'],
                    'active': data['active'],
                    'source': 'connected' if data.get('fromInput') else 'saved',
                    'fromInput': data.get('fromInput', False),
                })
                seen.add(word_lower)

        # Then add connected words that aren't already saved
        for word in connected_words:
            word_lower = word.lower()
            if word_lower not in seen:
                result.append({
                    'text': word,
                    'active': True,
                    'source': 'connected',
                    'fromInput': True,
                })
                seen.add(word_lower)

        return result

    def _get_active_trigger_words(self, toggle_data, connected_trigger_words=None):
        """
        Get list of active trigger word strings from toggle data and connected input.
        Connected-origin words are pruned when missing, while prompt-saved words remain.
        """
        active_words = []
        active_seen = set()
        saved_seen = set()
        has_authoritative_toggle_data = False
        connected_word_set = set()

        if connected_trigger_words and isinstance(connected_trigger_words, str):
            connected_word_set = {
                word.strip().lower()
                for word in connected_trigger_words.split(',,')
                if word.strip()
            }

        # First, get active words from saved toggle data
        if toggle_data is not None:
            try:
                if isinstance(toggle_data, str):
                    toggle_data = json.loads(toggle_data)
                if isinstance(toggle_data, list):
                    has_authoritative_toggle_data = True
                    for item in toggle_data:
                        if not isinstance(item, dict):
                            continue
                        word = item.get('text', '').strip()
                        if not word:
                            continue

                        word_lower = word.lower()
                        if item.get('fromInput') is True and word_lower not in connected_word_set:
                            continue
                        saved_seen.add(word_lower)

                        if item.get('active', True) and word_lower not in active_seen:
                            active_words.append(word)
                            active_seen.add(word_lower)
            except (json.JSONDecodeError, TypeError):
                pass

        # Also include connected trigger words that aren't already in the saved data
        # but only when no authoritative toggle payload exists yet.
        if (not has_authoritative_toggle_data) and connected_trigger_words and isinstance(connected_trigger_words, str):
            for word in connected_trigger_words.split(',,'):
                word = word.strip()
                word_lower = word.lower()
                if word and word_lower not in saved_seen and word_lower not in active_seen:
                    active_words.append(word)
                    active_seen.add(word_lower)

        return active_words

    def _append_trigger_words_to_prompt(self, prompt, trigger_words):
        """
        Prepend trigger words before the prompt.
        Ensures the trigger-word block ends with punctuation; if it doesn't end with
        period/comma/colon, append a comma.
        """
        if not trigger_words:
            return prompt

        trigger_block = ', '.join(trigger_words).strip()
        if not trigger_block:
            return prompt

        if not trigger_block.endswith(('.', ',', ':')):
            trigger_block += ','

        prompt = str(prompt or '').strip()
        if not prompt:
            return trigger_block

        return f"{trigger_block} {prompt}"


# API Routes for Advanced Prompt Manager

@server.PromptServer.instance.routes.post("/prompt-manager-advanced/save-prompt")
async def save_prompt_advanced(request):
    """API endpoint to save a prompt with lora configurations"""
    try:
        data = await request.json()
        category = data.get("category", "").strip()
        name = data.get("name", "").strip()
        text = data.get("text", "").strip()
        old_category = data.get("old_category", "").strip() or category
        old_name = data.get("old_name", "").strip() or name
        loras_a = data.get("loras_a", [])
        loras_b = data.get("loras_b", [])
        loras_c = data.get("loras_c", [])
        loras_d = data.get("loras_d", [])
        trigger_words = data.get("trigger_words", [])

        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name are required"})

        prompts = load_prompts()

        if category not in prompts:
            prompts[category] = {}

        source_prompts = prompts.get(old_category, {}) if isinstance(prompts.get(old_category), dict) else {}
        target_prompts = prompts[category]

        existing_old_name = None
        existing_prompt = {}
        if old_category in prompts:
            existing_old_name = next((key for key in source_prompts.keys() if str(key).lower() == old_name.lower()), None)
            if existing_old_name:
                existing_prompt = source_prompts.get(existing_old_name, {})

        existing_target_name = next((key for key in target_prompts.keys() if str(key).lower() == name.lower() and not is_hidden_key(key)), None)
        if existing_target_name:
            same_entry = (
                existing_old_name is not None
                and old_category == category
                and str(existing_target_name).lower() == str(existing_old_name).lower()
            )
            if not same_entry:
                return server.web.json_response({
                    "success": False,
                    "error": f"A prompt named '{existing_target_name}' already exists in '{category}'"
                })

        if existing_old_name and (old_category != category or existing_old_name != name):
            existing_prompt = source_prompts.pop(existing_old_name, existing_prompt)

        # Case-insensitive check for existing prompt
        # Get existing prompt data BEFORE potentially deleting it (to preserve thumbnail)
        existing_prompts_lower = {
            k.lower(): k
            for k in prompts[category].keys()
            if not is_hidden_key(k)
        }
        if name.lower() in existing_prompts_lower:
            casing_name = existing_prompts_lower[name.lower()]
            existing_prompt = prompts[category].get(casing_name, existing_prompt)
            if casing_name != name:
                # Delete the old casing version
                print(f"[PromptManagerAdvanced] Removing old casing '{casing_name}' before saving as '{name}'")
                del prompts[category][casing_name]

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
            prompt_data["saved_from"] = str(wf_to_save.get("_source") or "RecipeManager")
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
        save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptManagerAdvanced] Error in save_prompt API: {e}")
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
