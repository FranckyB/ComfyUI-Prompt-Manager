"""LoRA stacks and recipe data as the Prompt Manager nodes read them: four model slots A-D, connected stacks
normalized into them, and the prompt and LoRAs a v2 recipe (manager_data) carries."""

import os

from .lora_utils import normalize_path_separators
from .workflow_data_utils import ensure_v2_recipe_data


def has_meaningful_workflow_data(workflow_data):
    """Return True when v2 recipe data contains authored prompt/model/lora content."""
    for block in workflow_data.get("models", {}).values():
        if not isinstance(block, dict):
            continue
        loras = block.get("loras")
        has_loras = isinstance(loras, list) and any(isinstance(lora, dict) and str(lora.get("name", "")).strip() for lora in loras)
        if str(block.get("positive_prompt", "")).strip() or str(block.get("model", "")).strip() or has_loras:
            return True
    return False


def normalize_workflow_loras(loras):
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


def derive_prompt_fields(workflow_data):
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
        "loras_a": normalize_workflow_loras(model_a_block.get("loras", [])),
        "loras_b": normalize_workflow_loras(model_b_block.get("loras", [])),
        "loras_c": normalize_workflow_loras(model_c_block.get("loras", [])),
        "loras_d": normalize_workflow_loras(model_d_block.get("loras", [])),
        "model_a": str(model_a_block.get("model", "") or ""),
        "model_b": str(model_b_block.get("model", "") or ""),
        "model_c": str(model_c_block.get("model", "") or ""),
        "model_d": str(model_d_block.get("model", "") or ""),
    }


MODEL_SLOT_KEYS = ("a", "b", "c", "d")
LORA_MODE_PROMPT_ONLY = "Prompt LoRAs Only"
LORA_MODE_COMBINE = "Combine LoRAs"
LORA_MODE_INPUT_ONLY = "Input LoRAs Only"
_PROMPT_COMPOSER_RECIPE_KEY = "prompt_composer"
# Marks manager_data produced by Prompt Manager, next to "prompt_composer" and "prompt_layout".
PROMPT_MANAGER_RECIPE_KEY = "prompt_manager"
LORA_MODES = (
    LORA_MODE_PROMPT_ONLY,
    LORA_MODE_COMBINE,
    LORA_MODE_INPUT_ONLY,
)


def coerce_lora_stack(raw_stack):
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


def normalize_lora_slots(raw_input, primary_slot):
    """Normalize one PMA input socket into 4 slots (A/B/C/D)."""
    slots = {k: [] for k in MODEL_SLOT_KEYS}

    plain = coerce_lora_stack(raw_input)
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
        slots[suffix] = coerce_lora_stack(candidate)

    return slots


def combine_lora_inputs(input_a, input_b):
    """Combine PMA A/B sockets into a single multi-slot payload."""
    normalized_a = normalize_lora_slots(input_a, "a")
    normalized_b = normalize_lora_slots(input_b, "b")

    out = {}
    for slot in MODEL_SLOT_KEYS:
        out[slot] = [*(normalized_a.get(slot) or []), *(normalized_b.get(slot) or [])]
    return out


def flatten_lora_slots(multi_slots):
    flat = []
    if not isinstance(multi_slots, dict):
        return flat
    for slot in MODEL_SLOT_KEYS:
        flat.extend(coerce_lora_stack(multi_slots.get(slot)))
    return flat


def _normalize_prompt_composer_input_loras(raw_loras):
    normalized = []
    for path, model_strength, clip_strength in coerce_lora_stack(raw_loras):
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


def merge_prompt_composer_input_data(recipe_data, input_prompt="", prompt_enabled=False, input_lora_stack=None):
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
