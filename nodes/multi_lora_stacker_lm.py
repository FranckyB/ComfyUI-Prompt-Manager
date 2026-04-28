"""Multi-slot LoRA stacker outputting four independent LORA_STACK outputs (A/B/C/D)."""

import json
import os
import sys

from ..py.lora_utils import resolve_lora_path


def _try_get_lm_get_lora_info():
    """Return LM's get_lora_info function if it is already loaded in sys.modules."""
    for mod in sys.modules.values():
        fn = getattr(mod, "get_lora_info", None)
        if callable(fn):
            module_name = getattr(fn, "__module__", "") or ""
            if "lora_manager" in module_name.lower():
                return fn
    return None


def _build_lora_stack(loras_json):
    """
    Build a LORA_STACK list from the JSON state persisted by the JS widget.

    Each entry: {"name": str, "strength": float, "clipStrength": float, "active": bool}
    Returns a list of (lora_path, model_strength, clip_strength) tuples.
    """
    if isinstance(loras_json, str):
        raw = loras_json.strip()
        if not raw or raw == "[]":
            return []
        try:
            loras = json.loads(raw)
        except Exception:
            return []
    elif isinstance(loras_json, dict) and "__value__" in loras_json:
        loras = loras_json["__value__"]
    elif isinstance(loras_json, list):
        loras = loras_json
    else:
        return []

    if not isinstance(loras, list):
        return []

    get_lora_info = _try_get_lm_get_lora_info()
    stack = []

    for lora in loras:
        if not isinstance(lora, dict):
            continue
        if not lora.get("active", True):
            continue

        name = str(lora.get("name", "")).strip()
        if not name:
            continue

        try:
            strength = float(lora.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0

        try:
            clip_strength = float(lora.get("clipStrength", strength))
        except (TypeError, ValueError):
            clip_strength = strength

        # Prefer LM's path resolver (has full cache); fall back to folder_paths scan.
        lora_path = None
        if get_lora_info is not None:
            try:
                result = get_lora_info(name)
                if isinstance(result, tuple) and result[0]:
                    lora_path = result[0]
            except Exception:
                pass

        if not lora_path:
            resolved, available = resolve_lora_path(name)
            lora_path = resolved if (available and resolved) else name

        stack.append((lora_path.replace("/", os.sep), strength, clip_strength))

    return stack


class MultiLoraStackerLM:
    """Multi-slot LoRA stacker with four independent stacks (A / B / C / D)."""

    NAME = "Multi Lora Stacker (LoraManager)"
    CATEGORY = "Prompt Manager"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # JSON arrays persisted by the JS loras widget – one per slot.
                "loras_state_a": ("STRING", {"default": "[]", "multiline": True}),
                "loras_state_b": ("STRING", {"default": "[]", "multiline": True}),
                "loras_state_c": ("STRING", {"default": "[]", "multiline": True}),
                "loras_state_d": ("STRING", {"default": "[]", "multiline": True}),
            },
        }

    RETURN_TYPES = ("LORA_STACK", "LORA_STACK", "LORA_STACK", "LORA_STACK")
    RETURN_NAMES = ("lora_stack_a", "lora_stack_b", "lora_stack_c", "lora_stack_d")
    FUNCTION = "stack_multi"
    DESCRIPTION = (
        "Multi-slot LoRA stacker with a visual 4-panel UI (A / B / C / D). "
        "Each panel outputs a standard LORA_STACK compatible with Lora-Manager's loaders."
    )

    def stack_multi(
        self,
        loras_state_a="[]",
        loras_state_b="[]",
        loras_state_c="[]",
        loras_state_d="[]",
        **kwargs,
    ):
        stack_a = _build_lora_stack(loras_state_a)
        stack_b = _build_lora_stack(loras_state_b)
        stack_c = _build_lora_stack(loras_state_c)
        stack_d = _build_lora_stack(loras_state_d)
        return (stack_a, stack_b, stack_c, stack_d)
