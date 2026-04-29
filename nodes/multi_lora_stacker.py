"""Multi-slot LoRA stacker outputting four independent LORA_STACK outputs (A/B/C/D)."""

import json
import os
import sys

from ..py.lora_utils import resolve_lora_path
try:
    from ..py.lora_utils import get_lora_relative_path
except Exception:
    def get_lora_relative_path(lora_name):
        return str(lora_name or ""), False


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

    def _normalize_to_relative_lora(raw_value, fallback_name):
        candidate = str(raw_value or "").strip()
        if candidate:
            rel_path, available = get_lora_relative_path(candidate)
            if available and rel_path:
                return rel_path

            candidate_base = os.path.basename(candidate.replace("\\", "/")).strip()
            if candidate_base:
                rel_path, available = get_lora_relative_path(candidate_base)
                if available and rel_path:
                    return rel_path

        fallback = str(fallback_name or "").strip()
        if fallback:
            rel_path, available = get_lora_relative_path(fallback)
            if available and rel_path:
                return rel_path
        return fallback

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
                    lora_path = _normalize_to_relative_lora(result[0], name)
            except Exception:
                pass

        if not lora_path:
            resolved, available = resolve_lora_path(name)
            lora_path = _normalize_to_relative_lora(resolved if (available and resolved) else name, name)

        stack.append((lora_path, strength, clip_strength))

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

    RETURN_TYPES = ("MULTI_LORA_STACK",)
    RETURN_NAMES = ("multi_lora_stack",)
    FUNCTION = "stack_multi"
    DESCRIPTION = (
        "Multi-slot LoRA stacker with a visual 4-panel UI (A / B / C / D). "
        "Outputs one MULTI_LORA_STACK payload containing all four stacks."
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
        multi_lora_stack = {
            "a": stack_a,
            "b": stack_b,
            "c": stack_c,
            "d": stack_d,
            "stacks": {
                "a": stack_a,
                "b": stack_b,
                "c": stack_c,
                "d": stack_d,
            },
            "order": ["a", "b", "c", "d"],
            "version": 1,
        }
        return (multi_lora_stack,)


def _coerce_lora_stack(raw_stack):
    """Normalize an incoming LORA_STACK-like payload to list[(path, model, clip)]."""
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


class MultiLoraCombine:
    """Combine optional multi stack + optional A/B/C/D stacks into one MULTI_LORA_STACK."""

    NAME = "Multi LoRA Combine"
    CATEGORY = "Prompt Manager"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "multi_lora_stack": ("MULTI_LORA_STACK", {
                    "tooltip": "Optional existing MULTI_LORA_STACK to merge into.",
                }),
                "lora_stack_a": ("LORA_STACK", {
                    "tooltip": "Optional stack to append to slot A.",
                }),
                "lora_stack_b": ("LORA_STACK", {
                    "tooltip": "Optional stack to append to slot B.",
                }),
                "lora_stack_c": ("LORA_STACK", {
                    "tooltip": "Optional stack to append to slot C.",
                }),
                "lora_stack_d": ("LORA_STACK", {
                    "tooltip": "Optional stack to append to slot D.",
                }),
            },
        }

    RETURN_TYPES = ("MULTI_LORA_STACK",)
    RETURN_NAMES = ("multi_lora_stack",)
    FUNCTION = "combine_multi"
    DESCRIPTION = (
        "Combine an optional MULTI_LORA_STACK with optional LORA_STACK A/B/C/D "
        "inputs and output one MULTI_LORA_STACK payload."
    )

    def combine_multi(
        self,
        multi_lora_stack=None,
        lora_stack_a=None,
        lora_stack_b=None,
        lora_stack_c=None,
        lora_stack_d=None,
        **kwargs,
    ):
        # Start from empty payload when no multi input is connected.
        out_a = []
        out_b = []
        out_c = []
        out_d = []

        if isinstance(multi_lora_stack, dict):
            stacks_obj = multi_lora_stack.get("stacks") if isinstance(multi_lora_stack.get("stacks"), dict) else {}
            out_a = _coerce_lora_stack(multi_lora_stack.get("a", stacks_obj.get("a")))
            out_b = _coerce_lora_stack(multi_lora_stack.get("b", stacks_obj.get("b")))
            out_c = _coerce_lora_stack(multi_lora_stack.get("c", stacks_obj.get("c")))
            out_d = _coerce_lora_stack(multi_lora_stack.get("d", stacks_obj.get("d")))

        # Append separate stack inputs to each slot (if connected).
        out_a.extend(_coerce_lora_stack(lora_stack_a))
        out_b.extend(_coerce_lora_stack(lora_stack_b))
        out_c.extend(_coerce_lora_stack(lora_stack_c))
        out_d.extend(_coerce_lora_stack(lora_stack_d))

        merged = {
            "a": out_a,
            "b": out_b,
            "c": out_c,
            "d": out_d,
            "stacks": {
                "a": out_a,
                "b": out_b,
                "c": out_c,
                "d": out_d,
            },
            "order": ["a", "b", "c", "d"],
            "version": 1,
        }
        return (merged,)
