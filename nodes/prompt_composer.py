"""
Prompt Composer - build a prompt from multiple selectable prompt parts.

Each part stores category, selected prompt names (multi-select), and strength.
"""
import json
import math
import random
import time

from ..py.prompt_composer_store import (
    PromptMixerStore,
    _find_category_case_insensitive,
    _find_prompt_case_insensitive,
)


def _normalize_strength(strength):
    try:
        value = float(strength)
    except (TypeError, ValueError):
        value = 1.0
    if not math.isfinite(value):
        value = 1.0
    return max(0.0, min(5.0, value))


def _format_fragment(text, strength=1.0):
    trimmed = str(text or "").strip()
    if not trimmed:
        return ""
    strength_value = _normalize_strength(strength)
    if strength_value == 1.0:
        return trimmed
    return f"({trimmed}:{strength_value:.15g})"


def _prefix_with_category(category, text):
    trimmed_text = str(text or "").strip()
    if not trimmed_text:
        return ""
    trimmed_category = str(category or "").strip()
    if not trimmed_category:
        return trimmed_text
    if trimmed_category.endswith(":"):
        return f"{trimmed_category} {trimmed_text}"
    return f"{trimmed_category}: {trimmed_text}"


def _resolve_prompt_prefix(category_data, fallback_category):
    def _normalize_prefix(value):
        prefix = str(value or "").strip()
        if not prefix:
            return ""
        return prefix if prefix.endswith(":") else f"{prefix}:"

    if isinstance(category_data, dict):
        raw_prefix = category_data.get("_prompt_prefix_", "")
        prefix = _normalize_prefix(raw_prefix)
        if prefix:
            return prefix
    return _normalize_prefix(fallback_category)


def _parse_parts(parts_data):
    try:
        parts = json.loads(parts_data or "[]")
    except Exception:
        parts = []

    if not isinstance(parts, list):
        return []

    normalized = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        category = str(part.get("category") or "").strip()
        prompts = part.get("prompts")
        if isinstance(prompts, list):
            names = [str(name).strip() for name in prompts if str(name).strip()]
        else:
            names = []
        if not names:
            single_name = str(part.get("name") or "").strip()
            if single_name:
                names = [single_name]
        strength = _normalize_strength(part.get("strength", 1.0))
        if not names:
            continue
        normalized.append({
            "category": category,
            "prompts": names,
            "strength": strength,
        })
    return normalized


def _has_multi_part_selection(parts):
    for part in parts:
        names = part.get("prompts") or []
        if len(names) > 1:
            return True
    return False


def _resolve_run_seed(seed):
    try:
        seed_value = int(seed)
    except Exception:
        seed_value = 0
    if seed_value != 0:
        return seed_value
    return random.SystemRandom().randrange(0, 0xFFFFFFFFFFFFFFFF)


class PromptComposer:
    """Compose prompt fragments from multiple parts in one node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parts_data": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "dynamicPrompts": False,
                    "tooltip": "Internal: JSON list of prompt composer parts",
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Optional base prompt. Composed parts are appended after it.",
                }),
            },
            "hidden": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                }),
            },
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Compose multiple prompt fragments with per-part strength in one node."
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "compose"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, parts_data="[]", seed=0, prompt="", **kwargs):
        parts = _parse_parts(parts_data)
        if _has_multi_part_selection(parts):
            dynamic_seed = time.time_ns()
            return (parts_data, dynamic_seed, prompt)
        return (parts_data, seed, prompt)

    def compose(self, parts_data="[]", seed=0, prompt=""):
        prompts_data = PromptMixerStore.load_prompts()
        parts = _parse_parts(parts_data)

        run_seed = _resolve_run_seed(seed)
        rng = random.Random(run_seed)

        fragments = []
        for part in parts:
            raw_category = part.get("category") or ""
            category = _find_category_case_insensitive(prompts_data, raw_category) or raw_category
            category_data = prompts_data.get(category, {})

            names = part.get("prompts") or []
            if not names:
                continue

            chosen_name = names[0] if len(names) == 1 else rng.choice(names)
            entry, canonical_name = _find_prompt_case_insensitive(category_data, chosen_name)
            if not isinstance(entry, dict):
                continue

            text = entry.get("prompt", "") or ""
            prompt_prefix = _resolve_prompt_prefix(category_data, category)
            labeled = _prefix_with_category(prompt_prefix, text)
            formatted = _format_fragment(labeled, part.get("strength", 1.0))
            if formatted:
                fragments.append(formatted)

        base = str(prompt or "").strip() if isinstance(prompt, str) else ""
        if base and fragments:
            output = f"{base}\n" + "\n".join(fragments)
        elif base:
            output = base
        else:
            output = "\n".join(fragments)

        return (output,)
