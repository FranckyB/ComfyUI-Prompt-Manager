"""
Prompt Mixer Selector - pick prompt fragments from the isolated mixer library.

Supports single selection, multi-select random mode, and Comfy-style strength.
"""
import json
import math
import random
import time
import server

from .prompt_mixer_store import PromptMixerStore, _find_category_case_insensitive, _find_prompt_case_insensitive


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
    return f"{trimmed_category}: {trimmed_text}"


def _pick_from_selected(selected_json, seed):
    """Return one prompt name from a JSON list.

    Uses true random selection by default; if a non-zero seed is provided,
    selection becomes deterministic.
    """
    try:
        names = json.loads(selected_json or "[]")
    except Exception:
        names = []
    if not isinstance(names, list) or not names:
        return ""

    try:
        seed_value = int(seed)
    except Exception:
        seed_value = 0

    if seed_value == 0:
        return random.choice(names)

    rng = random.Random(seed_value)
    return rng.choice(names)


def _has_multiple_selection(selected_json):
    """Return True when more than one prompt is selected."""
    try:
        names = json.loads(selected_json or "[]")
    except Exception:
        names = []
    return isinstance(names, list) and len(names) > 1


def _resolve_run_seed(seed):
    """Return effective seed for this execution.

    Seed 0 means auto-random per run; non-zero seeds stay deterministic.
    """
    try:
        seed_value = int(seed)
    except Exception:
        seed_value = 0

    if seed_value != 0:
        return seed_value

    # Auto-random mode: derive a fresh seed each execution.
    return random.SystemRandom().randrange(0, 0xFFFFFFFFFFFFFFFF)


class PromptMixerSelector:
    """Select a prompt fragment and mix it into a growing prompt."""

    @classmethod
    def INPUT_TYPES(s):
        prompts_data = PromptMixerStore.load_prompts()
        categories = sorted([c for c in prompts_data.keys() if c != "__meta__"], key=str.lower)
        if not categories:
            categories = [""]

        all_prompts = []
        first_prompt = ""
        first_text = ""
        for cat in categories:
            entries = prompts_data.get(cat, {})
            if not isinstance(entries, dict):
                continue
            for name, entry in entries.items():
                if name == "__meta__":
                    continue
                all_prompts.append(name)
                if not first_prompt:
                    first_prompt = name
                    if isinstance(entry, dict):
                        first_text = entry.get("prompt", "") or ""

        all_prompts = sorted(all_prompts, key=str.lower)
        if not all_prompts:
            all_prompts = [""]

        return {
            "required": {
                "category": (categories, {"default": categories[0]}),
                "name": (all_prompts, {"default": first_prompt}),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Comfy-style prompt strength for the selected fragment. 1 keeps the default behavior.",
                }),
                "selected_prompts": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "dynamicPrompts": False,
                    "tooltip": "Internal: JSON list of selected fragment names",
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Connect a base prompt. The selected fragment will be appended after it. Works unconnected.",
                }),
            },
            "hidden": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                }),
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "api_prompt": "PROMPT",
            }
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Select a Prompt Mixer fragment and append it to a prompt."
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "select_fragment"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, name, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, category, name, strength=1.0,
                   selected_prompts="", seed=0, prompt="", **kwargs):
        if _has_multiple_selection(selected_prompts):
            # Force re-evaluation in multi mode so each queued run can pick anew.
            dynamic_seed = time.time_ns()
            return (category, name, strength, selected_prompts, dynamic_seed, prompt)
        return (category, name, strength, selected_prompts, seed, prompt)

    def select_fragment(self, category, name, strength=1.0,
                        selected_prompts="", seed=0, prompt="",
                        unique_id=None, extra_pnginfo=None, api_prompt=None):
        prompts_data = PromptMixerStore.load_prompts()
        canonical_category = _find_category_case_insensitive(prompts_data, category) or category
        category_data = prompts_data.get(canonical_category, {})

        chosen_name = name
        if _has_multiple_selection(selected_prompts):
            run_seed = _resolve_run_seed(seed)
            picked = _pick_from_selected(selected_prompts, run_seed)
            if picked:
                chosen_name = picked

        entry, canonical_name = _find_prompt_case_insensitive(category_data, chosen_name)
        fragment_text = ""
        thumbnail = None
        if isinstance(entry, dict):
            fragment_text = entry.get("prompt", "") or ""
            thumbnail = entry.get("thumbnail")

        labeled_fragment = _prefix_with_category(canonical_category, fragment_text)
        formatted = _format_fragment(labeled_fragment, strength)

        base = ""
        if isinstance(prompt, str):
            base = prompt.strip()

        if not formatted:
            output_text = base
        elif not base:
            output_text = formatted
        else:
            output_text = f"{base}\n{formatted}"

        if unique_id is not None:
            server.PromptServer.instance.send_sync("prompt-mixer-selector-update", {
                "node_id": unique_id,
                "category": canonical_category,
                "name": canonical_name or chosen_name,
                "fragment_text": fragment_text,
                "output_text": output_text,
                "thumbnail": thumbnail,
            })

        # Persist resolved text into API metadata for downstream nodes.
        node_id = str(unique_id) if unique_id is not None else ""
        if node_id:
            if isinstance(api_prompt, dict):
                prompt_node = api_prompt.get(node_id)
                if isinstance(prompt_node, dict):
                    inputs = prompt_node.get("inputs")
                    if isinstance(inputs, dict):
                        prompt_node["inputs"] = {**inputs, "prompt": output_text}

        return (output_text,)

