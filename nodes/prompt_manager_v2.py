"""Prompt Manager: a saved prompt, typed or taken from the prompt input, with the LoRA stacks and trigger words that
go with it. The node face only shows what is in use: one LoRA panel per slot that has LoRAs, trigger words when there
are any. On/off and strength edits live in the `state` widget, as overrides by slot and LoRA name."""

import json
import os

import server

from ..py.lora_stacks import (
    LORA_MODE_COMBINE,
    LORA_MODE_INPUT_ONLY,
    LORA_MODE_PROMPT_ONLY,
    LORA_MODES,
    MODEL_SLOT_KEYS,
    PROMPT_MANAGER_RECIPE_KEY,
    derive_prompt_fields,
    flatten_lora_slots,
    merge_prompt_composer_input_data,
    normalize_lora_slots,
)
from ..py.lora_utils import get_available_loras, get_lora_relative_path, normalize_path_separators, strip_lora_extension
from ..py.prompt_metadata import record_prompt
from ..py.prompt_store import is_hidden_key, load_prompts
from ..py.workflow_data_utils import build_v2_recipe_data_from_prompt, ensure_v2_recipe_data


def _saved_entry(category, name):
    entry = load_prompts().get(category, {}).get(name, {})
    return entry if isinstance(entry, dict) else {}


def _parse_state(state):
    """The node's overrides: {"loras": {slot: {name: {"active", "strength"}}}, "trigger_words": {text: active}},
    names and texts in lower case."""
    try:
        parsed = json.loads(state) if isinstance(state, str) and state else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _lora_name(path):
    return strip_lora_extension(os.path.basename(normalize_path_separators(path)))


def _saved_row(lora):
    """A LoRA saved with the prompt (or in its recipe), as a panel row; found among the local LoRAs by name."""
    name = str(lora.get("name", "")).strip()
    strength = float(lora.get("strength", lora.get("model_strength", 1.0)))
    path, available = get_lora_relative_path(name)
    return {
        "name": name,
        "path": path if available else name,
        "strength": strength,
        "clip_strength": float(lora.get("clip_strength", strength)),
        "active": lora.get("active", True) is not False,
        "available": available,
        "from_input": False,
    }


def _input_row(path, model_strength, clip_strength, lora_files):
    """A LoRA from the lora_stacks input, as a panel row."""
    name = _lora_name(path)
    available = path in lora_files
    if not available:
        found_path, available = get_lora_relative_path(name)
        if available:
            path = found_path
    return {
        "name": name,
        "path": path,
        "strength": float(model_strength),
        "clip_strength": float(clip_strength),
        "active": True,
        "available": available,
        "from_input": True,
    }


def _apply_override(row, override):
    """The row with the node's on/off and strength edits; clip strength follows the strength by the same ratio."""
    if "active" in override:
        row["active"] = override["active"] is not False
    if isinstance(override.get("strength"), (int, float)):
        strength = float(override["strength"])
        row["clip_strength"] *= strength / row["strength"] if row["strength"] else 1.0
        row["strength"] = strength
    return row


def _slot_rows(prompt_loras, input_stack, mode, overrides, lora_files):
    """One slot's panel rows: the prompt's LoRAs, the input's, or both (the prompt's first, an input LoRA of the same
    name left out), with the node's edits."""
    rows = []
    if mode != LORA_MODE_INPUT_ONLY:
        rows += [_saved_row(lora) for lora in prompt_loras if isinstance(lora, dict) and str(lora.get("name", "")).strip()]
    if mode != LORA_MODE_PROMPT_ONLY:
        rows += [_input_row(*item, lora_files) for item in input_stack]
    unique, seen = [], set()
    for row in rows:
        key = row["name"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(_apply_override(row, overrides.get(key, {})))
    return unique


def _trigger_rows(saved_words, connected_words, overrides):
    """Trigger words saved with the prompt, then the connected ones not already there, with the node's on/off edits.
    Connected words come as one string, groups split by ",," (as LoRA Manager sends them)."""
    rows, seen = [], set()
    candidates = [(word.get("text", "") if isinstance(word, dict) else word, word.get("active", True) if isinstance(word, dict) else True, False)
                  for word in saved_words or []]
    if isinstance(connected_words, str):
        candidates += [(word, True, True) for word in connected_words.split(",,")]
    for text, active, from_input in candidates:
        text = str(text or "").strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            rows.append({"text": text, "active": overrides.get(text.lower(), active is not False), "from_input": from_input})
    return rows


def _with_trigger_words(prompt, words):
    """Trigger words before the prompt, the block ending in punctuation."""
    block = ", ".join(words).strip()
    if not block:
        return prompt
    if not block.endswith((".", ",", ":")):
        block += ","
    prompt = str(prompt or "").strip()
    return f"{block} {prompt}" if prompt else block


class PromptManagerV2:
    CATEGORY = "Prompt Manager"
    DESCRIPTION = "A saved prompt, typed or from the prompt input, with its LoRA stacks and trigger words. LoRA stacks go out as a plain LoRA stack when one slot has LoRAs, as a multi LoRA stack when several do."

    @classmethod
    def INPUT_TYPES(cls):
        prompts = load_prompts()
        categories = list(prompts) or ["Default"]
        names = sorted({key for entries in prompts.values() for key in entries if not is_hidden_key(key)} | {""}, key=str.lower)
        return {
            "required": {
                "category": (categories, {"default": categories[0]}),
                "name": (names, {"default": ""}),
                "use_prompt_input": ("BOOLEAN", {"default": False, "tooltip": "Use the connected prompt instead of the typed one."}),
                "lora_mode": (list(LORA_MODES), {"default": LORA_MODE_COMBINE, "tooltip": "Which LoRAs go out: the prompt's, both (the prompt's first), or the input's."}),
                "text": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": False}),
            },
            "optional": {
                "prompt": ("STRING", {"forceInput": True, "lazy": True, "tooltip": "A prompt to use instead of the typed one."}),
                "lora_stacks": ("LORA_STACK,MULTI_LORA_STACK", {"forceInput": True, "tooltip": "LoRA stacks to combine with the prompt's; a plain LoRA stack is slot A."}),
                "manager_data": ("MANAGER_DATA", {"forceInput": True, "tooltip": "A recipe: its LoRAs are the prompt's, its prompt is used when none is typed."}),
                "trigger_words": ("STRING", {"forceInput": True, "tooltip": "Trigger words to put before the prompt, groups separated by ,,"}),
                "state": ("STRING", {"default": "{}"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "api_prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("STRING", "LORA_STACK,MULTI_LORA_STACK", "MANAGER_DATA")
    RETURN_NAMES = ("prompt", "lora_stacks", "manager_data")
    OUTPUT_TOOLTIPS = (
        "The prompt, its active trigger words first.",
        "The active LoRAs: a plain LoRA stack when one slot has LoRAs, a multi LoRA stack (slots A-D) when several do.",
        "The prompt and its LoRAs as a recipe.",
    )
    FUNCTION = "get_prompt"

    @classmethod
    def VALIDATE_INPUTS(cls, category, name):
        # A prompt or category renamed or deleted since the workflow was saved still runs, from its typed text.
        return True

    @classmethod
    def IS_CHANGED(cls, category, name, **kwargs):
        # The prompt's saved LoRAs and trigger words can change in the library without any input changing.
        return json.dumps(_saved_entry(category, name), sort_keys=True, default=str)

    def check_lazy_status(self, use_prompt_input, prompt=None, **kwargs):
        return ["prompt"] if use_prompt_input else []

    def get_prompt(self, category, name, use_prompt_input, lora_mode, text, prompt=None, lora_stacks=None, manager_data=None,
                   trigger_words=None, state="{}", unique_id=None, extra_pnginfo=None, api_prompt=None):
        overrides = _parse_state(state)
        entry = _saved_entry(category, name)
        stored_recipe = entry.get("workflow_data")
        recipe = (ensure_v2_recipe_data(manager_data, source="PromptManager") if isinstance(manager_data, dict)
                  else ensure_v2_recipe_data(stored_recipe, source="PromptManager") if isinstance(stored_recipe, dict) else None)
        fields = derive_prompt_fields(recipe)

        # The prompt input when it is on and has text, else the typed text, else the recipe's prompt.
        if use_prompt_input and prompt:
            output_text = prompt
        else:
            output_text = text if text.strip() else fields["prompt"]

        inputs = normalize_lora_slots(lora_stacks, "a")
        lora_files = get_available_loras()
        lora_overrides = overrides.get("loras") if isinstance(overrides.get("loras"), dict) else {}
        rows, stacks = {}, {}
        for slot in MODEL_SLOT_KEYS:
            # The prompt's LoRAs: the connected recipe's, else the saved prompt's, else its saved recipe's.
            prompt_loras = fields[f"loras_{slot}"] if isinstance(manager_data, dict) else (entry.get(f"loras_{slot}") or fields[f"loras_{slot}"])
            slot_overrides = lora_overrides.get(slot) if isinstance(lora_overrides.get(slot), dict) else {}
            rows[slot] = _slot_rows(prompt_loras, inputs[slot], lora_mode, slot_overrides, lora_files)
            stacks[slot] = [(row["path"], row["strength"], row["clip_strength"]) for row in rows[slot] if row["active"] and row["available"]]

        word_overrides = overrides.get("trigger_words") if isinstance(overrides.get("trigger_words"), dict) else {}
        words = _trigger_rows(entry.get("trigger_words"), trigger_words, word_overrides)
        final_output = _with_trigger_words(output_text, [word["text"] for word in words if word["active"]])

        if unique_id is not None:
            server.PromptServer.instance.send_sync("prompt-manager-v2-update", {"node_id": unique_id, "loras": rows, "trigger_words": words})

        # Saver nodes embed the final prompt as typed text, not the link that fed it.
        record_prompt(unique_id, final_output, extra_pnginfo, api_prompt, toggle="use_prompt_input", link_input="prompt", text_index=4)

        filled = [slot for slot in MODEL_SLOT_KEYS if stacks[slot]]
        lora_output = stacks if len(filled) > 1 else (stacks[filled[0]] if filled else [])

        recipe_out = build_v2_recipe_data_from_prompt(
            prompt_text=final_output,
            negative_prompt=fields["negative_prompt"],
            **{f"loras_{slot}": rows[slot] for slot in MODEL_SLOT_KEYS},
            source="PromptManager",
            base_recipe_data=recipe,
        )
        prompt_input_text = prompt or ""
        recipe_out = merge_prompt_composer_input_data(
            recipe_out,
            input_prompt=prompt_input_text,
            prompt_enabled=bool(use_prompt_input and prompt_input_text),
            input_lora_stack=flatten_lora_slots(stacks),
        )
        recipe_out[PROMPT_MANAGER_RECIPE_KEY] = {"category": category, "name": name}
        return (final_output, lora_output, recipe_out)
