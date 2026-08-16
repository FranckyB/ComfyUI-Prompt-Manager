"""Prompt Browser node with source-aware prompt selection and random multi-pick."""
import json
import random
import time

import server

from ..py.prompt_composer_store import PromptComposerStore, _find_category_case_insensitive, _find_prompt_case_insensitive
from .prompt_manager_basic import _get_workflow_node
from .prompt_manager_basic import PromptManager
from .prompt_generator import PromptGeneratorDataStore


def _is_hidden_category_entry_key(name):
    normalized = str(name or "").strip().lower()
    return normalized in {"__meta__", "_base_prompt_", "_prompt_type_"}


class PromptBrowser:
    """Browse prompts, select one or many, and emit a resolved prompt."""

    SOURCE_COMPOSE = "Compose Data"
    SOURCE_PROMPT = "Prompt Data"
    SOURCE_SYSTEM_PROMPTS = "System Prompts"

    @staticmethod
    def _load_prompts_for_source(source):
        if str(source or "") == PromptBrowser.SOURCE_PROMPT:
            return PromptManager.load_prompts()
        if str(source or "") == PromptBrowser.SOURCE_SYSTEM_PROMPTS:
            return PromptGeneratorDataStore.load()
        return PromptComposerStore.load_prompts()

    @staticmethod
    def _pick_from_selected(selected_json, seed):
        try:
            names = json.loads(selected_json or "[]")
        except Exception:
            names = []
        if not isinstance(names, list) or not names:
            return ""

        normalized = [str(name).strip() for name in names if str(name).strip()]
        if not normalized:
            return ""

        try:
            seed_value = int(seed)
        except Exception:
            seed_value = 0

        if seed_value == 0:
            return random.choice(normalized)

        rng = random.Random(seed_value)
        return rng.choice(normalized)

    @staticmethod
    def _has_multiple_selection(selected_json):
        try:
            names = json.loads(selected_json or "[]")
        except Exception:
            names = []
        return isinstance(names, list) and len(names) > 1

    @classmethod
    def INPUT_TYPES(s):
        compose_data = PromptComposerStore.load_prompts()
        prompt_data = PromptManager.load_prompts()
        system_data = PromptGeneratorDataStore.load()

        categories_set = set()
        for source_data in (compose_data, prompt_data, system_data):
            for category_name in source_data.keys():
                if category_name == "__meta__":
                    continue
                categories_set.add(category_name)

        categories = sorted(categories_set, key=str.lower)
        if not categories:
            categories = [""]

        all_prompts = []
        first_prompt_text = ""
        first_prompt = ""
        for source_data in (compose_data, prompt_data, system_data):
            for cat in categories:
                entries = source_data.get(cat, {})
                if not isinstance(entries, dict):
                    continue
                for name, entry in entries.items():
                    if _is_hidden_category_entry_key(name):
                        continue
                    all_prompts.append(name)
                    if not first_prompt:
                        first_prompt = name
                        if isinstance(entry, dict):
                            first_prompt_text = entry.get("prompt", "") or ""

        all_prompts = sorted(all_prompts, key=str.lower)
        if not all_prompts:
            all_prompts = [""]

        return {
            "required": {
                "source": ([s.SOURCE_COMPOSE, s.SOURCE_PROMPT, s.SOURCE_SYSTEM_PROMPTS], {"default": s.SOURCE_SYSTEM_PROMPTS}),
                "category": (categories, {"default": categories[0]}),
                "name": (all_prompts, {"default": first_prompt}),
                "selected_prompts": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "dynamicPrompts": False,
                    "tooltip": "Internal: JSON list of selected prompts",
                }),
                "use_prompt_input": ("BOOLEAN", {
                    "default": False,
                    "label_on": "on",
                    "label_off": "off",
                    "tooltip": "Toggle to use connected prompt input instead of internal text",
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": first_prompt_text,
                    "placeholder": "Enter prompt fragment text",
                    "dynamicPrompts": False,
                    "tooltip": "Enter prompt fragment text directly",
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Connect a prompt input to pass through",
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
    DESCRIPTION = "Browse prompts from Prompt Data, Compose Data, or System Prompts with single or multi-random selection."
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_prompt"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, name, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, source, category, name, selected_prompts="[]", use_prompt_input=False, text="", seed=0, prompt="", **kwargs):
        if cls._has_multiple_selection(selected_prompts):
            return (source, category, name, selected_prompts, use_prompt_input, text, time.time_ns(), prompt)
        return (source, category, name, selected_prompts, use_prompt_input, text, seed, prompt)

    @staticmethod
    def _patch_runtime_prompt_metadata(unique_id, output_text, extra_pnginfo=None, api_prompt=None):
        """Persist resolved prompt into workflow/api metadata for downstream save nodes."""
        node_id = str(unique_id) if unique_id is not None else ""
        if not node_id:
            return

        workflow_node = _get_workflow_node(extra_pnginfo, node_id)
        if isinstance(workflow_node, dict):
            widgets = workflow_node.get("widgets_values")
            if isinstance(widgets, list):
                # Current layout: [source, category, name, selected_prompts, use_prompt_input, text]
                if len(widgets) > 4:
                    widgets[4] = False
                if len(widgets) > 5:
                    widgets[5] = output_text
                elif len(widgets) > 4:
                    widgets[4] = output_text

        if isinstance(api_prompt, dict):
            prompt_node = api_prompt.get(node_id)
            if isinstance(prompt_node, dict):
                inputs = prompt_node.get("inputs")
                if isinstance(inputs, dict):
                    inputs["use_prompt_input"] = False
                    inputs["text"] = output_text

    def get_prompt(self, source, category, name, selected_prompts="[]", use_prompt_input=False, text="", prompt=None,
                   seed=0, unique_id=None, extra_pnginfo=None, api_prompt=None):
        prompts_data = self._load_prompts_for_source(source)
        canonical_category = _find_category_case_insensitive(prompts_data, category) or category
        category_data = prompts_data.get(canonical_category, {})

        chosen_name = name
        if self._has_multiple_selection(selected_prompts):
            picked = self._pick_from_selected(selected_prompts, seed)
            if picked:
                chosen_name = picked

        entry, canonical_name = _find_prompt_case_insensitive(category_data, chosen_name)
        resolved_text = ""
        if isinstance(entry, dict):
            resolved_text = str(entry.get("prompt", "") or "")

        if use_prompt_input and isinstance(prompt, str):
            output_text = prompt
        elif resolved_text.strip():
            output_text = resolved_text
        else:
            output_text = text

        if unique_id is not None:
            server.PromptServer.instance.send_sync("prompt-manager-update-text", {
                "node_id": unique_id,
                "prompt": output_text,
                "use_prompt_input": use_prompt_input,
                "selected_name": canonical_name or chosen_name,
                "source": source,
            })

        self._patch_runtime_prompt_metadata(unique_id, output_text, extra_pnginfo, api_prompt)
        return (output_text,)
