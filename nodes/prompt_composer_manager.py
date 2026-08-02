"""
Prompt Composer Manager - simplified writer for Prompt Composer fragments.

Mirrors the basic Prompt Manager flow but stores data in prompt_composer_data.json
instead of the main Prompt Manager library.
"""
import server

from ..py.prompt_composer_store import PromptComposerStore
from .prompt_manager_basic import _get_workflow_node


def _is_hidden_category_entry_key(name):
    normalized = str(name or "").strip().lower()
    return normalized in {"__meta__", "_base_prompt_", "_prompt_type_"}


class PromptComposerManager:
    """Save/load and edit prompt composer fragments."""

    @classmethod
    def INPUT_TYPES(s):
        prompts_data = PromptComposerStore.load_prompts()
        categories = sorted([c for c in prompts_data.keys() if c != "__meta__"], key=str.lower)
        if not categories:
            categories = [""]

        all_prompts = []
        first_prompt_text = ""
        first_prompt = ""
        for cat in categories:
            entries = prompts_data.get(cat, {})
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
                "category": (categories, {"default": categories[0]}),
                "name": (all_prompts, {"default": first_prompt}),
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
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "api_prompt": "PROMPT",
            }
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Save and edit Prompt Composer fragments in an isolated library."
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_prompt"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, name, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, category, name, use_prompt_input, text="", **kwargs):
        return (category, name, use_prompt_input, text)

    @staticmethod
    def _patch_runtime_prompt_metadata(unique_id, output_text, extra_pnginfo=None, api_prompt=None):
        """Persist resolved prompt into workflow/api metadata for downstream save nodes."""
        node_id = str(unique_id) if unique_id is not None else ""
        if not node_id:
            return

        workflow_node = _get_workflow_node(extra_pnginfo, node_id)
        if isinstance(workflow_node, dict):
            widgets = workflow_node.get("widgets_values")
            if isinstance(widgets, list) and len(widgets) > 2:
                widgets[2] = False
            if len(widgets) > 4:
                widgets[4] = output_text

        if isinstance(api_prompt, dict):
            prompt_node = api_prompt.get(node_id)
            if isinstance(prompt_node, dict):
                inputs = prompt_node.get("inputs")
                if isinstance(inputs, dict):
                    inputs["use_prompt_input"] = False
                    inputs["text"] = output_text

    def get_prompt(self, category, name, use_prompt_input, text="", prompt=None,
                   unique_id=None, extra_pnginfo=None, api_prompt=None):
        output_text = prompt if use_prompt_input and isinstance(prompt, str) else text

        if unique_id is not None:
            server.PromptServer.instance.send_sync("prompt-manager-update-text", {
                "node_id": unique_id,
                "prompt": output_text,
                "use_prompt_input": use_prompt_input,
            })

        self._patch_runtime_prompt_metadata(unique_id, output_text, extra_pnginfo, api_prompt)
        return (output_text,)
