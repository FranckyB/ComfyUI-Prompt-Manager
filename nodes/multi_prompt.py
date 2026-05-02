"""Bundled multi-slot helper node for Recipe Builder."""

import json


class RecipeBuilderMultiPrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model_a": ("STRING", {"forceInput": True}),
                "model_b": ("STRING", {"forceInput": True}),
                "model_c": ("STRING", {"forceInput": True}),
                "model_d": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("multi_prompt",)
    FUNCTION = "bundle"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Bundle model_a/model_b/model_c/model_d prompts into a JSON string payload."

    def bundle(self,
               model_a="", model_b="", model_c="", model_d=""):
        payload = {
            "model_a": str(model_a or ""),
            "model_b": str(model_b or ""),
            "model_c": str(model_c or ""),
            "model_d": str(model_d or ""),
        }
        return (json.dumps(payload, ensure_ascii=True),)
