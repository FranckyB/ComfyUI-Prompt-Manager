"""Bundled multi-slot helper node for Recipe Builder."""


class RecipeBuilderPromptBundle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "pos_a": ("STRING", {"forceInput": True}),
                "pos_b": ("STRING", {"forceInput": True}),
                "pos_c": ("STRING", {"forceInput": True}),
                "pos_d": ("STRING", {"forceInput": True}),
                "neg_a": ("STRING", {"forceInput": True}),
                "neg_b": ("STRING", {"forceInput": True}),
                "neg_c": ("STRING", {"forceInput": True}),
                "neg_d": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("MULTI_PROMPT",)
    RETURN_NAMES = ("multi_prompt",)
    FUNCTION = "bundle"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Bundle A/B/C/D prompts into a multi_prompt payload for Recipe Builder."

    def bundle(self,
               pos_a="", pos_b="", pos_c="", pos_d="",
               neg_a="", neg_b="", neg_c="", neg_d=""):
        payload = {
            "pos_a": str(pos_a or ""),
            "pos_b": str(pos_b or ""),
            "pos_c": str(pos_c or ""),
            "pos_d": str(pos_d or ""),
            "neg_a": str(neg_a or ""),
            "neg_b": str(neg_b or ""),
            "neg_c": str(neg_c or ""),
            "neg_d": str(neg_d or ""),
        }
        return (payload,)
