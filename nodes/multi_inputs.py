"""Bundled multi-slot helper node for Recipe Builder (Multi Model)."""


class RecipeBuilderDataBundle:
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
                "seed_a": ("INT", {"forceInput": True}),
                "seed_b": ("INT", {"forceInput": True}),
                "seed_c": ("INT", {"forceInput": True}),
                "seed_d": ("INT", {"forceInput": True}),
                "loras_a": ("LORA_STACK",),
                "loras_b": ("LORA_STACK",),
                "loras_c": ("LORA_STACK",),
                "loras_d": ("LORA_STACK",),
            },
        }

    RETURN_TYPES = ("BUILDER_DATA",)
    RETURN_NAMES = ("builder_data",)
    FUNCTION = "bundle"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Bundle A/B/C/D prompts, seeds, and LoRA stacks into one builder_data payload for Recipe Builder."

    def bundle(self,
               pos_a="", pos_b="", pos_c="", pos_d="",
               neg_a="", neg_b="", neg_c="", neg_d="",
               seed_a=0, seed_b=0, seed_c=0, seed_d=0,
               loras_a=None, loras_b=None, loras_c=None, loras_d=None):
        payload = {
            "pos_a": str(pos_a or ""),
            "pos_b": str(pos_b or ""),
            "pos_c": str(pos_c or ""),
            "pos_d": str(pos_d or ""),
            "neg_a": str(neg_a or ""),
            "neg_b": str(neg_b or ""),
            "neg_c": str(neg_c or ""),
            "neg_d": str(neg_d or ""),
            "seed_a": int(seed_a or 0),
            "seed_b": int(seed_b or 0),
            "seed_c": int(seed_c or 0),
            "seed_d": int(seed_d or 0),
            "loras_a": list(loras_a) if loras_a else [],
            "loras_b": list(loras_b) if loras_b else [],
            "loras_c": list(loras_c) if loras_c else [],
            "loras_d": list(loras_d) if loras_d else [],
        }
        return (payload,)
