"""Bundled multi-slot helper nodes for Recipe Builder (Multi Model)."""


class MultiPromptBundle:
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

    RETURN_TYPES = ("MULTI_PROMPTS",)
    RETURN_NAMES = ("multi_prompts",)
    FUNCTION = "bundle"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Bundle 4 prompt inputs into one multi-slot prompt payload. Use one node for positive and another for negative prompts."

    def bundle(self, model_a="", model_b="", model_c="", model_d=""):
        payload = {
            "model_a": str(model_a or ""),
            "model_b": str(model_b or ""),
            "model_c": str(model_c or ""),
            "model_d": str(model_d or ""),
        }
        return (payload,)


class MultiSeedBundle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model_a": ("INT", {"forceInput": True}),
                "model_b": ("INT", {"forceInput": True}),
                "model_c": ("INT", {"forceInput": True}),
                "model_d": ("INT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("MULTI_SEEDS",)
    RETURN_NAMES = ("multi_seeds",)
    FUNCTION = "bundle"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Bundle A/B/C/D seeds into one multi-slot seed payload."

    def bundle(self, model_a=0, model_b=0, model_c=0, model_d=0):
        payload = {
            "model_a": int(model_a or 0),
            "model_b": int(model_b or 0),
            "model_c": int(model_c or 0),
            "model_d": int(model_d or 0),
        }
        return (payload,)


class MultiLoraStacker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model_a": ("LORA_STACK",),
                "model_b": ("LORA_STACK",),
                "model_c": ("LORA_STACK",),
                "model_d": ("LORA_STACK",),
            },
        }

    RETURN_TYPES = ("MULTI_LORAS",)
    RETURN_NAMES = ("multi_loras",)
    FUNCTION = "bundle"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Bundle 4 LoRA stack inputs into one multi-slot LoRA payload."

    def bundle(self, model_a=None, model_b=None, model_c=None, model_d=None):
        payload = {
            "model_a": list(model_a) if model_a else [],
            "model_b": list(model_b) if model_b else [],
            "model_c": list(model_c) if model_c else [],
            "model_d": list(model_d) if model_d else [],
        }
        return (payload,)
