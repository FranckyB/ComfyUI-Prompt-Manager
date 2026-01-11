"""
ComfyUI Prompt Apply LoRA - Apply a LORA_STACK to model and clip
"""
import folder_paths
import comfy.sd
import comfy.utils


class PromptApplyLora:
    """
    Apply a LoRA stack to a model and optionally a CLIP.
    Takes a LORA_STACK (list of tuples) and applies each LoRA sequentially.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_stack": ("LORA_STACK",),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_stack"
    CATEGORY = "Prompt Manager"

    def apply_stack(self, model, lora_stack, clip=None):
        if not lora_stack:
            return (model, clip)

        model_out = model
        clip_out = clip

        for lora_name, model_strength, clip_strength in lora_stack:
            # Skip if no strength
            if model_strength == 0 and clip_strength == 0:
                continue

            # Get the LoRA path
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                print(f"[PromptApplyLora] Warning: LoRA not found: {lora_name}")
                continue

            # Load the LoRA
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

            # Apply to model and clip
            model_out, clip_out = comfy.sd.load_lora_for_models(
                model_out, clip_out, lora, model_strength, clip_strength
            )

        return (model_out, clip_out)
