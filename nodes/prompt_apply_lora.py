"""
ComfyUI Prompt Apply LoRA - Apply a LORA_STACK to model
"""
import ast
import json
import os

import folder_paths
import comfy.sd
import comfy.utils

from ..py.lora_utils import resolve_lora_path


def parse_lora_stack_text(text):
    """Parse a lora stack from a text string.

    Accepts formats:
      JSON:   [["lora_name", 1.0, 1.0], ...]
      Python: [("lora_name", 1.0, 1.0), ...]
      Plain:  One path/name per line (strength defaults to 1.0)
    Returns a list of (name, model_strength, clip_strength) tuples.
    """
    text = text.strip()
    if not text:
        return []

    # Try JSON first
    try:
        data = json.loads(text)
        return [(str(item[0]), float(item[1]), float(item[2])) for item in data]
    except (json.JSONDecodeError, ValueError, IndexError, TypeError):
        pass

    # Fall back to Python literal (handles tuples)
    try:
        data = ast.literal_eval(text)
        return [(str(item[0]), float(item[1]), float(item[2])) for item in data]
    except (ValueError, SyntaxError, IndexError, TypeError):
        pass

    # Fall back to newline-separated paths/names (strength defaults to 1.0)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return [(line, 1.0, 1.0) for line in lines]

    return []


class PromptApplyLora:
    """
    Apply a LoRA stack to a model.
    Takes a LORA_STACK (list of tuples) and applies each LoRA sequentially.
    Uses fuzzy matching to find LoRAs on disk — LoRAs not found are skipped.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "lora_stack": ("LORA_STACK",),
                "lora_stack_text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_stack"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Apply a LoRA stack to a model. Connect a LORA_STACK, paste one as text (list of [name, model_str, clip_str]), or list full file paths one per line. Full paths can reference any location on disk."

    def apply_stack(self, model, lora_stack=None, lora_stack_text=""):
        # Merge both sources
        stack = list(lora_stack) if lora_stack else []
        if lora_stack_text:
            stack.extend(parse_lora_stack_text(lora_stack_text))

        if not stack:
            return (model,)

        model_out = model

        for lora_name, model_strength, clip_strength in stack:
            # Skip if no strength
            if model_strength == 0 and clip_strength == 0:
                continue

            # If it looks like a full path and exists on disk, use it directly
            if os.path.isabs(lora_name) and os.path.isfile(lora_name):
                lora_path = lora_name
            else:
                # Resolve LoRA using fuzzy matching (handles renamed LoRAs, WAN tokens, etc.)
                lora_path, found = resolve_lora_path(lora_name)
                if not found:
                    print(f"[PromptApplyLora] Warning: LoRA not found, skipping: {lora_name}")
                    continue

            # Load the LoRA
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

            # Apply to model
            model_out, _ = comfy.sd.load_lora_for_models(
                model_out, None, lora, model_strength, clip_strength
            )

        return (model_out,)
