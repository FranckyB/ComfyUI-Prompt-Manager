from .model_manager import get_all_models, is_model_local, download_model
import time

# Global timestamp to track when models were last updated
_last_model_update = time.time()

def trigger_model_list_refresh():
    """Call this after downloading a model to trigger UI refresh"""
    global _last_model_update
    _last_model_update = time.time()

class PromptGenOptions:
    """Node that provides optional configuration for llama.cpp servers"""

    @classmethod
    def INPUT_TYPES(cls):
        available_models = get_all_models()

        # If no models found, show a placeholder
        if not available_models:
            available_models = ["No models found - check HuggingFace"]

        return {
            "optional": {
                "model": (available_models, {
                    "default": available_models[0] if available_models else "",
                    "tooltip": "Select model to use (local models listed first, then HuggingFace models)"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Replace LLM Instructions...",
                    "tooltip": "Custom LLM Instructions (leave empty to use default)"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Controls randomness (0.0 = deterministic, 2.0 = very random)"
                }),
                "top_k": ("INT", {
                    "default": 40,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Sample from top K most likely tokens (0 = disabled)"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Nucleus sampling: consider tokens with top_p probability mass"
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum probability threshold relative to top token"
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Penalty for repeating tokens (1.0 = no penalty)"
                }),
            }
        }

    CATEGORY = "Prompt Manager"
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "create_options"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force refresh when models are updated"""
        return _last_model_update

    def create_options(self, model: str = None,
                       system_prompt: str = None, temperature: float = None,
                       top_k: int = None, top_p: float = None, min_p: float = None,
                       repeat_penalty: float = None) -> dict:
        """Create options dictionary with model and LLM parameters"""

        options = {}

        # Handle model selection and download if needed
        if model and model != "No models found - check HuggingFace":
            # Re-check if model is local (in case it was just downloaded but UI not refreshed)
            if not is_model_local(model):
                print(f"[Prompt Generator Options] Model not found locally, downloading: {model}")
                downloaded_path = download_model(model)
                if downloaded_path:
                    trigger_model_list_refresh()
                    options["model"] = model
                else:
                    print(f"[Prompt Generator Options] Failed to download model: {model}")
            else:
                options["model"] = model

        # Only include LLM parameters that are provided
        if system_prompt and system_prompt.strip():
            options["system_prompt"] = system_prompt
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        if min_p is not None:
            options["min_p"] = min_p
        if repeat_penalty is not None:
            options["repeat_penalty"] = repeat_penalty

        return (options,)


NODE_CLASS_MAPPINGS = {
    "PromptGenOptions": PromptGenOptions
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGenOptions": "Prompt Generator Options"
}
