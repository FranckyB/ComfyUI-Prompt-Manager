from ..py.model_manager import get_all_models, is_model_local, download_model, get_mmproj_for_model, get_mmproj_path, _preferences_cache
from ..py.ollama_wrapper import discover_ollama_models, is_ollama_available
import time

# Global timestamp to track when models were last updated
_last_model_update = time.time()

def trigger_model_list_refresh():
    """Call this after downloading a model to trigger UI refresh"""
    global _last_model_update
    _last_model_update = time.time()

def get_default_context_size():
    """Return a context size scaled to the active GPU's total VRAM.

    Tiers:
        >= 24 GB  →  8192
        >= 16 GB  →  4096
         < 16 GB →   2048
        No GPU / unknown → 4096 (safe CPU default)

    Note:
        This uses GPU capacity (total VRAM), not current free/available VRAM.
    """
    try:
        import importlib
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            try:
                device_index = torch.cuda.current_device()
            except Exception:
                device_index = 0

            total_vram_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
            if total_vram_gb >= 24:
                return 8192
            elif total_vram_gb >= 16:
                return 4096
            else:
                return 2048
    except Exception:
        pass
    return 4096  # CPU or VRAM detection failed

class PromptGenOptions:
    """Node that provides optional configuration for llama.cpp servers"""

    @classmethod
    def INPUT_TYPES(cls):
        backend = _preferences_cache.get("llm_backend", "llama.cpp")
        preferred_model = str(_preferences_cache.get("preferred_model", "") or "").strip()

        if backend == "ollama":
            # Discover models from Ollama
            ollama_models, _status = discover_ollama_models(_preferences_cache)
            if not ollama_models:
                available_models = ["No Ollama models found - run 'ollama pull <model>'"]
            else:
                available_models = ollama_models
        else:
            available_models = get_all_models()
            if not available_models:
                available_models = ["No models found - check HuggingFace"]

        default_model = available_models[0] if available_models else ""
        if preferred_model and preferred_model in available_models:
            default_model = preferred_model

        return {
            "optional": {
                "model": (available_models, {
                    "default": default_model,
                    "tooltip": "Select model to use (local models listed first, then HuggingFace models)\nDownload sizes: UD-Q4_K_XL ~6GB | Q8_0 ~9.5GB | UD-Q8_K_XL ~13GB"
                }),
                "image2": ("IMAGE", {
                    "tooltip": "Connect an image (required for 'Analyze Image' and 'Analyze Image with Prompt' modes)"
                }),
                "image3": ("IMAGE", {
                    "tooltip": "Connect an image (required for 'Analyze Image' and 'Analyze Image with Prompt' modes)"
                }),
                "image4": ("IMAGE", {
                    "tooltip": "Connect an image (required for 'Analyze Image' and 'Analyze Image with Prompt' modes)"
                }),
                "image5": ("IMAGE", {
                    "tooltip": "Connect an image (required for 'Analyze Image' and 'Analyze Image with Prompt' modes)"
                }),
                "system_prompt_mode": (["replace", "append"], {
                    "default": "append",
                    "tooltip": "replace: the text below fully replaces the default LLM instructions\nappend: the text below is added after the default LLM instructions"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Replace LLM Instructions...",
                    "tooltip": "Custom LLM Instructions (leave empty to use default)\nThe default instructions are designed for generating\ndetailed and imaginative prompts for text-to-image generation."
                }),
                "use_model_default_sampling": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use the model's default sampling parameters (overrides temperature, top_p, etc)"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Controls randomness (0.0 = deterministic, 2.0 = very random)"
                }),
                "top_k": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Sample from top K most likely tokens (0 = disabled)"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
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
                "context_size": ("INT", {
                    "default": get_default_context_size(),
                    "min": 512,
                    "max": 32768,
                    "step": 512,
                    "tooltip": "Context size (increase for vision models or large prompts)"
                }),
                "show_everything_in_console": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print system prompt, user prompt, thinking process, and raw model response to console"
                })
            }
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Configure LLM model, sampling parameters, and system prompt for the Prompt Generator."
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "create_options"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force refresh when models are updated"""
        return _last_model_update

    def create_options(self, model: str = None, image2=None, image3=None, image4=None, image5=None,
                       system_prompt: str = None, system_prompt_mode: str = "replace",
                       use_model_default_sampling: bool = None, temperature: float = None,
                       top_k: int = None, top_p: float = None, min_p: float = None,
                       repeat_penalty: float = None, context_size: int = None,
                       show_everything_in_console: bool = None) -> dict:
        """Create options dictionary with model, LLM parameters, and extra images"""

        # Backward compatibility for workflows saved before `system_prompt_mode` existed.
        # Old widget arrays can shift values by one slot when loaded by newer node definitions.
        mode_valid = system_prompt_mode in ("replace", "append")
        legacy_shape = isinstance(system_prompt, bool) and (
            not mode_valid or isinstance(use_model_default_sampling, (int, float))
        )
        if legacy_shape:
            old_system_prompt = system_prompt_mode
            old_use_default_sampling = system_prompt
            old_temperature = use_model_default_sampling
            old_top_k = temperature
            old_top_p = top_k
            old_min_p = top_p
            old_repeat_penalty = min_p
            old_context_size = repeat_penalty
            old_show_console = context_size

            system_prompt_mode = "replace"
            system_prompt = old_system_prompt if isinstance(old_system_prompt, str) else ""
            use_model_default_sampling = bool(old_use_default_sampling) if old_use_default_sampling is not None else None
            temperature = old_temperature
            top_k = old_top_k
            top_p = old_top_p
            min_p = old_min_p
            repeat_penalty = old_repeat_penalty
            context_size = old_context_size
            show_everything_in_console = bool(old_show_console) if old_show_console is not None else None

        options = {}

        # Handle model selection and download if needed
        backend = _preferences_cache.get("llm_backend", "llama.cpp")

        if backend == "ollama":
            # Ollama mode — no download needed, just validate the selection
            if model and model != "No Ollama models found - run 'ollama pull <model>'":
                options["model"] = model
                options["llm_backend"] = "ollama"
        elif model and model != "No models found - check HuggingFace":
            # Check if model is local
            model_exists = is_model_local(model)

            # For vision models, also check if mmproj exists
            needs_download = not model_exists
            if model_exists:
                mmproj_name = get_mmproj_for_model(model)
                if mmproj_name:
                    # This is a vision model, check if mmproj is present
                    mmproj_path = get_mmproj_path(model)
                    if not mmproj_path:
                        print(f"[Prompt Generator Options] Vision model found but mmproj missing, downloading: {mmproj_name}")
                        needs_download = True

            if needs_download:
                if not model_exists:
                    print(f"[Prompt Generator Options] Model not found locally, downloading: {model}")
                downloaded_path = download_model(model)
                if downloaded_path:
                    trigger_model_list_refresh()
                    options["model"] = model
                else:
                    print(f"[Prompt Generator Options] Failed to download model: {model}")
            else:
                options["model"] = model

        # Add extra images if provided
        if image2 is not None:
            options["image2"] = image2
        if image3 is not None:
            options["image3"] = image3
        if image4 is not None:
            options["image4"] = image4
        if image5 is not None:
            options["image5"] = image5

        # Only include LLM parameters that are provided
        if system_prompt and system_prompt.strip():
            options["system_prompt"] = system_prompt
            options["system_prompt_mode"] = system_prompt_mode

        options["use_model_default_sampling"] = use_model_default_sampling
        options["temperature"] = temperature
        options["top_p"] = top_p
        options["top_k"] = top_k
        options["min_p"] = min_p
        options["repeat_penalty"] = repeat_penalty
        options["context_size"] = context_size
        options["show_everything_in_console"] = show_everything_in_console

        return (options,)


NODE_CLASS_MAPPINGS = {
    "PromptGenOptions": PromptGenOptions
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGenOptions": "Prompt Generator Options"
}
