# prompt_generator_options.py

class PromptGenOptions:
    """Options node for Prompt Generator"""

    @classmethod
    def INPUT_TYPES(cls):
        from .model_manager import get_local_models, get_huggingface_models, get_mmproj_models
        
        local_models = get_local_models()
        hf_models = get_huggingface_models()
        mmproj_models = get_mmproj_models()

        # Filter out mmproj files from local models
        local_models = [m for m in local_models if 'mmproj' not in m.lower()]
        
        # Combine: local models first, then downloadable ones marked with ⬇
        all_models = list(local_models)
        for m in hf_models:
            if m not in local_models:
                all_models.append(f"⬇ {m}")
        
        if not all_models:
            all_models = ["No models available"]
        
        # mmproj options: None first, then available mmproj files
        mmproj_options = ["None"] + list(mmproj_models)

        return {
            "required": {
                "model": (all_models, {
                    "default": all_models[0] if all_models else "No models available",
                    "tooltip": "Select model to use. Models with ⬇ will be downloaded automatically."
                }),
                "gpu_layers": ("STRING", {
                    "default": "",
                    "placeholder": "gpu0:0.7, gpu0:0.5,gpu1:0.4",
                    "tooltip": "GPU layer distribution. Examples:\n• empty -> All layers go to the first GPU (default)\n• gpu0:0.7 -> 70% to GPU:0, 30% to CPU\n• gpu0:0.5, gpu1:0.4 -> 50% GPU:0, 40% GPU:1, 10% CPU"
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable thinking/reasoning mode (model thinks before answering)"
                }),
                "context_size": ("INT", {
                    "default": 4096,
                    "min": 256,
                    "max": 1000000,
                    "step": 1,
                    "tooltip": "Context window size (total tokens for input + output). Higher values use more VRAM."
                }),
                "max_tokens": ("INT", {
                    "default": 4096,
                    "min": 1,
                    "max": 1000000,
                    "step": 1,
                    "tooltip": "Maximum tokens to generate"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Custom system prompt (leave empty for default)",
                    "tooltip": "Override the default system prompt"
                }),
                "use_model_default_sampling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use the model's default sampling parameters (overrides temperature, top_p, etc)"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Temperature for generation (higher = more creative, lower = more focused)"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Top-p (nucleus) sampling"
                }),
                "top_k": ("INT", {
                    "default": 40,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Top-k sampling (0 = disabled)"
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Min-p sampling threshold"
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Repetition penalty (1.0 = no penalty, higher = less repetition)"
                }),
            },
            "optional": {
                "image_1": ("IMAGE", {
                    "tooltip": "Optional image input for VLM models (image 1 of 5). Requires mmproj file."
                }),
                "image_2": ("IMAGE", {
                    "tooltip": "Optional image input for VLM models (image 2 of 5). Requires mmproj file."
                }),
                "image_3": ("IMAGE", {
                    "tooltip": "Optional image input for VLM models (image 3 of 5). Requires mmproj file."
                }),
                "image_4": ("IMAGE", {
                    "tooltip": "Optional image input for VLM models (image 4 of 5). Requires mmproj file."
                }),
                "image_5": ("IMAGE", {
                    "tooltip": "Optional image input for VLM models (image 5 of 5). Requires mmproj file."
                }),
            }
        }


    CATEGORY = "Prompt Manager"
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "create_options"

    def create_options(self, model, gpu_layers, enable_thinking, context_size, max_tokens, 
                    use_model_default_sampling, temperature, top_p, top_k, min_p, 
                    repeat_penalty, system_prompt="", 
                    image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):
        
        from .model_manager import get_matching_mmproj
        
        # Handle downloadable models (remove ⬇ prefix)
        if model.startswith("⬇ "):
            model = model[2:]
        
        # Auto-detect mmproj file
        mmproj = get_matching_mmproj(model)
        
        options = {
            "model": model,
            "mmproj": mmproj,  # Automatically detected
            "gpu_config": gpu_layers.strip() if gpu_layers.strip() else "auto",
            "enable_thinking": enable_thinking,
            "context_size": context_size,
            "max_tokens": max_tokens,
            "use_model_default_sampling": use_model_default_sampling,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
        }
        
        if system_prompt.strip():
            options["system_prompt"] = system_prompt
        
        # Collect images (filter out None values)
        images = []
        for img in [image_1, image_2, image_3, image_4, image_5]:
            if img is not None:
                images.append(img)
        
        if images:
            options["images"] = images
            if not mmproj:
                print(f"[Prompt Generator Options] Warning: Images provided but no matching mmproj file found for model: {model}")
        
        if mmproj:
            print(f"[Prompt Generator Options] Using mmproj: {mmproj}")
        
        return (options,)


NODE_CLASS_MAPPINGS = {
    "PromptGenOptions": PromptGenOptions
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGenOptions": "Prompt Generator Options"
}
