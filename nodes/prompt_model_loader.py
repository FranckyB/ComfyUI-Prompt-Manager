"""
ComfyUI Model Loader - Load checkpoint or diffusion model from a path string.
Automatically determines the model type based on available folders.
"""
import os
import folder_paths
import comfy.sd
import torch


class PromptModelLoader:
    """
    Load a checkpoint or diffusion model from a model name/path string.
    Automatically determines whether the model is a checkpoint (returns MODEL, CLIP, VAE)
    or a diffusion model (returns MODEL only, CLIP and VAE are None).
    Designed to accept output from the Prompt Extractor node.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": "Model name or path (e.g. from Prompt Extractor). Auto-detects checkpoint vs diffusion model."
                }),
            },
            "optional": {
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {
                    "default": "default",
                    "tooltip": "Weight precision for diffusion models. Ignored for checkpoints."
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_model"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Load a checkpoint or diffusion model from a path string. Auto-detects model type. CLIP and VAE are None for diffusion models."

    def load_model(self, model_path, weight_dtype="default"):
        model_path = model_path.strip()
        if not model_path:
            raise ValueError("[ModelLoader] No model path provided")

        # Try checkpoint first, then diffusion_models
        ckpt_path = folder_paths.get_full_path("checkpoints", model_path)
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"[ModelLoader] Loading checkpoint: {model_path}")
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
            return out[:3]

        # Try diffusion_models
        diff_path = folder_paths.get_full_path("diffusion_models", model_path)
        if diff_path and os.path.isfile(diff_path):
            print(f"[ModelLoader] Loading diffusion model: {model_path}")
            model_options = {}
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2

            model = comfy.sd.load_diffusion_model(diff_path, model_options=model_options)
            return (model, None, None)

        # Try unet (some setups use this folder)
        unet_path = folder_paths.get_full_path("unet", model_path)
        if unet_path and os.path.isfile(unet_path):
            print(f"[ModelLoader] Loading UNET model: {model_path}")
            model_options = {}
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2

            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
            return (model, None, None)

        raise FileNotFoundError(
            f"[ModelLoader] Model not found: '{model_path}'. \n"
            f"Searched in checkpoints and diffusion_models folders."
        )
