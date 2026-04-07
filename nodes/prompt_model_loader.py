"""
ComfyUI Model Loader - Load checkpoint, diffusion, UNET, or GGUF model from a path string.
Automatically determines the model type based on available folders.
Supports GGUF models when ComfyUI-GGUF custom node is installed.
"""
import os
import logging
import folder_paths
import comfy.sd
import torch


def _get_gguf_module():
    """Get the ComfyUI-GGUF module if the node is loaded, None otherwise."""
    try:
        import nodes as comfy_nodes
        gguf_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
        if gguf_cls is None:
            return None
        # Get the module the class came from
        import sys
        mod = sys.modules.get(gguf_cls.__module__)
        if mod is None:
            return None
        # The ops/loader are siblings in the same package
        package = gguf_cls.__module__.rsplit('.', 1)[0]
        ops_mod = sys.modules.get(f"{package}.ops")
        loader_mod = sys.modules.get(f"{package}.loader")
        if ops_mod and loader_mod:
            return {
                "GGMLOps": ops_mod.GGMLOps,
                "gguf_sd_loader": loader_mod.gguf_sd_loader,
                "GGUFModelPatcher": mod.GGUFModelPatcher,
            }
        return None
    except Exception:
        return None


def _load_gguf_unet(unet_path):
    """Load a GGUF model using ComfyUI-GGUF's loader. Returns MODEL."""
    import inspect

    gguf = _get_gguf_module()
    if gguf is None:
        raise RuntimeError("[ModelLoader] ComfyUI-GGUF module not available")

    ops = gguf["GGMLOps"]()
    sd, extra = gguf["gguf_sd_loader"](unet_path)

    kwargs = {}
    valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
    if "metadata" in valid_params:
        kwargs["metadata"] = extra.get("metadata", {})

    model = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options={"custom_operations": ops}, **kwargs,
    )
    if model is None:
        raise RuntimeError(f"[ModelLoader] Could not detect model type of GGUF: {unet_path}")

    model = gguf["GGUFModelPatcher"].clone(model)
    model.patch_on_device = False
    return model


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
    OUTPUT_NODE = True
    DESCRIPTION = "Load a checkpoint, diffusion, UNET, or GGUF model from a path string. Auto-detects model type. Supports GGUF when ComfyUI-GGUF is installed. CLIP and VAE are None for non-checkpoint models."

    def _build_model_options(self, weight_dtype):
        """Build model_options dict from weight_dtype selection."""
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        return model_options

    def load_model(self, model_path, weight_dtype="default"):
        model_path = model_path.strip()
        if not model_path:
            return {
                "ui": {"model_info": [{"model_type": "NOT FOUND", "model_name": "(no model path provided)"}]},
                "result": (None, None, None)
            }

        # Extract display name (basename without directory)
        display_name = os.path.basename(model_path.replace('\\', '/'))
        is_gguf = model_path.lower().endswith('.gguf')

        def _result(model_type, model, clip, vae):
            return {
                "ui": {"model_info": [{"model_type": model_type, "model_name": display_name}]},
                "result": (model, clip, vae)
            }

        # GGUF models — use ComfyUI-GGUF if active
        if is_gguf:
            if _get_gguf_module() is None:
                print(f"[ModelLoader] GGUF model detected but ComfyUI-GGUF is not installed: {model_path}")
                return _result("NOT FOUND", None, None, None)

            # Search unet_gguf (registered by ComfyUI-GGUF), unet, and diffusion_models folders
            for folder_name in ['unet_gguf', 'unet', 'diffusion_models']:
                try:
                    full_path = folder_paths.get_full_path(folder_name, model_path)
                    if full_path and os.path.isfile(full_path):
                        print(f"[ModelLoader] Loading GGUF model via ComfyUI-GGUF: {model_path} (from {folder_name})")
                        model = _load_gguf_unet(full_path)
                        return _result("GGUF", model, None, None)
                except Exception:
                    continue

            print(f"[ModelLoader] GGUF model not found: '{model_path}' — searched unet_gguf, unet, diffusion_models")
            return _result("NOT FOUND", None, None, None)

        # Try checkpoint first
        ckpt_path = folder_paths.get_full_path("checkpoints", model_path)
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"[ModelLoader] Loading checkpoint: {model_path}")
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
            return _result("Checkpoint", out[0], out[1], out[2])

        # Try diffusion_models and unet folders
        model_options = self._build_model_options(weight_dtype)
        for folder_name in ['diffusion_models', 'unet']:
            full_path = folder_paths.get_full_path(folder_name, model_path)
            if full_path and os.path.isfile(full_path):
                print(f"[ModelLoader] Loading diffusion/UNET model: {model_path} (from {folder_name})")
                model = comfy.sd.load_diffusion_model(full_path, model_options=model_options)
                return _result("Diffusion", model, None, None)

        print(f"[ModelLoader] Model not found: '{model_path}' — searched checkpoints, diffusion_models, unet")
        return _result("NOT FOUND", None, None, None)
