"""
ComfyUI Model Loader - Load checkpoint, diffusion, UNET, or GGUF model from workflow_data.
Automatically determines the model type based on available folders.
Supports GGUF models when ComfyUI-GGUF custom node is installed.
Dynamically shows Model B outputs when two models are found in workflow_data.
"""
import json
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


# Known model file extensions for stripping during name matching
MODEL_EXTENSIONS = ['.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf', '.sft']


def _resolve_model_name(model_name):
    """
    Resolve a model name (which may be a basename, basename without extension,
    or full relative path) to the actual relative path in ComfyUI's folder system.
    Returns (relative_path, folder_name) or (None, None) if not found.
    """
    if not model_name:
        return None, None

    model_name_clean = model_name.strip().replace('\\', '/')
    name_base = os.path.basename(model_name_clean).lower()

    # Strip extension for extensionless matching
    name_no_ext = name_base
    for ext in MODEL_EXTENSIONS:
        if name_no_ext.endswith(ext):
            name_no_ext = name_no_ext[:-len(ext)]
            break

    # Build lookup in a single pass per folder
    for folder_name in ['checkpoints', 'diffusion_models', 'unet', 'unet_gguf']:
        try:
            file_list = folder_paths.get_filename_list(folder_name)
        except Exception:
            continue

        for f in file_list:
            f_normalized = f.replace('\\', '/')

            # Exact relative path match
            if f_normalized == model_name_clean:
                return f, folder_name

            f_base = os.path.basename(f_normalized).lower()

            # Basename match (with extension)
            if f_base == name_base:
                return f, folder_name

            # Basename match without extension
            f_no_ext = f_base
            for ext in MODEL_EXTENSIONS:
                if f_no_ext.endswith(ext):
                    f_no_ext = f_no_ext[:-len(ext)]
                    break
            if f_no_ext == name_no_ext:
                return f, folder_name

    return None, None


def _build_model_options(weight_dtype):
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


def _load_single_model(model_path, weight_dtype="default"):
    """
    Load one model by name/path. Returns (model_type, model, clip, vae, display_name).
    model_type is "Checkpoint", "Diffusion", "GGUF", or "NOT FOUND".
    """
    model_path = (model_path or "").strip()
    if not model_path:
        return "NOT FOUND", None, None, None, "(empty)"

    display_name = os.path.basename(model_path.replace('\\', '/'))
    resolved_path, resolved_folder = _resolve_model_name(model_path)

    if resolved_path is None:
        print(f"[ModelLoader] Model not found: '{model_path}' — searched checkpoints, diffusion_models, unet, unet_gguf")
        return "NOT FOUND", None, None, None, display_name

    display_name = os.path.basename(resolved_path.replace('\\', '/'))
    is_gguf = resolved_path.lower().endswith('.gguf')

    # GGUF models — use ComfyUI-GGUF if active
    if is_gguf:
        if _get_gguf_module() is None:
            print(f"[ModelLoader] GGUF model detected but ComfyUI-GGUF is not installed: {resolved_path}")
            return "NOT FOUND", None, None, None, display_name

        full_path = folder_paths.get_full_path(resolved_folder, resolved_path)
        if full_path and os.path.isfile(full_path):
            print(f"[ModelLoader] Loading GGUF model via ComfyUI-GGUF: {resolved_path} (from {resolved_folder})")
            model = _load_gguf_unet(full_path)
            return "GGUF", model, None, None, display_name

        return "NOT FOUND", None, None, None, display_name

    # Checkpoint
    if resolved_folder == 'checkpoints':
        ckpt_path = folder_paths.get_full_path("checkpoints", resolved_path)
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"[ModelLoader] Loading checkpoint: {resolved_path}")
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
            return "Checkpoint", out[0], out[1], out[2], display_name

    # Diffusion / UNET
    model_options = _build_model_options(weight_dtype)
    full_path = folder_paths.get_full_path(resolved_folder, resolved_path)
    if full_path and os.path.isfile(full_path):
        print(f"[ModelLoader] Loading diffusion/UNET model: {resolved_path} (from {resolved_folder})")
        model = comfy.sd.load_diffusion_model(full_path, model_options=model_options)
        return "Diffusion", model, None, None, display_name

    print(f"[ModelLoader] Model file not accessible after resolution: {resolved_path}")
    return "NOT FOUND", None, None, None, display_name


class PromptModelLoader:
    """
    Load checkpoint or diffusion models from workflow_data (WORKFLOW_DATA).
    Reads model_a (and optionally model_b) from the workflow_data JSON.
    Dynamically shows Model B outputs when two models are present.
    Auto-detects checkpoint vs diffusion model type.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_data": ("WORKFLOW_DATA", {
                    "tooltip": "Workflow data from Prompt Extractor or Workflow Generator containing model paths."
                }),
            },
            "optional": {
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {
                    "default": "default",
                    "tooltip": "Weight precision for diffusion models. Ignored for checkpoints."
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model_a", "clip_a", "vae_a", "model_b", "clip_b", "vae_b")
    FUNCTION = "load_model"
    CATEGORY = "Prompt Manager"
    OUTPUT_NODE = True
    DESCRIPTION = "Load models from workflow_data. Reads model_a and model_b paths. Auto-detects checkpoint vs diffusion type. Shows Model B outputs when two models are present. CLIP and VAE are None for non-checkpoint models."

    def load_model(self, workflow_data, weight_dtype="default"):
        # Parse workflow_data JSON
        wf = {}
        if isinstance(workflow_data, str) and workflow_data.strip():
            try:
                wf = json.loads(workflow_data)
            except (json.JSONDecodeError, TypeError):
                pass

        model_a_name = (wf.get('model_a') or "").strip()
        model_b_name = (wf.get('model_b') or "").strip()
        has_b = bool(model_b_name)

        # Load model A
        if model_a_name:
            type_a, model_a, clip_a, vae_a, display_a = _load_single_model(model_a_name, weight_dtype)
        else:
            type_a, model_a, clip_a, vae_a, display_a = "NOT FOUND", None, None, None, "(no model in workflow_data)"

        # Load model B (only if present)
        if has_b:
            type_b, model_b, clip_b, vae_b, display_b = _load_single_model(model_b_name, weight_dtype)
        else:
            type_b, model_b, clip_b, vae_b, display_b = None, None, None, None, ""

        # Build UI info for JavaScript
        ui_info = {
            "model_type_a": type_a,
            "model_name_a": display_a,
            "has_model_b": has_b,
        }
        if has_b:
            ui_info["model_type_b"] = type_b
            ui_info["model_name_b"] = display_b

        return {
            "ui": {"model_info": [ui_info]},
            "result": (model_a, clip_a, vae_a, model_b, clip_b, vae_b)
        }
