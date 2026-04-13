"""
ComfyUI Workflow Model Loader — Load MODEL/CLIP/VAE from workflow_data names.

Takes WORKFLOW_DATA and loads the referenced models. Since the actual loading
inputs are the model name strings (which rarely change), ComfyUI's execution
cache keeps this node from reloading on every run.
"""
import json
import os
import folder_paths
import comfy.sd
import torch


# ── Model extensions & helpers ───────────────────────────────────────
MODEL_EXTENSIONS = ['.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf', '.sft']


def _get_gguf_module():
    """Get the ComfyUI-GGUF module if the node is loaded, None otherwise."""
    try:
        import nodes as comfy_nodes
        gguf_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
        if gguf_cls is None:
            return None
        import sys
        mod = sys.modules.get(gguf_cls.__module__)
        if mod is None:
            return None
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
        raise RuntimeError("[WorkflowModelLoader] ComfyUI-GGUF module not available")

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
        raise RuntimeError(f"[WorkflowModelLoader] Could not detect model type of GGUF: {unet_path}")

    model = gguf["GGUFModelPatcher"].clone(model)
    model.patch_on_device = False
    return model


def _resolve_model_name(model_name):
    """
    Resolve a model name (basename, basename without extension, or relative path)
    to the actual relative path in ComfyUI's folder system.
    Returns (relative_path, folder_name) or (None, None) if not found.
    """
    if not model_name:
        return None, None

    model_name_clean = model_name.strip().replace('\\', '/')
    name_base = os.path.basename(model_name_clean).lower()

    name_no_ext = name_base
    for ext in MODEL_EXTENSIONS:
        if name_no_ext.endswith(ext):
            name_no_ext = name_no_ext[:-len(ext)]
            break

    for folder_name in ['checkpoints', 'diffusion_models', 'unet', 'unet_gguf']:
        try:
            file_list = folder_paths.get_filename_list(folder_name)
        except Exception:
            continue

        for f in file_list:
            f_normalized = f.replace('\\', '/')
            if f_normalized == model_name_clean:
                return f, folder_name

            f_base = os.path.basename(f_normalized).lower()
            if f_base == name_base:
                return f, folder_name

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
        print(f"[WorkflowModelLoader] Model not found: '{model_path}' — searched checkpoints, diffusion_models, unet, unet_gguf")
        return "NOT FOUND", None, None, None, display_name

    display_name = os.path.basename(resolved_path.replace('\\', '/'))
    is_gguf = resolved_path.lower().endswith('.gguf')

    # GGUF models
    if is_gguf:
        if _get_gguf_module() is None:
            print(f"[WorkflowModelLoader] GGUF model detected but ComfyUI-GGUF is not installed: {resolved_path}")
            return "NOT FOUND", None, None, None, display_name
        full_path = folder_paths.get_full_path(resolved_folder, resolved_path)
        if full_path and os.path.isfile(full_path):
            print(f"[WorkflowModelLoader] Loading GGUF model: {resolved_path}")
            model = _load_gguf_unet(full_path)
            return "GGUF", model, None, None, display_name
        return "NOT FOUND", None, None, None, display_name

    # Checkpoint
    if resolved_folder == 'checkpoints':
        ckpt_path = folder_paths.get_full_path("checkpoints", resolved_path)
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"[WorkflowModelLoader] Loading checkpoint: {resolved_path}")
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
        print(f"[WorkflowModelLoader] Loading diffusion/UNET model: {resolved_path}")
        model = comfy.sd.load_diffusion_model(full_path, model_options=model_options)
        return "Diffusion", model, None, None, display_name

    print(f"[WorkflowModelLoader] Model file not accessible: {resolved_path}")
    return "NOT FOUND", None, None, None, display_name


class WorkflowModelLoader:
    """
    Load MODEL_A, MODEL_B, CLIP, VAE from model names in WORKFLOW_DATA.
    Wire the outputs to WorkflowContext's optional model/clip/vae inputs,
    or directly to sampler/conditioning nodes.

    Because this node's effective inputs are model name strings (which rarely
    change), ComfyUI's execution cache prevents redundant reloads when only
    seeds or prompts change upstream.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_data": ("WORKFLOW_DATA", {
                    "tooltip": "Workflow data containing model name strings to load.",
                }),
            },
            "optional": {
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {
                    "default": "default",
                    "tooltip": "Weight precision for diffusion models. Ignored for checkpoints."
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model_a", "model_b", "clip", "vae")
    FUNCTION = "load_models"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = (
        "Load MODEL / CLIP / VAE from the model names in workflow_data. "
        "Wire outputs to WorkflowContext or directly to sampler/conditioning nodes. "
        "Cached by ComfyUI when model names don't change."
    )

    def load_models(self, workflow_data, weight_dtype="default"):
        # Accept both dict and JSON string
        if isinstance(workflow_data, str):
            try:
                wf = json.loads(workflow_data)
            except (json.JSONDecodeError, TypeError):
                wf = {}
        elif isinstance(workflow_data, dict):
            wf = dict(workflow_data)
        else:
            wf = {}

        model_a = None
        model_b = None
        clip = None
        vae = None

        # Load Model A (may also yield CLIP + VAE if it's a checkpoint)
        ma_name = (wf.get('model_a') or "").strip()
        if ma_name:
            _, model_a, clip_loaded, vae_loaded, _ = _load_single_model(ma_name, weight_dtype)
            clip = clip_loaded
            vae = vae_loaded

        # Load Model B
        mb_name = (wf.get('model_b') or "").strip()
        if mb_name:
            _, model_b, _, _, _ = _load_single_model(mb_name, weight_dtype)

        return (model_a, model_b, clip, vae)
