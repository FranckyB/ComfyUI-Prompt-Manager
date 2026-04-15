import os
import sys
import inspect

import folder_paths
import nodes
import comfy.sd
import comfy.utils
import torch

from comfy_execution.graph import ExecutionBlocker


# =========================================================
# GGUF support
# =========================================================

def _get_gguf_module():
    """Get the ComfyUI-GGUF module if available, otherwise None."""
    try:
        gguf_cls = nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
        if gguf_cls is None:
            return None

        mod = sys.modules.get(gguf_cls.__module__)
        if mod is None:
            return None

        package = gguf_cls.__module__.rsplit(".", 1)[0]
        ops_mod = sys.modules.get(f"{package}.ops")
        loader_mod = sys.modules.get(f"{package}.loader")

        if ops_mod is None or loader_mod is None:
            return None

        return {
            "GGMLOps": ops_mod.GGMLOps,
            "gguf_sd_loader": loader_mod.gguf_sd_loader,
            "GGUFModelPatcher": mod.GGUFModelPatcher,
        }
    except Exception:
        return None


def _load_gguf_unet(unet_path):
    """Load a GGUF diffusion model and return MODEL."""
    gguf = _get_gguf_module()
    if gguf is None:
        raise RuntimeError(
            "[WorkflowGetModel] This workflow requires ComfyUI-GGUF, "
            "but UnetLoaderGGUF is not available."
        )

    ops = gguf["GGMLOps"]()
    sd, extra = gguf["gguf_sd_loader"](unet_path)

    kwargs = {}
    valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
    if "metadata" in valid_params:
        kwargs["metadata"] = extra.get("metadata", {})

    model = comfy.sd.load_diffusion_model_state_dict(
        sd,
        model_options={"custom_operations": ops},
        **kwargs,
    )

    if model is None:
        raise RuntimeError(
            f"[WorkflowGetModel] Could not detect model type of GGUF: {unet_path}"
        )

    model = gguf["GGUFModelPatcher"].clone(model)
    model.patch_on_device = False
    return model


# =========================================================
# Generic loading helpers
# =========================================================

MODEL_EXTENSIONS = ['.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf', '.sft']

WEIGHT_DTYPE_OPTIONS = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]


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


def _resolve_path(name, folder_key_candidates):
    """
    Resolve a model filename against one or more Comfy folder keys.

    Tries get_full_path first (exact / relative path), then falls back to
    scanning get_filename_list and matching by basename (with or without
    extension).  This handles models stored in subdirectories.

    Returns full path or None.
    """
    if not name:
        return None

    raw = str(name).strip()
    if not raw:
        return None

    normalized = raw.replace("\\", "/")
    base = os.path.basename(normalized).lower()

    # Strip extension for fuzzy matching
    base_no_ext = base
    for ext in MODEL_EXTENSIONS:
        if base_no_ext.endswith(ext):
            base_no_ext = base_no_ext[:-len(ext)]
            break

    # Pass 1: try get_full_path (works when name is already a valid relative path)
    for candidate in (raw, os.path.basename(normalized)):
        for key in folder_key_candidates:
            try:
                path = folder_paths.get_full_path(key, candidate)
                if path:
                    return path
            except Exception:
                pass

    # Pass 2: scan filename lists and match by basename (handles subdirectories)
    for key in folder_key_candidates:
        try:
            file_list = folder_paths.get_filename_list(key)
        except Exception:
            continue

        for f in file_list:
            f_normalized = f.replace("\\", "/")

            # Exact relative-path match
            if f_normalized == normalized:
                return folder_paths.get_full_path(key, f)

            f_base = os.path.basename(f_normalized).lower()

            # Basename match (with extension)
            if f_base == base:
                return folder_paths.get_full_path(key, f)

            # Basename match (without extension)
            f_no_ext = f_base
            for ext in MODEL_EXTENSIONS:
                if f_no_ext.endswith(ext):
                    f_no_ext = f_no_ext[:-len(ext)]
                    break
            if f_no_ext == base_no_ext:
                return folder_paths.get_full_path(key, f)

    return None


def _load_checkpoint_by_name(ckpt_name):
    """
    Load checkpoint and return (MODEL, CLIP, VAE).
    Uses standard checkpoint loading semantics.
    """
    ckpt_path = _resolve_path(ckpt_name, ["checkpoints", "diffusion_models", "unet"])
    if not ckpt_path:
        raise FileNotFoundError(f"[WorkflowGetModel] Checkpoint not found: {ckpt_name}")

    out = comfy.sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )

    model, clip, vae = out[:3]
    return model, clip, vae


def _load_diffusion_model_by_name(model_name, weight_dtype="default"):
    """
    Load a diffusion/UNet model and return MODEL.
    If the name ends with .gguf, use the GGUF loader path.
    """
    model_path = _resolve_path(model_name, ["diffusion_models", "unet", "unet_gguf", "checkpoints"])
    if not model_path:
        raise FileNotFoundError(f"[WorkflowGetModel] Diffusion model not found: {model_name}")

    if model_name.lower().endswith(".gguf"):
        return _load_gguf_unet(model_path)

    model_options = _build_model_options(weight_dtype)
    model = comfy.sd.load_diffusion_model(model_path, model_options=model_options)
    if model is None:
        raise RuntimeError(f"[WorkflowGetModel] Could not load diffusion model: {model_name}")
    return model


def _load_vae_by_name(vae_name):
    """
    Load a VAE and return VAE.
    """
    vae_path = _resolve_path(vae_name, ["vae", "checkpoints"])
    if not vae_path:
        raise FileNotFoundError(f"[WorkflowGetModel] VAE not found: {vae_name}")

    sd = comfy.utils.load_torch_file(vae_path)
    return comfy.sd.VAE(sd=sd)


def _load_clip_by_names(clip_names, clip_type=""):
    """
    Load CLIP from one or more filenames.
    clip_names should be a list[str].
    """
    if not clip_names:
        return None

    clip_paths = []
    for name in clip_names:
        clip_path = _resolve_path(name, ["text_encoders", "clip", "checkpoints"])
        if not clip_path:
            raise FileNotFoundError(f"[WorkflowGetModel] CLIP not found: {name}")
        clip_paths.append(clip_path)

    clip_type = (clip_type or "").strip().lower()

    # Map a few common values. If unknown, let Comfy infer/default.
    clip_type_map = {
        "stable_diffusion": comfy.sd.CLIPType.STABLE_DIFFUSION,
        "sd": comfy.sd.CLIPType.STABLE_DIFFUSION,
        "sdxl": comfy.sd.CLIPType.STABLE_DIFFUSION,
        "wan": comfy.sd.CLIPType.WAN,
        "flux": comfy.sd.CLIPType.FLUX,
        "hunyuan_video": comfy.sd.CLIPType.HUNYUAN_VIDEO,
    }

    load_kwargs = {
        "ckpt_paths": clip_paths,
        "embedding_directory": folder_paths.get_folder_paths("embeddings"),
    }

    if clip_type in clip_type_map:
        load_kwargs["clip_type"] = clip_type_map[clip_type]

    return comfy.sd.load_clip(**load_kwargs)


# =========================================================
# Node
# =========================================================

class WorkflowGetModel:
    CATEGORY = "WorkflowBuilder"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "get_model"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_data": ("WORKFLOW_DATA", {
                    "tooltip": "Workflow data containing model name strings to load.",
                    "forceInput": True,
                }),
                "model_key": (["model_a", "model_b"], {
                    "default": "model_a",
                    "tooltip": "Which model slot to load from workflow_data.",
                }),
                "weight_dtype": (WEIGHT_DTYPE_OPTIONS, {
                    "default": "default",
                    "tooltip": "Weight precision override for diffusion/UNet models. Ignored for checkpoints.",
                }),
            }
        }

    def get_model(self, workflow_data, model_key="model_a", weight_dtype="default"):
        spec = self._normalize_spec(workflow_data)
        self._model_key = model_key
        self._weight_dtype = weight_dtype
        loader_type = spec.get("loader_type", "").lower()

        if loader_type == "checkpoint":
            return self._load_checkpoint_mode(spec)
        elif loader_type in ("unet", "diffusion"):
            return self._load_unet_mode(spec)
        else:
            raise ValueError(
                f"[WorkflowGetModel] Unsupported loader_type: {spec.get('loader_type')}"
            )

    @staticmethod
    def _is_placeholder(value):
        """True for display-only strings like '(from checkpoint)', '(none)', etc."""
        return isinstance(value, str) and value.startswith("(")

    def _normalize_spec(self, spec):
        spec = dict(spec or {})
        spec["model_a"] = (spec.get("model_a") or "").strip()
        spec["model_b"] = (spec.get("model_b") or "").strip()
        spec["loader_type"] = (spec.get("loader_type") or "").strip()
        spec["clip_type"] = (spec.get("clip_type") or "").strip()

        # VAE / CLIP: placeholders like '(from checkpoint)' mean "use what
        # the checkpoint already provides" — normalise to empty.
        vae = (spec.get("vae") or "").strip()
        spec["vae"] = "" if self._is_placeholder(vae) else vae

        clip = spec.get("clip") or []
        spec["clip"] = [c for c in clip if c and not self._is_placeholder(c)]

        return spec

    def _load_checkpoint_mode(self, spec):
        ckpt_name = spec.get(self._model_key, "")
        if not ckpt_name:
            return (
                ExecutionBlocker(f"Missing checkpoint in '{self._model_key}'"),
                ExecutionBlocker("No CLIP available"),
                ExecutionBlocker("No VAE available"),
            )

        model, clip, vae = _load_checkpoint_by_name(ckpt_name)

        clip_override = self._maybe_load_clip_override(spec)
        if clip_override is not None:
            clip = clip_override

        vae_override = self._maybe_load_vae_override(spec)
        if vae_override is not None:
            vae = vae_override

        if clip is None:
            clip = ExecutionBlocker(f"No CLIP available for checkpoint '{ckpt_name}'")
        if vae is None:
            vae = ExecutionBlocker(f"No VAE available for checkpoint '{ckpt_name}'")

        return (model, clip, vae)

    def _load_unet_mode(self, spec):
        model_name = spec.get(self._model_key, "")
        if not model_name:
            return (
                ExecutionBlocker(f"Missing model in '{self._model_key}'"),
                ExecutionBlocker("No CLIP available"),
                ExecutionBlocker("No VAE available"),
            )

        model = _load_diffusion_model_by_name(model_name, self._weight_dtype)

        clip = self._maybe_load_clip_override(spec)
        vae = self._maybe_load_vae_override(spec)

        if clip is None:
            clip = ExecutionBlocker(
                f"No explicit CLIP provided for diffusion model '{model_name}'"
            )
        if vae is None:
            vae = ExecutionBlocker(
                f"No explicit VAE provided for diffusion model '{model_name}'"
            )

        return (model, clip, vae)

    def _maybe_load_clip_override(self, spec):
        clip_list = spec.get("clip") or []
        if len(clip_list) == 0:
            return None
        return _load_clip_by_names(clip_list, spec.get("clip_type", ""))

    def _maybe_load_vae_override(self, spec):
        vae_name = spec.get("vae", "")
        if not vae_name:
            return None
        return _load_vae_by_name(vae_name)


NODE_CLASS_MAPPINGS = {
    "WorkflowGetModel": WorkflowGetModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WorkflowGetModel": "Workflow Get Model",
}
