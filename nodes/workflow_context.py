"""
ComfyUI Workflow Context — Unpack/repack workflow_data into individual outputs.
Any connected optional input overrides the corresponding field before output.
Loads actual MODEL / CLIP / VAE objects from model names in workflow_data.
Analogous to rgthree's "Context Big" but for WORKFLOW_DATA.
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
        raise RuntimeError("[WorkflowContext] ComfyUI-GGUF module not available")

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
        raise RuntimeError(f"[WorkflowContext] Could not detect model type of GGUF: {unet_path}")

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
        print(f"[WorkflowContext] Model not found: '{model_path}' — searched checkpoints, diffusion_models, unet, unet_gguf")
        return "NOT FOUND", None, None, None, display_name

    display_name = os.path.basename(resolved_path.replace('\\', '/'))
    is_gguf = resolved_path.lower().endswith('.gguf')

    # GGUF models
    if is_gguf:
        if _get_gguf_module() is None:
            print(f"[WorkflowContext] GGUF model detected but ComfyUI-GGUF is not installed: {resolved_path}")
            return "NOT FOUND", None, None, None, display_name
        full_path = folder_paths.get_full_path(resolved_folder, resolved_path)
        if full_path and os.path.isfile(full_path):
            print(f"[WorkflowContext] Loading GGUF model: {resolved_path}")
            model = _load_gguf_unet(full_path)
            return "GGUF", model, None, None, display_name
        return "NOT FOUND", None, None, None, display_name

    # Checkpoint
    if resolved_folder == 'checkpoints':
        ckpt_path = folder_paths.get_full_path("checkpoints", resolved_path)
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"[WorkflowContext] Loading checkpoint: {resolved_path}")
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
        print(f"[WorkflowContext] Loading diffusion/UNET model: {resolved_path}")
        model = comfy.sd.load_diffusion_model(full_path, model_options=model_options)
        return "Diffusion", model, None, None, display_name

    print(f"[WorkflowContext] Model file not accessible: {resolved_path}")
    return "NOT FOUND", None, None, None, display_name


class WorkflowContext:
    """
    Unpack workflow_data into individual typed outputs.
    Connect optional inputs to override any field — the updated workflow_data
    is re-emitted so downstream nodes always see the latest values.
    Models are loaded and returned as MODEL / CLIP / VAE objects.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_data": ("WORKFLOW_DATA", {
                    "tooltip": "Workflow data to unpack. All fields are exposed as outputs.",
                }),
            },
            "optional": {
                "positive_prompt": ("STRING",  {"forceInput": True, "tooltip": "Override positive prompt"}),
                "negative_prompt": ("STRING",  {"forceInput": True, "tooltip": "Override negative prompt"}),
                "lora_stack_a":    ("LORA_STACK", {"tooltip": "Override LoRA stack A"}),
                "lora_stack_b":    ("LORA_STACK", {"tooltip": "Override LoRA stack B"}),
                "width":           ("INT",     {"forceInput": True, "tooltip": "Override width"}),
                "height":          ("INT",     {"forceInput": True, "tooltip": "Override height"}),
                "batch_size":      ("INT",     {"forceInput": True, "tooltip": "Override batch size"}),
                "length":          ("INT",     {"forceInput": True, "tooltip": "Override video length"}),
                "steps":           ("INT",     {"forceInput": True, "tooltip": "Override sampling steps"}),
                "cfg":             ("FLOAT",   {"forceInput": True, "tooltip": "Override CFG scale"}),
                "seed_a":          ("INT",     {"forceInput": True, "tooltip": "Override seed A"}),
                "seed_b":          ("INT",     {"forceInput": True, "tooltip": "Override seed B (dual-sampler)"}),
                "sampler_name":    ("STRING",  {"forceInput": True, "tooltip": "Override sampler name"}),
                "scheduler":       ("STRING",  {"forceInput": True, "tooltip": "Override scheduler"}),
                "denoise":         ("FLOAT",   {"forceInput": True, "tooltip": "Override denoise strength"}),
                "guidance":        ("FLOAT",   {"forceInput": True, "tooltip": "Override guidance value"}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {
                    "default": "default",
                    "tooltip": "Weight precision for diffusion models. Ignored for checkpoints."
                }),
            },
        }

    RETURN_TYPES = (
        "WORKFLOW_DATA",
        "MODEL", "MODEL", "CLIP", "VAE",
        "STRING", "STRING",
        "LORA_STACK", "LORA_STACK",
        "INT", "INT", "INT", "INT",
        "INT", "FLOAT", "INT", "INT",
        "STRING", "STRING", "FLOAT", "FLOAT",
        "STRING",
    )
    RETURN_NAMES = (
        "workflow_data",
        "model_a", "model_b", "clip", "vae",
        "positive_prompt", "negative_prompt",
        "lora_stack_a", "lora_stack_b",
        "width", "height", "batch_size", "length",
        "steps", "cfg", "seed_a", "seed_b",
        "sampler_name", "scheduler", "denoise", "guidance",
        "family",
    )
    FUNCTION = "unpack"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = (
        "Unpack workflow_data into individual outputs — loads models, unpacks prompts, "
        "sampler settings, resolution, and LoRA stacks. "
        "Connect any optional input to override that field before output."
    )

    # ------------------------------------------------------------------

    def unpack(self, workflow_data, weight_dtype="default", **kwargs):
        # Accept both dict and JSON string (Extractor outputs JSON string)
        if isinstance(workflow_data, str):
            try:
                wf = json.loads(workflow_data)
            except (json.JSONDecodeError, TypeError):
                wf = {}
        elif isinstance(workflow_data, dict):
            wf = dict(workflow_data)
        else:
            wf = {}
        sampler = dict(wf.get('sampler', {}))
        resolution = dict(wf.get('resolution', {}))

        # ── Apply top-level overrides ────────────────────────────────
        for key in ('positive_prompt', 'negative_prompt'):
            if kwargs.get(key) is not None:
                wf[key] = kwargs[key]

        # ── LoRA stack overrides ─────────────────────────────────────
        if kwargs.get('lora_stack_a') is not None:
            wf['loras_a'] = [
                {'name': n, 'model_strength': ms, 'clip_strength': cs}
                for n, ms, cs in kwargs['lora_stack_a']
            ]
        if kwargs.get('lora_stack_b') is not None:
            wf['loras_b'] = [
                {'name': n, 'model_strength': ms, 'clip_strength': cs}
                for n, ms, cs in kwargs['lora_stack_b']
            ]

        # ── Resolution overrides ─────────────────────────────────────
        for key in ('width', 'height', 'batch_size', 'length'):
            if kwargs.get(key) is not None:
                resolution[key] = kwargs[key]
        wf['resolution'] = resolution

        # ── Sampler overrides ────────────────────────────────────────
        for key in ('steps', 'cfg', 'seed_a', 'seed_b', 'sampler_name', 'scheduler', 'denoise', 'guidance'):
            if kwargs.get(key) is not None:
                sampler[key] = kwargs[key]
        wf['sampler'] = sampler

        # ── Load models ──────────────────────────────────────────────
        model_a_name = (wf.get('model_a') or "").strip()
        model_b_name = (wf.get('model_b') or "").strip()

        if model_a_name:
            _, model_a, clip, vae, _ = _load_single_model(model_a_name, weight_dtype)
        else:
            model_a = clip = vae = None

        if model_b_name:
            _, model_b, _, _, _ = _load_single_model(model_b_name, weight_dtype)
        else:
            model_b = None

        # ── Build lora stacks as tuples for LORA_STACK output ────────
        lora_stack_a = [
            (l['name'], l.get('model_strength', 1.0), l.get('clip_strength', 1.0))
            for l in wf.get('loras_a', []) if isinstance(l, dict) and l.get('name')
        ]
        lora_stack_b = [
            (l['name'], l.get('model_strength', 1.0), l.get('clip_strength', 1.0))
            for l in wf.get('loras_b', []) if isinstance(l, dict) and l.get('name')
        ]

        # ── Re-serialize workflow_data as JSON string (WORKFLOW_DATA convention) ─
        wf_out = json.dumps(wf, indent=2)

        # ── Extract all output values ────────────────────────────────
        return (
            wf_out,
            model_a, model_b, clip, vae,
            wf.get('positive_prompt', ''),
            wf.get('negative_prompt', ''),
            lora_stack_a,
            lora_stack_b,
            resolution.get('width', 512),
            resolution.get('height', 512),
            resolution.get('batch_size', 1),
            resolution.get('length', 0) or 0,
            sampler.get('steps', 20),
            sampler.get('cfg', 7.0),
            sampler.get('seed_a', sampler.get('seed', 0)),
            sampler.get('seed_b', 0) or 0,
            sampler.get('sampler_name', 'euler'),
            sampler.get('scheduler', 'normal'),
            sampler.get('denoise', 1.0),
            sampler.get('guidance', 0.0) or 0.0,
            wf.get('family', ''),
        )
