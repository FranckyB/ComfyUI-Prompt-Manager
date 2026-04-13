"""
ComfyUI Workflow Context — Unpack/repack workflow_data into individual outputs.
Any connected optional input overrides the corresponding field before output.
By default a pure data gateway (passes MODEL/CLIP/VAE through if connected).
Enable 'load_models' to fall back to loading from workflow_data names when
no model inputs are connected.
"""
import json

from .workflow_model_loader import _load_single_model


class WorkflowContext:
    """
    Unpack workflow_data into individual typed outputs.
    By default a pure data gateway — passes MODEL/CLIP/VAE through if
    connected. Enable 'load_models' to fall back to loading from
    workflow_data names when no model inputs are connected.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_data": ("WORKFLOW_DATA", {
                    "tooltip": "Workflow data to unpack. All fields are exposed as outputs.",
                }),
                "override_data": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {
                "model_a":         ("MODEL",   {"tooltip": "Pass-through Model A (from WorkflowModelLoader or other source)"}),
                "model_b":         ("MODEL",   {"tooltip": "Pass-through Model B (from WorkflowModelLoader or other source)"}),
                "clip":            ("CLIP",    {"tooltip": "Pass-through CLIP (from WorkflowModelLoader or other source)"}),
                "vae":             ("VAE",     {"tooltip": "Pass-through VAE (from WorkflowModelLoader or other source)"}),
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
                "load_models": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, load models from workflow_data names if no MODEL/CLIP/VAE inputs are connected."
                }),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {
                    "default": "default",
                    "tooltip": "Weight precision for diffusion models (only used when load_models is enabled)."
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
    )
    RETURN_NAMES = (
        "workflow_data",
        "model_a", "model_b", "clip", "vae",
        "positive_prompt", "negative_prompt",
        "lora_stack_a", "lora_stack_b",
        "width", "height", "batch_size", "length",
        "steps", "cfg", "seed_a", "seed_b",
        "sampler_name", "scheduler", "denoise", "guidance",
    )
    FUNCTION = "unpack"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = (
        "Unpack workflow_data into individual outputs — passes through MODEL/CLIP/VAE "
        "from connected inputs, or loads them from workflow_data names when "
        "'load_models' is enabled. Unpacks prompts, sampler settings, resolution, "
        "and LoRA stacks."
    )

    # ------------------------------------------------------------------

    def unpack(self, workflow_data, override_data="{}",
               load_models=False, weight_dtype="default", **kwargs):
        # Accept both dict and JSON string (backward compat)
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

        # ── Parse JS override_data blob ──────────────────────────────
        try:
            overrides = json.loads(override_data) if override_data else {}
        except (json.JSONDecodeError, TypeError):
            overrides = {}

        # Apply dropdown overrides from JS → update workflow_data dict
        if overrides.get('_family'):
            wf['family'] = overrides['_family']
        if overrides.get('model_a'):
            wf['model_a'] = overrides['model_a']
        if overrides.get('model_b'):
            wf['model_b'] = overrides['model_b']
        if overrides.get('vae'):
            wf['vae'] = overrides['vae']
        if overrides.get('clip_names'):
            wf['clip'] = overrides['clip_names']

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

        # ── Pass-through MODEL / CLIP / VAE ──────────────────────────
        # Whatever is connected gets forwarded. If load_models is enabled
        # and nothing is connected, fall back to loading from name strings.
        model_a = kwargs.get('model_a')
        model_b = kwargs.get('model_b')
        clip = kwargs.get('clip')
        vae = kwargs.get('vae')

        if load_models:
            if model_a is None:
                ma_name = (wf.get('model_a') or "").strip()
                if ma_name:
                    _, model_a, clip_loaded, vae_loaded, _ = _load_single_model(ma_name, weight_dtype)
                    if clip is None:
                        clip = clip_loaded
                    if vae is None:
                        vae = vae_loaded

            if model_b is None:
                mb_name = (wf.get('model_b') or "").strip()
                if mb_name:
                    _, model_b, _, _, _ = _load_single_model(mb_name, weight_dtype)

        # Store in dict so downstream nodes see them
        if model_a is not None:
            wf['MODEL_A'] = model_a
        if model_b is not None:
            wf['MODEL_B'] = model_b
        if clip is not None:
            wf['CLIP'] = clip
        if vae is not None:
            wf['VAE'] = vae

        # ── Build lora stacks as tuples for LORA_STACK output ────────
        lora_stack_a = [
            (l['name'], l.get('model_strength', 1.0), l.get('clip_strength', 1.0))
            for l in wf.get('loras_a', []) if isinstance(l, dict) and l.get('name')
        ]
        lora_stack_b = [
            (l['name'], l.get('model_strength', 1.0), l.get('clip_strength', 1.0))
            for l in wf.get('loras_b', []) if isinstance(l, dict) and l.get('name')
        ]

        # ── Extract all output values ────────────────────────────────
        return (
            wf,
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
        )
