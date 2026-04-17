"""
ComfyUI Workflow Bridge - Pure passthrough node.
Unpacks workflow_data into individual typed outputs.
Any connected optional input overrides the corresponding field before output.
Models (MODEL/CLIP/VAE) are passed through only - use WorkflowModelLoader for loading.
"""
import json
import os
import comfy.samplers
from ..py.lora_utils import resolve_lora_path


def _get_live_ksampler_combo_types():
    """Mirror the active KSampler enum sockets to avoid stale type mismatches."""
    default_samplers = comfy.samplers.KSampler.SAMPLERS
    default_schedulers = comfy.samplers.KSampler.SCHEDULERS
    try:
        from nodes import KSampler as CoreKSampler
        required = CoreKSampler.INPUT_TYPES().get("required", {})
        samplers = required.get("sampler_name", (default_samplers,))[0]
        schedulers = required.get("scheduler", (default_schedulers,))[0]
        return samplers, schedulers
    except Exception:
        return default_samplers, default_schedulers


_LIVE_SAMPLERS, _LIVE_SCHEDULERS = _get_live_ksampler_combo_types()


class WorkflowBridge:
    """
    Pure passthrough bridge node.
    Unpacks workflow_data into individual typed outputs and forwards
    any connected MODEL/CLIP/VAE inputs. No model loading.
    """

    @classmethod
    def _sync_combo_types(cls):
        samplers, schedulers = _get_live_ksampler_combo_types()
        cls._SAMPLER_ENUM = samplers
        cls._SCHEDULER_ENUM = schedulers
        cls.RETURN_TYPES = (
            "WORKFLOW_DATA",
            "MODEL", "MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "VAE", "LATENT", "IMAGE",
            "INT", "INT", "INT", "INT", "FLOAT",
            cls._SAMPLER_ENUM, cls._SCHEDULER_ENUM,
            "STRING", "STRING",
            "LORA_STACK", "LORA_STACK",
            "INT", "INT", "INT", "INT",
            "STRING",
        )

    @classmethod
    def INPUT_TYPES(cls):
        cls._sync_combo_types()
        return {
            "required": {
                "workflow_data": ("WORKFLOW_DATA", {
                    "tooltip": "Workflow data to unpack. All fields are exposed as outputs.",
                }),
            },
            "optional": {
                "model_a":         ("MODEL",        {"tooltip": "Pass-through Model A"}),
                "model_b":         ("MODEL",        {"tooltip": "Pass-through Model B"}),
                "clip":            ("CLIP",         {"tooltip": "Pass-through CLIP"}),
                "positive":        ("CONDITIONING", {"tooltip": "Pass-through positive conditioning"}),
                "negative":        ("CONDITIONING", {"tooltip": "Pass-through negative conditioning"}),
                "vae":             ("VAE",          {"tooltip": "Pass-through VAE"}),
                "latent":          ("LATENT",       {"tooltip": "Pass-through latent"}),
                "image":           ("IMAGE",        {"tooltip": "Pass-through image"}),
                "seed_a":          ("INT",     {"forceInput": True, "tooltip": "Override seed A"}),
                "seed_b":          ("INT",     {"forceInput": True, "tooltip": "Override seed B (dual-sampler)"}),
                "steps_a":         ("INT",     {"forceInput": True, "tooltip": "Override sampling steps A"}),
                "steps_b":         ("INT",     {"forceInput": True, "tooltip": "Override sampling steps B (dual-sampler)"}),
                "cfg":             ("FLOAT",   {"forceInput": True, "tooltip": "Override CFG scale"}),
                "sampler_name":    (cls._SAMPLER_ENUM, {"forceInput": True, "tooltip": "Override sampler name"}),
                "scheduler":       (cls._SCHEDULER_ENUM, {"forceInput": True, "tooltip": "Override scheduler"}),
                "pos_text":        ("STRING",  {"forceInput": True, "tooltip": "Override positive prompt"}),
                "neg_text":        ("STRING",  {"forceInput": True, "tooltip": "Override negative prompt"}),
                "lora_stack_a":    ("LORA_STACK", {"tooltip": "Override LoRA stack A"}),
                "lora_stack_b":    ("LORA_STACK", {"tooltip": "Override LoRA stack B"}),
                "width":           ("INT",     {"forceInput": True, "tooltip": "Override width"}),
                "height":          ("INT",     {"forceInput": True, "tooltip": "Override height"}),
                "batch_size":      ("INT",     {"forceInput": True, "tooltip": "Override batch size"}),
                "length":          ("INT",     {"forceInput": True, "tooltip": "Override video length"}),
            },
        }

    RETURN_TYPES = (
        "WORKFLOW_DATA",
        "MODEL", "MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "VAE", "LATENT", "IMAGE",
        "INT", "INT", "INT", "INT", "FLOAT",
        _LIVE_SAMPLERS, _LIVE_SCHEDULERS,
        "STRING", "STRING",
        "LORA_STACK", "LORA_STACK",
        "INT", "INT", "INT", "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "workflow_data",
        "model_a", "model_b", "clip", "positive", "negative", "vae", "latent", "image",
        "seed_a", "seed_b", "steps_a", "steps_b", "cfg",
        "sampler_name", "scheduler",
        "pos_text", "neg_text",
        "lora_stack_a", "lora_stack_b",
        "width", "height", "batch_size", "length",
        "model_name",
    )
    FUNCTION = "unpack"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = (
        "Pure passthrough bridge node. Unpacks workflow_data into individual "
        "typed outputs. Models are forwarded from connected inputs only."
    )

    # ------------------------------------------------------------------

    def unpack(self, workflow_data, **kwargs):
        def _short_display_name(name):
            raw = str(name or "").strip()
            if not raw:
                return ""
            base = os.path.basename(raw.replace("\\", "/"))
            return os.path.splitext(base)[0]

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

        # -- Apply top-level overrides -------------------------------
        pos_text = kwargs.get('pos_text')
        if pos_text is not None:
            wf['positive_prompt'] = pos_text

        neg_text = kwargs.get('neg_text')
        if neg_text is not None:
            wf['negative_prompt'] = neg_text

        # -- LoRA stack overrides ------------------------------------
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

        # -- Resolution overrides ------------------------------------
        for key in ('width', 'height', 'batch_size', 'length'):
            if kwargs.get(key) is not None:
                resolution[key] = kwargs[key]
        wf['resolution'] = resolution

        # -- Sampler overrides ---------------------------------------
        for key in ('steps_a', 'steps_b', 'cfg', 'seed_a', 'seed_b', 'sampler_name', 'scheduler'):
            if kwargs.get(key) is not None:
                sampler[key] = kwargs[key]
        wf['sampler'] = sampler

        # -- Pass-through MODEL / CLIP / VAE -------------------------
        # Priority: explicit connected inputs > embedded objects in workflow_data.
        model_a = kwargs.get('model_a')
        if model_a is None:
            model_a = wf.get('MODEL_A')

        model_b = kwargs.get('model_b')
        if model_b is None:
            model_b = wf.get('MODEL_B')

        clip = kwargs.get('clip')
        if clip is None:
            clip = wf.get('CLIP')

        vae = kwargs.get('vae')
        if vae is None:
            vae = wf.get('VAE')

        positive = kwargs.get('positive')
        if positive is None:
            positive = wf.get('POSITIVE')

        negative = kwargs.get('negative')
        if negative is None:
            negative = wf.get('NEGATIVE')

        latent = kwargs.get('latent')
        if latent is None:
            latent = wf.get('LATENT')

        image = kwargs.get('image')
        if image is None:
            image = wf.get('IMAGE')

        if model_a is not None:
            wf['MODEL_A'] = model_a
        if model_b is not None:
            wf['MODEL_B'] = model_b
        if clip is not None:
            wf['CLIP'] = clip
        if vae is not None:
            wf['VAE'] = vae
        if positive is not None:
            wf['POSITIVE'] = positive
        if negative is not None:
            wf['NEGATIVE'] = negative
        if latent is not None:
            wf['LATENT'] = latent
        if image is not None:
            wf['IMAGE'] = image

        # -- Build lora stacks as tuples for LORA_STACK output -------
        # Filter not-found LoRAs here so downstream nodes receiving
        # LORA_STACK don't get unavailable entries. Keep wf['loras_*']
        # unchanged to preserve authored workflow_data for chaining.
        _lora_found_cache = {}

        def _lora_is_found(name):
            if not name:
                return False
            if name in _lora_found_cache:
                return _lora_found_cache[name]
            _, found = resolve_lora_path(name)
            _lora_found_cache[name] = bool(found)
            return _lora_found_cache[name]

        def _is_active_available_found(lora_item):
            if not isinstance(lora_item, dict):
                return False
            lora_name = lora_item.get('name')
            if not lora_name:
                return False
            if lora_item.get('active', True) is False:
                return False
            if lora_item.get('available', True) is False:
                return False
            return _lora_is_found(lora_name)

        lora_stack_a = [
            (item['name'], item.get('model_strength', 1.0), item.get('clip_strength', 1.0))
            for item in wf.get('loras_a', [])
            if _is_active_available_found(item)
        ]
        lora_stack_b = [
            (item['name'], item.get('model_strength', 1.0), item.get('clip_strength', 1.0))
            for item in wf.get('loras_b', [])
            if _is_active_available_found(item)
        ]

        # -- Extract all output values -------------------------------
        model_name = wf.get('model_name', '')
        if not model_name:
            model_name = _short_display_name(wf.get('model_a', ''))

        return (
            wf,
            model_a, model_b, clip, positive, negative, vae, latent, image,
            sampler.get('seed_a', sampler.get('seed', 0)),
            sampler.get('seed_b', 0) or 0,
            sampler.get('steps_a', sampler.get('steps', 20)),
            sampler.get('steps_b', sampler.get('steps_a', sampler.get('steps', 20))),
            sampler.get('cfg', 7.0),
            sampler.get('sampler_name', 'euler'),
            sampler.get('scheduler', 'normal'),
            wf.get('positive_prompt', ''),
            wf.get('negative_prompt', ''),
            lora_stack_a,
            lora_stack_b,
            resolution.get('width', 512),
            resolution.get('height', 512),
            resolution.get('batch_size', 1),
            resolution.get('length', 0) or 0,
            model_name,
        )
