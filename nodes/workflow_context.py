"""
ComfyUI Workflow Context — Pure passthrough node.
Unpacks workflow_data into individual typed outputs.
Any connected optional input overrides the corresponding field before output.
Models (MODEL/CLIP/VAE) are passed through only — use WorkflowGetModel for loading.
"""
import json


class WorkflowContext:
    """
    Pure passthrough context node.
    Unpacks workflow_data into individual typed outputs and forwards
    any connected MODEL/CLIP/VAE inputs. No model loading.
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
                "model_a":         ("MODEL",   {"tooltip": "Pass-through Model A"}),
                "model_b":         ("MODEL",   {"tooltip": "Pass-through Model B"}),
                "clip":            ("CLIP",    {"tooltip": "Pass-through CLIP"}),
                "vae":             ("VAE",     {"tooltip": "Pass-through VAE"}),
                "positive_prompt": ("STRING",  {"forceInput": True, "tooltip": "Override positive prompt"}),
                "negative_prompt": ("STRING",  {"forceInput": True, "tooltip": "Override negative prompt"}),
                "lora_stack_a":    ("LORA_STACK", {"tooltip": "Override LoRA stack A"}),
                "lora_stack_b":    ("LORA_STACK", {"tooltip": "Override LoRA stack B"}),
                "width":           ("INT",     {"forceInput": True, "tooltip": "Override width"}),
                "height":          ("INT",     {"forceInput": True, "tooltip": "Override height"}),
                "batch_size":      ("INT",     {"forceInput": True, "tooltip": "Override batch size"}),
                "length":          ("INT",     {"forceInput": True, "tooltip": "Override video length"}),
                "steps_a":         ("INT",     {"forceInput": True, "tooltip": "Override sampling steps A"}),
                "steps_b":         ("INT",     {"forceInput": True, "tooltip": "Override sampling steps B (dual-sampler)"}),
                "cfg":             ("FLOAT",   {"forceInput": True, "tooltip": "Override CFG scale"}),
                "seed_a":          ("INT",     {"forceInput": True, "tooltip": "Override seed A"}),
                "seed_b":          ("INT",     {"forceInput": True, "tooltip": "Override seed B (dual-sampler)"}),
                "sampler_name":    ("STRING",  {"forceInput": True, "tooltip": "Override sampler name"}),
                "scheduler":       ("STRING",  {"forceInput": True, "tooltip": "Override scheduler"}),
            },
        }

    RETURN_TYPES = (
        "WORKFLOW_DATA",
        "MODEL", "MODEL", "CLIP", "VAE",
        "STRING", "STRING",
        "LORA_STACK", "LORA_STACK",
        "INT", "INT", "INT", "INT",
        "INT", "INT", "FLOAT", "INT", "INT",
        "STRING", "STRING",
    )
    RETURN_NAMES = (
        "workflow_data",
        "model_a", "model_b", "clip", "vae",
        "positive_prompt", "negative_prompt",
        "lora_stack_a", "lora_stack_b",
        "width", "height", "batch_size", "length",
        "steps_a", "steps_b", "cfg", "seed_a", "seed_b",
        "sampler_name", "scheduler",
    )
    FUNCTION = "unpack"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = (
        "Pure passthrough context node. Unpacks workflow_data into individual "
        "typed outputs. Models are forwarded from connected inputs only."
    )

    # ------------------------------------------------------------------

    def unpack(self, workflow_data, **kwargs):
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
        for key in ('steps_a', 'steps_b', 'cfg', 'seed_a', 'seed_b', 'sampler_name', 'scheduler'):
            if kwargs.get(key) is not None:
                sampler[key] = kwargs[key]
        wf['sampler'] = sampler

        # ── Pass-through MODEL / CLIP / VAE ──────────────────────────
        model_a = kwargs.get('model_a')
        model_b = kwargs.get('model_b')
        clip = kwargs.get('clip')
        vae = kwargs.get('vae')

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
            sampler.get('steps_a', sampler.get('steps', 20)),
            sampler.get('steps_b', sampler.get('steps_a', sampler.get('steps', 20))),
            sampler.get('cfg', 7.0),
            sampler.get('seed_a', sampler.get('seed', 0)),
            sampler.get('seed_b', 0) or 0,
            sampler.get('sampler_name', 'euler'),
            sampler.get('scheduler', 'normal'),
        )
