"""
ComfyUI Workflow Bridge - Pure passthrough node.
Unpacks workflow_data into individual typed outputs.
Any connected optional input overrides the corresponding field before output.
Models (MODEL/CLIP/VAE) are passed through only - use RecipeModelLoader for loading.
"""
import json
import os
import folder_paths
import comfy.samplers
from ..py.lora_utils import resolve_lora_path
from ..py.workflow_extraction_utils import resolve_model_name
from ..py.workflow_families import get_model_family, MODEL_FAMILIES


MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".gguf")


class _AnyType(str):
    """Comfy wildcard type that can connect to any socket type."""

    def __ne__(self, _other):
        return False


ANY_TYPE = _AnyType("ANY")


def _strip_model_ext(name):
    base = os.path.basename(str(name or "").replace("\\", "/"))
    lower = base.lower()
    for ext in MODEL_EXTENSIONS:
        if lower.endswith(ext):
            return base[:-len(ext)]
    return os.path.splitext(base)[0]


def _resolve_rel_path(raw_name, folder_keys):
    """Resolve a model-like name to a relative Comfy path in the provided folder keys."""
    if not raw_name:
        return "", False

    raw = str(raw_name).strip().replace("\\", "/")
    if not raw:
        return "", False

    base = os.path.basename(raw).lower()
    base_no_ext = _strip_model_ext(base).lower()

    for key in folder_keys:
        try:
            file_list = folder_paths.get_filename_list(key)
        except Exception:
            continue

        for rel in file_list:
            rel_norm = rel.replace("\\", "/")
            if rel_norm == raw:
                return rel, True

            rel_base = os.path.basename(rel_norm).lower()
            if rel_base == base:
                return rel, True

            if _strip_model_ext(rel_base).lower() == base_no_ext:
                return rel, True

    return os.path.basename(raw), False


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


DEFAULT_WORKFLOW_DATA = {
    "_source": "RecipeRelay",
    "family": "",
    "model_a": "",
    "model_b": "",
    "positive_prompt": "",
    "negative_prompt": "",
    "loras_a": [],
    "loras_b": [],
    "vae": "",
    "clip": [],
    "clip_type": "",
    "loader_type": "",
    "sampler": {
        "steps_a": 10,
        "steps_b": 10,
        "cfg": 5,
        "sampler_name": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "seed_a": 0,
        "seed_b": 0,
    },
    "resolution": {
        "width": 512,
        "height": 512,
        "batch_size": 1,
        "length": 1,
    },
}


class WorkflowRelay:
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
            "MODEL", "MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "VAE", "LATENT", "IMAGE", "MASK", ANY_TYPE,
            "INT", "INT", "INT", "INT", "FLOAT",
            cls._SAMPLER_ENUM, cls._SCHEDULER_ENUM, "FLOAT",
            "STRING", "STRING",
            "LORA_STACK", "LORA_STACK",
            "INT", "INT", "INT", "INT",
            "STRING", "STRING",
        )

    @classmethod
    def INPUT_TYPES(cls):
        cls._sync_combo_types()
        return {
            "required": {},
            "optional": {
                "workflow_data":   ("WORKFLOW_DATA", {"forceInput": True, "tooltip": "Optional workflow data input. If omitted, bridge starts from an empty workflow_data dict."}),
                "model_a":         ("MODEL",        {"tooltip": "Pass-through Model A"}),
                "model_b":         ("MODEL",        {"tooltip": "Pass-through Model B"}),
                "clip":            ("CLIP",         {"tooltip": "Pass-through CLIP"}),
                "positive":        ("CONDITIONING", {"tooltip": "Pass-through positive conditioning"}),
                "negative":        ("CONDITIONING", {"tooltip": "Pass-through negative conditioning"}),
                "vae":             ("VAE",          {"tooltip": "Pass-through VAE"}),
                "latent":          ("LATENT",       {"tooltip": "Pass-through latent"}),
                "image":           ("IMAGE",        {"tooltip": "Pass-through image"}),
                "mask":            ("MASK",         {"tooltip": "Pass-through mask"}),
                "extra":           (ANY_TYPE,         {"tooltip": "Pass-through extra data (any type)"}),
                "seed_a":          ("INT",     {"forceInput": True, "tooltip": "Override seed A"}),
                "seed_b":          ("INT",     {"forceInput": True, "tooltip": "Override seed B (dual-sampler)"}),
                "steps_a":         ("INT",     {"forceInput": True, "tooltip": "Override sampling steps A"}),
                "steps_b":         ("INT",     {"forceInput": True, "tooltip": "Override sampling steps B (dual-sampler)"}),
                "cfg":             ("FLOAT",   {"forceInput": True, "tooltip": "Override CFG scale"}),
                "sampler_name":    (cls._SAMPLER_ENUM, {"forceInput": True, "tooltip": "Override sampler name"}),
                "scheduler":       (cls._SCHEDULER_ENUM, {"forceInput": True, "tooltip": "Override scheduler"}),
                "denoise":         ("FLOAT",   {"forceInput": True, "tooltip": "Override denoise"}),
                "pos_prompt":      ("STRING",  {"forceInput": True, "tooltip": "Override positive prompt"}),
                "neg_prompt":      ("STRING",  {"forceInput": True, "tooltip": "Override negative prompt"}),
                "lora_stack_a":    ("LORA_STACK", {"tooltip": "Override LoRA stack A"}),
                "lora_stack_b":    ("LORA_STACK", {"tooltip": "Override LoRA stack B"}),
                "width":           ("INT",     {"forceInput": True, "tooltip": "Override width"}),
                "height":          ("INT",     {"forceInput": True, "tooltip": "Override height"}),
                "batch_size":      ("INT",     {"forceInput": True, "tooltip": "Override batch size"}),
                "length":          ("INT",     {"forceInput": True, "tooltip": "Override video length"}),
                "model_a_name":    ("STRING",  {"forceInput": True, "tooltip": "Optional Model A name/path. Resolves and writes workflow_data['model_a']."}),
                "model_b_name":    ("STRING",  {"forceInput": True, "tooltip": "Optional Model B name/path. Resolves and writes workflow_data['model_b']."}),
            },
        }

    RETURN_TYPES = (
        "WORKFLOW_DATA",
        "MODEL", "MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "VAE", "LATENT", "IMAGE", "MASK", ANY_TYPE,
        "INT", "INT", "INT", "INT", "FLOAT",
        _LIVE_SAMPLERS, _LIVE_SCHEDULERS, "FLOAT",
        "STRING", "STRING",
        "LORA_STACK", "LORA_STACK",
        "INT", "INT", "INT", "INT",
        "STRING", "STRING",
    )
    RETURN_NAMES = (
        "workflow_data",
        "model_a", "model_b", "clip", "positive", "negative", "vae", "latent", "image", "mask", "extra",
        "seed_a", "seed_b", "steps_a", "steps_b", "cfg",
        "sampler_name", "scheduler", "denoise",
        "pos_prompt", "neg_prompt",
        "lora_stack_a", "lora_stack_b",
        "width", "height", "batch_size", "length",
        "model_a_name", "model_b_name",
    )
    FUNCTION = "unpack"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = (
        "Pure passthrough bridge node. Unpacks workflow_data into individual "
        "typed outputs. Models are forwarded from connected inputs only."
    )

    # ------------------------------------------------------------------

    def unpack(self, workflow_data=None, **kwargs):
        def _short_display_name(name):
            raw = str(name or "").strip()
            if not raw:
                return ""
            base = os.path.basename(raw.replace("\\", "/"))
            return os.path.splitext(base)[0]

        # Accept both dict and JSON string
        if isinstance(workflow_data, str):
            try:
                incoming_wf = json.loads(workflow_data)
            except (json.JSONDecodeError, TypeError):
                incoming_wf = {}
        elif isinstance(workflow_data, dict):
            incoming_wf = dict(workflow_data)
        else:
            incoming_wf = {}

        # Start from a full empty-workflow template, then layer incoming values.
        wf = dict(DEFAULT_WORKFLOW_DATA)
        if isinstance(incoming_wf, dict):
            wf.update(incoming_wf)

            incoming_sampler = incoming_wf.get('sampler')
            if isinstance(incoming_sampler, dict):
                wf['sampler'] = dict(DEFAULT_WORKFLOW_DATA['sampler'])
                wf['sampler'].update(incoming_sampler)

            incoming_resolution = incoming_wf.get('resolution')
            if isinstance(incoming_resolution, dict):
                wf['resolution'] = dict(DEFAULT_WORKFLOW_DATA['resolution'])
                wf['resolution'].update(incoming_resolution)

        sampler = dict(wf.get('sampler', DEFAULT_WORKFLOW_DATA['sampler']))
        resolution = dict(wf.get('resolution', DEFAULT_WORKFLOW_DATA['resolution']))

        # -- Apply top-level overrides -------------------------------
        pos_text = kwargs.get('pos_prompt')
        if pos_text is not None:
            wf['positive_prompt'] = pos_text

        neg_text = kwargs.get('neg_prompt')
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
        for key in ('steps_a', 'steps_b', 'cfg', 'denoise', 'seed_a', 'seed_b', 'sampler_name', 'scheduler'):
            if kwargs.get(key) is not None:
                sampler[key] = kwargs[key]
        wf['sampler'] = sampler

        # -- Model/CLIP/VAE name overrides ---------------------------
        model_a_name_in = kwargs.get('model_a_name')
        model_b_name_in = kwargs.get('model_b_name')

        if isinstance(model_a_name_in, str) and model_a_name_in.strip():
            resolved_a, _ = resolve_model_name(model_a_name_in)
            wf['model_a'] = resolved_a or os.path.basename(model_a_name_in.strip().replace("\\", "/"))

            family = get_model_family(wf['model_a'])
            if family:
                wf['family'] = family
                family_spec = MODEL_FAMILIES.get(family, {})
                wf['clip_type'] = family_spec.get('clip_type', '')
                wf['loader_type'] = 'checkpoint' if family_spec.get('checkpoint', False) else 'unet'

        if isinstance(model_b_name_in, str) and model_b_name_in.strip():
            resolved_b, _ = resolve_model_name(model_b_name_in)
            wf['model_b'] = resolved_b or os.path.basename(model_b_name_in.strip().replace("\\", "/"))

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

        mask = kwargs.get('mask')
        if mask is None:
            mask = wf.get('MASK')

        extra = kwargs.get('extra')
        if extra is None:
            extra = wf.get('EXTRA')

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
        if mask is not None:
            wf['MASK'] = mask
        if extra is not None:
            wf['EXTRA'] = extra

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
        model_a_name = _short_display_name(wf.get('model_a', ''))
        model_b_name = _short_display_name(wf.get('model_b', ''))

        return (
            wf,
            model_a, model_b, clip, positive, negative, vae, latent, image, mask, extra,
            sampler.get('seed_a', sampler.get('seed', 0)),
            sampler.get('seed_b', 0) or 0,
            sampler.get('steps_a', sampler.get('steps', 10)),
            sampler.get('steps_b', sampler.get('steps_a', sampler.get('steps', 10))),
            sampler.get('cfg', 5.0),
            sampler.get('sampler_name', 'euler'),
            sampler.get('scheduler', 'simple'),
            sampler.get('denoise', 1.0),
            wf.get('positive_prompt', ''),
            wf.get('negative_prompt', ''),
            lora_stack_a,
            lora_stack_b,
            resolution.get('width', 512),
            resolution.get('height', 512),
            resolution.get('batch_size', 1),
            resolution.get('length', 0) or 0,
            model_a_name,
            model_b_name,
        )
