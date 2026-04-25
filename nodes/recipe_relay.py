"""
ComfyUI Workflow Bridge - Pure passthrough node.
Unpacks recipe_data into individual typed outputs.
Any connected optional input overrides the corresponding field before output.
Models (MODEL/CLIP/VAE) are passed through only - use RecipeModelLoader for loading.
"""
import json
import os
import folder_paths
import nodes
import comfy.samplers
from ..py.lora_utils import resolve_lora_path, get_lora_relative_path, strip_lora_extension
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


def _get_live_loader_combo_types():
    """Mirror active loader combo socket types so Relay connects to core load nodes."""
    def _from_node(node_names, key, fallback):
        for node_name in node_names:
            try:
                cls = nodes.NODE_CLASS_MAPPINGS.get(node_name)
                if cls is None:
                    continue
                required = cls.INPUT_TYPES().get("required", {})
                combo = required.get(key)
                if isinstance(combo, tuple) and len(combo) >= 1:
                    return combo[0]
            except Exception:
                continue
        return fallback

    ckpt_enum = _from_node(
        ["CheckpointLoaderSimple", "CheckpointLoader"],
        "ckpt_name",
        folder_paths.get_filename_list("checkpoints"),
    )
    unet_enum = _from_node(
        ["UNETLoader", "UnetLoader", "DiffusionModelLoader"],
        "unet_name",
        folder_paths.get_filename_list("diffusion_models"),
    )
    vae_enum = _from_node(
        ["VAELoader"],
        "vae_name",
        folder_paths.get_filename_list("vae"),
    )
    clip_enum = _from_node(
        ["CLIPLoader"],
        "clip_name",
        folder_paths.get_filename_list("text_encoders"),
    )
    return ckpt_enum, unet_enum, vae_enum, clip_enum


_LIVE_SAMPLERS, _LIVE_SCHEDULERS = _get_live_ksampler_combo_types()
_LIVE_CKPTS, _LIVE_UNETS, _LIVE_VAES, _LIVE_CLIPS = _get_live_loader_combo_types()


_MODEL_KEYS = ("model_a", "model_b", "model_c", "model_d")


def _normalize_model_slot(value):
    key = str(value or "model_a").strip().lower()
    return key if key in _MODEL_KEYS else "model_a"


def _relay_default_sampler():
    return {
        "steps": 10,
        "cfg": 5,
        "sampler_name": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "seed": 0,
    }


def _relay_default_resolution():
    return {
        "width": 512,
        "height": 512,
        "batch_size": 1,
        "length": 1,
    }


def _relay_default_model_block():
    return {
        "positive_prompt": "",
        "negative_prompt": "",
        "family": "sdxl",
        "model": "",
        "loras": [],
        "clip_type": "",
        "loader_type": "",
        "vae": "",
        "clip": [],
        "sampler": _relay_default_sampler(),
        "resolution": _relay_default_resolution(),
    }


def _selected_v2_block(wf, model_slot, create=False):
    if int(wf.get("version", 0) or 0) < 2:
        return None
    if not isinstance(wf.get("models"), dict):
        if not create:
            return None
        wf["models"] = {}

    models = wf["models"]
    slot = _normalize_model_slot(model_slot)
    block = models.get(slot)
    if isinstance(block, dict):
        return block
    if not create:
        return None

    block = dict(_relay_default_model_block())
    models[slot] = block
    return block


DEFAULT_RECIPE_DATA = {
    "_source": "RecipeRelay",
    "version": 2,
    "models": {},
}


class WorkflowRelay:
    """
    Pure passthrough bridge node.
    Unpacks recipe_data into individual typed outputs and forwards
    any connected MODEL/CLIP/VAE inputs. No model loading.
    """

    @classmethod
    def _sync_combo_types(cls):
        samplers, schedulers = _get_live_ksampler_combo_types()
        ckpts, unets, vaes, clips = _get_live_loader_combo_types()
        cls._SAMPLER_ENUM = samplers
        cls._SCHEDULER_ENUM = schedulers
        cls._UNET_ENUM = unets
        cls._CKPT_ENUM = ckpts
        cls._VAE_ENUM = vaes
        cls._CLIP_ENUM = clips
        cls.RETURN_TYPES = (
            "RECIPE_DATA",
            "MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "VAE", "LATENT", "IMAGE", "MASK", ANY_TYPE,
            "INT", "INT", "FLOAT",
            cls._SAMPLER_ENUM, cls._SCHEDULER_ENUM, "FLOAT",
            "STRING", "STRING",
            "LORA_STACK",
            "INT", "INT", "INT", "INT",
            "STRING",
            cls._CKPT_ENUM, cls._UNET_ENUM, cls._VAE_ENUM, cls._CLIP_ENUM,
        )

    @classmethod
    def INPUT_TYPES(cls):
        cls._sync_combo_types()
        return {
            "required": {},
            "optional": {
                "recipe_data":     ("RECIPE_DATA",  {"forceInput": True, "tooltip": "Optional recipe_data input. If omitted, bridge starts from an empty payload dict."}),
                "model_slot":      (_MODEL_KEYS,    {"default": "model_a", "tooltip": "Select which model slot to read/update in recipe_data."}),
                "model":           ("MODEL",        {"tooltip": "Pass-through Model"}),
                "clip":            ("CLIP",         {"tooltip": "Pass-through CLIP"}),
                "positive":        ("CONDITIONING", {"tooltip": "Pass-through positive conditioning"}),
                "negative":        ("CONDITIONING", {"tooltip": "Pass-through negative conditioning"}),
                "vae":             ("VAE",          {"tooltip": "Pass-through VAE"}),
                "latent":          ("LATENT",       {"tooltip": "Pass-through latent"}),
                "image":           ("IMAGE",        {"tooltip": "Pass-through image"}),
                "mask":            ("MASK",         {"tooltip": "Pass-through mask"}),
                "extra":           (ANY_TYPE,       {"tooltip": "Pass-through extra data (any type)"}),
                "seed":            ("INT",          {"forceInput": True, "tooltip": "Override sampler seed"}),
                "steps":           ("INT",          {"forceInput": True, "tooltip": "Override sampling steps"}),
                "cfg":             ("FLOAT",        {"forceInput": True, "tooltip": "Override CFG scale"}),
                "sampler_name":    (cls._SAMPLER_ENUM,   {"forceInput": True, "tooltip": "Override sampler name"}),
                "scheduler":       (cls._SCHEDULER_ENUM, {"forceInput": True, "tooltip": "Override scheduler"}),
                "denoise":         ("FLOAT",        {"forceInput": True, "tooltip": "Override denoise"}),
                "pos_prompt":      ("STRING",       {"forceInput": True, "tooltip": "Override positive prompt"}),
                "neg_prompt":      ("STRING",       {"forceInput": True, "tooltip": "Override negative prompt"}),
                "lora_stack":      ("LORA_STACK",   {"tooltip": "Override LoRA stack"}),
                "width":           ("INT",          {"forceInput": True, "tooltip": "Override width"}),
                "height":          ("INT",          {"forceInput": True, "tooltip": "Override height"}),
                "batch_size":      ("INT",          {"forceInput": True, "tooltip": "Override batch size"}),
                "length":          ("INT",          {"forceInput": True, "tooltip": "Override video length"}),
                "model_name":      ("STRING",       {"forceInput": True, "tooltip": "Optional model name/path. Resolves and writes recipe model field(s)."}),
                "model_data":      (ANY_TYPE,       {"forceInput": True, "tooltip": "Optional structured model payload from Recipe Model Picker."}),
            },
        }

    RETURN_TYPES = (
        "RECIPE_DATA",
        "MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "VAE", "LATENT", "IMAGE", "MASK", ANY_TYPE,
        "INT", "INT", "FLOAT",
        _LIVE_SAMPLERS, _LIVE_SCHEDULERS, "FLOAT",
        "STRING", "STRING",
        "LORA_STACK",
        "INT", "INT", "INT", "INT",
        "STRING",
        _LIVE_CKPTS, _LIVE_UNETS, _LIVE_VAES, _LIVE_CLIPS,
    )
    RETURN_NAMES = (
        "recipe_data",
        "model", "clip", "positive", "negative", "vae", "latent", "image", "mask", "extra",
        "seed", "steps", "cfg",
        "sampler_name", "scheduler", "denoise",
        "pos_prompt", "neg_prompt",
        "lora_stack",
        "width", "height", "batch_size", "length",
        "model_name",
        "ckpt_name", "unet_name", "vae_name", "clip_name",
    )
    FUNCTION = "unpack"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = (
        "Pure passthrough bridge node. Unpacks recipe_data into individual "
        "typed outputs. Models are forwarded from connected inputs only."
    )

    # ------------------------------------------------------------------

    def unpack(self, recipe_data=None, model_slot="model_a", **kwargs):
        def _normalize_lora_stack_input(raw_stack):
            rows = []
            if raw_stack is None:
                return rows

            for entry in raw_stack:
                if not (isinstance(entry, (list, tuple)) and len(entry) >= 1):
                    continue

                raw_name = str(entry[0] or "").strip()
                if not raw_name:
                    continue

                try:
                    model_strength = float(entry[1]) if len(entry) >= 2 else 1.0
                except Exception:
                    model_strength = 1.0

                try:
                    clip_strength = float(entry[2]) if len(entry) >= 3 else model_strength
                except Exception:
                    clip_strength = model_strength

                rel_path, available = get_lora_relative_path(raw_name)
                if not available or not rel_path:
                    rel_path = raw_name

                display_name = strip_lora_extension(os.path.basename(str(rel_path).replace("\\", "/")))
                rows.append({
                    'name': display_name,
                    'path': rel_path,
                    'model_strength': model_strength,
                    'clip_strength': clip_strength,
                    'active': True,
                    'available': bool(available),
                })

            return rows

        def _resolve_lora_output_path(lora_item):
            if not isinstance(lora_item, dict):
                return "", False

            candidate = str(lora_item.get('path') or lora_item.get('name') or '').strip()
            if not candidate:
                return "", False

            rel_path, available = get_lora_relative_path(candidate)
            if available and rel_path:
                return rel_path, True

            _, resolved_available = resolve_lora_path(candidate)
            return candidate, bool(resolved_available)

        def _short_display_name(name):
            raw = str(name or "").strip()
            if not raw:
                return ""
            base = os.path.basename(raw.replace("\\", "/"))
            return os.path.splitext(base)[0]

        # Accept both dict and JSON string
        if isinstance(recipe_data, str):
            try:
                incoming_wf = json.loads(recipe_data)
            except (json.JSONDecodeError, TypeError):
                incoming_wf = {}
        elif isinstance(recipe_data, dict):
            incoming_wf = dict(recipe_data)
        else:
            incoming_wf = {}

        # Preserve incoming recipe_data exactly when present.
        # Only fall back to the empty template when no input payload exists.
        if isinstance(incoming_wf, dict) and incoming_wf:
            wf = dict(incoming_wf)
            if isinstance(incoming_wf.get('sampler'), dict):
                wf['sampler'] = dict(incoming_wf['sampler'])
            if isinstance(incoming_wf.get('resolution'), dict):
                wf['resolution'] = dict(incoming_wf['resolution'])
            if isinstance(incoming_wf.get('models'), dict):
                wf['models'] = {
                    k: dict(v) if isinstance(v, dict) else v
                    for k, v in incoming_wf.get('models', {}).items()
                }
        else:
            wf = dict(DEFAULT_RECIPE_DATA)
            wf['models'] = {}

        selected_slot = _normalize_model_slot(model_slot)
        selected_block = _selected_v2_block(wf, selected_slot, create=False)

        v2_family = selected_block.get('family', '') if isinstance(selected_block, dict) else ''
        v2_model_name = selected_block.get('model', '') if isinstance(selected_block, dict) else ''
        v2_positive = selected_block.get('positive_prompt', '') if isinstance(selected_block, dict) else ''
        v2_negative = selected_block.get('negative_prompt', '') if isinstance(selected_block, dict) else ''
        v2_vae = selected_block.get('vae', '') if isinstance(selected_block, dict) else ''
        v2_clip = selected_block.get('clip', []) if isinstance(selected_block, dict) else []
        v2_clip_type = selected_block.get('clip_type', '') if isinstance(selected_block, dict) else ''
        v2_loader_type = selected_block.get('loader_type', '') if isinstance(selected_block, dict) else ''
        v2_loras = selected_block.get('loras', []) if isinstance(selected_block, dict) else []
        v2_sampler = selected_block.get('sampler', {}) if isinstance(selected_block, dict) and isinstance(selected_block.get('sampler'), dict) else {}
        v2_resolution = selected_block.get('resolution', {}) if isinstance(selected_block, dict) and isinstance(selected_block.get('resolution'), dict) else {}

        sampler = dict(_relay_default_sampler())
        if isinstance(wf.get('sampler'), dict):
            sampler.update(wf.get('sampler', {}))
        if isinstance(v2_sampler, dict):
            sampler.update(v2_sampler)

        resolution = dict(_relay_default_resolution())
        if isinstance(wf.get('resolution'), dict):
            resolution.update(wf.get('resolution', {}))
        if isinstance(v2_resolution, dict):
            resolution.update(v2_resolution)

        if isinstance(v2_positive, str) and v2_positive and not wf.get('positive_prompt'):
            wf['positive_prompt'] = v2_positive
        if isinstance(v2_negative, str) and v2_negative and not wf.get('negative_prompt'):
            wf['negative_prompt'] = v2_negative
        if isinstance(v2_family, str) and v2_family and not wf.get('family'):
            wf['family'] = v2_family
        if isinstance(v2_vae, str) and v2_vae and not wf.get('vae'):
            wf['vae'] = v2_vae
        if isinstance(v2_clip, list) and v2_clip and not wf.get('clip'):
            wf['clip'] = v2_clip
        if isinstance(v2_clip_type, str) and v2_clip_type and not wf.get('clip_type'):
            wf['clip_type'] = v2_clip_type
        if isinstance(v2_loader_type, str) and v2_loader_type and not wf.get('loader_type'):
            wf['loader_type'] = v2_loader_type
        if isinstance(v2_model_name, str) and v2_model_name and not wf.get('model_a'):
            wf['model_a'] = v2_model_name
        if isinstance(v2_loras, list) and v2_loras and not wf.get('loras_a'):
            wf['loras_a'] = v2_loras

        # -- Apply top-level overrides -------------------------------
        pos_text = kwargs.get('pos_prompt')
        if pos_text is not None:
            wf['positive_prompt'] = pos_text

        neg_text = kwargs.get('neg_prompt')
        if neg_text is not None:
            wf['negative_prompt'] = neg_text

        # -- LoRA stack overrides ------------------------------------
        if kwargs.get('lora_stack') is not None:
            normalized_stack = _normalize_lora_stack_input(kwargs['lora_stack'])
            wf['loras_a'] = normalized_stack
            # Keep dual-stack recipes coherent when a single override is used.
            if 'loras_b' in wf:
                wf['loras_b'] = normalized_stack

            if int(wf.get('version', 0) or 0) >= 2:
                target_block = _selected_v2_block(wf, selected_slot, create=True)
                target_block['loras'] = normalized_stack

        # -- Resolution overrides ------------------------------------
        for key in ('width', 'height', 'batch_size', 'length'):
            if kwargs.get(key) is not None:
                resolution[key] = kwargs[key]
        wf['resolution'] = resolution

        # -- Sampler overrides ---------------------------------------
        if kwargs.get('steps') is not None:
            sampler['steps_a'] = kwargs['steps']
            sampler['steps_b'] = kwargs['steps']
        if kwargs.get('seed') is not None:
            sampler['seed_a'] = kwargs['seed']
            sampler['seed_b'] = kwargs['seed']

        for key in ('cfg', 'denoise', 'sampler_name', 'scheduler'):
            if kwargs.get(key) is not None:
                sampler[key] = kwargs[key]
        wf['sampler'] = sampler

        if int(wf.get('version', 0) or 0) >= 2:
            target_block = _selected_v2_block(wf, selected_slot, create=True)
            bs = target_block.get('sampler', {}) if isinstance(target_block.get('sampler'), dict) else {}
            if kwargs.get('steps') is not None:
                bs['steps'] = kwargs['steps']
            if kwargs.get('seed') is not None:
                bs['seed'] = kwargs['seed']
            for key in ('cfg', 'denoise', 'sampler_name', 'scheduler'):
                if kwargs.get(key) is not None:
                    bs[key] = kwargs[key]
            target_block['sampler'] = bs

            br = target_block.get('resolution', {}) if isinstance(target_block.get('resolution'), dict) else {}
            for key in ('width', 'height', 'batch_size', 'length'):
                if kwargs.get(key) is not None:
                    br[key] = kwargs[key]
            if br:
                target_block['resolution'] = br

        # -- Model/CLIP/VAE name overrides ---------------------------
        model_name_in = kwargs.get('model_name')
        if isinstance(model_name_in, str) and model_name_in.strip():
            resolved_name, _ = resolve_model_name(model_name_in)
            chosen_name = resolved_name or os.path.basename(model_name_in.strip().replace("\\", "/"))
            wf['model_a'] = chosen_name

            if int(wf.get('version', 0) or 0) >= 2:
                target_block = _selected_v2_block(wf, selected_slot, create=True)
                target_block['model'] = chosen_name

            family = get_model_family(chosen_name)
            if family:
                wf['family'] = family
                family_spec = MODEL_FAMILIES.get(family, {})
                wf['clip_type'] = family_spec.get('clip_type', '')
                wf['loader_type'] = 'checkpoint' if family_spec.get('checkpoint', False) else 'unet'

        # -- Pass-through MODEL / CLIP / VAE -------------------------
        # Priority: explicit connected inputs > embedded objects in workflow_data.
        model = kwargs.get('model')
        if model is None:
            model = selected_block.get('MODEL') if isinstance(selected_block, dict) else None
        if model is None:
            model = wf.get('MODEL')
        if model is None:
            model = wf.get('MODEL_A')

        selected_loader_type = (
            str((selected_block or {}).get('loader_type', '')).strip().lower()
            if isinstance(selected_block, dict)
            else str(wf.get('loader_type', '')).strip().lower()
        )

        clip = kwargs.get('clip')
        if clip is None:
            clip = selected_block.get('CLIP') if isinstance(selected_block, dict) else None
        if clip is None:
            clip = wf.get('CLIP')

        vae = kwargs.get('vae')
        if vae is None:
            vae = selected_block.get('VAE') if isinstance(selected_block, dict) else None
        if vae is None:
            vae = wf.get('VAE')

        positive = kwargs.get('positive')
        if positive is None:
            positive = selected_block.get('POSITIVE') if isinstance(selected_block, dict) else None
        if positive is None:
            positive = wf.get('POSITIVE')

        negative = kwargs.get('negative')
        if negative is None:
            negative = selected_block.get('NEGATIVE') if isinstance(selected_block, dict) else None
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

        if model is not None:
            wf['MODEL'] = model
            wf['MODEL_A'] = model
            if int(wf.get('version', 0) or 0) >= 2:
                _selected_v2_block(wf, selected_slot, create=True)['MODEL'] = model
        if clip is not None:
            wf['CLIP'] = clip
            if int(wf.get('version', 0) or 0) >= 2:
                _selected_v2_block(wf, selected_slot, create=True)['CLIP'] = clip
        if vae is not None:
            wf['VAE'] = vae
            if int(wf.get('version', 0) or 0) >= 2:
                _selected_v2_block(wf, selected_slot, create=True)['VAE'] = vae
        if positive is not None:
            wf['POSITIVE'] = positive
            if int(wf.get('version', 0) or 0) >= 2:
                _selected_v2_block(wf, selected_slot, create=True)['POSITIVE'] = positive
        if negative is not None:
            wf['NEGATIVE'] = negative
            if int(wf.get('version', 0) or 0) >= 2:
                _selected_v2_block(wf, selected_slot, create=True)['NEGATIVE'] = negative
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

        def _is_active_available(lora_item):
            if not isinstance(lora_item, dict):
                return False
            resolved_path, available = _resolve_lora_output_path(lora_item)
            if not resolved_path:
                return False
            if lora_item.get('active', True) is False:
                return False
            if lora_item.get('available', True) is False:
                return False
            return available

        lora_stack = [
            (_resolve_lora_output_path(item)[0], item.get('model_strength', 1.0), item.get('clip_strength', 1.0))
            for item in (wf.get('loras_a', []) if isinstance(wf.get('loras_a', []), list) else [])
            if _is_active_available(item)
        ]
        final_selected = _selected_v2_block(wf, selected_slot, create=False)
        selected_loras = final_selected.get('loras', []) if isinstance(final_selected, dict) and isinstance(final_selected.get('loras'), list) else []
        if selected_loras:
            lora_stack = [
                (_resolve_lora_output_path(item)[0], item.get('model_strength', 1.0), item.get('clip_strength', 1.0))
                for item in selected_loras
                if _is_active_available(item)
            ]
        elif not lora_stack and isinstance(v2_loras, list):
            lora_stack = [
                (_resolve_lora_output_path(item)[0], item.get('model_strength', 1.0), item.get('clip_strength', 1.0))
                for item in v2_loras
                if _is_active_available(item)
            ]

        final_sampler = dict(sampler)
        if isinstance(final_selected, dict) and isinstance(final_selected.get('sampler'), dict):
            final_sampler.update(final_selected.get('sampler', {}))

        final_resolution = dict(resolution)
        if isinstance(final_selected, dict) and isinstance(final_selected.get('resolution'), dict):
            final_resolution.update(final_selected.get('resolution', {}))

        final_model_name = ''
        if isinstance(final_selected, dict):
            final_model_name = _short_display_name(final_selected.get('model', ''))
        if not final_model_name:
            final_model_name = _short_display_name(wf.get('model_a', '') or v2_model_name)

        final_pos = (final_selected.get('positive_prompt', '') if isinstance(final_selected, dict) else '') or wf.get('positive_prompt', '') or v2_positive
        final_neg = (final_selected.get('negative_prompt', '') if isinstance(final_selected, dict) else '') or wf.get('negative_prompt', '') or v2_negative

        raw_model_data = kwargs.get('model_data')
        if isinstance(raw_model_data, str):
            try:
                model_data_in = json.loads(raw_model_data)
            except Exception:
                model_data_in = {}
        elif isinstance(raw_model_data, dict):
            model_data_in = dict(raw_model_data)
        else:
            model_data_in = {}

        ckpt_name_in = str(model_data_in.get('ckpt_name', '') or '').strip()
        unet_name_in = str(model_data_in.get('unet_name', '') or '').strip()
        vae_name_in = str(model_data_in.get('vae_name', model_data_in.get('vae', '')) or '').strip()
        clip_name_in = str(model_data_in.get('clip_name', '') or '').strip()
        if not clip_name_in and isinstance(model_data_in.get('clip'), list) and model_data_in.get('clip'):
            clip_name_in = str(model_data_in.get('clip')[0] or '').strip()
        model_name_from_data = str(model_data_in.get('model', '') or '').strip()
        loader_type_from_data = str(model_data_in.get('loader_type', '') or '').strip().lower()
        family_from_data = str(model_data_in.get('family', '') or '').strip()
        clip_type_from_data = str(model_data_in.get('clip_type', '') or '').strip()

        selected_model_raw = (final_selected.get('model', '') if isinstance(final_selected, dict) else '') or wf.get('model_a', '') or v2_model_name or ''
        selected_vae_raw = (final_selected.get('vae', '') if isinstance(final_selected, dict) else '') or wf.get('vae', '') or v2_vae or ''
        selected_clip_raw = ''
        if isinstance(final_selected, dict) and isinstance(final_selected.get('clip'), list) and final_selected.get('clip'):
            selected_clip_raw = str(final_selected.get('clip')[0] or '').strip()
        elif isinstance(wf.get('clip'), list) and wf.get('clip'):
            selected_clip_raw = str(wf.get('clip')[0] or '').strip()
        elif isinstance(v2_clip, list) and v2_clip:
            selected_clip_raw = str(v2_clip[0] or '').strip()

        selected_model_name = str(selected_model_raw or '').strip()
        selected_vae_name = str(selected_vae_raw or '').strip()
        selected_clip_name = str(selected_clip_raw or '').strip()

        # Structured model_data / combo-name overrides are authoritative.
        if model_name_from_data:
            selected_model_name = model_name_from_data
        if loader_type_from_data in ('checkpoint', 'unet', 'diffusion'):
            selected_loader_type = 'unet' if loader_type_from_data == 'diffusion' else loader_type_from_data

        # If checkpoint/unet names are provided, update model name and loader type
        # even when recipe_data did not previously declare a loader type.
        if ckpt_name_in:
            selected_model_name = ckpt_name_in
            selected_loader_type = 'checkpoint'
        elif unet_name_in:
            selected_model_name = unet_name_in
            selected_loader_type = 'unet'

        if vae_name_in:
            selected_vae_name = vae_name_in
        if clip_name_in:
            selected_clip_name = clip_name_in

        is_v2_recipe = int(wf.get('version', 0) or 0) >= 2

        if is_v2_recipe:
            target_block = _selected_v2_block(wf, selected_slot, create=True)
            if selected_model_name:
                target_block['model'] = selected_model_name
            if selected_loader_type:
                target_block['loader_type'] = selected_loader_type
            if family_from_data:
                target_block['family'] = family_from_data
            if clip_type_from_data:
                target_block['clip_type'] = clip_type_from_data
            if selected_vae_name:
                target_block['vae'] = selected_vae_name
            if selected_clip_name:
                target_block['clip'] = [selected_clip_name]
        else:
            # Legacy payload fallback: keep top-level writes for non-v2 dicts.
            if selected_model_name:
                wf['model_a'] = selected_model_name
            if selected_loader_type:
                wf['loader_type'] = selected_loader_type
            if family_from_data:
                wf['family'] = family_from_data
            if clip_type_from_data:
                wf['clip_type'] = clip_type_from_data
            if selected_vae_name:
                wf['vae'] = selected_vae_name
            if selected_clip_name:
                wf['clip'] = [selected_clip_name]

        if selected_model_name:
            detected_family = get_model_family(selected_model_name)
            if detected_family:
                if is_v2_recipe:
                    _selected_v2_block(wf, selected_slot, create=True)['family'] = detected_family
                else:
                    wf['family'] = detected_family

        # -- Extract all output values -------------------------------
        model_name = _short_display_name(selected_model_name) or final_model_name

        # Always emit string outputs for combo-name sockets to avoid None flowing
        # into downstream standard loader nodes.
        ckpt_out = ckpt_name_in or (selected_model_name if selected_loader_type == 'checkpoint' else "")
        unet_out = unet_name_in or (selected_model_name if selected_loader_type in ('unet', 'diffusion') else "")
        vae_name_out = selected_vae_name or ""
        clip_name_out = selected_clip_name or ""

        return (
            wf,
            model, clip, positive, negative, vae, latent, image, mask, extra,
            final_sampler.get('seed_a', final_sampler.get('seed', 0)),
            final_sampler.get('steps_a', final_sampler.get('steps', 10)),
            final_sampler.get('cfg', 5.0),
            final_sampler.get('sampler_name', 'euler'),
            final_sampler.get('scheduler', 'simple'),
            final_sampler.get('denoise', 1.0),
            final_pos,
            final_neg,
            lora_stack,
            final_resolution.get('width', 512),
            final_resolution.get('height', 512),
            final_resolution.get('batch_size', 1),
            final_resolution.get('length', 0) or 0,
            model_name,
            ckpt_out,
            unet_out,
            vae_name_out,
            clip_name_out,
        )
