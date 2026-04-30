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


def _selected_model_block(wf, model_slot, create=False):
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
            "MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "VAE", "LATENT", "IMAGE", "MASK", "*", "*",
            "INT", "INT", "FLOAT",
            cls._SAMPLER_ENUM, cls._SCHEDULER_ENUM, "FLOAT",
            "STRING", "STRING",
            "LORA_STACK",
            "INT", "INT", "INT", "INT",
            "STRING", "STRING",
            cls._CKPT_ENUM, cls._UNET_ENUM, cls._VAE_ENUM, cls._CLIP_ENUM,
        )

    @classmethod
    def INPUT_TYPES(cls):
        cls._sync_combo_types()
        return {
            "required": {},
            "optional": {
                "recipe_data":     ("RECIPE_DATA",       {"forceInput": True, "tooltip": "Optional recipe_data input. If omitted, bridge starts from an empty payload dict."}),
                "model_slot":      (_MODEL_KEYS,         {"default": "model_a", "tooltip": "Select which model slot to read/update in recipe_data."}),
                "model":           ("MODEL",             {"tooltip": "Pass-through Model"}),
                "clip":            ("CLIP",              {"tooltip": "Pass-through CLIP"}),
                "positive":        ("CONDITIONING",      {"tooltip": "Pass-through positive conditioning"}),
                "negative":        ("CONDITIONING",      {"tooltip": "Pass-through negative conditioning"}),
                "vae":             ("VAE",               {"tooltip": "Pass-through VAE"}),
                "latent":          ("LATENT",            {"tooltip": "Pass-through latent"}),
                "image":           ("IMAGE",             {"tooltip": "Pass-through image"}),
                "mask":            ("MASK",              {"tooltip": "Pass-through mask"}),
                "extra_1":         ("*",                 {"tooltip": "Pass-through extra data (any type)"}),
                "extra_2":         ("*",                 {"tooltip": "Pass-through extra data (any type)"}),
                "seed":            ("INT",               {"forceInput": True, "tooltip": "Override sampler seed"}),
                "steps":           ("INT",               {"forceInput": True, "tooltip": "Override sampling steps"}),
                "cfg":             ("FLOAT",             {"forceInput": True, "tooltip": "Override CFG scale"}),
                "sampler_name":    (cls._SAMPLER_ENUM,   {"forceInput": True, "tooltip": "Override sampler name"}),
                "scheduler":       (cls._SCHEDULER_ENUM, {"forceInput": True, "tooltip": "Override scheduler"}),
                "denoise":         ("FLOAT",             {"forceInput": True, "tooltip": "Override denoise"}),
                "pos_prompt":      ("STRING",            {"forceInput": True, "tooltip": "Override positive prompt"}),
                "neg_prompt":      ("STRING",            {"forceInput": True, "tooltip": "Override negative prompt"}),
                "lora_stack":      ("LORA_STACK",        {"tooltip": "Override LoRA stack"}),
                "width":           ("INT",               {"forceInput": True, "tooltip": "Override width"}),
                "height":          ("INT",               {"forceInput": True, "tooltip": "Override height"}),
                "batch_size":      ("INT",               {"forceInput": True, "tooltip": "Override batch size"}),
                "length":          ("INT",               {"forceInput": True, "tooltip": "Override video length"}),
                "model_name":      ("STRING",            {"forceInput": True, "tooltip": "Optional model name/path. Resolves and writes recipe model field(s)."}),
                "family":          ("STRING",            {"forceInput": True, "tooltip": "Optional model family override. Writes selected slot family in recipe_data."}),
                "model_data":      ("*",                 {"forceInput": True, "tooltip": "Optional structured model payload from Recipe Model Picker."}),
            },
        }

    RETURN_TYPES = (
        "RECIPE_DATA",
        "MODEL", "CLIP", "CONDITIONING", "CONDITIONING", "VAE", "LATENT", "IMAGE", "MASK", "*", "*",
        "INT", "INT", "FLOAT",
        _LIVE_SAMPLERS, _LIVE_SCHEDULERS, "FLOAT",
        "STRING", "STRING",
        "LORA_STACK",
        "INT", "INT", "INT", "INT",
        "STRING", "STRING",
        _LIVE_CKPTS, _LIVE_UNETS, _LIVE_VAES, _LIVE_CLIPS,
    )
    RETURN_NAMES = (
        "recipe_data",
        "model", "clip", "positive", "negative", "vae", "latent", "image", "mask", "extra_1", "extra_2",
        "seed", "steps", "cfg",
        "sampler_name", "scheduler", "denoise",
        "pos_prompt", "neg_prompt",
        "lora_stack",
        "width", "height", "batch_size", "length",
        "model_name", "family",
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
                    # Some valid LoRA identifiers resolve via full-path lookup
                    # but not via relative-name lookup. Keep availability in
                    # sync with relay output path resolution behavior.
                    _resolved_full, resolved_available = resolve_lora_path(raw_name)
                    if resolved_available:
                        available = True
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

            # Some stored payloads contain absolute paths or names with extensions.
            # Retry with basename and extension-stripped basename before declaring missing.
            candidate_base = os.path.basename(candidate.replace("\\", "/")).strip()
            if candidate_base:
                rel_path, available = get_lora_relative_path(candidate_base)
                if available and rel_path:
                    return rel_path, True

                candidate_base_no_ext = strip_lora_extension(candidate_base)
                if candidate_base_no_ext and candidate_base_no_ext != candidate_base:
                    rel_path, available = get_lora_relative_path(candidate_base_no_ext)
                    if available and rel_path:
                        return rel_path, True

            _, resolved_available = resolve_lora_path(candidate)
            if resolved_available:
                rel_path, available = get_lora_relative_path(candidate)
                if available and rel_path:
                    return rel_path, True

            if candidate_base:
                _, resolved_available = resolve_lora_path(candidate_base)
                if resolved_available:
                    rel_path, available = get_lora_relative_path(candidate_base)
                    if available and rel_path:
                        return rel_path, True
                candidate_base_no_ext = strip_lora_extension(candidate_base)
                if candidate_base_no_ext and candidate_base_no_ext != candidate_base:
                    _, resolved_available = resolve_lora_path(candidate_base_no_ext)
                    if resolved_available:
                        rel_path, available = get_lora_relative_path(candidate_base_no_ext)
                        if available and rel_path:
                            return rel_path, True

            return candidate_base or candidate, False

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

        has_incoming_data = isinstance(incoming_wf, dict) and bool(incoming_wf)
        has_slot_write_request = any([
            kwargs.get('lora_stack') is not None,
            any(kwargs.get(k) is not None for k in ('pos_prompt', 'neg_prompt')),
            any(kwargs.get(k) is not None for k in ('steps', 'seed', 'cfg', 'denoise', 'sampler_name', 'scheduler')),
            any(kwargs.get(k) is not None for k in ('width', 'height', 'batch_size', 'length')),
            isinstance(kwargs.get('model_name'), str) and kwargs.get('model_name').strip() != '',
            kwargs.get('family') is not None,
            kwargs.get('model_data') is not None,
        ])
        create_block_on_write = (not has_incoming_data) or has_slot_write_request

        # Preserve incoming recipe_data exactly when present.
        # Only fall back to the empty template when no input payload exists.
        if has_incoming_data:
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

        # Relay now treats recipe payloads as v2-only canonical data.
        wf['version'] = 2
        if not isinstance(wf.get('models'), dict):
            wf['models'] = {}

        selected_slot = _normalize_model_slot(model_slot)
        selected_block = _selected_model_block(wf, selected_slot, create=create_block_on_write)

        slot_family = selected_block.get('family', '') if isinstance(selected_block, dict) else ''
        slot_model_name = selected_block.get('model', '') if isinstance(selected_block, dict) else ''
        slot_positive = selected_block.get('positive_prompt', '') if isinstance(selected_block, dict) else ''
        slot_negative = selected_block.get('negative_prompt', '') if isinstance(selected_block, dict) else ''
        slot_vae = selected_block.get('vae', '') if isinstance(selected_block, dict) else ''
        slot_clip = selected_block.get('clip', []) if isinstance(selected_block, dict) else []
        slot_loras = selected_block.get('loras', []) if isinstance(selected_block, dict) else []
        slot_sampler = selected_block.get('sampler', {}) if isinstance(selected_block, dict) and isinstance(selected_block.get('sampler'), dict) else {}
        slot_resolution = selected_block.get('resolution', {}) if isinstance(selected_block, dict) and isinstance(selected_block.get('resolution'), dict) else {}

        sampler = dict(_relay_default_sampler())
        if isinstance(wf.get('sampler'), dict):
            sampler.update(wf.get('sampler', {}))
        if isinstance(slot_sampler, dict):
            sampler.update(slot_sampler)

        resolution = dict(_relay_default_resolution())
        if isinstance(wf.get('resolution'), dict):
            resolution.update(wf.get('resolution', {}))
        if isinstance(slot_resolution, dict):
            resolution.update(slot_resolution)

        # -- Apply top-level overrides -------------------------------
        pos_text = kwargs.get('pos_prompt')
        if pos_text is not None:
            target_block = _selected_model_block(wf, selected_slot, create=create_block_on_write)
            if isinstance(target_block, dict):
                target_block['positive_prompt'] = pos_text

        neg_text = kwargs.get('neg_prompt')
        if neg_text is not None:
            target_block = _selected_model_block(wf, selected_slot, create=create_block_on_write)
            if isinstance(target_block, dict):
                target_block['negative_prompt'] = neg_text

        # -- LoRA stack overrides ------------------------------------
        if kwargs.get('lora_stack') is not None:
            normalized_stack = _normalize_lora_stack_input(kwargs['lora_stack'])
            target_block = _selected_model_block(wf, selected_slot, create=create_block_on_write)
            if isinstance(target_block, dict):
                target_block['loras'] = normalized_stack

        # -- Resolution overrides ------------------------------------
        for key in ('width', 'height', 'batch_size', 'length'):
            if kwargs.get(key) is not None:
                resolution[key] = kwargs[key]
        wf['resolution'] = resolution

        # # -- Sampler overrides ---------------------------------------
        # if kwargs.get('steps') is not None:
        #     sampler['steps_a'] = kwargs['steps']
        #     sampler['steps_b'] = kwargs['steps']
        # if kwargs.get('seed') is not None:
        #     sampler['seed_a'] = kwargs['seed']
        #     sampler['seed_b'] = kwargs['seed']

        for key in ('cfg', 'denoise', 'sampler_name', 'scheduler'):
            if kwargs.get(key) is not None:
                sampler[key] = kwargs[key]
        wf['sampler'] = sampler

        target_block = _selected_model_block(wf, selected_slot, create=create_block_on_write)
        if isinstance(target_block, dict):
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
            target_block = _selected_model_block(wf, selected_slot, create=create_block_on_write)
            if isinstance(target_block, dict):
                target_block['model'] = chosen_name

            family = get_model_family(chosen_name)
            if family:
                family_spec = MODEL_FAMILIES.get(family, {})
                target_block = _selected_model_block(wf, selected_slot, create=create_block_on_write)
                if isinstance(target_block, dict):
                    target_block['family'] = family
                    target_block['clip_type'] = family_spec.get('clip_type', '')
                    target_block['loader_type'] = 'checkpoint' if family_spec.get('checkpoint', False) else 'unet'

        # -- Pass-through MODEL / CLIP / VAE -------------------------
        # Priority: explicit connected inputs > embedded objects in workflow_data.
        model = kwargs.get('model')
        if model is None:
            model = selected_block.get('MODEL') if isinstance(selected_block, dict) else None
        # if model is None:
        #     model = wf.get('MODEL')
        # if model is None:
        #     model = wf.get('MODEL_A')

        selected_loader_type = (
            str((selected_block or {}).get('loader_type', '')).strip().lower()
            if isinstance(selected_block, dict)
            else ''
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

        extra_1 = kwargs.get('extra_1')
        if extra_1 is None:
            extra_1 = wf.get('EXTRA_1')

        extra_2 = kwargs.get('extra_2')
        if extra_2 is None:
            extra_2 = wf.get('EXTRA_2')

        if latent is not None:
            wf['LATENT'] = latent
        if image is not None:
            wf['IMAGE'] = image
        if mask is not None:
            wf['MASK'] = mask
        if extra_1 is not None:
            wf['EXTRA_1'] = extra_1
        if extra_2 is not None:
            wf['EXTRA_2'] = extra_2

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
            # Ignore persisted availability flags; they may be stale across sessions.
            # Use current filesystem resolution as source of truth.
            return available

        legacy_slot_loras = []
        lora_stack = [
            (_resolve_lora_output_path(item)[0], item.get('model_strength', 1.0), item.get('clip_strength', 1.0))
            for item in legacy_slot_loras
            if _is_active_available(item)
        ]
        final_selected = _selected_model_block(wf, selected_slot, create=create_block_on_write)
        selected_loras = final_selected.get('loras', []) if isinstance(final_selected, dict) and isinstance(final_selected.get('loras'), list) else []
        if selected_loras:
            lora_stack = [
                (_resolve_lora_output_path(item)[0], item.get('model_strength', 1.0), item.get('clip_strength', 1.0))
                for item in selected_loras
                if _is_active_available(item)
            ]
        elif not lora_stack and isinstance(slot_loras, list):
            lora_stack = [
                (_resolve_lora_output_path(item)[0], item.get('model_strength', 1.0), item.get('clip_strength', 1.0))
                for item in slot_loras
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
            final_model_name = _short_display_name(slot_model_name)

        final_pos = (final_selected.get('positive_prompt', '') if isinstance(final_selected, dict) else '') or slot_positive
        final_neg = (final_selected.get('negative_prompt', '') if isinstance(final_selected, dict) else '') or slot_negative

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

        selected_model_raw = (final_selected.get('model', '') if isinstance(final_selected, dict) else '') or slot_model_name or ''
        selected_vae_raw = (final_selected.get('vae', '') if isinstance(final_selected, dict) else '') or slot_vae or ''
        selected_clip_raw = ''
        if isinstance(final_selected, dict) and isinstance(final_selected.get('clip'), list) and final_selected.get('clip'):
            selected_clip_raw = str(final_selected.get('clip')[0] or '').strip()
        elif isinstance(slot_clip, list) and slot_clip:
            selected_clip_raw = str(slot_clip[0] or '').strip()

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

        target_block = _selected_model_block(wf, selected_slot, create=create_block_on_write)
        if isinstance(target_block, dict):
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

        if selected_model_name:
            detected_family = get_model_family(selected_model_name)
            if detected_family:
                detected_block = _selected_model_block(wf, selected_slot, create=create_block_on_write)
                if isinstance(detected_block, dict):
                    detected_block['family'] = detected_family

        family_in = kwargs.get('family')
        if family_in is not None:
            family_override = str(family_in or '').strip()
            family_block = _selected_model_block(wf, selected_slot, create=create_block_on_write)
            if isinstance(family_block, dict):
                family_block['family'] = family_override

        # -- Extract all output values -------------------------------
        model_name = _short_display_name(selected_model_name) or final_model_name
        family_out = ''
        current_block = _selected_model_block(wf, selected_slot, create=False)
        if isinstance(current_block, dict):
            family_out = str(current_block.get('family', '') or '').strip()
        if not family_out:
            family_out = str(slot_family or '').strip()

        resolved_ckpt_name, ckpt_found = _resolve_rel_path(selected_model_name, ("checkpoints",))
        resolved_unet_name, unet_found = _resolve_rel_path(selected_model_name, ("diffusion_models", "unet", "unet_gguf"))

        # Infer loader type when metadata is missing but model path resolves.
        if selected_loader_type not in ('checkpoint', 'unet', 'diffusion'):
            if ckpt_found and not unet_found:
                selected_loader_type = 'checkpoint'
            elif unet_found and not ckpt_found:
                selected_loader_type = 'unet'
            elif ckpt_found and unet_found:
                family_spec = MODEL_FAMILIES.get(family_out, {}) if family_out else {}
                selected_loader_type = 'checkpoint' if family_spec.get('checkpoint', False) else 'unet'

        # Always emit string outputs for combo-name sockets to avoid None flowing
        # into downstream standard loader nodes.
        if ckpt_name_in:
            ckpt_out = ckpt_name_in
        elif selected_loader_type == 'checkpoint':
            ckpt_out = resolved_ckpt_name if ckpt_found else (selected_model_name or "")
        elif ckpt_found and not unet_found:
            ckpt_out = resolved_ckpt_name
        else:
            ckpt_out = ""

        if unet_name_in:
            unet_out = unet_name_in
        elif selected_loader_type in ('unet', 'diffusion'):
            unet_out = resolved_unet_name if unet_found else (selected_model_name or "")
        elif unet_found and not ckpt_found:
            unet_out = resolved_unet_name
        else:
            unet_out = ""
        vae_name_out = selected_vae_name or ""
        clip_name_out = selected_clip_name or ""

        return (
            wf,
            model, clip, positive, negative, vae, latent, image, mask, extra_1, extra_2,
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
            family_out,
            ckpt_out,
            unet_out,
            vae_name_out,
            clip_name_out,
        )
