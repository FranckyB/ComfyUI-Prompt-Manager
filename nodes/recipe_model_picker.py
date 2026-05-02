"""
Recipe Model Picker

Builder-style model selector that outputs structured model_data for Recipe Relay.
"""

import folder_paths
from ..py.workflow_families import (
    MODEL_FAMILIES,
    get_all_family_labels,
    get_model_family,
    get_compatible_families,
    list_compatible_vaes,
    list_compatible_clips,
)


DEFAULT_ASSET = "(Default)"


def _gather_models(folder_names):
    gathered = []
    seen = set()
    for folder_name in folder_names:
        try:
            for model_name in folder_paths.get_filename_list(folder_name):
                if model_name not in seen:
                    seen.add(model_name)
                    gathered.append(model_name)
        except Exception:
            continue
    return gathered


def _list_models_for_family(family_key):
    family_key = str(family_key or "").strip()
    if not family_key:
        return sorted(_gather_models(["checkpoints", "diffusion_models", "unet", "unet_gguf"]))

    compat = get_compatible_families(family_key)
    compat_specs = [MODEL_FAMILIES.get(fam, {}) for fam in compat]

    preferred_folders = []
    for spec in compat_specs:
        for folder_name in (spec.get("model_folders", []) or []):
            if folder_name and folder_name not in preferred_folders:
                preferred_folders.append(folder_name)

    if preferred_folders:
        model_folders = preferred_folders
    elif family_key == "sdxl":
        model_folders = ["checkpoints"]
    else:
        model_folders = ["diffusion_models", "unet", "unet_gguf"]

    all_models = _gather_models(model_folders)
    models = [m for m in all_models if get_model_family(m) in compat]

    if not models:
        compat_has_checkpoint = any(spec.get("checkpoint", False) for spec in compat_specs if spec)
        if compat_has_checkpoint and family_key == "sdxl":
            models = _gather_models(["checkpoints"])
        else:
            folder_patterns = []
            name_patterns = []
            for fam in compat:
                spec = MODEL_FAMILIES.get(fam, {})
                folder_patterns.extend([
                    p.lower().replace("\\", "/").strip("/")
                    for p in spec.get("folders", []) if p
                ])
                name_patterns.extend([p.lower() for p in spec.get("names", []) if p])

            relaxed = []
            for model_name in all_models:
                model_lower = model_name.lower().replace("\\", "/")
                folder_hit = any(fp and fp in model_lower for fp in folder_patterns)
                name_hit = any(pat and pat in model_lower for pat in name_patterns)
                if folder_hit or name_hit:
                    relaxed.append(model_name)
            models = relaxed

    if family_key == "wan_video_i2v" and models:
        i2v_t2v = []
        for model_name in models:
            lower_name = model_name.lower().replace("\\", "/")
            if ("i2v" in lower_name) or ("t2v" in lower_name):
                i2v_t2v.append(model_name)
        if i2v_t2v:
            models = i2v_t2v

    return sorted(models)


def _family_maps():
    labels = dict(get_all_family_labels() or {})
    labels.pop("ltxv", None)
    keys = sorted(labels.keys(), key=lambda k: (k != "sdxl", labels.get(k, k).lower()))
    label_to_key = {labels[k]: k for k in keys}
    type_values = [labels[k] for k in keys]
    return labels, label_to_key, type_values


def _family_key_from_type_value(type_value):
    labels, label_to_key, _ = _family_maps()
    value = str(type_value or "").strip()
    if value in labels:
        return value
    if value in label_to_key:
        return label_to_key[value]
    return "sdxl"

class RecipeModelPicker:
    """Select family/model/vae/clip and output a single model_data payload."""

    @classmethod
    def _sync_inputs(cls):
        labels, _label_to_key, type_values = _family_maps()
        default_family = "sdxl" if "sdxl" in labels else (next(iter(labels.keys()), "sdxl"))
        default_type = labels.get(default_family, "SDXL")

        model_values = _list_models_for_family(default_family)
        if not model_values:
            model_values = [""]

        vae_values = [DEFAULT_ASSET] + list_compatible_vaes(default_family)
        clip_values = [DEFAULT_ASSET] + list_compatible_clips(default_family)

        cls._TYPE_VALUES = tuple(type_values or ["SDXL"])
        cls._MODEL_VALUES = tuple(model_values)
        cls._VAE_VALUES = tuple(vae_values)
        cls._CLIP_VALUES = tuple(clip_values)
        cls._DEFAULT_TYPE = default_type

    @classmethod
    def INPUT_TYPES(cls):
        cls._sync_inputs()
        return {
            "required": {
                "type": (cls._TYPE_VALUES, {"default": cls._DEFAULT_TYPE, "tooltip": "Model family (Builder type)."}),
                "model": (cls._MODEL_VALUES, {"tooltip": "Primary model for selected family."}),
                "vae": (cls._VAE_VALUES, {"default": DEFAULT_ASSET, "tooltip": "Optional VAE override; (Default) keeps family defaults."}),
                "clip": (cls._CLIP_VALUES, {"default": DEFAULT_ASSET, "tooltip": "Optional CLIP override; (Default) keeps family defaults."}),
            }
        }

    RETURN_TYPES = ("ANY",)
    RETURN_NAMES = ("model_data",)
    FUNCTION = "pick"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Builder-style Type/Model/VAE/CLIP selector that outputs model_data for Recipe Relay."

    def pick(self, type, model, vae, clip):
        family = _family_key_from_type_value(type)
        family_spec = MODEL_FAMILIES.get(family, {})

        selected_model = str(model or "").strip()
        if selected_model:
            detected = get_model_family(selected_model)
            if detected and not family:
                family = detected
                family_spec = MODEL_FAMILIES.get(family, {})

        loader_type = "checkpoint" if family_spec.get("checkpoint", False) else "unet"
        clip_type = family_spec.get("clip_type", "")

        selected_vae = str(vae or "").strip()
        if selected_vae == DEFAULT_ASSET:
            selected_vae = ""

        selected_clip = str(clip or "").strip()
        if selected_clip == DEFAULT_ASSET:
            selected_clip = ""

        model_data = {
            "model": selected_model,
            "family": family,
            "loader_type": loader_type,
            "clip_type": clip_type,
            "vae": selected_vae,
            "clip": [selected_clip] if selected_clip else [],
        }

        return (model_data,)
