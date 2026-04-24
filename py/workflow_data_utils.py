"""
Utility functions for RECIPE_DATA dicts.
Runtime object keys (MODEL_A, MODEL_B, CLIP, VAE, POSITIVE, NEGATIVE, etc.)
store non-serializable ComfyUI objects and must be stripped before JSON
serialization.
"""

import json
import math

# Keys that hold runtime (non-JSON-serializable) objects/conditioning payloads.
# These should never be persisted in saved prompt/workflow metadata.
RUNTIME_OBJECT_KEYS = frozenset({
    "MODEL",
    "MODEL_A",
    "MODEL_B",
    "CLIP",
    "VAE",
    "LATENT",
    "IMAGE",
    "MASK",
    "EXTRA",
    "POSITIVE",
    "NEGATIVE",
})

_MODEL_KEYS = ("model_a", "model_b", "model_c", "model_d")
_LEGACY_TOP_LEVEL_KEYS = {
    "family",
    "model_a",
    "model_b",
    "positive_prompt",
    "negative_prompt",
    "loras_a",
    "loras_b",
    "vae",
    "clip",
    "clip_type",
    "loader_type",
    "sampler",
    "resolution",
}


def _default_v2_model_block():
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
        "sampler": {
            "steps": 20,
            "cfg": 5.0,
            "denoise": 1.0,
            "seed": 0,
            "sampler_name": "euler",
            "scheduler": "simple",
        },
        "resolution": {
            "width": 768,
            "height": 1280,
            "batch_size": 1,
            "length": None,
        },
    }


def get_v2_model_block(recipe_data, model_key):
    """Return a v2 model block dict for model_key, or None if unavailable."""
    if not isinstance(recipe_data, dict):
        return None
    if int(recipe_data.get("version", 0) or 0) < 2:
        return None
    models = recipe_data.get("models", {}) if isinstance(recipe_data.get("models"), dict) else {}
    block = models.get(model_key)
    return block if isinstance(block, dict) else None


def ensure_v2_recipe_data(recipe_data, source="RecipeData"):
    """Normalize recipe_data to v2 shape.

    - Accepts both legacy v1-ish payloads and current v2 payloads.
    - Preserves unknown/root passthrough fields (e.g. IMAGE/LATENT/MASK).
    - Applies legacy top-level recipe fields onto model_a/model_b blocks.
    """
    if not isinstance(recipe_data, dict):
        return {
            "_source": source,
            "version": 2,
            "models": {},
        }

    is_v2 = int(recipe_data.get("version", 0) or 0) >= 2 and isinstance(recipe_data.get("models"), dict)

    out = {
        "_source": str(recipe_data.get("_source") or source),
        "version": 2,
        "models": {},
    }

    # Preserve non-legacy root keys as passthrough metadata/runtime payload.
    for key, val in recipe_data.items():
        if key in {"_source", "version", "models"}:
            continue
        if key in _LEGACY_TOP_LEVEL_KEYS:
            continue
        out[key] = val

    if is_v2:
        for mk in _MODEL_KEYS:
            block = recipe_data.get("models", {}).get(mk)
            if isinstance(block, dict):
                out["models"][mk] = dict(block)

    family = str(recipe_data.get("family", "") or "").strip()
    vae = str(recipe_data.get("vae", "") or "")
    clip_raw = recipe_data.get("clip", [])
    if isinstance(clip_raw, list):
        clip_names = clip_raw
    elif isinstance(clip_raw, str) and clip_raw:
        clip_names = [clip_raw]
    else:
        clip_names = []
    clip_type = str(recipe_data.get("clip_type", "") or "")
    loader_type = str(recipe_data.get("loader_type", "") or "")

    sampler = recipe_data.get("sampler", {}) if isinstance(recipe_data.get("sampler"), dict) else {}
    resolution = recipe_data.get("resolution", {}) if isinstance(recipe_data.get("resolution"), dict) else {}

    def _overlay_slot(slot_key, model_name, loras, prompt_override=True):
        if not model_name and not loras and not prompt_override and slot_key not in out["models"]:
            return
        block = dict(out["models"].get(slot_key, _default_v2_model_block()))
        if prompt_override:
            block["positive_prompt"] = str(recipe_data.get("positive_prompt", block.get("positive_prompt", "")) or "")
            block["negative_prompt"] = str(recipe_data.get("negative_prompt", block.get("negative_prompt", "")) or "")
        if family:
            block["family"] = family
        if model_name:
            block["model"] = model_name
        if isinstance(loras, list):
            block["loras"] = loras
        if clip_type:
            block["clip_type"] = clip_type
        if loader_type:
            block["loader_type"] = loader_type
        if vae:
            block["vae"] = vae
        if clip_names:
            block["clip"] = clip_names

        bs = block.get("sampler", {}) if isinstance(block.get("sampler"), dict) else {}
        if slot_key == "model_b":
            if sampler.get("steps_b") is not None:
                bs["steps"] = sampler.get("steps_b")
            elif sampler.get("steps_a") is not None:
                bs["steps"] = sampler.get("steps_a")
            elif sampler.get("steps") is not None:
                bs["steps"] = sampler.get("steps")
            if sampler.get("seed_b") is not None:
                bs["seed"] = sampler.get("seed_b")
            elif sampler.get("seed_a") is not None:
                bs["seed"] = sampler.get("seed_a")
            elif sampler.get("seed") is not None:
                bs["seed"] = sampler.get("seed")
        else:
            if sampler.get("steps_a") is not None:
                bs["steps"] = sampler.get("steps_a")
            elif sampler.get("steps") is not None:
                bs["steps"] = sampler.get("steps")
            if sampler.get("seed_a") is not None:
                bs["seed"] = sampler.get("seed_a")
            elif sampler.get("seed") is not None:
                bs["seed"] = sampler.get("seed")

        for key in ("cfg", "denoise", "sampler_name", "scheduler"):
            if sampler.get(key) is not None:
                bs[key] = sampler.get(key)
        if bs:
            block["sampler"] = bs

        br = block.get("resolution", {}) if isinstance(block.get("resolution"), dict) else {}
        for key in ("width", "height", "batch_size", "length"):
            if resolution.get(key) is not None:
                br[key] = resolution.get(key)
        if br:
            block["resolution"] = br

        out["models"][slot_key] = block

    model_a = str(recipe_data.get("model_a", "") or "")
    model_b = str(recipe_data.get("model_b", "") or "")
    loras_a = recipe_data.get("loras_a", []) if isinstance(recipe_data.get("loras_a"), list) else []
    loras_b = recipe_data.get("loras_b", []) if isinstance(recipe_data.get("loras_b"), list) else []

    if model_a or loras_a or recipe_data.get("positive_prompt") or recipe_data.get("negative_prompt"):
        _overlay_slot("model_a", model_a, loras_a, prompt_override=True)
    if model_b or loras_b:
        _overlay_slot("model_b", model_b, loras_b, prompt_override=False)

    # Legacy runtime object mapping into v2 model blocks.
    if "MODEL_A" in recipe_data:
        block = dict(out["models"].get("model_a", _default_v2_model_block()))
        block["MODEL"] = recipe_data.get("MODEL_A")
        out["models"]["model_a"] = block
    if "MODEL_B" in recipe_data:
        block = dict(out["models"].get("model_b", _default_v2_model_block()))
        block["MODEL"] = recipe_data.get("MODEL_B")
        out["models"]["model_b"] = block
    if "CLIP" in recipe_data:
        block = dict(out["models"].get("model_a", _default_v2_model_block()))
        block["CLIP"] = recipe_data.get("CLIP")
        out["models"]["model_a"] = block
    if "VAE" in recipe_data:
        block = dict(out["models"].get("model_a", _default_v2_model_block()))
        block["VAE"] = recipe_data.get("VAE")
        out["models"]["model_a"] = block
    if "POSITIVE" in recipe_data:
        block = dict(out["models"].get("model_a", _default_v2_model_block()))
        block["POSITIVE"] = recipe_data.get("POSITIVE")
        out["models"]["model_a"] = block
    if "NEGATIVE" in recipe_data:
        block = dict(out["models"].get("model_a", _default_v2_model_block()))
        block["NEGATIVE"] = recipe_data.get("NEGATIVE")
        out["models"]["model_a"] = block

    return out


def _strip_runtime_keys_deep(value):
    """Recursively strip runtime-only keys from dict/list containers."""
    if isinstance(value, dict):
        return {
            k: _strip_runtime_keys_deep(v)
            for k, v in value.items()
            if k not in RUNTIME_OBJECT_KEYS
        }
    if isinstance(value, list):
        return [_strip_runtime_keys_deep(v) for v in value]
    if isinstance(value, tuple):
        return [_strip_runtime_keys_deep(v) for v in value]
    if isinstance(value, set):
        return [_strip_runtime_keys_deep(v) for v in value]
    return value


def strip_runtime_objects(wf):
    """Return a deep copy of wf with all runtime object keys removed.

    Use before JSON serialization (saving prompts, send_sync to JS, etc.).
    """
    if not isinstance(wf, dict):
        return wf
    return _strip_runtime_keys_deep(wf)


def _json_default(value):
    """Best-effort fallback for objects that JSON cannot encode directly."""
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def to_json_safe_workflow_data(wf):
    """Return a deep JSON-safe copy of workflow data with runtime keys removed.

    This is stricter than strip_runtime_objects(): it also sanitizes nested
    containers and unknown object types so metadata embedding cannot fail
    during Save Image serialization.
    """
    if not isinstance(wf, dict):
        return {}

    stripped = strip_runtime_objects(wf)

    # Normalize NaN/Inf because Python's json allows them by default but strict
    # JSON consumers may reject them.
    def _normalize_numbers(value):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, dict):
            return {k: _normalize_numbers(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_normalize_numbers(v) for v in value]
        if isinstance(value, tuple):
            return [_normalize_numbers(v) for v in value]
        if isinstance(value, set):
            return [_normalize_numbers(v) for v in value]
        return value

    normalized = _normalize_numbers(stripped)

    try:
        return json.loads(json.dumps(normalized, default=_json_default, allow_nan=False))
    except Exception:
        # Last-resort safety: stringify opaque values while preserving structure.
        return json.loads(json.dumps(str(normalized)))
