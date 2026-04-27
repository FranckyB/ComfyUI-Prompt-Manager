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


def _as_clip_list(raw_clip):
    if isinstance(raw_clip, list):
        return list(raw_clip)
    if isinstance(raw_clip, tuple):
        return list(raw_clip)
    if raw_clip:
        return [raw_clip]
    return []


def _has_meaningful_legacy_slot(recipe_data, slot_key, sampler):
    if slot_key not in ("model_a", "model_b"):
        return False

    suffix = "a" if slot_key == "model_a" else "b"
    model_name = str(recipe_data.get(slot_key, "") or "").strip()
    loras_key = f"loras_{suffix}"
    loras = recipe_data.get(loras_key, []) if isinstance(recipe_data.get(loras_key), list) else []
    has_loras = len(loras) > 0

    steps_key = f"steps_{suffix}"
    seed_key = f"seed_{suffix}"
    has_slot_sampler = isinstance(sampler.get(steps_key), (int, float)) or isinstance(sampler.get(seed_key), (int, float))

    return bool(model_name or has_loras or has_slot_sampler)


def _legacy_to_v2_recipe_data(recipe_data, source):
    sampler = recipe_data.get("sampler", {}) if isinstance(recipe_data.get("sampler"), dict) else {}
    resolution = recipe_data.get("resolution", {}) if isinstance(recipe_data.get("resolution"), dict) else {}

    def _build_model_block(slot_key):
        is_b = slot_key == "model_b"
        suffix = "b" if is_b else "a"
        loras_key = f"loras_{suffix}"

        steps_key = f"steps_{suffix}"
        seed_key = f"seed_{suffix}"
        steps = sampler.get(steps_key)
        if steps is None:
            steps = sampler.get("steps_a", sampler.get("steps", 20))

        seed = sampler.get(seed_key)
        if seed is None:
            seed = sampler.get("seed_a", sampler.get("seed", 0))

        return {
            "positive_prompt": str(recipe_data.get("positive_prompt", "") or ""),
            "negative_prompt": str(recipe_data.get("negative_prompt", "") or ""),
            "family": str(recipe_data.get("family", "") or ""),
            "model": str(recipe_data.get(slot_key, "") or ""),
            "loras": list(recipe_data.get(loras_key, [])) if isinstance(recipe_data.get(loras_key), list) else [],
            "clip_type": str(recipe_data.get("clip_type", "") or ""),
            "loader_type": str(recipe_data.get("loader_type", "") or ""),
            "vae": str(recipe_data.get("vae", "") or ""),
            "clip": _as_clip_list(recipe_data.get("clip", [])),
            "sampler": {
                "steps": int(steps if steps is not None else 20),
                "cfg": float(sampler.get("cfg", 5.0)),
                "denoise": float(sampler.get("denoise", 1.0)),
                "seed": int(seed if seed is not None else 0),
                "sampler_name": str(sampler.get("sampler_name", "euler") or "euler"),
                "scheduler": str(sampler.get("scheduler", "simple") or "simple"),
            },
            "resolution": {
                "width": int(resolution.get("width", 768)),
                "height": int(resolution.get("height", 1280)),
                "batch_size": int(resolution.get("batch_size", 1)),
                "length": resolution.get("length", None),
            },
        }

    out = {
        "_source": str(recipe_data.get("_source") or source),
        "version": 2,
        "models": {},
    }

    # Preserve non-legacy metadata/runtime keys at root.
    for key, val in recipe_data.items():
        if key in {"_source", "version", "models"}:
            continue
        if key in _LEGACY_TOP_LEVEL_KEYS:
            continue
        out[key] = val

    if _has_meaningful_legacy_slot(recipe_data, "model_a", sampler):
        out["models"]["model_a"] = _build_model_block("model_a")

    if _has_meaningful_legacy_slot(recipe_data, "model_b", sampler):
        out["models"]["model_b"] = _build_model_block("model_b")

    return out


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
    """Normalize recipe_data to strict v2 shape.

    v2 contract is authoritative. Legacy top-level recipe fields are converted
    into v2 model blocks.
    """
    if not isinstance(recipe_data, dict):
        return {
            "_source": source,
            "version": 2,
            "models": {},
        }

    is_v2 = int(recipe_data.get("version", 0) or 0) >= 2 and isinstance(recipe_data.get("models"), dict)
    if not is_v2:
        return _legacy_to_v2_recipe_data(recipe_data, source)

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
