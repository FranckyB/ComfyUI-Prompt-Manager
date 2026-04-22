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
