"""
Utility functions for WORKFLOW_DATA dicts.
Runtime object keys (MODEL_A, MODEL_B, CLIP, VAE) store non-serializable
ComfyUI objects and must be stripped before JSON serialization.
"""

# Keys that hold runtime (non-JSON-serializable) objects
RUNTIME_OBJECT_KEYS = frozenset({"MODEL_A", "MODEL_B", "CLIP", "VAE"})


def strip_runtime_objects(wf):
    """Return a shallow copy of wf with all runtime object keys removed.

    Use before JSON serialization (saving prompts, send_sync to JS, etc.).
    """
    if not isinstance(wf, dict):
        return wf
    return {k: v for k, v in wf.items() if k not in RUNTIME_OBJECT_KEYS}
