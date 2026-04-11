"""
workflow_executor.py — Template-driven execution engine for WorkflowGenerator.

Loads an API-format workflow JSON + companion map JSON, patches values from
the node's override_data / extracted workflow, and executes the graph
in-process (model load → LoRA → sample → decode), returning (IMAGE, LATENT).

The template-based approach replaces the hardcoded Python graph-building
used previously.  Families without a template fall back to the existing
hardcoded paths in workflow_generator.py.
"""

import os
import json
import copy
import traceback

import folder_paths

from .workflow_families import FAMILY_WORKFLOW_STEMS


# ── Template directory ────────────────────────────────────────────────────────

def _get_template_dir():
    """Return the path to the workflows/api/ directory (inside the addon)."""
    addon_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(addon_root, "workflows", "api")


def load_template(family_key):
    """
    Load (api_dict, map_dict) for the given family key.
    Returns (None, None) if no template exists for this family.
    """
    stem = FAMILY_WORKFLOW_STEMS.get(family_key)
    if not stem:
        return None, None

    tdir = _get_template_dir()
    api_path = os.path.join(tdir, f"{stem}_api.json")
    map_path = os.path.join(tdir, f"{stem}_map.json")

    if not os.path.isfile(api_path) or not os.path.isfile(map_path):
        return None, None

    try:
        with open(api_path, "r", encoding="utf-8") as f:
            api = json.load(f)
        with open(map_path, "r", encoding="utf-8") as f:
            wmap = json.load(f)
        return api, wmap
    except Exception as e:
        print(f"[WorkflowExecutor] Failed to load template for {family_key}: {e}")
        return None, None


# ── Value patching ────────────────────────────────────────────────────────────

def _set(api, node_id, field, value):
    """Set a value in the API dict.  Silently ignores missing node/field."""
    if node_id is None or field is None:
        return
    node = api.get(str(node_id))
    if node is None:
        return
    node["inputs"][field] = value


def _get(api, node_id, field):
    """Get a value from the API dict, or None if missing."""
    if node_id is None or field is None:
        return None
    node = api.get(str(node_id))
    if node is None:
        return None
    return node["inputs"].get(field)


def patch_template(api, wmap, params):
    """
    Patch an API-format workflow dict in-place with the given params dict.

    params keys (all optional):
        positive_prompt   str
        negative_prompt   str
        model_a           str  (unet_name or ckpt_name)
        model_b           str
        vae               str  (vae_name, '' or '(default)' = keep template value)
        clip              str  (clip_name, '' or '(default)' = keep template value)
        clip_1            str  (DualCLIPLoader first encoder)
        clip_2            str  (DualCLIPLoader second encoder)
        width             int
        height            int
        length            int  (video frames)
        batch_size        int
        seed              int
        steps             int
        steps_high        int  (WAN Video high sampler)
        steps_low         int  (WAN Video low sampler)
        cfg               float
        sampler_name      str
        scheduler         str
        denoise           float
        guidance          float  (Flux1)
        lora_stack_a_text str  (JSON/text for PromptApplyLora)
        lora_stack_b_text str
    """
    def _m(key):
        """Return (node_id, field) tuple for a map key, or (None, None)."""
        v = wmap.get(key)
        if not v or not isinstance(v, list) or len(v) < 2:
            return None, None
        return str(v[0]), v[1]

    def _override(map_key, param_key, transform=None):
        val = params.get(param_key)
        if val is None:
            return
        nid, field = _m(map_key)
        if field is None:
            return
        _set(api, nid, field, transform(val) if transform else val)

    # ── Prompts ───────────────────────────────────────────────────────────────
    _override("positive_prompt", "positive_prompt")
    _override("negative_prompt", "negative_prompt")

    # ── Models ────────────────────────────────────────────────────────────────
    if params.get("model_a"):
        # Try unet_name first, then ckpt_name
        for mk in ["model_a", "checkpoint"]:
            nid, field = _m(mk)
            if field:
                _set(api, nid, field, params["model_a"])
                break
    if params.get("model_b"):
        nid, field = _m("model_b")
        if field:
            _set(api, nid, field, params["model_b"])

    # ── VAE (only override if non-empty and not "(default)") ─────────────────
    vae_val = params.get("vae", "")
    if vae_val and not str(vae_val).startswith("("):
        nid, field = _m("vae")
        if field:
            _set(api, nid, field, vae_val)

    # ── CLIP ─────────────────────────────────────────────────────────────────
    clip_val = params.get("clip", "")
    if clip_val and not str(clip_val).startswith("("):
        # Single CLIPLoader
        nid, field = _m("clip")
        if field:
            _set(api, nid, field, clip_val)
        # DualCLIPLoader first encoder
        nid1, f1 = _m("clip_1")
        if f1:
            _set(api, nid1, f1, clip_val)

    if params.get("clip_1") and not str(params["clip_1"]).startswith("("):
        nid, field = _m("clip_1")
        if field:
            _set(api, nid, field, params["clip_1"])
    if params.get("clip_2") and not str(params["clip_2"]).startswith("("):
        nid, field = _m("clip_2")
        if field:
            _set(api, nid, field, params["clip_2"])

    # ── Resolution ────────────────────────────────────────────────────────────
    if params.get("width") is not None:
        for mk in ["latent_width", "resize_width", "flux2_width"]:
            nid, field = _m(mk)
            if field:
                _set(api, nid, field, int(params["width"]))
    if params.get("height") is not None:
        for mk in ["latent_height", "resize_height", "flux2_height"]:
            nid, field = _m(mk)
            if field:
                _set(api, nid, field, int(params["height"]))
    if params.get("length") is not None:
        nid, field = _m("latent_length")
        if field:
            _set(api, nid, field, int(params["length"]))
    if params.get("batch_size") is not None:
        nid, field = _m("latent_batch")
        if field:
            _set(api, nid, field, int(params["batch_size"]))

    # ── Standard KSampler ─────────────────────────────────────────────────────
    _override("ksampler_seed",      "seed",         int)
    _override("ksampler_steps",     "steps",        int)
    _override("ksampler_cfg",       "cfg",          float)
    _override("ksampler_sampler",   "sampler_name")
    _override("ksampler_scheduler", "scheduler")
    _override("ksampler_denoise",   "denoise",      float)

    # ── Flux / SamplerCustomAdvanced ─────────────────────────────────────────
    _override("flux_seed",      "seed",      int)
    _override("flux_steps",     "steps",     int)
    _override("flux_guidance",  "guidance",  float)
    _override("flux_sampler",   "sampler_name")
    _override("flux_scheduler", "scheduler")
    _override("flux_denoise",   "denoise",   float)

    # ── Flux2 / CFGGuider ─────────────────────────────────────────────────────
    _override("flux_seed",   "seed",   int)   # RandomNoise
    _override("flux2_cfg",   "cfg",    float)
    _override("flux2_steps", "steps",  int)
    if params.get("width") is not None:
        nid, field = _m("flux2_width")
        if field:
            _set(api, nid, field, int(params["width"]))
    if params.get("height") is not None:
        nid, field = _m("flux2_height")
        if field:
            _set(api, nid, field, int(params["height"]))

    # ── WAN Video dual-sampler ────────────────────────────────────────────────
    steps_high = params.get("steps_high")
    steps_low  = params.get("steps_low")
    seed       = params.get("seed")
    cfg        = params.get("cfg")
    sampler    = params.get("sampler_name")
    scheduler  = params.get("scheduler")

    if steps_high is not None and steps_low is not None:
        total = int(steps_high) + int(steps_low)
        # High sampler: start=0, end=steps_high
        nid_h, _ = _m("ksampler_high_steps")
        if nid_h:
            _set(api, nid_h, "steps", total)
            _set(api, nid_h, "start_at_step", 0)
            _set(api, nid_h, "end_at_step", int(steps_high))
        # Low sampler: start=steps_high, end=100 (ComfyUI default max)
        nid_l, _ = _m("ksampler_low_steps")
        if nid_l:
            _set(api, nid_l, "steps", total)
            _set(api, nid_l, "start_at_step", int(steps_high))
            _set(api, nid_l, "end_at_step", 100)
    elif steps_high is not None:
        # Only high provided — use as total
        nid_h, _ = _m("ksampler_high_steps")
        if nid_h:
            _set(api, nid_h, "steps", int(steps_high))

    if seed is not None:
        for mk in ["ksampler_high_seed", "ksampler_low_seed"]:
            nid, field = _m(mk)
            if field:
                _set(api, nid, field, int(seed))
    if cfg is not None:
        for mk in ["ksampler_high_cfg", "ksampler_low_cfg"]:
            nid, field = _m(mk)
            if field:
                _set(api, nid, field, float(cfg))
    if sampler:
        for mk in ["ksampler_high_sampler", "ksampler_low_sampler"]:
            nid, field = _m(mk)
            if field:
                _set(api, nid, field, sampler)
    if scheduler:
        for mk in ["ksampler_high_scheduler", "ksampler_low_scheduler"]:
            nid, field = _m(mk)
            if field:
                _set(api, nid, field, scheduler)

    # ── LoRA stacks ───────────────────────────────────────────────────────────
    if params.get("lora_stack_a_text") is not None:
        nid, field = _m("lora_stack_a")
        if field:
            _set(api, nid, field, params["lora_stack_a_text"])
    if params.get("lora_stack_b_text") is not None:
        nid, field = _m("lora_stack_b")
        if field:
            _set(api, nid, field, params["lora_stack_b_text"])


# ── LoRA stack → text serialisation ──────────────────────────────────────────

def loras_to_text(lora_list, lora_overrides=None, stack_key=""):
    """
    Convert a list of LoRA dicts (from extracted workflow_data) to the JSON
    text format accepted by PromptApplyLora's lora_stack_text input.

    Format: JSON array of [name, model_strength, clip_strength]
    """
    lora_overrides = lora_overrides or {}
    result = []
    has_key = bool(stack_key)
    for lora in lora_list:
        name = lora.get("name", "")
        if not name:
            continue
        state_key = f"{stack_key}:{name}" if has_key else name
        ov = lora_overrides.get(state_key, lora_overrides.get(name, {}))
        if ov.get("active") is False:
            continue
        ms = float(ov.get("model_strength", lora.get("model_strength", 1.0)))
        cs = float(ov.get("clip_strength",  lora.get("clip_strength",  1.0)))
        result.append([name, ms, cs])
    return json.dumps(result)


# ── In-process node execution ─────────────────────────────────────────────────

def execute_template(api, wmap, family_key, source_image=None):
    """
    Execute a patched API-format workflow in-process.

    This imports and calls the ComfyUI node classes directly (same approach
    as the existing hardcoded samplers), but driven by the template graph
    rather than hardcoded Python.

    Returns (IMAGE tensor, LATENT dict) on success.
    Raises on failure.

    source_image: optional IMAGE tensor for i2v workflows.
    """
    import torch
    import comfy.sd
    import comfy.utils
    import comfy.samplers
    import comfy.sample
    import comfy.model_management
    import latent_preview

    from .lora_utils import resolve_lora_path

    def _m(key):
        v = wmap.get(key)
        if not v or not isinstance(v, list) or len(v) < 2:
            return None, None
        return str(v[0]), v[1]

    def _val(key):
        nid, field = _m(key)
        if field is None:
            return None
        return _get(api, nid, field)

    def _ref(key):
        """Get [src_node_id, src_slot] link reference for a map key."""
        nid, field = _m(key)
        if field is None:
            return None
        v = api.get(str(nid), {}).get("inputs", {}).get(field)
        return v if isinstance(v, list) else None

    # ── Step 1: Load models ───────────────────────────────────────────────────
    from ..nodes.workflow_generator import (
        _load_model_from_path, _load_vae, _load_clip, _apply_loras,
    )
    from .workflow_extraction_utils import resolve_model_name, resolve_vae_name, resolve_clip_names

    # Model A
    model_a_name = _val("model_a") or _val("checkpoint")
    if not model_a_name:
        raise ValueError("[WorkflowExecutor] No model_a / checkpoint in template")
    resolved_a, folder_a = resolve_model_name(model_a_name)
    if resolved_a is None:
        raise FileNotFoundError(f"[WorkflowExecutor] Model A not found: {model_a_name}")
    full_path_a = folder_paths.get_full_path(folder_a, resolved_a)
    model_a, clip_a, vae_a = _load_model_from_path(resolved_a, folder_a, full_path_a)

    # Model B (WAN Video only)
    model_b = clip_b_raw = None
    model_b_name = _val("model_b")
    if model_b_name:
        resolved_b, folder_b = resolve_model_name(model_b_name)
        if resolved_b:
            full_path_b = folder_paths.get_full_path(folder_b, resolved_b)
            model_b, clip_b_raw, _ = _load_model_from_path(resolved_b, folder_b, full_path_b)

    # VAE — read from the (already-patched) template.
    # If the user chose (Default), patch_template left the original template value
    # intact, so _val("vae") returns the template's filename (e.g.
    # "flux2-vae.safetensors"), which is what we load.  Only crashes if the file
    # is genuinely missing from disk.
    vae_name = _val("vae")
    vae = _load_vae(vae_name, existing_vae=vae_a)
    if vae is None:
        raise FileNotFoundError(
            f"[WorkflowExecutor] VAE not found on disk: {vae_name!r}. "
            f"Make sure the file exists in your ComfyUI/models/vae/ folder."
        )

    # CLIP — determine type from template CLIPLoader node
    clip_type_str = ""
    for nid, node in api.items():
        if node["class_type"] in ("CLIPLoader", "DualCLIPLoader"):
            clip_type_str = node["inputs"].get("type", "")
            break

    # Build clip_info
    clip_names = []
    c1 = _val("clip_1")
    c2 = _val("clip_2")
    c0 = _val("clip")
    if c1:
        clip_names.append(c1)
    if c2:
        clip_names.append(c2)
    if c0 and not clip_names:
        clip_names.append(c0)

    clip_info = {"names": clip_names, "type": clip_type_str, "source": "template"}
    clip = _load_clip(clip_info, {}, existing_clip=clip_a)
    if clip is None:
        raise FileNotFoundError("[WorkflowExecutor] No CLIP available")

    # ── Step 2: Apply LoRAs ───────────────────────────────────────────────────
    # lora_stack_text is already serialised JSON — parse it back to list
    def _parse_lora_text(text):
        if not text:
            return []
        try:
            return json.loads(text)
        except Exception:
            return []

    loras_a_text = _val("lora_stack_a") or ""
    loras_a = [{"name": r[0], "model_strength": r[1], "clip_strength": r[2]}
               for r in _parse_lora_text(loras_a_text) if len(r) >= 3]
    model_a, clip = _apply_loras(model_a, clip, loras_a, {})

    clip_b = clip_b_raw or clip
    if model_b is not None:
        loras_b_text = _val("lora_stack_b") or ""
        loras_b = [{"name": r[0], "model_strength": r[1], "clip_strength": r[2]}
                   for r in _parse_lora_text(loras_b_text) if len(r) >= 3]
        model_b, clip_b = _apply_loras(model_b, clip_b, loras_b, {})

    # ── Step 3: Encode prompts ────────────────────────────────────────────────
    pos_prompt = _val("positive_prompt") or ""
    neg_prompt = _val("negative_prompt") or ""

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos   = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg   = clip.encode_from_tokens_scheduled(tokens_neg)

    # ── Step 4: Latent ────────────────────────────────────────────────────────
    width  = int(_val("latent_width")  or _val("resize_width")  or 768)
    height = int(_val("latent_height") or _val("resize_height") or 1280)

    # Detect strategy from family
    strategy = _family_to_strategy(family_key)

    if strategy == "wan_video" and family_key == "wan_video_i2v" and source_image is not None:
        # i2v: latent comes from WanImageToVideo node — we call it directly
        latent_dict = _make_wan_i2v_latent(
            source_image, vae, cond_pos, cond_neg, width, height,
            int(_val("latent_length") or 81)
        )
        cond_pos, cond_neg = latent_dict.pop("_cond_pos"), latent_dict.pop("_cond_neg")
    else:
        length = _val("latent_length")
        batch  = 1
        if length is not None:
            # Video latent — channels=16 for WAN
            latent_tensor = torch.zeros(
                [int(length), 16, height // 8, width // 8],
                device=comfy.model_management.intermediate_device(),
            )
        else:
            latent_tensor = torch.zeros(
                [batch, 4, height // 8, width // 8],
                device=comfy.model_management.intermediate_device(),
            )
        latent_dict = {
            "samples": latent_tensor,
            "downscale_ratio_spacial": 8,
            "_width": width, "_height": height,
        }

    # ── Step 5: Sample ────────────────────────────────────────────────────────
    from ..nodes.workflow_generator import (
        _run_standard_ksampler, _run_flux_sampler, _run_flux2_sampler, _run_wan_sampler,
    )

    seed = int(_val("ksampler_seed") or _val("flux_seed") or
               _val("ksampler_high_seed") or 0)

    sampler_params = {
        "seed":        seed,
        "steps":       int(_val("ksampler_steps") or _val("flux_steps") or
                           _val("flux2_steps") or _val("ksampler_high_steps") or 20),
        "cfg":         float(_val("ksampler_cfg") or _val("flux2_cfg") or
                             _val("ksampler_high_cfg") or 5.0),
        "sampler_name": _val("ksampler_sampler") or _val("flux_sampler") or "euler",
        "scheduler":    _val("ksampler_scheduler") or _val("flux_scheduler") or "simple",
        "denoise":     float(_val("ksampler_denoise") or _val("flux_denoise") or 1.0),
        "guidance":    float(_val("flux_guidance") or 3.5),
    }

    if strategy == "wan_video":
        sampler_params_b = {
            "steps":       int(_val("ksampler_low_steps") or sampler_params["steps"]),
            "cfg":         float(_val("ksampler_low_cfg")  or sampler_params["cfg"]),
            "sampler_name": _val("ksampler_low_sampler")  or sampler_params["sampler_name"],
            "scheduler":    _val("ksampler_low_scheduler") or sampler_params["scheduler"],
            "seed":        seed + 1,
        }
        # For WAN Video, steps in our sampler_params are steps_high
        sampler_params["steps"] = int(_val("ksampler_high_steps") or 4)

        # Encode prompts for model B
        tokens_pos_b = clip_b.tokenize(pos_prompt)
        cond_pos_b   = clip_b.encode_from_tokens_scheduled(tokens_pos_b)
        tokens_neg_b = clip_b.tokenize(neg_prompt)
        cond_neg_b   = clip_b.encode_from_tokens_scheduled(tokens_neg_b)

        samples = _run_wan_sampler(
            model_a, cond_pos, cond_neg, latent_dict, sampler_params,
            model_b=model_b, cond_pos_b=cond_pos_b, cond_neg_b=cond_neg_b,
            sampler_params_b=sampler_params_b,
        )
    elif strategy == "wan_image":
        samples = _run_standard_ksampler(model_a, cond_pos, cond_neg, latent_dict, sampler_params)
    elif strategy == "flux2":
        samples = _run_flux2_sampler(model_a, cond_pos, cond_neg, latent_dict, sampler_params)
    elif strategy == "flux":
        samples = _run_flux_sampler(model_a, cond_pos, cond_neg, latent_dict, sampler_params)
    else:
        samples = _run_standard_ksampler(model_a, cond_pos, cond_neg, latent_dict, sampler_params)

    # ── Step 6: Decode ────────────────────────────────────────────────────────
    decoded = vae.decode(samples)
    out_latent = {"samples": samples}

    return decoded, out_latent


def _family_to_strategy(family_key):
    from .workflow_families import MODEL_FAMILIES
    spec = MODEL_FAMILIES.get(family_key, {})
    return spec.get("sampler", "standard")


def _make_wan_i2v_latent(source_image, vae, cond_pos, cond_neg, width, height, length):
    """
    Call WanImageToVideo node to produce the i2v latent conditioning.
    Returns a latent dict with _cond_pos and _cond_neg injected.
    """
    import torch
    import comfy.model_management
    try:
        from comfy_extras.nodes_wan import WanImageToVideo
        node = WanImageToVideo()
        result = node.encode(
            positive=cond_pos,
            negative=cond_neg,
            vae=vae,
            start_image=source_image,
            width=width,
            height=height,
            length=length,
            batch_size=1,
        )
        # result = (positive, negative, latent)
        new_cond_pos, new_cond_neg, latent = result
        latent["_cond_pos"] = new_cond_pos
        latent["_cond_neg"] = new_cond_neg
        return latent
    except Exception as e:
        print(f"[WorkflowExecutor] WanImageToVideo failed ({e}), falling back to empty latent")
        import comfy.model_management
        latent_tensor = torch.zeros(
            [length, 16, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return {
            "samples": latent_tensor,
            "_cond_pos": cond_pos,
            "_cond_neg": cond_neg,
        }
