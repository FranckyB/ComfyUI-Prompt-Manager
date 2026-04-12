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

    This keeps the JSON graph consistent with the user's actual values,
    which is useful for debugging / logging.  execute_template() reads
    params directly (not from the patched JSON) to avoid double-interpretation
    bugs, but having a fully-patched JSON makes it possible to inspect
    exactly what was sent (and would be needed if we ever switch to true
    graph execution via ComfyUI's engine).

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

    seed_b = params.get("seed_b")  # None means use same seed for both
    if seed is not None:
        nid, field = _m("ksampler_high_seed")
        if field:
            _set(api, nid, field, int(seed))
        nid, field = _m("ksampler_low_seed")
        if field:
            _set(api, nid, field, int(seed_b) if seed_b is not None else int(seed))
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

    # ── Source image for i2v ─────────────────────────────────────────────────
    if params.get("source_image_path") is not None:
        # wan_i2v_image maps directly to the LoadImage node's 'image' input
        nid, field = _m("wan_i2v_image")
        if nid is not None and field:
            _set(api, nid, field, params["source_image_path"])
            print(f"[patch_template] Set LoadImage node {nid}.{field} = {params['source_image_path']}")
        else:
            print("[patch_template] wan_i2v_image not in map — source image NOT patched")

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

def execute_template(api, wmap, family_key, params):
    """
    Execute a patched API-format workflow in-process.

    The template JSON + map are used as the source of truth for default
    values (model names, VAE, CLIP type, etc.).  The ``params`` dict
    carries the actual user-facing values already resolved by the caller
    (workflow_generator.py) — we use those directly instead of reading
    them back from the patched JSON, which avoids the double-interpretation
    bugs that plagued the WAN video dual-sampler path.

    params keys (from workflow_generator.py):
        model_a, model_b, vae, clip, clip_1, clip_2,
        positive_prompt, negative_prompt,
        width, height, length, batch_size,
        seed, seed_b, steps, steps_high, steps_low,
        cfg, sampler_name, scheduler, denoise, guidance,
        lora_stack_a_text, lora_stack_b_text,
        source_image_path  (i2v only)

    Returns (IMAGE tensor, LATENT dict) on success.
    Raises on failure.
    """
    stem = FAMILY_WORKFLOW_STEMS.get(family_key, family_key)
    strategy = _family_to_strategy(family_key)
    print(f"[WorkflowExecutor] Running template: {stem} "
          f"(family: {family_key}, strategy: {strategy})")

    import torch
    import comfy.sd
    import comfy.utils
    import comfy.samplers
    import comfy.sample
    import comfy.model_management
    import latent_preview

    # Helper: read a value from the patched template (used only for
    # defaults that aren't in params, like CLIP type).
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

    # Convenience: get from params first, then fall back to template value.
    def _p(param_key, *template_keys, transform=None, default=None):
        v = params.get(param_key)
        if v is None:
            for tk in template_keys:
                v = _val(tk)
                if v is not None:
                    break
        if v is None:
            return default
        return transform(v) if transform else v

    # ── Step 1: Load models ───────────────────────────────────────────────────
    from ..nodes.workflow_generator import (
        _load_model_from_path, _load_vae, _load_clip, _apply_loras,
    )
    from .workflow_extraction_utils import resolve_model_name

    # Model A
    model_a_name = _p("model_a", "model_a", "checkpoint")
    if not model_a_name:
        raise ValueError("[WorkflowExecutor] No model_a / checkpoint specified")
    resolved_a, folder_a = resolve_model_name(model_a_name)
    if resolved_a is None:
        raise FileNotFoundError(f"[WorkflowExecutor] Model A not found: {model_a_name}")
    full_path_a = folder_paths.get_full_path(folder_a, resolved_a)
    model_a, clip_a, vae_a = _load_model_from_path(resolved_a, folder_a, full_path_a)

    # Model B (WAN Video dual-model only)
    model_b = clip_b_raw = None
    model_b_name = _p("model_b", "model_b")
    if model_b_name:
        resolved_b, folder_b = resolve_model_name(model_b_name)
        if resolved_b:
            full_path_b = folder_paths.get_full_path(folder_b, resolved_b)
            model_b, clip_b_raw, _ = _load_model_from_path(resolved_b, folder_b, full_path_b)

    # VAE — params['vae'] is '' or '(Default)' when user wants the template
    # default.  patch_template already left the template value intact in that
    # case, so we read it from the template.
    vae_name = params.get("vae", "")
    if not vae_name or str(vae_name).startswith("("):
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
        if node.get("class_type") in ("CLIPLoader", "DualCLIPLoader"):
            clip_type_str = node["inputs"].get("type", "")
            break

    clip_names = []
    c1 = params.get("clip_1") or _val("clip_1")
    c2 = params.get("clip_2") or _val("clip_2")
    c0 = params.get("clip")   or _val("clip")
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
    def _parse_lora_text(text):
        if not text:
            return []
        try:
            return json.loads(text)
        except Exception:
            return []

    loras_a_text = params.get("lora_stack_a_text", "") or _val("lora_stack_a") or ""
    loras_a = [{"name": r[0], "model_strength": r[1], "clip_strength": r[2]}
               for r in _parse_lora_text(loras_a_text) if len(r) >= 3]
    model_a, clip = _apply_loras(model_a, clip, loras_a, {})

    clip_b = clip_b_raw or clip
    if model_b is not None:
        loras_b_text = params.get("lora_stack_b_text", "") or _val("lora_stack_b") or ""
        loras_b = [{"name": r[0], "model_strength": r[1], "clip_strength": r[2]}
                   for r in _parse_lora_text(loras_b_text) if len(r) >= 3]
        model_b, clip_b = _apply_loras(model_b, clip_b, loras_b, {})

    # ── Step 3: Encode prompts ────────────────────────────────────────────────
    pos_prompt = params.get("positive_prompt", "") or ""
    neg_prompt = params.get("negative_prompt", "") or ""

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos   = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg   = clip.encode_from_tokens_scheduled(tokens_neg)

    # ── Step 4: Latent ────────────────────────────────────────────────────────
    width  = int(params.get("width")  or _val("latent_width")  or _val("resize_width")  or 768)
    height = int(params.get("height") or _val("latent_height") or _val("resize_height") or 1280)
    length = params.get("length")
    if length is None:
        length = _val("latent_length")

    if strategy == "wan_video" and family_key == "wan_video_i2v":
        # i2v: load source image, run WanImageToVideo in-process
        i2v_image_name = params.get("source_image_path") or _val("wan_i2v_image")
        if not i2v_image_name:
            raise ValueError(
                "[WorkflowExecutor] wan_video_i2v has no source image "
                "— connect an image to the source_image input"
            )
        latent_dict = _run_i2v_from_template(
            i2v_image_name, vae, cond_pos, cond_neg,
            width, height, int(length or 81),
        )
        cond_pos, cond_neg = latent_dict.pop("_cond_pos"), latent_dict.pop("_cond_neg")
    else:
        batch = int(params.get("batch_size", 1))
        if length is not None:
            # Video latent — channels=16 for WAN/Hunyuan
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
    # Use params directly — no re-reading from patched JSON.
    from ..nodes.workflow_generator import (
        _run_standard_ksampler, _run_flux_sampler, _run_flux2_sampler, _run_wan_sampler,
    )

    seed = int(params.get("seed", 0))
    sampler_params = {
        "seed":         seed,
        "steps":        int(params.get("steps", 20)),
        "cfg":          float(params.get("cfg", 5.0)),
        "sampler_name": params.get("sampler_name", "euler"),
        "scheduler":    params.get("scheduler", "simple"),
        "denoise":      float(params.get("denoise", 1.0)),
        "guidance":     float(params.get("guidance") or 3.5),
    }

    if strategy == "wan_video":
        # steps_high / steps_low come directly from params — no reverse-
        # engineering of KSamplerAdvanced start/end boundaries needed.
        steps_high = int(params.get("steps_high", sampler_params["steps"]))
        steps_low  = int(params.get("steps_low",  steps_high))
        seed_b     = int(params.get("seed_b", seed + 1))
        print(f"[WorkflowExecutor] WAN dual-sampler: high={steps_high}, low={steps_low}")
        sampler_params["steps"] = steps_high
        sampler_params_b = {
            "steps":        steps_low,
            "cfg":          sampler_params["cfg"],
            "sampler_name": sampler_params["sampler_name"],
            "scheduler":    sampler_params["scheduler"],
            "seed":         seed_b,
        }

        # Conditioning for model B:
        # - i2v: cond_pos/cond_neg already contain WanImageToVideo output
        #   (source image encoded into conditioning). BOTH models must use
        #   these — the original workflow connects WanImageToVideo output
        #   to both KSamplerAdvanced nodes.
        # - t2v: re-encode with clip_b since model B has its own text encoder.
        if family_key == "wan_video_i2v":
            cond_pos_b = cond_pos
            cond_neg_b = cond_neg
        else:
            tokens_pos_b = clip_b.tokenize(pos_prompt)
            cond_pos_b   = clip_b.encode_from_tokens_scheduled(tokens_pos_b)
            tokens_neg_b = clip_b.tokenize(neg_prompt)
            cond_neg_b   = clip_b.encode_from_tokens_scheduled(tokens_neg_b)

        samples = _run_wan_sampler(
            model_a, cond_pos, cond_neg, latent_dict, sampler_params,
            model_b=model_b, cond_pos_b=cond_pos_b, cond_neg_b=cond_neg_b,
            sampler_params_b=sampler_params_b,
        )
    elif strategy == "flux2":
        samples = _run_flux2_sampler(model_a, cond_pos, cond_neg, latent_dict, sampler_params)
    elif strategy == "flux":
        samples = _run_flux_sampler(model_a, cond_pos, cond_neg, latent_dict, sampler_params)
    else:
        # wan_image, sdxl, qwen, z_image, etc. — all use standard KSampler
        samples = _run_standard_ksampler(model_a, cond_pos, cond_neg, latent_dict, sampler_params)

    # ── Step 6: Decode ────────────────────────────────────────────────────────
    decoded = vae.decode(samples)
    if len(decoded.shape) == 5:          # video: (batch, frames, H, W, 3) → (B*F, H, W, 3)
        decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
    out_latent = {"samples": samples}

    return decoded, out_latent


def _family_to_strategy(family_key):
    from .workflow_families import MODEL_FAMILIES
    spec = MODEL_FAMILIES.get(family_key, {})
    return spec.get("sampler", "standard")


def _run_i2v_from_template(image_name, vae, cond_pos, cond_neg,
                           width, height, length):
    """
    Run the WAN i2v pipeline in-process:
      Load source image from disk → WanImageToVideo

    Returns a latent dict with _cond_pos and _cond_neg injected
    (consumed by the downstream dual-sampler).
    """
    import torch
    import numpy as np
    import comfy.model_management
    from PIL import Image as PILImage

    # ── Load source image ─────────────────────────────────────────────────
    input_dir = folder_paths.get_input_directory()
    image_path = folder_paths.get_annotated_filepath(image_name)
    if not os.path.isfile(image_path):
        image_path = os.path.join(input_dir, image_name)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"[WorkflowExecutor] i2v source image not found: {image_name} "
            f"(looked in {input_dir})"
        )
    pil_img = PILImage.open(image_path).convert("RGB")
    img_array = np.array(pil_img).astype(np.float32) / 255.0
    source_image = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W, 3)
    print(f"[WorkflowExecutor] Loaded i2v source image: "
          f"{image_name} → shape {source_image.shape}")

    # ── WanImageToVideo ───────────────────────────────────────────────────
    try:
        from comfy_extras.nodes_wan import WanImageToVideo
        result = WanImageToVideo.execute(
            positive=cond_pos,
            negative=cond_neg,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=1,
            start_image=source_image,
        )
        new_cond_pos, new_cond_neg, latent = result[0], result[1], result[2]
        latent["_cond_pos"] = new_cond_pos
        latent["_cond_neg"] = new_cond_neg
        return latent
    except Exception as e:
        print(f"[WorkflowExecutor] WanImageToVideo failed ({e}), falling back to empty latent")
        traceback.print_exc()
        latent_tensor = torch.zeros(
            [length, 16, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return {
            "samples": latent_tensor,
            "_cond_pos": cond_pos,
            "_cond_neg": cond_neg,
        }
