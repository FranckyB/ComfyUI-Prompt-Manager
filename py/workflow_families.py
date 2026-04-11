"""
Model family definitions and compatibility helpers for ComfyUI Prompt Manager.

Shared by workflow_generator, workflow_generator, workflow_manager, and workflow_extraction_utils.

Each family dict specifies:
  "label"        — human-readable name shown in UI dropdowns
  "folders"      — lowercase path prefixes (matched against relative model path)
  "names"        — lowercase substrings matched against filename only
  "vae"          — VAE filename substrings compatible with this family
  "vae_exact"    — VAE exact basenames (full filename match, e.g. "ae.safetensors")
  "clip"         — CLIP/text-encoder filename substrings compatible with this family
  "clip_exact"   — CLIP exact basenames (full filename match)
  "clip_exclude" — substrings that DISQUALIFY a CLIP match (applied after include patterns)
  "sampler"      — internal sampling strategy key (see SAMPLER_STRATEGIES below)

Matching rules:
  VAE:  vae_exact checked first (full basename), then vae substring patterns.
  CLIP: clip_exclude applied first — any match disqualifies the file.
        Then clip_exact checked (full basename), then clip substring patterns.
  Folder prefixes checked before name patterns. First match wins for family detection.
To add a new family: add a dict entry and handle its "sampler" key in workflow_generator.
"""
import os
import folder_paths

# ─── Family definitions ──────────────────────────────────────────────────────

MODEL_FAMILIES = {
    # ── Flux 2 / Klein ───────────────────────────────────────────────────────
    "flux2": {
        "label":   "Flux 2",
        "folders": ["klein/", "flux2/"],
        "names":   ["flux2", "flux-2", "flux_2", "klein", "flux.2"],
        # Flux2/Klein uses its own VAE — NOT ae.safetensors
        "vae":     ["flux2-vae", "flux2_vae"],
        # Klein uses Qwen text encoders (NOT T5/CLIP-L):
        #   4B model → qwen_3_4b.safetensors
        #   9B model → qwen_3_8b_fp8mixed.safetensors
        # CLIPLoader type must be "flux2" so CLIPType.FLUX2 is used.
        "clip":         ["qwen_3_4b", "qwen_3_8b", "qwen_3"],
        "clip_exclude": ["qwen_2.5", "qwen_2_5", "t5xxl", "umt5", "clip_l", "clip_g", "gemma"],
        "clip_type": "flux2",
        "sampler": "flux2",
    },
    # ── Flux 1 (Dev + Schnell merged) ────────────────────────────────────────
    "flux1": {
        "label":   "Flux 1",
        "folders": ["flux/", "flux1/"],
        "names":   ["flux-dev", "flux_dev", "fluxdev", "flux1-dev", "flux1_dev",
                    "flux-schnell", "flux_schnell", "fluxschnell", "flux1-schnell",
                    "flux1_schnell", "flux1-", "flux1_", "flux.1"],
        # ae.safetensors is the canonical Flux1 VAE — exact match required to
        # avoid catching "mae.safetensors" or other short-named files.
        "vae":       ["ultrafluxvae"],
        "vae_exact": ["ae.safetensors"],
        # Flux1 uses T5-XXL + CLIP-L dual encoders.
        # umt5_xxl also contains "t5xxl" as substring — exclude it explicitly.
        "clip":         ["t5xxl", "clip_l"],
        "clip_exclude": ["umt5", "clip_g", "qwen", "gemma"],
        "sampler": "flux",
    },
    # ── Z-Image (Base + Turbo merged) ────────────────────────────────────────
    "zimage": {
        "label":   "Z-Image",
        "folders": ["zib/", "zit/", "zimage/"],
        "names":   ["z_image", "z-image", "zimage", "blitz"],
        # Z-Image shares ae.safetensors with Flux1 for its base VAE.
        "vae":       ["zimageturbo_vae"],
        "vae_exact": ["ae.safetensors"],
        # Z-Image uses Qwen 3 4B as its text encoder (not the larger 8B,
        # not T5, not CLIP-L). The 4B file is typically named
        # "qwen-4b-zimage.safetensors" or "qwen_3_4b.safetensors".
        "clip":         ["qwen-4b-zimage", "qwen_3_4b"],
        "clip_exclude": ["qwen_3_8b", "qwen_2.5", "qwen_2_5", "t5xxl", "umt5",
                         "clip_l", "clip_g", "gemma"],
        "sampler": "flux",
    },
    # ── WAN 2.x Video — Image-to-Video (dual KSampler + i2v latent) ──────────
    "wan_video_i2v": {
        "label":   "WAN Video (i2v)",
        "folders": ["wan2_2/i2v/", "wan2_1/i2v/"],
        "names":   ["wan2.2_i2v", "wan2.1_i2v", "wan_i2v", "i2v"],
        "vae":     ["wan_2.1_vae", "wan2.1_vae", "wan2.2_vae", "wan_2.2_vae",
                    "wan2.1-vae", "wan_2.1-vae"],
        "clip":         ["umt5_xxl", "umt5-xxl"],
        "clip_exclude": ["t5xxl", "clip_l", "clip_g", "qwen", "gemma"],
        "sampler": "wan_video",
    },
    # ── WAN 2.x Video — Text-to-Video (dual KSampler) ────────────────────────
    # DETECTION PRIORITY: must appear before wan_image — they share the same
    # folders/names, so first-match wins.  Auto-detected T2V models land here;
    # user selects wan_image explicitly from the family dropdown.
    "wan_video_t2v": {
        "label":   "WAN Video (t2v)",
        "folders": ["wan2_2/", "wan2_1/", "wan/"],
        "names":   ["wan2.2", "wan2.1", "wan2_2", "wan2_1", "wanvideo", "wan_t2v",
                    "wan2.2_t2v", "wan2.1_t2v"],
        "vae":     ["wan_2.1_vae", "wan2.1_vae", "wan2.2_vae", "wan_2.2_vae",
                    "wan2.1-vae", "wan_2.1-vae"],
        "clip":         ["umt5_xxl", "umt5-xxl"],
        "clip_exclude": ["t5xxl", "clip_l", "clip_g", "qwen", "gemma"],
        "sampler": "wan_video",
    },
    # ── WAN 2.x Image (single KSampler, text-to-image) ──────────────────────
    # Same model files as WAN Video T2V — identical folders/names/vae/clip.
    # Must appear AFTER wan_video_t2v so auto-detection resolves T2V models
    # to wan_video_t2v.  User picks this family explicitly for still images.
    # Sampler stays wan_image (single KSampler, no dual Steps rows).
    "wan_image": {
        "label":   "WAN Image",
        "folders": ["wan2_2/", "wan2_1/", "wan/"],
        "names":   ["wan2.2", "wan2.1", "wan2_2", "wan2_1", "wanvideo", "wan_t2v",
                    "wan2.2_t2v", "wan2.1_t2v"],
        "vae":     ["wan_2.1_vae", "wan2.1_vae", "wan2.2_vae", "wan_2.2_vae",
                    "wan2.1-vae", "wan_2.1-vae"],
        # WAN uses UMT5-XXL exclusively — NOT t5xxl, NOT clip_l/g.
        "clip":         ["umt5_xxl", "umt5-xxl"],
        "clip_exclude": ["t5xxl", "clip_l", "clip_g", "qwen", "gemma"],
        "sampler": "wan_image",
    },
    # ── LTX-Video ────────────────────────────────────────────────────────────
    "ltxv": {
        "label":   "LTX-Video",
        "folders": ["ltx/"],
        "names":   ["ltxv", "ltx-video", "ltx-2", "ltx2"],
        # LTX has its own VAE — NOT ae.safetensors, NOT SDXL/SD VAE.
        # taeltx* are TAE-based approximation VAEs for LTX.
        "vae":     ["ltx23_video_vae", "ltxv_vae", "ltx_vae", "taeltx"],
        # LTX-Video uses T5-XXL only (same encoder family as Flux1 but
        # NOT the dual CLIP-L+T5 combo — single T5 only).
        # umt5_xxl contains "t5xxl" → must exclude it.
        "clip":         ["t5xxl"],
        "clip_exclude": ["umt5", "clip_l", "clip_g", "qwen", "gemma"],
        "sampler": "flux",
    },
    # ── SDXL (+ Pony, Illustrious, Turbo merged) ────────────────────────────
    "sdxl": {
        "label":   "SDXL",
        "folders": ["illustrious/", "pony/", "sdxl/"],
        "names":   ["sdxl", "illustrious", "noob", "pony"],
        # SDXL uses its own finely-tuned VAE (sdxl_vae / sdxl-vae) or the
        # widely-used vae-ft-mse-840000 (originally SD1.5 but SDXL-compatible).
        "vae":     ["sdxl_vae", "sdxl-vae", "vae-ft-mse"],
        # SDXL uses dual text encoders: CLIP-L + CLIP-G.
        "clip":         ["clip_l", "clip_g"],
        "clip_exclude": ["t5xxl", "umt5", "qwen", "gemma"],
        "sampler": "standard",
        "checkpoint": True,
    },
    # ── SD 1.5 ───────────────────────────────────────────────────────────────
    "sd15": {
        "label":   "SD 1.5",
        "folders": [],
        "names":   ["sd-1", "sd_1", "sd1", "v1-5", "v1_5", "v1.5", "dreamshaper", "realistic"],
        # SD1.5 canonical VAE: vae-ft-mse-840000 or vae-ft-ema-560000.
        "vae":     ["vae-ft-mse-840000", "vae-ft-ema-560000", "orangemix", "blessed2"],
        # SD1.5 uses a single CLIP-L encoder only (NOT CLIP-G like SDXL).
        "clip":         ["clip_l"],
        "clip_exclude": ["clip_g", "t5xxl", "umt5", "qwen", "gemma"],
        "sampler": "standard",
        "checkpoint": True,
    },
    # ── Qwen Image ───────────────────────────────────────────────────────────
    "qwen_image": {
        "label":   "Qwen Image",
        "folders": ["qwen/"],
        "names":   ["qwen_image"],
        "vae":     ["qwen_image_vae"],
        "clip":         ["qwen_2.5_vl", "qwen_2_5_vl"],
        "clip_exclude": ["qwen_3", "t5xxl", "umt5", "clip_l", "clip_g", "gemma"],
        "sampler": "standard",
    },
}

# Compatibility groups — checkpoint families that can share each other's models.
# VAE/CLIP are NOT grouped (they differ per family) — model weights only.
MODEL_COMPAT_GROUPS = [
    {"sdxl"},                          # SDXL-arch — all merged into one now
    {"flux1"},                         # Flux1 variants — all merged into one now
    {"zimage"},                        # Z-Image — all merged
    {"flux2"},                         # Flux2 — all merged
]

# Sampling strategy keys (referenced by workflow_generator.py)
# Each key maps to a specific sampling code path.
SAMPLER_STRATEGIES = {
    "standard":  "standard",   # KSampler — SD1.5, SDXL, Qwen
    "flux":      "flux",       # SamplerCustomAdvanced + BasicGuider (Flux1, Z-Image, LTX-Video)
    "flux2":     "flux2",      # Flux2Scheduler + SamplerCustomAdvanced + CFGGuider (Klein/Flux2)
    "wan_image": "wan_image",  # Single KSampler — WAN Image
    "wan_video": "wan_video",  # Dual KSamplerAdvanced high/low (WAN Video i2v + t2v)
}

# Workflow template filenames (in workflows/api/) keyed by family.
# Maps each family key to its _api.json and _map.json stem.
FAMILY_WORKFLOW_STEMS = {
    "sdxl":          "sdxl",
    "sd15":          None,          # No template yet — falls back to hardcoded
    "flux1":         "flux_1",
    "flux2":         "flux_2",
    "zimage":        "z_image",
    "wan_image":     "wan_image",
    "wan_video_i2v": "wan_video_i2v",
    "wan_video_t2v": "wan_video_t2v",
    "ltxv":          None,          # No template yet
    "qwen_image":    "qwen_image",
}


# ─── Family lookup ───────────────────────────────────────────────────────────

def get_model_family(model_path):
    """
    Return the family key for a model, or None if unknown.
    Accepts a relative path like 'Illustrious/Anime/illustrij_v20.safetensors'.
    """
    if not model_path:
        return None
    path_lower = model_path.lower().replace("\\", "/")
    name_lower = os.path.basename(path_lower)

    for family, spec in MODEL_FAMILIES.items():
        for prefix in spec.get("folders", []):
            if path_lower.startswith(prefix.lower().replace("\\", "/")):
                return family
        for pat in spec.get("names", []):
            if pat in name_lower:
                return family
    return None


def get_family_label(family):
    """Return the human-readable label for a family, or the key itself."""
    if not family:
        return "Unknown"
    return MODEL_FAMILIES.get(family, {}).get("label", family)


def get_family_sampler_strategy(family):
    """Return the sampler strategy key for a family (default: 'standard')."""
    if not family:
        return "standard"
    return MODEL_FAMILIES.get(family, {}).get("sampler", "standard")


def get_compatible_families(family):
    """Return the set of all checkpoint-families compatible with the given one."""
    if not family:
        return set()
    result = {family}
    for group in MODEL_COMPAT_GROUPS:
        if family in group:
            result |= group
    return result


def get_all_family_labels():
    """Return a dict of {family_key: label} for all known families."""
    return {k: v.get("label", k) for k, v in MODEL_FAMILIES.items()}


# ─── Model listing ───────────────────────────────────────────────────────────

def list_compatible_models(reference_model):
    """
    Given a reference model name/path, return a sorted list of compatible
    models found on disk.  If the family is unknown, returns ALL models.
    """
    family = get_model_family(reference_model)
    compat = get_compatible_families(family)

    all_models = []
    for folder_name in ['checkpoints', 'diffusion_models', 'unet', 'unet_gguf']:
        try:
            all_models.extend(folder_paths.get_filename_list(folder_name))
        except Exception:
            continue

    # Deduplicate preserving order
    seen = set()
    unique = []
    for m in all_models:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    if not compat:
        return sorted(unique)

    compatible = []
    for m in unique:
        m_family = get_model_family(m)
        if m_family and m_family in compat:
            compatible.append(m)

    return sorted(compatible)


def list_all_models():
    """Return all models from all known model folder types (deduplicated, sorted)."""
    all_models = []
    seen = set()
    for folder_name in ['checkpoints', 'diffusion_models', 'unet', 'unet_gguf']:
        try:
            for m in folder_paths.get_filename_list(folder_name):
                if m not in seen:
                    seen.add(m)
                    all_models.append(m)
        except Exception:
            continue
    return sorted(all_models)


def list_compatible_vaes(family, return_recommended=False):
    """Return sorted list of VAE files compatible with the given family.
    If return_recommended=True, returns (list, recommended_str_or_None).

    Matching order:
      1. vae_exact  — full basename match (e.g. "ae.safetensors"), case-insensitive
      2. vae        — substring match against basename or full relative path
    Falls back to all VAEs when no patterns are defined or nothing matched.
    """
    def _ret(lst, rec):
        return (sorted(lst), rec) if return_recommended else sorted(lst)

    if not family:
        try:
            return _ret(folder_paths.get_filename_list("vae"), None)
        except Exception:
            return _ret([], None)

    spec = MODEL_FAMILIES.get(family, {})
    exact_names = [e.lower() for e in spec.get("vae_exact", [])]
    patterns    = [p.lower() for p in spec.get("vae", [])]

    if not exact_names and not patterns:
        try:
            return _ret(folder_paths.get_filename_list("vae"), None)
        except Exception:
            return _ret([], None)

    try:
        all_vaes = folder_paths.get_filename_list("vae")
    except Exception:
        return _ret([], None)

    matched = []
    recommended = None
    is_checkpoint = spec.get("checkpoint", False)

    for v in all_vaes:
        v_lower = v.lower().replace("\\", "/")
        v_name  = os.path.basename(v_lower)

        # 1. Exact basename match (e.g. "ae.safetensors")
        if v_name in exact_names:
            matched.append(v)
            if recommended is None and not is_checkpoint:
                recommended = v
            continue

        # 2. Substring patterns
        for pat in patterns:
            if pat in v_name or pat in v_lower:
                matched.append(v)
                if recommended is None and not is_checkpoint:
                    recommended = v
                break

    if matched:
        return _ret(matched, recommended)
    return _ret(all_vaes, None)


def list_compatible_clips(family, return_recommended=False):
    """Return sorted list of CLIP/text-encoder files compatible with the given family.
    If return_recommended=True, returns (list, recommended_str_or_None).

    Matching order:
      1. clip_exclude — if ANY exclude pattern is found in the basename/path, skip the file
      2. clip_exact   — full basename match, case-insensitive
      3. clip         — substring match against basename or full relative path
    Falls back to all CLIPs when no patterns are defined or nothing matched.
    """
    def _ret(lst, rec):
        return (sorted(lst), rec) if return_recommended else sorted(lst)

    if not family:
        clips = _gather_all_clips()
        return _ret(clips, None)

    spec = MODEL_FAMILIES.get(family, {})
    exclude_pats = [e.lower() for e in spec.get("clip_exclude", [])]
    exact_names  = [e.lower() for e in spec.get("clip_exact", [])]
    patterns     = [p.lower() for p in spec.get("clip", [])]

    if not exact_names and not patterns:
        clips = _gather_all_clips()
        return _ret(clips, None)

    all_clips = _gather_all_clips()

    matched = []
    recommended = None
    is_checkpoint = spec.get("checkpoint", False)

    for c in all_clips:
        c_lower = c.lower().replace("\\", "/")
        c_name  = os.path.basename(c_lower)

        # 1. Exclusion check — disqualify if any exclude pattern is present
        if any(ex in c_name or ex in c_lower for ex in exclude_pats):
            continue

        # 2. Exact basename match
        if c_name in exact_names:
            matched.append(c)
            if recommended is None and not is_checkpoint:
                recommended = c
            continue

        # 3. Substring include patterns
        for pat in patterns:
            if pat in c_name or pat in c_lower:
                matched.append(c)
                if recommended is None and not is_checkpoint:
                    recommended = c
                break

    if matched:
        return _ret(matched, recommended)
    return _ret(all_clips, None)


def _gather_all_clips():
    """Gather CLIP files from text_encoders and clip folders."""
    clips = []
    seen = set()
    for folder in ['text_encoders', 'clip']:
        try:
            for c in folder_paths.get_filename_list(folder):
                if c not in seen:
                    seen.add(c)
                    clips.append(c)
        except Exception:
            pass
    return clips
