"""
Model family definitions and compatibility helpers for ComfyUI Prompt Manager.

Shared by workflow_extractor, workflow_generator, workflow_manager, and workflow_extractor_utils.

Each family dict specifies:
  "label"   — human-readable name shown in UI dropdowns
  "folders" — lowercase path prefixes (matched against relative model path)
  "names"   — lowercase substrings matched against filename only
  "vae"     — VAE filename substrings compatible with this family
  "clip"    — CLIP/text-encoder filename substrings compatible with this family
  "sampler" — internal sampling strategy key (see SAMPLER_STRATEGIES below)

Matching: folder prefixes first, then name patterns. First match wins.
To add a new family: add a dict entry and handle its "sampler" key in workflow_generator.
"""
import os
import folder_paths

# ─── Family definitions ──────────────────────────────────────────────────────

MODEL_FAMILIES = {
    # ── Flux 2 / Klein ───────────────────────────────────────────────────────
    "flux2": {
        "label":   "Flux 2 (Klein)",
        "folders": ["klein/"],
        "names":   ["flux2", "flux-2", "flux_2", "klein"],
        "vae":     ["flux2-vae", "flux2_vae"],
        "clip":    ["mistral_3_small_flux2"],
        "sampler": "flux2",
    },
    # ── Flux 1 variants ──────────────────────────────────────────────────────
    "flux_dev": {
        "label":   "Flux 1 Dev",
        "folders": [],
        "names":   ["flux-dev", "flux_dev", "fluxdev", "flux1-dev", "flux1_dev"],
        "vae":     ["ae.safetensors", "ultrafluxvae"],
        "clip":    ["t5xxl", "clip_l"],
        "sampler": "flux",
    },
    "flux_schnell": {
        "label":   "Flux 1 Schnell",
        "folders": [],
        "names":   ["flux-schnell", "flux_schnell", "fluxschnell", "flux1-schnell"],
        "vae":     ["ae.safetensors", "ultrafluxvae"],
        "clip":    ["t5xxl", "clip_l"],
        "sampler": "flux",
    },
    "flux": {
        "label":   "Flux 1",
        "folders": ["flux/"],
        "names":   ["flux"],
        "vae":     ["ae.safetensors", "ultrafluxvae"],
        "clip":    ["t5xxl", "clip_l"],
        "sampler": "flux",
    },
    # ── Z-Image (Flux-architecture, own VAE/CLIP) ────────────────────────────
    "zib": {
        "label":   "Z-Image Base",
        "folders": ["zib/"],
        "names":   ["z_image_bf16", "z-image-base"],
        "vae":     ["zimageturbo_vae", "ae.safetensors"],
        "clip":    ["qwen-4b-zimage", "qwen_3_4b", "qwen_3_8b"],
        "sampler": "flux",
    },
    "zit": {
        "label":   "Z-Image Turbo",
        "folders": ["zit/"],
        "names":   ["z_image_turbo", "z-image-turbo", "blitz"],
        "vae":     ["zimageturbo_vae", "ae.safetensors"],
        "clip":    ["qwen-4b-zimage", "qwen_3_4b", "qwen_3_8b"],
        "sampler": "flux",
    },
    # ── SD3 ──────────────────────────────────────────────────────────────────
    "sd3.5": {
        "label":   "SD 3.5",
        "folders": [],
        "names":   ["sd3.5", "sd35"],
        "vae":     [],
        "clip":    ["t5xxl", "clip_l", "clip_g"],
        "sampler": "standard",
    },
    "sd3": {
        "label":   "SD 3",
        "folders": [],
        "names":   ["sd3"],
        "vae":     [],
        "clip":    ["t5xxl", "clip_l", "clip_g"],
        "sampler": "standard",
    },
    # ── SDXL family ──────────────────────────────────────────────────────────
    "sdxl_turbo": {
        "label":   "SDXL Turbo",
        "folders": [],
        "names":   ["sdxl-turbo", "sdxl_turbo", "sdxlturbo"],
        "vae":     ["sdxl_vae", "sdxl-vae"],
        "clip":    ["clip_l", "clip_g"],
        "sampler": "standard",
    },
    "sdxl": {
        "label":   "SDXL",
        "folders": [],
        "names":   ["sdxl"],
        "vae":     ["sdxl_vae", "sdxl-vae"],
        "clip":    ["clip_l", "clip_g"],
        "sampler": "standard",
    },
    "illustrious": {
        "label":   "Illustrious (SDXL)",
        "folders": ["illustrious/"],
        "names":   ["illustrious", "noob"],
        "vae":     ["sdxl_vae", "sdxl-vae"],
        "clip":    ["clip_l", "clip_g"],
        "sampler": "standard",
    },
    "pony": {
        "label":   "Pony (SDXL)",
        "folders": [],
        "names":   ["pony"],
        "vae":     ["sdxl_vae", "sdxl-vae"],
        "clip":    ["clip_l", "clip_g"],
        "sampler": "standard",
    },
    # ── SD 1.x / 2.x ────────────────────────────────────────────────────────
    "sd2": {
        "label":   "SD 2.x",
        "folders": [],
        "names":   ["sd-2", "sd_2", "sd2", "v2-", "v2_", "768-v"],
        "vae":     ["vae-ft-mse-840000"],
        "clip":    ["clip_l"],
        "sampler": "standard",
    },
    "sd1.5": {
        "label":   "SD 1.5",
        "folders": [],
        "names":   ["sd-1", "sd_1", "sd1", "v1-5", "v1_5", "v1.5"],
        "vae":     ["vae-ft-mse-840000"],
        "clip":    ["clip_l"],
        "sampler": "standard",
    },
    # ── Video models ─────────────────────────────────────────────────────────
    "wan": {
        "label":   "WAN 2.x",
        "folders": ["wan2_2/", "wan/"],
        "names":   ["wan2", "wan_", "wanvideo"],
        "vae":     ["wan_2.1_vae", "wan2.1_vae", "wan2.2_vae"],
        "clip":    ["umt5_xxl"],
        "sampler": "wan",
    },
    "ltxv": {
        "label":   "LTX-Video",
        "folders": ["ltx/"],
        "names":   ["ltxv", "ltx-video", "ltx-2"],
        "vae":     ["ltx23_video_vae", "taeltx"],
        "clip":    ["t5xxl", "ltx-2", "gemma_3"],
        "sampler": "standard",
    },
    "cogvideo": {
        "label":   "CogVideo",
        "folders": ["cogvideo/"],
        "names":   ["cogvideo"],
        "vae":     [],
        "clip":    ["t5xxl"],
        "sampler": "standard",
    },
    "mochi": {
        "label":   "Mochi",
        "folders": [],
        "names":   ["mochi"],
        "vae":     [],
        "clip":    ["t5xxl"],
        "sampler": "standard",
    },
    "hunyuan": {
        "label":   "HunyuanVideo",
        "folders": ["hunyuan/"],
        "names":   ["hunyuan"],
        "vae":     [],
        "clip":    ["llama_3", "clip_l"],
        "sampler": "standard",
    },
    # ── Qwen Image ───────────────────────────────────────────────────────────
    "qwen_image": {
        "label":   "Qwen Image",
        "folders": ["qwen/"],
        "names":   ["qwen_image"],
        "vae":     ["qwen_image_vae"],
        "clip":    ["qwen_2.5_vl"],
        "sampler": "standard",
    },
    # ── Other ────────────────────────────────────────────────────────────────
    "cascade": {
        "label":   "Stable Cascade",
        "folders": [],
        "names":   ["cascade", "stable-cascade"],
        "vae":     [],
        "clip":    ["clip_g"],
        "sampler": "standard",
    },
    "pixart": {
        "label":   "PixArt",
        "folders": [],
        "names":   ["pixart"],
        "vae":     [],
        "clip":    ["t5xxl"],
        "sampler": "standard",
    },
    "kolors": {
        "label":   "Kolors",
        "folders": [],
        "names":   ["kolors"],
        "vae":     ["sdxl_vae"],
        "clip":    [],
        "sampler": "standard",
    },
    "auraflow": {
        "label":   "AuraFlow",
        "folders": [],
        "names":   ["auraflow"],
        "vae":     [],
        "clip":    [],
        "sampler": "standard",
    },
}

# Compatibility groups — checkpoint families that can share each other's models.
# VAE/CLIP are NOT grouped (they differ per family) — model weights only.
MODEL_COMPAT_GROUPS = [
    {"sdxl", "sdxl_turbo", "pony", "illustrious"},  # SDXL-arch fine-tunes
    {"flux_dev", "flux_schnell", "flux"},             # Flux1 variants
    {"zib", "zit"},                                   # Z-Image variants
]

# Sampling strategy keys (referenced by workflow_generator.py)
# Each key maps to a specific sampling code path.
SAMPLER_STRATEGIES = {
    "standard": "standard",  # KSampler — SD1.5, SDXL, SD3, LTX, etc.
    "flux":     "flux",      # SamplerCustomAdvanced + BasicGuider (Flux1, Z-Image)
    "flux2":    "flux2",     # Flux2Scheduler + SamplerCustomAdvanced (Klein/Flux2)
    "wan":      "wan",       # Dual KSampler high/low (WAN 2.x)
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


def list_compatible_vaes(family):
    """Return sorted list of VAE files compatible with the given family."""
    if not family:
        try:
            return sorted(folder_paths.get_filename_list("vae"))
        except Exception:
            return []

    spec = MODEL_FAMILIES.get(family, {})
    patterns = [p.lower() for p in spec.get("vae", [])]

    if not patterns:
        try:
            return sorted(folder_paths.get_filename_list("vae"))
        except Exception:
            return []

    try:
        all_vaes = folder_paths.get_filename_list("vae")
    except Exception:
        return []

    matched = []
    for v in all_vaes:
        v_lower = v.lower().replace("\\", "/")
        v_name = os.path.basename(v_lower)
        for pat in patterns:
            if pat in v_name or pat in v_lower:
                matched.append(v)
                break

    return sorted(matched) if matched else sorted(all_vaes)


def list_compatible_clips(family):
    """Return sorted list of CLIP/text-encoder files compatible with the given family."""
    if not family:
        clips = _gather_all_clips()
        return sorted(clips)

    spec = MODEL_FAMILIES.get(family, {})
    patterns = [p.lower() for p in spec.get("clip", [])]

    if not patterns:
        clips = _gather_all_clips()
        return sorted(clips)

    all_clips = _gather_all_clips()

    matched = []
    for c in all_clips:
        c_lower = c.lower().replace("\\", "/")
        c_name = os.path.basename(c_lower)
        for pat in patterns:
            if pat in c_name or pat in c_lower:
                matched.append(c)
                break

    return sorted(matched) if matched else sorted(all_clips)


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
