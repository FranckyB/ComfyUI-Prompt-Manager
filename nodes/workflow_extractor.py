"""
ComfyUI Workflow Extractor - Extract ALL generation parameters from an image/video
and regenerate it internally. Loads model, applies LoRAs, runs KSampler, outputs IMAGE + LATENT.

Part of ComfyUI-Prompt-Manager — shares extraction logic with PromptExtractor.
"""
import os
import json
import traceback
import folder_paths
import server

# Shared extraction functions from prompt_extractor (same package)
from .prompt_extractor import (
    parse_workflow_for_prompts,
    extract_metadata_from_png,
    extract_metadata_from_jpeg,
    extract_metadata_from_json,
    extract_metadata_from_video,
    resolve_lora_path,
    build_node_map,
    build_link_map,
)

# ─── Constants ───────────────────────────────────────────────────────────────

MODEL_EXTENSIONS = ['.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf', '.sft']

# ─── Model family system ────────────────────────────────────────────────────
# Each family specifies:
#   "folders" — lowercase path prefixes (most reliable, matched against relative path)
#   "names"   — lowercase substrings matched against filename only
#   "vae"     — list of VAE filename substrings compatible with this family
#   "clip"    — list of CLIP/text-encoder filename substrings compatible
#   "label"   — human-readable name shown in UI
#
# Matching: folder prefixes checked first, then name patterns. First match wins.
# To add a new family: add a dict entry + optionally add to MODEL_COMPAT_GROUPS.

MODEL_FAMILIES = {
    # ── Flux 2 / Klein ───────────────────────────────────────────────────────
    "flux2": {
        "label":   "Flux 2 (Klein)",
        "folders": ["klein/"],
        "names":   ["flux2", "flux-2", "flux_2", "klein"],
        "vae":     ["flux2-vae", "flux2_vae"],
        "clip":    ["mistral_3_small_flux2"],
    },
    # ── Flux 1 variants ──────────────────────────────────────────────────────
    "flux_dev": {
        "label":   "Flux 1 Dev",
        "folders": [],
        "names":   ["flux-dev", "flux_dev", "fluxdev", "flux1-dev", "flux1_dev"],
        "vae":     ["ae.safetensors", "ultrafluxvae"],
        "clip":    ["t5xxl", "clip_l"],
    },
    "flux_schnell": {
        "label":   "Flux 1 Schnell",
        "folders": [],
        "names":   ["flux-schnell", "flux_schnell", "fluxschnell", "flux1-schnell"],
        "vae":     ["ae.safetensors", "ultrafluxvae"],
        "clip":    ["t5xxl", "clip_l"],
    },
    "flux": {
        "label":   "Flux 1",
        "folders": ["flux/"],
        "names":   ["flux"],
        "vae":     ["ae.safetensors", "ultrafluxvae"],
        "clip":    ["t5xxl", "clip_l"],
    },
    # ── Z-Image (Flux-architecture, own VAE/CLIP) ────────────────────────────
    "zib": {
        "label":   "Z-Image Base",
        "folders": ["zib/"],
        "names":   ["z_image_bf16", "z-image-base"],
        "vae":     ["zimageturbo_vae", "ae.safetensors"],
        "clip":    ["qwen-4b-zimage", "qwen_3_4b", "qwen_3_8b"],
    },
    "zit": {
        "label":   "Z-Image Turbo",
        "folders": ["zit/"],
        "names":   ["z_image_turbo", "z-image-turbo", "blitz"],
        "vae":     ["zimageturbo_vae", "ae.safetensors"],
        "clip":    ["qwen-4b-zimage", "qwen_3_4b", "qwen_3_8b"],
    },
    # ── SD3 ──────────────────────────────────────────────────────────────────
    "sd3.5": {
        "label":   "SD 3.5",
        "folders": [],
        "names":   ["sd3.5", "sd35"],
        "vae":     [],  # typically bundled in checkpoint
        "clip":    ["t5xxl", "clip_l", "clip_g"],
    },
    "sd3": {
        "label":   "SD 3",
        "folders": [],
        "names":   ["sd3"],
        "vae":     [],
        "clip":    ["t5xxl", "clip_l", "clip_g"],
    },
    # ── SDXL family ──────────────────────────────────────────────────────────
    "sdxl_turbo": {
        "label":   "SDXL Turbo",
        "folders": [],
        "names":   ["sdxl-turbo", "sdxl_turbo", "sdxlturbo"],
        "vae":     ["sdxl_vae", "sdxl-vae"],
        "clip":    ["clip_l", "clip_g"],
    },
    "sdxl": {
        "label":   "SDXL",
        "folders": [],
        "names":   ["sdxl"],
        "vae":     ["sdxl_vae", "sdxl-vae"],
        "clip":    ["clip_l", "clip_g"],
    },
    "illustrious": {
        "label":   "Illustrious (SDXL)",
        "folders": ["illustrious/"],
        "names":   ["illustrious", "noob"],
        "vae":     ["sdxl_vae", "sdxl-vae"],
        "clip":    ["clip_l", "clip_g"],
    },
    "pony": {
        "label":   "Pony (SDXL)",
        "folders": [],
        "names":   ["pony"],
        "vae":     ["sdxl_vae", "sdxl-vae"],
        "clip":    ["clip_l", "clip_g"],
    },
    # ── SD 1.x / 2.x ────────────────────────────────────────────────────────
    "sd2": {
        "label":   "SD 2.x",
        "folders": [],
        "names":   ["sd-2", "sd_2", "sd2", "v2-", "v2_", "768-v"],
        "vae":     ["vae-ft-mse-840000"],
        "clip":    ["clip_l"],
    },
    "sd1.5": {
        "label":   "SD 1.5",
        "folders": [],
        "names":   ["sd-1", "sd_1", "sd1", "v1-5", "v1_5", "v1.5"],
        "vae":     ["vae-ft-mse-840000"],
        "clip":    ["clip_l"],
    },
    # ── Video models ─────────────────────────────────────────────────────────
    "wan": {
        "label":   "WAN 2.x",
        "folders": ["wan2_2/", "wan/"],
        "names":   ["wan2", "wan_", "wanvideo"],
        "vae":     ["wan_2.1_vae", "wan2.1_vae", "wan2.2_vae"],
        "clip":    ["umt5_xxl"],
    },
    "ltxv": {
        "label":   "LTX-Video",
        "folders": ["ltx/"],
        "names":   ["ltxv", "ltx-video", "ltx-2"],
        "vae":     ["ltx23_video_vae", "taeltx"],
        "clip":    ["t5xxl", "ltx-2", "gemma_3"],
    },
    "cogvideo": {
        "label":   "CogVideo",
        "folders": ["cogvideo/"],
        "names":   ["cogvideo"],
        "vae":     [],
        "clip":    ["t5xxl"],
    },
    "mochi": {
        "label":   "Mochi",
        "folders": [],
        "names":   ["mochi"],
        "vae":     [],
        "clip":    ["t5xxl"],
    },
    "hunyuan": {
        "label":   "HunyuanVideo",
        "folders": ["hunyuan/"],
        "names":   ["hunyuan"],
        "vae":     [],
        "clip":    ["llama_3", "clip_l"],
    },
    # ── Qwen Image ───────────────────────────────────────────────────────────
    "qwen_image": {
        "label":   "Qwen Image",
        "folders": ["qwen/"],
        "names":   ["qwen_image"],
        "vae":     ["qwen_image_vae"],
        "clip":    ["qwen_2.5_vl"],
    },
    # ── Other ────────────────────────────────────────────────────────────────
    "cascade": {
        "label":   "Stable Cascade",
        "folders": [],
        "names":   ["cascade", "stable-cascade"],
        "vae":     [],
        "clip":    ["clip_g"],
    },
    "pixart": {
        "label":   "PixArt",
        "folders": [],
        "names":   ["pixart"],
        "vae":     [],
        "clip":    ["t5xxl"],
    },
    "kolors": {
        "label":   "Kolors",
        "folders": [],
        "names":   ["kolors"],
        "vae":     ["sdxl_vae"],
        "clip":    [],
    },
    "auraflow": {
        "label":   "AuraFlow",
        "folders": [],
        "names":   ["auraflow"],
        "vae":     [],
        "clip":    [],
    },
}

# Compatibility groups — checkpoint families that can use each other's models.
# VAE/CLIP are per-family (not grouped) since they differ.
MODEL_COMPAT_GROUPS = [
    {"sdxl", "sdxl_turbo", "pony", "illustrious"},  # SDXL-arch fine-tunes
    {"flux_dev", "flux_schnell", "flux"},             # Flux1 variants
    {"zib", "zit"},                                   # Z-Image variants
]


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


def get_compatible_families(family):
    """Return the set of all checkpoint-families compatible with the given one."""
    if not family:
        return set()
    result = {family}
    for group in MODEL_COMPAT_GROUPS:
        if family in group:
            result |= group
    return result


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


def list_compatible_vaes(family):
    """Return sorted list of VAE files compatible with the given family."""
    if not family:
        # Unknown family — return all VAEs
        try:
            return sorted(folder_paths.get_filename_list("vae"))
        except Exception:
            return []

    # Collect VAE patterns from the family (NOT from compat group — VAE is family-specific)
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


def get_all_family_labels():
    """Return a dict of {family_key: label} for all known families."""
    return {k: v.get("label", k) for k, v in MODEL_FAMILIES.items()}

# Node types that contain sampler parameters
KSAMPLER_TYPES = [
    'KSampler', 'KSamplerAdvanced', 'KSamplerSelect',
    'SamplerCustomAdvanced', 'SamplerCustom',
    'WanMoeKSamplerAdvanced',
]

# Node types for the advanced sampler pattern (Flux/WAN)
SCHEDULER_TYPES = ['BasicScheduler', 'Flux2Scheduler', 'SDTurboScheduler', 'KarrasScheduler']
GUIDER_TYPES = ['BasicGuider', 'CFGGuider', 'DualCFGGuider']
NOISE_TYPES = ['RandomNoise', 'DisableNoise']

# Node types for VAE
VAE_LOADER_TYPES = ['VAELoader', 'VAELoaderConsistencyDecoder']

# Node types for CLIP
CLIP_LOADER_TYPES = ['CLIPLoader', 'DualCLIPLoader', 'TripleCLIPLoader']

# Node types for empty latent (resolution)
LATENT_TYPES = ['EmptyLatentImage', 'EmptyFlux2LatentImage', 'EmptySD3LatentImage']
VIDEO_LATENT_TYPES = ['WanVideoLatentImage', 'WanImageToVideo']

# All available samplers and schedulers from ComfyUI
import comfy.samplers
SAMPLERS = comfy.samplers.KSampler.SAMPLERS
SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS


# ─── Extraction helpers ─────────────────────────────────────────────────────

def extract_sampler_params(prompt_data, workflow_data):
    """
    Extract sampler parameters (steps, cfg, seed, sampler, scheduler, denoise)
    from KSampler nodes in both API format and workflow format.
    Returns a dict with the found parameters.
    """
    params = {
        'steps': 20,
        'cfg': 7.0,
        'seed': 0,
        'sampler_name': 'euler',
        'scheduler': 'normal',
        'denoise': 1.0,
        'guidance': None,  # Flux guidance
    }

    found = False

    # Try API format first (prompt_data)
    if prompt_data and isinstance(prompt_data, dict):
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            class_type = node_data.get('class_type', '')
            inputs = node_data.get('inputs', {})

            # Standard KSampler
            if class_type in ('KSampler', 'KSamplerAdvanced'):
                params['steps'] = inputs.get('steps', params['steps'])
                params['cfg'] = inputs.get('cfg', params['cfg'])
                params['seed'] = inputs.get('seed', inputs.get('noise_seed', params['seed']))
                params['sampler_name'] = inputs.get('sampler_name', params['sampler_name'])
                params['scheduler'] = inputs.get('scheduler', params['scheduler'])
                params['denoise'] = inputs.get('denoise', params['denoise'])
                found = True
                break

            # SamplerCustomAdvanced — parameters are split across multiple nodes
            if class_type == 'SamplerCustomAdvanced':
                found = True
                # We need to trace connected nodes for sampler/scheduler/noise/guider
                # Continue scanning to find those nodes
                continue

        # For advanced pattern, scan for component nodes
        if found or not found:
            for node_id, node_data in prompt_data.items():
                if not isinstance(node_data, dict):
                    continue
                class_type = node_data.get('class_type', '')
                inputs = node_data.get('inputs', {})

                if class_type == 'KSamplerSelect':
                    params['sampler_name'] = inputs.get('sampler_name', params['sampler_name'])

                elif class_type in ('BasicScheduler', 'Flux2Scheduler'):
                    params['steps'] = inputs.get('steps', params['steps'])
                    params['scheduler'] = inputs.get('scheduler', params['scheduler'])
                    params['denoise'] = inputs.get('denoise', params['denoise'])
                    if 'max_shift' in inputs:
                        params['guidance'] = inputs.get('max_shift')

                elif class_type == 'CFGGuider':
                    params['cfg'] = inputs.get('cfg', params['cfg'])

                elif class_type == 'BasicGuider':
                    # BasicGuider doesn't have cfg — used with Flux
                    pass

                elif class_type == 'RandomNoise':
                    params['seed'] = inputs.get('noise_seed', params['seed'])

                elif class_type == 'WanMoeKSamplerAdvanced':
                    params['steps'] = inputs.get('steps', params['steps'])
                    params['cfg'] = inputs.get('cfg', params['cfg'])
                    params['seed'] = inputs.get('seed', inputs.get('noise_seed', params['seed']))
                    params['sampler_name'] = inputs.get('sampler_name', params['sampler_name'])
                    params['scheduler'] = inputs.get('scheduler', params['scheduler'])
                    found = True

    # Fallback: try workflow format
    if not found and workflow_data and isinstance(workflow_data, dict):
        node_map = build_node_map(workflow_data)
        for node_id, node in node_map.items():
            node_type = node.get('type', '')
            widgets = node.get('widgets_values', [])

            if node_type == 'KSampler' and len(widgets) >= 6:
                # KSampler widgets: [seed, control_after_generate, steps, cfg, sampler_name, scheduler, denoise]
                try:
                    params['seed'] = int(widgets[0]) if widgets[0] is not None else 0
                    params['steps'] = int(widgets[2]) if widgets[2] is not None else 20
                    params['cfg'] = float(widgets[3]) if widgets[3] is not None else 7.0
                    params['sampler_name'] = str(widgets[4]) if widgets[4] else 'euler'
                    params['scheduler'] = str(widgets[5]) if widgets[5] else 'normal'
                    if len(widgets) > 6 and widgets[6] is not None:
                        params['denoise'] = float(widgets[6])
                    found = True
                    break
                except (ValueError, IndexError):
                    pass

            elif node_type == 'KSamplerSelect' and len(widgets) >= 1:
                params['sampler_name'] = str(widgets[0]) if widgets[0] else params['sampler_name']

            elif node_type in ('BasicScheduler', 'Flux2Scheduler') and len(widgets) >= 2:
                try:
                    params['scheduler'] = str(widgets[0]) if widgets[0] else params['scheduler']
                    params['steps'] = int(widgets[1]) if widgets[1] is not None else params['steps']
                    if len(widgets) > 2 and widgets[2] is not None:
                        params['denoise'] = float(widgets[2])
                except (ValueError, IndexError):
                    pass

            elif node_type == 'CFGGuider' and len(widgets) >= 1:
                try:
                    params['cfg'] = float(widgets[0]) if widgets[0] is not None else params['cfg']
                except (ValueError, IndexError):
                    pass

            elif node_type == 'RandomNoise' and len(widgets) >= 1:
                try:
                    params['seed'] = int(widgets[0]) if widgets[0] is not None else 0
                except (ValueError, IndexError):
                    pass

    return params


def extract_vae_info(prompt_data, workflow_data):
    """Extract VAE loader information from the workflow."""
    vae_info = {'name': '', 'source': 'unknown'}

    # Try API format
    if prompt_data and isinstance(prompt_data, dict):
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            class_type = node_data.get('class_type', '')
            inputs = node_data.get('inputs', {})

            if class_type in VAE_LOADER_TYPES:
                vae_info['name'] = inputs.get('vae_name', '')
                vae_info['source'] = 'separate'
                return vae_info

            # Checkpoint includes VAE
            if class_type in ('CheckpointLoaderSimple', 'CheckpointLoader', 'CheckpointLoaderNF4'):
                ckpt_name = inputs.get('ckpt_name', '')
                if ckpt_name:
                    vae_info['name'] = f"(bundled with {os.path.basename(ckpt_name)})"
                    vae_info['source'] = 'checkpoint'
                    # Don't return yet — a separate VAE loader overrides checkpoint VAE

    # Fallback: workflow format
    if not vae_info['name'] and workflow_data and isinstance(workflow_data, dict):
        node_map = build_node_map(workflow_data)
        for node_id, node in node_map.items():
            node_type = node.get('type', '')
            widgets = node.get('widgets_values', [])

            if node_type in VAE_LOADER_TYPES and widgets:
                vae_info['name'] = str(widgets[0]) if widgets[0] else ''
                vae_info['source'] = 'separate'
                return vae_info

            if node_type in ('CheckpointLoaderSimple', 'CheckpointLoader') and widgets:
                ckpt_name = str(widgets[0]) if widgets[0] else ''
                if ckpt_name and not vae_info['name']:
                    vae_info['name'] = f"(bundled with {os.path.basename(ckpt_name)})"
                    vae_info['source'] = 'checkpoint'

    return vae_info


def extract_clip_info(prompt_data, workflow_data):
    """Extract CLIP loader information from the workflow."""
    clip_info = {'names': [], 'type': '', 'source': 'unknown'}

    # Try API format
    if prompt_data and isinstance(prompt_data, dict):
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            class_type = node_data.get('class_type', '')
            inputs = node_data.get('inputs', {})

            if class_type == 'CLIPLoader':
                clip_info['names'] = [inputs.get('clip_name', '')]
                clip_info['type'] = inputs.get('type', '')
                clip_info['source'] = 'separate'
                return clip_info

            if class_type == 'DualCLIPLoader':
                clip_info['names'] = [
                    inputs.get('clip_name1', ''),
                    inputs.get('clip_name2', '')
                ]
                clip_info['type'] = inputs.get('type', '')
                clip_info['source'] = 'separate'
                return clip_info

            if class_type == 'TripleCLIPLoader':
                clip_info['names'] = [
                    inputs.get('clip_name1', ''),
                    inputs.get('clip_name2', ''),
                    inputs.get('clip_name3', '')
                ]
                clip_info['source'] = 'separate'
                return clip_info

            # Checkpoint includes CLIP
            if class_type in ('CheckpointLoaderSimple', 'CheckpointLoader', 'CheckpointLoaderNF4'):
                ckpt_name = inputs.get('ckpt_name', '')
                if ckpt_name:
                    clip_info['names'] = [f"(bundled with {os.path.basename(ckpt_name)})"]
                    clip_info['source'] = 'checkpoint'

    # Fallback: workflow format
    if not clip_info['names'] and workflow_data and isinstance(workflow_data, dict):
        node_map = build_node_map(workflow_data)
        for node_id, node in node_map.items():
            node_type = node.get('type', '')
            widgets = node.get('widgets_values', [])

            if node_type == 'CLIPLoader' and widgets:
                clip_info['names'] = [str(widgets[0]) if widgets[0] else '']
                clip_info['type'] = str(widgets[1]) if len(widgets) > 1 and widgets[1] else ''
                clip_info['source'] = 'separate'
                return clip_info

            if node_type == 'DualCLIPLoader' and len(widgets) >= 2:
                clip_info['names'] = [
                    str(widgets[0]) if widgets[0] else '',
                    str(widgets[1]) if widgets[1] else ''
                ]
                clip_info['type'] = str(widgets[2]) if len(widgets) > 2 and widgets[2] else ''
                clip_info['source'] = 'separate'
                return clip_info

            if node_type in ('CheckpointLoaderSimple', 'CheckpointLoader') and widgets:
                ckpt_name = str(widgets[0]) if widgets[0] else ''
                if ckpt_name and not clip_info['names']:
                    clip_info['names'] = [f"(bundled with {os.path.basename(ckpt_name)})"]
                    clip_info['source'] = 'checkpoint'

    return clip_info


def extract_resolution(prompt_data, workflow_data):
    """Extract image/video resolution from Empty Latent nodes."""
    resolution = {'width': 512, 'height': 512, 'batch_size': 1, 'length': None}

    # Try API format
    if prompt_data and isinstance(prompt_data, dict):
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            class_type = node_data.get('class_type', '')
            inputs = node_data.get('inputs', {})

            if class_type in LATENT_TYPES + VIDEO_LATENT_TYPES:
                resolution['width'] = inputs.get('width', resolution['width'])
                resolution['height'] = inputs.get('height', resolution['height'])
                resolution['batch_size'] = inputs.get('batch_size', resolution['batch_size'])
                if 'length' in inputs:
                    resolution['length'] = inputs.get('length')
                return resolution

    # Fallback: workflow format
    if workflow_data and isinstance(workflow_data, dict):
        node_map = build_node_map(workflow_data)
        for node_id, node in node_map.items():
            node_type = node.get('type', '')
            widgets = node.get('widgets_values', [])

            if node_type in LATENT_TYPES and len(widgets) >= 3:
                try:
                    resolution['width'] = int(widgets[0]) if widgets[0] else 512
                    resolution['height'] = int(widgets[1]) if widgets[1] else 512
                    resolution['batch_size'] = int(widgets[2]) if widgets[2] else 1
                except (ValueError, IndexError):
                    pass
                return resolution

            if node_type in VIDEO_LATENT_TYPES and len(widgets) >= 3:
                try:
                    resolution['width'] = int(widgets[0]) if widgets[0] else 512
                    resolution['height'] = int(widgets[1]) if widgets[1] else 512
                    if len(widgets) > 2:
                        resolution['length'] = int(widgets[2]) if widgets[2] else 81
                    resolution['batch_size'] = int(widgets[3]) if len(widgets) > 3 and widgets[3] else 1
                except (ValueError, IndexError):
                    pass
                return resolution

    return resolution


# ─── Model resolution (reuse pattern from PromptModelLoader) ────────────────

def resolve_model_name(model_name):
    """Resolve model name to (relative_path, folder_name) or (None, None)."""
    if not model_name:
        return None, None

    model_name_clean = model_name.strip().replace('\\', '/')
    name_base = os.path.basename(model_name_clean).lower()

    name_no_ext = name_base
    for ext in MODEL_EXTENSIONS:
        if name_no_ext.endswith(ext):
            name_no_ext = name_no_ext[:-len(ext)]
            break

    for folder_name in ['checkpoints', 'diffusion_models', 'unet', 'unet_gguf']:
        try:
            file_list = folder_paths.get_filename_list(folder_name)
        except Exception:
            continue

        for f in file_list:
            f_normalized = f.replace('\\', '/')
            if f_normalized == model_name_clean:
                return f, folder_name
            f_base = os.path.basename(f_normalized).lower()
            if f_base == name_base:
                return f, folder_name
            f_no_ext = f_base
            for ext in MODEL_EXTENSIONS:
                if f_no_ext.endswith(ext):
                    f_no_ext = f_no_ext[:-len(ext)]
                    break
            if f_no_ext == name_no_ext:
                return f, folder_name

    return None, None


def resolve_vae_name(vae_name):
    """Resolve VAE name to full path, or None."""
    if not vae_name or vae_name.startswith('('):
        return None
    try:
        vae_list = folder_paths.get_filename_list("vae")
        name_lower = os.path.basename(vae_name).lower()
        for v in vae_list:
            if v == vae_name or os.path.basename(v).lower() == name_lower:
                return folder_paths.get_full_path("vae", v)
    except Exception:
        pass
    return None


def resolve_clip_names(clip_names, clip_type=''):
    """Resolve CLIP names to full paths. Returns list of paths (or None for not found)."""
    paths = []
    for name in clip_names:
        if not name or name.startswith('('):
            paths.append(None)
            continue
        found = None
        name_lower = os.path.basename(name).lower()
        for folder in ['text_encoders', 'clip']:
            try:
                file_list = folder_paths.get_filename_list(folder)
                for f in file_list:
                    if f == name or os.path.basename(f).lower() == name_lower:
                        found = folder_paths.get_full_path(folder, f)
                        break
                if found:
                    break
            except Exception:
                continue
        paths.append(found)
    return paths


# ─── Full extraction from file ──────────────────────────────────────────────

def extract_all_from_file(file_path, source_folder='input'):
    """
    Extract ALL generation parameters from a file.
    Returns a dict with: prompt, loras, model, vae, clip, sampler_params, resolution
    """
    result = {
        'positive_prompt': '',
        'negative_prompt': '',
        'loras_a': [],
        'loras_b': [],
        'model_a': '',
        'model_b': '',
        'vae': {'name': '', 'source': 'unknown'},
        'clip': {'names': [], 'type': '', 'source': 'unknown'},
        'sampler': {
            'steps': 20, 'cfg': 7.0, 'seed': 0,
            'sampler_name': 'euler', 'scheduler': 'normal',
            'denoise': 1.0, 'guidance': None,
        },
        'resolution': {'width': 512, 'height': 512, 'batch_size': 1, 'length': None},
        'is_video': False,
    }

    # Determine file type and extract metadata
    ext = os.path.splitext(file_path)[1].lower()
    prompt_data = None
    workflow_data = None

    if ext in ('.png',):
        prompt_data, workflow_data = extract_metadata_from_png(file_path)
    elif ext in ('.jpg', '.jpeg', '.webp'):
        prompt_data, workflow_data = extract_metadata_from_jpeg(file_path)
    elif ext in ('.json',):
        prompt_data, workflow_data = extract_metadata_from_json(file_path)
    elif ext in ('.mp4', '.webm', '.mov', '.avi'):
        prompt_data, workflow_data = extract_metadata_from_video(file_path)
        result['is_video'] = True

    if not prompt_data and not workflow_data:
        return result

    # Use Prompt Manager's extraction for prompts, LoRAs, models
    extracted = parse_workflow_for_prompts(prompt_data, workflow_data)
    result['positive_prompt'] = extracted.get('positive_prompt', '')
    result['negative_prompt'] = extracted.get('negative_prompt', '')
    result['loras_a'] = extracted.get('loras_a', [])
    result['loras_b'] = extracted.get('loras_b', [])

    models_a = extracted.get('models_a', [])
    models_b = extracted.get('models_b', [])
    result['model_a'] = os.path.basename(models_a[0].replace('\\', '/')) if models_a else ''
    result['model_b'] = os.path.basename(models_b[0].replace('\\', '/')) if models_b else ''

    # Extract additional parameters not covered by Prompt Manager
    result['sampler'] = extract_sampler_params(prompt_data, workflow_data)
    result['vae'] = extract_vae_info(prompt_data, workflow_data)
    result['clip'] = extract_clip_info(prompt_data, workflow_data)
    result['resolution'] = extract_resolution(prompt_data, workflow_data)

    # Detect if video workflow
    if result['resolution']['length'] is not None:
        result['is_video'] = True

    return result


# ─── API endpoints ──────────────────────────────────────────────────────────

@server.PromptServer.instance.routes.post("/workflow-extractor/extract")
async def api_extract(request):
    """Extract all parameters from a file and return to JS for UI display."""
    try:
        data = await request.json()
        filename = data.get('filename', '')
        source = data.get('source', 'input')

        if not filename:
            return server.web.json_response({"error": "No filename"}, status=400)

        # Build full path
        if source == 'output':
            base_dir = folder_paths.get_output_directory()
        else:
            base_dir = folder_paths.get_input_directory()

        file_path = os.path.join(base_dir, filename.replace('/', os.sep))

        # Validate path stays within base directory
        real_base = os.path.realpath(base_dir)
        real_path = os.path.realpath(file_path)
        if not real_path.startswith(real_base):
            return server.web.json_response({"error": "Invalid path"}, status=403)

        if not os.path.exists(file_path):
            return server.web.json_response({"error": "File not found"}, status=404)

        result = extract_all_from_file(file_path, source)

        # Check LoRA availability
        lora_availability = {}
        for stack_label, loras in [('a', result['loras_a']), ('b', result['loras_b'])]:
            for lora in loras:
                name = lora.get('name', '')
                if name:
                    _, found = resolve_lora_path(name)
                    lora_availability[name] = found

        result['lora_availability'] = lora_availability

        # Check model availability
        for key in ['model_a', 'model_b']:
            model_name = result[key]
            if model_name:
                resolved, folder = resolve_model_name(model_name)
                result[f'{key}_found'] = resolved is not None
                # Use resolved path for family detection (has folder prefix)
                result[f'{key}_resolved'] = resolved
            else:
                result[f'{key}_found'] = True
                result[f'{key}_resolved'] = None

        # Detect model family from resolved path (most reliable) or raw name
        ref_path = result.get('model_a_resolved') or result.get('model_a', '')
        family = get_model_family(ref_path)
        result['model_family'] = family
        result['model_family_label'] = get_family_label(family)

        # Check VAE availability
        if result['vae']['name'] and not result['vae']['name'].startswith('('):
            result['vae_found'] = resolve_vae_name(result['vae']['name']) is not None
        else:
            result['vae_found'] = True

        # Check CLIP availability
        clip_found = []
        for name in result['clip']['names']:
            if name and not name.startswith('('):
                paths = resolve_clip_names([name])
                clip_found.append(paths[0] is not None)
            else:
                clip_found.append(True)
        result['clip_found'] = clip_found

        return server.web.json_response(result)
    except Exception as e:
        print(f"[WorkflowExtractor] API error: {e}")
        traceback.print_exc()
        return server.web.json_response({"error": str(e)}, status=500)


@server.PromptServer.instance.routes.get("/workflow-extractor/list-models")
async def api_list_models(request):
    """List compatible models. Pass ?ref=<model> or ?family=<key>."""
    try:
        family_key = request.rel_url.query.get('family', '')
        ref = request.rel_url.query.get('ref', '')
        if family_key:
            # Direct family override from UI
            compat = get_compatible_families(family_key)
            all_models = []
            for fn in ['checkpoints', 'diffusion_models', 'unet', 'unet_gguf']:
                try:
                    all_models.extend(folder_paths.get_filename_list(fn))
                except Exception:
                    continue
            seen = set()
            models = []
            for m in all_models:
                if m not in seen:
                    seen.add(m)
                    mf = get_model_family(m)
                    if mf and mf in compat:
                        models.append(m)
            family = family_key
        elif ref:
            models = list_compatible_models(ref)
            family = get_model_family(ref)
        else:
            model_type = request.rel_url.query.get('type', 'checkpoints')
            models = sorted(folder_paths.get_filename_list(model_type))
            family = None
        return server.web.json_response({
            "models": sorted(models),
            "family": family,
            "family_label": get_family_label(family),
        })
    except Exception as e:
        return server.web.json_response({"models": [], "error": str(e)})


@server.PromptServer.instance.routes.get("/workflow-extractor/list-vaes")
async def api_list_vaes(request):
    """List compatible VAEs. Pass ?family=<key> to filter."""
    try:
        family = request.rel_url.query.get('family', '') or None
        vaes = list_compatible_vaes(family)
        return server.web.json_response({"vaes": vaes})
    except Exception as e:
        return server.web.json_response({"vaes": [], "error": str(e)})


@server.PromptServer.instance.routes.get("/workflow-extractor/list-clips")
async def api_list_clips(request):
    """List compatible CLIPs. Pass ?family=<key> to filter."""
    try:
        family = request.rel_url.query.get('family', '') or None
        clips = list_compatible_clips(family)
        return server.web.json_response({"clips": clips})
    except Exception as e:
        return server.web.json_response({"clips": [], "error": str(e)})


@server.PromptServer.instance.routes.get("/workflow-extractor/list-families")
async def api_list_families(request):
    """List all known model families for the type selector."""
    try:
        return server.web.json_response({"families": get_all_family_labels()})
    except Exception as e:
        return server.web.json_response({"families": {}, "error": str(e)})


# ─── Main Node ──────────────────────────────────────────────────────────────

class WorkflowExtractor:
    """
    Extract ALL generation parameters from an image/video and regenerate internally.
    Loads model, applies LoRAs, runs KSampler, outputs IMAGE + LATENT.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # Get available files
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()
        supported = ['.png', '.jpg', '.jpeg', '.webp', '.json', '.mp4', '.webm', '.mov', '.avi']

        files = []
        for base_dir in [input_dir, output_dir]:
            if os.path.exists(base_dir):
                for root, dirs, filenames in os.walk(base_dir):
                    for fn in filenames:
                        if os.path.splitext(fn)[1].lower() in supported:
                            rel = os.path.relpath(os.path.join(root, fn), base_dir).replace('\\', '/')
                            files.append(rel)
        files = sorted(set(files))
        if not files:
            files = [""]

        return {
            "required": {
                "source_folder": (["input", "output"], {"default": "input"}),
                "image": (files, {}),
            },
            "optional": {
                # Override toggles — stored as JSON strings from JS
                "override_data": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "hidden": True,
                }),
                # LoRA toggle state — stored as JSON string from JS
                "lora_state": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "hidden": True,
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "execute"
    CATEGORY = "FBnodes"
    OUTPUT_NODE = False
    DESCRIPTION = "Extract and display ALL generation parameters from an image/video. Pure display node — does not execute."

    def execute(self, source_folder="input", image="", override_data="{}", lora_state="{}",
                unique_id=None, extra_pnginfo=None, prompt=None):
        """No-op: this node is display-only. Parameters are extracted via the API."""
        return {}

