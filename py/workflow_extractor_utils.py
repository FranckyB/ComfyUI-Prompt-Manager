"""
Shared extraction helpers for ComfyUI Prompt Manager.

Used by both WorkflowExtractor and WorkflowGenerator nodes.
Handles: sampler params, VAE info, CLIP info, resolution, model resolution,
and the full extract_all_from_file() entry point.
"""
import os
import json
import folder_paths

from .workflow_families import get_model_family, get_family_label

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_EXTENSIONS = ['.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf', '.sft']

# Node types that contain sampler parameters
KSAMPLER_TYPES = [
    'KSampler', 'KSamplerAdvanced', 'KSamplerSelect',
    'SamplerCustomAdvanced', 'SamplerCustom',
    'WanMoeKSamplerAdvanced',
]

SCHEDULER_TYPES = ['BasicScheduler', 'Flux2Scheduler', 'SDTurboScheduler', 'KarrasScheduler']
GUIDER_TYPES    = ['BasicGuider', 'CFGGuider', 'DualCFGGuider']
NOISE_TYPES     = ['RandomNoise', 'DisableNoise']
VAE_LOADER_TYPES  = ['VAELoader', 'VAELoaderConsistencyDecoder']
CLIP_LOADER_TYPES = ['CLIPLoader', 'DualCLIPLoader', 'TripleCLIPLoader']
LATENT_TYPES      = ['EmptyLatentImage', 'EmptyFlux2LatentImage', 'EmptySD3LatentImage']
VIDEO_LATENT_TYPES = ['WanVideoLatentImage', 'WanImageToVideo']

CHECKPOINT_TYPES = ('CheckpointLoaderSimple', 'CheckpointLoader', 'CheckpointLoaderNF4')

# ─── Sampler extraction ───────────────────────────────────────────────────────

def extract_sampler_params(prompt_data, workflow_data):
    """
    Extract sampler parameters from KSampler nodes in API or workflow format.
    Returns a dict with steps, cfg, seed, sampler_name, scheduler, denoise, guidance.
    """
    params = {
        'steps': 20,
        'cfg': 7.0,
        'seed': 0,
        'sampler_name': 'euler',
        'scheduler': 'normal',
        'denoise': 1.0,
        'guidance': None,  # Flux guidance / max_shift
    }

    # ── API format ────────────────────────────────────────────────────────────
    if prompt_data and isinstance(prompt_data, dict):
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            ct = node_data.get('class_type', '')
            inp = node_data.get('inputs', {})

            if ct in ('KSampler', 'KSamplerAdvanced'):
                params['steps']        = inp.get('steps', params['steps'])
                params['cfg']          = inp.get('cfg', params['cfg'])
                params['seed']         = inp.get('seed', inp.get('noise_seed', params['seed']))
                params['sampler_name'] = inp.get('sampler_name', params['sampler_name'])
                params['scheduler']    = inp.get('scheduler', params['scheduler'])
                params['denoise']      = inp.get('denoise', params['denoise'])
                return params  # standard KSampler found — done

            if ct == 'WanMoeKSamplerAdvanced':
                params['steps']        = inp.get('steps', params['steps'])
                params['cfg']          = inp.get('cfg', params['cfg'])
                params['seed']         = inp.get('seed', inp.get('noise_seed', params['seed']))
                params['sampler_name'] = inp.get('sampler_name', params['sampler_name'])
                params['scheduler']    = inp.get('scheduler', params['scheduler'])
                return params

        # Second pass for split-node pattern (SamplerCustomAdvanced)
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            ct  = node_data.get('class_type', '')
            inp = node_data.get('inputs', {})

            if ct == 'KSamplerSelect':
                params['sampler_name'] = inp.get('sampler_name', params['sampler_name'])
            elif ct in ('BasicScheduler', 'Flux2Scheduler'):
                params['steps']     = inp.get('steps', params['steps'])
                params['scheduler'] = inp.get('scheduler', params['scheduler'])
                params['denoise']   = inp.get('denoise', params['denoise'])
                if 'max_shift' in inp:
                    params['guidance'] = inp.get('max_shift')
            elif ct == 'CFGGuider':
                params['cfg'] = inp.get('cfg', params['cfg'])
            elif ct == 'RandomNoise':
                params['seed'] = inp.get('noise_seed', params['seed'])

    # ── Workflow (node graph) format fallback ─────────────────────────────────
    if workflow_data and isinstance(workflow_data, dict):
        from .workflow_node_utils import build_node_map
        node_map = build_node_map(workflow_data)
        for node_id, node in node_map.items():
            ntype   = node.get('type', '')
            widgets = node.get('widgets_values', [])

            if ntype == 'KSampler' and len(widgets) >= 6:
                try:
                    params['seed']         = int(widgets[0])   if widgets[0] is not None else 0
                    params['steps']        = int(widgets[2])   if widgets[2] is not None else 20
                    params['cfg']          = float(widgets[3]) if widgets[3] is not None else 7.0
                    params['sampler_name'] = str(widgets[4])   if widgets[4] else 'euler'
                    params['scheduler']    = str(widgets[5])   if widgets[5] else 'normal'
                    if len(widgets) > 6 and widgets[6] is not None:
                        params['denoise'] = float(widgets[6])
                    return params
                except (ValueError, IndexError):
                    pass

            elif ntype == 'KSamplerSelect' and widgets:
                params['sampler_name'] = str(widgets[0]) if widgets[0] else params['sampler_name']
            elif ntype in ('BasicScheduler', 'Flux2Scheduler') and len(widgets) >= 2:
                try:
                    params['scheduler'] = str(widgets[0]) if widgets[0] else params['scheduler']
                    params['steps']     = int(widgets[1]) if widgets[1] is not None else params['steps']
                    if len(widgets) > 2 and widgets[2] is not None:
                        params['denoise'] = float(widgets[2])
                except (ValueError, IndexError):
                    pass
            elif ntype == 'CFGGuider' and widgets:
                try:
                    params['cfg'] = float(widgets[0]) if widgets[0] is not None else params['cfg']
                except (ValueError, IndexError):
                    pass
            elif ntype == 'RandomNoise' and widgets:
                try:
                    params['seed'] = int(widgets[0]) if widgets[0] is not None else 0
                except (ValueError, IndexError):
                    pass

    return params


# ─── VAE extraction ───────────────────────────────────────────────────────────

def extract_vae_info(prompt_data, workflow_data):
    """Extract VAE loader information from the workflow. Returns {'name', 'source'}."""
    vae_info = {'name': '', 'source': 'unknown'}

    if prompt_data and isinstance(prompt_data, dict):
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            ct  = node_data.get('class_type', '')
            inp = node_data.get('inputs', {})

            if ct in VAE_LOADER_TYPES:
                vae_info['name']   = inp.get('vae_name', '')
                vae_info['source'] = 'separate'
                return vae_info

            if ct in CHECKPOINT_TYPES and inp.get('ckpt_name'):
                if not vae_info['name']:
                    vae_info['name']   = '(from checkpoint)'
                    vae_info['source'] = 'checkpoint'
                # Don't return — a separate VAE loader later overrides this

    if not vae_info['name'] and workflow_data and isinstance(workflow_data, dict):
        from .workflow_node_utils import build_node_map
        node_map = build_node_map(workflow_data)
        for node_id, node in node_map.items():
            ntype   = node.get('type', '')
            widgets = node.get('widgets_values', [])

            if ntype in VAE_LOADER_TYPES and widgets:
                vae_info['name']   = str(widgets[0]) if widgets[0] else ''
                vae_info['source'] = 'separate'
                return vae_info

            if ntype in CHECKPOINT_TYPES and widgets and not vae_info['name']:
                vae_info['name']   = '(from checkpoint)'
                vae_info['source'] = 'checkpoint'

    return vae_info


# ─── CLIP extraction ──────────────────────────────────────────────────────────

def extract_clip_info(prompt_data, workflow_data):
    """Extract CLIP loader information. Returns {'names', 'type', 'source'}."""
    clip_info = {'names': [], 'type': '', 'source': 'unknown'}

    if prompt_data and isinstance(prompt_data, dict):
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            ct  = node_data.get('class_type', '')
            inp = node_data.get('inputs', {})

            if ct == 'CLIPLoader':
                clip_info['names']  = [inp.get('clip_name', '')]
                clip_info['type']   = inp.get('type', '')
                clip_info['source'] = 'separate'
                return clip_info

            if ct == 'DualCLIPLoader':
                clip_info['names']  = [inp.get('clip_name1', ''), inp.get('clip_name2', '')]
                clip_info['type']   = inp.get('type', '')
                clip_info['source'] = 'separate'
                return clip_info

            if ct == 'TripleCLIPLoader':
                clip_info['names']  = [
                    inp.get('clip_name1', ''),
                    inp.get('clip_name2', ''),
                    inp.get('clip_name3', ''),
                ]
                clip_info['source'] = 'separate'
                return clip_info

            if ct in CHECKPOINT_TYPES and inp.get('ckpt_name') and not clip_info['names']:
                clip_info['names']  = ['(from checkpoint)']
                clip_info['source'] = 'checkpoint'

    if not clip_info['names'] and workflow_data and isinstance(workflow_data, dict):
        from .workflow_node_utils import build_node_map
        node_map = build_node_map(workflow_data)
        for node_id, node in node_map.items():
            ntype   = node.get('type', '')
            widgets = node.get('widgets_values', [])

            if ntype == 'CLIPLoader' and widgets:
                clip_info['names']  = [str(widgets[0]) if widgets[0] else '']
                clip_info['type']   = str(widgets[1]) if len(widgets) > 1 and widgets[1] else ''
                clip_info['source'] = 'separate'
                return clip_info

            if ntype == 'DualCLIPLoader' and len(widgets) >= 2:
                clip_info['names']  = [
                    str(widgets[0]) if widgets[0] else '',
                    str(widgets[1]) if widgets[1] else '',
                ]
                clip_info['type']   = str(widgets[2]) if len(widgets) > 2 and widgets[2] else ''
                clip_info['source'] = 'separate'
                return clip_info

            if ntype in CHECKPOINT_TYPES and widgets and not clip_info['names']:
                clip_info['names']  = ['(from checkpoint)']
                clip_info['source'] = 'checkpoint'

    return clip_info


# ─── Resolution extraction ────────────────────────────────────────────────────

def extract_resolution(prompt_data, workflow_data):
    """Extract image/video resolution from Empty Latent nodes."""
    resolution = {'width': 512, 'height': 512, 'batch_size': 1, 'length': None}

    if prompt_data and isinstance(prompt_data, dict):
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            ct  = node_data.get('class_type', '')
            inp = node_data.get('inputs', {})

            if ct in LATENT_TYPES + VIDEO_LATENT_TYPES:
                resolution['width']      = inp.get('width',      resolution['width'])
                resolution['height']     = inp.get('height',     resolution['height'])
                resolution['batch_size'] = inp.get('batch_size', resolution['batch_size'])
                if 'length' in inp:
                    resolution['length'] = inp.get('length')
                return resolution

    if workflow_data and isinstance(workflow_data, dict):
        from .workflow_node_utils import build_node_map
        node_map = build_node_map(workflow_data)
        for node_id, node in node_map.items():
            ntype   = node.get('type', '')
            widgets = node.get('widgets_values', [])

            if ntype in LATENT_TYPES and len(widgets) >= 3:
                try:
                    resolution['width']      = int(widgets[0]) if widgets[0] else 512
                    resolution['height']     = int(widgets[1]) if widgets[1] else 512
                    resolution['batch_size'] = int(widgets[2]) if widgets[2] else 1
                except (ValueError, IndexError):
                    pass
                return resolution

            if ntype in VIDEO_LATENT_TYPES and len(widgets) >= 3:
                try:
                    resolution['width']  = int(widgets[0]) if widgets[0] else 512
                    resolution['height'] = int(widgets[1]) if widgets[1] else 512
                    if len(widgets) > 2:
                        resolution['length'] = int(widgets[2]) if widgets[2] else 81
                    resolution['batch_size'] = int(widgets[3]) if len(widgets) > 3 and widgets[3] else 1
                except (ValueError, IndexError):
                    pass
                return resolution

    return resolution


# ─── Model resolution ─────────────────────────────────────────────────────────

def resolve_model_name(model_name):
    """Resolve model name to (relative_path, folder_name) or (None, None)."""
    if not model_name:
        return None, None

    model_name_clean = model_name.strip().replace('\\', '/')
    name_base        = os.path.basename(model_name_clean).lower()
    name_no_ext      = name_base
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
            f_base   = os.path.basename(f_normalized).lower()
            f_no_ext = f_base
            for ext in MODEL_EXTENSIONS:
                if f_no_ext.endswith(ext):
                    f_no_ext = f_no_ext[:-len(ext)]
                    break
            if f_base == name_base or f_no_ext == name_no_ext:
                return f, folder_name

    return None, None


def resolve_vae_name(vae_name):
    """Resolve VAE name to full path, or None if not found / from checkpoint."""
    if not vae_name or vae_name.startswith('('):
        return None
    try:
        vae_list   = folder_paths.get_filename_list("vae")
        name_lower = os.path.basename(vae_name).lower()
        for v in vae_list:
            if v == vae_name or os.path.basename(v).lower() == name_lower:
                return folder_paths.get_full_path("vae", v)
    except Exception:
        pass
    return None


def resolve_clip_names(clip_names, clip_type=''):
    """
    Resolve a list of CLIP names to full paths.
    Returns list of paths (None for any not found).
    """
    paths = []
    for name in clip_names:
        if not name or name.startswith('('):
            paths.append(None)
            continue
        found      = None
        name_lower = os.path.basename(name).lower()
        for folder in ['text_encoders', 'clip']:
            try:
                for f in folder_paths.get_filename_list(folder):
                    if f == name or os.path.basename(f).lower() == name_lower:
                        found = folder_paths.get_full_path(folder, f)
                        break
                if found:
                    break
            except Exception:
                continue
        paths.append(found)
    return paths


# ─── Full extraction ──────────────────────────────────────────────────────────

def extract_all_from_file(file_path, source_folder='input'):
    """
    Extract ALL generation parameters from a file.

    Returns a dict with:
      positive_prompt, negative_prompt,
      loras_a, loras_b,
      model_a, model_b,
      vae  {'name', 'source'},
      clip {'names', 'type', 'source'},
      sampler {'steps', 'cfg', 'seed', 'sampler_name', 'scheduler', 'denoise', 'guidance'},
      resolution {'width', 'height', 'batch_size', 'length'},
      is_video (bool)
    """
    # Import extraction functions from prompt_extractor (lives in nodes/).
    # Late import to avoid circular deps at module load time.
    # Relative import works because py/ and nodes/ are siblings in the same package.
    from ..nodes.prompt_extractor import (
        parse_workflow_for_prompts,
        extract_metadata_from_png,
        extract_metadata_from_jpeg,
        extract_metadata_from_json,
        extract_metadata_from_video,
        resolve_lora_path,
    )

    result = {
        'positive_prompt': '',
        'negative_prompt': '',
        'loras_a':  [],
        'loras_b':  [],
        'model_a':  '',
        'model_b':  '',
        'vae':      {'name': '', 'source': 'unknown'},
        'clip':     {'names': [], 'type': '', 'source': 'unknown'},
        'sampler':  {
            'steps': 20, 'cfg': 7.0, 'seed': 0,
            'sampler_name': 'euler', 'scheduler': 'normal',
            'denoise': 1.0, 'guidance': None,
        },
        'resolution': {'width': 512, 'height': 512, 'batch_size': 1, 'length': None},
        'is_video': False,
    }

    ext           = os.path.splitext(file_path)[1].lower()
    prompt_data   = None
    workflow_data = None

    if ext == '.png':
        prompt_data, workflow_data = extract_metadata_from_png(file_path)
    elif ext in ('.jpg', '.jpeg', '.webp'):
        prompt_data, workflow_data = extract_metadata_from_jpeg(file_path)
    elif ext == '.json':
        prompt_data, workflow_data = extract_metadata_from_json(file_path)
    elif ext in ('.mp4', '.webm', '.mov', '.avi'):
        prompt_data, workflow_data = extract_metadata_from_video(file_path)
        result['is_video'] = True

    if not prompt_data and not workflow_data:
        return result

    extracted = parse_workflow_for_prompts(prompt_data, workflow_data)
    result['positive_prompt'] = extracted.get('positive_prompt', '')
    result['negative_prompt'] = extracted.get('negative_prompt', '')
    result['loras_a']         = extracted.get('loras_a', [])
    result['loras_b']         = extracted.get('loras_b', [])

    models_a         = extracted.get('models_a', [])
    models_b         = extracted.get('models_b', [])
    result['model_a'] = os.path.basename(models_a[0].replace('\\', '/')) if models_a else ''
    result['model_b'] = os.path.basename(models_b[0].replace('\\', '/')) if models_b else ''

    result['sampler']    = extract_sampler_params(prompt_data, workflow_data)
    result['vae']        = extract_vae_info(prompt_data, workflow_data)
    result['clip']       = extract_clip_info(prompt_data, workflow_data)
    result['resolution'] = extract_resolution(prompt_data, workflow_data)

    if result['resolution']['length'] is not None:
        result['is_video'] = True

    return result


def enrich_with_availability(result):
    """
    Add availability flags (lora_availability, model_a_found, etc.)
    to an extract_all_from_file() result dict in-place.
    Returns the same dict.
    """
    from ..nodes.prompt_extractor import resolve_lora_path

    # LoRA availability
    lora_availability = {}
    for loras in [result['loras_a'], result['loras_b']]:
        for lora in loras:
            name = lora.get('name', '')
            if name and name not in lora_availability:
                _, found = resolve_lora_path(name)
                lora_availability[name] = found
    result['lora_availability'] = lora_availability

    # Model A / B availability
    for key in ['model_a', 'model_b']:
        model_name = result[key]
        if model_name:
            resolved, folder = resolve_model_name(model_name)
            result[f'{key}_found']    = resolved is not None
            result[f'{key}_resolved'] = resolved
        else:
            result[f'{key}_found']    = True
            result[f'{key}_resolved'] = None

    # Model family detection (use resolved path — has folder prefix)
    ref_path = result.get('model_a_resolved') or result.get('model_a', '')
    family   = get_model_family(ref_path)
    result['model_family']       = family
    result['model_family_label'] = get_family_label(family)

    # VAE availability
    vae_name = result['vae'].get('name', '')
    if vae_name and not vae_name.startswith('('):
        result['vae_found'] = resolve_vae_name(vae_name) is not None
    else:
        result['vae_found'] = True

    # CLIP availability
    clip_found = []
    for name in result['clip'].get('names', []):
        if name and not name.startswith('('):
            paths = resolve_clip_names([name])
            clip_found.append(paths[0] is not None)
        else:
            clip_found.append(True)
    result['clip_found'] = clip_found

    return result
