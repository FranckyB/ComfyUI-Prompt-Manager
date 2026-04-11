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
VIDEO_LATENT_TYPES = ['WanVideoLatentImage', 'WanImageToVideo', 'EmptyHunyuanLatentVideo']

CHECKPOINT_TYPES = ('CheckpointLoaderSimple', 'CheckpointLoader', 'CheckpointLoaderNF4')

# Node types that carry embedded extracted_data
_EMBEDDED_NODE_TYPES = ('WorkflowGenerator', 'PromptExtractor')


def _get_embedded_extracted_data(workflow_data):
    """Return the best ``extracted_data`` dict from a WG or PE node, or *None*.

    Prioritises WorkflowGenerator over PromptExtractor, and only returns
    data that actually contains ``sampler`` or ``resolution`` keys.
    """
    if not workflow_data or not isinstance(workflow_data, dict):
        return None

    best = None
    best_is_wg = False

    for wf_node in workflow_data.get('nodes', []):
        ntype = wf_node.get('type', '')
        if ntype not in _EMBEDDED_NODE_TYPES:
            continue
        ed = wf_node.get('extracted_data')
        if not ed or not isinstance(ed, dict):
            continue
        has_useful = bool(ed.get('sampler') or ed.get('resolution'))
        if not has_useful:
            continue
        is_wg = ntype == 'WorkflowGenerator'
        # Prefer WG; among same type, first wins
        if best is None or (is_wg and not best_is_wg):
            best = ed
            best_is_wg = is_wg
        if best_is_wg:
            break  # WG is highest priority — stop early

    return best


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

    # ── Fallback: embedded extracted_data from WG / PE nodes ──────────────
    ed = _get_embedded_extracted_data(workflow_data)
    if ed:
        s = ed.get('sampler')
        if s and isinstance(s, dict) and s.get('sampler_name'):
            params['steps']        = s.get('steps', params['steps'])
            params['cfg']          = s.get('cfg', params['cfg'])
            params['seed']         = s.get('seed', params['seed'])
            params['sampler_name'] = s.get('sampler_name', params['sampler_name'])
            params['scheduler']    = s.get('scheduler', params['scheduler'])
            params['denoise']      = s.get('denoise', params['denoise'])
            params['guidance']     = s.get('guidance', params['guidance'])

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

    # ── Fallback: embedded extracted_data from WG / PE nodes ──────────────
    if not vae_info['name']:
        ed = _get_embedded_extracted_data(workflow_data)
        if ed:
            vae = ed.get('vae', '')
            if vae and isinstance(vae, str) and vae not in ('', '\u2014'):
                vae_info['name']   = vae
                vae_info['source'] = 'embedded'

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

    # ── Fallback: embedded extracted_data from WG / PE nodes ──────────────
    if not clip_info['names']:
        ed = _get_embedded_extracted_data(workflow_data)
        if ed:
            clip = ed.get('clip', [])
            names = clip if isinstance(clip, list) else ([clip] if clip else [])
            if any(n for n in names):
                clip_info['names']  = names
                clip_info['source'] = 'embedded'

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
                def _scalar(val, fallback, field_name=None):
                    """Resolve node-ref lists to scalar by looking up the
                    referenced node in prompt_data, or return fallback."""
                    if isinstance(val, list):
                        # Node reference: [node_id, output_index]
                        ref_id = str(val[0])
                        ref_node = prompt_data.get(ref_id, {})
                        ref_inp = ref_node.get('inputs', {})
                        # PrimitiveInt / ImageResizeKJv2 / similar
                        if 'value' in ref_inp:
                            return int(ref_inp['value'])
                        # ImageResizeKJv2: width is output slot 1, height slot 2
                        if field_name == 'width' and 'width' in ref_inp:
                            v = ref_inp['width']
                            return int(v) if not isinstance(v, list) else fallback
                        if field_name == 'height' and 'height' in ref_inp:
                            v = ref_inp['height']
                            return int(v) if not isinstance(v, list) else fallback
                        return fallback
                    try:
                        return int(val)
                    except (TypeError, ValueError):
                        return fallback

                w = inp.get('width',      resolution['width'])
                h = inp.get('height',     resolution['height'])
                b = inp.get('batch_size', resolution['batch_size'])
                # Track whether width/height came from node-refs (runtime values)
                # so execute() knows it can use source_image dims as a better fallback.
                resolution['_width_from_node_ref']  = isinstance(w, list)
                resolution['_height_from_node_ref'] = isinstance(h, list)
                resolution['width']      = _scalar(w, resolution['width'],  'width')
                resolution['height']     = _scalar(h, resolution['height'], 'height')
                # batch_size in WanImageToVideo can equal 'length' — clamp to 1 if > 64
                b_val = _scalar(b, resolution['batch_size'])
                resolution['batch_size'] = b_val if b_val <= 64 else 1
                if 'length' in inp:
                    l = inp.get('length')
                    resolution['length'] = _scalar(l, None) if l is not None else None
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

    # ── Fallback: embedded extracted_data from WG / PE nodes ──────────────
    ed = _get_embedded_extracted_data(workflow_data)
    if ed:
        r = ed.get('resolution')
        if r and isinstance(r, dict) and r.get('width'):
            resolution['width']      = r.get('width', resolution['width'])
            resolution['height']     = r.get('height', resolution['height'])
            resolution['batch_size'] = r.get('batch_size', resolution['batch_size'])
            if r.get('length') is not None:
                resolution['length'] = r['length']

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


# ─── Embedded generation data (WG / PE) ──────────────────────────────────────

def _find_embedded_generation_data(workflow_data, prompt_data):
    """
    Look for sampler / resolution / model data embedded by WorkflowGenerator
    or PromptExtractor.  Checks (in priority order):

      1. WG node ``extracted_data`` (full dict with sampler + resolution)
      2. WG node ``widgets_values`` containing the ``override_data`` JSON
      3. WG ``override_data`` in the prompt API inputs
      4. PE node ``extracted_data``

    Returns a dict with ``sampler``, ``resolution``, ``vae``, ``clip``,
    ``model_a``, ``model_b`` keys — or *None* if nothing was found.
    """
    out = {}

    # --- Helper: merge extracted_data dict into *out* ---
    def _apply_extracted(ed, source_label):
        s = ed.get('sampler')
        if s and isinstance(s, dict) and s.get('sampler_name'):
            out['sampler'] = {
                'steps':        s.get('steps', 20),
                'cfg':          s.get('cfg', 7.0),
                'seed':         s.get('seed', 0),
                'sampler_name': s.get('sampler_name', 'euler'),
                'scheduler':    s.get('scheduler', 'normal'),
                'denoise':      s.get('denoise', 1.0),
                'guidance':     s.get('guidance'),
            }
        r = ed.get('resolution')
        if r and isinstance(r, dict) and r.get('width'):
            out['resolution'] = {
                'width':      r.get('width', 512),
                'height':     r.get('height', 512),
                'batch_size': r.get('batch_size', 1),
                'length':     r.get('length'),
            }
        vae = ed.get('vae', '')
        if vae and isinstance(vae, str) and vae not in ('', '\u2014'):
            out['vae'] = {'name': vae, 'source': source_label}
        clip = ed.get('clip', [])
        if clip:
            names = clip if isinstance(clip, list) else [clip]
            if any(n for n in names):
                out['clip'] = {'names': names, 'type': '', 'source': source_label}
        if ed.get('model_a'):
            out.setdefault('model_a', ed['model_a'])
        if ed.get('model_b'):
            out.setdefault('model_b', ed['model_b'])

    # --- Helper: parse override_data JSON (flat keys) ---
    def _apply_override_json(ov_str, source_label):
        try:
            ov = json.loads(ov_str) if ov_str else {}
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(ov, dict):
            return
        if ov.get('width') or ov.get('steps'):
            if 'sampler' not in out and ov.get('sampler_name'):
                out['sampler'] = {
                    'steps':        ov.get('steps', 20),
                    'cfg':          ov.get('cfg', 7.0),
                    'seed':         ov.get('seed', 0),
                    'sampler_name': ov.get('sampler_name', 'euler'),
                    'scheduler':    ov.get('scheduler', 'normal'),
                    'denoise':      1.0,
                    'guidance':     None,
                }
            if 'resolution' not in out and ov.get('width'):
                out['resolution'] = {
                    'width':      ov.get('width', 512),
                    'height':     ov.get('height', 512),
                    'batch_size': ov.get('batch_size', 1),
                    'length':     ov.get('length'),
                }
            if ov.get('model_a'):
                out.setdefault('model_a', ov['model_a'])

    # --- 1. Workflow nodes (extracted_data + widgets_values) ---
    if workflow_data and isinstance(workflow_data, dict):
        for wf_node in workflow_data.get('nodes', []):
            ntype = wf_node.get('type', '')
            if ntype == 'WorkflowGenerator':
                # Priority 1: extracted_data
                ed = wf_node.get('extracted_data')
                if ed and isinstance(ed, dict):
                    _apply_extracted(ed, 'WorkflowGenerator')
                # Priority 2: override_data in widgets_values
                if 'sampler' not in out or 'resolution' not in out:
                    for v in (wf_node.get('widgets_values') or []):
                        if isinstance(v, str) and len(v) > 5 and v.strip().startswith('{'):
                            _apply_override_json(v, 'WorkflowGenerator')
                            if 'sampler' in out and 'resolution' in out:
                                break
                if 'sampler' in out or 'resolution' in out:
                    break  # WG found — done

        # Priority 4: PromptExtractor extracted_data (only if WG wasn't found)
        if 'sampler' not in out and 'resolution' not in out:
            for wf_node in workflow_data.get('nodes', []):
                if wf_node.get('type') == 'PromptExtractor':
                    ed = wf_node.get('extracted_data')
                    if ed and isinstance(ed, dict):
                        _apply_extracted(ed, 'PromptExtractor')
                        if 'sampler' in out or 'resolution' in out:
                            break

    # --- 3. Prompt API (WG override_data in inputs) ---
    if ('sampler' not in out or 'resolution' not in out) and prompt_data and isinstance(prompt_data, dict):
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            if node_data.get('class_type') == 'WorkflowGenerator':
                inp = node_data.get('inputs', {})
                _apply_override_json(inp.get('override_data', ''), 'WorkflowGenerator')
                break

    return out if out else None


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
    )
    from .lora_utils import resolve_lora_path

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

    # ── A1111 / Forge override ────────────────────────────────────────────
    # When prompt_data is a parsed A1111 dict (has 'prompt' + 'loras' keys)
    # the ComfyUI extraction functions won't find anything.  Apply the
    # values that parse_a1111_parameters() already extracted.
    if isinstance(prompt_data, dict) and 'prompt' in prompt_data and 'loras' in prompt_data:
        _s = result['sampler']
        if prompt_data.get('sampler_name'):
            _s['sampler_name'] = prompt_data['sampler_name']
        if prompt_data.get('scheduler'):
            _s['scheduler'] = prompt_data['scheduler']
        if prompt_data.get('steps'):
            _s['steps'] = prompt_data['steps']
        if prompt_data.get('cfg'):
            _s['cfg'] = prompt_data['cfg']
        if prompt_data.get('seed'):
            _s['seed'] = prompt_data['seed']
        if prompt_data.get('denoise') is not None:
            _s['denoise'] = prompt_data['denoise']
        _r = result['resolution']
        if prompt_data.get('width') and prompt_data.get('height'):
            _r['width'] = prompt_data['width']
            _r['height'] = prompt_data['height']

    # ── Embedded data override (WorkflowGenerator / PromptExtractor) ──────
    # When a WorkflowGenerator generated the image there are no standalone
    # KSampler / EmptyLatentImage nodes.  Look for authoritative values in:
    #   1. extracted_data on the WG or PE node
    #   2. widgets_values override_data JSON on the WG node
    # Also check prompt API for WG override_data.
    _embedded = _find_embedded_generation_data(workflow_data, prompt_data)
    if _embedded:
        for key in ('sampler', 'resolution'):
            if key in _embedded and isinstance(_embedded[key], dict):
                result[key] = _embedded[key]
        if _embedded.get('vae'):
            result['vae'] = _embedded['vae']
        if _embedded.get('clip'):
            result['clip'] = _embedded['clip']
        if not result['model_a'] and _embedded.get('model_a'):
            result['model_a'] = _embedded['model_a']
        if not result['model_b'] and _embedded.get('model_b'):
            result['model_b'] = _embedded['model_b']

    if result['resolution']['length'] is not None:
        result['is_video'] = True

    return result


def enrich_with_availability(result):
    """
    Add availability flags (lora_availability, model_a_found, etc.)
    to an extract_all_from_file() result dict in-place.
    Returns the same dict.
    """
    from .lora_utils import resolve_lora_path

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


def build_simplified_workflow_data(extracted, overrides=None, sampler_params=None):
    """
    Build the shared workflow_data dict that both WorkflowGenerator and PromptExtractor output.

    Parameters
    ----------
    extracted : dict  — result of extract_all_from_file() or an equivalent dict with the same keys
    overrides : dict  — optional JS-side override values (model_a, positive_prompt, …)
    sampler_params : dict — optional sampler values (already-merged steps/cfg/seed/…)

    Returns a serialisable dict (same schema for both nodes).
    """
    if overrides is None:
        overrides = {}
    sampler = sampler_params if sampler_params is not None else extracted.get('sampler', {
        'steps': 20, 'cfg': 7.0, 'seed': 0,
        'sampler_name': 'euler', 'scheduler': 'normal',
        'denoise': 1.0, 'guidance': None,
    })

    family      = extracted.get('model_family', '')
    family_strat = extracted.get('model_family_label', '')

    clip_info = extracted.get('clip', {})
    clip_source = clip_info.get('source', '')
    if clip_source == 'checkpoint':
        loader_type = 'checkpoint'
    elif clip_source in ('separate', 'workflow_data'):
        loader_type = 'unet'
    else:
        loader_type = ''

    return {
        "_version":        1,
        "_source":         overrides.get('_source', 'PromptExtractor'),
        "family":          family,
        "family_strategy": family_strat,
        "model_a":         overrides.get('model_a',   extracted.get('model_a', '')),
        "model_b":         overrides.get('model_b',   extracted.get('model_b', '')),
        "positive_prompt": overrides.get('positive_prompt', extracted.get('positive_prompt', '')),
        "negative_prompt": overrides.get('negative_prompt', extracted.get('negative_prompt', '')),
        "loras_a":         extracted.get('loras_a', []),
        "loras_b":         extracted.get('loras_b', []),
        "vae":             overrides.get('vae',  extracted.get('vae', {}).get('name', '')),
        "clip":            overrides.get('clip_names', extracted.get('clip', {}).get('names', [])),
        "clip_type":       clip_info.get('type', ''),
        "loader_type":     loader_type,
        "sampler":         sampler,
        "resolution": {
            "width":      overrides.get('width',      extracted.get('resolution', {}).get('width',      512)),
            "height":     overrides.get('height',     extracted.get('resolution', {}).get('height',     512)),
            "batch_size": overrides.get('batch_size', extracted.get('resolution', {}).get('batch_size', 1)),
            "length":     overrides.get('length',     extracted.get('resolution', {}).get('length',     None)),
            # Propagate node-ref flags so WorkflowGenerator knows to use
            # source_image dimensions rather than the stale template values.
            "_width_from_node_ref":  extracted.get('resolution', {}).get('_width_from_node_ref',  False),
            "_height_from_node_ref": extracted.get('resolution', {}).get('_height_from_node_ref', False),
        },
    }
