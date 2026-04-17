"""
Shared extraction helpers for ComfyUI Prompt Manager.

Used by both WorkflowExtractor and WorkflowBuilder / WorkflowRenderer nodes.
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

# Node types that carry embedded extracted_data.
# Keep 'WorkflowGenerator' for backward compat with images generated before rename.
# WorkflowBuilder stores authoritative extracted_data for Builder+Renderer graphs.
_EMBEDDED_NODE_TYPES = (
    'WorkflowRenderer',
    'WorkflowGenerator',
    'WorkflowBuilder',
    'PromptExtractor',
)


def _get_embedded_extracted_data(workflow_data):
    """Return the best ``extracted_data`` dict from a WR/WG or PE node, or *None*.

    Prioritises WorkflowRenderer/WorkflowGenerator over WorkflowBuilder over
    PromptExtractor, and only returns data that actually contains ``sampler``
    or ``resolution`` keys.
    """
    if not workflow_data or not isinstance(workflow_data, dict):
        return None

    best = None
    best_rank = -1

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
        if ntype in ('WorkflowRenderer', 'WorkflowGenerator'):
            rank = 3
        elif ntype == 'WorkflowBuilder':
            rank = 2
        else:
            rank = 1

        # Prefer higher-priority source; among same type, first wins.
        if best is None or rank > best_rank:
            best = ed
            best_rank = rank
        if best_rank == 3:
            break  # WR/WG is highest priority — stop early

    return best


# ─── Sampler extraction ───────────────────────────────────────────────────────

def extract_sampler_params(prompt_data, workflow_data):
    """
    Extract sampler parameters from KSampler nodes in API or workflow format.
    Returns a dict with steps_a, steps_b, cfg, seed_a, seed_b, sampler_name, scheduler.
    """
    params = {
        'steps_a': 20,
        'steps_b': None,   # WAN Video low-pass steps (None = same as steps_a)
        'cfg': 7.0,
        'seed_a': 0,
        'seed_b': None,   # WAN Video second-sampler seed (None = use same as seed_a)
        'sampler_name': 'euler',
        'scheduler': 'normal',
    }

    # ── API format ────────────────────────────────────────────────────────────
    if prompt_data and isinstance(prompt_data, dict):
        # WAN Video uses two KSamplerAdvanced nodes (high + low pass).
        # Collect all of them so we can expose both seeds.
        ksampler_advanced_nodes = []
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            ct = node_data.get('class_type', '')
            inp = node_data.get('inputs', {})

            if ct == 'KSampler':
                params['steps_a']      = inp.get('steps', params['steps_a'])
                params['cfg']          = inp.get('cfg', params['cfg'])
                seed_val               = inp.get('seed', inp.get('noise_seed', params['seed_a']))
                params['seed_a']       = seed_val if not isinstance(seed_val, list) else params['seed_a']
                params['sampler_name'] = inp.get('sampler_name', params['sampler_name'])
                params['scheduler']    = inp.get('scheduler', params['scheduler'])
                return params  # standard KSampler found — done

            if ct == 'KSamplerAdvanced':
                ksampler_advanced_nodes.append(inp)

            if ct == 'WanMoeKSamplerAdvanced':
                params['steps_a']      = inp.get('steps', params['steps_a'])
                params['cfg']          = inp.get('cfg', params['cfg'])
                seed_val               = inp.get('seed', inp.get('noise_seed', params['seed_a']))
                params['seed_a']       = seed_val if not isinstance(seed_val, list) else params['seed_a']
                params['sampler_name'] = inp.get('sampler_name', params['sampler_name'])
                params['scheduler']    = inp.get('scheduler', params['scheduler'])
                return params

        # Handle KSamplerAdvanced — including WAN Video dual-sampler pattern
        if ksampler_advanced_nodes:
            first = ksampler_advanced_nodes[0]
            params['steps_a']      = first.get('steps', params['steps_a'])
            params['cfg']          = first.get('cfg', params['cfg'])
            seed_val               = first.get('seed', first.get('noise_seed', params['seed_a']))
            params['seed_a']       = seed_val if not isinstance(seed_val, list) else params['seed_a']
            params['sampler_name'] = first.get('sampler_name', params['sampler_name'])
            params['scheduler']    = first.get('scheduler', params['scheduler'])
            if len(ksampler_advanced_nodes) >= 2:
                # Second node = low-pass sampler — expose its seed as seed_b
                second   = ksampler_advanced_nodes[1]
                seed_b   = second.get('seed', second.get('noise_seed', params['seed_a']))
                params['seed_b'] = seed_b if not isinstance(seed_b, list) else params['seed_a']
                # Extract steps_b if the second sampler has different steps
                steps_b_val = second.get('steps')
                if steps_b_val is not None and not isinstance(steps_b_val, list):
                    params['steps_b'] = int(steps_b_val)
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
                params['steps_a']   = inp.get('steps', params['steps_a'])
                params['scheduler'] = inp.get('scheduler', params['scheduler'])
            elif ct == 'CFGGuider':
                params['cfg'] = inp.get('cfg', params['cfg'])
            elif ct == 'RandomNoise':
                params['seed_a'] = inp.get('noise_seed', params['seed_a'])

    # ── Workflow (node graph) format fallback ─────────────────────────────────
    if workflow_data and isinstance(workflow_data, dict):
        from .workflow_node_utils import build_node_map
        node_map = build_node_map(workflow_data)
        for node_id, node in node_map.items():
            ntype   = node.get('type', '')
            widgets = node.get('widgets_values', [])

            if ntype == 'KSampler' and len(widgets) >= 6:
                try:
                    params['seed_a']       = int(widgets[0])   if widgets[0] is not None else 0
                    params['steps_a']      = int(widgets[2])   if widgets[2] is not None else 20
                    params['cfg']          = float(widgets[3]) if widgets[3] is not None else 7.0
                    params['sampler_name'] = str(widgets[4])   if widgets[4] else 'euler'
                    params['scheduler']    = str(widgets[5])   if widgets[5] else 'normal'
                    return params
                except (ValueError, IndexError):
                    pass

            elif ntype == 'KSamplerSelect' and widgets:
                params['sampler_name'] = str(widgets[0]) if widgets[0] else params['sampler_name']
            elif ntype in ('BasicScheduler', 'Flux2Scheduler') and len(widgets) >= 2:
                try:
                    params['scheduler'] = str(widgets[0]) if widgets[0] else params['scheduler']
                    params['steps_a']   = int(widgets[1]) if widgets[1] is not None else params['steps_a']
                except (ValueError, IndexError):
                    pass
            elif ntype == 'CFGGuider' and widgets:
                try:
                    params['cfg'] = float(widgets[0]) if widgets[0] is not None else params['cfg']
                except (ValueError, IndexError):
                    pass
            elif ntype == 'RandomNoise' and widgets:
                try:
                    params['seed_a'] = int(widgets[0]) if widgets[0] is not None else 0
                except (ValueError, IndexError):
                    pass

    # ── Fallback: embedded extracted_data from WG / PE nodes ──────────────
    ed = _get_embedded_extracted_data(workflow_data)
    if ed:
        s = ed.get('sampler')
        if s and isinstance(s, dict) and s.get('sampler_name'):
            params['steps_a']      = s.get('steps_a', s.get('steps', params['steps_a']))
            params['steps_b']      = s.get('steps_b', params['steps_b'])
            params['cfg']          = s.get('cfg', params['cfg'])
            params['seed_a']       = s.get('seed_a', s.get('seed', params['seed_a']))
            params['sampler_name'] = s.get('sampler_name', params['sampler_name'])
            params['scheduler']    = s.get('scheduler', params['scheduler'])

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
                resolution['width']      = _scalar(w, resolution['width'],  'width')
                resolution['height']     = _scalar(h, resolution['height'], 'height')
                if ct in VIDEO_LATENT_TYPES:
                    # For video latent nodes (WanImageToVideo, etc.), batch_size
                    # is the frame count — store it as length, keep batch_size=1.
                    b_val = _scalar(b, resolution['batch_size'])
                    resolution['batch_size'] = 1
                    if 'length' in inp:
                        l = inp.get('length')
                        resolution['length'] = _scalar(l, None) if l is not None else None
                    elif b_val > 1:
                        # batch_size field holds frame count
                        resolution['length'] = b_val
                else:
                    b_val = _scalar(b, resolution['batch_size'])
                    resolution['batch_size'] = b_val
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
    Look for sampler / resolution / model data embedded by WorkflowRenderer
    (or legacy WorkflowGenerator) or PromptExtractor.  Checks (in priority order):

    1. WR/WG/WB node ``extracted_data`` (full dict with sampler + resolution)
    2. WR/WG/WB node ``widgets_values`` containing the ``override_data`` JSON
    3. WR/WG/WB ``override_data`` in the prompt API inputs
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
                'steps_a':      s.get('steps_a', s.get('steps', 20)),
                'steps_b':      s.get('steps_b'),
                'cfg':          s.get('cfg', 7.0),
                'seed_a':       s.get('seed_a', s.get('seed', 0)),
                'seed_b':       s.get('seed_b'),
                'sampler_name': s.get('sampler_name', 'euler'),
                'scheduler':    s.get('scheduler', 'normal'),
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
        if ov.get('width') or ov.get('steps_a') or ov.get('steps'):
            if 'sampler' not in out and ov.get('sampler_name'):
                out['sampler'] = {
                    'steps_a':      ov.get('steps_a', ov.get('steps', 20)),
                    'steps_b':      ov.get('steps_b'),
                    'cfg':          ov.get('cfg', 7.0),
                    'seed_a':       ov.get('seed_a', ov.get('seed', 0)),
                    'seed_b':       ov.get('seed_b'),
                    'sampler_name': ov.get('sampler_name', 'euler'),
                    'scheduler':    ov.get('scheduler', 'normal'),
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
            if ntype in ('WorkflowRenderer', 'WorkflowGenerator', 'WorkflowBuilder'):
                # Priority 1: extracted_data
                ed = wf_node.get('extracted_data')
                if ed and isinstance(ed, dict):
                    _apply_extracted(ed, ntype)
                # Priority 2: override_data in widgets_values
                if 'sampler' not in out or 'resolution' not in out:
                    for v in (wf_node.get('widgets_values') or []):
                        if isinstance(v, str) and len(v) > 5 and v.strip().startswith('{'):
                            _apply_override_json(v, ntype)
                            if 'sampler' in out and 'resolution' in out:
                                break
                if 'sampler' in out or 'resolution' in out:
                    break  # WR/WG/WB found — done

        # Priority 4: PromptExtractor extracted_data (only if WG wasn't found)
        if 'sampler' not in out and 'resolution' not in out:
            for wf_node in workflow_data.get('nodes', []):
                if wf_node.get('type') == 'PromptExtractor':
                    ed = wf_node.get('extracted_data')
                    if ed and isinstance(ed, dict):
                        _apply_extracted(ed, 'PromptExtractor')
                        if 'sampler' in out or 'resolution' in out:
                            break

    # --- 3. Prompt API (WR/WG override_data in inputs) ---
    if ('sampler' not in out or 'resolution' not in out) and prompt_data and isinstance(prompt_data, dict):
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            if node_data.get('class_type') in ('WorkflowRenderer', 'WorkflowGenerator', 'WorkflowBuilder'):
                inp = node_data.get('inputs', {})
                _apply_override_json(inp.get('override_data', ''), node_data['class_type'])
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
      sampler {'steps_a', 'steps_b', 'cfg', 'seed_a', 'seed_b', 'sampler_name', 'scheduler'},
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
            'steps_a': 20, 'steps_b': None, 'cfg': 7.0, 'seed_a': 0, 'seed_b': None,
            'sampler_name': 'euler', 'scheduler': 'normal',
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
            _s['steps_a'] = prompt_data['steps']
        if prompt_data.get('cfg'):
            _s['cfg'] = prompt_data['cfg']
        if prompt_data.get('seed'):
            _s['seed_a'] = prompt_data['seed']
        _r = result['resolution']
        if prompt_data.get('width') and prompt_data.get('height'):
            _r['width'] = prompt_data['width']
            _r['height'] = prompt_data['height']

    # ── Embedded data override (WorkflowRenderer / PromptExtractor) ──────
    # When a WorkflowRenderer generated the image there are no standalone
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


def _build_sampler_dict(sampler, family):
    """Return sampler dict with steps_a/steps_b for all families."""
    import math as _math
    d = {**sampler}
    # Normalize: if only a plain 'seed' key exists, map it to seed_a
    if 'seed' in d:
        d.setdefault('seed_a', d.pop('seed'))
    d.setdefault('seed_a', 0)
    d.setdefault('seed_b', sampler.get('seed_b'))
    # Normalize legacy 'steps' key → steps_a
    if 'steps' in d and 'steps_a' not in d:
        d['steps_a'] = d.pop('steps')
    elif 'steps' in d:
        d.pop('steps')
    d.setdefault('steps_a', 20)
    if family in ('wan_video_i2v', 'wan_video_t2v'):
        # For WAN dual-sampler: if steps_b not set, split steps_a evenly
        if d.get('steps_b') is None:
            total = d['steps_a']
            d['steps_a'] = _math.ceil(total / 2)
            d['steps_b'] = total - d['steps_a']
    # Remove legacy keys
    d.pop('steps_high', None)
    d.pop('steps_low', None)
    return d


def build_simplified_workflow_data(extracted, overrides=None, sampler_params=None):
    """
    Build the shared workflow_data dict that both WorkflowBuilder and PromptExtractor output.

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
        'steps_a': 20, 'steps_b': None, 'cfg': 7.0, 'seed_a': 0, 'seed_b': None,
        'sampler_name': 'euler', 'scheduler': 'normal',
    })

    family      = extracted.get('model_family', '')

    # If family is unknown, infer from clip type — each family requires a
    # specific CLIPLoader type, so this is a reliable signal.
    clip_info = extracted.get('clip', {})
    if not family:
        _clip_type = clip_info.get('type', '').lower()
        _clip_src  = clip_info.get('source', '')
        if _clip_src == 'checkpoint':
            family = 'sdxl'
        elif 'flux2' in _clip_type:
            family = 'flux2'
        elif 'flux' in _clip_type:
            family = 'flux1'
        elif 'sd3' in _clip_type:
            family = 'sd3'
        elif 'wan' in _clip_type:
            family = 'wan_video_t2v'
        elif 'qwen_image' in _clip_type:
            family = 'qwen_image'
        elif 'lumina2' in _clip_type:
            family = 'zimage'

    # Determine loader_type: family spec is authoritative when available.
    # The clip source heuristic ('checkpoint' vs 'separate') is unreliable
    # when data passes through multiple nodes (e.g. PE → WB → Renderer)
    # because intermediate steps may set source to 'workflow_data'.
    from .workflow_families import MODEL_FAMILIES
    family_spec = MODEL_FAMILIES.get(family, {})
    if family_spec:
        is_ckpt = family_spec.get('checkpoint', False)
        loader_type = 'checkpoint' if is_ckpt else 'unet'
    else:
        # No family spec — fall back to clip source heuristic
        clip_source = clip_info.get('source', '')
        if clip_source == 'checkpoint':
            loader_type = 'checkpoint'
        elif clip_source in ('separate', 'workflow_data'):
            loader_type = 'unet'
        else:
            loader_type = ''

    vae_override_explicit  = 'vae' in overrides
    clip_override_explicit = 'clip_names' in overrides

    vae_val  = overrides.get('vae',  extracted.get('vae', {}).get('name', ''))
    clip_val = overrides.get('clip_names', extracted.get('clip', {}).get('names', []))

    # For checkpoint models: if the user didn't explicitly choose a separate
    # VAE / CLIP file, the checkpoint's own VAE / CLIP should be used.
    # The extracted values may contain real filenames from the source workflow
    # (e.g. clip_g.safetensors) but those came from a DualCLIPLoader in the
    # original graph — they shouldn't override the checkpoint's built-in ones.
    if loader_type == 'checkpoint':
        if not vae_override_explicit or not vae_val or vae_val.startswith('('):
            vae_val = '(from checkpoint)'
        if not clip_override_explicit or not clip_val or clip_val == [] or \
                (len(clip_val) == 1 and (not clip_val[0] or clip_val[0].startswith('('))):
            clip_val = ['(from checkpoint)']

    return {
        "_source":         overrides.get('_source', 'PromptExtractor'),
        "family":          family,
        "model_a":         overrides.get('model_a',   extracted.get('model_a', '')),
        "model_b":         overrides.get('model_b',   extracted.get('model_b', '')),
        "positive_prompt": overrides.get('positive_prompt', extracted.get('positive_prompt', '')),
        "negative_prompt": overrides.get('negative_prompt', extracted.get('negative_prompt', '')),
        "loras_a":         extracted.get('loras_a', []),
        "loras_b":         extracted.get('loras_b', []),
        "vae":             vae_val,
        "clip":            clip_val,
        "clip_type":       clip_info.get('type', '') or family_spec.get('clip_type', ''),
        "loader_type":     loader_type,
        "sampler":         _build_sampler_dict(sampler, family),
        "resolution": {
            "width":      overrides.get('width',      extracted.get('resolution', {}).get('width',      512)),
            "height":     overrides.get('height',     extracted.get('resolution', {}).get('height',     512)),
            "batch_size": overrides.get('batch_size', extracted.get('resolution', {}).get('batch_size', 1)),
            "length":     overrides.get('length',     extracted.get('resolution', {}).get('length',     None)),
        },
    }
