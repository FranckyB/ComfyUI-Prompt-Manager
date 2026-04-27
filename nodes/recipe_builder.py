"""
ComfyUI Workflow Builder
Extracts ALL generation parameters from an image/video, provides a full UI
for editing them, and outputs RECIPE_DATA (JSON) for the Workflow Renderer
render node.

Part of ComfyUI-Prompt-Manager — shares extraction logic with PromptExtractor.
Family system and extraction helpers live in py/ for reuse.
"""
import os
import json
import copy
import traceback
import numpy as np
import torch
from PIL import Image as PILImage, ImageOps
import folder_paths
import comfy.sd
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.model_management
import server

# ── Shared helpers from py/ ──────────────────────────────────────────────────
from ..py.workflow_families import (
    MODEL_FAMILIES,
    get_model_family,
    get_family_label,
    get_family_sampler_strategy,
    get_compatible_families,
    get_all_family_labels,
    list_compatible_models,
    list_compatible_vaes,
    list_compatible_clips,
)
from ..py.workflow_extraction_utils import (
    extract_sampler_params,
    extract_vae_info,
    extract_clip_info,
    extract_resolution,
    resolve_model_name,
    resolve_vae_name,
    resolve_clip_names,
    build_simplified_workflow_data,
)
from ..py.workflow_data_utils import strip_runtime_objects, to_json_safe_workflow_data

# ── Shared extraction functions from PromptExtractor ────────────────────────
from .prompt_extractor import (
    parse_workflow_for_prompts,
    extract_metadata_from_png,
    extract_metadata_from_jpeg,
    extract_metadata_from_json,
    extract_metadata_from_video,
    build_node_map,
    build_link_map,
    extract_video_frame_av_to_tensor,
    get_cached_video_frame,
    get_placeholder_image_tensor,
)
from ..py.lora_utils import resolve_lora_path, strip_lora_extension

# ── Sampler/scheduler lists ──────────────────────────────────────────────────
SAMPLERS   = comfy.samplers.KSampler.SAMPLERS
SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS

# Cache for last extracted info per RecipeBuilder node (keyed by unique_id).
# Used by another RecipeBuilder's "Update Workflow" button for live pull.
_last_workflow_builder_info = {}
_MODEL_KEYS = ("model_a", "model_b", "model_c", "model_d")


def _normalize_model_slot(slot, default="model_a"):
    key = str(slot or default).strip().lower()
    return key if key in _MODEL_KEYS else default


def _v2_get_model_block(recipe_data, model_key):
    if not isinstance(recipe_data, dict):
        return None
    if int(recipe_data.get("version", 0) or 0) < 2:
        return None
    models = recipe_data.get("models", {}) if isinstance(recipe_data.get("models"), dict) else {}
    block = models.get(model_key)
    return block if isinstance(block, dict) else None


def _builder_output_to_v2(legacy_wf, base_recipe_data=None, overwrite_existing_slots=True):
    """Convert Builder legacy workflow_data shape to v2 models shape."""
    wf = legacy_wf if isinstance(legacy_wf, dict) else {}
    sampler = wf.get("sampler", {}) if isinstance(wf.get("sampler"), dict) else {}
    resolution = wf.get("resolution", {}) if isinstance(wf.get("resolution"), dict) else {}

    def _make_model_block(slot_key):
        is_b = slot_key == "model_b"
        steps = sampler.get("steps_b") if is_b and sampler.get("steps_b") is not None else sampler.get("steps_a", sampler.get("steps", 20))
        seed = sampler.get("seed_b") if is_b and sampler.get("seed_b") is not None else sampler.get("seed_a", sampler.get("seed", 0))
        loras_key = "loras_b" if is_b else "loras_a"
        model_key = "model_b" if is_b else "model_a"

        return {
            "positive_prompt": wf.get("positive_prompt", ""),
            "negative_prompt": wf.get("negative_prompt", ""),
            "family": wf.get("family", "") or "sdxl",
            "model": wf.get(model_key, ""),
            "loras": wf.get(loras_key, []) if isinstance(wf.get(loras_key), list) else [],
            "clip_type": wf.get("clip_type", ""),
            "loader_type": wf.get("loader_type", ""),
            "vae": wf.get("vae", ""),
            "clip": wf.get("clip", []) if isinstance(wf.get("clip"), list) else ([wf.get("clip")] if wf.get("clip") else []),
            "sampler": {
                "steps": steps,
                "cfg": sampler.get("cfg", 5.0),
                "denoise": sampler.get("denoise", 1.0),
                "seed": seed,
                "sampler_name": sampler.get("sampler_name", "euler"),
                "scheduler": sampler.get("scheduler", "simple"),
            },
            "resolution": {
                "width": resolution.get("width", 768),
                "height": resolution.get("height", 1280),
                "batch_size": resolution.get("batch_size", 1),
                "length": resolution.get("length", None),
            },
        }

    out = {
        "_source": wf.get("_source", "RecipeBuilder"),
        "version": 2,
        "models": {},
    }

    def _safe_deepcopy(value):
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    if isinstance(base_recipe_data, dict) and int(base_recipe_data.get("version", 0) or 0) >= 2 and isinstance(base_recipe_data.get("models"), dict):
        out["_source"] = str(base_recipe_data.get("_source") or out["_source"])
        for key, val in base_recipe_data.items():
            if key in ("_source", "version", "models"):
                continue
            out[key] = _safe_deepcopy(val)
        for key, block in (base_recipe_data.get("models") or {}).items():
            if isinstance(block, dict):
                # Preserve runtime objects without deep-copying Comfy internals
                # (e.g. ModelPatcher), which can raise during deepcopy.
                out["models"][key] = dict(block)

    primary_slot = "model_a"
    has_secondary_block = any([
        str(wf.get("model_b", "")).strip() != "",
        isinstance(wf.get("loras_b"), list) and len(wf.get("loras_b") or []) > 0,
        isinstance(sampler.get("steps_b"), (int, float)),
        isinstance(sampler.get("seed_b"), (int, float)),
    ])
    secondary_slot = "model_b" if has_secondary_block else None

    if overwrite_existing_slots or primary_slot not in out["models"]:
        out["models"][primary_slot] = _make_model_block("model_a")
    if secondary_slot and (overwrite_existing_slots or secondary_slot not in out["models"]):
        out["models"][secondary_slot] = _make_model_block("model_b")

    for key in ("LATENT", "IMAGE", "MASK", "EXTRA"):
        if key in wf:
            out[key] = wf.get(key)

    return out


# ─── API endpoints ──────────────────────────────────────────────────────────

@server.PromptServer.instance.routes.get("/workflow-extractor/list-models")
async def api_list_models(request):
    """List compatible models. Pass ?ref=<model> or ?family=<key>."""
    try:
        family_key = request.rel_url.query.get('family', '')
        ref        = request.rel_url.query.get('ref', '')
        show_all_q = request.rel_url.query.get('show_all', '').strip().lower()
        show_all   = show_all_q in ('1', 'true', 'yes', 'on')

        def _gather_models(folder_names):
            gathered = []
            seen = set()
            for fn in folder_names:
                try:
                    for m in folder_paths.get_filename_list(fn):
                        if m not in seen:
                            seen.add(m)
                            gathered.append(m)
                except Exception:
                    continue
            return gathered

        if family_key:
            compat = get_compatible_families(family_key)
            compat_specs = [MODEL_FAMILIES.get(f, {}) for f in compat]

            preferred_folders = []
            for spec in compat_specs:
                for folder_name in spec.get('model_folders', []) or []:
                    if folder_name and folder_name not in preferred_folders:
                        preferred_folders.append(folder_name)

            # Policy: only SDXL should source from checkpoints.
            # All other families should stay on diffusion/unet folders.
            if preferred_folders:
                model_folders = preferred_folders
            elif family_key == 'sdxl':
                model_folders = ['checkpoints']
            else:
                model_folders = ['diffusion_models', 'unet', 'unet_gguf']
            all_models = _gather_models(model_folders)

            if show_all:
                models = list(all_models)
            else:
                models = [m for m in all_models if get_model_family(m) in compat]

                # Fallbacks for installations with inconsistent folder taxonomy:
                # 1) Checkpoint families: if no match, show all checkpoints.
                # 2) Diffusion/UNet families: if no match, do relaxed name/path matching.
                if not models:
                    compat_has_checkpoint = any(spec.get('checkpoint', False) for spec in compat_specs if spec)

                    if compat_has_checkpoint and family_key == 'sdxl':
                        models = _gather_models(['checkpoints'])
                    else:
                        folder_patterns = []
                        name_patterns = []
                        for fam in compat:
                            spec = MODEL_FAMILIES.get(fam, {})
                            folder_patterns.extend([
                                p.lower().replace('\\', '/').strip('/')
                                for p in spec.get('folders', []) if p
                            ])
                            name_patterns.extend([p.lower() for p in spec.get('names', []) if p])

                        relaxed = []
                        for m in all_models:
                            m_lower = m.lower().replace('\\', '/')
                            m_name = os.path.basename(m_lower)
                            folder_hit = any(fp and fp in m_lower for fp in folder_patterns)
                            name_hit = any(pat and (pat in m_name or pat in m_lower) for pat in name_patterns)
                            if folder_hit or name_hit:
                                relaxed.append(m)
                        models = relaxed

                # WAN i2v can use i2v or t2v model files depending on naming.
                # Use lowercase full-path checks to avoid case-sensitivity misses.
                if family_key == 'wan_video_i2v' and models:
                    i2v_t2v = []
                    for m in models:
                        m_lower = m.lower().replace('\\', '/')
                        if ('i2v' in m_lower) or ('t2v' in m_lower):
                            i2v_t2v.append(m)
                    if i2v_t2v:
                        models = i2v_t2v
            family = family_key
        elif ref:
            family = get_model_family(ref) or 'sdxl'
            models = list_compatible_models(ref, family_override=family)
        else:
            model_type = request.rel_url.query.get('type', 'checkpoints')
            try:
                models = sorted(folder_paths.get_filename_list(model_type))
            except Exception:
                models = []
            family = None

        return server.web.json_response({
            "models":       sorted(models),
            "family":       family,
            "family_label": get_family_label(family),
        })
    except Exception as e:
        return server.web.json_response({"models": [], "error": str(e)})


@server.PromptServer.instance.routes.get("/workflow-extractor/list-vaes")
async def api_list_vaes(request):
    """List compatible VAEs. Pass ?family=<key> to filter.
    Returns {vaes: [...], recommended: str|null} — recommended is the best match on disk.
    """
    try:
        family = request.rel_url.query.get('family', '') or None
        vaes, recommended = list_compatible_vaes(family, return_recommended=True)
        return server.web.json_response({"vaes": vaes, "recommended": recommended})
    except Exception as e:
        return server.web.json_response({"vaes": [], "recommended": None, "error": str(e)})


@server.PromptServer.instance.routes.get("/workflow-extractor/list-clips")
async def api_list_clips(request):
    """List compatible CLIPs. Pass ?family=<key> to filter.
    Returns {clips: [...], recommended: str|null} — recommended is the best match on disk.
    """
    try:
        family = request.rel_url.query.get('family', '') or None
        clips, recommended = list_compatible_clips(family, return_recommended=True)
        return server.web.json_response({"clips": clips, "recommended": recommended})
    except Exception as e:
        return server.web.json_response({"clips": [], "recommended": None, "error": str(e)})


@server.PromptServer.instance.routes.get("/workflow-extractor/video-frame")
async def api_video_frame(request):
    """Extract first frame of a video file and return as PNG. Used by JS thumbnail fallback."""
    try:
        filename = request.rel_url.query.get('filename', '')
        source   = request.rel_url.query.get('source', 'input')
        position = float(request.rel_url.query.get('position', '0'))

        if not filename:
            return server.web.Response(status=400)

        base_dir  = folder_paths.get_output_directory() if source == 'output' else folder_paths.get_input_directory()
        file_path = os.path.join(base_dir, filename.replace('/', os.sep))

        real_base = os.path.realpath(base_dir)
        real_path = os.path.realpath(file_path)
        if not real_path.startswith(real_base) or not os.path.exists(file_path):
            return server.web.Response(status=404)

        # Try PyAV first
        try:
            import av
            import io
            container = av.open(file_path)
            video_stream = next((s for s in container.streams if s.type == 'video'), None)
            if video_stream is None:
                return server.web.Response(status=404)

            target_time = None
            if position > 0 and video_stream.duration:
                target_time = int(position * video_stream.duration)
                container.seek(target_time, stream=video_stream)

            frame = None
            for packet in container.demux(video_stream):
                for f in packet.decode():
                    frame = f
                    break
                if frame is not None:
                    break
            container.close()

            if frame is None:
                return server.web.Response(status=404)

            img = frame.to_image()
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            return server.web.Response(body=buf.read(), content_type='image/png')

        except ImportError:
            pass  # PyAV not available — try ffmpeg subprocess

        # Fallback: ffmpeg subprocess
        import subprocess, io, tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.close()
        try:
            cmd = [
                'ffmpeg', '-y', '-i', file_path,
                '-vframes', '1',
                '-f', 'image2', tmp.name,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode == 0 and os.path.exists(tmp.name):
                with open(tmp.name, 'rb') as f:
                    data = f.read()
                return server.web.Response(body=data, content_type='image/png')
        except Exception:
            pass
        finally:
            try:
                os.unlink(tmp.name)
            except:
                pass

        return server.web.Response(status=500)
    except Exception as e:
        print(f"[RecipeBuilder] video-frame API error: {e}")
        return server.web.Response(status=500)


@server.PromptServer.instance.routes.get("/workflow-extractor/list-families")
async def api_list_families(request):
    """List all known model families for the type selector."""
    try:
        families = get_all_family_labels() or {}
        families.pop("ltxv", None)
        return server.web.json_response({"families": families})
    except Exception as e:
        return server.web.json_response({"families": {}, "error": str(e)})


@server.PromptServer.instance.routes.get("/workflow-extractor/list-files")
async def api_list_files(request):
    """List supported media files from input or output directory (recursive).
    Used by JS to refresh the image dropdown when source_folder changes.
    """
    try:
        source = request.rel_url.query.get('source', 'input')
        base_dir = folder_paths.get_output_directory() if source == 'output' else folder_paths.get_input_directory()
        supported = {'.png', '.jpg', '.jpeg', '.webp', '.json', '.mp4', '.webm', '.mov', '.avi'}
        files = []
        if os.path.exists(base_dir):
            for root, dirs, filenames in os.walk(base_dir):
                for fn in filenames:
                    if os.path.splitext(fn)[1].lower() in supported:
                        rel = os.path.relpath(os.path.join(root, fn), base_dir).replace('\\', '/')
                        files.append(rel)
        files.sort()
        return server.web.json_response({"files": files})
    except Exception as e:
        return server.web.json_response({"files": [], "error": str(e)})


@server.PromptServer.instance.routes.post("/workflow-builder/process-extracted")
async def api_process_extracted(request):
    """Process raw extracted data through WB's normalization logic.
    Matches execute() behavior: family resolution, sampler normalization,
    VAE/CLIP defaults, availability checking.
    Used by the 'Update Workflow' button."""
    try:
        body = await request.json()
        raw = body.get('extracted', {})
        family_override = body.get('family_override', None)
        if isinstance(family_override, str):
            family_override = family_override.strip() or None
        else:
            family_override = None
        if not raw:
            return server.web.json_response({"error": "No extracted data"}, status=400)

        # ── Normalize sampler: preserve ALL keys, ensure seed_a exists ──
        raw_sampler = raw.get('sampler', {})
        sampler = {
            'steps_a': raw_sampler.get('steps_a', raw_sampler.get('steps', 20)),
            'steps_b': raw_sampler.get('steps_b'),
            'cfg': raw_sampler.get('cfg', 5.0),
            'denoise': raw_sampler.get('denoise', 1.0),
            'seed_a': raw_sampler.get('seed_a', raw_sampler.get('seed', 0)),
            'seed_b': raw_sampler.get('seed_b'),
            'sampler_name': raw_sampler.get('sampler_name', 'euler'),
            'scheduler': raw_sampler.get('scheduler', 'simple'),
        }

        # ── Normalize VAE ───────────────────────────────────────────────
        raw_vae = raw.get('vae', {})
        if isinstance(raw_vae, dict):
            vae = {'name': raw_vae.get('name', ''), 'source': raw_vae.get('source', 'workflow_data')}
        elif isinstance(raw_vae, str):
            vae = {'name': raw_vae, 'source': 'workflow_data'}
        else:
            vae = {'name': '', 'source': 'unknown'}

        # ── Normalize CLIP ──────────────────────────────────────────────
        raw_clip = raw.get('clip', {})
        if isinstance(raw_clip, dict):
            clip = {
                'names': raw_clip.get('names', []),
                'type': raw_clip.get('type', ''),
                'source': raw_clip.get('source', 'workflow_data'),
            }
        elif isinstance(raw_clip, list):
            clip = {'names': raw_clip, 'type': '', 'source': 'workflow_data'}
        elif isinstance(raw_clip, str):
            clip = {'names': [raw_clip] if raw_clip else [], 'type': '', 'source': 'workflow_data'}
        else:
            clip = {'names': [], 'type': '', 'source': 'unknown'}

        # ── Normalize resolution ────────────────────────────────────────
        raw_res = raw.get('resolution', {})
        resolution = {
            'width': raw_res.get('width', 768),
            'height': raw_res.get('height', 1280),
            'batch_size': raw_res.get('batch_size', 1),
            'length': raw_res.get('length'),
        }

        extracted = {
            'positive_prompt': raw.get('positive_prompt', ''),
            'negative_prompt': raw.get('negative_prompt', ''),
            'loras_a': raw.get('loras_a', []),
            'loras_b': raw.get('loras_b', []),
            'model_a': raw.get('model_a', ''),
            'model_b': raw.get('model_b', ''),
            'vae': vae,
            'clip': clip,
            'sampler': sampler,
            'resolution': resolution,
            'is_video': raw.get('is_video', resolution.get('length') is not None),
            'model_family': raw.get('model_family', ''),
            'model_family_label': raw.get('model_family_label', ''),
        }

        model_name_a = extracted['model_a']
        model_name_b = extracted['model_b']

        # ── Resolve family ──────────────────────────────────────────────
        family_key = family_override or extracted.get('model_family') or None
        if not family_key and model_name_a:
            resolved_ref, _ = resolve_model_name(model_name_a)
            family_key = get_model_family(resolved_ref or model_name_a)
        if not family_key:
            family_key = 'sdxl'

        extracted['model_family'] = family_key
        extracted['model_family_label'] = get_family_label(family_key)

        from ..py.workflow_families import MODEL_FAMILIES
        spec = MODEL_FAMILIES.get(family_key, {})

        # ── Default VAE/CLIP fallback ───────────────────────────────────
        # For checkpoint models, empty means "use from checkpoint".
        # For non-checkpoint (unet) models, resolve to a compatible file.
        is_ckpt = spec.get('checkpoint', False)
        vae_name = vae.get('name', '')
        if is_ckpt:
            # Checkpoint models default to built-in VAE, but respect an
            # explicit separate VAE selection from the incoming data.
            if not vae_name or str(vae_name).startswith('('):
                vae = {'name': '(Default)', 'source': 'checkpoint'}
                extracted['vae'] = vae
            else:
                vae = {'name': vae_name, 'source': vae.get('source', 'separate')}
                extracted['vae'] = vae
        elif not vae_name or vae_name.startswith('('):
            vaes, rec_vae = list_compatible_vaes(family_key, return_recommended=True)
            fallback_vae = rec_vae or (vaes[0] if vaes else '')
            if fallback_vae:
                vae = {'name': fallback_vae, 'source': 'default'}
                extracted['vae'] = vae

        # ── Default CLIP fallback ───────────────────────────────────────
        clip_names = clip.get('names', [])
        clip_is_placeholder = bool(clip_names) and all(
            (not n) or str(n).startswith('(') for n in clip_names
        )
        if is_ckpt:
            # Checkpoint models default to built-in CLIP, but respect an
            # explicit separate CLIP selection from the incoming data.
            if not clip_names or clip_is_placeholder:
                clip = {'names': ['(Default)'], 'type': clip.get('type', ''), 'source': 'checkpoint'}
            else:
                clip = {
                    'names': clip_names,
                    'type': clip.get('type', ''),
                    'source': clip.get('source', 'separate'),
                }
            extracted['clip'] = clip
        elif not clip_names or clip_is_placeholder:
            clip_type_from_spec = spec.get('clip_type', '')
            compatible_clips, rec_clip = list_compatible_clips(family_key, return_recommended=True)
            if compatible_clips:
                clip_slots = spec.get('clip_slots', 1)
                clip_patterns = [p.lower() for p in spec.get('clip', [])]
                selected = []
                if clip_patterns and clip_slots >= 2:
                    for pat in clip_patterns:
                        for c in compatible_clips:
                            if pat in os.path.basename(c).lower() and c not in selected:
                                selected.append(c)
                                break
                        if len(selected) >= clip_slots:
                            break
                if not selected:
                    selected = [compatible_clips[0]]
                clip = {'names': selected, 'type': clip_type_from_spec, 'source': 'default'}
                extracted['clip'] = clip

        # ── Ensure clip_type and loader_type from family spec ───────────
        clip_type = clip.get('type', '') or spec.get('clip_type', '')
        loader_type = 'checkpoint' if is_ckpt else 'unet'

        # ── Availability checks ─────────────────────────────────────────
        model_a_found = True
        model_b_found = True
        if model_name_a:
            resolved_a, _ = resolve_model_name(model_name_a)
            model_a_found = resolved_a is not None
        if model_name_b:
            resolved_b, _ = resolve_model_name(model_name_b)
            model_b_found = resolved_b is not None

        vae_found = True
        vae_name_str = vae.get('name', '')
        if vae_name_str and not vae_name_str.startswith('('):
            vae_found = resolve_vae_name(vae_name_str) is not None

        lora_availability = {}
        for lora in extracted.get('loras_a', []) + extracted.get('loras_b', []):
            lora_name = lora.get('name', '')
            if lora_name and lora_name not in lora_availability:
                _, found = resolve_lora_path(lora_name)
                lora_availability[lora_name] = found

        # ── Build response matching execute() ui_info['extracted'] ──────
        ui_info = {
            'positive_prompt':    extracted['positive_prompt'],
            'negative_prompt':    extracted['negative_prompt'],
            'model_a':            model_name_a,
            'model_b':            model_name_b,
            'model_a_found':      model_a_found,
            'model_b_found':      model_b_found,
            'loras_a':            extracted['loras_a'],
            'loras_b':            extracted['loras_b'],
            'vae':                extracted['vae'],
            'vae_found':          vae_found,
            'clip':               extracted['clip'],
            'sampler':            sampler,
            'resolution':         resolution,
            'is_video':           extracted.get('is_video', False),
            'model_family':       family_key,
            'model_family_label': get_family_label(family_key),
            'lora_availability':  lora_availability,
        }

        print(f"[RecipeBuilder] process-extracted: family={family_key}, "
              f"model_a={model_name_a}, vae={vae.get('name', '')}, "
              f"loras={len(extracted.get('loras_a', []))}+{len(extracted.get('loras_b', []))}")
        return server.web.json_response({"extracted": ui_info})
    except Exception as e:
        print(f"[RecipeBuilder] process-extracted error: {e}")
        traceback.print_exc()
        return server.web.json_response({"error": str(e)}, status=500)


@server.PromptServer.instance.routes.get("/workflow-builder/get-extracted-data")
async def api_get_workflow_builder_extracted_data(request):
    """Return the last extracted info cached by RecipeBuilder after execution."""
    try:
        node_id = request.rel_url.query.get('node_id', '')
        if node_id:
            data = _last_workflow_builder_info.get(str(node_id))
            if data:
                return server.web.json_response({"extracted": data, "node_id": node_id})
            return server.web.json_response({
                "extracted": None,
                "node_id": node_id,
                "error": "No cached data for this node. Execute RecipeBuilder first.",
            })

        available = {nid: bool(d) for nid, d in _last_workflow_builder_info.items()}
        return server.web.json_response({"available": available})
    except Exception as e:
        print(f"[RecipeBuilder] Error in get-extracted-data: {e}")
        return server.web.json_response({"extracted": None, "error": str(e)}, status=500)


# ─── Main Node ──────────────────────────────────────────────────────────────

class WorkflowBuilder:
    """
    Workflow Builder — UI and extraction node.

    Can run standalone with manual settings, or accept recipe_data (from
    PromptExtractor) and/or lora_stack inputs to pre-fill all parameters.

    Widget order:  Resolution → Model / VAE / CLIP → Prompts → Sampler → LoRAs
    Outputs RECIPE_DATA (JSON string) for the Workflow Renderer render node.
    """

    # Class-level cache so models persist across executions.
    # ComfyUI creates a new node instance on every queue run, so instance-level
    # caches are always empty.  Keyed by (unique_id, full_path, family_key) so
    # each canvas node has its own cache slot and model changes are detected.
    _class_model_cache: dict = {}
    # Cache last effective LoRA UI state per node to protect against transient
    # empty lora_state payloads when connected input stacks are unchanged.
    _class_lora_ui_cache: dict = {}

    def __init__(self):
        pass  # cache lives at class level — see _class_model_cache

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                # ── Connectable inputs ────────────────────────────────
                "recipe_data": ("RECIPE_DATA", {
                    "tooltip": "Optional recipe_data input for prefill/update.",
                }),
                "builder_data": ("BUILDER_DATA", {
                    "tooltip": "Optional A/B/C/D bundle of prompts, seeds, and LoRA stacks.",
                }),
                # ── Hidden state widgets — written by JS, read by Python ──
                "override_data":   ("STRING", {"default": "{}", "multiline": True}),
                "lora_state":      ("STRING", {"default": "{}", "multiline": True}),
            },
            "hidden": {
                "unique_id":     "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES  = ("RECIPE_DATA",)
    RETURN_NAMES  = ("recipe_data",)
    FUNCTION      = "execute"
    CATEGORY      = "Prompt Manager"
    OUTPUT_NODE   = True
    DESCRIPTION   = (
        "Workflow Builder. Extracts parameters from images/workflows, provides "
        "a full editing UI, and outputs recipe_data for the Workflow Renderer."
    )

    def execute(self,
                recipe_data=None,
                pos_prompt=None, neg_prompt=None,
                seed=None, seed_a=None, seed_b=None, denoise=None,
                lora_stack=None, lora_stack_a=None, lora_stack_b=None,
                builder_data=None,
                multi_pos_prompts=None, multi_neg_prompts=None,
                multi_seeds=None, multi_loras=None,
                override_data="{}", lora_state="{}",
                unique_id=None, extra_pnginfo=None, prompt=None):
        """
        Main execution:
          1. Parse recipe_data (if connected)
          2. Apply JS overrides
          3. Merge connected lora inputs (always)
          4. Build recipe_data JSON
          5. Return RECIPE_DATA
        """
        # Backward-compat input mapping: route legacy scalar inputs to slot A.
        if seed is not None and seed_a is None:
            seed_a = seed
        if lora_stack is not None and lora_stack_a is None:
            lora_stack_a = lora_stack

        try:
            overrides = json.loads(override_data) if override_data else {}
        except json.JSONDecodeError:
            overrides = {}
        if not isinstance(overrides, dict):
            overrides = {}
        raw_active_model_slot = str(overrides.get("_model_slot", "") or "").strip().lower()
        active_model_slot = raw_active_model_slot if raw_active_model_slot in _MODEL_KEYS else None

        # Multi-slot builder semantics: selected slot drives read/display only.
        load_model_slot = active_model_slot
        load_secondary_pull_slot = None

        def _extract_multi_prompt_value(payload, slot_key, prompt_key=None):
            if not isinstance(payload, dict):
                return None
            slot_val = payload.get(slot_key)
            if isinstance(slot_val, str):
                return slot_val if slot_val.strip() else None
            if isinstance(slot_val, dict):
                val = slot_val.get(prompt_key) if prompt_key is not None else None
                if val is None:
                    return None
                val_str = str(val)
                return val_str if val_str.strip() else None
            return None

        def _extract_multi_seed_value(payload, slot_key):
            if not isinstance(payload, dict):
                return None
            if slot_key not in payload:
                return None
            try:
                val = int(payload.get(slot_key))
                return val if val != 0 else None
            except Exception:
                return None

        def _extract_multi_lora_stack(payload, slot_key):
            if not isinstance(payload, dict):
                return None
            stack = payload.get(slot_key)
            return stack if isinstance(stack, list) else None

        # Unified builder_data payload (from RecipeBuilderDataBundle) maps to
        # the existing multi_* structures used by slot merge/lock logic.
        if isinstance(builder_data, dict):
            slot_map = {
                "model_a": "a",
                "model_b": "b",
                "model_c": "c",
                "model_d": "d",
            }
            if not isinstance(multi_pos_prompts, dict):
                multi_pos_prompts = {}
            if not isinstance(multi_neg_prompts, dict):
                multi_neg_prompts = {}
            if not isinstance(multi_seeds, dict):
                multi_seeds = {}
            if not isinstance(multi_loras, dict):
                multi_loras = {}

            for slot_key, suffix in slot_map.items():
                pos_val = builder_data.get(f"pos_{suffix}")
                if pos_val is not None:
                    multi_pos_prompts[slot_key] = str(pos_val or "")

                neg_val = builder_data.get(f"neg_{suffix}")
                if neg_val is not None:
                    multi_neg_prompts[slot_key] = str(neg_val or "")

                if f"seed_{suffix}" in builder_data:
                    try:
                        multi_seeds[slot_key] = int(builder_data.get(f"seed_{suffix}") or 0)
                    except Exception:
                        pass

                lora_val = builder_data.get(f"loras_{suffix}")
                if isinstance(lora_val, list):
                    multi_loras[slot_key] = lora_val

        # ── Parse recipe_data input (if connected) ─────────────────────
        wf_data = None
        if recipe_data is not None:
            if isinstance(recipe_data, dict):
                wf_data = recipe_data
            elif isinstance(recipe_data, str):
                try:
                    wf_data = json.loads(recipe_data)
                except (json.JSONDecodeError, TypeError):
                    print("[RecipeBuilder] Warning: could not parse recipe_data")

        # RecipeBuilder consumes v2 recipe_data only.
        if isinstance(wf_data, dict):
            is_v2_recipe = int(wf_data.get("version", 0) or 0) >= 2 and isinstance(wf_data.get("models"), dict)
            if not is_v2_recipe:
                print("[RecipeBuilder] Warning: non-v2 recipe_data received; ignoring input payload")
                wf_data = None

        # ── Build extracted dict from recipe_data or defaults ───────────
        if wf_data:
            v2_a = _v2_get_model_block(wf_data, load_model_slot) if load_model_slot else None
            v2_b = _v2_get_model_block(wf_data, load_secondary_pull_slot) if load_secondary_pull_slot else None

            # Strict multi behavior: if selected slot has no model, treat as
            # empty selection and clear extracted UI state.
            if isinstance(v2_a, dict) and str(v2_a.get('model', '') or '').strip() == '':
                v2_a = None
                v2_b = None

            if isinstance(v2_a, dict):
                sa = v2_a.get('sampler', {}) if isinstance(v2_a.get('sampler'), dict) else {}
                sb = v2_b.get('sampler', {}) if isinstance(v2_b, dict) and isinstance(v2_b.get('sampler'), dict) else {}
                ra = v2_a.get('resolution', {}) if isinstance(v2_a.get('resolution'), dict) else {}
                extracted = {
                    'positive_prompt': v2_a.get('positive_prompt', ''),
                    'negative_prompt': v2_a.get('negative_prompt', ''),
                    'loras_a': v2_a.get('loras', []) if isinstance(v2_a.get('loras'), list) else [],
                    'loras_b': v2_b.get('loras', []) if isinstance(v2_b, dict) and isinstance(v2_b.get('loras'), list) else [],
                    'model_a': v2_a.get('model', ''),
                    'model_b': v2_b.get('model', '') if isinstance(v2_b, dict) else '',
                    'vae':     {'name': v2_a.get('vae', ''), 'source': 'workflow_data'},
                    'clip':    {
                        'names': v2_a.get('clip', []) if isinstance(v2_a.get('clip'), list) else ([v2_a['clip']] if v2_a.get('clip') else []),
                        'type': v2_a.get('clip_type', ''), 'source': 'workflow_data',
                    },
                    'sampler': {
                        'steps_a': sa.get('steps', 20),
                        'steps_b': sb.get('steps') if isinstance(v2_b, dict) else None,
                        'cfg': sa.get('cfg', 5.0),
                        'denoise': sa.get('denoise', 1.0),
                        'seed_a': sa.get('seed', 0),
                        'seed_b': sb.get('seed') if isinstance(v2_b, dict) else None,
                        'sampler_name': sa.get('sampler_name', 'euler'),
                        'scheduler': sa.get('scheduler', 'simple'),
                    },
                    'resolution': {
                        'width': ra.get('width', 768),
                        'height': ra.get('height', 1280),
                        'batch_size': ra.get('batch_size', 1),
                        'length': ra.get('length'),
                    },
                    'is_video': ra.get('length') is not None,
                    'model_family': v2_a.get('family', ''),
                    'model_family_label': get_family_label(v2_a.get('family', '')),
                    'model_slot': load_model_slot or '',
                }
            else:
                extracted = {
                    'positive_prompt': '',
                    'negative_prompt': '',
                    'loras_a': [],
                    'loras_b': [],
                    'model_a': '',
                    'model_b': '',
                    'vae':     {'name': '', 'source': 'workflow_data'},
                    'clip':    {
                        'names': [],
                        'type': '', 'source': 'workflow_data',
                    },
                    'sampler': {
                        'steps_a': 20,
                        'steps_b': None,
                        'cfg': 5.0,
                        'denoise': 1.0,
                        'seed_a': 0,
                        'seed_b': None,
                        'sampler_name': 'euler',
                        'scheduler': 'simple',
                    },
                    'resolution': {
                        'width': 768,
                        'height': 1280,
                        'batch_size': 1,
                        'length': None,
                    },
                    'is_video': False,
                    'model_family': '',
                    'model_family_label': get_family_label(''),
                    'model_slot': load_model_slot or '',
                }
        else:
            extracted = {
                'positive_prompt': '',
                'negative_prompt': '',
                'loras_a': [],
                'loras_b': [],
                'model_a': '',
                'model_b': '',
                'vae':     {'name': '', 'source': 'unknown'},
                'clip':    {'names': [], 'type': '', 'source': 'unknown'},
                'sampler': {
                    'steps_a': 20, 'steps_b': None, 'cfg': 5.0, 'denoise': 1.0, 'seed_a': 0,
                    'sampler_name': 'euler', 'scheduler': 'simple',
                },
                'resolution': {
                    'width': 768, 'height': 1280, 'batch_size': 1, 'length': None,
                },
                'is_video': False,
                'model_slot': load_model_slot or '',
            }

        # ── Merge connected LoRA stacks with workflow LoRAs ──────────────
        # Keep workflow LoRAs and add connected input LoRAs.
        # Dedup by name; if same LoRA in both, input wins (user intent).
        workflow_loras_a = list(extracted.get('loras_a', []))
        workflow_loras_b = list(extracted.get('loras_b', []))
        input_loras_a = []
        input_loras_b = []

        def _merge_lora_lists(wf_list, inp_list):
            by_name = {}
            for lst in wf_list:
                by_name[lst.get('name', '')] = lst
            for lst in inp_list:
                by_name[lst.get('name', '')] = lst
            return list(by_name.values())

        def _normalize_input_lora_stack(raw_stack):
            normalized = []
            if not raw_stack:
                return normalized

            for entry in raw_stack:
                name = None
                model_strength = 1.0
                clip_strength = 1.0

                if isinstance(entry, (list, tuple)) and len(entry) >= 1:
                    name = entry[0]
                    if len(entry) >= 2:
                        try:
                            model_strength = float(entry[1])
                        except Exception:
                            model_strength = 1.0
                    if len(entry) >= 3:
                        try:
                            clip_strength = float(entry[2])
                        except Exception:
                            clip_strength = model_strength
                    else:
                        clip_strength = model_strength
                elif isinstance(entry, dict):
                    name = entry.get('path') or entry.get('name')
                    try:
                        model_strength = float(entry.get('model_strength', entry.get('strength', 1.0)))
                    except Exception:
                        model_strength = 1.0
                    try:
                        clip_strength = float(entry.get('clip_strength', model_strength))
                    except Exception:
                        clip_strength = model_strength
                else:
                    name = entry

                if not name:
                    continue

                path_name = str(name)
                norm_name = strip_lora_extension(os.path.basename(path_name))
                normalized.append({
                    'name': norm_name,
                    'path': path_name,
                    'model_strength': model_strength,
                    'clip_strength': clip_strength,
                    'active': True,
                    'source_input': True,
                })

            return normalized

        input_loras_a = _normalize_input_lora_stack(lora_stack_a)
        input_loras_b = _normalize_input_lora_stack(lora_stack_b)
        has_connected_lora_input = (lora_stack_a is not None) or (lora_stack_b is not None)
        extracted['loras_a'] = _merge_lora_lists(workflow_loras_a, input_loras_a)
        extracted['loras_b'] = _merge_lora_lists(workflow_loras_b, input_loras_b)

        def _lora_input_signature(lora_list):
            rows = []
            for lora_item in (lora_list or []):
                if not isinstance(lora_item, dict):
                    continue
                rows.append((
                    str(lora_item.get('name', '')),
                    str(lora_item.get('path', '')),
                    float(lora_item.get('model_strength', 1.0)),
                    float(lora_item.get('clip_strength', lora_item.get('model_strength', 1.0))),
                ))
            rows.sort()
            return json.dumps(rows, separators=(',', ':'))

        node_state_key = str(unique_id) if unique_id is not None else None
        input_lora_sig = _lora_input_signature(input_loras_a) + "||" + _lora_input_signature(input_loras_b)

        # ── Parse overrides from JS ──────────────────────────────────────
        try:
            lora_overrides = json.loads(lora_state) if lora_state else {}
        except json.JSONDecodeError:
            lora_overrides = {}

        # If connected input stacks did not change but lora_state arrived empty,
        # reuse last known effective LoRA UI state for this node.
        if node_state_key is not None and (not isinstance(lora_overrides, dict) or len(lora_overrides) == 0):
            cached_lora_state = WorkflowBuilder._class_lora_ui_cache.get(node_state_key)
            if isinstance(cached_lora_state, dict) and cached_lora_state.get('input_sig') == input_lora_sig:
                cached_overrides = cached_lora_state.get('overrides')
                if isinstance(cached_overrides, dict) and cached_overrides:
                    lora_overrides = dict(cached_overrides)

        section_locks = overrides.get('_section_locks', {}) if isinstance(overrides, dict) else {}
        # Pull From none means connected recipe_data should not drive section
        # values during execute; keep current UI state unless explicit inputs
        # (prompt/seed/lora stacks) are connected.
        has_workflow_input = (wf_data is not None)
        upstream_source = str((wf_data or {}).get('_source', '')).strip().lower() if isinstance(wf_data, dict) else ""
        # Sources that should keep RecipeBuilder in manual-edit mode while connected.
        # This includes extractors and manager nodes that users actively edit.
        manual_override_sources = {
            "promptextractor",
            "workflowextractor",
            "promptmanageradvanced",
            "workflowmanager",
        }
        manual_sourced_input = upstream_source in manual_override_sources

        def _allow_override(section_name):
            # Extractor/Manager sourced workflow_data keeps WB in manual-edit mode.
            # Other chained workflow_data sources are lock-gated to stay in sync.
            if has_workflow_input and manual_sourced_input:
                return True
            # If workflow_data is connected from non-extractor sources,
            # only locked sections keep local UI overrides.
            if has_workflow_input:
                return bool(section_locks.get(section_name, False))
            # Standalone mode (no workflow_data): local UI overrides apply normally.
            return True

        def _normalize_override_lora_list(raw_list):
            out = []
            if not isinstance(raw_list, list):
                return out

            for item in raw_list:
                if not isinstance(item, dict):
                    continue
                name = str(item.get('name', '')).strip()
                if not name:
                    continue

                path = str(item.get('path', name)).strip() or name
                try:
                    model_strength = float(item.get('model_strength', item.get('strength', 1.0)))
                except Exception:
                    model_strength = 1.0
                try:
                    clip_strength = float(item.get('clip_strength', model_strength))
                except Exception:
                    clip_strength = model_strength

                row = {
                    'name': name,
                    'path': path,
                    'model_strength': model_strength,
                    'clip_strength': clip_strength,
                    'active': item.get('active', True) is not False,
                }
                if 'source_input' in item:
                    row['source_input'] = item.get('source_input', False) is True
                if 'available' in item:
                    row['available'] = item.get('available', True) is not False
                out.append(row)

            return out

        def _apply_preferred_lora_order(current_list, preferred_list):
            current = list(current_list) if isinstance(current_list, list) else []
            preferred = preferred_list if isinstance(preferred_list, list) else []
            if not current or not preferred:
                return current

            current_by_name = {}
            for row in current:
                if not isinstance(row, dict):
                    continue
                name = str(row.get('name', '')).strip()
                if not name or name in current_by_name:
                    continue
                current_by_name[name] = row

            ordered = []
            used_names = set()
            for row in preferred:
                if not isinstance(row, dict):
                    continue
                name = str(row.get('name', '')).strip()
                if not name or name in used_names:
                    continue
                match = current_by_name.get(name)
                if match is not None:
                    ordered.append(match)
                    used_names.add(name)

            for row in current:
                if not isinstance(row, dict):
                    continue
                name = str(row.get('name', '')).strip()
                if not name or name in used_names:
                    continue
                ordered.append(row)
                used_names.add(name)

            return ordered

        # If JS persisted full LoRA stacks in override_data, treat those as
        # authoritative UI state for this run only when no connected LoRA
        # inputs are present. Connected stacks must be authoritative on the
        # current execute so add/remove changes apply immediately.
        override_loras_a = None
        override_loras_b = None
        if _allow_override('loras'):
            if 'loras_a' in overrides:
                override_loras_a = _normalize_override_lora_list(overrides.get('loras_a'))
            if 'loras_b' in overrides:
                override_loras_b = _normalize_override_lora_list(overrides.get('loras_b'))

            if not has_connected_lora_input:
                if override_loras_a is not None:
                    extracted['loras_a'] = override_loras_a
                if override_loras_b is not None:
                    extracted['loras_b'] = override_loras_b
            else:
                if override_loras_a is not None:
                    input_names_a = {
                        str(input_row.get('name', '')).strip()
                        for input_row in (input_loras_a or [])
                        if isinstance(input_row, dict) and str(input_row.get('name', '')).strip()
                    }
                    pruned_pref_a = []
                    local_pref_a = []
                    for row in override_loras_a:
                        name = str((row or {}).get('name', '')).strip() if isinstance(row, dict) else ''
                        if not name:
                            continue
                        source_input = bool(row.get('source_input', False)) if isinstance(row, dict) else False
                        # Drop stale rows that came from a previously connected
                        # lora_stack entry that has since been removed.
                        if source_input and name not in input_names_a:
                            continue
                        pruned_pref_a.append(row)
                        if not source_input:
                            local_pref_a.append(row)

                    # Keep local recipe/UI-authored rows, then merge current
                    # workflow + connected input rows, then apply preferred order.
                    merged_a = _merge_lora_lists(local_pref_a, extracted.get('loras_a', []))
                    extracted['loras_a'] = _apply_preferred_lora_order(merged_a, pruned_pref_a)
                if override_loras_b is not None:
                    input_names_b = {
                        str(input_row.get('name', '')).strip()
                        for input_row in (input_loras_b or [])
                        if isinstance(input_row, dict) and str(input_row.get('name', '')).strip()
                    }
                    pruned_pref_b = []
                    local_pref_b = []
                    for row in override_loras_b:
                        name = str((row or {}).get('name', '')).strip() if isinstance(row, dict) else ''
                        if not name:
                            continue
                        source_input = bool(row.get('source_input', False)) if isinstance(row, dict) else False
                        if source_input and name not in input_names_b:
                            continue
                        pruned_pref_b.append(row)
                        if not source_input:
                            local_pref_b.append(row)

                    merged_b = _merge_lora_lists(local_pref_b, extracted.get('loras_b', []))
                    extracted['loras_b'] = _apply_preferred_lora_order(merged_b, pruned_pref_b)

        # ── Apply overrides ──────────────────────────────────────────────
        pos_text = extracted['positive_prompt']
        neg_text = extracted['negative_prompt']
        if _allow_override('positive'):
            pos_text = overrides.get('positive_prompt', pos_text)
        if _allow_override('negative'):
            neg_text = overrides.get('negative_prompt', neg_text)

        # ── Prompt input override (if connected, use them) ─────────────
        if pos_prompt is not None:
            pos_text = pos_prompt
        if neg_prompt is not None:
            neg_text = neg_prompt

        multi_pos_text = _extract_multi_prompt_value(multi_pos_prompts, active_model_slot, 'positive')
        if multi_pos_text is not None:
            pos_text = multi_pos_text
        multi_neg_text = _extract_multi_prompt_value(multi_neg_prompts, active_model_slot, 'negative')
        if multi_neg_text is not None:
            neg_text = multi_neg_text
        model_name_a = extracted['model_a']
        model_name_b = extracted.get('model_b', '')
        vae_name = extracted['vae']['name']
        if _allow_override('model'):
            model_name_a = overrides.get('model_a', model_name_a)
            model_name_b = overrides.get('model_b', model_name_b)
            vae_name = overrides.get('vae', vae_name)
        sampler_params = extracted['sampler'].copy()
        if _allow_override('sampler'):
            for key in ['steps_a', 'steps_b', 'cfg', 'denoise', 'seed_a', 'seed_b', 'sampler_name', 'scheduler']:
                if key in overrides:
                    val = overrides[key]
                    # Guard: overrides could carry a stale list from a corrupt
                    # override_data blob — coerce numeric fields to scalar.
                    if key in ('steps_a', 'steps_b', 'seed_a', 'seed_b') and isinstance(val, list):
                        val = 0
                    elif key == 'cfg' and isinstance(val, list):
                        val = 5.0
                    elif key == 'denoise' and isinstance(val, list):
                        val = 1.0
                    sampler_params[key] = val
        # Also ensure extracted sampler seed/steps are never lists
        for key, default in (('seed_a', 0), ('seed_b', None), ('steps_a', 20), ('steps_b', None), ('cfg', 5.0), ('denoise', 1.0)):
            if isinstance(sampler_params.get(key), list):
                sampler_params[key] = default

        # Connected seed inputs always override local sampler seed controls,
        # similar to connected prompt inputs overriding prompt textareas.
        if seed_a is not None:
            try:
                sampler_params['seed_a'] = int(seed_a)
            except (TypeError, ValueError):
                pass
        if seed_b is not None:
            try:
                sampler_params['seed_b'] = int(seed_b)
            except (TypeError, ValueError):
                pass
        if denoise is not None:
            try:
                sampler_params['denoise'] = float(denoise)
            except (TypeError, ValueError):
                pass

        multi_seed_val = _extract_multi_seed_value(multi_seeds, active_model_slot)
        if multi_seed_val is not None:
            sampler_params['seed_a'] = multi_seed_val

        # Family + strategy
        # Priority: 1. JS override '_family' (user's explicit dropdown selection)
        #           2. Detected from actual model_name_a (reliable, based on disk)
        #           3. wf_data['family'] (from PE or previous run — may be stale)
        #           4. extracted['model_family'] (fallback)
        # The JS dropdown is the user's deliberate choice and must win.
        js_family = (overrides.get('_family') or None) if _allow_override('model') else None
        selected_v2_block = _v2_get_model_block(wf_data, load_model_slot) if isinstance(wf_data, dict) else None
        wf_family = selected_v2_block.get('family', '') if isinstance(selected_v2_block, dict) else ''
        incoming_family = extracted.get('model_family') or wf_family or ''
        model_detected_family = None
        if model_name_a:
            resolved_ref, _ = resolve_model_name(model_name_a)
            model_detected_family = get_model_family(resolved_ref or model_name_a)
        family_key = js_family or model_detected_family or wf_family or extracted.get('model_family') or None
        if not family_key and isinstance(selected_v2_block, dict):
            clip_type = str(selected_v2_block.get('clip_type', '')).lower()
            loader_type = selected_v2_block.get('loader_type', '')
            if loader_type == 'checkpoint':
                family_key = 'sdxl'
            elif 'flux2' in clip_type:
                family_key = 'flux2'
            elif 'flux' in clip_type:
                family_key = 'flux1'
            elif 'sd3' in clip_type:
                family_key = 'sd3'
            elif 'wan' in clip_type:
                family_key = 'wan_video_t2v'  # closest generic WAN family
            elif 'qwen_image' in clip_type:
                family_key = 'qwen_image'
            elif 'lumina2' in clip_type:
                family_key = 'zimage'
            # other: fall through to sdxl default — better than wrong family
        if not family_key:
            family_key = "" if not str(model_name_a or "").strip() else "sdxl"

        strategy = get_family_sampler_strategy(family_key)

        print(f"[RecipeBuilder] Family: {get_family_label(family_key)} "
              f"(strategy={strategy}), model_a={model_name_a}, "
              f"model_b={model_name_b or '—'}")

        # If user changed family in UI, don't carry stale extracted VAE/CLIP
        # from the previous family unless they explicitly selected them.
        family_switched_in_ui = bool(
            _allow_override('model') and js_family and incoming_family and js_family != incoming_family
        )

        def _families_compatible(source_family, target_family):
            src = str(source_family or "").strip()
            dst = str(target_family or "").strip()
            if not src or not dst:
                return True
            if src == dst:
                return True
            try:
                src_compat = set(get_compatible_families(src) or [])
            except Exception:
                src_compat = set()
            try:
                dst_compat = set(get_compatible_families(dst) or [])
            except Exception:
                dst_compat = set()
            return (dst in src_compat) or (src in dst_compat)

        # Builder-to-Builder chaining guard:
        # if upstream workflow family and selected execution family are
        # incompatible, drop carried workflow LoRAs. Keep explicitly connected
        # input stacks only, so user-intent inputs still work.
        if not _families_compatible(incoming_family, family_key):
            extracted['loras_a'] = list(input_loras_a)
            extracted['loras_b'] = list(input_loras_b)
            if override_loras_a is not None:
                extracted['loras_a'] = _apply_preferred_lora_order(extracted['loras_a'], override_loras_a)
            if override_loras_b is not None:
                extracted['loras_b'] = _apply_preferred_lora_order(extracted['loras_b'], override_loras_b)
            print(
                f"[RecipeBuilder] Dropped incompatible upstream LoRAs "
                f"(incoming_family={incoming_family}, selected_family={family_key})"
            )

        # ── Build workflow JSON + UI info BEFORE model loading ────────────
        # This ensures the JS UI is always populated, even if generation
        # fails (e.g. model not found).  The user can then edit settings
        # and re-queue.
        wf_overrides = {'_source': 'RecipeBuilder'}
        if _allow_override('model'):
            for key in ('model_a', 'model_b', 'vae', 'clip_names', '_family'):
                if key in overrides:
                    wf_overrides[key] = overrides[key]
            if family_switched_in_ui:
                if 'vae' not in overrides:
                    wf_overrides['vae'] = ''
                if 'clip_names' not in overrides:
                    wf_overrides['clip_names'] = []
        if _allow_override('sampler'):
            for key in ('steps_a', 'steps_b', 'cfg', 'denoise', 'seed_a', 'seed_b', 'sampler_name', 'scheduler'):
                if key in overrides:
                    wf_overrides[key] = overrides[key]
        if _allow_override('resolution'):
            for key in ('width', 'height', 'batch_size', 'length'):
                if key in overrides:
                    wf_overrides[key] = overrides[key]
        # Inject prompt input overrides so build_simplified_workflow_data uses them
        wf_overrides['positive_prompt'] = pos_text
        wf_overrides['negative_prompt'] = neg_text
        extracted['model_family'] = family_key
        extracted['model_family_label'] = get_family_label(family_key)

        # ── Apply lora_overrides: set active flag + update strengths ─────
        # Keep ALL loras (like PMA) but mark inactive ones with active=False.
        # Only _apply_loras filters out inactive when actually loading models.
        def _resolve_lora_override(overrides_map, lora_name, stack_key):
            """Resolve LoRA UI state across prefixed/unprefixed key variants.

            Historical/UI states may use either `<name>` or `<stack>:<name>` keys,
            and stack shape can change between runs. Prefer stack-specific key
            then fall back to bare-name key.
            """
            name = str(lora_name or "")
            if not name or not isinstance(overrides_map, dict):
                return {}
            pref_key = f"{stack_key}:{name}" if stack_key else name
            ov = overrides_map.get(pref_key)
            if isinstance(ov, dict):
                return ov
            ov = overrides_map.get(name)
            return ov if isinstance(ov, dict) else {}

        for stack_key, list_key in [('a', 'loras_a'), ('b', 'loras_b')]:
            updated_list = []
            for lora in extracted.get(list_key, []):
                lora_name = lora.get('name', '')
                lora_st = _resolve_lora_override(lora_overrides, lora_name, stack_key)
                updated = dict(lora)
                # Preserve incoming active state unless user explicitly toggled it in UI.
                updated['active'] = lora_st.get('active', lora.get('active', True)) is not False
                if 'model_strength' in lora_st:
                    updated['model_strength'] = float(lora_st['model_strength'])
                if 'clip_strength' in lora_st:
                    updated['clip_strength'] = float(lora_st['clip_strength'])
                updated_list.append(updated)
            extracted[list_key] = updated_list

        # Persist last effective LoRA UI state for unchanged-stack protection.
        if node_state_key is not None:
            effective_map = {}
            for stack_key, list_key in [('a', 'loras_a'), ('b', 'loras_b')]:
                for lora in extracted.get(list_key, []):
                    name = str(lora.get('name', '')).strip()
                    if not name:
                        continue
                    st = {
                        'active': lora.get('active', True) is not False,
                        'model_strength': float(lora.get('model_strength', lora.get('strength', 1.0))),
                        'clip_strength': float(lora.get('clip_strength', lora.get('model_strength', lora.get('strength', 1.0)))),
                    }
                    effective_map[name] = st
                    effective_map[f"{stack_key}:{name}"] = st
            WorkflowBuilder._class_lora_ui_cache[node_state_key] = {
                'input_sig': input_lora_sig,
                'overrides': effective_map,
            }

        simplified_wf = build_simplified_workflow_data(
            extracted, wf_overrides, sampler_params
        )

        # ── Ensure clip_type and loader_type are always set from family ───
        # build_simplified_workflow_data derives these from extracted['clip'],
        # which may have empty type/source when the user is in manual mode
        # (no workflow_data connected) or the source file had no metadata.
        # The family spec is authoritative — always fill from it.
        from ..py.workflow_families import MODEL_FAMILIES
        spec = MODEL_FAMILIES.get(family_key, {})
        if spec:
            # Family selection is authoritative in WB: always align clip/loader type.
            simplified_wf['clip_type'] = spec.get('clip_type', '')
            is_ckpt = spec.get('checkpoint', False)
            simplified_wf['loader_type'] = 'checkpoint' if is_ckpt else 'unet'
        else:
            if not simplified_wf.get('clip_type'):
                simplified_wf['clip_type'] = ''
            if not simplified_wf.get('loader_type'):
                simplified_wf['loader_type'] = 'unet'

        # ── Default VAE/CLIP: resolve empty values ─────────────────────────
        # For checkpoint models, empty VAE/CLIP means "use from checkpoint"
        # — the renderer uses the checkpoint's built-in VAE/CLIP.
        # For non-checkpoint (unet) models, resolve to a compatible file.
        is_checkpoint = simplified_wf.get('loader_type') == 'checkpoint'

        # Checkpoint models: use (Default) when user selected "(Default)"
        # (i.e. override value is empty/missing). If user explicitly picked a
        # file in the dropdown, respect that choice.
        if is_checkpoint:
            vae_ov = overrides.get('vae', '')
            if not vae_ov or vae_ov.startswith('('):
                simplified_wf['vae'] = '(Default)'
            clip_ov = overrides.get('clip_names', [])
            if not clip_ov or all(not c or c.startswith('(') for c in clip_ov):
                simplified_wf['clip'] = ['(Default)']
        else:
            # Non-checkpoint: resolve empty VAE/CLIP to compatible files
            vae_val = simplified_wf.get('vae')
            if not vae_val or str(vae_val).startswith('('):
                vaes, rec_vae = list_compatible_vaes(family_key, return_recommended=True)
                fallback_vae = rec_vae or (vaes[0] if vaes else '')
                if fallback_vae:
                    simplified_wf['vae'] = fallback_vae
                    print(f"[RecipeBuilder] VAE defaulted to: {fallback_vae}")
            clip_val = simplified_wf.get('clip')
            clip_is_placeholder = bool(clip_val) and all(
                (not n) or str(n).startswith('(') for n in clip_val
            )
            if not clip_val or clip_val == [] or clip_is_placeholder:
                clip_type_from_spec = spec.get('clip_type', '')
                compatible_clips, rec_clip = list_compatible_clips(family_key, return_recommended=True)
                if compatible_clips:
                    clip_slots = spec.get('clip_slots', 1)
                    clip_patterns = [p.lower() for p in spec.get('clip', [])]
                    selected = []
                    if clip_patterns and clip_slots >= 2:
                        for pat in clip_patterns:
                            for c in compatible_clips:
                                if pat in os.path.basename(c).lower() and c not in selected:
                                    selected.append(c)
                                    break
                            if len(selected) >= clip_slots:
                                break
                    if not selected:
                        selected = [compatible_clips[0]]
                    simplified_wf['clip'] = selected
                    if clip_type_from_spec:
                        simplified_wf['clip_type'] = clip_type_from_spec
                    print(f"[RecipeBuilder] CLIP defaulted to: {selected}")
        lora_availability = {}
        for lora in extracted.get('loras_a', []) + extracted.get('loras_b', []):
            lora_name = lora.get('name', '')
            if lora_name and lora_name not in lora_availability:
                _, available = resolve_lora_path(lora_name)
                lora_availability[lora_name] = available

        # ── Check model / VAE availability for JS display ─────────────────
        model_a_found = True
        model_b_found = True
        if model_name_a:
            resolved_a, _ = resolve_model_name(model_name_a)
            model_a_found = resolved_a is not None
        if model_name_b:
            resolved_b, _ = resolve_model_name(model_name_b)
            model_b_found = resolved_b is not None

        vae_found = True
        vae_info = extracted.get('vae', {})
        vae_name_str = vae_info.get('name', '') if isinstance(vae_info, dict) else (vae_info or '')
        if vae_name_str and not vae_name_str.startswith('('):
            vae_found = resolve_vae_name(vae_name_str) is not None

        # ── Fallback: replace not-found names in workflow_data ────────────
        # JS shows original names in red; workflow_data gets valid defaults
        # so downstream nodes (ModelLoader, Renderer) never get broken names.
        # Always respect the workflow family when picking fallbacks.
        if not model_a_found:
            compat_models = list_compatible_models(model_name_a, family_override=family_key)
            if compat_models:
                fallback = compat_models[0]
                print(f"[RecipeBuilder] Model A '{model_name_a}' not found, workflow_data will use: {fallback}")
                simplified_wf['model_a'] = fallback

        if model_name_b and not model_b_found:
            compat_models = list_compatible_models(model_name_b, family_override=family_key)
            if compat_models:
                fallback = compat_models[0]
                print(f"[RecipeBuilder] Model B '{model_name_b}' not found, workflow_data will use: {fallback}")
                simplified_wf['model_b'] = fallback

        if not vae_found and vae_name_str:
            vaes, recommended = list_compatible_vaes(family_key, return_recommended=True)
            fallback_vae = recommended or (vaes[0] if vaes else '')
            if fallback_vae:
                print(f"[RecipeBuilder] VAE '{vae_name_str}' not found, workflow_data will use: {fallback_vae}")
                simplified_wf['vae'] = fallback_vae

        # CLIP fallback: check each clip name, replace not-found ones
        # Skip for "(Default)" — those aren't real files on disk.
        clip_names_out = simplified_wf.get('clip', [])
        if clip_names_out and not all(n.startswith('(') for n in clip_names_out):
            clip_paths = resolve_clip_names(clip_names_out, simplified_wf.get('clip_type', ''))
            if any(p is None for p in clip_paths):
                compatible_clips = list_compatible_clips(family_key)
                if compatible_clips:
                    fixed_clips = []
                    for i, (name, path) in enumerate(zip(clip_names_out, clip_paths)):
                        if path is None and i < len(compatible_clips):
                            print(f"[RecipeBuilder] CLIP '{name}' not found, workflow_data will use: {compatible_clips[i]}")
                            fixed_clips.append(compatible_clips[i])
                        else:
                            fixed_clips.append(name)
                    simplified_wf['clip'] = fixed_clips

        # Preserve runtime passthrough objects from incoming workflow_data
        # only when the effective selection remains unchanged. If the user
        # changed model/clip/vae in this Builder, drop stale runtime objects
        # so downstream Renderer reloads the new assets.
        if isinstance(wf_data, dict):
            def _norm_name(v):
                return str(v or "").strip()

            def _norm_clip_list(v):
                if isinstance(v, list):
                    return [str(x or "").strip() for x in v if str(x or "").strip()]
                if isinstance(v, str):
                    s = v.strip()
                    return [s] if s else []
                return []

            in_v2_a = _v2_get_model_block(wf_data, load_model_slot)
            in_v2_b = _v2_get_model_block(wf_data, load_secondary_pull_slot) if load_secondary_pull_slot else None

            in_model_a = _norm_name((in_v2_a or {}).get("model", ""))
            out_model_a = _norm_name(simplified_wf.get("model_a", ""))
            in_model_b = _norm_name((in_v2_b or {}).get("model", ""))
            out_model_b = _norm_name(simplified_wf.get("model_b", ""))
            in_vae = _norm_name((in_v2_a or {}).get("vae", ""))
            out_vae = _norm_name(simplified_wf.get("vae", ""))
            in_clip = _norm_clip_list((in_v2_a or {}).get("clip", []))
            out_clip = _norm_clip_list(simplified_wf.get("clip", []))

            in_model_obj_a = (in_v2_a or {}).get("MODEL") if isinstance(in_v2_a, dict) else None
            in_model_obj_b = (in_v2_b or {}).get("MODEL") if isinstance(in_v2_b, dict) else None
            in_clip_obj = (in_v2_a or {}).get("CLIP") if isinstance(in_v2_a, dict) else None
            in_vae_obj = (in_v2_a or {}).get("VAE") if isinstance(in_v2_a, dict) else None

            if out_model_a and in_model_a == out_model_a and in_model_obj_a is not None:
                simplified_wf["MODEL_A"] = in_model_obj_a
            if in_model_b == out_model_b and in_model_obj_b is not None:
                simplified_wf["MODEL_B"] = in_model_obj_b
            if in_clip == out_clip and in_clip_obj is not None:
                simplified_wf["CLIP"] = in_clip_obj
            if in_vae == out_vae and in_vae_obj is not None:
                simplified_wf["VAE"] = in_vae_obj

            # Keep conditioning/latent/image passthrough when present.
            for key in ("POSITIVE", "NEGATIVE", "LATENT", "IMAGE", "MASK"):
                if key in wf_data:
                    simplified_wf[key] = wf_data.get(key)

        # Keep missing LoRAs in workflow_data (Prompt Manager-compatible).
        # Annotate each LoRA with available flag so downstream nodes can
        # skip missing entries without losing authored stack information.
        for lora_key in ('loras_a', 'loras_b'):
            annotated = []
            for lora in list(simplified_wf.get(lora_key, [])):
                row = dict(lora or {})
                # Internal provenance flag used during execute-time merging
                # must never leak into serialized recipe_data.
                row.pop('source_input', None)
                name = str(row.get('name', '')).strip()
                available = lora_availability.get(name, True) if name else True
                row['available'] = bool(available)
                annotated.append(row)

            simplified_wf[lora_key] = annotated

        # Optional multi-slot profile payload from JS. When present, this node
        # can author model_a..model_d blocks in one builder instance.
        base_recipe_for_output = wf_data if isinstance(wf_data, dict) else None
        slot_profiles = overrides.get('_slot_profiles') if isinstance(overrides, dict) else None

        def _slot_profile_is_meaningful(raw_profile):
            if not isinstance(raw_profile, dict):
                return False

            ov_profile = raw_profile.get('ov') if isinstance(raw_profile.get('ov'), dict) else None
            profile = ov_profile if isinstance(ov_profile, dict) else raw_profile

            # Any explicitly provided prompt/model/asset value makes profile meaningful.
            for k in ('positive_prompt', 'negative_prompt', 'model_a', 'model', 'vae', 'family', '_family', 'clip_type', 'loader_type'):
                val = profile.get(k)
                if val is not None and str(val).strip() != '':
                    return True

            clip_names = profile.get('clip_names', profile.get('clip'))
            if isinstance(clip_names, list) and any(str(x or '').strip() for x in clip_names):
                return True

            loras = profile.get('loras_a', profile.get('loras'))
            if isinstance(loras, list) and len(loras) > 0:
                return True

            # Non-default sampler/resolution values indicate authored state.
            try:
                if int(profile.get('steps_a', profile.get('steps', 20)) or 20) != 20:
                    return True
                if float(profile.get('cfg', 5.0) or 5.0) != 5.0:
                    return True
                if float(profile.get('denoise', 1.0) or 1.0) != 1.0:
                    return True
                if int(profile.get('seed_a', profile.get('seed', 0)) or 0) != 0:
                    return True
                if int(profile.get('width', 768) or 768) != 768:
                    return True
                if int(profile.get('height', 1280) or 1280) != 1280:
                    return True
                if int(profile.get('batch_size', 1) or 1) != 1:
                    return True
                if profile.get('length') not in (None, '', 0):
                    return True
            except Exception:
                pass

            locks = profile.get('_section_locks')
            if isinstance(locks, dict) and any(bool(v) for v in locks.values()):
                return True

            return False

        has_meaningful_slot_profiles = isinstance(slot_profiles, dict) and any(
            _slot_profile_is_meaningful(slot_profiles.get(slot_key)) for slot_key in _MODEL_KEYS
        )

        has_multi_slot_inputs = any([
            isinstance(multi_pos_prompts, dict),
            isinstance(multi_neg_prompts, dict),
            isinstance(multi_seeds, dict),
            isinstance(multi_loras, dict),
        ])

        # Absolute-truth execute path: when recipe_data is connected and no
        # multi-slot override inputs are connected, do not let stale UI slot
        # profiles rewrite incoming model blocks.
        allow_slot_profile_authoring = not (isinstance(base_recipe_for_output, dict) and not has_multi_slot_inputs)

        if allow_slot_profile_authoring and isinstance(slot_profiles, dict) and (has_meaningful_slot_profiles or not isinstance(base_recipe_for_output, dict)):
            try:
                if isinstance(base_recipe_for_output, dict) and \
                        int(base_recipe_for_output.get('version', 0) or 0) >= 2 and \
                        isinstance(base_recipe_for_output.get('models'), dict):
                    base_recipe_for_output = dict(base_recipe_for_output)
                    base_recipe_for_output['models'] = {
                        k: dict(v) for k, v in (base_recipe_for_output.get('models') or {}).items()
                        if isinstance(v, dict)
                    }
                else:
                    base_recipe_for_output = {
                        '_source': 'RecipeBuilder',
                        'version': 2,
                        'models': {},
                    }

                def _norm_num(v, default):
                    try:
                        return float(v)
                    except Exception:
                        return float(default)

                def _norm_int(v, default):
                    try:
                        return int(v)
                    except Exception:
                        return int(default)

                def _norm_loras(raw_list):
                    out = []
                    if not isinstance(raw_list, list):
                        return out
                    for item in raw_list:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get('name', '')).strip()
                        if not name:
                            continue
                        path = str(item.get('path', name)).strip() or name
                        model_strength = _norm_num(item.get('model_strength', item.get('strength', 1.0)), 1.0)
                        clip_strength = _norm_num(item.get('clip_strength', model_strength), model_strength)
                        out.append({
                            'name': name,
                            'path': path,
                            'model_strength': model_strength,
                            'clip_strength': clip_strength,
                            'active': item.get('active', True) is not False,
                            'available': item.get('available', True) is not False,
                        })
                    return out

                for slot_key in _MODEL_KEYS:
                    raw_profile = slot_profiles.get(slot_key)
                    if not isinstance(raw_profile, dict):
                        continue

                    if not _slot_profile_is_meaningful(raw_profile):
                        # Keep incoming recipe_data slot block untouched when
                        # profile only contains default placeholder values.
                        continue

                    # Current JS shape stores per-slot profile as {ov, ls}; keep
                    # backward compatibility with legacy raw slot payload.
                    ov_profile = raw_profile.get('ov') if isinstance(raw_profile.get('ov'), dict) else None
                    profile = ov_profile if isinstance(ov_profile, dict) else raw_profile
                    profile_input_ghosts = profile.get('_input_ghosts', {}) if isinstance(profile.get('_input_ghosts'), dict) else {}

                    if isinstance(ov_profile, dict):
                        loras_src = profile.get('loras_a', [])
                        clip_src = profile.get('clip_names', [])
                        sampler_in = {
                            'steps': profile.get('steps_a', 20),
                            'cfg': profile.get('cfg', 5.0),
                            'denoise': profile.get('denoise', 1.0),
                            'seed': profile.get('seed_a', 0),
                            'sampler_name': profile.get('sampler_name', 'euler'),
                            'scheduler': profile.get('scheduler', 'simple'),
                        }
                        resolution_in = {
                            'width': profile.get('width', 768),
                            'height': profile.get('height', 1280),
                            'batch_size': profile.get('batch_size', 1),
                            'length': profile.get('length'),
                        }
                        profile_model = profile.get('model_a', profile.get('model', ''))
                        profile_family = profile.get('_family', profile.get('family', ''))
                        profile_clip_type = profile.get('clip_type', '')
                        profile_loader_type = profile.get('loader_type', '')
                    else:
                        loras_src = profile.get('loras', [])
                        clip_src = profile.get('clip', [])
                        sampler_in = profile.get('sampler', {}) if isinstance(profile.get('sampler'), dict) else {}
                        resolution_in = profile.get('resolution', {}) if isinstance(profile.get('resolution'), dict) else {}
                        profile_model = profile.get('model', '')
                        profile_family = profile.get('family', '')
                        profile_clip_type = profile.get('clip_type', '')
                        profile_loader_type = profile.get('loader_type', '')

                    clip_raw = clip_src if isinstance(clip_src, list) else ([clip_src] if clip_src else [])

                    multi_pos_slot = _extract_multi_prompt_value(multi_pos_prompts, slot_key, 'positive')
                    multi_neg_slot = _extract_multi_prompt_value(multi_neg_prompts, slot_key, 'negative')
                    multi_seed_slot = _extract_multi_seed_value(multi_seeds, slot_key)
                    multi_lora_slot = _extract_multi_lora_stack(multi_loras, slot_key)

                    existing_slot_block = None
                    if isinstance(base_recipe_for_output, dict):
                        existing_models = base_recipe_for_output.get('models')
                        if isinstance(existing_models, dict):
                            candidate_slot = existing_models.get(slot_key)
                            if isinstance(candidate_slot, dict):
                                existing_slot_block = candidate_slot

                    existing_slot_loras = []
                    existing_slot_seed = 0
                    if isinstance(existing_slot_block, dict):
                        existing_slot_loras = _norm_loras(existing_slot_block.get('loras', []))
                        existing_slot_sampler = existing_slot_block.get('sampler') if isinstance(existing_slot_block.get('sampler'), dict) else {}
                        existing_slot_seed = _norm_int(existing_slot_sampler.get('seed', 0), 0)

                    raw_profile_loras = loras_src if isinstance(loras_src, list) else []
                    local_profile_loras = []
                    for item in raw_profile_loras:
                        if not isinstance(item, dict):
                            continue
                        if item.get('source_input', False) is True:
                            continue
                        local_profile_loras.append(item)

                    prior_lora_ghost = bool(profile_input_ghosts.get('loras', False))
                    if multi_lora_slot is not None:
                        merged_profile_loras = _merge_lora_lists(
                            _norm_loras(local_profile_loras),
                            _normalize_input_lora_stack(multi_lora_slot),
                        )
                    elif prior_lora_ghost:
                        merged_profile_loras = _merge_lora_lists(
                            _norm_loras(local_profile_loras),
                            existing_slot_loras,
                        )
                    else:
                        merged_profile_loras = _norm_loras(raw_profile_loras)

                    profile_pos = profile.get('positive_prompt', '')
                    if multi_pos_slot is None and bool(profile_input_ghosts.get('positive', False)) and isinstance(existing_slot_block, dict):
                        profile_pos = existing_slot_block.get('positive_prompt', '')
                    if (profile_pos is None or str(profile_pos) == '') and isinstance(existing_slot_block, dict):
                        profile_pos = existing_slot_block.get('positive_prompt', '')

                    profile_neg = profile.get('negative_prompt', '')
                    if multi_neg_slot is None and bool(profile_input_ghosts.get('negative', False)) and isinstance(existing_slot_block, dict):
                        profile_neg = existing_slot_block.get('negative_prompt', '')
                    if (profile_neg is None or str(profile_neg) == '') and isinstance(existing_slot_block, dict):
                        profile_neg = existing_slot_block.get('negative_prompt', '')

                    sampler_seed = sampler_in.get('seed', 0)
                    if multi_seed_slot is None and bool(profile_input_ghosts.get('seed', False)):
                        sampler_seed = existing_slot_seed

                    slot_block = {
                        'positive_prompt': str(multi_pos_slot if multi_pos_slot is not None else (profile_pos or '')),
                        'negative_prompt': str(multi_neg_slot if multi_neg_slot is not None else (profile_neg or '')),
                        'family': str(profile_family or ''),
                        'model': str(profile_model or ''),
                        'loras': merged_profile_loras,
                        'clip_type': str(profile_clip_type or ''),
                        'loader_type': str(profile_loader_type or ''),
                        'vae': str(profile.get('vae', '') or ''),
                        'clip': [str(x or '') for x in clip_raw if str(x or '').strip()],
                        'sampler': {
                            'steps': _norm_int(sampler_in.get('steps', 20), 20),
                            'cfg': _norm_num(sampler_in.get('cfg', 5.0), 5.0),
                            'denoise': _norm_num(sampler_in.get('denoise', 1.0), 1.0),
                            'seed': _norm_int(multi_seed_slot if multi_seed_slot is not None else sampler_seed, 0),
                            'sampler_name': str(sampler_in.get('sampler_name', 'euler') or 'euler'),
                            'scheduler': str(sampler_in.get('scheduler', 'simple') or 'simple'),
                        },
                        'resolution': {
                            'width': _norm_int(resolution_in.get('width', 768), 768),
                            'height': _norm_int(resolution_in.get('height', 1280), 1280),
                            'batch_size': _norm_int(resolution_in.get('batch_size', 1), 1),
                            'length': resolution_in.get('length'),
                        },
                    }
                    base_recipe_for_output['models'][slot_key] = slot_block
            except Exception as e:
                print(f"[RecipeBuilder] Warning: failed to apply _slot_profiles: {e}")

        output_wf = _builder_output_to_v2(
            simplified_wf,
            base_recipe_data=base_recipe_for_output,
            overwrite_existing_slots=False,
        )

        # ── Build UI info for JS (always, even if generation fails) ───────
        # Echo back the *effective* values (overrides applied) so the JS
        # pre-update handler sees what the user actually has set and does
        # NOT mistakenly treat it as a source change that clears all fields.
        effective_sampler = dict(sampler_params)
        if _allow_override('sampler'):
            for key in ['steps_a', 'steps_b', 'cfg', 'denoise', 'seed_a', 'seed_b', 'sampler_name', 'scheduler']:
                if key in overrides:
                    effective_sampler[key] = overrides[key]

        # Ensure UI info mirrors connected seed inputs too.
        if seed_a is not None:
            try:
                effective_sampler['seed_a'] = int(seed_a)
            except (TypeError, ValueError):
                pass
        if seed_b is not None:
            try:
                effective_sampler['seed_b'] = int(seed_b)
            except (TypeError, ValueError):
                pass
        if denoise is not None:
            try:
                effective_sampler['denoise'] = float(denoise)
            except (TypeError, ValueError):
                pass

        effective_resolution = dict(extracted['resolution'])
        if _allow_override('resolution'):
            for key in ['width', 'height', 'batch_size', 'length']:
                if key in overrides:
                    effective_resolution[key] = overrides[key]

        effective_vae = extracted['vae']
        if _allow_override('model') and overrides.get('vae'):
            effective_vae = {'name': overrides['vae'], 'source': 'override'}

        effective_clip = extracted['clip']
        if _allow_override('model') and overrides.get('clip_names'):
            effective_clip = {'names': overrides['clip_names'], 'type': '', 'source': 'override'}

        ui_slot_profiles = {}
        slot_profiles_src = slot_profiles if isinstance(slot_profiles, dict) else {}
        output_models = output_wf.get('models', {}) if isinstance(output_wf, dict) else {}
        for slot_key in _MODEL_KEYS:
            row = slot_profiles_src.get(slot_key) if isinstance(slot_profiles_src.get(slot_key), dict) else {}
            row_saved_ov = dict(row.get('ov')) if isinstance(row.get('ov'), dict) else {}
            # Recipe data is authoritative on execute: rebuild each slot from a
            # clean baseline, then apply this slot's model block if present.
            # This guarantees missing slots are explicitly cleared.
            row_ov = {
                'positive_prompt': '',
                'negative_prompt': '',
                'model_a': '',
                '_family': '',
                'clip_type': '',
                'loader_type': '',
                'vae': '',
                'clip_names': [],
                'loras_a': [],
                'steps_a': 20,
                'cfg': 5.0,
                'denoise': 1.0,
                'seed_a': 0,
                'sampler_name': 'euler',
                'scheduler': 'simple',
                'width': 768,
                'height': 1280,
                'batch_size': 1,
                'length': None,
            }
            row_ls = dict(row.get('ls')) if isinstance(row.get('ls'), dict) else {}
            existing_locks = dict(row_saved_ov.get('_section_locks')) if isinstance(row_saved_ov.get('_section_locks'), dict) else {}

            # Check if loras are locked BEFORE overwriting from execution output.
            _pre_locks = existing_locks
            _loras_locked = bool(_pre_locks.get('loras', False))
            _saved_loras_a = list(row_saved_ov.get('loras_a', [])) if _loras_locked else None

            block = output_models.get(slot_key) if isinstance(output_models, dict) else None
            if isinstance(block, dict):
                sampler_block = block.get('sampler') if isinstance(block.get('sampler'), dict) else {}
                resolution_block = block.get('resolution') if isinstance(block.get('resolution'), dict) else {}
                row_ov.update({
                    'positive_prompt': str(block.get('positive_prompt', '') or ''),
                    'negative_prompt': str(block.get('negative_prompt', '') or ''),
                    'model_a': str(block.get('model', '') or ''),
                    '_family': str(block.get('family', '') or ''),
                    'clip_type': str(block.get('clip_type', '') or ''),
                    'loader_type': str(block.get('loader_type', '') or ''),
                    'vae': str(block.get('vae', '') or ''),
                    'clip_names': list(block.get('clip', [])) if isinstance(block.get('clip'), list) else [],
                    'loras_a': list(block.get('loras', [])) if isinstance(block.get('loras'), list) else [],
                    'steps_a': int(sampler_block.get('steps', 20) or 20),
                })
                # Restore loras if they were locked before this execution.
                if _loras_locked and _saved_loras_a is not None:
                    row_ov['loras_a'] = _saved_loras_a
                row_ov.update({
                    'cfg': float(sampler_block.get('cfg', 5.0) or 5.0),
                    'denoise': float(sampler_block.get('denoise', 1.0) or 1.0),
                    'seed_a': int(sampler_block.get('seed', 0) or 0),
                    'sampler_name': str(sampler_block.get('sampler_name', 'euler') or 'euler'),
                    'scheduler': str(sampler_block.get('scheduler', 'simple') or 'simple'),
                    'width': int(resolution_block.get('width', 768) or 768),
                    'height': int(resolution_block.get('height', 1280) or 1280),
                    'batch_size': int(resolution_block.get('batch_size', 1) or 1),
                    'length': resolution_block.get('length'),
                })

            # Execution-time builder_data should ghost the affected inputs for
            # this slot without changing the user's section lock booleans.
            lock_pos  = _extract_multi_prompt_value(multi_pos_prompts, slot_key, 'positive') is not None
            lock_neg  = _extract_multi_prompt_value(multi_neg_prompts, slot_key, 'negative') is not None
            lock_seed = _extract_multi_seed_value(multi_seeds, slot_key) is not None
            row_ov['_section_locks'] = existing_locks
            row_ov['_input_ghosts'] = {
                'positive': lock_pos,
                'negative': lock_neg,
                'seed': lock_seed,
            }

            ui_slot_profiles[slot_key] = {
                'ov': row_ov,
                'ls': row_ls,
            }

        ui_info = {
            'extracted': {
                'positive_prompt':    pos_text,
                'negative_prompt':    neg_text,
                'model_a':            model_name_a,
                'model_b':            model_name_b,
                'model_a_found':      model_a_found,
                'model_b_found':      model_b_found,
                'loras_a':            extracted['loras_a'],
                'loras_b':            extracted['loras_b'],
                # Report only non-input workflow-side rows here; connected
                # lora_stack rows are sent separately in input_loras_*.
                'workflow_loras_a':   [
                    lora_row for lora_row in (extracted.get('loras_a', []) or [])
                    if not (isinstance(lora_row, dict) and lora_row.get('source_input', False) is True)
                ],
                'workflow_loras_b':   [
                    lora_row for lora_row in (extracted.get('loras_b', []) or [])
                    if not (isinstance(lora_row, dict) and lora_row.get('source_input', False) is True)
                ],
                'input_loras_a':      input_loras_a,
                'input_loras_b':      input_loras_b,
                'vae':                effective_vae,
                'vae_found':          vae_found,
                'clip':               effective_clip,
                'sampler':            effective_sampler,
                'resolution':         effective_resolution,
                'is_video':           extracted.get('is_video', False),
                'model_family':       family_key,
                'model_family_label': get_family_label(family_key),
                'model_slot':         active_model_slot or '',
                'lora_availability':  lora_availability,
                '_slot_profiles':     ui_slot_profiles,
            }
        }

        # ── Embed extracted_data into workflow for re-extraction ──────────
        # Uses the same schema as workflow_data (build_simplified_workflow_data)
        # with LoRA entries enriched with active/available state.
        if extra_pnginfo is not None:
            pnginfo = extra_pnginfo
            if isinstance(pnginfo, list) and len(pnginfo) > 0:
                pnginfo = pnginfo[0]
            workflow = None
            if hasattr(pnginfo, 'get'):
                workflow = pnginfo.get('workflow', {})
            elif hasattr(pnginfo, 'workflow'):
                workflow = pnginfo.workflow

            if workflow and isinstance(workflow, dict):
                # Build enriched LoRA lists with active + available state
                def _enrich_loras(lora_list, overrides_map, stack_prefix):
                    enriched = []
                    for lora in lora_list:
                        name = lora.get('name', '')
                        # Read active state from JS lora_overrides
                        ov = _resolve_lora_override(overrides_map, name, stack_prefix)
                        active = ov.get('active', lora.get('active', True)) if ov else lora.get('active', True)
                        enriched.append({
                            'name': name,
                            'path': lora.get('path', name),
                            'strength': float(ov.get('model_strength', lora.get('model_strength', 1.0))),
                            'clip_strength': float(ov.get('clip_strength', lora.get('clip_strength', 1.0))),
                            'active': active,
                            'available': lora_availability.get(name, True),
                        })
                    return enriched

                loras_a_enriched = _enrich_loras(
                    extracted.get('loras_a', []), lora_overrides,
                    'a'
                )
                loras_b_enriched = _enrich_loras(
                    extracted.get('loras_b', []), lora_overrides, 'b'
                )

                # Start from output_wf (v2), but strip runtime objects so
                # workflow metadata stays JSON-serializable for Save Image.
                extracted_data = to_json_safe_workflow_data(dict(output_wf))
                extracted_data['loras_a'] = loras_a_enriched
                extracted_data['loras_b'] = loras_b_enriched

                wf_nodes = workflow.get('nodes', [])
                for wf_node in wf_nodes:
                    if str(wf_node.get('id')) == str(unique_id):
                        wf_node['extracted_data'] = extracted_data
                        break

        # ── Push UI info to JS immediately (before generation) ─────────────
        # This makes the node feel responsive: LoRAs, model info, prompts
        # all appear while the model is still loading / sampling.
        try:
            server.PromptServer.instance.send_sync(
                "workflow-generator-pre-update",
                {"node_id": str(unique_id), "info": ui_info},
            )
        except Exception:
            pass  # Non-critical — JS will still get data from onExecuted

        # Cache last extracted info so downstream RecipeBuilder nodes can
        # pull this node's live state via /workflow-builder/get-extracted-data.
        if unique_id is not None:
            _last_workflow_builder_info[str(unique_id)] = ui_info.get('extracted', {})

        return {
            "ui":     {"workflow_info": [ui_info]},
            "result": (output_wf,),
        }


class WorkflowBuilderMulti(WorkflowBuilder):
    RETURN_TYPES = ("RECIPE_DATA",)
    RETURN_NAMES = ("recipe_data",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "recipe_data": ("RECIPE_DATA", {
                    "tooltip": "Optional recipe_data input for prefill/update.",
                }),
                "builder_data": ("BUILDER_DATA", {
                    "tooltip": "Optional A/B/C/D bundle of prompts, seeds, and LoRA stacks.",
                }),
                "override_data": ("STRING", {"default": "{}", "multiline": True}),
                "lora_state": ("STRING", {"default": "{}", "multiline": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    DESCRIPTION = (
        "Recipe Builder (Multi Model). Single builder UI with 4 model slots "
        "(A/B/C/D) and per-slot settings."
    )

    def execute(self,
                recipe_data=None,
                builder_data=None,
                override_data="{}", lora_state="{}",
                unique_id=None, extra_pnginfo=None, prompt=None):
        result = super().execute(
            recipe_data=recipe_data,
            builder_data=builder_data,
            override_data=override_data,
            lora_state=lora_state,
            unique_id=unique_id,
            extra_pnginfo=extra_pnginfo,
            prompt=prompt,
        )
        if isinstance(result, dict):
            raw = result.get("result", ())
            recipe_only = raw[:1] if isinstance(raw, tuple) else tuple(raw[:1]) if isinstance(raw, list) else ()
            return {
                "ui": result.get("ui", {}),
                "result": recipe_only,
            }
        if isinstance(result, tuple):
            return result[:1]
        if isinstance(result, list):
            return tuple(result[:1])
        return result
