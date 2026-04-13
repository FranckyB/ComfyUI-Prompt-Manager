"""
ComfyUI Workflow Builder
Extracts ALL generation parameters from an image/video, provides a full UI
for editing them, and outputs WORKFLOW_DATA (JSON) for the Workflow Renderer
render node.

Part of ComfyUI-Prompt-Manager — shares extraction logic with PromptExtractor.
Family system and extraction helpers live in py/ for reuse.
"""
import os
import json
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
from ..py.lora_utils import resolve_lora_path

# ── Sampler/scheduler lists ──────────────────────────────────────────────────
SAMPLERS   = comfy.samplers.KSampler.SAMPLERS
SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS


# ─── API endpoints ──────────────────────────────────────────────────────────

@server.PromptServer.instance.routes.get("/workflow-extractor/list-models")
async def api_list_models(request):
    """List compatible models. Pass ?ref=<model> or ?family=<key>."""
    try:
        family_key = request.rel_url.query.get('family', '')
        ref        = request.rel_url.query.get('ref', '')

        if family_key:
            compat    = get_compatible_families(family_key)
            all_models = []
            for fn in ['checkpoints', 'diffusion_models', 'unet', 'unet_gguf']:
                try:
                    all_models.extend(folder_paths.get_filename_list(fn))
                except Exception:
                    continue
            seen   = set()
            models = []
            for m in all_models:
                if m not in seen:
                    seen.add(m)
                    if get_model_family(m) in compat:
                        models.append(m)
            family = family_key
        elif ref:
            models = list_compatible_models(ref)
            family = get_model_family(ref)
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
        print(f"[WorkflowBuilder] video-frame API error: {e}")
        return server.web.Response(status=500)


@server.PromptServer.instance.routes.get("/workflow-extractor/list-families")
async def api_list_families(request):
    """List all known model families for the type selector."""
    try:
        return server.web.json_response({"families": get_all_family_labels()})
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

# ─── Main Node ──────────────────────────────────────────────────────────────

class WorkflowBuilder:
    """
    Workflow Builder — UI and extraction node.

    Can run standalone with manual settings, or accept workflow_data (from
    PromptExtractor) and/or lora_stack inputs to pre-fill all parameters.

    Widget order:  Resolution → Model / VAE / CLIP → Prompts → Sampler → LoRAs
    Outputs WORKFLOW_DATA (JSON string) for the Workflow Renderer render node.
    """

    # Class-level cache so models persist across executions.
    # ComfyUI creates a new node instance on every queue run, so instance-level
    # caches are always empty.  Keyed by (unique_id, full_path, family_key) so
    # each canvas node has its own cache slot and model changes are detected.
    _class_model_cache: dict = {}

    def __init__(self):
        pass  # cache lives at class level — see _class_model_cache

    @classmethod
    def INPUT_TYPES(cls):
        # Family labels sorted alphabetically with SDXL default
        family_labels = get_all_family_labels()
        family_keys = sorted(family_labels.keys(), key=lambda k: family_labels[k].lower())
        # Move sdxl to front as default
        if "sdxl" in family_keys:
            family_keys.remove("sdxl")
            family_keys.insert(0, "sdxl")

        # Gather all models for initial model_a default
        all_models = []
        for fn in ['checkpoints', 'diffusion_models', 'unet', 'unet_gguf']:
            try:
                all_models.extend(folder_paths.get_filename_list(fn))
            except Exception:
                continue
        # Deduplicate
        seen = set()
        unique_models = []
        for m in all_models:
            if m not in seen:
                seen.add(m)
                unique_models.append(m)
        # Filter to SDXL-compatible for default
        sdxl_compat = get_compatible_families("sdxl")
        sdxl_models = sorted([m for m in unique_models if get_model_family(m) in sdxl_compat])
        first_model = sdxl_models[0] if sdxl_models else (sorted(unique_models)[0] if unique_models else "")

        return {
            "required": {
                "use_prompt_input": ("BOOLEAN", {
                    "default": False,
                    "label_on": "on",
                    "label_off": "off",
                    "tooltip": "When on, use the connected prompt input (overrides prompt from workflow_data). "
                               "When off, use the prompt from workflow_data or manual entry.",
                }),
                "use_lora_input": ("BOOLEAN", {
                    "default": False,
                    "label_on": "on",
                    "label_off": "off",
                    "tooltip": "When on, use LoRA stacks from connected inputs. "
                               "When off, use only the LoRAs shown in the node UI.",
                }),
                "use_workflow_data": ("BOOLEAN", {
                    "default": False,
                    "label_on": "on",
                    "label_off": "off",
                    "tooltip": "When on, populate all fields from connected workflow_data input. "
                               "When off, use manual values (keeps last loaded values).",
                }),
            },
            "optional": {
                # ── Connectable inputs ────────────────────────────────
                "positive_prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Positive prompt text. Overrides positive prompt from workflow_data when use_prompt_input is on.",
                }),
                "negative_prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Negative prompt text. Overrides negative prompt from workflow_data when use_prompt_input is on.",
                }),
                "lora_stack_a": ("LORA_STACK",),
                "lora_stack_b": ("LORA_STACK",),
                "workflow_data": ("WORKFLOW_DATA", {
                    "forceInput": True,
                    "lazy": True,
                    "tooltip": "Connect workflow_data output from PromptExtractor",
                }),
                # ── Hidden state widgets — written by JS, read by Python ──
                "override_data":   ("STRING", {"default": "{}", "multiline": True}),
                "lora_state":      ("STRING", {"default": "{}", "multiline": True}),
            },
            "hidden": {
                "unique_id":     "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt":        "PROMPT",
            },
        }

    RETURN_TYPES  = ("WORKFLOW_DATA",)
    RETURN_NAMES  = ("workflow_data",)
    FUNCTION      = "execute"
    CATEGORY      = "Prompt Manager"
    OUTPUT_NODE   = True
    DESCRIPTION   = (
        "Workflow Builder. Extracts parameters from images/workflows, provides "
        "a full editing UI, and outputs workflow_data for the Workflow Renderer."
    )

    def check_lazy_status(self, use_workflow_data=False, use_lora_input=False,
                          workflow_data=None, **kwargs):
        """Tell ComfyUI which lazy inputs need evaluation this run."""
        needed = []
        if use_workflow_data and workflow_data is None:
            needed.append('workflow_data')
        return needed

    def execute(self, use_prompt_input=False, use_workflow_data=False, use_lora_input=False,
                positive_prompt=None, negative_prompt=None, workflow_data=None,
                lora_stack_a=None, lora_stack_b=None,
                override_data="{}", lora_state="{}",
                unique_id=None, extra_pnginfo=None, prompt=None):
        """
        Main execution:
          1. Parse workflow_data or use defaults
          2. Apply JS overrides
          3. Merge lora inputs if enabled
          4. Build workflow_data JSON
          5. Return WORKFLOW_DATA
        """
        # ── Parse workflow_data input (if enabled and connected) ─────────
        wf_data = None
        if use_workflow_data and workflow_data:
            if isinstance(workflow_data, dict):
                wf_data = workflow_data
            elif isinstance(workflow_data, str):
                try:
                    wf_data = json.loads(workflow_data)
                except (json.JSONDecodeError, TypeError):
                    print("[WorkflowBuilder] Warning: could not parse workflow_data")

        # ── Build extracted dict from workflow_data or defaults ───────────
        if wf_data:
            wf_sampler = wf_data.get('sampler', {})
            wf_res = wf_data.get('resolution', {})
            extracted = {
                'positive_prompt': wf_data.get('positive_prompt', ''),
                'negative_prompt': wf_data.get('negative_prompt', ''),
                'loras_a': wf_data.get('loras_a', []),
                'loras_b': wf_data.get('loras_b', []),
                'model_a': wf_data.get('model_a', ''),
                'model_b': wf_data.get('model_b', ''),
                'vae':     {'name': wf_data.get('vae', ''), 'source': 'workflow_data'},
                'clip':    {
                    'names': wf_data.get('clip', []) if isinstance(wf_data.get('clip'), list)
                             else ([wf_data['clip']] if wf_data.get('clip') else []),
                    'type': wf_data.get('clip_type', ''), 'source': 'workflow_data',
                },
                'sampler': {
                    'steps': wf_sampler.get('steps', 20),
                    'cfg': wf_sampler.get('cfg', 5.0),
                    'seed_a': wf_sampler.get('seed_a', wf_sampler.get('seed', 0)),
                    'seed_b': wf_sampler.get('seed_b'),  # None = same as seed_a
                    'sampler_name': wf_sampler.get('sampler_name', 'euler'),
                    'scheduler': wf_sampler.get('scheduler', 'simple'),
                    'denoise': 1.0,
                    'guidance': wf_sampler.get('guidance'),
                },
                'resolution': {
                    'width': wf_res.get('width', 768),
                    'height': wf_res.get('height', 1280),
                    'batch_size': wf_res.get('batch_size', 1),
                    'length': wf_res.get('length'),
                    '_width_from_node_ref':  wf_res.get('_width_from_node_ref',  False),
                    '_height_from_node_ref': wf_res.get('_height_from_node_ref', False),
                },
                'is_video': wf_res.get('length') is not None,
                'model_family': wf_data.get('family', ''),
                'model_family_label': get_family_label(wf_data.get('family', '')),
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
                    'steps': 20, 'cfg': 5.0, 'seed_a': 0,
                    'sampler_name': 'euler', 'scheduler': 'simple',
                    'denoise': 1.0, 'guidance': None,
                },
                'resolution': {
                    'width': 768, 'height': 1280, 'batch_size': 1, 'length': None,
                },
                'is_video': False,
            }

        # ── Merge connected LoRA stacks if use_lora_input is enabled ─────
        if use_lora_input:
            if lora_stack_a:
                extracted['loras_a'] = [
                    {'name': name, 'model_strength': ms, 'clip_strength': cs}
                    for name, ms, cs in lora_stack_a
                ]
            if lora_stack_b:
                extracted['loras_b'] = [
                    {'name': name, 'model_strength': ms, 'clip_strength': cs}
                    for name, ms, cs in lora_stack_b
                ]

        # ── Parse overrides from JS ──────────────────────────────────────
        try:
            overrides = json.loads(override_data) if override_data else {}
        except json.JSONDecodeError:
            overrides = {}
        try:
            lora_overrides = json.loads(lora_state) if lora_state else {}
        except json.JSONDecodeError:
            lora_overrides = {}

        # ── Apply overrides ──────────────────────────────────────────────
        pos_text = overrides.get('positive_prompt', extracted['positive_prompt'])
        neg_text = overrides.get('negative_prompt', extracted['negative_prompt'])

        # ── Prompt input override (highest priority) ─────────────────────
        if use_prompt_input:
            if positive_prompt is not None:
                pos_text = positive_prompt
            if negative_prompt is not None:
                neg_text = negative_prompt
        model_name_a    = overrides.get('model_a', extracted['model_a'])
        model_name_b    = overrides.get('model_b', extracted['model_b'])
        vae_name        = overrides.get('vae', extracted['vae']['name'])

        sampler_params = extracted['sampler'].copy()
        sampler_params['denoise'] = 1.0
        for key in ['steps', 'cfg', 'seed_a', 'seed_b', 'sampler_name', 'scheduler']:
            if key in overrides:
                val = overrides[key]
                # Guard: overrides could carry a stale list from a corrupt
                # override_data blob — coerce numeric fields to scalar.
                if key in ('steps', 'seed_a', 'seed_b') and isinstance(val, list):
                    val = 0
                elif key == 'cfg' and isinstance(val, list):
                    val = 5.0
                sampler_params[key] = val
        # Also ensure extracted sampler seed/steps are never lists
        for key, default in (('seed_a', 0), ('seed_b', None), ('steps', 20), ('cfg', 5.0)):
            if isinstance(sampler_params.get(key), list):
                sampler_params[key] = default
        # WAN Video step overrides (steps_high / steps_low)
        if 'steps_high' in overrides:
            sampler_params['steps_high'] = int(overrides['steps_high'])
        if 'steps_low' in overrides:
            sampler_params['steps_low'] = int(overrides['steps_low'])

        # Family + strategy
        # wf_data['family'] is authoritative when workflow_data is driving the node
        # — it must take priority over the stale '_family' the JS serialised into
        # override_data (which may still say 'sdxl' if the async updateUI hadn't
        # finished before syncHidden ran).
        wf_family = wf_data.get('family', '') if wf_data else ''
        family_key = wf_family or overrides.get('_family') or extracted.get('model_family') or None
        if not family_key:
            resolved_ref, _ = resolve_model_name(model_name_a)
            family_key = get_model_family(resolved_ref or model_name_a)
        if not family_key and wf_data:
            clip_type = wf_data.get('clip_type', '').lower()
            loader_type = wf_data.get('loader_type', '')
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
            family_key = "sdxl"
        strategy = get_family_sampler_strategy(family_key)

        print(f"[WorkflowBuilder] Family: {get_family_label(family_key)} "
              f"(strategy={strategy}), model_a={model_name_a}, "
              f"model_b={model_name_b or '—'}")

        # ── Build workflow JSON + UI info BEFORE model loading ────────────
        # This ensures the JS UI is always populated, even if generation
        # fails (e.g. model not found).  The user can then edit settings
        # and re-queue.
        wf_overrides = dict(overrides)
        wf_overrides['_source'] = 'WorkflowBuilder'
        # Inject prompt input overrides so build_simplified_workflow_data uses them
        wf_overrides['positive_prompt'] = pos_text
        wf_overrides['negative_prompt'] = neg_text
        extracted['model_family'] = family_key
        extracted['model_family_label'] = get_family_label(family_key)

        # ── Apply lora_overrides: set active flag + update strengths ─────
        # Keep ALL loras (like PMA) but mark inactive ones with active=False.
        # Only _apply_loras filters out inactive when actually loading models.
        has_both = bool(extracted.get('loras_a')) and bool(extracted.get('loras_b'))
        for stack_key, list_key in [('a', 'loras_a'), ('b', 'loras_b')]:
            sk = stack_key if has_both else ''
            updated_list = []
            for lora in extracted.get(list_key, []):
                lora_name = lora.get('name', '')
                state_key = f"{sk}:{lora_name}" if sk else lora_name
                lora_st = lora_overrides.get(state_key, lora_overrides.get(lora_name, {}))
                updated = dict(lora)
                updated['active'] = lora_st.get('active', True) is not False
                if 'model_strength' in lora_st:
                    updated['model_strength'] = float(lora_st['model_strength'])
                if 'clip_strength' in lora_st:
                    updated['clip_strength'] = float(lora_st['clip_strength'])
                updated_list.append(updated)
            extracted[list_key] = updated_list

        simplified_wf = build_simplified_workflow_data(
            extracted, wf_overrides, sampler_params
        )

        # ── Check LoRA availability for JS display ────────────────────────
        lora_availability = {}
        for lora in extracted.get('loras_a', []) + extracted.get('loras_b', []):
            lora_name = lora.get('name', '')
            if lora_name and lora_name not in lora_availability:
                _, found = resolve_lora_path(lora_name)
                lora_availability[lora_name] = found

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
            compat_models = list_compatible_models(model_name_a)
            if compat_models:
                fallback = compat_models[0]
                print(f"[WorkflowBuilder] Model A '{model_name_a}' not found, workflow_data will use: {fallback}")
                simplified_wf['model_a'] = fallback

        if model_name_b and not model_b_found:
            compat_models = list_compatible_models(model_name_b)
            if compat_models:
                fallback = compat_models[0]
                print(f"[WorkflowBuilder] Model B '{model_name_b}' not found, workflow_data will use: {fallback}")
                simplified_wf['model_b'] = fallback

        if not vae_found and vae_name_str:
            vaes, recommended = list_compatible_vaes(family_key, return_recommended=True)
            fallback_vae = recommended or (vaes[0] if vaes else '')
            if fallback_vae:
                print(f"[WorkflowBuilder] VAE '{vae_name_str}' not found, workflow_data will use: {fallback_vae}")
                simplified_wf['vae'] = fallback_vae

        # CLIP fallback: check each clip name, replace not-found ones
        clip_names_out = simplified_wf.get('clip', [])
        if clip_names_out:
            clip_paths = resolve_clip_names(clip_names_out, simplified_wf.get('clip_type', ''))
            if any(p is None for p in clip_paths):
                compatible_clips = list_compatible_clips(family_key)
                if compatible_clips:
                    fixed_clips = []
                    for i, (name, path) in enumerate(zip(clip_names_out, clip_paths)):
                        if path is None and i < len(compatible_clips):
                            print(f"[WorkflowBuilder] CLIP '{name}' not found, workflow_data will use: {compatible_clips[i]}")
                            fixed_clips.append(compatible_clips[i])
                        else:
                            fixed_clips.append(name)
                    simplified_wf['clip'] = fixed_clips

        # LoRA fallback: remove not-found LoRAs from workflow_data output
        # (like PMA does) so downstream nodes never get broken LoRA names.
        for lora_key in ('loras_a', 'loras_b'):
            cleaned = [
                l for l in simplified_wf.get(lora_key, [])
                if lora_availability.get(l.get('name', ''), True)
            ]
            if len(cleaned) != len(simplified_wf.get(lora_key, [])):
                removed = [l.get('name') for l in simplified_wf.get(lora_key, [])
                           if not lora_availability.get(l.get('name', ''), True)]
                for r in removed:
                    print(f"[WorkflowBuilder] LoRA '{r}' not found, removed from workflow_data")
                simplified_wf[lora_key] = cleaned

        # ── Build UI info for JS (always, even if generation fails) ───────
        # Echo back the *effective* values (overrides applied) so the JS
        # pre-update handler sees what the user actually has set and does
        # NOT mistakenly treat it as a source change that clears all fields.
        effective_sampler = dict(extracted['sampler'])
        for key in ['steps', 'cfg', 'seed_a', 'sampler_name', 'scheduler']:
            if key in overrides:
                effective_sampler[key] = overrides[key]
        if 'steps_high' in overrides:
            effective_sampler['steps_high'] = overrides['steps_high']
        if 'steps_low' in overrides:
            effective_sampler['steps_low'] = overrides['steps_low']

        effective_resolution = dict(extracted['resolution'])
        for key in ['width', 'height', 'batch_size', 'length']:
            if key in overrides:
                effective_resolution[key] = overrides[key]

        effective_vae = extracted['vae']
        if overrides.get('vae'):
            effective_vae = {'name': overrides['vae'], 'source': 'override'}

        effective_clip = extracted['clip']
        if overrides.get('clip_names'):
            effective_clip = {'names': overrides['clip_names'], 'type': '', 'source': 'override'}

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
                'vae':                effective_vae,
                'vae_found':          vae_found,
                'clip':               effective_clip,
                'sampler':            effective_sampler,
                'resolution':         effective_resolution,
                'is_video':           extracted.get('is_video', False),
                'model_family':       family_key,
                'model_family_label': get_family_label(family_key),
                'lora_availability':  lora_availability,
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
                        key = f"{stack_prefix}:{name}" if stack_prefix else name
                        ov = overrides_map.get(key, {})
                        active = ov.get('active', True) if ov else True
                        enriched.append({
                            'name': name,
                            'path': lora.get('path', name),
                            'strength': float(ov.get('model_strength', lora.get('model_strength', 1.0))),
                            'clip_strength': float(ov.get('clip_strength', lora.get('clip_strength', 1.0))),
                            'active': active,
                            'available': lora_availability.get(name, True),
                        })
                    return enriched

                has_both = bool(extracted.get('loras_a')) and bool(extracted.get('loras_b'))
                loras_a_enriched = _enrich_loras(
                    extracted.get('loras_a', []), lora_overrides,
                    'a' if has_both else ''
                )
                loras_b_enriched = _enrich_loras(
                    extracted.get('loras_b', []), lora_overrides, 'b'
                )

                # Start from the already-built simplified_wf dict
                extracted_data = dict(simplified_wf)
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

        return {
            "ui":     {"workflow_info": [ui_info]},
            "result": (simplified_wf,),
        }
