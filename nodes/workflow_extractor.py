"""
ComfyUI Workflow Extractor
Extracts ALL generation parameters from an image/video and regenerates internally.
Loads model, applies LoRAs, runs KSampler (family-aware), outputs IMAGE + LATENT + WORKFLOW_JSON.

Part of ComfyUI-Prompt-Manager — shares extraction logic with PromptExtractor.
Family system and extraction helpers live in py/ for reuse by WorkflowGenerator.
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
from ..py.workflow_extractor_utils import (
    extract_sampler_params,
    extract_vae_info,
    extract_clip_info,
    extract_resolution,
    resolve_model_name,
    resolve_vae_name,
    resolve_clip_names,
    extract_all_from_file,
    enrich_with_availability,
    build_simplified_workflow_data,
)

# ── Shared extraction functions from PromptExtractor ────────────────────────
from .prompt_extractor import (
    parse_workflow_for_prompts,
    extract_metadata_from_png,
    extract_metadata_from_jpeg,
    extract_metadata_from_json,
    extract_metadata_from_video,
    resolve_lora_path,
    build_node_map,
    build_link_map,
    extract_video_frame_av_to_tensor,
    get_cached_video_frame,
    get_placeholder_image_tensor,
)

# ── Optional GGUF support ────────────────────────────────────────────────────
try:
    from comfyui_gguf.nodes import load_gguf_unet as _load_gguf_unet
    GGUF_SUPPORT = True
except (ImportError, ModuleNotFoundError, AttributeError):
    try:
        import gguf_connector
        GGUF_SUPPORT = True
    except (ImportError, ModuleNotFoundError):
        GGUF_SUPPORT = False
    if not GGUF_SUPPORT:
        _load_gguf_unet = None

# ── Sampler/scheduler lists ──────────────────────────────────────────────────
SAMPLERS   = comfy.samplers.KSampler.SAMPLERS
SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS


# ─── API endpoints ──────────────────────────────────────────────────────────

@server.PromptServer.instance.routes.post("/workflow-extractor/extract")
async def api_extract(request):
    """Extract all parameters from a file and return to JS for UI display."""
    try:
        data     = await request.json()
        filename = data.get('filename', '')
        source   = data.get('source', 'input')

        if not filename:
            return server.web.json_response({"error": "No filename"}, status=400)

        base_dir = folder_paths.get_output_directory() if source == 'output' \
                   else folder_paths.get_input_directory()
        file_path = os.path.join(base_dir, filename.replace('/', os.sep))

        real_base = os.path.realpath(base_dir)
        real_path = os.path.realpath(file_path)
        if not real_path.startswith(real_base):
            return server.web.json_response({"error": "Invalid path"}, status=403)
        if not os.path.exists(file_path):
            return server.web.json_response({"error": "File not found"}, status=404)

        result = extract_all_from_file(file_path, source)
        enrich_with_availability(result)
        return server.web.json_response(result)
    except Exception as e:
        print(f"[WorkflowGenerator] API error: {e}")
        traceback.print_exc()
        return server.web.json_response({"error": str(e)}, status=500)


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

        base_dir  = folder_paths.get_output_directory() if source == 'output' \
                    else folder_paths.get_input_directory()
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
            try: os.unlink(tmp.name)
            except: pass

        return server.web.Response(status=500)
    except Exception as e:
        print(f"[WorkflowGenerator] video-frame API error: {e}")
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
        base_dir = folder_paths.get_output_directory() if source == 'output' \
                   else folder_paths.get_input_directory()
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


# ─── Internal sampling strategies ───────────────────────────────────────────

def _run_standard_ksampler(model, cond_pos, cond_neg, latent_dict, sampler_params):
    """
    Standard KSampler path — used for SD1.5, SDXL, Pony, Illustrious, SD3, LTX, etc.
    """
    import latent_preview

    steps        = int(sampler_params['steps'])
    cfg          = float(sampler_params['cfg'])
    seed         = int(sampler_params['seed'])
    sampler_name = sampler_params['sampler_name']
    scheduler    = sampler_params['scheduler']
    denoise      = float(sampler_params['denoise'])

    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_dict["samples"],
        latent_dict.get("downscale_ratio_spacial", None)
    )
    noise    = comfy.sample.prepare_noise(latent_image, seed)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        cond_pos, cond_neg, latent_image,
        denoise=denoise, disable_noise=False,
        start_step=None, last_step=None,
        force_full_denoise=False, noise_mask=None,
        callback=callback, disable_pbar=disable_pbar, seed=seed,
    )
    return samples


def _run_flux_sampler(model, cond_pos, cond_neg, latent_dict, sampler_params):
    """
    Flux-style sampler path — BasicGuider + SamplerCustomAdvanced.
    Used for Flux1, Z-Image, LTX-Video.
    """
    import latent_preview
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider, KSamplerSelect, BasicScheduler, RandomNoise, SamplerCustomAdvanced
    )

    steps        = int(sampler_params['steps'])
    seed         = int(sampler_params['seed'])
    sampler_name = sampler_params['sampler_name']
    scheduler    = sampler_params.get('scheduler', 'simple')
    denoise      = float(sampler_params.get('denoise', 1.0))
    guidance     = float(sampler_params.get('guidance') or 3.5)

    # Patch guidance into model via ModelSamplingFlux if needed
    try:
        from comfy_extras.nodes_model_advanced import ModelSamplingFlux
        width  = latent_dict.get('_width', 512)
        height = latent_dict.get('_height', 512)
        (model,) = ModelSamplingFlux().patch(model, max_shift=guidance, base_shift=0.5,
                                              width=width, height=height)
    except Exception:
        pass

    guider_node   = BasicGuider()
    sampler_node  = KSamplerSelect()
    scheduler_node = BasicScheduler()
    noise_node    = RandomNoise()
    custom_node   = SamplerCustomAdvanced()

    (guider,)    = guider_node.get_guider(model, cond_pos)
    (sampler,)   = sampler_node.get_sampler(sampler_name)
    (sigmas,)    = scheduler_node.get_sigmas(model, scheduler, steps, denoise)
    (noise_obj,) = noise_node.get_noise(seed)

    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_dict["samples"],
        latent_dict.get("downscale_ratio_spacial", None)
    )
    latent_in = {"samples": latent_image}

    out, out_denoised = custom_node.sample(noise_obj, guider, sampler, sigmas, latent_in)
    return out_denoised["samples"]


def _run_flux2_sampler(model, cond_pos, cond_neg, latent_dict, sampler_params):
    """
    Flux2/Klein: CFGGuider + BasicScheduler('beta') + SamplerCustomAdvanced.
    Uses CFGGuider (has CFG scale) like standard SD, but custom advanced sampler
    path like Flux1.  The ComfyUI Flux2Scheduler node is dimension-based and
    cannot be called programmatically here; BasicScheduler with 'beta' schedule
    is the correct equivalent.
    """
    import latent_preview
    from comfy_extras.nodes_custom_sampler import (
        CFGGuider, KSamplerSelect, BasicScheduler, SamplerCustomAdvanced, RandomNoise
    )

    steps        = int(sampler_params.get('steps', 20))
    cfg          = float(sampler_params.get('cfg', 1.0))
    seed         = int(sampler_params.get('seed', 0))
    sampler_name = sampler_params.get('sampler_name', 'euler')
    scheduler    = sampler_params.get('scheduler', 'beta')
    denoise      = float(sampler_params.get('denoise', 1.0))

    guider_node    = CFGGuider()
    sampler_node   = KSamplerSelect()
    scheduler_node = BasicScheduler()
    noise_node     = RandomNoise()
    custom_node    = SamplerCustomAdvanced()

    (guider,)    = guider_node.get_guider(model, cond_pos, cond_neg, cfg)
    (sampler,)   = sampler_node.get_sampler(sampler_name)
    (sigmas,)    = scheduler_node.get_sigmas(model, scheduler, steps, denoise)
    (noise_obj,) = noise_node.get_noise(seed)

    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_dict["samples"],
        latent_dict.get("downscale_ratio_spacial", None)
    )
    latent_in = {"samples": latent_image}
    out, out_denoised = custom_node.sample(noise_obj, guider, sampler, sigmas, latent_in)
    return out_denoised["samples"]


def _run_wan_sampler(model, cond_pos, cond_neg, latent_dict, sampler_params,
                     model_b=None, cond_pos_b=None, cond_neg_b=None,
                     sampler_params_b=None):
    """
    WAN 2.x dual-sampler path.
    model   / cond_pos   / cond_neg   = High (model A / stack A)
    model_b / cond_pos_b / cond_neg_b = Low  (model B / stack B)

    Falls back to single standard sampler if no model_b is provided.
    """
    import latent_preview

    # Apply ModelSamplingSD3 shift=5.0 to all WAN models before sampling
    try:
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        (model,) = ModelSamplingSD3().patch(model, shift=5.0, multiplier=1000)
    except Exception:
        pass

    # If no second model, fall back to standard path
    if model_b is None:
        return _run_standard_ksampler(model, cond_pos, cond_neg, latent_dict, sampler_params)

    # Apply shift to model B as well
    try:
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        (model_b,) = ModelSamplingSD3().patch(model_b, shift=5.0, multiplier=1000)
    except Exception:
        pass

    steps_high   = int(sampler_params.get('steps', 20))
    cfg_high     = float(sampler_params.get('cfg', 6.0))
    seed         = int(sampler_params.get('seed', 0))
    sampler_name = sampler_params.get('sampler_name', 'uni_pc')
    scheduler    = sampler_params.get('scheduler', 'simple')

    steps_low    = int((sampler_params_b or sampler_params).get('steps', steps_high))
    cfg_low      = float((sampler_params_b or sampler_params).get('cfg', cfg_high))

    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_dict["samples"],
        latent_dict.get("downscale_ratio_spacial", None)
    )

    # High pass
    noise = comfy.sample.prepare_noise(latent_image, seed)
    callback = latent_preview.prepare_callback(model, steps_high)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples_high = comfy.sample.sample(
        model, noise, steps_high, cfg_high, sampler_name, scheduler,
        cond_pos, cond_neg, latent_image,
        denoise=1.0, disable_noise=False,
        start_step=None, last_step=None,
        force_full_denoise=False, noise_mask=None,
        callback=callback, disable_pbar=disable_pbar, seed=seed,
    )

    # Low pass (refine with model B)
    latent_low = comfy.sample.fix_empty_latent_channels(
        model_b, samples_high,
        latent_dict.get("downscale_ratio_spacial", None)
    )
    noise_low    = comfy.sample.prepare_noise(latent_low, seed + 1)
    callback_low = latent_preview.prepare_callback(model_b, steps_low)

    samples_final = comfy.sample.sample(
        model_b, noise_low, steps_low, cfg_low, sampler_name, scheduler,
        cond_pos_b or cond_pos, cond_neg_b or cond_neg, latent_low,
        denoise=0.7, disable_noise=False,
        start_step=None, last_step=None,
        force_full_denoise=True, noise_mask=None,
        callback=callback_low, disable_pbar=disable_pbar, seed=seed + 1,
    )
    return samples_final


def _load_model_from_path(resolved_path, resolved_folder, full_model_path):
    """
    Load a model from disk. Returns (model, clip, vae).
    clip and vae are None for non-checkpoint model types.
    """
    is_gguf       = resolved_path.lower().endswith('.gguf')
    is_checkpoint = resolved_folder == 'checkpoints'

    model = clip = vae = None

    if is_gguf and GGUF_SUPPORT and _load_gguf_unet:
        print(f"[WorkflowGenerator] Loading GGUF model: {resolved_path}")
        model = _load_gguf_unet(full_model_path)
    elif is_checkpoint:
        print(f"[WorkflowGenerator] Loading checkpoint: {resolved_path}")
        out   = comfy.sd.load_checkpoint_guess_config(
            full_model_path, output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        model, clip, vae = out[0], out[1], out[2]
    else:
        print(f"[WorkflowGenerator] Loading diffusion model: {resolved_path}")
        # ComfyUI's load_diffusion_model internally calls load_torch_file
        # which always uses weights_only=True in PyTorch >= 2.6. Some model
        # formats (e.g. Klein .pt/.bin) contain non-standard objects that
        # fail with weights_only=True ("Unsupported operand 0"). In that
        # case we fall back to loading the state_dict with weights_only=False
        # and pass it directly to load_diffusion_model_state_dict.
        try:
            model = comfy.sd.load_diffusion_model(full_model_path)
        except Exception as e:
            if 'weights_only' in str(e).lower() or 'unpickling' in str(e).lower() or 'unsupported operand' in str(e).lower():
                print(f"[WorkflowGenerator] weights_only load failed, retrying with weights_only=False: {e}")
                import torch
                pl_sd = torch.load(full_model_path, map_location='cpu', weights_only=False)
                sd = pl_sd.get('state_dict', pl_sd)
                if hasattr(comfy.sd, 'load_diffusion_model_state_dict'):
                    model = comfy.sd.load_diffusion_model_state_dict(sd)
                else:
                    raise RuntimeError(
                        f"[WorkflowGenerator] Cannot load model '{resolved_path}': "
                        f"weights_only=False is needed but load_diffusion_model_state_dict "
                        f"is not available in this ComfyUI version. Update ComfyUI."
                    ) from e
            else:
                raise

    return model, clip, vae


def _load_vae(vae_name, existing_vae=None):
    """
    Load VAE by name. If existing_vae is provided and vae_name starts with '(',
    returns existing_vae unchanged.
    """
    if vae_name and not vae_name.startswith('('):
        vae_path = resolve_vae_name(vae_name)
        if vae_path:
            print(f"[WorkflowGenerator] Loading VAE: {vae_name}")
            sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
            v = comfy.sd.VAE(sd=sd, metadata=metadata)
            v.throw_exception_if_invalid()
            return v
    return existing_vae


def _load_clip(clip_info, overrides, existing_clip=None):
    """
    Load CLIP encoders. Returns loaded clip or existing_clip if from checkpoint.
    """
    if existing_clip is not None:
        # Checkpoint already provided CLIP — only override if explicitly set
        override_names = overrides.get('clip_names')
        if not override_names:
            return existing_clip

    clip_names  = overrides.get('clip_names') or clip_info.get('names', [])
    clip_paths  = resolve_clip_names(clip_names, clip_info.get('type', ''))
    valid_paths = [p for p in clip_paths if p is not None]

    if not valid_paths:
        return existing_clip

    clip_type_str = clip_info.get('type', '')
    if clip_type_str and 'flux' in clip_type_str.lower():
        clip_type = comfy.sd.CLIPType.FLUX
    elif clip_type_str and 'sd3' in clip_type_str.lower():
        clip_type = comfy.sd.CLIPType.SD3
    else:
        clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

    print(f"[WorkflowGenerator] Loading CLIP: {clip_names}")
    return comfy.sd.load_clip(
        ckpt_paths=valid_paths,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type,
    )


def _apply_loras(model, clip, loras, lora_overrides, stack_key=''):
    """
    Apply a list of LoRAs to model+clip. Respects active/strength overrides from JS.
    Returns updated (model, clip).
    """
    has_key = bool(stack_key)
    for lora in loras:
        lora_name = lora.get('name', '')
        state_key = f"{stack_key}:{lora_name}" if has_key else lora_name
        lora_st   = lora_overrides.get(state_key, lora_overrides.get(lora_name, {}))

        if lora_st.get('active') is False:
            print(f"[WorkflowGenerator] Skipping disabled LoRA: {lora_name}")
            continue

        model_strength = float(lora_st.get('model_strength', lora.get('model_strength', 1.0)))
        clip_strength  = float(lora_st.get('clip_strength',  lora.get('clip_strength',  1.0)))

        lora_path, found = resolve_lora_path(lora_name)
        if not found:
            print(f"[WorkflowGenerator] LoRA not found, skipping: {lora_name}")
            continue

        print(f"[WorkflowGenerator] Applying LoRA: {lora_name} "
              f"(model={model_strength:.2f}, clip={clip_strength:.2f})")
        lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model, clip = comfy.sd.load_lora_for_models(model, clip, lora_data,
                                                     model_strength, clip_strength)
    return model, clip


# ─── Main Node ──────────────────────────────────────────────────────────────

class WorkflowGenerator:
    """
    Extract generation parameters from an image/video and regenerate,
    OR generate from scratch using manually entered values (standalone mode).
    Shows model, LoRAs, VAE, CLIP, sampler settings with override controls.
    Supports family-aware sampling: Standard, Flux, Flux2, WAN dual-sampler.
    Outputs: IMAGE, LATENT, WORKFLOW_DATA (string).
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # Load only input directory files at startup (like PromptExtractor).
        # JS will refresh the list dynamically when source_folder changes.
        input_dir = folder_paths.get_input_directory()
        supported = {'.png', '.jpg', '.jpeg', '.webp', '.json', '.mp4', '.webm', '.mov', '.avi'}
        files = ["(none)"]
        if os.path.exists(input_dir):
            for root, dirs, filenames in os.walk(input_dir):
                for fn in filenames:
                    if os.path.splitext(fn)[1].lower() in supported:
                        rel = os.path.relpath(os.path.join(root, fn), input_dir).replace('\\', '/')
                        files.append(rel)
        files[1:] = sorted(files[1:])

        return {
            "required": {
                "source_folder": (["input", "output"], {"default": "input"}),
                "image":         (files, {"default": "(none)"}),
            },
            "optional": {
                # Hidden state widgets — written by JS, read by Python
                "override_data":   ("STRING", {"default": "{}", "multiline": True}),
                "lora_state":      ("STRING", {"default": "{}", "multiline": True}),
                "extracted_cache": ("STRING", {"default": "{}", "multiline": True}),
            },
            "hidden": {
                "unique_id":     "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt":        "PROMPT",
            },
        }

    RETURN_TYPES  = ("IMAGE", "LATENT", "STRING")
    RETURN_NAMES  = ("image", "latent", "workflow_data")
    FUNCTION      = "execute"
    CATEGORY      = "FBnodes"
    OUTPUT_NODE   = True
    DESCRIPTION   = (
        "Extract ALL generation parameters from an image/video and regenerate, "
        "or generate from scratch using manually entered values (standalone mode). "
        "Shows model, LoRAs, VAE, CLIP, sampler settings with override controls. "
        "Supports Standard, Flux, Flux2, and WAN dual-sampler strategies."
    )

    def execute(self, source_folder="input", image="", override_data="{}",
                lora_state="{}", extracted_cache="{}",
                unique_id=None, extra_pnginfo=None, prompt=None):
        """
        Main execution:
          1. Extract parameters from file (or use defaults in standalone mode)
          2. Apply overrides from JS UI
          3. Load model (+ VAE + CLIP)
          4. Apply LoRAs
          5. Encode prompts
          6. Create latent
          7. Sample (family-aware strategy)
          8. Decode
          9. Return IMAGE, LATENT, WORKFLOW_DATA
        """
        # ── Standalone mode: no file selected ───────────────────────────
        standalone = not image or image in ("", "(none)")

        # ── Extract all parameters (or use blank defaults) ───────────────
        if standalone:
            extracted = {
                'positive_prompt': '',
                'negative_prompt': '',
                'loras_a': [],
                'loras_b': [],
                'model_a': '',
                'model_b': '',
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
        else:
            base_dir  = folder_paths.get_output_directory() if source_folder == 'output' \
                        else folder_paths.get_input_directory()
            file_path = os.path.join(base_dir, image.replace('/', os.sep))
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"[WorkflowGenerator] File not found: {file_path}")
            extracted = extract_all_from_file(file_path, source_folder)

        # ── Parse overrides ──────────────────────────────────────────────
        try:
            overrides = json.loads(override_data) if override_data else {}
        except json.JSONDecodeError:
            overrides = {}
        try:
            lora_overrides = json.loads(lora_state) if lora_state else {}
        except json.JSONDecodeError:
            lora_overrides = {}

        # ── Apply overrides ──────────────────────────────────────────────
        positive_prompt = overrides.get('positive_prompt', extracted['positive_prompt'])
        negative_prompt = overrides.get('negative_prompt', extracted['negative_prompt'])
        model_name_a    = overrides.get('model_a', extracted['model_a'])
        model_name_b    = overrides.get('model_b', extracted['model_b'])
        vae_name        = overrides.get('vae', extracted['vae']['name'])

        sampler_params = extracted['sampler'].copy()
        # Denoise is always forced to 1.0 — not user-controllable
        sampler_params['denoise'] = 1.0
        for key in ['steps', 'cfg', 'seed', 'sampler_name', 'scheduler']:
            if key in overrides:
                sampler_params[key] = overrides[key]

        # Family + strategy
        family_key = overrides.get('_family') or None
        if not family_key:
            resolved_ref, _ = resolve_model_name(model_name_a)
            family_key = get_model_family(resolved_ref or model_name_a)
        strategy = get_family_sampler_strategy(family_key)

        print(f"[WorkflowGenerator] Family: {get_family_label(family_key)} "
              f"(strategy={strategy}), model_a={model_name_a}, model_b={model_name_b or '—'}")

        # ── Load Model A ─────────────────────────────────────────────────
        resolved_a, folder_a = resolve_model_name(model_name_a)
        if resolved_a is None:
            raise RuntimeError(f"[WorkflowGenerator] Model not found: {model_name_a}")
        full_path_a = folder_paths.get_full_path(folder_a, resolved_a)
        model_a, clip_a, vae_a = _load_model_from_path(resolved_a, folder_a, full_path_a)

        # ── Load VAE ─────────────────────────────────────────────────────
        vae = _load_vae(vae_name, existing_vae=vae_a)
        if vae is None:
            raise RuntimeError(
                f"[WorkflowGenerator] No VAE available. Model={model_name_a}, VAE={vae_name}"
            )

        # ── Load CLIP ────────────────────────────────────────────────────
        clip = _load_clip(extracted['clip'], overrides, existing_clip=clip_a)
        if clip is None:
            raise RuntimeError("[WorkflowGenerator] No CLIP available for text encoding")

        # ── Apply LoRAs (Stack A → model A) ──────────────────────────────
        has_both_stacks = bool(extracted['loras_a']) and bool(extracted['loras_b'])
        stack_key_a     = "a" if has_both_stacks else ""
        model_a, clip   = _apply_loras(model_a, clip, extracted['loras_a'],
                                        lora_overrides, stack_key=stack_key_a)

        # ── Encode prompts ───────────────────────────────────────────────
        tokens_pos = clip.tokenize(positive_prompt)
        cond_pos   = clip.encode_from_tokens_scheduled(tokens_pos)
        tokens_neg = clip.tokenize(negative_prompt)
        cond_neg   = clip.encode_from_tokens_scheduled(tokens_neg)

        # ── Resolution ───────────────────────────────────────────────────
        res    = extracted['resolution']
        width  = int(overrides.get('width',      res['width']))
        height = int(overrides.get('height',     res['height']))
        length = overrides.get('length', res.get('length'))
        batch  = int(length) if length is not None else int(overrides.get('batch_size', res.get('batch_size', 1)))

        # ── Create latent ────────────────────────────────────────────────
        print(f"[WorkflowGenerator] Latent: {width}x{height}, batch={batch}")
        latent_tensor = torch.zeros(
            [batch, 4, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        latent_dict = {
            "samples": latent_tensor,
            "downscale_ratio_spacial": 8,
            "_width":  width,
            "_height": height,
        }

        # ── Sample (family-aware) ─────────────────────────────────────────
        print(f"[WorkflowGenerator] Sampling strategy: {strategy}")
        print(f"[WorkflowGenerator] steps={sampler_params['steps']}, "
              f"cfg={sampler_params['cfg']}, seed={sampler_params['seed']}, "
              f"sampler={sampler_params['sampler_name']}, scheduler={sampler_params['scheduler']}")

        if strategy == "wan" and model_name_b:
            # WAN dual-sampler: load model B and stack B
            resolved_b, folder_b = resolve_model_name(model_name_b)
            model_b_obj = clip_b = None
            if resolved_b:
                full_path_b = folder_paths.get_full_path(folder_b, resolved_b)
                model_b_obj, clip_b_raw, _ = _load_model_from_path(resolved_b, folder_b, full_path_b)

                # Apply stack B LoRAs to model B
                clip_b = clip_b_raw or clip  # Use checkpoint clip or fall back to A's clip
                stack_key_b = "b" if has_both_stacks else ""
                model_b_obj, clip_b = _apply_loras(model_b_obj, clip_b, extracted['loras_b'],
                                                    lora_overrides, stack_key=stack_key_b)

                # Encode with clip B
                tokens_pos_b = clip_b.tokenize(positive_prompt)
                cond_pos_b   = clip_b.encode_from_tokens_scheduled(tokens_pos_b)
                tokens_neg_b = clip_b.tokenize(negative_prompt)
                cond_neg_b   = clip_b.encode_from_tokens_scheduled(tokens_neg_b)
            else:
                model_b_obj = cond_pos_b = cond_neg_b = None
                print(f"[WorkflowGenerator] WAN model B not found: {model_name_b} — using single sampler")

            samples = _run_wan_sampler(
                model_a, cond_pos, cond_neg, latent_dict, sampler_params,
                model_b=model_b_obj,
                cond_pos_b=cond_pos_b,
                cond_neg_b=cond_neg_b,
            )

        elif strategy == "flux2":
            samples = _run_flux2_sampler(model_a, cond_pos, cond_neg, latent_dict, sampler_params)

        elif strategy == "flux":
            samples = _run_flux_sampler(model_a, cond_pos, cond_neg, latent_dict, sampler_params)

        else:
            # Standard KSampler — SD1.5, SDXL, Pony, Illustrious, SD3, etc.
            samples = _run_standard_ksampler(model_a, cond_pos, cond_neg, latent_dict, sampler_params)

        # ── Build output latent ──────────────────────────────────────────
        out_latent = {"samples": samples}

        # ── Decode ───────────────────────────────────────────────────────
        print("[WorkflowGenerator] Decoding latent…")
        decoded = vae.decode(samples)

        # ── Build simplified workflow JSON ────────────────────────────────
        wf_overrides = dict(overrides)
        wf_overrides['_source'] = 'WorkflowGenerator'
        extracted['model_family']       = family_key
        extracted['model_family_label'] = strategy
        simplified_wf = build_simplified_workflow_data(extracted, wf_overrides, sampler_params)
        workflow_data_str = json.dumps(simplified_wf, indent=2)

        # ── Build UI info for JS ──────────────────────────────────────────
        ui_info = {
            'extracted': {
                'positive_prompt':   extracted['positive_prompt'],
                'negative_prompt':   extracted['negative_prompt'],
                'model_a':           extracted['model_a'],
                'model_b':           extracted['model_b'],
                'loras_a':           extracted['loras_a'],
                'loras_b':           extracted['loras_b'],
                'vae':               extracted['vae'],
                'clip':              extracted['clip'],
                'sampler':           extracted['sampler'],
                'resolution':        extracted['resolution'],
                'is_video':          extracted.get('is_video', False),
                'model_family':      family_key,
                'model_family_label': get_family_label(family_key),
            }
        }

        return {
            "ui":     {"workflow_info": [ui_info]},
            "result": (decoded, out_latent, workflow_data_str),
        }
