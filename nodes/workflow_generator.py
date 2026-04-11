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
from ..py.workflow_executor import (
    load_template,
    patch_template,
    loras_to_text,
    execute_template,
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
        (model,) = ModelSamplingFlux().patch(model, max_shift=guidance, base_shift=0.5, width=width, height=height)
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
    t = clip_type_str.lower() if clip_type_str else ''
    # Check flux2 BEFORE flux — 'flux2' also contains 'flux' so order matters.
    # CLIPType.FLUX2 is required for Klein (Qwen text encoder); CLIPType.FLUX
    # is for Flux1 (T5-XXL + CLIP-L). Using the wrong type causes shape mismatch.
    if t and 'flux2' in t:
        clip_type = comfy.sd.CLIPType.FLUX2
    elif t and 'flux' in t:
        clip_type = comfy.sd.CLIPType.FLUX
    elif t and 'sd3' in t:
        clip_type = comfy.sd.CLIPType.SD3
    elif t and 'wan' in t:
        clip_type = comfy.sd.CLIPType.WAN  # umt5-xxl encoder (vocab 256384)
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
        model, clip = comfy.sd.load_lora_for_models(model, clip, lora_data, model_strength, clip_strength)
    return model, clip


# ─── Main Node ──────────────────────────────────────────────────────────────

class WorkflowGenerator:
    """
    One-stop generation node: load model, apply LoRAs, sample, decode.

    Can run standalone with manual settings, or accept workflow_data (from
    PromptExtractor) and/or lora_stack inputs to pre-fill all parameters.

    Widget order:  Resolution → Model / VAE / CLIP → Prompts → Sampler → LoRAs
    Supports family-aware sampling: Standard, Flux, Flux2, WAN dual-sampler.
    Outputs: IMAGE, LATENT, WORKFLOW_DATA (string).
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
                "lora_stack_a": ("LORA_STACK",),
                "lora_stack_b": ("LORA_STACK",),
                "workflow_data_input": ("WORKFLOW_DATA", {
                    "forceInput": True,
                    "lazy": True,
                    "tooltip": "Connect workflow_data output from PromptExtractor",
                }),
                "source_image": ("IMAGE", {
                    "tooltip": "Input image for WAN Video i2v (image-to-video) generation.",
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

    RETURN_TYPES  = ("IMAGE", "LATENT", "WORKFLOW_DATA")
    RETURN_NAMES  = ("image", "latent", "workflow_data")
    FUNCTION      = "execute"
    CATEGORY      = "FBnodes"
    OUTPUT_NODE   = False
    DESCRIPTION   = (
        "One-stop generation node. Accepts workflow_data from PromptExtractor "
        "and/or LoRA stacks, or works standalone with manual settings. "
        "Supports Standard, Flux, Flux2, and WAN dual-sampler strategies."
    )

    def check_lazy_status(self, use_workflow_data=False, use_lora_input=False,
                          workflow_data_input=None, **kwargs):
        """Tell ComfyUI which lazy inputs need evaluation this run."""
        needed = []
        if use_workflow_data and workflow_data_input is None:
            needed.append('workflow_data_input')
        return needed

    def execute(self, use_workflow_data=False, use_lora_input=False,
                workflow_data_input=None, lora_stack_a=None, lora_stack_b=None,
                source_image=None,
                override_data="{}", lora_state="{}",
                unique_id=None, extra_pnginfo=None, prompt=None):
        """
        Main execution:
          1. Parse workflow_data or use defaults
          2. Apply JS overrides
          3. Merge lora inputs if enabled
          4. Load model (+ VAE + CLIP)
          5. Apply LoRAs
          6. Encode prompts → Sample → Decode
          7. Return IMAGE, LATENT, WORKFLOW_DATA
        """
        # ── Parse workflow_data input (if enabled and connected) ─────────
        wf_data = None
        if use_workflow_data and workflow_data_input:
            try:
                wf_data = json.loads(workflow_data_input)
            except (json.JSONDecodeError, TypeError):
                print("[WorkflowGenerator] Warning: could not parse workflow_data_input")

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
                    'seed': wf_sampler.get('seed', 0),
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
                },
                'is_video': wf_res.get('length') is not None,
                'model_family': wf_data.get('family', ''),
                'model_family_label': wf_data.get('family_strategy', ''),
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
                    'steps': 20, 'cfg': 5.0, 'seed': 0,
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
        positive_prompt = overrides.get('positive_prompt', extracted['positive_prompt'])
        negative_prompt = overrides.get('negative_prompt', extracted['negative_prompt'])
        model_name_a    = overrides.get('model_a', extracted['model_a'])
        model_name_b    = overrides.get('model_b', extracted['model_b'])
        vae_name        = overrides.get('vae', extracted['vae']['name'])

        sampler_params = extracted['sampler'].copy()
        sampler_params['denoise'] = 1.0
        for key in ['steps', 'cfg', 'seed', 'sampler_name', 'scheduler']:
            if key in overrides:
                sampler_params[key] = overrides[key]
        # WAN Video step overrides (steps_high / steps_low)
        if 'steps_high' in overrides:
            sampler_params['steps_high'] = int(overrides['steps_high'])
        if 'steps_low' in overrides:
            sampler_params['steps_low'] = int(overrides['steps_low'])

        # Family + strategy
        family_key = overrides.get('_family') or extracted.get('model_family') or None
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
            # other: fall through to sdxl default — better than wrong family
        if not family_key:
            family_key = "sdxl"
        strategy = get_family_sampler_strategy(family_key)

        print(f"[WorkflowGenerator] Family: {get_family_label(family_key)} "
              f"(strategy={strategy}), model_a={model_name_a}, "
              f"model_b={model_name_b or '—'}")

        # ── Build workflow JSON + UI info BEFORE model loading ────────────
        # This ensures the JS UI is always populated, even if generation
        # fails (e.g. model not found).  The user can then edit settings
        # and re-queue.
        wf_overrides = dict(overrides)
        wf_overrides['_source'] = 'WorkflowGenerator'
        extracted['model_family'] = family_key
        extracted['model_family_label'] = strategy

        simplified_wf = build_simplified_workflow_data(
            extracted, wf_overrides, sampler_params
        )

        workflow_data_str = json.dumps(simplified_wf, indent=2)

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

        # ── Build UI info for JS (always, even if generation fails) ───────
        # Echo back the *effective* values (overrides applied) so the JS
        # pre-update handler sees what the user actually has set and does
        # NOT mistakenly treat it as a source change that clears all fields.
        effective_sampler = dict(extracted['sampler'])
        for key in ['steps', 'cfg', 'seed', 'sampler_name', 'scheduler']:
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
                'positive_prompt':    positive_prompt,
                'negative_prompt':    negative_prompt,
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

        # ── Load, sample, decode ──────────────────────────────────────────
        # The UI is always populated (ui_info built above).  If generation
        # fails the error is surfaced visibly on the node so the user can
        # adjust settings and re-queue.
        gen_error = None
        decoded = None
        out_latent = None

        try:
            # ── Load Model A ─────────────────────────────────────────────
            resolved_a, folder_a = resolve_model_name(model_name_a)
            if resolved_a is None:
                gen_error = f"Model A not found: {model_name_a}"
                raise FileNotFoundError(gen_error)
            full_path_a = folder_paths.get_full_path(folder_a, resolved_a)
            _cache = WorkflowGenerator._class_model_cache
            _cache_key_a = (str(unique_id), full_path_a, family_key)
            if _cache_key_a not in _cache:
                print(f"[WorkflowGenerator] Loading model (not cached): {resolved_a}")
                _cache[_cache_key_a] = _load_model_from_path(
                    resolved_a, folder_a, full_path_a
                )
            else:
                print(f"[WorkflowGenerator] Using cached model: {resolved_a}")
            model_a, clip_a, vae_a = _cache[_cache_key_a]

            # ── Resolution ───────────────────────────────────────────────
            res    = extracted['resolution']
            width  = int(overrides.get('width', res['width']))
            height = int(overrides.get('height', res['height']))
            length = overrides.get('length', res.get('length'))
            batch  = (
                int(length) if length is not None
                else int(overrides.get('batch_size', res.get('batch_size', 1)))
            )

            # ── Template-driven path (preferred) ─────────────────────────
            # Try to execute via workflow template. Falls back to hardcoded
            # samplers if no template exists for this family.
            # NOTE: VAE and CLIP loading is intentionally deferred to
            # execute_template (template path) or the hardcoded block below.
            # This avoids crashing when (Default) is selected on diffusion
            # model families that have no bundled VAE/CLIP in the checkpoint.
            api_template, wmap = load_template(family_key)

            # has_both_stacks is needed by both the template path (loras_to_text)
            # and the hardcoded fallback path (_apply_loras), so define it here.
            has_both_stacks = (
                bool(extracted['loras_a']) and bool(extracted['loras_b'])
            )

            if api_template is not None:
                import copy
                api = copy.deepcopy(api_template)

                # Build params for patch_template
                patch_params = {
                    'positive_prompt':   positive_prompt,
                    'negative_prompt':   negative_prompt,
                    'model_a':           model_name_a,
                    'model_b':           model_name_b or None,
                    'vae':               vae_name or '',
                    'width':             width,
                    'height':            height,
                    'batch_size':        batch,
                    'seed':              sampler_params.get('seed', 0),
                    'cfg':               sampler_params.get('cfg', 5.0),
                    'sampler_name':      sampler_params.get('sampler_name', 'euler'),
                    'scheduler':         sampler_params.get('scheduler', 'simple'),
                    'denoise':           sampler_params.get('denoise', 1.0),
                    'guidance':          sampler_params.get('guidance'),
                    'lora_stack_a_text': loras_to_text(
                        extracted['loras_a'], lora_overrides,
                        'a' if has_both_stacks else ''
                    ),
                    'lora_stack_b_text': loras_to_text(
                        extracted['loras_b'], lora_overrides, 'b'
                    ),
                }

                # Standard steps or WAN Video dual-steps
                if strategy == 'wan_video':
                    sh = sampler_params.get('steps_high', sampler_params.get('steps', 4))
                    sl = sampler_params.get('steps_low',  sampler_params.get('steps', 20))
                    patch_params['steps_high'] = sh
                    patch_params['steps_low']  = sl
                    print(f"[WorkflowGenerator] WAN Video dual-steps: high={sh}, low={sl}")
                else:
                    patch_params['steps'] = sampler_params.get('steps', 20)

                if length is not None:
                    patch_params['length'] = int(length)

                # Clip names from overrides or extracted
                clip_info = dict(extracted['clip'])
                if overrides.get('clip_names'):
                    patch_params['clip'] = overrides['clip_names'][0]
                elif clip_info.get('names'):
                    patch_params['clip'] = clip_info['names'][0]
                    if len(clip_info['names']) > 1:
                        patch_params['clip_1'] = clip_info['names'][0]
                        patch_params['clip_2'] = clip_info['names'][1]

                patch_template(api, wmap, patch_params)

                print(f"[WorkflowGenerator] Template execution: family={family_key}")
                decoded, out_latent = execute_template(
                    api, wmap, family_key,
                    source_image=source_image,
                )

            else:
                # ── Fallback: hardcoded sampler paths ────────────────────
                # For the hardcoded path we need VAE + CLIP loaded in Python.
                # Diffusion-model families (flux1, flux2, wan, ltxv, zimage)
                # return vae_a=None / clip_a=None from _load_model_from_path.
                # When (Default) is selected and no bundled VAE exists we must
                # not crash — instead surface a friendly error so the user can
                # pick a specific VAE/CLIP from the dropdown.
                print(f"[WorkflowGenerator] Hardcoded sampler fallback: {strategy}")

                vae = _load_vae(vae_name, existing_vae=vae_a)
                if vae is None:
                    gen_error = (
                        f"No VAE available — please select a specific VAE "
                        f"(model={model_name_a}, vae={vae_name!r}). "
                        f"(Default) requires a VAE bundled in the checkpoint, "
                        f"which diffusion-model families do not provide."
                    )
                    raise FileNotFoundError(gen_error)

                clip_info = dict(extracted['clip'])
                if not clip_info.get('type') and family_key:
                    from ..py.workflow_families import MODEL_FAMILIES
                    family_clip_type = MODEL_FAMILIES.get(
                        family_key, {}
                    ).get('clip_type', '')
                    if family_clip_type:
                        clip_info['type'] = family_clip_type
                clip = _load_clip(clip_info, overrides, existing_clip=clip_a)
                if clip is None:
                    gen_error = "No CLIP available for text encoding"
                    raise FileNotFoundError(gen_error)

                stack_key_a = "a" if has_both_stacks else ""
                model_a, clip = _apply_loras(
                    model_a, clip, extracted['loras_a'],
                    lora_overrides, stack_key=stack_key_a,
                )

                tokens_pos = clip.tokenize(positive_prompt)
                cond_pos   = clip.encode_from_tokens_scheduled(tokens_pos)
                tokens_neg = clip.tokenize(negative_prompt)
                cond_neg   = clip.encode_from_tokens_scheduled(tokens_neg)

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

                print(f"[WorkflowGenerator] Sampling strategy: {strategy}")
                print(
                    f"[WorkflowGenerator] steps={sampler_params['steps']}, "
                    f"cfg={sampler_params['cfg']}, seed={sampler_params['seed']}, "
                    f"sampler={sampler_params['sampler_name']}, "
                    f"scheduler={sampler_params['scheduler']}"
                )

                if strategy == "wan_video" and model_name_b:
                    resolved_b, folder_b = resolve_model_name(model_name_b)
                    model_b_obj = clip_b = None
                    if resolved_b:
                        full_path_b = folder_paths.get_full_path(folder_b, resolved_b)
                        _cache_key_b = (str(unique_id), full_path_b, family_key + "_b")
                        if _cache_key_b not in _cache:
                            print(f"[WorkflowGenerator] Loading model B: {resolved_b}")
                            _cache[_cache_key_b] = _load_model_from_path(
                                resolved_b, folder_b, full_path_b
                            )
                        else:
                            print(f"[WorkflowGenerator] Using cached model B: {resolved_b}")
                        model_b_obj, clip_b_raw, _ = _cache[_cache_key_b]

                        clip_b = clip_b_raw or clip
                        stack_key_b = "b" if has_both_stacks else ""
                        model_b_obj, clip_b = _apply_loras(
                            model_b_obj, clip_b, extracted['loras_b'],
                            lora_overrides, stack_key=stack_key_b,
                        )

                        tokens_pos_b = clip_b.tokenize(positive_prompt)
                        cond_pos_b = clip_b.encode_from_tokens_scheduled(tokens_pos_b)
                        tokens_neg_b = clip_b.tokenize(negative_prompt)
                        cond_neg_b = clip_b.encode_from_tokens_scheduled(tokens_neg_b)
                    else:
                        model_b_obj = cond_pos_b = cond_neg_b = None
                        print(
                            f"[WorkflowGenerator] WAN model B not found: "
                            f"{model_name_b} — using single sampler"
                        )

                    # Map steps_high/steps_low into sampler_params for _run_wan_sampler
                    wan_params_a = dict(sampler_params)
                    wan_params_b = dict(sampler_params)
                    if 'steps_high' in sampler_params:
                        wan_params_a['steps'] = sampler_params['steps_high']
                        wan_params_b['steps'] = sampler_params.get('steps_low', sampler_params['steps_high'])

                    samples = _run_wan_sampler(
                        model_a, cond_pos, cond_neg, latent_dict, wan_params_a,
                        model_b=model_b_obj,
                        cond_pos_b=cond_pos_b,
                        cond_neg_b=cond_neg_b,
                        sampler_params_b=wan_params_b,
                    )

                elif strategy == "flux2":
                    samples = _run_flux2_sampler(
                        model_a, cond_pos, cond_neg, latent_dict, sampler_params
                    )

                elif strategy == "flux":
                    samples = _run_flux_sampler(
                        model_a, cond_pos, cond_neg, latent_dict, sampler_params
                    )

                else:
                    samples = _run_standard_ksampler(
                        model_a, cond_pos, cond_neg, latent_dict, sampler_params
                    )

                # ── Decode ───────────────────────────────────────────────────
                print("[WorkflowGenerator] Decoding latent…")
                decoded = vae.decode(samples)
                out_latent = {"samples": samples}

        except FileNotFoundError:
            # Model/VAE/CLIP not found — gen_error already set above.
            # send_sync already fired so JS UI has the extracted data.
            raise
        except Exception as e:
            # send_sync already fired so JS UI has the extracted data.
            raise

        return {
            "ui":     {"workflow_info": [ui_info]},
            "result": (decoded, out_latent, workflow_data_str),
        }
