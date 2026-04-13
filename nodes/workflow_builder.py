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


# ─── Internal sampling strategies ───────────────────────────────────────────

def _run_standard_ksampler(model, cond_pos, cond_neg, latent_dict, sampler_params):
    """
    Standard KSampler path — used for SD1.5, SDXL, Pony, Illustrious, SD3, LTX, etc.
    Returns:
        LATENT dict (same shape as input, with "samples" replaced).
    """
    import torch
    import latent_preview

    if not isinstance(latent_dict, dict) or "samples" not in latent_dict:
        raise TypeError("latent_dict must be a LATENT dict containing 'samples'.")

    steps       = int(sampler_params['steps'])
    cfg         = float(sampler_params['cfg'])
    seed        = int(sampler_params.get('seed_a', sampler_params.get('seed', 0)))
    sampler_name = sampler_params['sampler_name']
    scheduler    = sampler_params['scheduler']
    denoise      = float(sampler_params['denoise'])

    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_dict["samples"],
        latent_dict.get("downscale_ratio_spacial", None),
    )

    batch_inds = latent_dict["batch_index"] if "batch_index" in latent_dict else None
    noise_mask = latent_dict["noise_mask"] if "noise_mask" in latent_dict else None

    noise    = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        cond_pos, cond_neg, latent_image,
        denoise=denoise, disable_noise=False,
        start_step=None, last_step=None,
        force_full_denoise=False, noise_mask=noise_mask,
        callback=callback, disable_pbar=disable_pbar, seed=seed,
    )

    out = latent_dict.copy()
    out["samples"] = samples
    return out


def _render_zimage(model, clip, vae, pos_prompt, neg_prompt,
                   width, height, batch, sampler_params,
                   loras_a=None, lora_overrides=None, lora_stack_key=''):
    """
    Z-Image render — uses the exact same ComfyUI node classes as the
    original z_image.json workflow:

        UNETLoader  → PromptApplyLora → ModelSamplingAuraFlow → KSampler
        CLIPLoader  → CLIPTextEncode (pos / neg)  ──────────↗
        EmptySD3LatentImage  ───────────────────────────────↗
        VAELoader  → VAEDecode  ← ──────────────────KSampler

    Model, CLIP, and VAE are loaded by the caller (cached).
    Returns (IMAGE tensor, LATENT dict).
    """
    import torch
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

    # ── 1. Apply LoRAs (PromptApplyLora equivalent) ───────────────────────
    if loras_a:
        model, clip = _apply_loras(
            model, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key,
        )

    # ── 2. Encode prompts (CLIPTextEncode) ────────────────────────────────
    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos   = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg   = clip.encode_from_tokens_scheduled(tokens_neg)

    # ── 3. Create latent (EmptySD3LatentImage — 16 channels) ─────────────
    latent = {
        "samples": torch.zeros(
            [batch, 16, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        ),
    }

    # ── 4. Patch model (ModelSamplingAuraFlow, shift from workflow) ───────
    shift = float(sampler_params.get('guidance') or 3.0)
    (model,) = ModelSamplingAuraFlow().patch(model, shift)

    # ── 5. KSampler ──────────────────────────────────────────────────────
    seed         = int(sampler_params.get('seed_a', sampler_params.get('seed', 0)))
    steps        = int(sampler_params['steps'])
    cfg          = float(sampler_params['cfg'])
    sampler_name = sampler_params['sampler_name']
    scheduler    = sampler_params['scheduler']
    denoise      = float(sampler_params.get('denoise', 1.0))

    import latent_preview
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent["samples"])
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

    # ── 6. VAEDecode ─────────────────────────────────────────────────────
    print("[_render_zimage] Decoding latent…")
    decoded = vae.decode(samples)
    if len(decoded.shape) == 5:
        decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])

    return decoded, {"samples": samples}


def _run_zimage_sampler(model, cond_pos, cond_neg, latent_dict, sampler_params):
    """
    Z-Image sampler path — ModelSamplingAuraFlow + standard KSampler.
    Matches the z_image.json workflow exactly (16-channel latent).
    NOTE: This is only used as a sampler-step fallback. Prefer _render_zimage
    for the full pipeline.
    """
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
    import torch

    shift = float(sampler_params.get('guidance') or 3.0)
    (model,) = ModelSamplingAuraFlow().patch(model, shift)

    # Z-Image uses 16-channel latent (EmptySD3LatentImage).
    # If caller passed a 4-channel latent, recreate with 16 channels.
    s = latent_dict["samples"]
    if s.shape[1] == 4:
        s = torch.zeros(
            [s.shape[0], 16, s.shape[2], s.shape[3]],
            device=s.device, dtype=s.dtype,
        )
        latent_dict = dict(latent_dict)
        latent_dict["samples"] = s

    return _run_standard_ksampler(model, cond_pos, cond_neg, latent_dict, sampler_params)


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
    seed         = int(sampler_params.get('seed_a', sampler_params.get('seed', 0)))
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

    # V3 API: classmethods return NodeOutput — extract .args[0]
    guider    = BasicGuider.execute(model, cond_pos).args[0]
    sampler   = KSamplerSelect.execute(sampler_name).args[0]
    sigmas    = BasicScheduler.execute(model, scheduler, steps, denoise).args[0]
    noise_obj = RandomNoise.execute(seed).args[0]

    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_dict["samples"],
        latent_dict.get("downscale_ratio_spacial", None)
    )
    latent_in = {"samples": latent_image}

    sca_result = SamplerCustomAdvanced.execute(noise_obj, guider, sampler, sigmas, latent_in)
    return sca_result.args[1]["samples"]  # denoised_output


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
    seed         = int(sampler_params.get('seed_a', 0))
    sampler_name = sampler_params.get('sampler_name', 'euler')
    scheduler    = sampler_params.get('scheduler', 'beta')
    denoise      = float(sampler_params.get('denoise', 1.0))

    # V3 API: classmethods return NodeOutput — extract .args[0]
    guider    = CFGGuider.execute(model, cond_pos, cond_neg, cfg).args[0]
    sampler   = KSamplerSelect.execute(sampler_name).args[0]
    sigmas    = BasicScheduler.execute(model, scheduler, steps, denoise).args[0]
    noise_obj = RandomNoise.execute(seed).args[0]

    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_dict["samples"],
        latent_dict.get("downscale_ratio_spacial", None)
    )
    latent_in = {"samples": latent_image}
    sca_result = SamplerCustomAdvanced.execute(noise_obj, guider, sampler, sigmas, latent_in)
    return sca_result.args[1]["samples"]  # denoised_output


def _run_wan_sampler(model, cond_pos, cond_neg, latent_dict, sampler_params,
                     model_b=None, cond_pos_b=None, cond_neg_b=None):
    """
    WAN 2.x dual-sampler path.

    model   / cond_pos   / cond_neg   = High-noise model (model A)
    model_b / cond_pos_b / cond_neg_b = Low-noise model  (model B)

    Returns:
        LATENT dict
    """
    import torch
    import comfy
    import latent_preview

    if not isinstance(latent_dict, dict) or "samples" not in latent_dict:
        raise TypeError("latent_dict must be a LATENT dict containing a 'samples' tensor.")

    def _patch_wan_sampling(m):
        try:
            from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        except Exception:
            return m

        # Try the most conservative call first, then fall back if needed.
        try:
            patched = ModelSamplingSD3().patch(m, shift=5.0)
            if isinstance(patched, tuple):
                return patched[0]
            return patched
        except TypeError:
            patched = ModelSamplingSD3().patch(m, shift=5.0, multiplier=1000)
            if isinstance(patched, tuple):
                return patched[0]
            return patched

    model = _patch_wan_sampling(model)

    if model_b is None:
        return _run_standard_ksampler(model, cond_pos, cond_neg, latent_dict, sampler_params)

    model_b = _patch_wan_sampling(model_b)

    steps_high = int(sampler_params.get("steps_high", 2))
    steps_low = int(sampler_params.get("steps_low", steps_high))
    total_steps = steps_high + steps_low

    cfg = float(sampler_params.get("cfg", 1.0))
    seed_a = int(sampler_params.get("seed_a", 0))
    seed_b = int(sampler_params.get("seed_b", seed_a))
    sampler_name = sampler_params.get("sampler_name", "euler")
    scheduler = sampler_params.get("scheduler", "simple")

    print(
        f"[_run_wan_sampler] total_steps={total_steps}, "
        f"high=0→{steps_high}, low={steps_high}→{total_steps}, "
        f"cfg={cfg}, seed_a={seed_a}, seed_b={seed_b}"
    )

    # ---- High pass ----
    latent_image = latent_dict["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent_dict.get("downscale_ratio_spacial", None)
    )

    batch_inds = latent_dict["batch_index"] if "batch_index" in latent_dict else None
    noise_mask = latent_dict["noise_mask"] if "noise_mask" in latent_dict else None

    noise_high = comfy.sample.prepare_noise(latent_image, seed_a, batch_inds)
    callback_high = latent_preview.prepare_callback(model, total_steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples_high = comfy.sample.sample(
        model, noise_high, total_steps, cfg, sampler_name, scheduler,
        cond_pos, cond_neg, latent_image,
        denoise=1.0, disable_noise=False,
        start_step=0, last_step=steps_high,
        force_full_denoise=False, noise_mask=noise_mask,
        callback=callback_high, disable_pbar=disable_pbar, seed=seed_a,
    )

    # ---- Low pass ----
    latent_low = comfy.sample.fix_empty_latent_channels(
        model_b,
        samples_high,
        latent_dict.get("downscale_ratio_spacial", None)
    )

    # Match documented common_ksampler behavior for disable_noise=True
    noise_low = torch.zeros(
        latent_low.size(),
        dtype=latent_low.dtype,
        layout=latent_low.layout,
        device="cpu",
    )

    callback_low = latent_preview.prepare_callback(model_b, total_steps)

    samples_final = comfy.sample.sample(
        model_b, noise_low, total_steps, cfg, sampler_name, scheduler,
        cond_pos_b if cond_pos_b is not None else cond_pos,
        cond_neg_b if cond_neg_b is not None else cond_neg,
        latent_low,
        denoise=1.0, disable_noise=True,
        start_step=steps_high, last_step=total_steps,
        force_full_denoise=True, noise_mask=noise_mask,
        callback=callback_low, disable_pbar=disable_pbar, seed=seed_b,
    )

    out = latent_dict.copy()
    out["samples"] = samples_final
    return out


# ─── Dedicated render functions (one per family) ────────────────────────────
# Each function mirrors the exact node graph from the original workflow JSON.
# Model, CLIP, and VAE are loaded by the caller (cached).
# Returns (IMAGE tensor, LATENT dict).

def _render_qwen_image(model, clip, vae, pos_prompt, neg_prompt,
                       width, height, batch, sampler_params,
                       loras_a=None, lora_overrides=None, lora_stack_key=''):
    """
    Qwen Image render — mirrors qwen_image.json.
    """
    import torch
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

    if loras_a:
        model, clip = _apply_loras(model, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key)

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos   = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg   = clip.encode_from_tokens_scheduled(tokens_neg)

    latent = {
        "samples": torch.zeros(
            [batch, 16, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        ),
    }

    shift = float(sampler_params.get('guidance') or 3.1)

    # Be forgiving about patch signature
    msa = ModelSamplingAuraFlow()
    try:
        patched = msa.patch(model, shift)
    except TypeError:
        patched = msa.patch(model, shift=shift)
    if isinstance(patched, tuple):
        model = patched[0]
    else:
        model = patched

    latent_out = _run_standard_ksampler(model, cond_pos, cond_neg, latent, sampler_params)

    print("[_render_qwen_image] Decoding latent…")
    decoded = vae.decode(latent_out["samples"])
    if len(decoded.shape) == 5:
        decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
    return decoded, latent_out


def _render_flux1(model, clip, vae, pos_prompt, neg_prompt,
                  width, height, batch, sampler_params,
                  loras_a=None, lora_overrides=None, lora_stack_key=''):
    """
    Flux 1 render — mirrors flux_1.json.
    """
    import torch
    import node_helpers
    from nodes import ConditioningZeroOut

    if loras_a:
        model, clip = _apply_loras(model, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key)

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos   = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg   = clip.encode_from_tokens_scheduled(tokens_neg)

    # FluxGuidance: apply guidance scale to positive conditioning
    guidance = float(sampler_params.get('guidance') or 3.5)
    cond_pos = node_helpers.conditioning_set_values(cond_pos, {"guidance": guidance})

    # ConditioningZeroOut: zero out the negative conditioning
    (cond_neg,) = ConditioningZeroOut().zero_out(cond_neg)

    latent = {
        "samples": torch.zeros(
            [batch, 16, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        ),
    }

    latent_out = _run_standard_ksampler(model, cond_pos, cond_neg, latent, sampler_params)

    print("[_render_flux1] Decoding latent…")
    decoded = vae.decode(latent_out["samples"])
    if len(decoded.shape) == 5:
        decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
    return decoded, latent_out


def _render_flux2(model, clip, vae, pos_prompt, neg_prompt,
                  width, height, batch, sampler_params,
                  loras_a=None, lora_overrides=None, lora_stack_key=''):
    """
    Flux 2 render — mirrors flux_2.json.
    """
    import torch
    from comfy_extras.nodes_custom_sampler import (
        CFGGuider, KSamplerSelect, BasicScheduler, SamplerCustomAdvanced, RandomNoise,
    )

    if loras_a:
        model, clip = _apply_loras(model, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key)

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos   = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg   = clip.encode_from_tokens_scheduled(tokens_neg)

    latent = {
        "samples": torch.zeros(
            [batch, 16, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        ),
    }

    steps        = int(sampler_params.get('steps', 20))
    cfg          = float(sampler_params.get('cfg', 5.0))
    seed         = int(sampler_params.get('seed_a', sampler_params.get('seed', 0)))
    sampler_name = sampler_params.get('sampler_name', 'euler')
    scheduler    = sampler_params.get('scheduler', 'beta')
    denoise      = float(sampler_params.get('denoise', 1.0))

    # Handle NodeOutput vs tuple vs direct return
    def _first_arg(result):
        if hasattr(result, "args"):
            return result.args[0]
        if isinstance(result, tuple):
            return result[0]
        return result

    guider    = _first_arg(CFGGuider.execute(model, cond_pos, cond_neg, cfg))
    sampler   = _first_arg(KSamplerSelect.execute(sampler_name))
    sigmas    = _first_arg(BasicScheduler.execute(model, scheduler, steps, denoise))
    noise_obj = _first_arg(RandomNoise.execute(seed))

    latent_image = comfy.sample.fix_empty_latent_channels(model, latent["samples"])
    latent_in = {"samples": latent_image}

    sca_result = SamplerCustomAdvanced.execute(noise_obj, guider, sampler, sigmas, latent_in)
    if hasattr(sca_result, "args") and len(sca_result.args) > 1:
        out_denoised = sca_result.args[1]  # denoised_output
    elif isinstance(sca_result, tuple) and len(sca_result) > 1:
        out_denoised = sca_result[1]
    else:
        raise TypeError(
            f"SamplerCustomAdvanced returned unexpected value: {type(sca_result)}"
        )

    samples = out_denoised["samples"]

    print("[_render_flux2] Decoding latent…")
    decoded = vae.decode(samples)
    if len(decoded.shape) == 5:
        decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
    return decoded, {"samples": samples}


def _render_sdxl(model, clip, vae, pos_prompt, neg_prompt,
                 width, height, batch, sampler_params,
                 loras_a=None, lora_overrides=None, lora_stack_key=''):
    """
    SDXL render — mirrors sdxl.json (also used for SD1.5).
    """
    import torch

    if loras_a:
        model, clip = _apply_loras(model, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key)

    # CLIPSetLastLayer(-2) — standard for SDXL
    clip = clip.clone()
    clip.clip_layer(-2)

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos   = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg   = clip.encode_from_tokens_scheduled(tokens_neg)

    latent = {
        "samples": torch.zeros(
            [batch, 4, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        ),
    }

    latent_out = _run_standard_ksampler(model, cond_pos, cond_neg, latent, sampler_params)

    print("[_render_sdxl] Decoding latent…")
    decoded = vae.decode(latent_out["samples"])
    if len(decoded.shape) == 5:
        decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
    return decoded, latent_out


def _render_wan_image(model, clip, vae, pos_prompt, neg_prompt,
                      width, height, batch, sampler_params,
                      loras_a=None, lora_overrides=None, lora_stack_key=''):
    """
    WAN Image render — mirrors wan_image.json:
        CLIPLoader(umt5_xxl/wan) → CLIPTextEncode (pos / neg)
        UNETLoader → PromptApplyLora → KSampler
        EmptyLatentImage → KSampler → VAEDecode

    Note: wan_image.json uses two CLIPTextEncode nodes with the same
    prompt text (one for positive, one for negative).
    """
    import torch
    import comfy

    if loras_a:
        model, clip = _apply_loras(
            model, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key
        )

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    # WAN is video-native; for single image generation use a 5D latent with T=1.
    latent = {
        "samples": torch.zeros(
            [batch, 16, 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        ),
        "downscale_ratio_spacial": 8,
    }

    latent_out = _run_standard_ksampler(model, cond_pos, cond_neg, latent, sampler_params)
    print(f"[_render_wan_image] samples shape={latent_out['samples'].shape}")

    print("[_render_wan_image] Decoding latent…")

    # Prefer passing the latent tensor to the VAE API you are already using.
    # If your local WAN VAE wrapper wants the whole latent dict instead, swap this line.
    decoded = vae.decode(latent_out["samples"])
    print(f"[_render_wan_image] decoded shape={decoded.shape}")

    if len(decoded.shape) == 5:
        # WAN video-style decode path, flatten [B, T, H, W, C] → [B*T, H, W, C]
        decoded = decoded.reshape(
            -1,
            decoded.shape[-3],
            decoded.shape[-2],
            decoded.shape[-1],
        )

    return decoded, latent_out


def _render_wan_video_t2v(model_a, model_b, clip, vae, pos_prompt, neg_prompt,
                          width, height, length, sampler_params,
                          loras_a=None, loras_b=None,
                          lora_overrides=None, lora_stack_key_a='', lora_stack_key_b=''):
    """
    WAN Video T2V render.
    Returns:
        decoded_frames, latent_dict
    """

    # Optional LoRAs for model A / CLIP A
    if loras_a:
        model_a, clip = _apply_loras(
            model_a, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key_a
        )

    # Optional LoRAs for model B / CLIP B
    clip_b = clip
    if model_b and loras_b:
        model_b, clip_b = _apply_loras(
            model_b, clip_b, loras_b, lora_overrides or {}, stack_key=lora_stack_key_b
        )

    # Encode prompts for model A
    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    # Encode prompts for model B if it has a distinct CLIP
    if model_b and clip_b is not clip:
        tokens_pos_b = clip_b.tokenize(pos_prompt)
        cond_pos_b = clip_b.encode_from_tokens_scheduled(tokens_pos_b)
        tokens_neg_b = clip_b.tokenize(neg_prompt)
        cond_neg_b = clip_b.encode_from_tokens_scheduled(tokens_neg_b)
    else:
        cond_pos_b = cond_pos
        cond_neg_b = cond_neg

    # Create empty latent video, tolerant to either old-style generate() or V3-style execute()
    L = int(length or 81)
    from comfy_extras.nodes_hunyuan import EmptyHunyuanLatentVideo

    latent = None
    latent_node = EmptyHunyuanLatentVideo()

    if hasattr(latent_node, "generate"):
        latent = latent_node.generate(width=width, height=height, length=L, batch_size=1)[0]
    elif hasattr(EmptyHunyuanLatentVideo, "execute"):
        result = EmptyHunyuanLatentVideo.execute(
            width=width, height=height, length=L, batch_size=1
        )
        if isinstance(result, tuple):
            latent = result[0]
        elif hasattr(result, "result"):
            latent = result.result[0]
        elif hasattr(result, "outputs"):
            latent = result.outputs[0]
        else:
            latent = result
    else:
        raise RuntimeError("EmptyHunyuanLatentVideo has neither generate() nor execute().")

    if not isinstance(latent, dict) or "samples" not in latent:
        raise TypeError(
            f"EmptyHunyuanLatentVideo returned unexpected value: {type(latent)}"
        )

    latent_out = _run_wan_sampler(
        model_a, cond_pos, cond_neg, latent, sampler_params,
        model_b=model_b, cond_pos_b=cond_pos_b, cond_neg_b=cond_neg_b,
    )

    print("[_render_wan_video_t2v] Decoding latent…")

    latent_samples = latent_out["samples"]

    # Try the direct decode API first
    try:
        decoded = vae.decode(latent_samples)
    except Exception:
        # Some setups expect decode on a latent dict or a tiled decode path elsewhere
        decoded = vae.decode(latent_out["samples"])

    if len(decoded.shape) == 5:
        decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])

    return decoded, latent_out


def _render_wan_video_i2v(model_a, model_b, clip, vae, pos_prompt, neg_prompt,
                          width, height, length, sampler_params,
                          source_image=None,
                          loras_a=None, loras_b=None,
                          lora_overrides=None, lora_stack_key_a='', lora_stack_key_b=''):
    """
    WAN Video I2V render — mirrors wan_video_i2v.json.
    Returns:
        decoded_frames, latent_dict
    """

    # Optional LoRAs for model A / CLIP A
    if loras_a:
        model_a, clip = _apply_loras(
            model_a, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key_a
        )

    # Optional LoRAs for model B / CLIP B
    clip_b = clip
    if model_b and loras_b:
        model_b, clip_b = _apply_loras(
            model_b, clip_b, loras_b, lora_overrides or {}, stack_key=lora_stack_key_b
        )

    # Base prompt encodings (WanImageToVideo will re-use/modify these)
    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    # WanImageToVideo: encode source image into conditioning + latent
    L = int(length or 81)
    from comfy_extras.nodes_wan import WanImageToVideo as WanI2VNode

    # Be tolerant to V3 NodeOutput or direct tuple return
    i2v_result = WanI2VNode.execute(
        positive=cond_pos, negative=cond_neg, vae=vae,
        width=width, height=height, length=L, batch_size=1,
        start_image=source_image,
    )

    if hasattr(i2v_result, "args"):
        i2v_pos, i2v_neg, i2v_latent = i2v_result.args
    elif isinstance(i2v_result, tuple) and len(i2v_result) == 3:
        i2v_pos, i2v_neg, i2v_latent = i2v_result
    else:
        raise TypeError(
            f"WanImageToVideo returned unexpected value: {type(i2v_result)}"
        )

    # WanImageToVideo should return a LATENT dict for the third output
    latent = i2v_latent
    if not isinstance(latent, dict) or "samples" not in latent:
        raise TypeError(
            f"WanImageToVideo latent output has unexpected type: {type(latent)}"
        )

    # Both high and low samplers use WanImageToVideo’s modified conds
    cond_pos = i2v_pos
    cond_neg = i2v_neg
    cond_pos_b = i2v_pos
    cond_neg_b = i2v_neg

    # Dual KSamplerAdvanced via _run_wan_sampler (same as T2V)
    latent_out = _run_wan_sampler(
        model_a, cond_pos, cond_neg, latent, sampler_params,
        model_b=model_b, cond_pos_b=cond_pos_b, cond_neg_b=cond_neg_b,
    )

    print("[_render_wan_video_i2v] Decoding latent…")

    latent_samples = latent_out["samples"]

    # Try the straightforward tensor decode first
    try:
        decoded = vae.decode(latent_samples)
    except Exception:
        decoded = vae.decode(latent_out["samples"])

    if len(decoded.shape) == 5:
        decoded = decoded.reshape(
            -1,
            decoded.shape[-3],
            decoded.shape[-2],
            decoded.shape[-1],
        )

    return decoded, latent_out


def _load_model_from_path(resolved_path, resolved_folder, full_model_path):
    """
    Load a model from disk. Returns (model, clip, vae).
    clip and vae are None for non-checkpoint model types.
    """
    is_gguf       = resolved_path.lower().endswith('.gguf')
    is_checkpoint = resolved_folder == 'checkpoints'

    model = clip = vae = None

    if is_gguf and GGUF_SUPPORT and _load_gguf_unet:
        print(f"[WorkflowBuilder] Loading GGUF model: {resolved_path}")
        model = _load_gguf_unet(full_model_path)
    elif is_checkpoint:
        print(f"[WorkflowBuilder] Loading checkpoint: {resolved_path}")
        out   = comfy.sd.load_checkpoint_guess_config(
            full_model_path, output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        model, clip, vae = out[0], out[1], out[2]
    else:
        print(f"[WorkflowBuilder] Loading diffusion model: {resolved_path}")
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
                print(f"[WorkflowBuilder] weights_only load failed, retrying with weights_only=False: {e}")
                import torch
                pl_sd = torch.load(full_model_path, map_location='cpu', weights_only=False)
                sd = pl_sd.get('state_dict', pl_sd)
                if hasattr(comfy.sd, 'load_diffusion_model_state_dict'):
                    model = comfy.sd.load_diffusion_model_state_dict(sd)
                else:
                    raise RuntimeError(
                        f"[WorkflowBuilder] Cannot load model '{resolved_path}': "
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
            print(f"[WorkflowBuilder] Loading VAE: {vae_name}")
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
        clip_type = comfy.sd.CLIPType.WAN        # umt5-xxl encoder (vocab 256384)
    elif t and 'qwen_image' in t:
        clip_type = comfy.sd.CLIPType.QWEN_IMAGE  # Qwen2.5-VL image encoder
    elif t and 'lumina2' in t:
        clip_type = comfy.sd.CLIPType.LUMINA2     # z-image / Lumina2 encoder
    else:
        clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

    print(f"[WorkflowBuilder] Loading CLIP: {clip_names}")
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

        if lora_st.get('active') is False or lora.get('active') is False:
            print(f"[WorkflowBuilder] Skipping disabled LoRA: {lora_name}")
            continue

        model_strength = float(lora_st.get('model_strength', lora.get('model_strength', 1.0)))
        clip_strength  = float(lora_st.get('clip_strength',  lora.get('clip_strength',  1.0)))

        lora_path, found = resolve_lora_path(lora_name)
        if not found:
            print(f"[WorkflowBuilder] LoRA not found, skipping: {lora_name}")
            continue

        print(f"[WorkflowBuilder] Applying LoRA: {lora_name} "
              f"(model={model_strength:.2f}, clip={clip_strength:.2f})")
        lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model, clip = comfy.sd.load_lora_for_models(model, clip, lora_data, model_strength, clip_strength)
    return model, clip


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
