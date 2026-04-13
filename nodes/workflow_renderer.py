"""
Workflow Renderer — render-only node.

Accepts WORKFLOW_DATA (JSON string from Workflow Builder or PromptExtractor),
loads models, samples, decodes, and outputs IMAGE + LATENT.

No UI, no extraction — purely a render engine.
"""
import json
import math
import torch
import folder_paths
import comfy.model_management

from ..py.workflow_families import (
    get_model_family,
    get_family_label,
    get_family_sampler_strategy,
    get_compatible_families,
)
from ..py.workflow_extraction_utils import (
    resolve_model_name,
    resolve_vae_name,
    resolve_clip_names,
)
from ..py.lora_utils import resolve_lora_path

class WorkflowRenderer:
    """
    Render-only generation node.

    Takes WORKFLOW_DATA (from Workflow Builder or PromptExtractor),
    loads models, applies LoRAs, samples, decodes.
    Outputs IMAGE + LATENT.
    """

    _class_model_cache: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_data": ("WORKFLOW_DATA", {
                    "forceInput": True,
                    "tooltip": "Connect workflow_data from Workflow Builder or PromptExtractor",
                }),
            },
            "optional": {
                "source_image": ("IMAGE", {
                    "tooltip": "Input image for WAN Video i2v (image-to-video) generation.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "execute"
    CATEGORY = "Prompt Manager"
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Render-only node. Accepts workflow_data, loads models, samples, "
        "and decodes. Outputs IMAGE + LATENT."
    )

    def execute(self, workflow_data, source_image=None, unique_id=None):
        # ── Parse workflow_data ───────────────────────────────────────────
        if isinstance(workflow_data, dict):
            wf = workflow_data
        elif isinstance(workflow_data, str):
            try:
                wf = json.loads(workflow_data)
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"[WorkflowRenderer] Invalid workflow_data: {e}")
        else:
            raise ValueError(f"[WorkflowRenderer] Invalid workflow_data type: {type(workflow_data)}")

        wf_sampler = wf.get("sampler", {})
        wf_res = wf.get("resolution", {})

        positive_prompt = wf.get("positive_prompt", "")
        negative_prompt = wf.get("negative_prompt", "")
        model_name_a = wf.get("model_a", "")
        model_name_b = wf.get("model_b", "") or None
        vae_name = wf.get("vae", "")
        family_key = wf.get("family", "")
        clip_names = wf.get("clip", [])
        if isinstance(clip_names, str):
            clip_names = [clip_names] if clip_names else []
        clip_type_str = wf.get("clip_type", "")
        loras_a = wf.get("loras_a", [])
        loras_b = wf.get("loras_b", [])

        # Sampler params
        sampler_params = {
            "steps": int(wf_sampler.get("steps", 20)),
            "cfg": float(wf_sampler.get("cfg", 5.0)),
            "seed_a": int(wf_sampler.get("seed_a", wf_sampler.get("seed", 0))),
            "sampler_name": wf_sampler.get("sampler_name", "euler"),
            "scheduler": wf_sampler.get("scheduler", "simple"),
            "denoise": 1.0,
            "guidance": wf_sampler.get("guidance"),
        }
        seed_b = wf_sampler.get("seed_b")

        # Resolution
        width = int(wf_res.get("width", 768))
        height = int(wf_res.get("height", 1280))
        batch = int(wf_res.get("batch_size", 1))
        length = wf_res.get("length")

        # ── Resolve family + strategy ─────────────────────────────────────
        if not family_key:
            resolved_ref, _ = resolve_model_name(model_name_a)
            family_key = get_model_family(resolved_ref or model_name_a)
        if not family_key:
            family_key = "sdxl"
        strategy = get_family_sampler_strategy(family_key)

        print(f"[WorkflowRenderer] Family: {get_family_label(family_key)} "
              f"(strategy={strategy}), model_a={model_name_a}, "
              f"model_b={model_name_b or '—'}")

        # ── Clamp resolution for i2v ──────────────────────────────────────
        if family_key == "wan_video_i2v":
            MAX_DIM = 1280
            if max(width, height) > MAX_DIM:
                scale = MAX_DIM / max(width, height)
                width = (round(width * scale) // 16) * 16
                height = (round(height * scale) // 16) * 16

        # ── Load Model A ──────────────────────────────────────────────────
        resolved_a, folder_a = resolve_model_name(model_name_a)
        if resolved_a is None:
            compat = get_compatible_families(family_key)
            all_on_disk = []
            for fn in ["checkpoints", "diffusion_models", "unet", "unet_gguf"]:
                try:
                    all_on_disk.extend(folder_paths.get_filename_list(fn))
                except Exception:
                    pass
            seen = set()
            fallbacks = []
            for m in all_on_disk:
                if m not in seen:
                    seen.add(m)
                    if get_model_family(m) in compat:
                        fallbacks.append(m)
            if fallbacks:
                model_name_a = sorted(fallbacks)[0]
                resolved_a, folder_a = resolve_model_name(model_name_a)
                print(f"[WorkflowRenderer] Using fallback model: {model_name_a}")
            else:
                raise FileNotFoundError(
                    f"Model A not found and no fallback for family {family_key}: {model_name_a}"
                )

        full_path_a = folder_paths.get_full_path(folder_a, resolved_a)
        _cache = WorkflowRenderer._class_model_cache
        _cache_key_a = (str(unique_id), full_path_a, family_key)
        if _cache_key_a not in _cache:
            print(f"[WorkflowRenderer] Loading model: {resolved_a}")
            _cache[_cache_key_a] = _load_model_from_path(resolved_a, folder_a, full_path_a)
        else:
            print(f"[WorkflowRenderer] Using cached model: {resolved_a}")
        model_a, clip_a, vae_a = _cache[_cache_key_a]
        has_both_stacks = bool(loras_a) and bool(loras_b)

        # ── Load VAE + CLIP ───────────────────────────────────────────────
        vae = _load_vae(vae_name, existing_vae=vae_a)
        if vae is None:
            raise FileNotFoundError(
                f"No VAE available (model={model_name_a}, vae={vae_name!r}). "
                f"Select a specific VAE in the Workflow Builder."
            )

        clip_info = {"names": clip_names, "type": clip_type_str, "source": "workflow_data"}
        clip = _load_clip(clip_info, {}, existing_clip=clip_a)
        if clip is None:
            raise FileNotFoundError("No CLIP available for text encoding")

        # ── Build LoRA overrides ──────────────────────────────────────────
        lora_overrides = {}
        for lora in loras_a:
            key = f"a:{lora['name']}" if has_both_stacks else lora["name"]
            lora_overrides[key] = {
                "active": lora.get("active", True),
                "model_strength": lora.get("strength", lora.get("model_strength", 1.0)),
                "clip_strength": lora.get("clip_strength", 1.0),
            }
        for lora in loras_b:
            key = f"b:{lora['name']}"
            lora_overrides[key] = {
                "active": lora.get("active", True),
                "model_strength": lora.get("strength", lora.get("model_strength", 1.0)),
                "clip_strength": lora.get("clip_strength", 1.0),
            }

        stack_key_a = "a" if has_both_stacks else ""
        stack_key_b = "b" if has_both_stacks else ""

        # ── Dispatch by family ────────────────────────────────────────────
        render_args = dict(
            model=model_a, clip=clip, vae=vae,
            pos_prompt=positive_prompt, neg_prompt=negative_prompt,
            width=width, height=height, batch=batch,
            sampler_params=sampler_params,
            loras_a=loras_a, lora_overrides=lora_overrides,
            lora_stack_key=stack_key_a,
        )

        # Unsupported families — error early with a clear message
        unsupported = ("ltxv",)
        if family_key in unsupported:
            raise ValueError(
                f"[WorkflowRenderer] Family '{family_key}' is not yet supported. "
                f"Unsupported families: {', '.join(unsupported)}"
            )

        if family_key == "zimage":
            decoded, out_latent = _render_zimage(**render_args)

        elif family_key == "qwen_image":
            decoded, out_latent = _render_qwen_image(**render_args)

        elif family_key == "flux1":
            decoded, out_latent = _render_flux1(**render_args)

        elif family_key == "flux2":
            decoded, out_latent = _render_flux2(**render_args)

        elif family_key in ("sdxl", "sd15"):
            decoded, out_latent = _render_sdxl(**render_args)

        elif family_key == "wan_image":
            decoded, out_latent = _render_wan_image(**render_args)

        elif family_key in ("wan_video_t2v", "wan_video_i2v"):
            # Load Model B for dual-sampler
            model_b_obj = None
            if model_name_b:
                resolved_b, folder_b = resolve_model_name(model_name_b)
                if resolved_b:
                    full_path_b = folder_paths.get_full_path(folder_b, resolved_b)
                    _cache_key_b = (str(unique_id), full_path_b, family_key + "_b")
                    if _cache_key_b not in _cache:
                        print(f"[WorkflowRenderer] Loading model B: {resolved_b}")
                        _cache[_cache_key_b] = _load_model_from_path(resolved_b, folder_b, full_path_b)
                    else:
                        print(f"[WorkflowRenderer] Using cached model B: {resolved_b}")
                    model_b_obj, _, _ = _cache[_cache_key_b]

            # WAN dual-sampler step counts
            total_steps = sampler_params["steps"]
            steps_high = int(wf_sampler.get("steps_high", math.ceil(total_steps / 2)))
            steps_low = int(wf_sampler.get("steps_low", total_steps - steps_high))
            sampler_params["steps_high"] = steps_high
            sampler_params["steps_low"] = steps_low
            if seed_b is not None:
                sampler_params["seed_b"] = int(seed_b)

            L = int(length or 81)

            if family_key == "wan_video_i2v":
                decoded, out_latent = _render_wan_video_i2v(
                    model_a=model_a, model_b=model_b_obj, clip=clip, vae=vae,
                    pos_prompt=positive_prompt, neg_prompt=negative_prompt,
                    width=width, height=height, length=L,
                    sampler_params=sampler_params,
                    source_image=source_image,
                    loras_a=loras_a, loras_b=loras_b,
                    lora_overrides=lora_overrides,
                    lora_stack_key_a=stack_key_a, lora_stack_key_b=stack_key_b,
                )
            else:
                decoded, out_latent = _render_wan_video_t2v(
                    model_a=model_a, model_b=model_b_obj, clip=clip, vae=vae,
                    pos_prompt=positive_prompt, neg_prompt=negative_prompt,
                    width=width, height=height, length=L,
                    sampler_params=sampler_params,
                    loras_a=loras_a, loras_b=loras_b,
                    lora_overrides=lora_overrides,
                    lora_stack_key_a=stack_key_a, lora_stack_key_b=stack_key_b,
                )

        else:
            raise ValueError(
                f"[WorkflowRenderer] Unsupported family '{family_key}'. "
                f"No render function available."
            )

        return (decoded, out_latent)


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

