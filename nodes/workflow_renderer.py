"""
Workflow Renderer — render-only node.

Accepts WORKFLOW_DATA (JSON string from Workflow Builder or PromptExtractor),
loads models, samples, decodes, and outputs IMAGE + LATENT.

No UI, no extraction — purely a render engine.
"""
import os
import json
import math
import traceback
import numpy as np
import torch
from PIL import Image as PILImage
import folder_paths
import comfy.sd
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.model_management

from ..py.workflow_families import (
    get_model_family,
    get_family_label,
    get_family_sampler_strategy,
    get_compatible_families,
)
from ..py.workflow_executor import (
    load_template,
    patch_template,
    loras_to_text,
    execute_template,
)
from ..py.workflow_extraction_utils import (
    resolve_model_name,
    resolve_vae_name,
    resolve_clip_names,
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

# ── Re-use sampler functions from the builder module ─────────────────────────
from .workflow_builder import (
    _run_standard_ksampler,
    _run_flux_sampler,
    _run_flux2_sampler,
    _run_wan_sampler,
    _load_model_from_path,
    _load_vae,
    _load_clip,
    _apply_loras,
)


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

        # ── Template-driven path (preferred) ──────────────────────────────
        api_template, wmap = load_template(family_key)
        has_both_stacks = bool(loras_a) and bool(loras_b)

        if api_template is not None:
            import copy
            api = copy.deepcopy(api_template)

            # Build lora override dicts from active/strength in workflow_data
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

            patch_params = {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "model_a": model_name_a,
                "model_b": model_name_b,
                "vae": vae_name,
                "width": width,
                "height": height,
                "batch_size": batch,
                "seed": sampler_params.get("seed_a", sampler_params.get("seed", 0)),
                "seed_b": seed_b,
                "cfg": sampler_params["cfg"],
                "sampler_name": sampler_params["sampler_name"],
                "scheduler": sampler_params["scheduler"],
                "denoise": 1.0,
                "guidance": sampler_params.get("guidance"),
                "lora_stack_a_text": loras_to_text(
                    loras_a, lora_overrides, "a" if has_both_stacks else ""
                ),
                "lora_stack_b_text": loras_to_text(
                    loras_b, lora_overrides, "b"
                ),
            }

            # Steps: WAN dual or standard
            if strategy == "wan_video":
                steps_high = wf_sampler.get("steps_high")
                steps_low = wf_sampler.get("steps_low")
                if steps_high is not None and steps_low is not None:
                    sh = int(steps_high)
                    sl = int(steps_low)
                else:
                    total = sampler_params["steps"]
                    sh = math.ceil(total / 2)
                    sl = total - sh
                patch_params["steps_high"] = sh
                patch_params["steps_low"] = sl
                if patch_params.get("seed_b") is None:
                    patch_params["seed_b"] = patch_params["seed"]
                print(f"[WorkflowRenderer] WAN dual-steps: high={sh}, low={sl}")
            else:
                patch_params["steps"] = sampler_params["steps"]

            if length is not None:
                patch_params["length"] = int(length)

            # Source image for i2v
            if source_image is not None and family_key == "wan_video_i2v":
                try:
                    input_dir = folder_paths.get_input_directory()
                    img_t = source_image
                    if isinstance(img_t, torch.Tensor):
                        while img_t.ndim > 3:
                            img_t = img_t.squeeze(0)
                        img_array = img_t.cpu().numpy()
                    else:
                        img_array = np.array(img_t)
                        while img_array.ndim > 3:
                            img_array = img_array[0]
                    img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
                    pil_img = PILImage.fromarray(img_array)
                    temp_name = "wg_i2v_source_image.png"
                    temp_path = os.path.join(input_dir, temp_name)
                    pil_img.save(temp_path)
                    patch_params["source_image_path"] = temp_name
                    print(f"[WorkflowRenderer] Source image saved: {temp_path}")
                except Exception as e:
                    print(f"[WorkflowRenderer] Failed to save source image: {e}")

            # CLIP names
            if clip_names:
                patch_params["clip"] = clip_names[0]
                if len(clip_names) > 1:
                    patch_params["clip_1"] = clip_names[0]
                    patch_params["clip_2"] = clip_names[1]

            patch_template(api, wmap, patch_params)

            print(f"[WorkflowRenderer] Template execution: family={family_key}")
            decoded, out_latent = execute_template(api, wmap, family_key, patch_params)

        else:
            # ── Fallback: hardcoded sampler paths ─────────────────────────
            print(f"[WorkflowRenderer] Hardcoded sampler fallback: {strategy}")

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

            # Build lora override dicts from active/strength in workflow_data
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
            model_a, clip = _apply_loras(model_a, clip, loras_a, lora_overrides, stack_key=stack_key_a)

            tokens_pos = clip.tokenize(positive_prompt)
            cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
            tokens_neg = clip.tokenize(negative_prompt)
            cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

            # Latent creation
            if strategy == "wan_video":
                L = int(length or 81)
                temporal = ((L - 1) // 4) + 1
                latent_tensor = torch.zeros(
                    [batch, 16, temporal, height // 8, width // 8],
                    device=comfy.model_management.intermediate_device(),
                )
            else:
                latent_tensor = torch.zeros(
                    [batch, 4, height // 8, width // 8],
                    device=comfy.model_management.intermediate_device(),
                )
            latent_dict = {
                "samples": latent_tensor,
                "downscale_ratio_spacial": 8,
                "_width": width,
                "_height": height,
            }

            # Model B for WAN dual-sampler
            if strategy == "wan_video" and model_name_b:
                resolved_b, folder_b = resolve_model_name(model_name_b)
                model_b_obj = clip_b = None
                if resolved_b:
                    full_path_b = folder_paths.get_full_path(folder_b, resolved_b)
                    _cache_key_b = (str(unique_id), full_path_b, family_key + "_b")
                    if _cache_key_b not in _cache:
                        print(f"[WorkflowRenderer] Loading model B: {resolved_b}")
                        _cache[_cache_key_b] = _load_model_from_path(resolved_b, folder_b, full_path_b)
                    else:
                        print(f"[WorkflowRenderer] Using cached model B: {resolved_b}")
                    model_b_obj, clip_b_raw, _ = _cache[_cache_key_b]
                    clip_b = clip_b_raw or clip
                    stack_key_b = "b" if has_both_stacks else ""
                    model_b_obj, clip_b = _apply_loras(
                        model_b_obj, clip_b, loras_b, lora_overrides, stack_key=stack_key_b,
                    )
                    tokens_pos_b = clip_b.tokenize(positive_prompt)
                    cond_pos_b = clip_b.encode_from_tokens_scheduled(tokens_pos_b)
                    tokens_neg_b = clip_b.tokenize(negative_prompt)
                    cond_neg_b = clip_b.encode_from_tokens_scheduled(tokens_neg_b)
                else:
                    model_b_obj = cond_pos_b = cond_neg_b = None

                total_steps = sampler_params["steps"]
                steps_high = int(wf_sampler.get("steps_high", math.ceil(total_steps / 2)))
                steps_low = int(wf_sampler.get("steps_low", total_steps - steps_high))

                wan_params_a = dict(sampler_params)
                wan_params_b = dict(sampler_params)
                wan_params_a["steps"] = steps_high
                wan_params_b["steps"] = steps_low

                samples = _run_wan_sampler(
                    model_a, cond_pos, cond_neg, latent_dict, wan_params_a,
                    model_b=model_b_obj, cond_pos_b=cond_pos_b, cond_neg_b=cond_neg_b,
                    sampler_params_b=wan_params_b,
                )
            elif strategy == "flux2":
                samples = _run_flux2_sampler(model_a, cond_pos, cond_neg, latent_dict, sampler_params)
            elif strategy == "flux":
                samples = _run_flux_sampler(model_a, cond_pos, cond_neg, latent_dict, sampler_params)
            else:
                samples = _run_standard_ksampler(model_a, cond_pos, cond_neg, latent_dict, sampler_params)

            print("[WorkflowRenderer] Decoding latent…")
            decoded = vae.decode(samples)
            if len(decoded.shape) == 5:
                decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
            out_latent = {"samples": samples}

        return (decoded, out_latent)
