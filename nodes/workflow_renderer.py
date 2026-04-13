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
from ..py.workflow_extraction_utils import resolve_model_name

# ── Re-use render functions from the builder module ──────────────────────────
from .workflow_builder import (
    _render_zimage,
    _render_qwen_image,
    _render_flux1,
    _render_flux2,
    _render_sdxl,
    _render_wan_image,
    _render_wan_video_t2v,
    _render_wan_video_i2v,
    _load_model_from_path,
    _load_vae,
    _load_clip,
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
