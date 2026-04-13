"""
Test Render Functions — self-contained ComfyUI test node.

Pick a family from the dropdown, hit Queue. The node auto-discovers the
first compatible model, VAE, and CLIP on disk, loads them, and runs the
full _render_* pipeline. Outputs IMAGE + LATENT for visual verification.
"""
import os
import torch
import folder_paths
import comfy.model_management
import comfy.sd
import comfy.utils

from ..py.workflow_families import (
    MODEL_FAMILIES,
    get_model_family,
    get_compatible_families,
    list_compatible_models,
    list_compatible_vaes,
    list_compatible_clips,
)
from ..py.workflow_extraction_utils import (
    resolve_model_name,
    resolve_vae_name,
    resolve_clip_names,
)
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

FAMILIES = sorted(MODEL_FAMILIES.keys())


class TestRenderFunctions:
    """
    Self-contained test node. Select a family → auto-loads the first
    compatible model / VAE / CLIP found on disk → runs the render pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "family": (FAMILIES, {"default": "sdxl"}),
                "positive_prompt": ("STRING", {
                    "default": "a photo of a cat sitting on a windowsill",
                    "multiline": True,
                }),
                "negative_prompt": ("STRING", {
                    "default": "blurry, low quality",
                    "multiline": True,
                }),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8,
                                  "tooltip": "Keep low for WAN video (512 recommended)"}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8,
                                   "tooltip": "Keep low for WAN video (512 recommended)"}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde", "uni_pc"],
                                 {"default": "euler"}),
                "scheduler": (["simple", "normal", "karras", "beta", "sgm_uniform"],
                              {"default": "simple"}),
            },
            "optional": {
                "length": ("INT", {"default": 33, "min": 1, "max": 257, "step": 4,
                                   "tooltip": "Video length (frames) for WAN video families"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "STRING")
    RETURN_NAMES = ("image", "latent", "report")
    FUNCTION = "execute"
    CATEGORY = "Prompt Manager/Test"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Self-contained test node. Select a family and queue — "
        "auto-discovers model/VAE/CLIP, runs the render pipeline, "
        "and outputs IMAGE + LATENT + a text report."
    )

    def execute(self, family, positive_prompt, negative_prompt,
                width, height, steps, cfg, seed, guidance, sampler_name, scheduler,
                length=33):

        report = []
        def log(msg):
            print(f"[TestRender] {msg}")
            report.append(msg)

        log(f"=== Testing family: {family} ===")

        # ── 1. Find first compatible model ────────────────────────────────
        compat = get_compatible_families(family)
        all_on_disk = []
        for fn in ["checkpoints", "diffusion_models", "unet", "unet_gguf"]:
            try:
                all_on_disk.extend(folder_paths.get_filename_list(fn))
            except Exception:
                pass
        seen = set()
        candidates = []
        for m in all_on_disk:
            if m not in seen:
                seen.add(m)
                if get_model_family(m) in compat:
                    candidates.append(m)
        if not candidates:
            raise FileNotFoundError(
                f"No compatible model found on disk for family '{family}'. "
                f"Compatible families: {compat}"
            )

        # WAN image uses low-quality models — filter out "high" models
        if family == "wan_image":
            low_candidates = [m for m in candidates if "high" not in m.lower()]
            if low_candidates:
                candidates = low_candidates

        # For WAN video families, explicitly pick HIGH and LOW models
        is_wan_video = family in ("wan_video_t2v", "wan_video_i2v")
        if is_wan_video:
            high_models = sorted([m for m in candidates if "high" in m.lower()])
            low_models  = sorted([m for m in candidates if "low" in m.lower()])
            if high_models:
                model_name = high_models[0]
            else:
                model_name = sorted(candidates)[0]
                log("WARNING: No 'high' model found — using first candidate")
        else:
            model_name = sorted(candidates)[0]

        log(f"Model A (high): {model_name}")

        resolved_a, folder_a = resolve_model_name(model_name)
        full_path_a = folder_paths.get_full_path(folder_a, resolved_a)
        model_a, clip_from_ckpt, vae_from_ckpt = _load_model_from_path(
            resolved_a, folder_a, full_path_a
        )
        comfy.model_management.soft_empty_cache()

        # ── 2. Resolve model B path (but DON'T load yet — saves VRAM) ────
        # Model B is loaded just before sampling so ComfyUI's memory
        # management can evict model A from GPU first.
        model_b_path_info = None  # (resolved, folder, full_path)
        if is_wan_video:
            if low_models:
                model_b_name = low_models[0]
                log(f"Model B (low): {model_b_name} (deferred load)")
                res_b, fld_b = resolve_model_name(model_b_name)
                path_b = folder_paths.get_full_path(fld_b, res_b)
                model_b_path_info = (res_b, fld_b, path_b)
            else:
                log("No 'low' model found — will reuse model A for B")

        # ── 3. Find VAE ──────────────────────────────────────────────────
        vae_list, rec_vae = list_compatible_vaes(family, return_recommended=True)
        vae_name = rec_vae or (vae_list[0] if vae_list else None)
        vae = _load_vae(vae_name, existing_vae=vae_from_ckpt)
        if vae is None:
            raise FileNotFoundError(f"No VAE found for family '{family}'")
        log(f"VAE: {vae_name or '(from checkpoint)'}")

        # ── 4. Find CLIP ─────────────────────────────────────────────────
        spec = MODEL_FAMILIES.get(family, {})
        clip_type_str = spec.get("clip_type", "")
        is_checkpoint = spec.get("checkpoint", False)
        clip_patterns = [p.lower() for p in spec.get("clip", [])]
        clip_slots = spec.get("clip_slots", 1)  # how many distinct CLIP files needed

        clip = None
        if not is_checkpoint and clip_patterns:
            clip_list = list_compatible_clips(family)
            if clip_slots >= 2:
                # Need one file per pattern (e.g. flux1: one t5xxl + one clip_l)
                selected = []
                for pat in clip_patterns:
                    for c in clip_list:
                        c_lower = os.path.basename(c).lower()
                        if pat in c_lower and c not in selected:
                            selected.append(c)
                            break
                    if len(selected) >= clip_slots:
                        break
            else:
                # Single encoder — pick first file matching patterns in order
                selected = []
                for pat in clip_patterns:
                    for c in clip_list:
                        c_lower = os.path.basename(c).lower()
                        if pat in c_lower:
                            selected = [c]
                            break
                    if selected:
                        break
                if not selected and clip_list:
                    selected = [clip_list[0]]
            if selected:
                log(f"CLIP files: {selected} type={clip_type_str or 'default'}")
                clip_info = {"names": selected, "type": clip_type_str, "source": "test"}
                clip = _load_clip(clip_info, {}, existing_clip=None)
        if clip is None:
            clip = clip_from_ckpt
        if clip is None:
            raise FileNotFoundError(f"No CLIP found for family '{family}'")
        log(f"CLIP loaded, type={clip_type_str or 'default'}")

        # ── 5. Build sampler params ──────────────────────────────────────
        # Apply family-specific defaults for cfg
        family_cfg_defaults = {
            "wan_image": 1.0, "wan_video_t2v": 1.0, "wan_video_i2v": 1.0,
            "flux1": 1.0,
        }
        effective_cfg = family_cfg_defaults.get(family, cfg)
        if family in family_cfg_defaults and cfg == 5.0:
            # Only override if user left cfg at default
            log(f"Using family default cfg={effective_cfg} (override with non-default value)")

        sampler_params = {
            "steps": steps,
            "cfg": effective_cfg,
            "seed_a": seed,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": 1.0,
            "guidance": guidance,
        }

        batch = 1
        common = dict(
            model=model_a, clip=clip, vae=vae,
            pos_prompt=positive_prompt, neg_prompt=negative_prompt,
            width=width, height=height, batch=batch,
            sampler_params=sampler_params,
        )

        log(f"Rendering {width}x{height}, steps={steps}, cfg={cfg}, guidance={guidance}")

        # ── 6. Dispatch ──────────────────────────────────────────────────
        unsupported = ("ltxv",)
        if family in unsupported:
            raise ValueError(
                f"Family '{family}' is not yet supported by the render pipeline. "
                f"Unsupported families: {', '.join(unsupported)}"
            )

        if family == "zimage":
            decoded, latent = _render_zimage(**common)

        elif family == "qwen_image":
            decoded, latent = _render_qwen_image(**common)

        elif family == "flux1":
            decoded, latent = _render_flux1(**common)

        elif family == "flux2":
            decoded, latent = _render_flux2(**common)

        elif family in ("sdxl", "sd15"):
            decoded, latent = _render_sdxl(**common)

        elif family == "wan_image":
            decoded, latent = _render_wan_image(**common)

        elif family in ("wan_video_t2v", "wan_video_i2v"):
            half = steps // 2 or 1
            sampler_params["steps_high"] = half
            sampler_params["steps_low"] = steps - half
            sampler_params["seed_b"] = seed + 1

            # Load model B NOW — deferred to keep VRAM free during CLIP encode
            model_b = None
            if model_b_path_info:
                log("Loading model B (low) …")
                res_b, fld_b, path_b = model_b_path_info
                model_b, _, _ = _load_model_from_path(res_b, fld_b, path_b)
                comfy.model_management.soft_empty_cache()
            else:
                model_b = model_a

            if family == "wan_video_i2v":
                source_image = torch.zeros(
                    [1, height, width, 3],
                    device=comfy.model_management.intermediate_device(),
                )
                log("Using blank source image for i2v test")
                decoded, latent = _render_wan_video_i2v(
                    model_a=model_a, model_b=model_b, clip=clip, vae=vae,
                    pos_prompt=positive_prompt, neg_prompt=negative_prompt,
                    width=width, height=height, length=length,
                    sampler_params=sampler_params, source_image=source_image,
                )
            else:
                decoded, latent = _render_wan_video_t2v(
                    model_a=model_a, model_b=model_b, clip=clip, vae=vae,
                    pos_prompt=positive_prompt, neg_prompt=negative_prompt,
                    width=width, height=height, length=length,
                    sampler_params=sampler_params,
                )

        else:
            raise ValueError(f"Unknown family: {family}")

        # ── 7. Cleanup — release models so ComfyUI can reclaim VRAM ──────
        del model_a, clip, vae
        if 'model_b' in dir():
            del model_b
        comfy.model_management.soft_empty_cache()

        log(f"OK  image={list(decoded.shape)}  latent={list(latent['samples'].shape)}")
        report_text = "\n".join(report)

        return {"ui": {"text": [report_text]}, "result": (decoded, latent, report_text)}
