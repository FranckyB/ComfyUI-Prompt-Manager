"""
ComfyUI Workflow Generator
Standalone generator node that accepts a WORKFLOW_DICT from WorkflowManager
(or a manually configured dict), loads all required models, applies LoRAs,
encodes prompts, samples (family-aware), decodes, and outputs IMAGE + LATENT + WORKFLOW_JSON.

When workflow_dict is connected: fully automatic — just hit Queue.
When workflow_dict is None: all fields are manually configurable via UI.
"""
import os
import json
import traceback
import torch
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
    get_all_family_labels,
    list_all_models,
    list_compatible_models,
    list_compatible_vaes,
    list_compatible_clips,
    MODEL_FAMILIES,
)
from ..py.workflow_extractor_utils import (
    resolve_model_name,
    resolve_vae_name,
    resolve_clip_names,
)

# ── Re-use sampling strategies from WorkflowExtractor ───────────────────────
from .workflow_extractor import (
    _load_model_from_path,
    _load_vae,
    _load_clip,
    _apply_loras,
    _run_standard_ksampler,
    _run_flux_sampler,
    _run_wan_sampler,
    _build_simplified_workflow_json,
)
from .prompt_extractor import resolve_lora_path

# ── Sampler / scheduler lists ────────────────────────────────────────────────
SAMPLERS   = comfy.samplers.KSampler.SAMPLERS
SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS


# ─── API endpoints ──────────────────────────────────────────────────────────

@server.PromptServer.instance.routes.get("/workflow-generator/list-families")
async def wg_list_families(request):
    return server.web.json_response({"families": get_all_family_labels()})

@server.PromptServer.instance.routes.get("/workflow-generator/list-models")
async def wg_list_models(request):
    family = request.rel_url.query.get('family', '') or None
    models = list_compatible_models(family) if family else list_all_models()
    return server.web.json_response({"models": models})

@server.PromptServer.instance.routes.get("/workflow-generator/list-vaes")
async def wg_list_vaes(request):
    family = request.rel_url.query.get('family', '') or None
    return server.web.json_response({"vaes": list_compatible_vaes(family)})

@server.PromptServer.instance.routes.get("/workflow-generator/list-clips")
async def wg_list_clips(request):
    family = request.rel_url.query.get('family', '') or None
    return server.web.json_response({"clips": list_compatible_clips(family)})


# ─── Main Node ──────────────────────────────────────────────────────────────

class WorkflowGenerator:
    """
    Workflow Generator — takes a WORKFLOW_DICT from WorkflowManager (or manual inputs)
    and generates images or videos with the correct family-aware sampling pipeline.

    Outputs: IMAGE, LATENT, WORKFLOW_JSON (string)
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get available models for the dropdowns
        all_models = list_all_models()
        if not all_models:
            all_models = [""]
        all_vaes = []
        try:
            all_vaes = sorted(folder_paths.get_filename_list("vae"))
        except Exception:
            pass
        all_clips = list_compatible_clips(None)

        family_labels = get_all_family_labels()
        family_keys   = ["auto"] + sorted(family_labels.keys())

        return {
            "required": {
                "family":       (family_keys, {"default": "auto"}),
                "model_a":      (["(from dict)"] + all_models, {"default": "(from dict)"}),
                "positive_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "placeholder": "Positive prompt (leave empty to use dict)",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "placeholder": "Negative prompt",
                }),
                "steps":    ("INT",   {"default": 20, "min": 1,  "max": 200}),
                "cfg":      ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "seed":     ("INT",   {"default": 0,  "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "sampler":  (SAMPLERS,   {"default": "euler"}),
                "scheduler":(SCHEDULERS, {"default": "normal"}),
                "denoise":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "width":    ("INT",   {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height":   ("INT",   {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1,  "min": 1,  "max": 128}),
            },
            "optional": {
                # When connected: all model/sampler/resolution fields come from the dict
                "workflow_dict": ("WORKFLOW_DICT", {}),
                # Additional LoRA stacks (merged with those in the dict)
                "lora_stack_a":  ("LORA_STACK", {}),
                "lora_stack_b":  ("LORA_STACK", {}),
                # Manual model B (for WAN — can be set from dict or here)
                "model_b":   (["(from dict)"] + all_models, {"default": "(from dict)"}),
                # Manual VAE / CLIP overrides
                "vae":  (["(from dict)", "(from checkpoint)"] + all_vaes, {"default": "(from dict)"}),
                "clip1": (["(from dict)", "(from checkpoint)"] + all_clips, {"default": "(from dict)"}),
                "clip2": (["none"]          + all_clips, {"default": "none"}),
                # Video length (frames)
                "length": ("INT", {"default": 0, "min": 0, "max": 1000,
                                   "tooltip": "Video frame count (0 = image mode)"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES  = ("IMAGE", "LATENT", "STRING")
    RETURN_NAMES  = ("image", "latent", "workflow_json")
    FUNCTION      = "generate"
    CATEGORY      = "FBnodes"
    OUTPUT_NODE   = True
    DESCRIPTION   = (
        "Workflow Generator — connect a WorkflowManager output (workflow_dict) for "
        "fully automatic generation, or configure manually. Supports Standard, Flux, "
        "Flux2, and WAN dual-sampler strategies."
    )

    def generate(
        self,
        family="auto",
        model_a="(from dict)",
        positive_prompt="",
        negative_prompt="",
        steps=20,
        cfg=7.0,
        seed=0,
        sampler="euler",
        scheduler="normal",
        denoise=1.0,
        width=512,
        height=512,
        batch_size=1,
        workflow_dict=None,
        lora_stack_a=None,
        lora_stack_b=None,
        model_b="(from dict)",
        vae="(from dict)",
        clip1="(from dict)",
        clip2="none",
        length=0,
        unique_id=None,
    ):
        """
        Generate an image or video using the family-aware sampling pipeline.

        Priority order (for each field):
          1. Explicit widget value (if not '(from dict)' / 'none')
          2. workflow_dict value
          3. Default / fallback
        """
        wd = workflow_dict or {}
        wf_cfg = wd.get("workflow_config", {}) or {}

        # ── Resolve family ────────────────────────────────────────────────
        family_key = None
        if family and family != "auto":
            family_key = family
        elif wf_cfg.get("family"):
            family_key = wf_cfg["family"]
        elif wd.get("family"):
            family_key = wd["family"]

        strategy = get_family_sampler_strategy(family_key)
        print(f"[WorkflowGenerator] Family: {get_family_label(family_key)}, strategy={strategy}")

        # ── Resolve model names ───────────────────────────────────────────
        model_name_a = None
        if model_a and model_a != "(from dict)":
            model_name_a = model_a
        elif wf_cfg.get("model_a"):
            model_name_a = os.path.basename(wf_cfg["model_a"].replace("\\", "/"))
        elif wd.get("model_a"):
            model_name_a = os.path.basename(wd["model_a"].replace("\\", "/"))

        model_name_b = None
        if model_b and model_b != "(from dict)":
            model_name_b = model_b
        elif wf_cfg.get("model_b"):
            model_name_b = os.path.basename(wf_cfg["model_b"].replace("\\", "/"))
        elif wd.get("model_b"):
            model_name_b = os.path.basename(wd["model_b"].replace("\\", "/"))

        if not model_name_a:
            raise RuntimeError("[WorkflowGenerator] No model specified. Connect a WorkflowManager or set model_a.")

        # ── Resolve prompts ───────────────────────────────────────────────
        pos_prompt = positive_prompt.strip() if positive_prompt else ""
        if not pos_prompt:
            pos_prompt = wd.get("prompt", "") or ""
        neg_prompt = negative_prompt.strip() if negative_prompt else ""
        if not neg_prompt:
            neg_prompt = wf_cfg.get("negative_prompt", "") or ""

        # ── Resolve sampler params ────────────────────────────────────────
        dict_sampler = wf_cfg.get("sampler") or wd.get("sampler") or {}
        sampler_params = {
            "steps":        steps,
            "cfg":          cfg,
            "seed":         seed,
            "sampler_name": sampler,
            "scheduler":    scheduler,
            "denoise":      denoise,
            "guidance":     dict_sampler.get("guidance"),
        }
        # If all sampler fields are at default values and dict has them, prefer dict
        if dict_sampler:
            if steps == 20  and dict_sampler.get("steps"):        sampler_params["steps"]        = int(dict_sampler["steps"])
            if cfg   == 7.0 and dict_sampler.get("cfg"):          sampler_params["cfg"]          = float(dict_sampler["cfg"])
            if seed  == 0   and dict_sampler.get("seed"):         sampler_params["seed"]         = int(dict_sampler["seed"])
            if sampler   == "euler"  and dict_sampler.get("sampler_name"):  sampler_params["sampler_name"] = dict_sampler["sampler_name"]
            if scheduler == "normal" and dict_sampler.get("scheduler"):     sampler_params["scheduler"]    = dict_sampler["scheduler"]
            if denoise   == 1.0     and dict_sampler.get("denoise") is not None: sampler_params["denoise"] = float(dict_sampler["denoise"])

        # ── Resolve resolution ────────────────────────────────────────────
        dict_res = wf_cfg.get("resolution") or wd.get("resolution") or {}
        if width == 512 and dict_res.get("width"):
            width = int(dict_res["width"])
        if height == 512 and dict_res.get("height"):
            height = int(dict_res["height"])
        if batch_size == 1 and dict_res.get("batch_size"):
            batch_size = int(dict_res["batch_size"])
        if length == 0 and dict_res.get("length"):
            length = int(dict_res["length"])

        # Use length as batch for video
        if length > 0:
            batch_size = length

        # ── Resolve VAE name ──────────────────────────────────────────────
        vae_name = None
        if vae and vae not in ("(from dict)", "(from checkpoint)"):
            vae_name = vae
        elif wf_cfg.get("vae"):
            vae_name = wf_cfg["vae"]
        elif wd.get("vae"):
            vae_name = wd["vae"]

        # ── Resolve CLIP names ────────────────────────────────────────────
        clip_names = []
        if clip1 and clip1 not in ("(from dict)", "(from checkpoint)", "none"):
            clip_names.append(clip1)
        if clip2 and clip2 != "none":
            clip_names.append(clip2)
        if not clip_names:
            dict_clip = wf_cfg.get("clip") or wd.get("clip") or []
            if isinstance(dict_clip, str):
                dict_clip = [dict_clip]
            clip_names = [c for c in dict_clip if c and not c.startswith("(")]

        # ── Build LoRA lists from dict + connected stacks ─────────────────
        dict_loras_a = wd.get("loras_a", [])
        dict_loras_b = wd.get("loras_b", [])

        # Convert connected LORA_STACK format to dict format
        def lora_stack_to_list(stack):
            if not stack:
                return []
            out = []
            for item in stack:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    name   = os.path.basename(item[0].replace("\\", "/"))
                    m_str  = float(item[1]) if len(item) > 1 else 1.0
                    c_str  = float(item[2]) if len(item) > 2 else m_str
                    out.append({"name": name, "model_strength": m_str, "clip_strength": c_str})
            return out

        extra_a = lora_stack_to_list(lora_stack_a)
        extra_b = lora_stack_to_list(lora_stack_b)
        all_loras_a = dict_loras_a + extra_a
        all_loras_b = dict_loras_b + extra_b

        # ── Load Model A ─────────────────────────────────────────────────
        resolved_a, folder_a = resolve_model_name(model_name_a)
        if resolved_a is None:
            raise RuntimeError(f"[WorkflowGenerator] Model A not found: {model_name_a}")
        full_path_a = folder_paths.get_full_path(folder_a, resolved_a)
        model_a_obj, clip_a_obj, vae_a_obj = _load_model_from_path(resolved_a, folder_a, full_path_a)
        print(f"[WorkflowGenerator] Model A loaded: {resolved_a}")

        # ── Load VAE ─────────────────────────────────────────────────────
        vae_obj = _load_vae(vae_name, existing_vae=vae_a_obj)
        if vae_obj is None:
            raise RuntimeError(f"[WorkflowGenerator] No VAE available. Model={model_name_a}, VAE={vae_name}")

        # ── Load CLIP ────────────────────────────────────────────────────
        clip_info    = {"names": clip_names, "type": "", "source": "separate" if clip_names else "checkpoint"}
        clip_overrides = {"clip_names": clip_names} if clip_names else {}
        clip_obj = _load_clip(clip_info, clip_overrides, existing_clip=clip_a_obj)
        if clip_obj is None:
            raise RuntimeError("[WorkflowGenerator] No CLIP available for encoding")

        # ── Apply LoRAs — Stack A ────────────────────────────────────────
        has_both = bool(all_loras_a) and bool(all_loras_b)
        model_a_obj, clip_obj = _apply_loras(
            model_a_obj, clip_obj, all_loras_a, {}, stack_key="a" if has_both else ""
        )

        # ── Encode prompts ───────────────────────────────────────────────
        tokens_pos = clip_obj.tokenize(pos_prompt)
        cond_pos   = clip_obj.encode_from_tokens_scheduled(tokens_pos)
        tokens_neg = clip_obj.tokenize(neg_prompt)
        cond_neg   = clip_obj.encode_from_tokens_scheduled(tokens_neg)

        # ── Create latent ────────────────────────────────────────────────
        print(f"[WorkflowGenerator] Latent: {width}x{height}, batch={batch_size}")
        latent_tensor = torch.zeros(
            [batch_size, 4, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        latent_dict = {
            "samples": latent_tensor,
            "downscale_ratio_spacial": 8,
            "_width":  width,
            "_height": height,
        }

        # ── Sample ───────────────────────────────────────────────────────
        print(f"[WorkflowGenerator] Strategy={strategy}, "
              f"steps={sampler_params['steps']}, cfg={sampler_params['cfg']}, "
              f"seed={sampler_params['seed']}, sampler={sampler_params['sampler_name']}, "
              f"scheduler={sampler_params['scheduler']}")

        if strategy == "wan" and model_name_b:
            resolved_b, folder_b = resolve_model_name(model_name_b)
            model_b_obj = cond_pos_b = cond_neg_b = None
            if resolved_b:
                full_path_b = folder_paths.get_full_path(folder_b, resolved_b)
                model_b_obj, clip_b_raw, _ = _load_model_from_path(resolved_b, folder_b, full_path_b)
                clip_b = clip_b_raw or clip_obj
                model_b_obj, clip_b = _apply_loras(
                    model_b_obj, clip_b, all_loras_b, {}, stack_key="b" if has_both else ""
                )
                tokens_pos_b = clip_b.tokenize(pos_prompt)
                cond_pos_b   = clip_b.encode_from_tokens_scheduled(tokens_pos_b)
                tokens_neg_b = clip_b.tokenize(neg_prompt)
                cond_neg_b   = clip_b.encode_from_tokens_scheduled(tokens_neg_b)
            else:
                print(f"[WorkflowGenerator] WAN model B not found: {model_name_b} — using single sampler")

            samples = _run_wan_sampler(
                model_a_obj, cond_pos, cond_neg, latent_dict, sampler_params,
                model_b=model_b_obj, cond_pos_b=cond_pos_b, cond_neg_b=cond_neg_b,
            )
        elif strategy in ("flux", "flux2"):
            samples = _run_flux_sampler(model_a_obj, cond_pos, cond_neg, latent_dict, sampler_params)
        else:
            samples = _run_standard_ksampler(model_a_obj, cond_pos, cond_neg, latent_dict, sampler_params)

        # ── Decode ───────────────────────────────────────────────────────
        print("[WorkflowGenerator] Decoding…")
        decoded    = vae_obj.decode(samples)
        out_latent = {"samples": samples}

        # ── Build output WORKFLOW_JSON ────────────────────────────────────
        fake_extracted = {
            "positive_prompt": pos_prompt, "negative_prompt": neg_prompt,
            "loras_a": all_loras_a, "loras_b": all_loras_b,
            "model_a": model_name_a, "model_b": model_name_b or "",
            "vae": {"name": vae_name or "", "source": "separate"},
            "clip": {"names": clip_names, "type": ""},
            "sampler": sampler_params,
            "resolution": {"width": width, "height": height,
                           "batch_size": batch_size, "length": length or None},
        }
        simplified = _build_simplified_workflow_json(
            model_name_a, fake_extracted, {}, sampler_params,
            family_key, strategy
        )
        wf_json_str = json.dumps(simplified, indent=2)

        return {
            "ui":     {"generated": [{"model": model_name_a, "family": get_family_label(family_key)}]},
            "result": (decoded, out_latent, wf_json_str),
        }
