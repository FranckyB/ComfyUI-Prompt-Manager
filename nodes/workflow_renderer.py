"""
Workflow Renderer — render-only node.

Accepts WORKFLOW_DATA (JSON string from Workflow Builder or PromptExtractor),
loads models, samples, decodes, and outputs IMAGE.

No UI, no extraction — purely a render engine.
"""
import json
import math
import os
import torch
import folder_paths
import comfy.model_management
import comfy.sd

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

# Reuse the robust GGUF loading helper used by WorkflowModelLoader when available.
try:
    from .workflow_model_loader import _load_gguf_unet as _load_gguf_unet_shared
except Exception:
    _load_gguf_unet_shared = None

class WorkflowRenderer:
    """
    Render-only generation node.

    Takes WORKFLOW_DATA (from Workflow Builder or PromptExtractor),
    loads models, applies LoRAs, samples, decodes.
    Outputs IMAGE.
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

    RETURN_TYPES = ("WORKFLOW_DATA", "IMAGE")
    RETURN_NAMES = ("workflow_data", "output_image")
    FUNCTION = "execute"
    CATEGORY = "Prompt Manager"
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Render-only node. Accepts workflow_data, loads models, samples, "
        "and decodes. Outputs IMAGE + WORKFLOW_DATA (with MODEL/CLIP/VAE passthrough)."
    )

    @classmethod
    def IS_CHANGED(cls, workflow_data, source_image=None, **kwargs):
        """Return a stable fingerprint so ComfyUI skips re-execution when nothing changed."""
        import hashlib, json as _json
        h = hashlib.sha256()
        if isinstance(workflow_data, dict):
            h.update(_json.dumps(workflow_data, sort_keys=True, default=str).encode())
        elif isinstance(workflow_data, str):
            h.update(workflow_data.encode())
        if source_image is not None:
            h.update(str(source_image.shape).encode())
            h.update(str(source_image.sum().item()).encode())
        return h.hexdigest()

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

        wf_out = dict(wf)
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
            "steps": int(wf_sampler.get("steps_a", wf_sampler.get("steps", 20))),
            "cfg": float(wf_sampler.get("cfg", 5.0)),
            "seed_a": int(wf_sampler.get("seed_a", wf_sampler.get("seed", 0))),
            "sampler_name": wf_sampler.get("sampler_name", "euler"),
            "scheduler": wf_sampler.get("scheduler", "simple"),
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
        from ..py.workflow_families import MODEL_FAMILIES
        _family_spec = MODEL_FAMILIES.get(family_key, {})
        _family_is_ckpt = _family_spec.get('checkpoint', False)

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
        # Verify resolved model belongs to a compatible family — resolve_model_name
        # does basename matching which can match unrelated models (e.g. audio files).
        if resolved_a is not None:
            compat_check = get_compatible_families(family_key)
            resolved_family = get_model_family(resolved_a)
            if resolved_family is not None and resolved_family not in compat_check:
                print(f"[WorkflowRenderer] Resolved model {resolved_a} is family "
                      f"{resolved_family}, not compatible with {family_key} — rejecting")
                resolved_a, folder_a = None, None
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
            _cache[_cache_key_a] = _load_model_from_path(resolved_a, folder_a, full_path_a, family_is_checkpoint=_family_is_ckpt)
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
        model_b_obj = None

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
                        _cache[_cache_key_b] = _load_model_from_path(resolved_b, folder_b, full_path_b, family_is_checkpoint=_family_is_ckpt)
                    else:
                        print(f"[WorkflowRenderer] Using cached model B: {resolved_b}")
                    model_b_obj, _, _ = _cache[_cache_key_b]

            # WAN dual-sampler step counts
            total_steps = sampler_params["steps"]
            steps_high = int(wf_sampler.get("steps_a", wf_sampler.get("steps_high", math.ceil(total_steps / 2))))
            steps_low = int(wf_sampler.get("steps_b", wf_sampler.get("steps_low", total_steps - steps_high)))
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

        def _short_display_name(name):
            raw = str(name or "").strip()
            if not raw:
                return ""
            base = os.path.basename(raw.replace("\\", "/"))
            return os.path.splitext(base)[0]

        wf_out["MODEL_A"] = model_a
        if model_b_obj is not None:
            wf_out["MODEL_B"] = model_b_obj
        else:
            wf_out.pop("MODEL_B", None)
        wf_out["CLIP"] = clip
        wf_out["VAE"] = vae
        cond_pos_out, cond_neg_out = _encode_text_conditioning(clip, positive_prompt, negative_prompt)
        wf_out["POSITIVE"] = cond_pos_out
        wf_out["NEGATIVE"] = cond_neg_out
        wf_out["LATENT"] = out_latent
        wf_out["IMAGE"] = decoded
        wf_out["model_name"] = _short_display_name(resolved_a or model_name_a)

        return (wf_out, decoded)


def _encode_text_conditioning(clip, positive_prompt, negative_prompt):
    """
    Resolve positive/negative text conditioning from a CLIP encoder.
    Returns (positive_conditioning, negative_conditioning), each possibly None.
    """
    if clip is None:
        return None, None

    try:
        tokens_pos = clip.tokenize(str(positive_prompt or ""))
        cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
    except Exception as e:
        print(f"[WorkflowRenderer] Failed to encode positive conditioning: {e}")
        cond_pos = None

    try:
        tokens_neg = clip.tokenize(str(negative_prompt or ""))
        cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)
    except Exception as e:
        print(f"[WorkflowRenderer] Failed to encode negative conditioning: {e}")
        cond_neg = None

    return cond_pos, cond_neg

# ── Model loading ──────────────────────────────────────────────────

def _load_model_from_path(resolved_path, resolved_folder, full_model_path, family_is_checkpoint=None):
    """
    Load a model from disk. Returns (model, clip, vae).
    clip and vae are None for non-checkpoint model types.

    family_is_checkpoint: if explicitly False, load as diffusion model even if
    the file is in the checkpoints folder (e.g. Z-Image models stored there).
    """
    resolved_path_l = str(resolved_path or "").lower()
    full_path_l = str(full_model_path or "").lower()
    folder_l = str(resolved_folder or "").lower()
    is_gguf = any([
        resolved_path_l.endswith('.gguf'),
        full_path_l.endswith('.gguf'),
        folder_l == 'unet_gguf',
    ])
    # Family spec overrides folder-based heuristic when provided
    if family_is_checkpoint is not None:
        is_checkpoint = family_is_checkpoint
    else:
        is_checkpoint = resolved_folder == 'checkpoints'

    model = clip = vae = None

    if is_gguf:
        print(f"[WorkflowRenderer] Loading GGUF model: {resolved_path}")
        if _load_gguf_unet_shared is not None:
            model = _load_gguf_unet_shared(full_model_path)
        elif GGUF_SUPPORT and _load_gguf_unet:
            model = _load_gguf_unet(full_model_path)
        else:
            raise RuntimeError(
                "[WorkflowRenderer] GGUF model selected but no GGUF loader is available. "
                "Install/enable ComfyUI-GGUF (UnetLoaderGGUF)."
            )
    elif is_checkpoint:
        print(f"[WorkflowRenderer] Loading checkpoint: {resolved_path}")
        out   = comfy.sd.load_checkpoint_guess_config(
            full_model_path, output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        model, clip, vae = out[0], out[1], out[2]
    else:
        print(f"[WorkflowRenderer] Loading diffusion model: {resolved_path}")
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
                print(f"[WorkflowRenderer] weights_only load failed, retrying with weights_only=False: {e}")
                pl_sd = torch.load(full_model_path, map_location='cpu', weights_only=False)
                sd = pl_sd.get('state_dict', pl_sd)
                if hasattr(comfy.sd, 'load_diffusion_model_state_dict'):
                    model = comfy.sd.load_diffusion_model_state_dict(sd)
                else:
                    raise RuntimeError(
                        f"[WorkflowRenderer] Cannot load model '{resolved_path}': "
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
            print(f"[WorkflowRenderer] Loading VAE: {vae_name}")
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

    print(f"[WorkflowRenderer] Loading CLIP: {clip_names}")
    print(f"[WorkflowRenderer] clip_info={clip_info}, overrides={overrides}")
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
            print(f"[WorkflowRenderer] Skipping disabled LoRA: {lora_name}")
            continue

        model_strength = float(lora_st.get('model_strength', lora.get('model_strength', 1.0)))
        clip_strength  = float(lora_st.get('clip_strength',  lora.get('clip_strength',  1.0)))

        lora_path, found = resolve_lora_path(lora_name)
        if not found:
            print(f"[WorkflowRenderer] LoRA not found, skipping: {lora_name}")
            continue

        print(f"[WorkflowRenderer] Applying LoRA: {lora_name} "
              f"(model={model_strength:.2f}, clip={clip_strength:.2f})")
        lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model, clip = comfy.sd.load_lora_for_models(model, clip, lora_data, model_strength, clip_strength)
    return model, clip


# ── Rendering functions ──────────────────────────────────────────────────

def _extract_node_outputs(result):
    """
    Normalize node outputs across old-style tuples and V3 NodeOutput-ish objects.
    Returns a tuple.
    """
    if hasattr(result, "args"):
        return tuple(result.args)
    if hasattr(result, "result"):
        r = result.result
        return tuple(r) if isinstance(r, (list, tuple)) else (r,)
    if hasattr(result, "outputs"):
        r = result.outputs
        return tuple(r) if isinstance(r, (list, tuple)) else (r,)
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    return (result,)


def _decode_latent_output(vae, latent_out, tag="render"):
    """
    Decode a LATENT dict and flatten video outputs to image batch format if needed.
    """
    if not isinstance(latent_out, dict) or "samples" not in latent_out:
        raise TypeError(f"[{tag}] Expected LATENT dict with 'samples', got: {type(latent_out)}")

    print(f"[{tag}] Decoding latent…")
    decoded = vae.decode(latent_out["samples"])
    print(f"[{tag}] latent shape={latent_out['samples'].shape}")
    print(f"[{tag}] decoded shape={decoded.shape}")

    if len(decoded.shape) == 5:
        decoded = decoded.reshape(
            -1,
            decoded.shape[-3],
            decoded.shape[-2],
            decoded.shape[-1],
        )
    return decoded


def _make_latent(channels, width, height, batch=1, frames=None, spacial_ratio=None):
    """
    Create a LATENT dict.
    For image models: [B, C, H/8, W/8]
    For WAN/video-native models: [B, C, T, H/8, W/8]
    """
    import comfy

    if frames is None:
        samples = torch.zeros(
            [batch, channels, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
    else:
        samples = torch.zeros(
            [batch, channels, frames, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )

    latent = {"samples": samples}
    if spacial_ratio is not None:
        latent["downscale_ratio_spacial"] = spacial_ratio
    return latent


def _patch_model_sampling_sd3(model, shift=5.0):
    """
    Apply ModelSamplingSD3 if available.
    """
    try:
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
    except Exception:
        return model

    patcher = ModelSamplingSD3()
    try:
        result = patcher.patch(model, shift=shift)
    except TypeError:
        result = patcher.patch(model, shift=shift, multiplier=1000)

    outs = _extract_node_outputs(result)
    return outs[0]


def _patch_model_sampling_auraflow(model, shift=3.0):
    """
    Apply ModelSamplingAuraFlow using the aura-specific patch path.
    """
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

    patcher = ModelSamplingAuraFlow()

    if hasattr(patcher, "patch_aura"):
        result = patcher.patch_aura(model, shift)
    else:
        try:
            result = patcher.patch(model, shift=shift)
        except TypeError:
            result = patcher.patch(model, shift)

    outs = _extract_node_outputs(result)
    return outs[0]


def _run_standard_ksampler(model, cond_pos, cond_neg, latent_dict, sampler_params):
    """
    Standard KSampler path.
    Input: LATENT dict
    Output: LATENT dict
    Mirrors Comfy common_ksampler behavior.
    """
    import comfy
    import latent_preview

    if not isinstance(latent_dict, dict) or "samples" not in latent_dict:
        raise TypeError("latent_dict must be a LATENT dict containing 'samples'.")

    steps        = int(sampler_params["steps"])
    cfg          = float(sampler_params["cfg"])
    seed         = int(sampler_params.get("seed_a", sampler_params.get("seed", 0)))
    sampler_name = sampler_params["sampler_name"]
    scheduler    = sampler_params["scheduler"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_dict["samples"],
        latent_dict.get("downscale_ratio_spacial", None),
    )

    batch_inds = latent_dict["batch_index"] if "batch_index" in latent_dict else None
    noise_mask = latent_dict["noise_mask"] if "noise_mask" in latent_dict else None

    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        cond_pos, cond_neg, latent_image,
        denoise=1.0, disable_noise=False,
        start_step=None, last_step=None,
        force_full_denoise=False, noise_mask=noise_mask,
        callback=callback, disable_pbar=disable_pbar, seed=seed,
    )

    out = latent_dict.copy()
    out["samples"] = samples
    return out


def _run_wan_sampler(model, cond_pos, cond_neg, latent_dict, sampler_params,
                     model_b=None, cond_pos_b=None, cond_neg_b=None):
    """
    WAN 2.x dual-sampler path.
    Input: LATENT dict
    Output: LATENT dict
    """
    import comfy
    import latent_preview

    if not isinstance(latent_dict, dict) or "samples" not in latent_dict:
        raise TypeError("latent_dict must be a LATENT dict containing 'samples'.")

    model = _patch_model_sampling_sd3(model, shift=5.0)

    if model_b is None:
        return _run_standard_ksampler(model, cond_pos, cond_neg, latent_dict, sampler_params)

    model_b = _patch_model_sampling_sd3(model_b, shift=5.0)

    steps_high  = int(sampler_params.get("steps_high", 2))
    steps_low   = int(sampler_params.get("steps_low", steps_high))
    total_steps = steps_high + steps_low

    cfg          = float(sampler_params.get("cfg", 1.0))
    seed_a       = int(sampler_params.get("seed_a", 0))
    seed_b       = int(sampler_params.get("seed_b", seed_a))
    sampler_name = sampler_params.get("sampler_name", "euler")
    scheduler    = sampler_params.get("scheduler", "simple")

    print(
        f"[_run_wan_sampler] total_steps={total_steps}, "
        f"high=0→{steps_high}, low={steps_high}→{total_steps}, "
        f"cfg={cfg}, seed_a={seed_a}, seed_b={seed_b}"
    )

    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_dict["samples"],
        latent_dict.get("downscale_ratio_spacial", None),
    )

    batch_inds = latent_dict["batch_index"] if "batch_index" in latent_dict else None
    noise_mask = latent_dict["noise_mask"] if "noise_mask" in latent_dict else None

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    noise_high = comfy.sample.prepare_noise(latent_image, seed_a, batch_inds)
    callback_high = latent_preview.prepare_callback(model, total_steps)

    samples_high = comfy.sample.sample(
        model, noise_high, total_steps, cfg, sampler_name, scheduler,
        cond_pos, cond_neg, latent_image,
        denoise=1.0, disable_noise=False,
        start_step=0, last_step=steps_high,
        force_full_denoise=False, noise_mask=noise_mask,
        callback=callback_high, disable_pbar=disable_pbar, seed=seed_a,
    )

    latent_low = comfy.sample.fix_empty_latent_channels(
        model_b,
        samples_high,
        latent_dict.get("downscale_ratio_spacial", None),
    )

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


def _render_sdxl(model, clip, vae, pos_prompt, neg_prompt,
                 width, height, batch, sampler_params,
                 loras_a=None, lora_overrides=None, lora_stack_key=''):
    """
    SDXL render — also suitable for SD1.5-style image flow if desired.
    """
    if loras_a:
        model, clip = _apply_loras(
            model, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key
        )

    clip = clip.clone()
    clip.clip_layer(-2)

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    latent = _make_latent(4, width, height, batch=batch)
    latent_out = _run_standard_ksampler(model, cond_pos, cond_neg, latent, sampler_params)
    decoded = _decode_latent_output(vae, latent_out, tag="_render_sdxl")
    return decoded, latent_out


def _render_qwen_image(model, clip, vae, pos_prompt, neg_prompt,
                       width, height, batch, sampler_params,
                       loras_a=None, lora_overrides=None, lora_stack_key=''):
    """
    Qwen Image render — mirrors qwen_image.json.
    """
    from comfy_extras.nodes_sd3 import EmptySD3LatentImage

    if loras_a:
        model, clip = _apply_loras(
            model, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key
        )

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    model = _patch_model_sampling_auraflow(
        model, shift=float(sampler_params.get("shift", 3.1))
    )

    latent_node = EmptySD3LatentImage()

    if hasattr(latent_node, "generate"):
        latent = latent_node.generate(width=width, height=height, batch_size=batch)[0]
    elif hasattr(EmptySD3LatentImage, "execute"):
        latent = _extract_node_outputs(
            EmptySD3LatentImage.execute(width=width, height=height, batch_size=batch)
        )[0]
    else:
        raise RuntimeError("EmptySD3LatentImage has neither generate() nor execute().")

    if not isinstance(latent, dict) or "samples" not in latent:
        raise TypeError("EmptySD3LatentImage did not return a LATENT dict.")

    latent_out = _run_standard_ksampler(
        model, cond_pos, cond_neg, latent, sampler_params
    )

    decoded = _decode_latent_output(vae, latent_out, tag="_render_qwen_image")
    return decoded, latent_out


def _render_flux1(model, clip, vae, pos_prompt, neg_prompt,
                  width, height, batch, sampler_params,
                  loras_a=None, lora_overrides=None, lora_stack_key=''):
    """
    Flux 1 render.
    """
    import node_helpers
    from nodes import ConditioningZeroOut

    if loras_a:
        model, clip = _apply_loras(
            model, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key
        )

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    cond_pos = node_helpers.conditioning_set_values(cond_pos, {"guidance": 3.5})
    (cond_neg,) = ConditioningZeroOut().zero_out(cond_neg)

    latent = _make_latent(16, width, height, batch=batch)
    latent_out = _run_standard_ksampler(model, cond_pos, cond_neg, latent, sampler_params)
    decoded = _decode_latent_output(vae, latent_out, tag="_render_flux1")
    return decoded, latent_out


def _render_flux2(model, clip, vae, pos_prompt, neg_prompt,
                  width, height, batch, sampler_params,
                  loras_a=None, lora_overrides=None, lora_stack_key=''):
    """
    Flux 2 render using custom sampler nodes.
    Returns decoded images and a LATENT dict.
    """
    import comfy
    from comfy_extras.nodes_custom_sampler import (
        CFGGuider, KSamplerSelect, BasicScheduler, SamplerCustomAdvanced, RandomNoise,
    )

    if loras_a:
        model, clip = _apply_loras(
            model, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key
        )

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    latent = _make_latent(16, width, height, batch=batch)

    steps        = int(sampler_params.get("steps", 20))
    cfg          = float(sampler_params.get("cfg", 5.0))
    seed         = int(sampler_params.get("seed_a", sampler_params.get("seed", 0)))
    sampler_name = sampler_params.get("sampler_name", "euler")
    scheduler    = sampler_params.get("scheduler", "beta")

    guider = _extract_node_outputs(CFGGuider.execute(model, cond_pos, cond_neg, cfg))[0]
    sampler = _extract_node_outputs(KSamplerSelect.execute(sampler_name))[0]
    sigmas = _extract_node_outputs(BasicScheduler.execute(model, scheduler, steps, 1.0))[0]
    noise_obj = _extract_node_outputs(RandomNoise.execute(seed))[0]

    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent["samples"],
        latent.get("downscale_ratio_spacial", None),
    )
    latent_in = latent.copy()
    latent_in["samples"] = latent_image

    sca_result = SamplerCustomAdvanced.execute(noise_obj, guider, sampler, sigmas, latent_in)
    sca_outs = _extract_node_outputs(sca_result)

    # Prefer final output; fall back to second if needed
    if len(sca_outs) >= 1 and isinstance(sca_outs[0], dict) and "samples" in sca_outs[0]:
        latent_out = sca_outs[0]
    elif len(sca_outs) >= 2 and isinstance(sca_outs[1], dict) and "samples" in sca_outs[1]:
        latent_out = sca_outs[1]
    else:
        raise TypeError(f"SamplerCustomAdvanced returned unexpected outputs: {type(sca_result)}")

    decoded = _decode_latent_output(vae, latent_out, tag="_render_flux2")
    return decoded, latent_out


def _render_zimage(model, clip, vae, pos_prompt, neg_prompt,
                   width, height, batch, sampler_params,
                   loras_a=None, lora_overrides=None, lora_stack_key=''):
    from comfy_extras.nodes_sd3 import EmptySD3LatentImage

    if loras_a:
        model, clip = _apply_loras(
            model, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key
        )

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    model = _patch_model_sampling_auraflow(
        model, shift=float(sampler_params.get("shift", 3.0))
    )

    latent_node = EmptySD3LatentImage()
    if hasattr(latent_node, "generate"):
        latent = latent_node.generate(width=width, height=height, batch_size=batch)[0]
    else:
        latent = _extract_node_outputs(
            EmptySD3LatentImage.execute(width=width, height=height, batch_size=batch)
        )[0]

    latent_out = _run_standard_ksampler(
        model, cond_pos, cond_neg, latent, sampler_params
    )

    decoded = _decode_latent_output(vae, latent_out, tag="_render_zimage")
    return decoded, latent_out


def _render_wan_image(model, clip, vae, pos_prompt, neg_prompt,
                      width, height, batch, sampler_params,
                      loras_a=None, lora_overrides=None, lora_stack_key=''):
    """
    WAN image render using a video-native latent with T=1.
    """
    if loras_a:
        model, clip = _apply_loras(
            model, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key
        )

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    latent = _make_latent(16, width, height, batch=batch, frames=1, spacial_ratio=8)
    latent_out = _run_standard_ksampler(model, cond_pos, cond_neg, latent, sampler_params)
    decoded = _decode_latent_output(vae, latent_out, tag="_render_wan_image")
    return decoded, latent_out


def _render_wan_video_t2v(model_a, model_b, clip, vae, pos_prompt, neg_prompt,
                          width, height, length, sampler_params,
                          loras_a=None, loras_b=None,
                          lora_overrides=None, lora_stack_key_a='', lora_stack_key_b=''):
    """
    WAN video T2V render.
    """
    if loras_a:
        model_a, clip = _apply_loras(
            model_a, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key_a
        )

    clip_b = clip
    if model_b and loras_b:
        model_b, clip_b = _apply_loras(
            model_b, clip_b, loras_b, lora_overrides or {}, stack_key=lora_stack_key_b
        )

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    if model_b and clip_b is not clip:
        tokens_pos_b = clip_b.tokenize(pos_prompt)
        cond_pos_b = clip_b.encode_from_tokens_scheduled(tokens_pos_b)
        tokens_neg_b = clip_b.tokenize(neg_prompt)
        cond_neg_b = clip_b.encode_from_tokens_scheduled(tokens_neg_b)
    else:
        cond_pos_b = cond_pos
        cond_neg_b = cond_neg

    L = int(length or 81)
    from comfy_extras.nodes_hunyuan import EmptyHunyuanLatentVideo

    latent = None
    node = EmptyHunyuanLatentVideo()

    if hasattr(node, "generate"):
        latent = node.generate(width=width, height=height, length=L, batch_size=1)[0]
    elif hasattr(EmptyHunyuanLatentVideo, "execute"):
        outs = _extract_node_outputs(
            EmptyHunyuanLatentVideo.execute(width=width, height=height, length=L, batch_size=1)
        )
        latent = outs[0]
    else:
        raise RuntimeError("EmptyHunyuanLatentVideo has neither generate() nor execute().")

    if not isinstance(latent, dict) or "samples" not in latent:
        raise TypeError("EmptyHunyuanLatentVideo did not return a LATENT dict.")

    latent_out = _run_wan_sampler(
        model_a, cond_pos, cond_neg, latent, sampler_params,
        model_b=model_b, cond_pos_b=cond_pos_b, cond_neg_b=cond_neg_b,
    )

    decoded = _decode_latent_output(vae, latent_out, tag="_render_wan_video_t2v")
    return decoded, latent_out


def _render_wan_video_i2v(model_a, model_b, clip, vae, pos_prompt, neg_prompt,
                          width, height, length, sampler_params,
                          source_image=None,
                          loras_a=None, loras_b=None,
                          lora_overrides=None, lora_stack_key_a='', lora_stack_key_b=''):
    """
    WAN video I2V render.
    """
    from comfy_extras.nodes_wan import WanImageToVideo as WanI2VNode

    if loras_a:
        model_a, clip = _apply_loras(
            model_a, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key_a
        )

    clip_b = clip
    if model_b and loras_b:
        model_b, clip_b = _apply_loras(
            model_b, clip_b, loras_b, lora_overrides or {}, stack_key=lora_stack_key_b
        )

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    L = int(length or 81)
    i2v_result = WanI2VNode.execute(
        positive=cond_pos,
        negative=cond_neg,
        vae=vae,
        width=width,
        height=height,
        length=L,
        batch_size=1,
        start_image=source_image,
    )
    i2v_outs = _extract_node_outputs(i2v_result)

    if len(i2v_outs) < 3:
        raise TypeError("WanImageToVideo did not return expected (positive, negative, latent).")

    cond_pos = i2v_outs[0]
    cond_neg = i2v_outs[1]
    latent = i2v_outs[2]

    if not isinstance(latent, dict) or "samples" not in latent:
        raise TypeError("WanImageToVideo latent output is not a LATENT dict.")

    cond_pos_b = cond_pos
    cond_neg_b = cond_neg

    latent_out = _run_wan_sampler(
        model_a, cond_pos, cond_neg, latent, sampler_params,
        model_b=model_b, cond_pos_b=cond_pos_b, cond_neg_b=cond_neg_b,
    )

    decoded = _decode_latent_output(vae, latent_out, tag="_render_wan_video_i2v")
    return decoded, latent_out
