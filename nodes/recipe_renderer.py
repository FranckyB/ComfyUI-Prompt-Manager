"""
Workflow Renderer — render-only node.

Accepts RECIPE_DATA (JSON string from Recipe Builder or PromptExtractor),
loads models, samples, decodes, and outputs IMAGE + LATENT.

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
    list_compatible_vaes,
    list_compatible_clips,
)
from ..py.workflow_extraction_utils import (
    resolve_model_name,
    resolve_vae_name,
    resolve_clip_names,
)
from ..py.workflow_data_utils import strip_runtime_objects
from ..py.lora_utils import resolve_lora_path

# Reuse the robust GGUF loading helper used by RecipeModelLoader when available.
try:
    from .recipe_model_loader import _load_gguf_unet as _load_gguf_unet_shared
except Exception:
    _load_gguf_unet_shared = None

# Keep renderer fallback sampler defaults aligned with Recipe Builder family defaults.
_FAMILY_SAMPLER_DEFAULTS = {
    "ernie": {"steps_a": 4, "cfg": 1.0, "sampler": "euler_ancestral", "scheduler": "beta"},
    "sdxl": {"steps_a": 20, "cfg": 5.0, "sampler": "dpmpp_2m_sde", "scheduler": "karras"},
    "flux1": {"steps_a": 20, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
    "flux2": {"steps_a": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
    "zimage": {"steps_a": 9, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
    "ltxv": {"steps_a": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
    "wan_image": {"steps_a": 8, "cfg": 1.0, "sampler": "lcm", "scheduler": "simple"},
    "wan_video_t2v": {"steps_a": 2, "cfg": 1.0, "sampler": "lcm", "scheduler": "simple", "steps_b": 2},
    "wan_video_i2v": {"steps_a": 2, "cfg": 1.0, "sampler": "lcm", "scheduler": "simple", "steps_b": 2},
    "qwen_image": {"steps_a": 10, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
}

# Runtime caches for non-checkpoint components reused across repeated renders.
_class_vae_cache = {}
_class_clip_cache = {}
_MODEL_KEYS = ("model_a", "model_b", "model_c", "model_d")


def _normalize_model_slot(value):
    key = str(value or "model_a").strip().lower()
    return key if key in _MODEL_KEYS else "model_a"

class WorkflowRenderer:
    """
    Render-only generation node.

    Takes RECIPE_DATA (from Workflow Builder or PromptExtractor),
    loads models, applies LoRAs, samples, decodes.
    Outputs IMAGE + LATENT.
    """

    _class_model_cache: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe_data": ("RECIPE_DATA", {
                    "forceInput": True,
                    "tooltip": "Connect recipe_data from Recipe Builder or PromptExtractor",
                }),
                "model_slot": (_MODEL_KEYS, {
                    "default": "model_a",
                    "tooltip": "Select which model slot to render from in recipe_data.",
                }),
                "clear_cache_after_render": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Clear Renderer model cache and request memory cleanup after rendering completes.",
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

    RETURN_TYPES = ("RECIPE_DATA", "IMAGE", "LATENT")
    RETURN_NAMES = ("recipe_data", "output_image", "output_latent")
    FUNCTION = "execute"
    CATEGORY = "Prompt Manager"
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Render-only node. Accepts recipe_data, loads models, samples, "
        "and decodes. Outputs IMAGE + LATENT + RECIPE_DATA "
        "(with MODEL/CLIP/VAE passthrough)."
    )

    @classmethod
    def IS_CHANGED(cls, recipe_data, source_image=None, **kwargs):
        """Return a slot-aware fingerprint so unrelated slot edits do not force rerender."""
        import hashlib
        import json as _json
        import time

        def _tensor_sig(tensor):
            try:
                return {
                    "shape": tuple(tensor.shape),
                    "sum": float(tensor.detach().sum().item()),
                }
            except Exception:
                return {
                    "shape": str(getattr(tensor, "shape", "")),
                    "sum": str(tensor),
                }

        h = hashlib.sha256()
        clear_cache_after_render = bool(kwargs.get("clear_cache_after_render", False))
        selected_slot = _normalize_model_slot(kwargs.get("model_slot", "model_a"))

        h.update(selected_slot.encode())
        h.update(str(clear_cache_after_render).encode())

        wf = None
        if isinstance(recipe_data, dict):
            wf = recipe_data
        elif isinstance(recipe_data, str):
            try:
                wf = json.loads(recipe_data)
            except Exception:
                wf = None
                h.update(recipe_data.encode())
        else:
            h.update(str(type(recipe_data)).encode())

        if isinstance(wf, dict):
            wf_sampler = wf.get("sampler", {}) if isinstance(wf.get("sampler"), dict) else {}
            wf_res = wf.get("resolution", {}) if isinstance(wf.get("resolution"), dict) else {}

            family_key = str(wf.get("family", "") or "")
            slot_block = {}
            secondary_block = {}

            if int(wf.get("version", 0) or 0) >= 2 and isinstance(wf.get("models"), dict):
                models = wf.get("models", {})
                primary = models.get(selected_slot)
                if isinstance(primary, dict):
                    slot_block = dict(primary)
                    family_key = str(primary.get("family", family_key) or family_key)
                    if isinstance(primary.get("sampler"), dict):
                        merged_sampler = dict(wf_sampler)
                        merged_sampler.update(primary.get("sampler", {}))
                        wf_sampler = merged_sampler
                    if isinstance(primary.get("resolution"), dict):
                        merged_res = dict(wf_res)
                        merged_res.update(primary.get("resolution", {}))
                        wf_res = merged_res

                slot_index = _MODEL_KEYS.index(selected_slot)
                if slot_index < len(_MODEL_KEYS) - 1:
                    secondary_key = _MODEL_KEYS[slot_index + 1]
                    secondary = models.get(secondary_key)
                    if isinstance(secondary, dict):
                        secondary_block = dict(secondary)
                        if isinstance(secondary.get("sampler"), dict):
                            sec_sampler = secondary.get("sampler", {})
                            if "steps_b" not in wf_sampler and sec_sampler.get("steps") is not None:
                                wf_sampler["steps_b"] = sec_sampler.get("steps")
                            if "seed_b" not in wf_sampler and sec_sampler.get("seed") is not None:
                                wf_sampler["seed_b"] = sec_sampler.get("seed")
            else:
                if selected_slot == "model_a":
                    slot_block = {
                        "family": wf.get("family", ""),
                        "model": wf.get("model_a", ""),
                        "positive_prompt": wf.get("positive_prompt", ""),
                        "negative_prompt": wf.get("negative_prompt", ""),
                        "loras": wf.get("loras_a", []),
                        "vae": wf.get("vae", ""),
                        "clip": wf.get("clip", []),
                        "clip_type": wf.get("clip_type", ""),
                        "loader_type": wf.get("loader_type", ""),
                    }
                elif selected_slot == "model_b":
                    slot_block = {
                        "family": wf.get("family", ""),
                        "model": wf.get("model_b", ""),
                        "loras": wf.get("loras_b", []),
                    }

            if not family_key:
                model_ref = ""
                if isinstance(slot_block, dict):
                    model_ref = str(slot_block.get("model", "") or "")
                if not model_ref:
                    model_ref = str(wf.get("model_a", "") or "")
                if model_ref:
                    try:
                        resolved_ref, _ = resolve_model_name(model_ref)
                        family_key = get_model_family(resolved_ref or model_ref) or ""
                    except Exception:
                        family_key = ""
            if not family_key:
                family_key = "sdxl"

            try:
                denoise_value = float(wf_sampler.get("denoise", 1.0))
            except (TypeError, ValueError):
                denoise_value = 1.0
            denoise_value = max(0.0, min(1.0, denoise_value))

            relevant = {
                "version": int(wf.get("version", 0) or 0),
                "source": str(wf.get("_source", "") or ""),
                "slot": selected_slot,
                "family": family_key,
                "slot_block": strip_runtime_objects(slot_block if isinstance(slot_block, dict) else {}),
                "secondary_block": strip_runtime_objects(secondary_block if isinstance(secondary_block, dict) else {}),
                "sampler": strip_runtime_objects(wf_sampler),
                "resolution": strip_runtime_objects(wf_res),
            }
            h.update(_json.dumps(relevant, sort_keys=True, default=str).encode())

            uses_image = (family_key == "wan_video_i2v") or (
                family_key not in ("wan_video_t2v", "wan_video_i2v") and denoise_value < 1.0
            )
            if uses_image:
                image_for_sig = source_image if isinstance(source_image, torch.Tensor) else wf.get("IMAGE")
                if isinstance(image_for_sig, torch.Tensor):
                    h.update(_json.dumps(_tensor_sig(image_for_sig), sort_keys=True).encode())
                else:
                    h.update(str(image_for_sig).encode())

        if clear_cache_after_render:
            # Force execution when post-render clear is enabled so purge always runs.
            return f"{h.hexdigest()}::{time.time_ns()}"
        return h.hexdigest()

    @classmethod
    def _clear_cached_models(cls):
        cleared = len(cls._class_model_cache)
        cls._class_model_cache.clear()
        cleared_vaes = len(_class_vae_cache)
        _class_vae_cache.clear()
        cleared_clips = len(_class_clip_cache)
        _class_clip_cache.clear()
        print(f"[RecipeRenderer] Cleared caches (models={cleared}, vaes={cleared_vaes}, clips={cleared_clips})")

        try:
            comfy.model_management.soft_empty_cache()
        except Exception as e:
            print(f"[RecipeRenderer] Cache cleanup call failed: {e}")

        try:
            import gc
            gc.collect()
        except Exception:
            pass

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[RecipeRenderer] CUDA cache clear failed: {e}")

    def execute(self, recipe_data, model_slot="model_a", clear_cache_after_render=False, source_image=None, unique_id=None):
        workflow_data = recipe_data

        # ── Parse workflow_data ───────────────────────────────────────────
        if isinstance(workflow_data, dict):
            wf = workflow_data
        elif isinstance(workflow_data, str):
            try:
                wf = json.loads(workflow_data)
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"[RecipeRenderer] Invalid workflow_data: {e}")
        else:
            raise ValueError(f"[RecipeRenderer] Invalid workflow_data type: {type(workflow_data)}")

        wf_out = dict(wf)

        wf_sampler = wf.get("sampler", {})
        wf_res = wf.get("resolution", {})
        if not isinstance(wf_sampler, dict):
            wf_sampler = {}
        if not isinstance(wf_res, dict):
            wf_res = {}

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
        loader_type_str = wf.get("loader_type", "")
        loras_a = wf.get("loras_a", [])
        loras_b = wf.get("loras_b", [])
        primary_key = None
        secondary_key = None
        primary = None
        secondary = None
        selected_slot = _normalize_model_slot(model_slot)
        slot_model_defined = False

        if int(wf.get("version", 0) or 0) >= 2 and isinstance(wf.get("models"), dict):
            models = wf.get("models", {})
            primary_key = selected_slot
            secondary = None
            idx = _MODEL_KEYS.index(selected_slot)
            if idx < len(_MODEL_KEYS) - 1:
                secondary_key = _MODEL_KEYS[idx + 1]

            primary = models.get(primary_key) if isinstance(models.get(primary_key), dict) else None
            if secondary_key and isinstance(models.get(secondary_key), dict):
                secondary = models.get(secondary_key)

            if isinstance(primary, dict):
                model_name_a = primary.get("model", model_name_a)
                positive_prompt = primary.get("positive_prompt", positive_prompt)
                negative_prompt = primary.get("negative_prompt", negative_prompt)
                family_key = primary.get("family", family_key)
                vae_name = primary.get("vae", vae_name)
                clip_names = primary.get("clip", clip_names)
                clip_type_str = primary.get("clip_type", clip_type_str)
                loader_type_str = primary.get("loader_type", loader_type_str)
                if isinstance(primary.get("loras"), list):
                    loras_a = primary.get("loras", [])
                if isinstance(primary.get("sampler"), dict):
                    merged_sampler = dict(wf_sampler) if isinstance(wf_sampler, dict) else {}
                    merged_sampler.update(primary.get("sampler", {}))
                    wf_sampler = merged_sampler
                if isinstance(primary.get("resolution"), dict):
                    merged_res = dict(wf_res) if isinstance(wf_res, dict) else {}
                    merged_res.update(primary.get("resolution", {}))
                    wf_res = merged_res

                if str(primary.get("model", "")).strip() or primary.get("MODEL") is not None:
                    slot_model_defined = True

            if isinstance(secondary, dict):
                model_name_b = secondary.get("model", model_name_b)
                if isinstance(secondary.get("loras"), list):
                    loras_b = secondary.get("loras", [])
                if isinstance(secondary.get("sampler"), dict):
                    sec_sampler = secondary.get("sampler", {})
                    if isinstance(wf_sampler, dict):
                        if "steps_b" not in wf_sampler and sec_sampler.get("steps") is not None:
                            wf_sampler["steps_b"] = sec_sampler.get("steps")
                        if "seed_b" not in wf_sampler and sec_sampler.get("seed") is not None:
                            wf_sampler["seed_b"] = sec_sampler.get("seed")
        else:
            # Legacy schema supports root model_a/model_b only.
            if selected_slot == "model_a":
                slot_model_defined = bool(str(model_name_a or "").strip()) or (wf.get("MODEL_A") is not None)
            elif selected_slot == "model_b":
                slot_model_defined = bool(str(model_name_b or "").strip()) or (wf.get("MODEL_B") is not None)
            else:
                slot_model_defined = False

        if not slot_model_defined:
            raise ValueError(f"[RecipeRenderer] Selected model is not defined for slot '{selected_slot}'.")

        if isinstance(clip_names, str):
            clip_names = [clip_names] if clip_names else []

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

        # ── Family defaults for VAE/CLIP when upstream data is incomplete ──
        clip_is_placeholder = bool(clip_names) and all(
            (not n) or str(n).startswith('(') for n in clip_names
        )

        if _family_is_ckpt:
            # Checkpoint families can use embedded VAE/CLIP.
            if not vae_name or str(vae_name).startswith('('):
                vae_name = "(Default)"
            if not clip_names or clip_is_placeholder:
                clip_names = ["(Default)"]
            if not clip_type_str:
                clip_type_str = str(_family_spec.get("clip_type", ""))
        else:
            # Non-checkpoint families require explicit compatible VAE/CLIP selections.
            if not vae_name or str(vae_name).startswith('('):
                vaes, rec_vae = list_compatible_vaes(family_key, return_recommended=True)
                vae_name = rec_vae or (vaes[0] if vaes else "")

            # Use the family-defined CLIP type for non-checkpoint families to avoid
            # stale type values from previously selected families.
            family_clip_type = str(_family_spec.get("clip_type", "") or "")
            if family_clip_type:
                clip_type_str = family_clip_type

            # Validate/repair clip_names against family-compatible CLIPs.
            # This prevents stale selections from other families (e.g. SDXL) from
            # reaching the sampler and causing dimension mismatches.
            compatible_clips, rec_clip = list_compatible_clips(family_key, return_recommended=True)
            clip_slots = int(_family_spec.get("clip_slots", 1) or 1)
            selected = []

            if compatible_clips:
                by_basename = {
                    os.path.basename(str(c)).lower(): c
                    for c in compatible_clips
                }

                if family_key == "ernie":
                    preferred_ernie = None
                    for c in compatible_clips:
                        base = os.path.basename(str(c)).lower()
                        if base in ("ministral-3-3b.safetensors", "ministral_3_3b.safetensors"):
                            preferred_ernie = c
                            break
                    if not preferred_ernie:
                        raise ValueError(
                            "[RecipeRenderer] Ernie requires the text encoder "
                            "'ministral-3-3b.safetensors'. Install/select that encoder "
                            "and retry."
                        )
                    rec_clip = preferred_ernie

                incoming = clip_names if isinstance(clip_names, list) else []
                for raw_name in incoming:
                    name = str(raw_name or "").strip()
                    if not name or name.startswith("("):
                        continue

                    # Accept direct exact match or basename-equivalent match.
                    chosen = None
                    if name in compatible_clips:
                        chosen = name
                    else:
                        chosen = by_basename.get(os.path.basename(name).lower())

                    if chosen and chosen not in selected:
                        selected.append(chosen)
                    if len(selected) >= clip_slots:
                        break

                if not selected or clip_is_placeholder:
                    clip_patterns = [p.lower() for p in _family_spec.get("clip", []) if p]

                    if clip_patterns and clip_slots >= 2:
                        for pat in clip_patterns:
                            for c in compatible_clips:
                                if pat in os.path.basename(str(c)).lower() and c not in selected:
                                    selected.append(c)
                                    break
                            if len(selected) >= clip_slots:
                                break

                    if not selected:
                        selected = [rec_clip] if rec_clip in compatible_clips else [compatible_clips[0]]

                if family_key == "ernie" and rec_clip in compatible_clips:
                    selected = [rec_clip]

            clip_names = selected

        family_defaults = _FAMILY_SAMPLER_DEFAULTS.get(
            family_key,
            {"steps_a": 20, "cfg": 5.0, "sampler": "euler", "scheduler": "simple"},
        )
        sampler_params = {
            "steps": int(wf_sampler.get("steps_a", wf_sampler.get("steps", family_defaults.get("steps_a", 20)))),
            "cfg": float(wf_sampler.get("cfg", family_defaults.get("cfg", 5.0))),
            "seed_a": int(wf_sampler.get("seed_a", wf_sampler.get("seed", 0))),
            "sampler_name": wf_sampler.get("sampler_name", family_defaults.get("sampler", "euler")),
            "scheduler": wf_sampler.get("scheduler", family_defaults.get("scheduler", "simple")),
        }
        try:
            denoise_value = float(wf_sampler.get("denoise", 1.0))
        except (TypeError, ValueError):
            denoise_value = 1.0
        denoise_value = max(0.0, min(1.0, denoise_value))
        sampler_params["denoise"] = denoise_value
        seed_b = wf_sampler.get("seed_b")
        workflow_image = wf.get("IMAGE")
        denoise_source_image = source_image if isinstance(source_image, torch.Tensor) else workflow_image

        # Only WAN video families use dual-model sampling. Ignore stale model_b
        # values for image families so logs and state remain unambiguous.
        if family_key not in ("wan_video_t2v", "wan_video_i2v"):
            model_name_b = None

        print(f"[RecipeRenderer] Family: {get_family_label(family_key)} "
              f"(strategy={strategy}), model_a={model_name_a}, "
              f"model_b={model_name_b or '—'}")

        # ── Clamp resolution for i2v ──────────────────────────────────────
        if family_key == "wan_video_i2v":
            MAX_DIM = 1280
            if max(width, height) > MAX_DIM:
                scale = MAX_DIM / max(width, height)
                width = (round(width * scale) // 16) * 16
                height = (round(height * scale) // 16) * 16

        # ── Load Model A (or reuse passthrough object) ────────────────────
        resolved_a = None
        folder_a = None
        _cache = WorkflowRenderer._class_model_cache
        passthrough_model_a = primary.get("MODEL") if isinstance(primary, dict) else None
        passthrough_clip = primary.get("CLIP") if isinstance(primary, dict) else None
        passthrough_vae = primary.get("VAE") if isinstance(primary, dict) else None

        if passthrough_model_a is None:
            passthrough_model_a = wf.get("MODEL_A", None)
        if passthrough_clip is None:
            passthrough_clip = wf.get("CLIP", None)
        if passthrough_vae is None:
            passthrough_vae = wf.get("VAE", None)

        if passthrough_model_a is not None:
            model_a = passthrough_model_a
            clip_a = passthrough_clip
            vae_a = passthrough_vae
        else:
            resolved_a, folder_a = resolve_model_name(model_name_a)
            # Verify resolved model belongs to a compatible family — resolve_model_name
            # does basename matching which can match unrelated models (e.g. audio files).
            if resolved_a is not None:
                compat_check = get_compatible_families(family_key)
                resolved_family = get_model_family(resolved_a)
                if resolved_family is not None and resolved_family not in compat_check:
                    print(f"[RecipeRenderer] Resolved model {resolved_a} is family "
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
                    print(f"[RecipeRenderer] Using fallback model: {model_name_a}")
                else:
                    raise FileNotFoundError(
                        f"Model A not found and no fallback for family {family_key}: {model_name_a}"
                    )

            full_path_a = folder_paths.get_full_path(folder_a, resolved_a)
            _cache_key_a = (str(unique_id), full_path_a, family_key)
            if _cache_key_a not in _cache:
                print(f"[RecipeRenderer] Loading model: {resolved_a}")
                _cache[_cache_key_a] = _load_model_from_path(resolved_a, folder_a, full_path_a, family_is_checkpoint=_family_is_ckpt)
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
        # Never trust/reuse prior LATENT directly across families.
        # If denoise < 1.0, regenerate a compatible latent from IMAGE via current VAE.
        input_latent_for_denoise = None
        if family_key not in ("wan_video_t2v", "wan_video_i2v") and denoise_value < 1.0:
            input_latent_for_denoise = _build_denoise_latent_from_image(vae, denoise_source_image, family_key)

        render_args = dict(
            model=model_a, clip=clip, vae=vae,
            pos_prompt=positive_prompt, neg_prompt=negative_prompt,
            width=width, height=height, batch=batch,
            sampler_params=sampler_params,
            input_latent=input_latent_for_denoise,
            loras_a=loras_a, lora_overrides=lora_overrides,
            lora_stack_key=stack_key_a,
        )

        # Unsupported families — error early with a clear message
        unsupported = ("ltxv",)
        if family_key in unsupported:
            raise ValueError(
                f"[RecipeRenderer] Family '{family_key}' is not yet supported. "
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

        elif family_key == "ernie":
            decoded, out_latent = _render_ernie(**render_args)

        elif family_key == "sdxl":
            decoded, out_latent = _render_sdxl(**render_args)

        elif family_key == "wan_image":
            decoded, out_latent = _render_wan_image(**render_args)

        elif family_key in ("wan_video_t2v", "wan_video_i2v"):
            # Load Model B for dual-sampler
            model_b_obj = None
            passthrough_model_b = secondary.get("MODEL") if isinstance(secondary, dict) else None
            if passthrough_model_b is None:
                passthrough_model_b = wf.get("MODEL_B", None)
            if passthrough_model_b is not None:
                model_b_obj = passthrough_model_b
            elif model_name_b:
                resolved_b, folder_b = resolve_model_name(model_name_b)
                if resolved_b:
                    full_path_b = folder_paths.get_full_path(folder_b, resolved_b)
                    _cache_key_b = (str(unique_id), full_path_b, family_key + "_b")
                    if _cache_key_b not in _cache:
                        print(f"[RecipeRenderer] Loading model B: {resolved_b}")
                        _cache[_cache_key_b] = _load_model_from_path(resolved_b, folder_b, full_path_b, family_is_checkpoint=_family_is_ckpt)
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
                f"[RecipeRenderer] Unsupported family '{family_key}'. "
                f"No render function available."
            )

        # Defensive guard: some Flux setups can decode at 2x the requested
        # size due to latent/vae scale mismatches in upstream backends.
        # Normalize to requested resolution so Builder->Renderer is stable.
        if isinstance(decoded, torch.Tensor) and decoded.ndim == 4:
            out_h = int(decoded.shape[1])
            out_w = int(decoded.shape[2])
            if family_key in ("flux1", "flux2") and out_w == width * 2 and out_h == height * 2:
                print(
                    f"[RecipeRenderer] Flux decode size {out_w}x{out_h} is 2x requested "
                    f"{width}x{height}; resizing output to requested resolution."
                )
                decoded_nchw = decoded.permute(0, 3, 1, 2)
                decoded_nchw = torch.nn.functional.interpolate(
                    decoded_nchw,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                decoded = decoded_nchw.permute(0, 2, 3, 1).contiguous()

        cond_pos_out, cond_neg_out = _encode_text_conditioning(clip, positive_prompt, negative_prompt)

        if int(wf_out.get("version", 0) or 0) >= 2 and isinstance(wf_out.get("models"), dict):
            models_out = wf_out.get("models", {})
            if primary_key and isinstance(models_out.get(primary_key), dict):
                primary_out = models_out[primary_key]
                primary_out["MODEL"] = model_a
                primary_out["CLIP"] = clip
                primary_out["VAE"] = vae
                primary_out["POSITIVE"] = cond_pos_out
                primary_out["NEGATIVE"] = cond_neg_out

            if secondary_key and model_b_obj is not None and isinstance(models_out.get(secondary_key), dict):
                secondary_out = models_out[secondary_key]
                secondary_out["MODEL"] = model_b_obj
                secondary_out["VAE"] = vae
                secondary_out["CLIP"] = clip

        wf_out["LATENT"] = out_latent
        wf_out["IMAGE"] = decoded

        if clear_cache_after_render:
            self._clear_cached_models()

        return (wf_out, decoded, out_latent)


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
        print(f"[RecipeRenderer] Failed to encode positive conditioning: {e}")
        cond_pos = None

    try:
        tokens_neg = clip.tokenize(str(negative_prompt or ""))
        cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)
    except Exception as e:
        print(f"[RecipeRenderer] Failed to encode negative conditioning: {e}")
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
        print(f"[RecipeRenderer] Loading GGUF model: {resolved_path}")
        if _load_gguf_unet_shared is not None:
            model = _load_gguf_unet_shared(full_model_path)
        elif GGUF_SUPPORT and _load_gguf_unet:
            model = _load_gguf_unet(full_model_path)
        else:
            raise RuntimeError(
                "[RecipeRenderer] GGUF model selected but no GGUF loader is available. "
                "Install/enable ComfyUI-GGUF (UnetLoaderGGUF)."
            )
    elif is_checkpoint:
        print(f"[RecipeRenderer] Loading checkpoint: {resolved_path}")
        out   = comfy.sd.load_checkpoint_guess_config(
            full_model_path, output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        model, clip, vae = out[0], out[1], out[2]
    else:
        print(f"[RecipeRenderer] Loading diffusion model: {resolved_path}")
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
                print(f"[RecipeRenderer] weights_only load failed, retrying with weights_only=False: {e}")
                pl_sd = torch.load(full_model_path, map_location='cpu', weights_only=False)
                sd = pl_sd.get('state_dict', pl_sd)
                if hasattr(comfy.sd, 'load_diffusion_model_state_dict'):
                    model = comfy.sd.load_diffusion_model_state_dict(sd)
                else:
                    raise RuntimeError(
                        f"[RecipeRenderer] Cannot load model '{resolved_path}': "
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
            cache_key = os.path.realpath(vae_path)
            cached_vae = _class_vae_cache.get(cache_key)
            if cached_vae is not None:
                return cached_vae
            print(f"[RecipeRenderer] Loading VAE: {vae_name}")
            sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
            v = comfy.sd.VAE(sd=sd, metadata=metadata)
            v.throw_exception_if_invalid()
            _class_vae_cache[cache_key] = v
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

    print(f"[RecipeRenderer] Loading CLIP: {clip_names}")
    cache_key = (tuple(os.path.realpath(p) for p in valid_paths), str(clip_type))
    cached_clip = _class_clip_cache.get(cache_key)
    if cached_clip is not None:
        return cached_clip

    loaded_clip = comfy.sd.load_clip(
        ckpt_paths=valid_paths,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type,
    )
    _class_clip_cache[cache_key] = loaded_clip
    return loaded_clip


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
            continue

        if lora.get('available', lora.get('available', True)) is False:
            continue

        model_strength = float(lora_st.get('model_strength', lora.get('model_strength', 1.0)))
        clip_strength  = float(lora_st.get('clip_strength',  lora.get('clip_strength',  1.0)))

        lora_path, available = resolve_lora_path(lora_name)
        if not available:
            continue

        print(f"[RecipeRenderer] Applying LoRA: {lora_name} "
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

    decoded = vae.decode(latent_out["samples"])

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

    scale = int(spacial_ratio) if spacial_ratio else 8
    scale = max(1, scale)

    if frames is None:
        samples = torch.zeros(
            [batch, channels, height // scale, width // scale],
            device=comfy.model_management.intermediate_device(),
        )
    else:
        samples = torch.zeros(
            [batch, channels, frames, height // scale, width // scale],
            device=comfy.model_management.intermediate_device(),
        )

    latent = {"samples": samples}
    if spacial_ratio is not None:
        latent["downscale_ratio_spacial"] = spacial_ratio
    return latent


def _build_denoise_latent_from_image(vae, image, family_key):
    """
    Build a LATENT dict from an IMAGE tensor using the active VAE.
    Used when denoise < 1.0 so the latent is compatible with the current model family.
    """
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        return None

    # Comfy IMAGE tensors are BHWC in [0,1]. Keep RGB channels only.
    image_rgb = image[..., :3].contiguous()

    try:
        if family_key == "wan_image" and image_rgb.shape[0] > 1:
            # Some WAN VAEs only encode the first item of a BHWC batch.
            # Encode per item and concatenate to preserve full batch behavior.
            encoded_parts = []
            for i in range(image_rgb.shape[0]):
                part = vae.encode(image_rgb[i:i + 1].contiguous())
                if not isinstance(part, torch.Tensor):
                    return None
                if part.ndim == 4:
                    part = part.unsqueeze(2)
                elif part.ndim != 5:
                    return None
                encoded_parts.append(part)
            encoded = torch.cat(encoded_parts, dim=0)
        else:
            encoded = vae.encode(image_rgb)
    except Exception as e:
        print(f"[RecipeRenderer] Failed to VAE-encode IMAGE for denoise latent: {e}")
        return None

    if not isinstance(encoded, torch.Tensor):
        return None

    # WAN image path uses video-native latent shape [B, C, T, H, W] with T=1.
    if family_key == "wan_image" and encoded.ndim == 4:
        encoded = encoded.unsqueeze(2)

    latent = {"samples": encoded}
    if family_key == "wan_image":
        latent["downscale_ratio_spacial"] = 8
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


def _run_standard_ksampler(model, cond_pos, cond_neg, latent_dict, sampler_params, denoise_override=None):
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
    denoise      = float(sampler_params.get("denoise", 1.0)) if denoise_override is None else float(denoise_override)
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
        denoise=denoise, disable_noise=False,
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

    steps_high  = int(sampler_params.get("steps_high", 3))
    steps_low   = int(sampler_params.get("steps_low", steps_high))
    total_steps = steps_high + steps_low

    cfg          = float(sampler_params.get("cfg", 1.0))
    seed_a       = int(sampler_params.get("seed_a", 0))
    seed_b       = int(sampler_params.get("seed_b", seed_a))
    sampler_name = sampler_params.get("sampler_name", "euler")
    scheduler    = sampler_params.get("scheduler", "simple")

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
                 input_latent=None,
                 loras_a=None, lora_overrides=None, lora_stack_key=''):
    """
    SDXL render.
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

    use_input_latent = isinstance(input_latent, dict) and ("samples" in input_latent)
    latent = input_latent if use_input_latent else _make_latent(4, width, height, batch=batch)
    denoise = float(sampler_params.get("denoise", 1.0)) if use_input_latent else 1.0
    latent_out = _run_standard_ksampler(model, cond_pos, cond_neg, latent, sampler_params, denoise_override=denoise)
    decoded = _decode_latent_output(vae, latent_out, tag="_render_sdxl")
    return decoded, latent_out


def _render_qwen_image(model, clip, vae, pos_prompt, neg_prompt,
                       width, height, batch, sampler_params,
                       input_latent=None,
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

    use_input_latent = isinstance(input_latent, dict) and ("samples" in input_latent)
    if use_input_latent:
        latent = input_latent
    else:
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

    denoise = float(sampler_params.get("denoise", 1.0)) if use_input_latent else 1.0
    latent_out = _run_standard_ksampler(
        model, cond_pos, cond_neg, latent, sampler_params, denoise_override=denoise
    )

    decoded = _decode_latent_output(vae, latent_out, tag="_render_qwen_image")
    return decoded, latent_out


def _render_flux1(model, clip, vae, pos_prompt, neg_prompt,
                  width, height, batch, sampler_params,
                  input_latent=None,
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

    use_input_latent = isinstance(input_latent, dict) and ("samples" in input_latent)
    latent = input_latent if use_input_latent else _make_latent(16, width, height, batch=batch, spacial_ratio=16)
    denoise = float(sampler_params.get("denoise", 1.0)) if use_input_latent else 1.0
    latent_out = _run_standard_ksampler(model, cond_pos, cond_neg, latent, sampler_params, denoise_override=denoise)
    decoded = _decode_latent_output(vae, latent_out, tag="_render_flux1")
    return decoded, latent_out


def _render_flux2(model, clip, vae, pos_prompt, neg_prompt,
                  width, height, batch, sampler_params,
                  input_latent=None,
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

    use_input_latent = isinstance(input_latent, dict) and ("samples" in input_latent)
    latent = input_latent if use_input_latent else _make_latent(16, width, height, batch=batch, spacial_ratio=16)

    steps        = int(sampler_params.get("steps", 4))
    cfg          = float(sampler_params.get("cfg", 1.0))
    seed         = int(sampler_params.get("seed_a", sampler_params.get("seed", 0)))
    sampler_name = sampler_params.get("sampler_name", "euler")
    scheduler    = sampler_params.get("scheduler", "simple")

    guider = _extract_node_outputs(CFGGuider.execute(model, cond_pos, cond_neg, cfg))[0]
    sampler = _extract_node_outputs(KSamplerSelect.execute(sampler_name))[0]
    denoise = float(sampler_params.get("denoise", 1.0)) if use_input_latent else 1.0
    sigmas = _extract_node_outputs(BasicScheduler.execute(model, scheduler, steps, denoise))[0]
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


def _render_ernie(model, clip, vae, pos_prompt, neg_prompt,
                  width, height, batch, sampler_params,
                  input_latent=None,
                  loras_a=None, lora_overrides=None, lora_stack_key=''):
    """
    Ernie render path aligned to the reference Ernie workflow:
    CLIPTextEncode -> ConditioningZeroOut -> KSampler -> VAEDecode
    with EmptyFlux2LatentImage-like latent shape (spacial_ratio=16).
    """
    from nodes import ConditioningZeroOut

    if loras_a:
        model, clip = _apply_loras(
            model, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key
        )

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)

    # Match the reference workflow: negative conditioning is zeroed from the
    # encoded positive path rather than encoded from a separate negative text.
    (cond_neg,) = ConditioningZeroOut().zero_out(cond_pos)

    use_input_latent = isinstance(input_latent, dict) and ("samples" in input_latent)
    latent = input_latent if use_input_latent else _make_latent(16, width, height, batch=batch, spacial_ratio=16)
    denoise = float(sampler_params.get("denoise", 1.0)) if use_input_latent else 1.0
    latent_out = _run_standard_ksampler(
        model, cond_pos, cond_neg, latent, sampler_params, denoise_override=denoise
    )
    decoded = _decode_latent_output(vae, latent_out, tag="_render_ernie")
    return decoded, latent_out


def _render_zimage(model, clip, vae, pos_prompt, neg_prompt,
                   width, height, batch, sampler_params,
                   input_latent=None,
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

    use_input_latent = isinstance(input_latent, dict) and ("samples" in input_latent)
    if use_input_latent:
        latent = input_latent
    else:
        latent_node = EmptySD3LatentImage()
        if hasattr(latent_node, "generate"):
            latent = latent_node.generate(width=width, height=height, batch_size=batch)[0]
        else:
            latent = _extract_node_outputs(
                EmptySD3LatentImage.execute(width=width, height=height, batch_size=batch)
            )[0]

    denoise = float(sampler_params.get("denoise", 1.0)) if use_input_latent else 1.0
    latent_out = _run_standard_ksampler(
        model, cond_pos, cond_neg, latent, sampler_params, denoise_override=denoise
    )

    decoded = _decode_latent_output(vae, latent_out, tag="_render_zimage")
    return decoded, latent_out


def _render_wan_image(model, clip, vae, pos_prompt, neg_prompt,
                      width, height, batch, sampler_params,
                      input_latent=None,
                      loras_a=None, lora_overrides=None, lora_stack_key=''):
    """
    WAN image render using a video-native latent with T=1.
    """
    if loras_a:
        model, clip = _apply_loras(
            model, clip, loras_a, lora_overrides or {}, stack_key=lora_stack_key
        )

    # Keep WAN image aligned with WAN video sampling path.
    model = _patch_model_sampling_sd3(model, shift=5.0)

    tokens_pos = clip.tokenize(pos_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)
    tokens_neg = clip.tokenize(neg_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    use_input_latent = isinstance(input_latent, dict) and ("samples" in input_latent)
    latent = input_latent if use_input_latent else _make_latent(16, width, height, batch=batch, frames=1, spacial_ratio=8)
    denoise = float(sampler_params.get("denoise", 1.0)) if use_input_latent else 1.0
    latent_out = _run_standard_ksampler(model, cond_pos, cond_neg, latent, sampler_params, denoise_override=denoise)
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
