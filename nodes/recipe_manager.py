"""
Workflow Manager - stripped-down workflow_data editor based on PromptManagerAdvanced.

Differences from PromptManagerAdvanced:
- workflow_data input/output only (no prompt/lora stack outputs)
- no use_* switches
- no trigger words pipeline
"""

import json
import base64
from io import BytesIO

import server

from ..py.workflow_data_utils import ensure_v2_recipe_data, get_v2_model_block, to_json_safe_workflow_data, build_v2_recipe_data_from_prompt
from .prompt_manager_adv import PromptManagerAdvanced

try:
    import numpy as np
    from PIL import Image
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False


def _image_to_base64_thumbnail(image_tensor, max_size=200):
    if not IMAGE_SUPPORT or image_tensor is None:
        return None

    try:
        img_array = image_tensor[0] if len(image_tensor.shape) == 4 else image_tensor
        if hasattr(img_array, "cpu"):
            img_array = img_array.cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        width, height = img.size
        min_dim = min(width, height)
        if min_dim > max_size:
            scale = max_size / min_dim
            img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}"
    except Exception as e:
        print(f"[RecipeManager] Thumbnail generation failed: {e}")
        return None


class WorkflowManager(PromptManagerAdvanced):
    """Workflow-focused manager that edits and forwards RECIPE_DATA."""

    @classmethod
    def INPUT_TYPES(cls):
        prompts_data = cls.load_prompts()
        categories = list(prompts_data.keys()) if prompts_data else ["Default"]

        all_prompts = set()
        for category_prompts in prompts_data.values():
            all_prompts.update(k for k in category_prompts.keys() if k != "__meta__")

        all_prompts.add("")
        all_prompts_list = sorted(list(all_prompts))

        first_category = categories[0] if categories else "Default"

        # Keep new node instances empty until the user explicitly selects a prompt.
        first_prompt = ""
        first_prompt_text = ""

        return {
            "required": {
                "category": (categories, {"default": first_category}),
                "name": (all_prompts_list, {"default": first_prompt}),
                "text": ("STRING", {
                    "multiline": True,
                    "default": first_prompt_text,
                    "placeholder": "Enter prompt text",
                    "dynamicPrompts": False,
                    "tooltip": "Edit prompt text directly",
                }),
            },
            "optional": {
                "recipe_data": ("RECIPE_DATA", {"forceInput": True, "tooltip": "Connected recipe_data to edit/forward"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "loras_a_toggle": "STRING",
                "loras_b_toggle": "STRING",
                "saved_workflow_data": "STRING",
            },
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Workflow-data focused manager (no use switches, no trigger words)."
    RETURN_TYPES = ("RECIPE_DATA",)
    RETURN_NAMES = ("recipe_data",)
    FUNCTION = "get_workflow"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, category, name, text="", **kwargs):
        recipe_data = kwargs.get("recipe_data", None)
        loras_a_toggle = kwargs.get("loras_a_toggle", "")
        loras_b_toggle = kwargs.get("loras_b_toggle", "")
        saved_workflow_data = kwargs.get("saved_workflow_data", "")
        return (
            category,
            name,
            text,
            str(recipe_data) if recipe_data else None,
            loras_a_toggle or "",
            loras_b_toggle or "",
            saved_workflow_data or "",
        )

    def get_workflow(
        self,
        category,
        name,
        text="",
        recipe_data=None,
        unique_id=None,
        loras_a_toggle=None,
        loras_b_toggle=None,
        saved_workflow_data=None,
    ):
        prompts_data = self.load_prompts()

        def _as_workflow_dict(raw):
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                    return parsed if isinstance(parsed, dict) else None
                except (json.JSONDecodeError, TypeError):
                    return None
            return None

        prompt_entry = prompts_data.get(category, {}).get(name, {}) if isinstance(prompts_data, dict) else {}
        # On reload, category/name can be temporarily out of sync in the UI.
        # If direct lookup misses, find the selected name in any category.
        if not isinstance(prompt_entry, dict) or not prompt_entry:
            if isinstance(prompts_data, dict) and isinstance(name, str) and name:
                for cat_prompts in prompts_data.values():
                    if isinstance(cat_prompts, dict) and isinstance(cat_prompts.get(name), dict):
                        prompt_entry = cat_prompts.get(name)
                        break

        stored_prompt_wf = _as_workflow_dict(prompt_entry.get("workflow_data")) if isinstance(prompt_entry, dict) else None

        live_workflow_data = _as_workflow_dict(recipe_data)

        hidden_saved_wf = _as_workflow_dict(saved_workflow_data)

        # WorkflowManager should prefer live connected workflow_data when present,
        # then fall back to local serialized state.
        resolved_workflow_data = ensure_v2_recipe_data(live_workflow_data, source="RecipeManager") if isinstance(live_workflow_data, dict) else None
        if resolved_workflow_data is None:
            resolved_workflow_data = ensure_v2_recipe_data(hidden_saved_wf, source="RecipeManager") if isinstance(hidden_saved_wf, dict) else None
        if resolved_workflow_data is None and isinstance(stored_prompt_wf, dict):
            resolved_workflow_data = ensure_v2_recipe_data(stored_prompt_wf, source="RecipeManager")

        # Prompt-only fallback: allow RecipeManager to open saved PMA prompts
        # (prompt + lora stacks) by synthesizing minimal v2 recipe_data.
        if resolved_workflow_data is None and isinstance(prompt_entry, dict):
            prompt_text = str(prompt_entry.get("prompt", "") or text or "")
            negative_text = str(prompt_entry.get("negative_prompt", "") or "")
            prompt_loras_a = prompt_entry.get("loras_a", []) if isinstance(prompt_entry.get("loras_a"), list) else []
            prompt_loras_b = prompt_entry.get("loras_b", []) if isinstance(prompt_entry.get("loras_b"), list) else []

            has_prompt_payload = bool(
                prompt_text.strip() or
                negative_text.strip() or
                len(prompt_loras_a) > 0 or
                len(prompt_loras_b) > 0
            )

            if has_prompt_payload:
                resolved_workflow_data = build_v2_recipe_data_from_prompt(
                    prompt_text=prompt_text,
                    negative_prompt=negative_text,
                    loras_a=prompt_loras_a,
                    loras_b=prompt_loras_b,
                    source="RecipeManager",
                )

        wf = resolved_workflow_data if isinstance(resolved_workflow_data, dict) else {}
        wf_model_a = get_v2_model_block(wf, "model_a") or {}
        wf_model_b = get_v2_model_block(wf, "model_b") or {}
        wf_has_model_b = isinstance(get_v2_model_block(wf, "model_b"), dict)
        incoming_wf = live_workflow_data if isinstance(live_workflow_data, dict) else None
        incoming_has_model_b = None
        if isinstance(incoming_wf, dict):
            incoming_v2 = ensure_v2_recipe_data(incoming_wf, source="RecipeManager")
            incoming_models = incoming_v2.get("models", {}) if isinstance(incoming_v2.get("models"), dict) else {}
            incoming_has_model_b = isinstance(incoming_models.get("model_b"), dict)
        output_text = (wf_model_a.get("positive_prompt", "") or text or "")
        generated_thumbnail = _image_to_base64_thumbnail(wf.get("IMAGE")) if isinstance(wf, dict) else None
        incoming_thumbnail = _image_to_base64_thumbnail(incoming_wf.get("IMAGE")) if isinstance(incoming_wf, dict) else None
        workflow_thumbnail = incoming_thumbnail if isinstance(incoming_thumbnail, str) and incoming_thumbnail else (
            generated_thumbnail if isinstance(generated_thumbnail, str) and generated_thumbnail else (
                incoming_wf.get("thumbnail") if isinstance(incoming_wf, dict) and isinstance(incoming_wf.get("thumbnail"), str) else (
                    wf.get("thumbnail") if isinstance(wf.get("thumbnail"), str) else None
                )
            )
        )

        # Build preset stacks from saved toggle state.
        preset_stack_a = self._build_stack_from_toggle(loras_a_toggle) if loras_a_toggle else []
        preset_stack_b = self._build_stack_from_toggle(loras_b_toggle) if (wf_has_model_b and loras_b_toggle) else []

        all_preset_loras_a = self._get_all_loras_from_toggle(loras_a_toggle) if loras_a_toggle else []
        all_preset_loras_b = self._get_all_loras_from_toggle(loras_b_toggle) if (wf_has_model_b and loras_b_toggle) else []

        wf_loras_a = [
            (lora["name"], lora.get("model_strength", 1.0), lora.get("clip_strength", 1.0))
            for lora in wf_model_a.get("loras", [])
            if isinstance(lora, dict) and lora.get("name")
        ]
        wf_loras_b = []
        if wf_has_model_b:
            wf_loras_b = [
                (lora["name"], lora.get("model_strength", 1.0), lora.get("clip_strength", 1.0))
                for lora in wf_model_b.get("loras", [])
                if isinstance(lora, dict) and lora.get("name")
            ]

        # WorkflowManager is workflow-data centric: preserve connected workflow order
        # and use toggles only to modify active/strength state for those entries.
        # Fallback to preset stack only when no workflow LoRAs are present.
        if wf_loras_a:
            merged_stack_a = list(wf_loras_a)
        else:
            merged_stack_a = list(preset_stack_a) if preset_stack_a else []

        if wf_has_model_b:
            if wf_loras_b:
                merged_stack_b = list(wf_loras_b)
            else:
                merged_stack_b = list(preset_stack_b) if preset_stack_b else []
        else:
            merged_stack_b = []

        processed_stack_a = self._process_lora_toggle(merged_stack_a, loras_a_toggle)
        processed_stack_b = self._process_lora_toggle(merged_stack_b, loras_b_toggle)

        loras_a_display = self._format_loras_for_display_with_unavailable(merged_stack_a, all_preset_loras_a)
        loras_b_display = self._format_loras_for_display_with_unavailable(merged_stack_b, all_preset_loras_b)

        unavailable_loras_a = [lora["name"] for lora in all_preset_loras_a if not lora.get("available", False)]
        unavailable_loras_b = [lora["name"] for lora in all_preset_loras_b if not lora.get("available", False)]

        if unique_id is not None:
            server.PromptServer.instance.send_sync("prompt-manager-advanced-update", {
                "node_id": unique_id,
                "prompt": output_text,
                "use_prompt_input": False,
                "use_workflow_data": True,
                "prompt_input": "",
                "workflow_data": to_json_safe_workflow_data(ensure_v2_recipe_data(recipe_data, source="RecipeManager")) if isinstance(recipe_data, dict) else (
                    to_json_safe_workflow_data(resolved_workflow_data) if isinstance(resolved_workflow_data, dict) else None
                ),
                "loras_a": loras_a_display,
                "loras_b": loras_b_display,
                "input_loras_a": self._format_loras_for_display(wf_loras_a) if wf_loras_a else [],
                "input_loras_b": self._format_loras_for_display(wf_loras_b) if wf_loras_b else [],
                "unavailable_loras_a": unavailable_loras_a,
                "unavailable_loras_b": unavailable_loras_b,
                "trigger_words": [],
                "connected_thumbnail": workflow_thumbnail,
                "should_reset": False,
                "original_strengths_a": {
                    lora_item.get("name", ""): lora_item.get("strength", lora_item.get("model_strength", 1.0))
                    for lora_item in loras_a_display
                    if lora_item.get("name")
                },
                "original_strengths_b": {
                    lora_item.get("name", ""): lora_item.get("strength", lora_item.get("model_strength", 1.0))
                    for lora_item in loras_b_display
                    if lora_item.get("name")
                },
            })

        out_workflow_data = ensure_v2_recipe_data(
            dict(resolved_workflow_data) if isinstance(resolved_workflow_data, dict) else {},
            source="RecipeManager",
        )

        models_out = out_workflow_data.get("models", {})
        if not isinstance(models_out, dict):
            models_out = {}
            out_workflow_data["models"] = models_out

        has_existing_model_b = isinstance(models_out.get("model_b"), dict)
        model_a_out = models_out.get("model_a") if isinstance(models_out.get("model_a"), dict) else {}
        model_b_out = models_out.get("model_b") if isinstance(models_out.get("model_b"), dict) else {}

        model_a_out["positive_prompt"] = output_text
        model_a_out["loras"] = [
            {
                "name": lora.get("name", ""),
                "path": lora.get("path", lora.get("name", "")),
                "model_strength": lora.get("model_strength", lora.get("strength", 1.0)),
                "clip_strength": lora.get("clip_strength", lora.get("strength", 1.0)),
                "active": lora.get("active", True),
                "available": lora.get("available", True),
            }
            for lora in loras_a_display
            if isinstance(lora, dict) and lora.get("name")
        ]
        model_b_loras = [
            {
                "name": lora.get("name", ""),
                "path": lora.get("path", lora.get("name", "")),
                "model_strength": lora.get("model_strength", lora.get("strength", 1.0)),
                "clip_strength": lora.get("clip_strength", lora.get("strength", 1.0)),
                "active": lora.get("active", True),
                "available": lora.get("available", True),
            }
            for lora in loras_b_display
            if isinstance(lora, dict) and lora.get("name")
        ]

        models_out["model_a"] = model_a_out
        model_a_name = str(model_a_out.get("model", "") or "").strip()
        model_b_name = str(model_b_out.get("model", "") or "").strip()
        model_b_pos = str(model_b_out.get("positive_prompt", "") or "").strip()
        model_b_neg = str(model_b_out.get("negative_prompt", "") or "").strip()
        model_b_has_runtime = any(
            model_b_out.get(k) is not None
            for k in ("MODEL", "CLIP", "VAE", "POSITIVE", "NEGATIVE")
        )
        # model_b is considered meaningful when it has authored content
        # that is not just a mirrored default of model_a.
        keep_model_b = bool(model_b_loras) or bool(model_b_pos) or bool(model_b_neg) or model_b_has_runtime
        if model_b_name and model_b_name != model_a_name:
            keep_model_b = True

        if (has_existing_model_b and model_b_name and model_b_name == model_a_name and
                not model_b_loras and not model_b_pos and not model_b_neg and not model_b_has_runtime):
            keep_model_b = False

        # Canonical rule: if incoming recipe_data had no model_b, scrub model_b from output.
        if incoming_has_model_b is False:
            keep_model_b = False

        if keep_model_b:
            model_b_out["loras"] = model_b_loras
            models_out["model_b"] = model_b_out
        else:
            models_out.pop("model_b", None)
        out_workflow_data["_source"] = "RecipeManager"

        return (out_workflow_data,)

    def check_lazy_status(self, category, name, text="", **kwargs):
        return []
