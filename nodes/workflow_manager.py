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

from ..py.workflow_data_utils import to_json_safe_workflow_data
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
        print(f"[WorkflowManager] Thumbnail generation failed: {e}")
        return None


class WorkflowManager(PromptManagerAdvanced):
    """Workflow-focused manager that edits and forwards WORKFLOW_DATA."""

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

        first_prompt = ""
        first_prompt_text = ""
        if prompts_data and first_category in prompts_data and prompts_data[first_category]:
            first_category_prompts = [k for k in prompts_data[first_category].keys() if k != "__meta__"]
            first_prompt = sorted(first_category_prompts, key=str.lower)[0] if first_category_prompts else ""
            if first_prompt:
                first_prompt_text = prompts_data[first_category][first_prompt].get("prompt", "")

        return {
            "required": {
                "category": (categories, {"default": first_category}),
                "name": (all_prompts_list, {"default": first_prompt}),
                "text": ("STRING", {
                    "multiline": True,
                    "default": first_prompt_text,
                    "placeholder": "Enter prompt text",
                    "dynamicPrompts": False,
                    "tooltip": "Edit workflow_data prompt text directly",
                }),
            },
            "optional": {
                "workflow_data": ("WORKFLOW_DATA", {"forceInput": True, "tooltip": "Connected workflow_data to edit/forward"}),
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
    RETURN_TYPES = ("WORKFLOW_DATA",)
    RETURN_NAMES = ("workflow_data",)
    FUNCTION = "get_workflow"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, category, name, text="", **kwargs):
        workflow_data = kwargs.get("workflow_data", None)
        loras_a_toggle = kwargs.get("loras_a_toggle", "")
        loras_b_toggle = kwargs.get("loras_b_toggle", "")
        saved_workflow_data = kwargs.get("saved_workflow_data", "")
        return (
            category,
            name,
            text,
            str(workflow_data) if workflow_data else None,
            loras_a_toggle or "",
            loras_b_toggle or "",
            saved_workflow_data or "",
        )

    def get_workflow(
        self,
        category,
        name,
        text="",
        workflow_data=None,
        unique_id=None,
        loras_a_toggle=None,
        loras_b_toggle=None,
        saved_workflow_data=None,
    ):
        prompts_data = self.load_prompts()
        prompt_entry = prompts_data.get(category, {}).get(name, {}) if isinstance(prompts_data, dict) else {}
        stored_prompt_wf = prompt_entry.get("workflow_data") if isinstance(prompt_entry, dict) else None

        live_workflow_data = None
        if isinstance(workflow_data, dict):
            live_workflow_data = workflow_data
        elif isinstance(workflow_data, str) and workflow_data.strip():
            try:
                parsed = json.loads(workflow_data)
                if isinstance(parsed, dict):
                    live_workflow_data = parsed
            except (json.JSONDecodeError, TypeError):
                live_workflow_data = None

        hidden_saved_wf = None
        if isinstance(saved_workflow_data, str) and saved_workflow_data.strip():
            try:
                parsed_saved = json.loads(saved_workflow_data)
                if isinstance(parsed_saved, dict):
                    hidden_saved_wf = parsed_saved
            except (json.JSONDecodeError, TypeError):
                hidden_saved_wf = None

        resolved_workflow_data = live_workflow_data or hidden_saved_wf
        if resolved_workflow_data is None and isinstance(stored_prompt_wf, dict):
            resolved_workflow_data = stored_prompt_wf

        wf = resolved_workflow_data if isinstance(resolved_workflow_data, dict) else {}
        output_text = (wf.get("positive_prompt", "") or text or "")
        generated_thumbnail = _image_to_base64_thumbnail(wf.get("IMAGE")) if isinstance(wf, dict) else None
        workflow_thumbnail = generated_thumbnail if isinstance(generated_thumbnail, str) and generated_thumbnail else (
            wf.get("thumbnail") if isinstance(wf.get("thumbnail"), str) else None
        )

        # Build preset stacks from saved toggle state.
        preset_stack_a = self._build_stack_from_toggle(loras_a_toggle) if loras_a_toggle else []
        preset_stack_b = self._build_stack_from_toggle(loras_b_toggle) if loras_b_toggle else []

        all_preset_loras_a = self._get_all_loras_from_toggle(loras_a_toggle) if loras_a_toggle else []
        all_preset_loras_b = self._get_all_loras_from_toggle(loras_b_toggle) if loras_b_toggle else []

        wf_loras_a = [
            (lora["name"], lora.get("model_strength", 1.0), lora.get("clip_strength", 1.0))
            for lora in wf.get("loras_a", [])
            if isinstance(lora, dict) and lora.get("name")
        ]
        wf_loras_b = [
            (lora["name"], lora.get("model_strength", 1.0), lora.get("clip_strength", 1.0))
            for lora in wf.get("loras_b", [])
            if isinstance(lora, dict) and lora.get("name")
        ]

        # WorkflowManager is workflow-data centric: workflow LoRAs are base, preset toggles merge in.
        merged_stack_a = self._merge_lora_stacks(wf_loras_a, preset_stack_a) if (wf_loras_a or preset_stack_a) else []
        merged_stack_b = self._merge_lora_stacks(wf_loras_b, preset_stack_b) if (wf_loras_b or preset_stack_b) else []

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
                "workflow_data": to_json_safe_workflow_data(resolved_workflow_data) if isinstance(resolved_workflow_data, dict) else None,
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

        out_workflow_data = dict(resolved_workflow_data) if isinstance(resolved_workflow_data, dict) else {}
        out_workflow_data["positive_prompt"] = output_text
        out_workflow_data["loras_a"] = [
            {
                "name": lora.get("name", ""),
                "model_strength": lora.get("model_strength", lora.get("strength", 1.0)),
                "clip_strength": lora.get("clip_strength", lora.get("strength", 1.0)),
                "active": lora.get("active", True),
                "available": lora.get("available", True),
                "found": lora.get("found", lora.get("available", True)),
            }
            for lora in loras_a_display
            if isinstance(lora, dict) and lora.get("name")
        ]
        out_workflow_data["loras_b"] = [
            {
                "name": lora.get("name", ""),
                "model_strength": lora.get("model_strength", lora.get("strength", 1.0)),
                "clip_strength": lora.get("clip_strength", lora.get("strength", 1.0)),
                "active": lora.get("active", True),
                "available": lora.get("available", True),
                "found": lora.get("found", lora.get("available", True)),
            }
            for lora in loras_b_display
            if isinstance(lora, dict) and lora.get("name")
        ]
        out_workflow_data["_source"] = "WorkflowManager"

        return (out_workflow_data,)

    def check_lazy_status(self, category, name, text="", **kwargs):
        return []
