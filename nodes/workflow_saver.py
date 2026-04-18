"""
Workflow Saver - Save workflow_data snapshots into prompt library entries.

MVP goals:
- Capture live workflow_data from execution.
- Save to Prompt Manager data with optional overwrite confirmation.
- Auto-generate thumbnail from workflow_data['IMAGE'].
"""
import base64
from io import BytesIO
from datetime import datetime

import server

from ..py.workflow_data_utils import to_json_safe_workflow_data
from .prompt_manager_adv import PromptManagerAdvanced

try:
    import numpy as np
    from PIL import Image
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False


_last_workflow_saver_snapshot = {}


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
        print(f"[WorkflowSaver] Thumbnail generation failed: {e}")
        return None


def _normalize_lora_data(loras):
    out = []
    if not isinstance(loras, list):
        return out
    for lora in loras:
        if not isinstance(lora, dict):
            continue
        name = str(lora.get("name", "")).strip()
        if not name:
            continue
        strength = float(lora.get("strength", lora.get("model_strength", 1.0)))
        clip_strength = float(lora.get("clip_strength", strength))
        available = bool(lora.get("available", lora.get("found", True)))
        found = bool(lora.get("found", available))
        out.append({
            "name": name,
            "strength": strength,
            "clip_strength": clip_strength,
            "active": bool(lora.get("active", True)),
            "available": available,
            "found": found,
        })
    return out


class WorkflowSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_data": ("WORKFLOW_DATA", {
                    "forceInput": True,
                    "tooltip": "Connect workflow_data from WorkflowBuilder/Renderer/Bridge.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("WORKFLOW_DATA",)
    RETURN_NAMES = ("workflow_data",)
    FUNCTION = "execute"
    CATEGORY = "Prompt Manager"
    OUTPUT_NODE = True
    DESCRIPTION = "Save-ready workflow snapshot node. Use the Save Snapshot button in node UI."

    def execute(self, workflow_data, unique_id=None):
        if isinstance(workflow_data, dict):
            wf = workflow_data
        elif isinstance(workflow_data, str):
            try:
                import json
                wf = json.loads(workflow_data)
                if not isinstance(wf, dict):
                    wf = {}
            except Exception:
                wf = {}
        else:
            wf = {}

        wf_safe = to_json_safe_workflow_data(wf)
        img_for_thumb = wf.get("IMAGE")
        thumbnail = _image_to_base64_thumbnail(img_for_thumb)

        snapshot = {
            "positive_prompt": str(wf.get("positive_prompt", "") or ""),
            "negative_prompt": str(wf.get("negative_prompt", "") or ""),
            "loras_a": _normalize_lora_data(wf.get("loras_a", [])),
            "loras_b": _normalize_lora_data(wf.get("loras_b", [])),
            "workflow_data": wf_safe,
            "thumbnail": thumbnail,
            "saved_from": "WorkflowSaver",
            "captured_at": datetime.utcnow().isoformat() + "Z",
        }

        if unique_id is not None:
            _last_workflow_saver_snapshot[str(unique_id)] = snapshot

        return {
            "ui": {
                "workflow_saver": [{
                    "node_id": str(unique_id) if unique_id is not None else "",
                    "has_snapshot": True,
                    "has_thumbnail": bool(thumbnail),
                }]
            },
            "result": (wf,)
        }


@server.PromptServer.instance.routes.get("/workflow-saver/list")
async def workflow_saver_list(request):
    try:
        prompts = PromptManagerAdvanced.load_prompts() or {}
        categories = sorted(list(prompts.keys()), key=str.lower)
        names_by_category = {}
        for category in categories:
            cdata = prompts.get(category, {})
            if isinstance(cdata, dict):
                names = [k for k in cdata.keys() if k != "__meta__"]
            else:
                names = []
            names_by_category[category] = sorted(names, key=str.lower)

        return server.web.json_response({
            "success": True,
            "categories": categories,
            "names_by_category": names_by_category,
        })
    except Exception as e:
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.get("/workflow-saver/snapshot")
async def workflow_saver_snapshot(request):
    try:
        node_id = str(request.rel_url.query.get("node_id", "") or "").strip()
        if not node_id:
            return server.web.json_response({"success": False, "error": "Missing node_id"}, status=400)

        snap = _last_workflow_saver_snapshot.get(node_id)
        if not snap:
            return server.web.json_response({
                "success": False,
                "error": "No snapshot cached for this node. Execute the workflow first.",
            }, status=404)

        return server.web.json_response({"success": True, "snapshot": snap})
    except Exception as e:
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/workflow-saver/save")
async def workflow_saver_save(request):
    try:
        data = await request.json()
        node_id = str(data.get("node_id", "") or "").strip()
        category = str(data.get("category", "") or "").strip()
        name = str(data.get("name", "") or "").strip()
        overwrite = bool(data.get("overwrite", False))

        if not node_id or not category or not name:
            return server.web.json_response({
                "success": False,
                "error": "node_id, category, and name are required",
            }, status=400)

        snapshot = _last_workflow_saver_snapshot.get(node_id)
        if not snapshot:
            return server.web.json_response({
                "success": False,
                "error": "No snapshot available for this node. Execute first.",
            }, status=404)

        prompts = PromptManagerAdvanced.load_prompts() or {}
        if category not in prompts or not isinstance(prompts.get(category), dict):
            prompts[category] = {}

        exists = name in prompts[category]
        if exists and not overwrite:
            return server.web.json_response({
                "success": False,
                "exists": True,
                "error": "Entry already exists",
            })

        existing = prompts[category].get(name, {}) if exists else {}

        prompt_data = {
            "prompt": snapshot.get("positive_prompt", ""),
            "negative_prompt": snapshot.get("negative_prompt", ""),
            "loras_a": snapshot.get("loras_a", []),
            "loras_b": snapshot.get("loras_b", []),
            "trigger_words": existing.get("trigger_words", []),
            "workflow_data": to_json_safe_workflow_data(snapshot.get("workflow_data", {})),
        }

        thumb = snapshot.get("thumbnail")
        if thumb:
            prompt_data["thumbnail"] = thumb
        elif isinstance(existing, dict) and existing.get("thumbnail"):
            prompt_data["thumbnail"] = existing.get("thumbnail")

        # Preserve optional nsfw flag from existing entry.
        if isinstance(existing, dict) and "nsfw" in existing:
            prompt_data["nsfw"] = existing.get("nsfw")

        prompt_data["saved_from"] = "WorkflowSaver"
        prompt_data["saved_at"] = datetime.utcnow().isoformat() + "Z"

        prompts[category][name] = prompt_data
        PromptManagerAdvanced.save_prompts(prompts)

        return server.web.json_response({
            "success": True,
            "saved": {
                "category": category,
                "name": name,
                "overwritten": bool(exists),
                "has_thumbnail": bool(prompt_data.get("thumbnail")),
            }
        })
    except Exception as e:
        return server.web.json_response({"success": False, "error": str(e)}, status=500)
