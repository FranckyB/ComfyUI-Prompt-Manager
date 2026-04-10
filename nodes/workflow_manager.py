"""
ComfyUI Workflow Manager
Big-brother of Prompt Manager Advanced that also stores workflow_config
(model family, model_a, model_b, vae, clip) alongside prompts in the
same prompt_manager_data.json file.

backward-compatible — old prompts load fine, WorkflowManager simply reads
and writes the extra `workflow_config` key inside each prompt entry.
PromptManager / PromptManagerAdvanced leave that key untouched thanks to the
preserve-guard added to save_prompt_advanced.
"""
import os
import json
import folder_paths
import server

# Share helpers from PromptManagerAdvanced
from .prompt_manager_advanced import PromptManagerAdvanced
from ..py.workflow_families import (
    get_all_family_labels,
    list_all_models,
    list_compatible_models,
    list_compatible_vaes,
    list_compatible_clips,
)


# ─── API endpoints ──────────────────────────────────────────────────────────

@server.PromptServer.instance.routes.get("/workflow-manager/prompts")
async def wm_get_prompts(request):
    """Return all prompts (same store as PM Advanced), including workflow_config fields."""
    try:
        prompts = PromptManagerAdvanced.load_prompts()
        return server.web.json_response({"prompts": prompts})
    except Exception as e:
        return server.web.json_response({"prompts": {}, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/workflow-manager/save-prompt")
async def wm_save_prompt(request):
    """
    Save a prompt with full workflow_config.
    All base fields from PM Advanced + optional workflow_config block.
    """
    try:
        data          = await request.json()
        category      = data.get("category", "").strip()
        name          = data.get("name", "").strip()
        text          = data.get("text", "").strip()
        loras_a       = data.get("loras_a", [])
        loras_b       = data.get("loras_b", [])
        trigger_words = data.get("trigger_words", [])
        workflow_config = data.get("workflow_config")  # may be None

        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name required"})

        prompts = PromptManagerAdvanced.load_prompts()
        if category not in prompts:
            prompts[category] = {}

        existing_lc = {k.lower(): k for k in prompts[category].keys()}
        existing    = {}
        if name.lower() in existing_lc:
            old_name = existing_lc[name.lower()]
            existing = prompts[category].get(old_name, {})
            if old_name != name:
                del prompts[category][old_name]

        def normalize_loras(loras):
            out = []
            for lr in loras:
                if isinstance(lr, dict) and lr.get('name'):
                    out.append({
                        "name":          lr.get('name'),
                        "strength":      lr.get('strength', lr.get('model_strength', 1.0)),
                        "clip_strength": lr.get('clip_strength', lr.get('strength', 1.0)),
                        "active":        lr.get('active', True),
                    })
            return out

        def normalize_trigger_words(words):
            out  = []
            seen = set()
            for w in words:
                if isinstance(w, dict) and w.get('text'):
                    t = w['text'].strip()
                    if t and t.lower() not in seen:
                        out.append({"text": t, "active": w.get('active', True)})
                        seen.add(t.lower())
                elif isinstance(w, str) and w.strip():
                    t = w.strip()
                    if t.lower() not in seen:
                        out.append({"text": t, "active": True})
                        seen.add(t.lower())
            return out

        # Thumbnail
        thumbnail = data.get("thumbnail")
        if thumbnail is None:
            thumbnail = existing.get("thumbnail")

        prompt_data = {
            "prompt":        text,
            "loras_a":       normalize_loras(loras_a),
            "loras_b":       normalize_loras(loras_b),
            "trigger_words": normalize_trigger_words(trigger_words),
        }
        if thumbnail:
            prompt_data["thumbnail"] = thumbnail

        nsfw = data.get("nsfw")
        if nsfw is not None:
            prompt_data["nsfw"] = bool(nsfw)
        elif existing.get("nsfw"):
            prompt_data["nsfw"] = existing["nsfw"]

        # Workflow config — save if provided, preserve existing if not
        if workflow_config is not None:
            prompt_data["workflow_config"] = workflow_config
        elif existing.get("workflow_config"):
            prompt_data["workflow_config"] = existing["workflow_config"]

        # Preserve any other unknown extension keys
        _MANAGED = {"prompt", "loras_a", "loras_b", "trigger_words", "thumbnail", "nsfw", "workflow_config"}
        for k, v in existing.items():
            if k not in _MANAGED:
                prompt_data[k] = v

        prompts[category][name] = prompt_data
        PromptManagerAdvanced.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[WorkflowManager] Save error: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.get("/workflow-manager/families")
async def wm_list_families(request):
    """Return all model families."""
    return server.web.json_response({"families": get_all_family_labels()})


@server.PromptServer.instance.routes.get("/workflow-manager/list-models")
async def wm_list_models(request):
    """List models, optionally filtered by ?family=<key>."""
    try:
        family = request.rel_url.query.get('family', '') or None
        if family:
            models = list_compatible_models(family)
        else:
            models = list_all_models()
        return server.web.json_response({"models": models})
    except Exception as e:
        return server.web.json_response({"models": [], "error": str(e)})


@server.PromptServer.instance.routes.get("/workflow-manager/list-vaes")
async def wm_list_vaes(request):
    family = request.rel_url.query.get('family', '') or None
    return server.web.json_response({"vaes": list_compatible_vaes(family)})


@server.PromptServer.instance.routes.get("/workflow-manager/list-clips")
async def wm_list_clips(request):
    family = request.rel_url.query.get('family', '') or None
    return server.web.json_response({"clips": list_compatible_clips(family)})


# ─── Node ───────────────────────────────────────────────────────────────────

class WorkflowManager:
    """
    Workflow Manager — Prompt Manager Advanced extended with full workflow config.
    Stores model family, model A/B, VAE, CLIP alongside prompts in the same JSON store.
    Outputs: prompt (STRING), lora_stack_a (LORA_STACK), lora_stack_b (LORA_STACK),
             workflow_dict (WORKFLOW_DICT — feeds into WorkflowGenerator).
    """

    @classmethod
    def INPUT_TYPES(cls):
        prompts_data = PromptManagerAdvanced.load_prompts()
        categories   = sorted(prompts_data.keys()) if prompts_data else ["Default"]
        all_prompts  = set()
        for cat_data in prompts_data.values():
            all_prompts.update(k for k in cat_data.keys() if k != "__meta__")
        all_prompts.add("")
        all_prompts_list = sorted(list(all_prompts))

        first_category = categories[0] if categories else "Default"
        first_prompt   = ""
        first_text     = ""
        if prompts_data and first_category in prompts_data:
            cat_keys = [k for k in prompts_data[first_category].keys() if k != "__meta__"]
            first_prompt = sorted(cat_keys, key=str.lower)[0] if cat_keys else ""
            if first_prompt:
                first_text = prompts_data[first_category][first_prompt].get("prompt", "")

        # Build family choices for the dropdown
        family_labels = get_all_family_labels()  # {key: label}
        family_keys   = [""] + sorted(family_labels.keys())

        return {
            "required": {
                "category":        (categories,        {"default": first_category}),
                "name":            (all_prompts_list,  {"default": first_prompt}),
                "use_prompt_input": ("BOOLEAN", {
                    "default": False,
                    "label_on": "on", "label_off": "off",
                    "tooltip": "Use connected prompt input instead of stored text",
                }),
                "use_lora_input": ("BOOLEAN", {
                    "default": False,
                    "label_on": "on", "label_off": "off",
                    "tooltip": "Merge connected LoRA stacks with stored LoRAs",
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": first_text,
                    "placeholder": "Enter prompt text",
                    "dynamicPrompts": False,
                }),
            },
            "optional": {
                "prompt_input":  ("STRING",     {"multiline": True, "forceInput": True, "lazy": True}),
                "lora_stack_a":  ("LORA_STACK", {}),
                "lora_stack_b":  ("LORA_STACK", {}),
                "trigger_words": ("STRING",     {"forceInput": True}),
                "thumbnail_image": ("IMAGE",    {}),
            },
            "hidden": {
                "unique_id":            "UNIQUE_ID",
                "loras_a_toggle":       "STRING",
                "loras_b_toggle":       "STRING",
                "trigger_words_toggle": "STRING",
            },
        }

    RETURN_TYPES  = ("STRING", "LORA_STACK", "LORA_STACK", "WORKFLOW_DICT")
    RETURN_NAMES  = ("prompt", "lora_stack_a", "lora_stack_b", "workflow_dict")
    FUNCTION      = "get_prompt"
    CATEGORY      = "FBnodes"
    OUTPUT_NODE   = True
    DESCRIPTION   = (
        "Workflow Manager — stores complete workflow recipes (prompts, LoRAs, model family, "
        "model A/B, VAE, CLIP) in the same JSON store as Prompt Manager Advanced. "
        "Fully backward-compatible — old prompts load unchanged. "
        "Connect workflow_dict to WorkflowGenerator to generate images/videos in one node."
    )

    @classmethod
    def IS_CHANGED(cls, category, name, text, use_prompt_input, use_lora_input, **kwargs):
        prompt_input = kwargs.get('prompt_input')
        lora_stack_a = kwargs.get('lora_stack_a')
        lora_stack_b = kwargs.get('lora_stack_b')
        return (category, name, text, use_prompt_input, use_lora_input,
                str(prompt_input), str(lora_stack_a), str(lora_stack_b))

    def get_prompt(self, category, name, use_prompt_input, text="", use_lora_input=True,
                   prompt_input=None, lora_stack_a=None, lora_stack_b=None,
                   trigger_words=None, thumbnail_image=None,
                   unique_id=None, loras_a_toggle=None, loras_b_toggle=None,
                   trigger_words_toggle=None):
        """
        Delegate most work to PromptManagerAdvanced.get_prompt, then augment
        the return value with a WORKFLOW_DICT output.
        """
        # Delegate to the original node for all prompt/lora logic
        pma = PromptManagerAdvanced()
        result = pma.get_prompt(
            category=category, name=name,
            use_prompt_input=use_prompt_input, text=text,
            use_lora_input=use_lora_input,
            prompt_input=prompt_input,
            lora_stack_a=lora_stack_a, lora_stack_b=lora_stack_b,
            trigger_words=trigger_words, thumbnail_image=thumbnail_image,
            unique_id=unique_id,
            loras_a_toggle=loras_a_toggle, loras_b_toggle=loras_b_toggle,
            trigger_words_toggle=trigger_words_toggle,
        )

        # Extract prompt text + stacks from result
        if isinstance(result, dict):
            # OUTPUT_NODE result format
            ui_data = result.get('ui', {})
            res     = result.get('result', ("", None, None))
        else:
            ui_data = {}
            res     = result if isinstance(result, tuple) else ("", None, None)

        prompt_out    = res[0] if len(res) > 0 else ""
        lora_stack_a_out = res[1] if len(res) > 1 else None
        lora_stack_b_out = res[2] if len(res) > 2 else None

        # Load workflow_config from stored prompt
        prompts    = PromptManagerAdvanced.load_prompts()
        prompt_data = prompts.get(category, {}).get(name, {})
        wf_config   = prompt_data.get("workflow_config", {})

        # Build WORKFLOW_DICT — everything WorkflowGenerator needs
        workflow_dict = {
            "prompt":       prompt_out,
            "loras_a":      prompt_data.get("loras_a", []),
            "loras_b":      prompt_data.get("loras_b", []),
            "trigger_words": prompt_data.get("trigger_words", []),
            "workflow_config": wf_config,
            # Convenience shorthand from workflow_config
            "family":   wf_config.get("family", ""),
            "model_a":  wf_config.get("model_a", ""),
            "model_b":  wf_config.get("model_b", ""),
            "vae":      wf_config.get("vae", ""),
            "clip":     wf_config.get("clip", []),
            "sampler":  wf_config.get("sampler", {}),
            "resolution": wf_config.get("resolution", {}),
        }

        # Build unified UI payload
        ui_out = dict(ui_data)
        ui_out["workflow_config"] = [wf_config]

        return {
            "ui":     ui_out,
            "result": (prompt_out, lora_stack_a_out, lora_stack_b_out, workflow_dict),
        }
