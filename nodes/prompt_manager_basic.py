import server
from ..py.prompt_metadata import record_prompt
from ..py.prompt_store import is_hidden_key, load_prompts, save_prompts


class PromptManager:
    """Legacy: kept for saved workflows; Prompt Manager (PromptManagerV2) replaces it."""

    DEPRECATED = True

    @classmethod
    def INPUT_TYPES(s):
        prompts_data = load_prompts()
        categories = list(prompts_data.keys()) if prompts_data else ["Character"]

        all_prompts = set()
        for category_prompts in prompts_data.values():
            all_prompts.update(k for k in category_prompts.keys() if not is_hidden_key(k))

        all_prompts.add("")
        all_prompts_list = sorted(list(all_prompts))

        first_category = categories[0] if categories else "Character"

        # Get first prompt from first category
        first_prompt = ""
        first_prompt_text = ""
        if prompts_data and first_category in prompts_data and prompts_data[first_category]:
            first_category_prompts = [k for k in prompts_data[first_category].keys() if not is_hidden_key(k)]
            first_prompt = sorted(first_category_prompts, key=str.lower)[0] if first_category_prompts else ""
            if first_prompt:
                first_prompt_text = prompts_data[first_category][first_prompt].get("prompt", "")

        return {
            "required": {
                "category": (categories, {"default": first_category}),
                "name": (all_prompts_list, {"default": first_prompt}),
                "use_external": ("BOOLEAN", {"default": False, "tooltip": "Toggle to use LLM input instead of internal text"}),
                "text": ("STRING", {"multiline": True, "default": first_prompt_text, "placeholder": "Enter prompt text", "dynamicPrompts": False, "tooltip": "Enter prompt text directly"}),
            },
            "optional": {
                "llm_input": ("STRING", {"multiline": True, "forceInput": True, "lazy": True, "tooltip": "Connect LLM text input here"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "api_prompt": "PROMPT",
            }
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Simple prompt manager for saving and loading prompts by category. Prompt-only, no LoRA support."
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_prompt"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, name, **kwargs):
        """Allow any name value, including temporary unsaved prompts"""
        # Return True to accept any value - this allows new prompts to be tested before saving
        return True

    def get_prompt(self, category, name, use_external, text="", llm_input="", unique_id=None, extra_pnginfo=None, api_prompt=None):
        """Return the appropriate text based on use_external toggle and broadcast update"""

        # Choose which text to use based on the toggle
        if use_external and llm_input:
            output_text = llm_input
        else:
            output_text = text

        # Broadcast the current prompt text to the frontend
        if unique_id is not None:
            server.PromptServer.instance.send_sync("prompt-manager-update-text", {
                "node_id": unique_id,
                "prompt": output_text,
                "use_external": use_external,
                "llm_input": llm_input if use_external else ""
            })

        # Save resolved runtime prompt into execution metadata so saver nodes
        # embed the final prompt value, not the unresolved linked input state.
        record_prompt(unique_id, output_text, extra_pnginfo, api_prompt, toggle="use_external", link_input="llm_input", text_index=3)
        return (output_text,)

    def check_lazy_status(self, category, name, use_external, text, llm_input=None, unique_id=None,
                          extra_pnginfo=None, api_prompt=None, **kwargs):
        needed = []
        if use_external:
            needed.append("llm_input")
        return needed


@server.PromptServer.instance.routes.post("/prompt-manager/save-prompt")
async def save_prompt(request):
    """API endpoint to save a prompt"""
    try:
        data = await request.json()
        category = data.get("category", "").strip()
        name = data.get("name", "").strip()
        text = data.get("text", "").strip()
        old_category = data.get("old_category", "").strip() or category
        old_name = data.get("old_name", "").strip() or name

        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name are required"})

        prompts = load_prompts()

        if category not in prompts:
            prompts[category] = {}

        source_prompts = prompts.get(old_category, {}) if isinstance(prompts.get(old_category), dict) else {}
        target_prompts = prompts[category]

        existing_data = {}
        existing_old_name = None
        if old_category in prompts:
            existing_old_name = next((key for key in source_prompts.keys() if str(key).lower() == old_name.lower()), None)
            if existing_old_name:
                existing_data = source_prompts.get(existing_old_name, {})

        target_conflict_name = next((key for key in target_prompts.keys() if str(key).lower() == name.lower()), None)
        if target_conflict_name:
            same_entry = (
                existing_old_name is not None
                and old_category == category
                and str(target_conflict_name).lower() == str(existing_old_name).lower()
            )
            if not same_entry:
                return server.web.json_response({
                    "success": False,
                    "error": f"A prompt named '{target_conflict_name}' already exists in '{category}'"
                })

        if existing_old_name and (old_category != category or existing_old_name != name):
            existing_data = source_prompts.pop(existing_old_name, existing_data)

        # Case-insensitive check - find if name exists with different casing
        existing_prompts_lower = {k.lower(): k for k in prompts[category].keys()}
        if name.lower() in existing_prompts_lower:
            casing_name = existing_prompts_lower[name.lower()]
            if casing_name != name:

                # Delete the old casing version
                print(f"[PromptManager] Removing old casing '{casing_name}' before saving as '{name}'")
                del prompts[category][casing_name]

        # Only the text changes: LoRAs, trigger words, recipe data, thumbnail and flags saved by the full
        # Prompt Manager stay with the prompt.
        prompts[category][name] = {**existing_data, "prompt": text} if isinstance(existing_data, dict) else {"prompt": text}

        save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptManager] Error in save_prompt API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)
