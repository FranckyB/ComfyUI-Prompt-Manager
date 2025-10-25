import os
import json
import folder_paths
import server

class PromptManagerNode:

    @classmethod
    def INPUT_TYPES(s):
        prompts_data = PromptManagerNode.load_prompts()
        categories = list(prompts_data.keys()) if prompts_data else ["Character"]

        all_prompts = set()
        for category_prompts in prompts_data.values():
            all_prompts.update(category_prompts.keys())

        all_prompts.add("")
        all_prompts_list = sorted(list(all_prompts))

        first_category = categories[0] if categories else "Character"
        # Get first prompt from first category, not from all prompts
        first_prompt = ""
        first_prompt_text = ""
        if prompts_data and first_category in prompts_data and prompts_data[first_category]:
            first_category_prompts = list(prompts_data[first_category].keys())
            first_prompt = sorted(first_category_prompts, key=str.lower)[0] if first_category_prompts else ""
            if first_prompt:
                first_prompt_text = prompts_data[first_category][first_prompt]

        return {
            "required": {
                "category": (categories, {"default": first_category}),
                "name": (all_prompts_list, {"default": first_prompt}),
            },
            "optional": {
                "text": ("STRING", {"multiline": True, "default": first_prompt_text, "placeholder": "Enter prompt text or connect input", "dynamicPrompts": False, "forceInput": False, "tooltip": "Enter prompt text directly or connect from another node"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    CATEGORY = "Prompt Manager"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_prompt"

    @staticmethod
    def get_prompts_path():
        """Get the path to the prompts JSON file in user/default folder"""
        return os.path.join(folder_paths.get_user_directory(), "default", "prompt_manager.json")

    @staticmethod
    def get_default_prompts_path():
        """Get the path to the default prompts JSON file"""
        return os.path.join(os.path.dirname(__file__), "default_prompts.json")

    @classmethod
    def load_prompts(cls):
        """Load prompts from user folder or default"""
        user_path = cls.get_prompts_path()
        default_path = cls.get_default_prompts_path()

        if os.path.exists(user_path):
            try:
                with open(user_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[PromptManager] Error loading user prompts: {e}")

        if os.path.exists(default_path):
            try:
                with open(default_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cls.save_prompts(data)
                    return data
            except Exception as e:
                print(f"[PromptManager] Error loading default prompts: {e}")

        default_data = {
            "Character": {
                "Fantasy Warrior": "a fantasy warrior, detailed armor, epic pose, dramatic lighting"
            },
            "Style": {
                "Cinematic": "cinematic lighting, dramatic atmosphere, film grain, depth of field"
            }
        }
        cls.save_prompts(default_data)
        return default_data

    @staticmethod
    def sort_prompts_data(data):
        """Sort categories and prompts alphabetically (case-insensitive)"""
        sorted_data = {}
        for category in sorted(data.keys(), key=str.lower):
            sorted_data[category] = dict(sorted(data[category].items(), key=lambda item: item[0].lower()))
        return sorted_data

    @classmethod
    def save_prompts(cls, data):
        """Save prompts to user folder"""
        user_path = cls.get_prompts_path()
        sorted_data = cls.sort_prompts_data(data)

        try:
            os.makedirs(os.path.dirname(user_path), exist_ok=True)
            with open(user_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[PromptManager] Error saving prompts: {e}")

    def get_prompt(self, category, name, text="", unique_id=None):
        """Return the current text in the text field and broadcast update"""
        # Broadcast the current prompt text to the frontend
        if unique_id is not None:
            server.PromptServer.instance.send_sync("prompt-manager-update-text", {
                "node_id": unique_id,
                "prompt": text
            })

        return (text,)


@server.PromptServer.instance.routes.get("/prompt-manager/get-prompts")
async def get_prompts(request):
    """API endpoint to get all prompts"""
    try:
        prompts = PromptManagerNode.load_prompts()
        return server.web.json_response(prompts)
    except Exception as e:
        print(f"[PromptManager] Error in get_prompts API: {e}")
        return server.web.json_response({"error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/save-category")
async def save_category(request):
    """API endpoint to create a new category"""
    try:
        data = await request.json()
        category_name = data.get("category_name", "").strip()

        if not category_name:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        prompts = PromptManagerNode.load_prompts()

        # Case-insensitive check for existing category
        existing_categories_lower = {k.lower(): k for k in prompts.keys()}
        if category_name.lower() in existing_categories_lower:
            return server.web.json_response({"success": False, "error": f"Category already exists as '{existing_categories_lower[category_name.lower()]}'"})

        prompts[category_name] = {}
        PromptManagerNode.save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptManager] Error in save_category API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/save-prompt")
async def save_prompt(request):
    """API endpoint to save a prompt"""
    try:
        data = await request.json()
        category = data.get("category", "").strip()
        name = data.get("name", "").strip()
        text = data.get("text", "").strip()

        if not category or not name or not text:
            return server.web.json_response({"success": False, "error": "All fields are required"})

        prompts = PromptManagerNode.load_prompts()

        if category not in prompts:
            prompts[category] = {}

        # Case-insensitive check - find if name exists with different casing
        existing_prompts_lower = {k.lower(): k for k in prompts[category].keys()}
        if name.lower() in existing_prompts_lower:
            old_name = existing_prompts_lower[name.lower()]
            if old_name != name:
                # Delete the old casing version
                print(f"[PromptManager] Removing old casing '{old_name}' before saving as '{name}'")
                del prompts[category][old_name]

        prompts[category][name] = text
        PromptManagerNode.save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptManager] Error in save_prompt API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/delete-category")
async def delete_category(request):
    """API endpoint to delete a category"""
    try:
        data = await request.json()
        category = data.get("category", "").strip()

        if not category:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        prompts = PromptManagerNode.load_prompts()

        if category not in prompts:
            return server.web.json_response({"success": False, "error": "Category not found"})

        del prompts[category]
        PromptManagerNode.save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptManager] Error in delete_category API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/delete-prompt")
async def delete_prompt(request):
    """API endpoint to delete a prompt"""
    try:
        data = await request.json()
        category = data.get("category", "").strip()
        name = data.get("name", "").strip()

        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and prompt name are required"})

        prompts = PromptManagerNode.load_prompts()

        if category not in prompts or name not in prompts[category]:
            return server.web.json_response({"success": False, "error": "Prompt not found"})

        del prompts[category][name]
        PromptManagerNode.save_prompts(prompts)

        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptManager] Error in delete_prompt API: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)
