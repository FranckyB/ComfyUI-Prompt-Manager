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
        if prompts_data and first_category in prompts_data and prompts_data[first_category]:
            first_category_prompts = list(prompts_data[first_category].keys())
            first_prompt = sorted(first_category_prompts, key=str.lower)[0] if first_category_prompts else ""

        return {
            "required": {
                "category": (categories, {"default": first_category}),
                "name": (all_prompts_list, {"default": first_prompt}),
            },
            "optional": {
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter prompt text or connect input", "dynamicPrompts": False, "forceInput": False, "tooltip": "Enter prompt text directly or connect from another node"}),
            },
        }

    CATEGORY = "Prompt Manager"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
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

    def get_prompt(self, category, name, text=""):
        """Return the current text in the text field"""
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

        if category_name in prompts:
            return server.web.json_response({"success": False, "error": "Category already exists"})

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
