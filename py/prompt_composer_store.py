"""
Prompt Composer Store - isolated JSON storage for composer fragments.

This module provides a separate data file and backend endpoints so Prompt
Composer nodes do not share the main Prompt Manager library.
"""
import json
import os
import shutil
import time

import folder_paths
import server
from .backup_manager import atomic_save, load_with_fallback, check_backup


class PromptComposerStore:
    """Load/save prompt fragments to their own user data file."""

    @staticmethod
    def get_data_path():
        """Path to the prompt composer JSON file in the user's default folder."""
        return os.path.join(folder_paths.get_user_directory(), "default", "prompt_composer_data.json")

    @staticmethod
    def get_default_prompts_path():
        """Path to the bundled default composer prompts JSON file."""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "default_composer_prompts.json")

    @classmethod
    def load_prompts(cls):
        """Load composer fragments, seeding from bundled defaults if no user file exists."""
        user_path = cls.get_data_path()

        if os.path.exists(user_path):
            try:
                with open(user_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                print(f"[PromptComposerStore] Error loading data: {e}")
                check_backup(user_path)
                data = load_with_fallback(user_path, "PromptComposerStore")
                if isinstance(data, dict):
                    return data
                print("[PromptComposerStore] User data file exists but could not be parsed; not overwriting.")
                return {}

        default_path = cls.get_default_prompts_path()
        if os.path.exists(default_path):
            try:
                with open(default_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    cls.save_prompts(data)
                    return data
            except Exception as e:
                print(f"[PromptComposerStore] Error loading default prompts: {e}")

        return {}

    @staticmethod
    def sort_prompts_data(data):
        """Sort categories and prompt names alphabetically while preserving __meta__."""
        sorted_data = {}
        for category in sorted(data.keys(), key=str.lower):
            cat_data = data[category]
            meta = cat_data.get("__meta__") if isinstance(cat_data, dict) else None
            sorted_prompts = dict(sorted(
                ((k, v) for k, v in cat_data.items() if k != "__meta__"),
                key=lambda item: item[0].lower(),
            ))
            if meta is not None:
                sorted_prompts["__meta__"] = meta
            sorted_data[category] = sorted_prompts
        return sorted_data

    @classmethod
    def save_prompts(cls, data):
        """Save composer fragments atomically with rotating backups."""
        user_path = cls.get_data_path()
        sorted_data = cls.sort_prompts_data(data)
        atomic_save(user_path, sorted_data, "PromptComposerStore")


def _find_category_case_insensitive(prompts_data, category):
    """Return the canonical category name or None."""
    if not isinstance(prompts_data, dict):
        return None
    for cat in prompts_data.keys():
        if cat.lower() == str(category or "").lower():
            return cat
    return None


def _is_hidden_category_entry_key(name):
    normalized = str(name or "").strip().lower()
    return normalized in {"__meta__", "_base_prompt_", "_prompt_prefix_", "_prompt_type_"}


def _find_prompt_case_insensitive(category_data, name):
    """Return entry dict and canonical name for a prompt in a category."""
    if not isinstance(category_data, dict):
        return None, None
    if _is_hidden_category_entry_key(name):
        return None, None
    if name in category_data and not _is_hidden_category_entry_key(name):
        return category_data[name], name
    name_lower = str(name or "").lower()
    for entry_name, entry in category_data.items():
        if _is_hidden_category_entry_key(entry_name):
            continue
        if entry_name.lower() == name_lower:
            return entry, entry_name
    return None, None


@server.PromptServer.instance.routes.get("/prompt-manager/compose/get-prompts")
async def compose_get_prompts(request):
    try:
        return server.web.json_response(PromptComposerStore.load_prompts())
    except Exception as e:
        print(f"[PromptComposerStore] Error in get-prompts: {e}")
        return server.web.json_response({"error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/save-category")
async def compose_save_category(request):
    try:
        data = await request.json()
        category_name = str(data.get("category_name", "")).strip()
        if not category_name:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        prompts = PromptComposerStore.load_prompts()
        existing = {k.lower(): k for k in prompts.keys()}
        if category_name.lower() in existing:
            return server.web.json_response({
                "success": False,
                "error": f"Category already exists as '{existing[category_name.lower()]}'",
            })

        cat_data = {}
        if data.get("nsfw"):
            cat_data["__meta__"] = {"nsfw": True}
        prompts[category_name] = cat_data
        PromptComposerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptComposerStore] Error in save-category: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/save-category-base-prompt")
async def compose_save_category_base_prompt(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        base_prompt = str(data.get("base_prompt", ""))

        if not category:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        prompts = PromptComposerStore.load_prompts()
        canonical_category = _find_category_case_insensitive(prompts, category)
        if canonical_category is None:
            return server.web.json_response({"success": False, "error": "Category not found"})

        cat_data = prompts.get(canonical_category)
        if not isinstance(cat_data, dict):
            cat_data = {}

        trimmed = base_prompt.strip()
        if trimmed:
            cat_data["_base_prompt_"] = base_prompt
        else:
            cat_data.pop("_base_prompt_", None)

        prompts[canonical_category] = cat_data
        PromptComposerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptComposerStore] Error in save-category-base-prompt: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/save-category-settings")
async def compose_save_category_settings(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        if not category:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        prompts = PromptComposerStore.load_prompts()
        canonical_category = _find_category_case_insensitive(prompts, category)
        if canonical_category is None:
            return server.web.json_response({"success": False, "error": "Category not found"})

        cat_data = prompts.get(canonical_category)
        if not isinstance(cat_data, dict):
            cat_data = {}

        base_prompt = str(data.get("base_prompt", ""))
        trimmed_base = base_prompt.strip()
        if trimmed_base:
            cat_data["_base_prompt_"] = base_prompt
        else:
            cat_data.pop("_base_prompt_", None)

        prompt_type = str(data.get("prompt_type", "")).strip()
        if prompt_type:
            cat_data["_prompt_type_"] = prompt_type
        else:
            cat_data.pop("_prompt_type_", None)

        # _prompt_prefix_ is replaced by _prompt_type_; remove legacy prefix.
        cat_data.pop("_prompt_prefix_", None)

        prompts[canonical_category] = cat_data
        PromptComposerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptComposerStore] Error in save-category-settings: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/rename-category")
async def compose_rename_category(request):
    try:
        data = await request.json()
        old_category = str(data.get("old_category", "")).strip()
        new_category = str(data.get("new_category", "")).strip()
        if not old_category or not new_category:
            return server.web.json_response({"success": False, "error": "Both old and new category names are required"})

        prompts = PromptComposerStore.load_prompts()
        if old_category not in prompts:
            return server.web.json_response({"success": False, "error": f"Category '{old_category}' not found"})

        existing = {k.lower(): k for k in prompts.keys() if k.lower() != old_category.lower()}
        if new_category.lower() in existing:
            return server.web.json_response({
                "success": False,
                "error": f"Category already exists as '{existing[new_category.lower()]}'",
            })

        prompts[new_category] = prompts[old_category]
        del prompts[old_category]
        PromptComposerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts, "new_category": new_category})
    except Exception as e:
        print(f"[PromptComposerStore] Error in rename-category: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/delete-category")
async def compose_delete_category(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        if not category:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        prompts = PromptComposerStore.load_prompts()
        if category in prompts:
            del prompts[category]
            PromptComposerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptComposerStore] Error in delete-category: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/save-prompt")
async def compose_save_prompt(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        name = str(data.get("name", "")).strip()
        text = str(data.get("text", "")).strip()
        thumbnail = data.get("thumbnail")

        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name are required"})

        prompts = PromptComposerStore.load_prompts()
        if category not in prompts:
            prompts[category] = {}

        # Case-insensitive prompt replacement.
        existing_lower = {
            k.lower(): k
            for k in prompts[category].keys()
            if not _is_hidden_category_entry_key(k)
        }
        existing_prompt = {}
        if name.lower() in existing_lower:
            old_name = existing_lower[name.lower()]
            existing_prompt = prompts[category].get(old_name, {})
            if old_name != name:
                print(f"[PromptComposerStore] Removing old casing '{old_name}' before saving as '{name}'")
                del prompts[category][old_name]

        entry = {"prompt": text}
        if thumbnail is not None:
            entry["thumbnail"] = thumbnail
        elif existing_prompt.get("thumbnail"):
            entry["thumbnail"] = existing_prompt["thumbnail"]
        if existing_prompt.get("nsfw"):
            entry["nsfw"] = existing_prompt["nsfw"]

        prompts[category][name] = entry
        PromptComposerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptComposerStore] Error in save-prompt: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/delete-prompt")
async def compose_delete_prompt(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        name = str(data.get("name", "")).strip()
        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name are required"})

        prompts = PromptComposerStore.load_prompts()
        if category in prompts and name in prompts[category]:
            del prompts[category][name]
            PromptComposerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptComposerStore] Error in delete-prompt: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/rename-prompt")
async def compose_rename_prompt(request):
    try:
        data = await request.json()
        old_category = str(data.get("category", "")).strip()
        old_name = str(data.get("old_name", "")).strip()
        new_name = str(data.get("new_name", "")).strip()
        new_category = str(data.get("new_category", old_category)).strip() or old_category

        if not old_category or not old_name or not new_name:
            return server.web.json_response({"success": False, "error": "Missing required fields"})

        prompts = PromptComposerStore.load_prompts()
        if old_category not in prompts or old_name not in prompts[old_category]:
            return server.web.json_response({"success": False, "error": "Prompt not found"})

        if new_category not in prompts:
            prompts[new_category] = {}

        entry = prompts[old_category].pop(old_name)
        prompts[new_category][new_name] = entry
        PromptComposerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptComposerStore] Error in rename-prompt: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/toggle-nsfw")
async def compose_toggle_nsfw(request):
    try:
        data = await request.json()
        toggle_type = data.get("type", "prompt")
        category = str(data.get("category", "")).strip()
        name = str(data.get("name", "")).strip()

        prompts = PromptComposerStore.load_prompts()
        if toggle_type == "category":
            if category not in prompts:
                return server.web.json_response({"success": False, "error": "Category not found"})
            meta = prompts[category].get("__meta__", {})
            meta["nsfw"] = not meta.get("nsfw", False)
            prompts[category]["__meta__"] = meta
        else:
            if category not in prompts or name not in prompts[category]:
                return server.web.json_response({"success": False, "error": "Prompt not found"})
            entry = prompts[category][name]
            if isinstance(entry, dict):
                entry["nsfw"] = not entry.get("nsfw", False)

        PromptComposerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptComposerStore] Error in toggle-nsfw: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/save-thumbnail")
async def compose_save_thumbnail(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        name = str(data.get("name", "")).strip()
        thumbnail = data.get("thumbnail")

        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name are required"})

        prompts = PromptComposerStore.load_prompts()
        if category not in prompts:
            prompts[category] = {}
        if name not in prompts[category]:
            prompts[category][name] = {"prompt": ""}

        entry = prompts[category][name]
        if not isinstance(entry, dict):
            entry = {"prompt": str(entry)}
            prompts[category][name] = entry

        if thumbnail:
            entry["thumbnail"] = thumbnail
        else:
            entry.pop("thumbnail", None)

        PromptComposerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptComposerStore] Error in save-thumbnail: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)
