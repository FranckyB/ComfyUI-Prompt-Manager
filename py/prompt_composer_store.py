"""
Prompt Composer Store - isolated JSON storage for composer fragments.

This module provides a separate data file and backend endpoints so Prompt
Composer/Prompt Mixer nodes do not share the main Prompt Manager library.
"""
import json
import os
import shutil
import time

import folder_paths
import server


class PromptMixerStore:
    """Load/save prompt fragments to their own user data file."""

    @staticmethod
    def get_data_path():
        """Path to the prompt composer JSON file in the user's default folder."""
        return os.path.join(folder_paths.get_user_directory(), "default", "prompt_composer_data.json")

    @staticmethod
    def get_legacy_data_path():
        """Legacy mixer file path kept for backward compatibility."""
        return os.path.join(folder_paths.get_user_directory(), "default", "prompt_mixer_data.json")

    @staticmethod
    def get_backup_path():
        return PromptMixerStore.get_data_path() + "_bak"

    @staticmethod
    def get_default_prompts_path():
        """Path to the bundled default composer prompts JSON file."""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "default_composer_prompts.json")

    @staticmethod
    def get_legacy_default_prompts_path():
        """Legacy bundled defaults path kept for backward compatibility."""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "default_mixer_prompts.json")

    @classmethod
    def _refresh_weekly_backup_if_due(cls, source_path, backup_path, interval_days=7):
        if not os.path.exists(source_path):
            return
        should_refresh = not os.path.exists(backup_path)
        if not should_refresh:
            try:
                backup_age = time.time() - os.path.getmtime(backup_path)
                should_refresh = backup_age > interval_days * 24 * 60 * 60
            except Exception:
                should_refresh = True
        if should_refresh:
            try:
                shutil.copy2(source_path, backup_path)
            except Exception:
                pass

    @classmethod
    def load_prompts(cls):
        """Load mixer fragments, seeding from bundled defaults if no user file exists."""
        user_path = cls.get_data_path()
        legacy_user_path = cls.get_legacy_data_path()
        default_path = cls.get_default_prompts_path()
        legacy_default_path = cls.get_legacy_default_prompts_path()

        source_user_path = user_path if os.path.exists(user_path) else legacy_user_path

        if os.path.exists(source_user_path):
            try:
                with open(source_user_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if source_user_path != user_path:
                        cls.save_prompts(data)
                    return data
            except Exception as e:
                print(f"[PromptMixerStore] Error loading data: {e}")
                backup_candidates = [
                    cls.get_backup_path(),
                    user_path + ".bak",
                    user_path + ".backup",
                    user_path + ".tmp",
                    legacy_user_path + ".bak",
                    legacy_user_path + ".backup",
                    legacy_user_path + ".tmp",
                ]
                for backup_path in backup_candidates:
                    if not os.path.exists(backup_path):
                        continue
                    try:
                        with open(backup_path, "r", encoding="utf-8") as f:
                            return json.load(f)
                    except Exception:
                        pass
                print("[PromptMixerStore] User data file exists but could not be parsed; not overwriting.")
                return {}

        default_source_path = default_path if os.path.exists(default_path) else legacy_default_path

        if os.path.exists(default_source_path):
            try:
                with open(default_source_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    cls.save_prompts(data)
                    return data
            except Exception as e:
                print(f"[PromptMixerStore] Error loading default prompts: {e}")

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
        """Save mixer fragments atomically with a weekly backup."""
        user_path = cls.get_data_path()
        sorted_data = cls.sort_prompts_data(data)
        tmp_path = user_path + ".tmp"
        bak_path = cls.get_backup_path()
        try:
            os.makedirs(os.path.dirname(user_path), exist_ok=True)
            cls._refresh_weekly_backup_if_due(user_path, bak_path)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(sorted_data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, user_path)
        except Exception as e:
            print(f"[PromptMixerStore] Error saving data: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass


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
    return normalized in {"__meta__", "_base_prompt_", "_prompt_prefix_"}


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


@server.PromptServer.instance.routes.get("/prompt-manager/mixer/get-prompts")
async def mixer_get_prompts(request):
    try:
        return server.web.json_response(PromptMixerStore.load_prompts())
    except Exception as e:
        print(f"[PromptMixerStore] Error in get-prompts: {e}")
        return server.web.json_response({"error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/mixer/save-category")
async def mixer_save_category(request):
    try:
        data = await request.json()
        category_name = str(data.get("category_name", "")).strip()
        if not category_name:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        prompts = PromptMixerStore.load_prompts()
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
        PromptMixerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptMixerStore] Error in save-category: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/mixer/save-category-base-prompt")
async def mixer_save_category_base_prompt(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        base_prompt = str(data.get("base_prompt", ""))

        if not category:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        prompts = PromptMixerStore.load_prompts()
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
        PromptMixerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptMixerStore] Error in save-category-base-prompt: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/mixer/save-category-prompt-prefix")
async def mixer_save_category_prompt_prefix(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        prompt_prefix = str(data.get("prompt_prefix", ""))

        if not category:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        prompts = PromptMixerStore.load_prompts()
        canonical_category = _find_category_case_insensitive(prompts, category)
        if canonical_category is None:
            return server.web.json_response({"success": False, "error": "Category not found"})

        cat_data = prompts.get(canonical_category)
        if not isinstance(cat_data, dict):
            cat_data = {}

        trimmed = prompt_prefix.strip()
        if trimmed:
            cat_data["_prompt_prefix_"] = prompt_prefix
        else:
            cat_data.pop("_prompt_prefix_", None)

        prompts[canonical_category] = cat_data
        PromptMixerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptMixerStore] Error in save-category-prompt-prefix: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/mixer/rename-category")
async def mixer_rename_category(request):
    try:
        data = await request.json()
        old_category = str(data.get("old_category", "")).strip()
        new_category = str(data.get("new_category", "")).strip()
        if not old_category or not new_category:
            return server.web.json_response({"success": False, "error": "Both old and new category names are required"})

        prompts = PromptMixerStore.load_prompts()
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
        PromptMixerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts, "new_category": new_category})
    except Exception as e:
        print(f"[PromptMixerStore] Error in rename-category: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/mixer/delete-category")
async def mixer_delete_category(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        if not category:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        prompts = PromptMixerStore.load_prompts()
        if category in prompts:
            del prompts[category]
            PromptMixerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptMixerStore] Error in delete-category: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/mixer/save-prompt")
async def mixer_save_prompt(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        name = str(data.get("name", "")).strip()
        text = str(data.get("text", "")).strip()
        thumbnail = data.get("thumbnail")

        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name are required"})

        prompts = PromptMixerStore.load_prompts()
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
                print(f"[PromptMixerStore] Removing old casing '{old_name}' before saving as '{name}'")
                del prompts[category][old_name]

        entry = {"prompt": text}
        if thumbnail is not None:
            entry["thumbnail"] = thumbnail
        elif existing_prompt.get("thumbnail"):
            entry["thumbnail"] = existing_prompt["thumbnail"]
        if existing_prompt.get("nsfw"):
            entry["nsfw"] = existing_prompt["nsfw"]

        prompts[category][name] = entry
        PromptMixerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptMixerStore] Error in save-prompt: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/mixer/delete-prompt")
async def mixer_delete_prompt(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        name = str(data.get("name", "")).strip()
        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name are required"})

        prompts = PromptMixerStore.load_prompts()
        if category in prompts and name in prompts[category]:
            del prompts[category][name]
            PromptMixerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptMixerStore] Error in delete-prompt: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/mixer/rename-prompt")
async def mixer_rename_prompt(request):
    try:
        data = await request.json()
        old_category = str(data.get("category", "")).strip()
        old_name = str(data.get("old_name", "")).strip()
        new_name = str(data.get("new_name", "")).strip()
        new_category = str(data.get("new_category", old_category)).strip() or old_category

        if not old_category or not old_name or not new_name:
            return server.web.json_response({"success": False, "error": "Missing required fields"})

        prompts = PromptMixerStore.load_prompts()
        if old_category not in prompts or old_name not in prompts[old_category]:
            return server.web.json_response({"success": False, "error": "Prompt not found"})

        if new_category not in prompts:
            prompts[new_category] = {}

        entry = prompts[old_category].pop(old_name)
        prompts[new_category][new_name] = entry
        PromptMixerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptMixerStore] Error in rename-prompt: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/mixer/toggle-nsfw")
async def mixer_toggle_nsfw(request):
    try:
        data = await request.json()
        toggle_type = data.get("type", "prompt")
        category = str(data.get("category", "")).strip()
        name = str(data.get("name", "")).strip()

        prompts = PromptMixerStore.load_prompts()
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

        PromptMixerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptMixerStore] Error in toggle-nsfw: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/mixer/save-thumbnail")
async def mixer_save_thumbnail(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        name = str(data.get("name", "")).strip()
        thumbnail = data.get("thumbnail")

        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name are required"})

        prompts = PromptMixerStore.load_prompts()
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

        PromptMixerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptMixerStore] Error in save-thumbnail: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)
