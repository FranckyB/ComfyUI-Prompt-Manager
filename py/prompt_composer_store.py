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
                    normalized = _normalize_prompts_data(data)
                    if normalized != data:
                        print("[PromptComposerStore] Migrating prompt_composer_data.json to nested _prompts_ schema")
                        cls.save_prompts(normalized)
                    return normalized
            except Exception as e:
                print(f"[PromptComposerStore] Error loading data: {e}")
                check_backup(user_path)
                data = load_with_fallback(user_path, "PromptComposerStore")
                if isinstance(data, dict):
                    normalized = _normalize_prompts_data(data)
                    if normalized != data:
                        print("[PromptComposerStore] Migrating recovered prompt_composer_data.json to nested _prompts_ schema")
                        cls.save_prompts(normalized)
                    return normalized
                print("[PromptComposerStore] User data file exists but could not be parsed; not overwriting.")
                return {}

        default_path = cls.get_default_prompts_path()
        if os.path.exists(default_path):
            try:
                with open(default_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    normalized = _normalize_prompts_data(data)
                    cls.save_prompts(normalized)
                    return normalized
            except Exception as e:
                print(f"[PromptComposerStore] Error loading default prompts: {e}")

        return {}

    @staticmethod
    def sort_prompts_data(data):
        """Sort categories and nested prompt names alphabetically while preserving metadata."""
        sorted_data = {}
        top_level_meta = data.get("__meta__") if isinstance(data, dict) else None
        if top_level_meta is not None:
            sorted_data["__meta__"] = top_level_meta
        for category in sorted(data.keys(), key=str.lower):
            if category == "__meta__":
                continue
            cat_data = data[category]
            normalized_category = _normalize_category_data(cat_data)
            sorted_category = {}

            meta = normalized_category.get("__meta__")
            if meta is not None:
                sorted_category["__meta__"] = meta

            base_prompt = normalized_category.get("_base_prompt_")
            if isinstance(base_prompt, str) and base_prompt.strip():
                sorted_category["_base_prompt_"] = base_prompt

            prompt_type = normalized_category.get("_prompt_type_")
            if isinstance(prompt_type, str) and prompt_type.strip():
                sorted_category["_prompt_type_"] = prompt_type

            prompt_prefix = normalized_category.get("_prompt_prefix_")
            if isinstance(prompt_prefix, str) and prompt_prefix.strip():
                sorted_category["_prompt_prefix_"] = prompt_prefix

            prompt_entries = _get_category_prompts_map(normalized_category)
            sorted_category["_prompts_"] = dict(sorted(
                prompt_entries.items(),
                key=lambda item: item[0].lower(),
            ))
            sorted_data[category] = sorted_category
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
    return normalized in {"__meta__", "_base_prompt_", "_prompt_prefix_", "_prompt_type_", "_prompts_"}


def _normalize_prompt_entry(entry):
    if isinstance(entry, dict):
        normalized = dict(entry)
        if "prompt" in normalized:
            normalized["prompt"] = str(normalized.get("prompt", "") or "")
        return normalized
    return {"prompt": str(entry or "")}


def _get_category_prompts_map(category_data):
    if not isinstance(category_data, dict):
        return {}
    prompt_entries = category_data.get("_prompts_")
    if isinstance(prompt_entries, dict):
        return prompt_entries
    return {
        key: value
        for key, value in category_data.items()
        if not _is_hidden_category_entry_key(key)
    }


def _ensure_category_prompts_map(category_data):
    if not isinstance(category_data, dict):
        category_data = {}
    prompt_entries = category_data.get("_prompts_")
    if not isinstance(prompt_entries, dict):
        prompt_entries = {
            key: _normalize_prompt_entry(value)
            for key, value in category_data.items()
            if not _is_hidden_category_entry_key(key)
        }
        category_data["_prompts_"] = prompt_entries
        for key in list(category_data.keys()):
            if key != "_prompts_" and not _is_hidden_category_entry_key(key):
                category_data.pop(key, None)
    return prompt_entries


def _normalize_category_data(category_data):
    if not isinstance(category_data, dict):
        category_data = {}

    normalized = {"_prompts_": {}}

    meta = category_data.get("__meta__")
    if meta is not None:
        normalized["__meta__"] = meta

    base_prompt = category_data.get("_base_prompt_")
    if isinstance(base_prompt, str):
        if base_prompt.strip():
            normalized["_base_prompt_"] = base_prompt
    else:
        base_prompt_text = str(base_prompt or "")
        if base_prompt_text.strip():
            normalized["_base_prompt_"] = base_prompt_text

    prompt_type = str(category_data.get("_prompt_type_") or "").strip()
    if prompt_type:
        normalized["_prompt_type_"] = prompt_type

    prompt_prefix = category_data.get("_prompt_prefix_")
    if isinstance(prompt_prefix, str):
        if prompt_prefix.strip():
            normalized["_prompt_prefix_"] = prompt_prefix
    else:
        prompt_prefix_text = str(prompt_prefix or "")
        if prompt_prefix_text.strip():
            normalized["_prompt_prefix_"] = prompt_prefix_text

    for name, entry in _get_category_prompts_map(category_data).items():
        normalized["_prompts_"][str(name)] = _normalize_prompt_entry(entry)

    return normalized


def _normalize_prompts_data(data):
    if not isinstance(data, dict):
        return {}

    normalized = {}
    top_level_meta = data.get("__meta__")
    if top_level_meta is not None:
        normalized["__meta__"] = top_level_meta

    for category, category_data in data.items():
        if category == "__meta__":
            continue
        normalized[str(category)] = _normalize_category_data(category_data)
    return normalized


def _ensure_category_data(prompts, category):
    category_data = _normalize_category_data(prompts.get(category, {}))
    prompts[category] = category_data
    return category_data


def _normalize_optional_string(value):
    if value is None:
        return ""
    return str(value or "").strip()


def _normalize_optional_float(value, default=1.0, minimum=None, maximum=None):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    if minimum is not None:
        numeric = max(float(minimum), numeric)
    if maximum is not None:
        numeric = min(float(maximum), numeric)
    return numeric


def _find_prompt_case_insensitive(category_data, name):
    """Return entry dict and canonical name for a prompt in a category."""
    prompt_entries = _get_category_prompts_map(category_data)
    if not isinstance(prompt_entries, dict):
        return None, None
    if str(name or "").strip().lower() == "__meta__":
        return None, None
    if name in prompt_entries:
        return prompt_entries[name], name
    name_lower = str(name or "").lower()
    for entry_name, entry in prompt_entries.items():
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
        prompt_type = str(data.get("prompt_type", "")).strip()
        if prompt_type:
            cat_data["_prompt_type_"] = prompt_type
        cat_data["_prompts_"] = {}
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

        cat_data = _ensure_category_data(prompts, canonical_category)

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

        cat_data = _ensure_category_data(prompts, canonical_category)

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

        prompt_prefix = str(data.get("prompt_prefix", ""))
        if prompt_prefix.strip():
            cat_data["_prompt_prefix_"] = prompt_prefix
        else:
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


@server.PromptServer.instance.routes.post("/prompt-manager/compose/import-prompts")
async def compose_import_prompts(request):
    try:
        data = await request.json()
        imported_data = _normalize_prompts_data(data.get("data", {}))
        mode = str(data.get("mode", "skip_existing") or "skip_existing").strip().lower()
        if mode not in {"skip_existing", "replace_existing"}:
            mode = "skip_existing"

        if not isinstance(imported_data, dict):
            return server.web.json_response({"success": False, "error": "Invalid data format"})

        prompts = PromptComposerStore.load_prompts()
        imported_prompts = 0
        skipped_prompts = 0
        imported_category_settings = 0
        skipped_category_settings = 0
        created_categories = 0

        for category, imported_category in imported_data.items():
            if category == "__meta__" or not isinstance(imported_category, dict):
                continue

            canonical_category = _find_category_case_insensitive(prompts, category)
            is_new_category = canonical_category is None
            if is_new_category:
                canonical_category = str(category)
                prompts[canonical_category] = {"_prompts_": {}}
                created_categories += 1

            category_data = _ensure_category_data(prompts, canonical_category)
            normalized_imported_category = _normalize_category_data(imported_category)

            imported_meta = normalized_imported_category.get("__meta__")
            if imported_meta is not None and (mode == "replace_existing" or is_new_category):
                category_data["__meta__"] = imported_meta

            imported_base_prompt = normalized_imported_category.get("_base_prompt_")
            imported_prompt_type = normalized_imported_category.get("_prompt_type_")
            has_category_settings = bool(
                (isinstance(imported_base_prompt, str) and imported_base_prompt.strip()) or
                (isinstance(imported_prompt_type, str) and imported_prompt_type.strip())
            )
            if has_category_settings:
                if mode == "replace_existing" or is_new_category:
                    if isinstance(imported_base_prompt, str) and imported_base_prompt.strip():
                        category_data["_base_prompt_"] = imported_base_prompt
                    if isinstance(imported_prompt_type, str) and imported_prompt_type.strip():
                        category_data["_prompt_type_"] = imported_prompt_type
                    imported_category_settings += 1
                else:
                    skipped_category_settings += 1

            prompt_entries = _ensure_category_prompts_map(category_data)
            for prompt_name, imported_entry in _get_category_prompts_map(normalized_imported_category).items():
                existing_entry, existing_name = _find_prompt_case_insensitive(category_data, prompt_name)
                if existing_entry is not None and mode != "replace_existing":
                    skipped_prompts += 1
                    continue
                if existing_name and existing_name != prompt_name:
                    del prompt_entries[existing_name]
                prompt_entries[prompt_name] = _normalize_prompt_entry(imported_entry)
                imported_prompts += 1

            prompts[canonical_category] = category_data

        PromptComposerStore.save_prompts(prompts)
        return server.web.json_response({
            "success": True,
            "prompts": prompts,
            "imported_prompts": imported_prompts,
            "skipped_prompts": skipped_prompts,
            "imported_category_settings": imported_category_settings,
            "skipped_category_settings": skipped_category_settings,
            "created_categories": created_categories,
        })
    except Exception as e:
        print(f"[PromptComposerStore] Error in import-prompts: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/save-prompt")
async def compose_save_prompt(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        name = str(data.get("name", "")).strip()
        text = str(data.get("text", "")).strip()
        thumbnail = data.get("thumbnail")
        lora = data.get("lora", None)
        lora_strength = data.get("lora_strength", None)
        lora_image = data.get("lora_image", None)
        lora_image_strength = data.get("lora_image_strength", None)
        lora_video = data.get("lora_video", None)
        lora_video_strength = data.get("lora_video_strength", None)
        refmod = data.get("refmod", None)
        refmod_weight = data.get("refmod_weight", None)

        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name are required"})

        prompts = PromptComposerStore.load_prompts()
        if category not in prompts:
            prompts[category] = {"_prompts_": {}}

        category_data = _ensure_category_data(prompts, category)
        prompt_entries = _ensure_category_prompts_map(category_data)

        # Case-insensitive prompt replacement.
        existing_lower = {
            k.lower(): k
            for k in prompt_entries.keys()
        }
        existing_prompt = {}
        if name.lower() in existing_lower:
            old_name = existing_lower[name.lower()]
            existing_prompt = prompt_entries.get(old_name, {})
            if old_name != name:
                print(f"[PromptComposerStore] Removing old casing '{old_name}' before saving as '{name}'")
                del prompt_entries[old_name]

        entry = {"prompt": text}
        if thumbnail is not None:
            entry["thumbnail"] = thumbnail
        elif existing_prompt.get("thumbnail"):
            entry["thumbnail"] = existing_prompt["thumbnail"]
        if lora_image is None:
            if existing_prompt.get("lora_image"):
                entry["lora_image"] = existing_prompt["lora_image"]
            elif existing_prompt.get("lora"):
                entry["lora_image"] = existing_prompt["lora"]
            if "lora_image_strength" in existing_prompt:
                entry["lora_image_strength"] = existing_prompt["lora_image_strength"]
            elif "lora_strength" in existing_prompt:
                entry["lora_image_strength"] = existing_prompt["lora_strength"]
        else:
            normalized_lora_image = _normalize_optional_string(lora_image)
            if normalized_lora_image:
                entry["lora_image"] = normalized_lora_image
                entry["lora_image_strength"] = _normalize_optional_float(lora_image_strength, default=1.0)
        if lora_video is None:
            if existing_prompt.get("lora_video"):
                entry["lora_video"] = existing_prompt["lora_video"]
            if "lora_video_strength" in existing_prompt:
                entry["lora_video_strength"] = existing_prompt["lora_video_strength"]
        else:
            normalized_lora_video = _normalize_optional_string(lora_video)
            if normalized_lora_video:
                entry["lora_video"] = normalized_lora_video
                entry["lora_video_strength"] = _normalize_optional_float(lora_video_strength, default=1.0)
        if lora is not None and lora_image is None:
            normalized_legacy_lora = _normalize_optional_string(lora)
            if normalized_legacy_lora and not entry.get("lora_image"):
                entry["lora_image"] = normalized_legacy_lora
                entry["lora_image_strength"] = _normalize_optional_float(lora_strength, default=1.0)
        if refmod is None:
            if existing_prompt.get("refmod"):
                entry["refmod"] = existing_prompt["refmod"]
            if "refmod_weight" in existing_prompt:
                entry["refmod_weight"] = existing_prompt["refmod_weight"]
        else:
            normalized_refmod = _normalize_optional_string(refmod)
            if normalized_refmod:
                entry["refmod"] = normalized_refmod
                entry["refmod_weight"] = _normalize_optional_float(refmod_weight, default=1.0, minimum=0.0, maximum=10.0)
        if existing_prompt.get("nsfw"):
            entry["nsfw"] = existing_prompt["nsfw"]

        prompt_entries[name] = entry
        prompts[category] = category_data
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
        if category in prompts:
            category_data = _ensure_category_data(prompts, category)
            prompt_entries = _ensure_category_prompts_map(category_data)
            if name in prompt_entries:
                del prompt_entries[name]
                prompts[category] = category_data
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
        if old_category not in prompts:
            return server.web.json_response({"success": False, "error": "Prompt not found"})

        old_category_data = _ensure_category_data(prompts, old_category)
        old_prompt_entries = _ensure_category_prompts_map(old_category_data)
        if old_name not in old_prompt_entries:
            return server.web.json_response({"success": False, "error": "Prompt not found"})

        if new_category not in prompts:
            prompts[new_category] = {"_prompts_": {}}

        new_category_data = _ensure_category_data(prompts, new_category)
        new_prompt_entries = _ensure_category_prompts_map(new_category_data)

        entry = old_prompt_entries.pop(old_name)
        new_prompt_entries[new_name] = entry
        prompts[old_category] = old_category_data
        prompts[new_category] = new_category_data
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
            category_data = _ensure_category_data(prompts, category)
            meta = category_data.get("__meta__", {})
            meta["nsfw"] = not meta.get("nsfw", False)
            category_data["__meta__"] = meta
            prompts[category] = category_data
        else:
            if category not in prompts:
                return server.web.json_response({"success": False, "error": "Prompt not found"})
            category_data = _ensure_category_data(prompts, category)
            prompt_entries = _ensure_category_prompts_map(category_data)
            if name not in prompt_entries:
                return server.web.json_response({"success": False, "error": "Prompt not found"})
            entry = prompt_entries[name]
            if isinstance(entry, dict):
                entry["nsfw"] = not entry.get("nsfw", False)
            prompts[category] = category_data

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
            prompts[category] = {"_prompts_": {}}

        category_data = _ensure_category_data(prompts, category)
        prompt_entries = _ensure_category_prompts_map(category_data)
        if name not in prompt_entries:
            prompt_entries[name] = {"prompt": ""}

        entry = prompt_entries[name]
        if not isinstance(entry, dict):
            entry = {"prompt": str(entry)}
            prompt_entries[name] = entry

        if thumbnail:
            entry["thumbnail"] = thumbnail
        else:
            entry.pop("thumbnail", None)

        prompts[category] = category_data
        PromptComposerStore.save_prompts(prompts)
        return server.web.json_response({"success": True, "prompts": prompts})
    except Exception as e:
        print(f"[PromptComposerStore] Error in save-thumbnail: {e}")
        return server.web.json_response({"success": False, "error": str(e)}, status=500)
