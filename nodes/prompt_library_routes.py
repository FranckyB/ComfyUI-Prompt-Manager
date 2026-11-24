"""Routes on the prompt library the Prompt Manager nodes share (py/prompt_store.py), each served under both the
Basic (/prompt-manager) and the full node's (/prompt-manager-advanced) prefix. Each node's own save route stays in
its module."""

import json
import os

import server

from ..py.backup_manager import atomic_save
from ..py.prompt_store import is_hidden_key, load_prompts, save_prompts

PREFIXES = ("/prompt-manager", "/prompt-manager-advanced")


def _route(method, name):
    """Serve the handler at `name` under both prefixes. A failure answers as JSON, which the frontend shows."""
    def register(handler):
        async def respond(request):
            try:
                return await handler(request)
            except Exception as e:
                print(f"[PromptManager] {name} failed: {e}")
                return server.web.json_response({"success": False, "error": str(e)}, status=500)
        for prefix in PREFIXES:
            getattr(server.PromptServer.instance.routes, method)(f"{prefix}/{name}")(respond)
        return handler
    return register


def _fail(error, status=200):
    return server.web.json_response({"success": False, "error": error}, status=status)


def _text(data, key):
    return str(data.get(key) or "").strip()


def _find(entries, name):
    """The prompt in `entries` named `name`, whatever its case (hidden category keys are not prompts)."""
    return next((key for key in entries if not is_hidden_key(key) and key.lower() == name.lower()), None)


def _safe_abspath(path):
    return os.path.abspath(os.path.expanduser(path or ""))


@_route("get", "get-prompts")
async def get_prompts(request):
    return server.web.json_response(load_prompts())


@_route("post", "save-category")
async def save_category(request):
    data = await request.json()
    category = _text(data, "category_name")
    if not category:
        return _fail("Category name is required")

    prompts = load_prompts()
    existing = {key.lower(): key for key in prompts}
    if category.lower() in existing:
        return _fail(f"Category already exists as '{existing[category.lower()]}'")

    prompts[category] = {"__meta__": {"nsfw": True}} if data.get("nsfw") else {}
    save_prompts(prompts)
    return server.web.json_response({"success": True, "prompts": prompts})


@_route("post", "rename-category")
async def rename_category(request):
    data = await request.json()
    old_category = _text(data, "old_category")
    new_category = _text(data, "new_category")
    if not old_category or not new_category:
        return _fail("Both old and new category names are required")

    prompts = load_prompts()
    if old_category not in prompts:
        return _fail(f"Category '{old_category}' not found")
    others = {key.lower(): key for key in prompts if key.lower() != old_category.lower()}
    if new_category.lower() in others:
        return _fail(f"Category already exists as '{others[new_category.lower()]}'")

    prompts[new_category] = prompts.pop(old_category)
    save_prompts(prompts)
    return server.web.json_response({"success": True, "prompts": prompts, "new_category": new_category})


@_route("post", "delete-category")
async def delete_category(request):
    data = await request.json()
    category = _text(data, "category")
    if not category:
        return _fail("Category name is required")

    prompts = load_prompts()
    if category not in prompts:
        return _fail("Category not found")

    del prompts[category]
    save_prompts(prompts)
    return server.web.json_response({"success": True, "prompts": prompts})


@_route("post", "rename-prompt")
async def rename_prompt(request):
    """Rename a prompt, or move it to another category."""
    data = await request.json()
    category = _text(data, "category")
    old_name = _text(data, "old_name")
    new_name = _text(data, "new_name")
    new_category = _text(data, "new_category") or category
    if not category or not old_name or not new_name:
        return _fail("Category, old name, and new name are required")

    prompts = load_prompts()
    if category not in prompts:
        return _fail("Prompt not found")
    if new_category not in prompts:
        return _fail(f"Category '{new_category}' not found")

    source, target = prompts[category], prompts[new_category]
    current = _find(source, old_name)
    if not current:
        return _fail("Prompt not found")
    clash = _find(target, new_name)
    if clash and not (new_category == category and clash == current):
        return _fail(f"A prompt named '{clash}' already exists in '{new_category}'")

    target[new_name] = source.pop(current)
    save_prompts(prompts)
    return server.web.json_response({"success": True, "prompts": prompts, "new_name": new_name, "new_category": new_category})


@_route("post", "delete-prompt")
async def delete_prompt(request):
    data = await request.json()
    category = _text(data, "category")
    name = _text(data, "name")
    if not category or not name:
        return _fail("Category and prompt name are required")

    prompts = load_prompts()
    if category not in prompts or name not in prompts[category]:
        return _fail("Prompt not found")

    del prompts[category][name]
    save_prompts(prompts)
    return server.web.json_response({"success": True, "prompts": prompts})


@_route("post", "save-thumbnail")
async def save_thumbnail(request):
    """Set a prompt's thumbnail, or remove it when none is sent."""
    data = await request.json()
    category = _text(data, "category")
    name = _text(data, "name")
    thumbnail = data.get("thumbnail")
    if not category or not name:
        return _fail("Category and name required")

    prompts = load_prompts()
    if category not in prompts or name not in prompts[category]:
        return _fail("Prompt not found")

    if thumbnail:
        prompts[category][name]["thumbnail"] = thumbnail
    else:
        prompts[category][name].pop("thumbnail", None)
    save_prompts(prompts)
    return server.web.json_response({"success": True})


@_route("post", "toggle-nsfw")
async def toggle_nsfw(request):
    """Flip the NSFW flag of a category (type "category") or of a prompt (type "prompt")."""
    data = await request.json()
    kind = _text(data, "type")
    category = _text(data, "category")
    name = _text(data, "name")
    if not category:
        return _fail("Category is required")

    prompts = load_prompts()
    if category not in prompts:
        return _fail("Category not found")

    if kind == "category":
        meta = prompts[category].get("__meta__", {})
        meta["nsfw"] = not meta.get("nsfw", False)
        prompts[category]["__meta__"] = meta
    elif kind == "prompt":
        if not name or name not in prompts[category]:
            return _fail("Prompt not found")
        prompts[category][name]["nsfw"] = not prompts[category][name].get("nsfw", False)
    else:
        return _fail("Invalid type, must be 'category' or 'prompt'")

    save_prompts(prompts)
    return server.web.json_response({"success": True, "prompts": prompts})


@_route("post", "import-prompts")
async def import_prompts(request):
    """Import a library: "replace" starts from nothing, "skip_existing" keeps prompts already there, any other mode
    overwrites them."""
    data = await request.json()
    imported_data = data.get("data", {})
    mode = str(data.get("mode", "replace_existing") or "replace_existing").strip().lower()
    if not isinstance(imported_data, dict):
        return _fail("Invalid data format")

    prompts = {} if mode == "replace" else load_prompts()
    imported_prompts = skipped_prompts = created_categories = 0

    for category, category_prompts in imported_data.items():
        if not isinstance(category_prompts, dict):
            continue
        if category not in prompts:
            prompts[category] = {}
            created_categories += 1

        for prompt_name, prompt_data in category_prompts.items():
            # Hidden category metadata keys (not prompt entries); the base prompt and prefix are plain text.
            if is_hidden_key(prompt_name):
                if mode != "skip_existing" or prompt_name not in prompts[category]:
                    prompts[category][prompt_name] = prompt_data
                continue

            # Older exports stored a prompt as its text alone.
            if isinstance(prompt_data, str):
                prompt_data = {"prompt": prompt_data}
            elif not isinstance(prompt_data, dict):
                continue

            if prompt_name in prompts[category] and mode == "skip_existing":
                skipped_prompts += 1
                continue

            # Every field the entry carries (recipe data, negative prompt, thumbnail, flags) comes along; the lists
            # a prompt always has default to empty.
            prompts[category][prompt_name] = {
                "prompt": "",
                "loras_a": [],
                "loras_b": [],
                "loras_c": [],
                "loras_d": [],
                "trigger_words": [],
                **prompt_data,
            }
            imported_prompts += 1

    save_prompts(prompts)
    return server.web.json_response({
        "success": True,
        "prompts": prompts,
        "imported_prompts": imported_prompts,
        "skipped_prompts": skipped_prompts,
        "created_categories": created_categories,
    })


@_route("post", "load-prompts-file")
async def load_prompts_file(request):
    """A library JSON file the user picked with the file browser."""
    data = await request.json()
    file_path = _safe_abspath(data.get("path", ""))
    if not file_path:
        return _fail("Path is required")
    if not file_path.lower().endswith(".json"):
        return _fail("Only JSON files can be opened", 400)
    if not os.path.isfile(file_path):
        return _fail("JSON file not found", 404)

    with open(file_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        return _fail("Invalid JSON data format", 400)
    return server.web.json_response({"success": True, "data": loaded, "path": file_path})


@_route("post", "export-prompts-file")
async def export_prompts_file(request):
    """Write a library to a JSON file the user picked with the file browser."""
    data = await request.json()
    raw_path = data.get("path", "")
    export_path = _safe_abspath(raw_path) if raw_path else ""
    if export_path and not export_path.lower().endswith(".json"):
        export_path = f"{export_path}.json"
    exported_data = data.get("data", {})
    if not export_path:
        return _fail("Export path is required")
    if not isinstance(exported_data, dict):
        return _fail("Invalid data format")

    parent_dir = os.path.dirname(export_path)
    if not parent_dir or not os.path.isdir(parent_dir):
        return _fail("Target folder does not exist")
    if not atomic_save(export_path, exported_data, "PromptManagerExport"):
        return _fail("Failed to save export file", 500)
    return server.web.json_response({"success": True, "path": export_path})
