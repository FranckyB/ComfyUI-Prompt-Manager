"""The prompt library the Prompt Manager nodes share: user/default/prompt_manager_data.json, seeded from
prompts/default_prompts.json the first time. Backups are made once a day at startup (backup_manager.check_backup)."""

import json
import os

import folder_paths

from .backup_manager import atomic_save, load_with_fallback

DEFAULT_PROMPTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "default_prompts.json")

# Category entries that are not prompts: the category's metadata, and its base prompt and prompt prefix.
HIDDEN_KEYS = {"__meta__", "_base_prompt_", "_prompt_prefix_"}


def is_hidden_key(name):
    return str(name or "").strip().lower() in HIDDEN_KEYS


def prompts_path():
    return os.path.join(folder_paths.get_user_directory(), "default", "prompt_manager_data.json")


def load_prompts():
    """The library; the default prompts when there is no library yet. An unreadable file with no readable backup
    loads as empty, never overwritten here."""
    data = load_with_fallback(prompts_path(), "PromptManager")
    if data is not None:
        return data
    if os.path.exists(prompts_path()):
        return {}
    with open(DEFAULT_PROMPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    save_prompts(data)
    return data


def save_prompts(data):
    """Save with categories and their prompts sorted case-insensitively, a category's __meta__ last."""
    sorted_data = {}
    for category in sorted(data, key=str.lower):
        entries = data[category]
        sorted_entries = dict(sorted(((k, v) for k, v in entries.items() if k != "__meta__"), key=lambda item: item[0].lower()))
        if "__meta__" in entries:
            sorted_entries["__meta__"] = entries["__meta__"]
        sorted_data[category] = sorted_entries
    atomic_save(prompts_path(), sorted_data, "PromptManager")
