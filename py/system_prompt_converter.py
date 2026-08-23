"""System-prompt storage format converter.

Old format: {"System Prompts": {"Name": {"prompt": "...", "category": "Image"}}}
New format: {"Image": {"Name": {"prompt": "..."}}, "Video": {...}}

This module is used both by the runtime auto-migration in prompt_generator.py
and by a standalone conversion helper/script.
"""
import json
import os
import shutil

SYSTEM_PROMPT_CATEGORIES = ("Image", "Video", "Audio", "Other")
LEGACY_BUCKET_NAMES = ("system prompts", "system_prompts")


def _is_legacy_bucket(key):
    return str(key).strip().lower() in LEGACY_BUCKET_NAMES


def _coerce_category(category):
    cat = str(category or "").strip()
    return cat if cat in SYSTEM_PROMPT_CATEGORIES else "Other"


def convert_system_prompts(data, preserve_non_system=True):
    """Convert old single-bucket system prompts into category buckets.

    Args:
        data: dict loaded from a prompt_generator_data.json-compatible file.
        preserve_non_system: if True, any top-level keys that are not legacy
            buckets and not system-prompt category buckets are kept as-is.

    Returns:
        A new dict in the category-bucket format. If no conversion is needed,
        the original dict is returned unchanged.
    """
    if not isinstance(data, dict):
        return data

    legacy_bucket_key = None
    for key in data.keys():
        if _is_legacy_bucket(key):
            legacy_bucket_key = key
            break

    if legacy_bucket_key is None:
        # Already in new format (or at least no legacy wrapper). Make sure the
        # known category buckets exist so callers can rely on them.
        return data

    converted = {}
    for key, bucket in data.items():
        if key == legacy_bucket_key:
            continue
        if isinstance(bucket, dict):
            converted[key] = dict(bucket)

    for name, entry in data[legacy_bucket_key].items():
        if str(name).startswith("_"):
            continue
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("prompt", "") or "").strip()
        if not text:
            continue
        category = _coerce_category(entry.get("category"))
        new_entry = {"prompt": text}
        if entry.get("nsfw"):
            new_entry["nsfw"] = True
        if entry.get("thumbnail"):
            new_entry["thumbnail"] = entry["thumbnail"]
        converted.setdefault(category, {})[name] = new_entry

    if preserve_non_system:
        for key, bucket in data.items():
            if key == legacy_bucket_key:
                continue
            if key in SYSTEM_PROMPT_CATEGORIES:
                continue
            if isinstance(bucket, dict):
                converted.setdefault(key, dict(bucket))

    return converted


def convert_system_prompts_file(path, backup=True, indent=2):
    """Convert a JSON file in-place, optionally creating a .bak backup first.

    Args:
        path: path to the JSON file to convert.
        backup: if True and the file exists, copy it to path + ".bak" first.
        indent: JSON indentation to use when writing.

    Returns:
        (converted: bool, message: str)
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return False, f"File not found: {path}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Failed to load {path}: {e}"

    if not isinstance(data, dict):
        return False, f"Unexpected top-level type in {path}: {type(data).__name__}"

    converted = convert_system_prompts(data)
    if converted is data:
        return False, f"No legacy format detected in {path}; nothing to convert."

    if backup and os.path.exists(path):
        backup_path = path + ".bak"
        try:
            shutil.copy2(path, backup_path)
        except Exception as e:
            return False, f"Failed to create backup {backup_path}: {e}"

    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=indent, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return False, f"Failed to write {path}: {e}"

    return True, f"Converted {path} to category buckets."


def ensure_category_buckets(data):
    """Make sure all four system-prompt category buckets exist in a dict."""
    if not isinstance(data, dict):
        return data
    for category in SYSTEM_PROMPT_CATEGORIES:
        if category not in data:
            data[category] = {}
    return data


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python system_prompt_converter.py <path/to/prompt_generator_data.json>")
        sys.exit(1)

    for target in sys.argv[1:]:
        converted, message = convert_system_prompts_file(target, backup=True)
        print(message)
        if not converted and "File not found" not in message:
            sys.exit(0 if "No legacy format detected" in message else 1)
