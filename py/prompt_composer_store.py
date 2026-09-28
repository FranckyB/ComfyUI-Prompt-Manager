"""
Prompt Composer Store.

Canonical storage lives in per-type JSON files under:
    ComfyUI/user/default/prompt_composer/*.json

Legacy v1 composer libraries are converted to canonical v2 on first load,
then the runtime continues using canonical v2 data only.
"""
import io
import json
import os
import re
import shutil
import zipfile
import base64
from datetime import datetime

import folder_paths
import server
from .backup_manager import atomic_save, check_backup, load_with_fallback


SCHEMA_VERSION = 2
COMPOSER_DIRNAME = "prompt_composer"
LEGACY_FILENAME = "prompt_composer_data.json"
LEGACY_BACKUP_FILENAME = "prompt_composer_data.legacy.json"
DEFAULT_PROMPTS_DIRNAME = "default_composer_prompts"
DEFAULT_LEGACY_FILENAME = "default_composer_prompts.json"
CANONICAL_TYPES_KEY = "_types_"
FALLBACK_TYPE_FILE = "misc.json"
TYPE_FILE_SUFFIX = ".json"
TYPE_ICON_SUFFIX = ".png"
DEFAULT_TYPE_ICON_FILENAME = "_default.png"
GROUP_PLACEHOLDER_ICON_FILENAME = "placeholder.png"
ALL_TYPES_ICON_FILENAME = "all.png"
COMPOSER_CATEGORY_KEY_SEPARATOR = "::"


class PromptComposerStore:
    """Load/save Prompt Composer libraries."""

    @staticmethod
    def get_storage_dir():
        return os.path.join(folder_paths.get_user_directory(), "default", COMPOSER_DIRNAME)

    @staticmethod
    def get_data_path():
        return os.path.join(folder_paths.get_user_directory(), "default", LEGACY_FILENAME)

    @staticmethod
    def get_legacy_backup_path():
        return os.path.join(folder_paths.get_user_directory(), "default", LEGACY_BACKUP_FILENAME)

    @staticmethod
    def get_default_prompts_dir():
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", DEFAULT_PROMPTS_DIRNAME)

    @staticmethod
    def get_default_prompts_path():
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", DEFAULT_LEGACY_FILENAME)

    @staticmethod
    def get_js_dir():
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "js")

    @classmethod
    def load_prompts(cls):
        """Return the flattened category-first browser view derived from canonical v2 data."""
        library = cls.load_canonical_prompts()
        return _flatten_canonical_library(library)

    @classmethod
    def load_canonical_prompts(cls):
        storage_dir = cls.get_storage_dir()
        default_dir = cls.get_default_prompts_dir()
        type_files = _list_type_files(storage_dir)
        if type_files:
            return _load_canonical_library_from_files(storage_dir, type_files, default_dir=default_dir)

        legacy_path = cls.get_data_path()
        if os.path.exists(legacy_path):
            legacy_data = _load_legacy_library(legacy_path)
            canonical = _convert_legacy_library_to_canonical(legacy_data)
            _populate_missing_type_icons(canonical, default_dir=default_dir, search_dirs=[storage_dir, default_dir])
            cls.save_prompts(canonical)
            _archive_legacy_file(legacy_path, cls.get_legacy_backup_path())
            return canonical

        default_type_files = _list_type_files(default_dir)
        if default_type_files:
            try:
                canonical = _load_canonical_library_from_files(default_dir, default_type_files, default_dir=default_dir)
                cls.save_prompts(canonical)
                return canonical
            except Exception as exc:
                print(f"[PromptComposerStore] Error loading bundled defaults: {exc}")

        default_path = cls.get_default_prompts_path()
        if os.path.exists(default_path):
            try:
                with open(default_path, "r", encoding="utf-8") as handle:
                    default_data = json.load(handle)
                canonical = _convert_legacy_library_to_canonical(_normalize_prompts_data(default_data))
                _populate_missing_type_icons(canonical, default_dir=default_dir, search_dirs=[default_dir])
                cls.save_prompts(canonical)
                return canonical
            except Exception as exc:
                print(f"[PromptComposerStore] Error loading legacy bundled defaults: {exc}")

        return _new_canonical_library()

    @classmethod
    def save_prompts(cls, data):
        library = _normalize_canonical_library(data)
        storage_dir = cls.get_storage_dir()
        os.makedirs(storage_dir, exist_ok=True)

        expected_files = set()
        for type_file, type_data in _iter_type_items(library):
            expected_files.add(type_file)
            payload = _serialize_type_data(type_file, type_data)
            atomic_save(os.path.join(storage_dir, type_file), payload, "PromptComposerStore")

        for existing_name in _list_type_files(storage_dir):
            if existing_name in expected_files:
                continue
            try:
                os.remove(os.path.join(storage_dir, existing_name))
            except FileNotFoundError:
                pass

    @classmethod
    def save_type_files(cls, data, touched_type_files=None, removed_type_files=None):
        library = _normalize_canonical_library(data)
        storage_dir = cls.get_storage_dir()
        os.makedirs(storage_dir, exist_ok=True)

        touched = _resolve_requested_type_files(library, touched_type_files)
        removed = {
            _normalize_type_file_name(type_file)
            for type_file in (removed_type_files or [])
            if str(type_file or "").strip()
        }
        removed.difference_update(touched)

        for type_file in touched:
            type_data = library.get(CANONICAL_TYPES_KEY, {}).get(type_file)
            if not isinstance(type_data, dict):
                continue
            payload = _serialize_type_data(type_file, type_data)
            atomic_save(os.path.join(storage_dir, type_file), payload, "PromptComposerStore")

        for type_file in removed:
            try:
                os.remove(os.path.join(storage_dir, type_file))
            except FileNotFoundError:
                pass

        return library

    @classmethod
    def sort_prompts_data(cls, data):
        """Compatibility wrapper used by older callers and tests."""
        return _normalize_canonical_library(data)


def _new_canonical_library():
    return {
        "__meta__": {
            "schema_version": SCHEMA_VERSION,
            "storage": "type_files",
        },
        CANONICAL_TYPES_KEY: {},
    }


def _safe_abspath(path):
    return os.path.abspath(os.path.expanduser(path or ""))


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "on"}


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


def _normalize_optional_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _slugify_type_name(value):
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or os.path.splitext(FALLBACK_TYPE_FILE)[0]


def _normalize_type_file_name(value, fallback=FALLBACK_TYPE_FILE):
    raw = str(value or "").strip()
    if not raw:
        return fallback
    if raw.lower().endswith(TYPE_FILE_SUFFIX):
        stem = os.path.splitext(os.path.basename(raw))[0]
    else:
        stem = raw
    slug = _slugify_type_name(stem)
    return f"{slug}{TYPE_FILE_SUFFIX}"


def _type_stem(type_file):
    return os.path.splitext(os.path.basename(str(type_file or "")))[0]


def _type_icon_file_name(type_file):
    stem = _type_stem(type_file)
    return f"{stem}{TYPE_ICON_SUFFIX}" if stem else DEFAULT_TYPE_ICON_FILENAME


def _copy_file_if_missing(source_path, target_path):
    if not source_path or not target_path:
        return
    if not os.path.isfile(source_path) or os.path.exists(target_path):
        return
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy2(source_path, target_path)


def _copy_matching_type_icons(source_dir, target_dir, type_files):
    if not source_dir or not target_dir or not os.path.isdir(source_dir):
        return

    _copy_file_if_missing(
        os.path.join(source_dir, DEFAULT_TYPE_ICON_FILENAME),
        os.path.join(target_dir, DEFAULT_TYPE_ICON_FILENAME),
    )

    for type_file in type_files or []:
        icon_name = _type_icon_file_name(type_file)
        _copy_file_if_missing(
            os.path.join(source_dir, icon_name),
            os.path.join(target_dir, icon_name),
        )


def _read_icon_data_url(path):
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:
        print(f"[PromptComposerStore] Failed to read icon '{path}': {exc}")
        return ""


def _normalize_type_icon(value):
    normalized = str(value or "").strip()
    return normalized if normalized.startswith("data:image/") else ""


def _load_packaged_type_icon(type_file, default_dir=None):
    if not default_dir or not type_file:
        return ""
    path = os.path.join(default_dir, _normalize_type_file_name(type_file))
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except Exception as exc:
        print(f"[PromptComposerStore] Failed to load packaged type JSON '{path}': {exc}")
        return ""
    if not isinstance(loaded, dict):
        return ""
    return _normalize_type_icon(loaded.get("icon") or loaded.get("icon_url") or loaded.get("_type_icon_"))


def _populate_missing_type_icons(library, default_dir=None, search_dirs=None):
    if not isinstance(library, dict):
        return False

    changed = False
    for type_file, type_data in _iter_type_items(library):
        if not isinstance(type_data, dict) or _normalize_type_icon(type_data.get("icon")):
            continue
        icon_data = _load_packaged_type_icon(type_file, default_dir=default_dir)
        if icon_data:
            type_data["icon"] = icon_data
            changed = True

    return changed


def _resolve_type_icon_path(type_file=None, storage_dir=None, fallback_dirs=None):
    normalized_type_file = _normalize_optional_string(type_file)
    icon_name = _type_icon_file_name(normalized_type_file)
    search_dirs = []
    if storage_dir:
        search_dirs.append(storage_dir)
    for directory in fallback_dirs or []:
        if directory and directory not in search_dirs:
            search_dirs.append(directory)

    if not normalized_type_file:
        for directory in search_dirs:
            all_path = os.path.join(directory, ALL_TYPES_ICON_FILENAME)
            if os.path.isfile(all_path):
                return all_path

    for directory in search_dirs:
        specific_path = os.path.join(directory, icon_name)
        if os.path.isfile(specific_path):
            return specific_path

    for directory in search_dirs:
        placeholder_path = os.path.join(directory, GROUP_PLACEHOLDER_ICON_FILENAME)
        if os.path.isfile(placeholder_path):
            return placeholder_path

    for directory in search_dirs:
        default_path = os.path.join(directory, DEFAULT_TYPE_ICON_FILENAME)
        if os.path.isfile(default_path):
            return default_path

    return None


def _default_type_name_from_file(type_file):
    return " ".join(part.capitalize() for part in _type_stem(type_file).split("_") if part) or "Misc"


def _default_subject_type():
    return "subject"


def _default_subject_kind(subject_type="subject"):
    return "character" if _normalize_subject_type(subject_type, _default_subject_type()) == "new_subject" else "other"


def _legacy_subject_type_for_name(name):
    stem = _type_stem(_normalize_type_file_name(name or ""))
    if stem in {"character", "environment"}:
        return "new_subject"
    if stem in {"style", "effect", "lighting", "mood", "composition", "camera"}:
        return "non_subject"
    return _default_subject_type()


def _legacy_subject_kind_for_name(name, subject_type="subject"):
    if _normalize_subject_type(subject_type, _default_subject_type()) != "new_subject":
        return "other"
    stem = _type_stem(_normalize_type_file_name(name or ""))
    if stem == "environment":
        return "environment"
    if stem == "animal":
        return "animal"
    if stem == "other":
        return "other"
    return "character"


def _normalize_subject_type(value, fallback="subject"):
    normalized = str(value or "").strip().lower()
    if normalized in {"new_subject", "subject", "non_subject"}:
        return normalized
    return fallback


def _normalize_subject_kind(value, fallback="other"):
    normalized = str(value or "").strip().lower()
    if normalized == "person":
        return "character"
    if normalized in {"character", "animal", "environment", "other"}:
        return normalized
    return fallback


def _is_hidden_category_entry_key(name):
    normalized = str(name or "").strip().lower()
    return normalized in {
        "__meta__",
        "_base_prompt_",
        "_prompt_prefix_",
        "_prompt_type_",
        "_prompts_",
        "_type_file_",
        "_type_name_",
        "_subject_type_",
        "_subject_kind_",
        "_type_prefix_",
        "_type_base_prompt_",
        "_type_nsfw_",
    }


def _normalize_prompt_entry(entry):
    if isinstance(entry, dict):
        normalized = dict(entry)
        normalized["prompt"] = str(normalized.get("prompt", "") or "")
        if "thumbnail" in normalized and normalized["thumbnail"] is None:
            normalized.pop("thumbnail", None)
        if "nsfw" in normalized:
            normalized["nsfw"] = _coerce_bool(normalized.get("nsfw"))
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
    return prompt_entries


def _normalize_type_category_data(category_name, category_data):
    if not isinstance(category_data, dict):
        category_data = {}

    normalized = {"_prompts_": {}}

    base_prompt = str(category_data.get("base_prompt") or category_data.get("_base_prompt_") or "")
    if base_prompt.strip():
        normalized["base_prompt"] = base_prompt

    prefix = str(category_data.get("prefix") or category_data.get("_prompt_prefix_") or "")
    if prefix.strip():
        normalized["prefix"] = prefix

    if _coerce_bool(category_data.get("nsfw", category_data.get("__meta__", {}).get("nsfw") if isinstance(category_data.get("__meta__"), dict) else False)):
        normalized["nsfw"] = True

    order = _normalize_optional_int(category_data.get("order"))
    if order is not None:
        normalized["order"] = order

    prompts = category_data.get("_prompts_")
    if not isinstance(prompts, dict):
        prompts = category_data.get("prompts")
    if not isinstance(prompts, dict):
        prompts = {
            key: value
            for key, value in category_data.items()
            if not _is_hidden_category_entry_key(key) and key not in {"base_prompt", "prefix", "nsfw", "order", "prompts"}
        }

    for prompt_name, prompt_entry in prompts.items():
        normalized["_prompts_"][str(prompt_name)] = _normalize_prompt_entry(prompt_entry)

    return normalized


def _normalize_category_data(category_data):
    """Normalize legacy flat category data during one-time v1 to v2 migration."""
    if not isinstance(category_data, dict):
        category_data = {}

    normalized = {"_prompts_": {}}
    if _coerce_bool(category_data.get("__meta__", {}).get("nsfw") if isinstance(category_data.get("__meta__"), dict) else False):
        normalized["__meta__"] = {"nsfw": True}

    base_prompt = str(category_data.get("_base_prompt_") or "")
    if base_prompt.strip():
        normalized["_base_prompt_"] = base_prompt

    prompt_type = _normalize_optional_string(category_data.get("_prompt_type_"))
    if prompt_type:
        normalized["_prompt_type_"] = prompt_type

    prompt_prefix = str(category_data.get("_prompt_prefix_") or "")
    if prompt_prefix.strip():
        normalized["_prompt_prefix_"] = prompt_prefix

    for name, entry in _get_category_prompts_map(category_data).items():
        normalized["_prompts_"][str(name)] = _normalize_prompt_entry(entry)

    return normalized


def _normalize_prompts_data(data):
    """Normalize a legacy flat category-first library for startup migration."""
    if not isinstance(data, dict):
        return {}

    normalized = {}
    top_level_meta = data.get("__meta__")
    if isinstance(top_level_meta, dict):
        normalized["__meta__"] = dict(top_level_meta)

    for category, category_data in data.items():
        if category == "__meta__":
            continue
        normalized[str(category)] = _normalize_category_data(category_data)
    return normalized


def _normalize_type_data(type_file, type_data):
    if not isinstance(type_data, dict):
        type_data = {}

    normalized_subject_type = _normalize_subject_type(type_data.get("subject_type"), _default_subject_type())

    normalized = {
        "file": type_file,
        "name": _normalize_optional_string(type_data.get("name")) or _default_type_name_from_file(type_file),
        "subject_type": normalized_subject_type,
        "subject_kind": _normalize_subject_kind(
            type_data.get("subject_kind"),
            _legacy_subject_kind_for_name(type_data.get("name") or type_file, normalized_subject_type),
        ),
        "categories": {},
    }

    prefix = str(type_data.get("prefix") or "")
    if prefix.strip():
        normalized["prefix"] = prefix

    base_prompt = str(type_data.get("base_prompt") or "")
    if base_prompt.strip():
        normalized["base_prompt"] = base_prompt

    if _coerce_bool(type_data.get("nsfw")):
        normalized["nsfw"] = True

    icon = _normalize_type_icon(type_data.get("icon") or type_data.get("icon_url") or type_data.get("_type_icon_"))
    if icon:
        normalized["icon"] = icon

    order = _normalize_optional_int(type_data.get("order"))
    if order is not None:
        normalized["order"] = order

    categories = type_data.get("categories")
    if not isinstance(categories, dict):
        categories = {}

    for category_name, category_data in categories.items():
        normalized["categories"][str(category_name)] = _normalize_type_category_data(category_name, category_data)

    return normalized


def _normalize_canonical_library(data):
    library = _new_canonical_library()
    if isinstance(data, dict) and isinstance(data.get("__meta__"), dict):
        library["__meta__"].update(dict(data["__meta__"]))
    library["__meta__"]["schema_version"] = SCHEMA_VERSION
    library["__meta__"]["storage"] = "type_files"

    raw_types = data.get(CANONICAL_TYPES_KEY) if isinstance(data, dict) else {}
    if not isinstance(raw_types, dict):
        raw_types = {}

    for raw_type_file, raw_type_data in _sorted_type_items(raw_types):
        type_file = _normalize_type_file_name(raw_type_file)
        library[CANONICAL_TYPES_KEY][type_file] = _normalize_type_data(type_file, raw_type_data)

    return library


def _serialize_type_data(type_file, type_data):
    normalized = _normalize_type_data(type_file, type_data)
    payload = {
        "name": normalized["name"],
        "subject_type": normalized["subject_type"],
        "subject_kind": normalized["subject_kind"],
        "categories": {},
    }

    if normalized.get("prefix"):
        payload["prefix"] = normalized["prefix"]
    if normalized.get("base_prompt"):
        payload["base_prompt"] = normalized["base_prompt"]
    if normalized.get("nsfw"):
        payload["nsfw"] = True
    if normalized.get("icon"):
        payload["icon"] = normalized["icon"]
    if "order" in normalized:
        payload["order"] = normalized["order"]

    for category_name, category_data in sorted(normalized["categories"].items(), key=lambda item: item[0].lower()):
        serialized_category = {"_prompts_": {}}
        if category_data.get("prefix"):
            serialized_category["prefix"] = category_data["prefix"]
        if category_data.get("base_prompt"):
            serialized_category["base_prompt"] = category_data["base_prompt"]
        if category_data.get("nsfw"):
            serialized_category["nsfw"] = True
        if "order" in category_data:
            serialized_category["order"] = category_data["order"]
        serialized_category["_prompts_"] = dict(sorted(
            ((name, _normalize_prompt_entry(entry)) for name, entry in _get_category_prompts_map(category_data).items()),
            key=lambda item: item[0].lower(),
        ))
        payload["categories"][category_name] = serialized_category

    return payload


def _sorted_type_items(types):
    if not isinstance(types, dict):
        return []

    entries = []
    has_explicit_order = False
    for type_file, type_data in types.items():
        order = _normalize_optional_int(type_data.get("order")) if isinstance(type_data, dict) else None
        if order is not None:
            has_explicit_order = True
        entries.append((str(type_file), type_data, order))

    if not has_explicit_order:
        return sorted(((type_file, type_data) for type_file, type_data, _order in entries), key=lambda item: str(item[0]).lower())

    return [
        (type_file, type_data)
        for type_file, type_data, _order in sorted(
            entries,
            key=lambda item: (
                item[2] is None,
                item[2] if item[2] is not None else 0,
                str(item[0]).lower(),
            ),
        )
    ]


def _iter_type_items(library):
    types = library.get(CANONICAL_TYPES_KEY, {}) if isinstance(library, dict) else {}
    return _sorted_type_items(types)


def _ordered_type_files(library):
    return [type_file for type_file, _type_data in _iter_type_items(library)]


def _resolve_requested_type_files(library, requested_type_files):
    resolved = []
    seen = set()
    for requested in requested_type_files or []:
        raw = str(requested or "").strip()
        if not raw:
            continue
        canonical = _find_type_case_insensitive(library, raw) or _normalize_type_file_name(raw)
        if canonical in seen:
            continue
        seen.add(canonical)
        resolved.append(canonical)
    return resolved


def _type_orders_are_dense(library):
    ordered_type_files = _ordered_type_files(library)
    if not ordered_type_files:
        return True

    types = library.get(CANONICAL_TYPES_KEY, {}) if isinstance(library, dict) else {}
    expected_order = list(range(len(ordered_type_files)))
    actual_order = [
        _normalize_optional_int(types.get(type_file, {}).get("order"))
        for type_file in ordered_type_files
    ]
    return actual_order == expected_order


def _assign_type_orders(library, ordered_type_files, start_index=0, end_index=None):
    types = library.get(CANONICAL_TYPES_KEY, {}) if isinstance(library, dict) else {}
    if not isinstance(types, dict) or not ordered_type_files:
        return

    max_index = len(ordered_type_files) - 1
    first_index = max(0, int(start_index or 0))
    last_index = max_index if end_index is None else min(max_index, int(end_index))
    if last_index < first_index:
        return

    for index in range(first_index, last_index + 1):
        type_file = ordered_type_files[index]
        type_data = types.get(type_file)
        if isinstance(type_data, dict):
            type_data["order"] = index


def _ensure_dense_type_orders(library):
    ordered_type_files = _ordered_type_files(library)
    if ordered_type_files and not _type_orders_are_dense(library):
        _assign_type_orders(library, ordered_type_files)
    return ordered_type_files


def _list_type_files(storage_dir):
    if not storage_dir or not os.path.isdir(storage_dir):
        return []
    try:
        return sorted(
            [
                name for name in os.listdir(storage_dir)
                if name.lower().endswith(TYPE_FILE_SUFFIX) and os.path.isfile(os.path.join(storage_dir, name))
            ],
            key=str.lower,
        )
    except FileNotFoundError:
        return []


def _load_canonical_library_from_files(storage_dir, type_files, default_dir=None):
    library = _new_canonical_library()
    for type_file in type_files:
        path = os.path.join(storage_dir, type_file)
        loaded = None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except Exception as exc:
            print(f"[PromptComposerStore] Error loading type file '{type_file}': {exc}")
            check_backup(path)
            loaded = load_with_fallback(path, "PromptComposerStore")
        if not isinstance(loaded, dict):
            print(f"[PromptComposerStore] Skipping invalid type file '{type_file}'")
            continue
        normalized = _normalize_type_data(type_file, loaded)
        library[CANONICAL_TYPES_KEY][type_file] = normalized
    return library


def _load_legacy_library(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            return _normalize_prompts_data(loaded)
    except Exception as exc:
        print(f"[PromptComposerStore] Error loading legacy data: {exc}")
        check_backup(path)
        loaded = load_with_fallback(path, "PromptComposerStore")
        if isinstance(loaded, dict):
            return _normalize_prompts_data(loaded)
        raise
    return {}


def _archive_legacy_file(source_path, backup_path):
    if not source_path or not os.path.exists(source_path):
        return
    target_path = backup_path
    if os.path.exists(target_path):
        stem, ext = os.path.splitext(target_path)
        counter = 1
        while os.path.exists(f"{stem}.{counter}{ext}"):
            counter += 1
        target_path = f"{stem}.{counter}{ext}"
    shutil.move(source_path, target_path)


def _convert_legacy_library_to_canonical(legacy_data):
    legacy = _normalize_prompts_data(legacy_data)
    library = _new_canonical_library()
    if isinstance(legacy.get("__meta__"), dict):
        library["__meta__"].update(dict(legacy["__meta__"]))

    ordered_type_files = []
    seen_type_files = set()
    for category_name, category_data in legacy.items():
        if category_name == "__meta__":
            continue
        prompt_type = _normalize_optional_string(category_data.get("_prompt_type_"))
        type_file = _normalize_type_file_name(prompt_type or category_name or FALLBACK_TYPE_FILE)
        type_entry = library[CANONICAL_TYPES_KEY].setdefault(type_file, {
            "file": type_file,
            "name": _default_type_name_from_file(type_file),
            "subject_type": _legacy_subject_type_for_name(prompt_type or category_name),
            "categories": {},
        })
        if type_file not in seen_type_files:
            ordered_type_files.append(type_file)
            seen_type_files.add(type_file)
        if prompt_type:
            type_entry["name"] = _default_type_name_from_file(type_file)
            type_entry["subject_type"] = _legacy_subject_type_for_name(prompt_type)

        type_entry["categories"][category_name] = _normalize_type_category_data(category_name, {
            "base_prompt": category_data.get("_base_prompt_"),
            "prefix": category_data.get("_prompt_prefix_"),
            "nsfw": _coerce_bool(category_data.get("__meta__", {}).get("nsfw") if isinstance(category_data.get("__meta__"), dict) else False),
            "_prompts_": _get_category_prompts_map(category_data),
        })

    _assign_type_orders(library, ordered_type_files)

    return _normalize_canonical_library(library)


def _normalize_library_input(data):
    if not isinstance(data, dict):
        raise ValueError("Prompt Composer library must be a JSON object")
    if CANONICAL_TYPES_KEY not in data:
        raise ValueError("Prompt Composer requires canonical v2 library data with _types_")
    return _normalize_canonical_library(data)


def _type_nsfw(type_data):
    return _coerce_bool(type_data.get("nsfw")) if isinstance(type_data, dict) else False


def _category_nsfw(category_data):
    return _coerce_bool(category_data.get("nsfw")) if isinstance(category_data, dict) else False


def _flat_category_key(type_file, category_name):
    normalized_category = str(category_name or "").strip()
    normalized_type_file = _normalize_type_file_name(type_file) if type_file else ""
    if not normalized_type_file:
        return normalized_category
    return f"{normalized_type_file}{COMPOSER_CATEGORY_KEY_SEPARATOR}{normalized_category}"


def _flatten_canonical_library(library):
    canonical = _normalize_canonical_library(library)
    flat = {"__meta__": dict(canonical.get("__meta__", {}))}
    for type_file, type_data in _iter_type_items(canonical):
        type_key = _type_stem(type_file)
        type_name = type_data.get("name") or _default_type_name_from_file(type_file)
        type_icon = _normalize_type_icon(type_data.get("icon"))
        type_prefix = str(type_data.get("prefix") or "")
        type_base_prompt = str(type_data.get("base_prompt") or "")
        type_subject_type = _normalize_subject_type(type_data.get("subject_type"), _default_subject_type())
        type_subject_kind = _normalize_subject_kind(
            type_data.get("subject_kind"),
            _default_subject_kind(type_subject_type),
        )
        type_is_nsfw = _type_nsfw(type_data)

        categories = type_data.get("categories", {}) if isinstance(type_data, dict) else {}
        for category_name, category_data in sorted(categories.items(), key=lambda item: item[0].lower()):
            flat_category = {"_prompts_": {}}
            flat_category["_category_name_"] = category_name
            category_base_prompt = str(category_data.get("base_prompt") or "")
            category_prefix = str(category_data.get("prefix") or "")
            effective_base_prompt = str(category_data.get("base_prompt") or type_base_prompt or "")
            effective_prefix = str(category_data.get("prefix") or type_prefix or "")
            if effective_base_prompt.strip():
                flat_category["_base_prompt_"] = effective_base_prompt
            if effective_prefix.strip():
                flat_category["_prompt_prefix_"] = effective_prefix
            if category_base_prompt.strip():
                flat_category["_category_base_prompt_"] = category_base_prompt
            if category_prefix.strip():
                flat_category["_category_prefix_"] = category_prefix
            flat_category["_prompt_type_"] = type_key
            flat_category["_type_file_"] = type_file
            flat_category["_type_name_"] = type_name
            if type_icon:
                flat_category["_type_icon_"] = type_icon
                flat_category["_type_icon_url_"] = type_icon
            flat_category["_subject_type_"] = type_subject_type
            flat_category["_subject_kind_"] = type_subject_kind
            if type_prefix.strip():
                flat_category["_type_prefix_"] = type_prefix
            if type_base_prompt.strip():
                flat_category["_type_base_prompt_"] = type_base_prompt
            if type_is_nsfw:
                flat_category["_type_nsfw_"] = True
            if _category_nsfw(category_data) or type_is_nsfw:
                flat_category["__meta__"] = {"nsfw": True}
            flat_category["_prompts_"] = dict(sorted(
                ((name, _normalize_prompt_entry(entry)) for name, entry in _get_category_prompts_map(category_data).items()),
                key=lambda item: item[0].lower(),
            ))
            flat[_flat_category_key(type_file, category_name)] = flat_category
    return flat


def _find_category_case_insensitive(prompts_data, category):
    if not isinstance(prompts_data, dict):
        return None
    target = str(category or "").strip().lower()
    if not target:
        return None
    for existing_category in prompts_data.keys():
        if existing_category == "__meta__":
            continue
        if str(existing_category).strip().lower() == target:
            return existing_category
    return None


def _find_prompt_case_insensitive(category_data, name):
    prompt_entries = _get_category_prompts_map(category_data)
    if not isinstance(prompt_entries, dict):
        return None, None
    target = str(name or "").strip().lower()
    if not target or target == "__meta__":
        return None, None
    if name in prompt_entries:
        return prompt_entries[name], name
    for entry_name, entry in prompt_entries.items():
        if str(entry_name).strip().lower() == target:
            return entry, entry_name
    return None, None


def _find_type_case_insensitive(library, type_file):
    types = library.get(CANONICAL_TYPES_KEY, {}) if isinstance(library, dict) else {}
    target = _normalize_type_file_name(type_file) if type_file else ""
    if not target:
        return None
    for existing_type_file in types.keys():
        if str(existing_type_file).lower() == target.lower():
            return existing_type_file
    return None


def _find_category_locations(library, category, type_file=None):
    types = library.get(CANONICAL_TYPES_KEY, {}) if isinstance(library, dict) else {}
    target = str(category or "").strip().lower()
    if not target:
        return []

    selected_type_files = []
    if type_file:
        canonical_type_file = _find_type_case_insensitive(library, type_file)
        if canonical_type_file:
            selected_type_files = [canonical_type_file]
    else:
        selected_type_files = list(types.keys())

    matches = []
    for current_type_file in selected_type_files:
        type_data = types.get(current_type_file, {})
        categories = type_data.get("categories", {}) if isinstance(type_data, dict) else {}
        for existing_category, category_data in categories.items():
            if str(existing_category).strip().lower() == target:
                matches.append((current_type_file, type_data, existing_category, category_data))
    return matches


def _locate_category(library, category, type_file=None):
    matches = _find_category_locations(library, category, type_file=type_file)
    if not matches:
        return None, "Category not found"
    if len(matches) > 1 and not type_file:
        return None, f"Category '{category}' exists in multiple types. Please reselect it from the composer browser."
    return matches[0], None


def _ensure_type(library, type_file, name=None, subject_type=None, nsfw=False, prefix="", base_prompt=""):
    normalized_type_file = _normalize_type_file_name(type_file)
    types = library.setdefault(CANONICAL_TYPES_KEY, {})
    type_data = types.get(normalized_type_file)
    if not isinstance(type_data, dict):
        normalized_subject_type = _normalize_subject_type(subject_type, _default_subject_type())
        type_data = {
            "file": normalized_type_file,
            "name": _normalize_optional_string(name) or _default_type_name_from_file(normalized_type_file),
            "subject_type": normalized_subject_type,
            "subject_kind": _legacy_subject_kind_for_name(name or normalized_type_file, normalized_subject_type),
            "categories": {},
        }
        types[normalized_type_file] = type_data
    if _normalize_optional_string(name):
        type_data["name"] = _normalize_optional_string(name)
    if subject_type is not None:
        type_data["subject_type"] = _normalize_subject_type(subject_type, type_data.get("subject_type", "subject"))
        type_data["subject_kind"] = _normalize_subject_kind(
            type_data.get("subject_kind"),
            _default_subject_kind(type_data.get("subject_type", _default_subject_type())),
        )
    if _coerce_bool(nsfw):
        type_data["nsfw"] = True
    if str(prefix or "").strip():
        type_data["prefix"] = str(prefix)
    if str(base_prompt or "").strip():
        type_data["base_prompt"] = str(base_prompt)
    return type_data


def _ensure_type_category(type_data, category_name):
    categories = type_data.setdefault("categories", {})
    category_data = categories.get(category_name)
    if not isinstance(category_data, dict):
        category_data = {"_prompts_": {}}
        categories[category_name] = category_data
    if "_prompts_" not in category_data or not isinstance(category_data.get("_prompts_"), dict):
        category_data["_prompts_"] = {
            key: _normalize_prompt_entry(value)
            for key, value in category_data.items()
            if key not in {"base_prompt", "prefix", "nsfw", "order", "_prompts_"}
        }
    return category_data


def _count_prompt_totals(data):
    library = _normalize_library_input(data)
    category_count = 0
    prompt_count = 0
    for _type_file, type_data in _iter_type_items(library):
        categories = type_data.get("categories", {}) if isinstance(type_data, dict) else {}
        category_count += len(categories)
        for category_data in categories.values():
            prompt_entries = _get_category_prompts_map(category_data)
            if isinstance(prompt_entries, dict):
                prompt_count += len(prompt_entries)
    return category_count, prompt_count


def _normalize_export_path(raw_path):
    candidate = _safe_abspath(raw_path)
    if not candidate:
        return ""
    if not candidate.lower().endswith(TYPE_FILE_SUFFIX):
        candidate = f"{candidate}{TYPE_FILE_SUFFIX}"
    return candidate


def _looks_like_single_type_payload(data):
    return (
        isinstance(data, dict)
        and not isinstance(data.get(CANONICAL_TYPES_KEY), dict)
        and isinstance(data.get("categories"), dict)
        and any(key in data for key in {"name", "subject_type", "subject_kind", "categories", "prefix", "base_prompt", "nsfw", "order"})
    )


def _merge_canonical_libraries(base_library, incoming_library):
    merged = _normalize_canonical_library(base_library)
    incoming = _normalize_canonical_library(incoming_library)
    target_types = merged.setdefault(CANONICAL_TYPES_KEY, {})
    for type_file, type_data in _iter_type_items(incoming):
        target_types[type_file] = _normalize_type_data(type_file, type_data)
    return _normalize_canonical_library(merged)


def _normalize_uploaded_json_document(file_name, loaded):
    if _looks_like_single_type_payload(loaded):
        type_file = _normalize_type_file_name(os.path.basename(file_name or "") or FALLBACK_TYPE_FILE)
        library = _new_canonical_library()
        library[CANONICAL_TYPES_KEY][type_file] = _normalize_type_data(type_file, loaded)
        return _normalize_canonical_library(library)
    if CANONICAL_TYPES_KEY not in loaded:
        raise ValueError("Only canonical v2 Prompt Composer libraries or single-type v2 JSON files are supported")
    return _normalize_library_input(loaded)


def _load_uploaded_library(file_name, raw_bytes):
    if not raw_bytes:
        raise ValueError("Uploaded file is empty")

    normalized_name = str(file_name or "upload").strip()
    lower_name = normalized_name.lower()

    if lower_name.endswith(".zip"):
        combined_library = _new_canonical_library()
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            json_members = [
                member for member in archive.namelist()
                if member and not member.endswith("/") and member.lower().endswith(TYPE_FILE_SUFFIX)
            ]
            if not json_members:
                raise ValueError("ZIP file does not contain any JSON files")
            for member in json_members:
                with archive.open(member, "r") as handle:
                    member_bytes = handle.read()
                try:
                    loaded = json.loads(member_bytes.decode("utf-8"))
                except Exception as exc:
                    raise ValueError(f"Failed to parse '{member}': {exc}") from exc
                if not isinstance(loaded, dict):
                    raise ValueError(f"Invalid JSON structure in '{member}'")
                combined_library = _merge_canonical_libraries(
                    combined_library,
                    _normalize_uploaded_json_document(member, loaded),
                )
        return _normalize_canonical_library(combined_library)

    try:
        loaded = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("Invalid JSON structure")
    return _normalize_uploaded_json_document(normalized_name, loaded)


def _backup_composer_storage(storage_dir):
    normalized_storage_dir = _safe_abspath(storage_dir)
    if not normalized_storage_dir or not os.path.isdir(normalized_storage_dir):
        return None

    existing_entries = [
        name for name in os.listdir(normalized_storage_dir)
        if name and name != "_backup"
    ]
    if not existing_entries:
        return None

    backup_root = os.path.join(normalized_storage_dir, "_backup")
    os.makedirs(backup_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(backup_root, timestamp)
    counter = 1
    while os.path.exists(backup_dir):
        counter += 1
        backup_dir = os.path.join(backup_root, f"{timestamp}_{counter:02d}")
    os.makedirs(backup_dir, exist_ok=True)

    for entry_name in existing_entries:
        shutil.move(
            os.path.join(normalized_storage_dir, entry_name),
            os.path.join(backup_dir, entry_name),
        )

    return backup_dir


async def _read_uploaded_library_request(request):
    post_data = await request.post()
    upload = post_data.get("file")
    if not upload:
        raise ValueError("File upload is required")

    raw_bytes = upload.file.read() if getattr(upload, "file", None) else upload
    file_name = getattr(upload, "filename", "upload")
    library = _load_uploaded_library(file_name, raw_bytes)
    return post_data, file_name, library


def _resolve_target_type_file(current_type_file, requested_prompt_type, fallback_category="misc"):
    normalized_prompt_type = _normalize_optional_string(requested_prompt_type)
    if normalized_prompt_type:
        return _normalize_type_file_name(normalized_prompt_type)
    if current_type_file:
        return current_type_file
    return _normalize_type_file_name(fallback_category or FALLBACK_TYPE_FILE)


@server.PromptServer.instance.routes.get("/prompt-manager/compose/get-prompts")
async def compose_get_prompts(request):
    try:
        return server.web.json_response(PromptComposerStore.load_canonical_prompts())
    except Exception as exc:
        print(f"[PromptComposerStore] Error in get-prompts: {exc}")
        return server.web.json_response({"error": str(exc)}, status=500)


@server.PromptServer.instance.routes.get("/prompt-manager/compose/type-icon")
async def compose_get_type_icon(request):
    try:
        requested_type_file = _normalize_optional_string(request.query.get("type_file"))
        icon_path = _resolve_type_icon_path(
            requested_type_file or None,
            storage_dir=PromptComposerStore.get_storage_dir(),
            fallback_dirs=[PromptComposerStore.get_default_prompts_dir(), PromptComposerStore.get_js_dir()],
        )
        if not icon_path:
            return server.web.Response(text="Type icon not found", status=404)
        return server.web.FileResponse(icon_path, headers={"Cache-Control": "no-store"})
    except Exception as exc:
        print(f"[PromptComposerStore] Error in type-icon: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/save-category")
async def compose_save_category(request):
    try:
        data = await request.json()
        category_name = str(data.get("category_name", "")).strip()
        if not category_name:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        library = PromptComposerStore.load_canonical_prompts()
        prompt_type = _normalize_optional_string(data.get("prompt_type"))
        target_type_file = _normalize_type_file_name(data.get("type_file") or prompt_type or FALLBACK_TYPE_FILE)
        location, error = _locate_category(library, category_name, type_file=target_type_file)
        if location:
            return server.web.json_response({
                "success": False,
                "error": f"Category already exists as '{location[2]}'",
            })

        target_type = _ensure_type(
            library,
            target_type_file,
            name=_normalize_optional_string(data.get("type_name")) or _default_type_name_from_file(target_type_file),
            subject_type=data.get("subject_type"),
            nsfw=_coerce_bool(data.get("type_nsfw")),
            prefix=str(data.get("type_prefix") or ""),
            base_prompt=str(data.get("type_base_prompt") or ""),
        )
        category_data = _ensure_type_category(target_type, category_name)
        if _coerce_bool(data.get("nsfw")):
            category_data["nsfw"] = True

        library = PromptComposerStore.save_type_files(library, [target_type_file])
        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in save-category: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/save-category-base-prompt")
async def compose_save_category_base_prompt(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        if not category:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        library = PromptComposerStore.load_canonical_prompts()
        location, error = _locate_category(library, category, type_file=data.get("type_file"))
        if error:
            return server.web.json_response({"success": False, "error": error})
        type_file, type_data, canonical_category, category_data = location

        base_prompt = str(data.get("base_prompt", ""))
        if base_prompt.strip():
            category_data["base_prompt"] = base_prompt
        else:
            category_data.pop("base_prompt", None)

        type_data.setdefault("categories", {})[canonical_category] = category_data
        library[CANONICAL_TYPES_KEY][type_file] = type_data
        library = PromptComposerStore.save_type_files(library, [type_file])
        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in save-category-base-prompt: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/save-category-settings")
async def compose_save_category_settings(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        if not category:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        library = PromptComposerStore.load_canonical_prompts()
        location, error = _locate_category(library, category, type_file=data.get("type_file"))
        if error:
            return server.web.json_response({"success": False, "error": error})
        current_type_file, current_type_data, canonical_category, category_data = location

        target_type_file = _resolve_target_type_file(current_type_file, data.get("prompt_type"), fallback_category=canonical_category)
        if target_type_file != current_type_file:
            existing_target_location, _existing_target_error = _locate_category(library, canonical_category, type_file=target_type_file)
            if existing_target_location:
                return server.web.json_response({
                    "success": False,
                    "error": f"Category already exists as '{existing_target_location[2]}'",
                })
            current_type_data.get("categories", {}).pop(canonical_category, None)
            target_type_data = _ensure_type(
                library,
                target_type_file,
                name=_normalize_optional_string(data.get("type_name")) or _default_type_name_from_file(target_type_file),
                subject_type=data.get("subject_type"),
                nsfw=_coerce_bool(data.get("type_nsfw")),
                prefix=str(data.get("type_prefix") or ""),
                base_prompt=str(data.get("type_base_prompt") or ""),
            )
            target_type_data.setdefault("categories", {})[canonical_category] = category_data
            current_type_data = library[CANONICAL_TYPES_KEY].get(current_type_file, current_type_data)
        else:
            target_type_data = current_type_data

        base_prompt = str(data.get("base_prompt", ""))
        if base_prompt.strip():
            category_data["base_prompt"] = base_prompt
        else:
            category_data.pop("base_prompt", None)

        prefix = str(data.get("prefix", data.get("prompt_prefix", "")) or "")
        if prefix.strip():
            category_data["prefix"] = prefix
        else:
            category_data.pop("prefix", None)

        if "nsfw" in data:
            if _coerce_bool(data.get("nsfw")):
                category_data["nsfw"] = True
            else:
                category_data.pop("nsfw", None)

        target_type_data.setdefault("categories", {})[canonical_category] = category_data
        if not current_type_data.get("categories"):
            library[CANONICAL_TYPES_KEY].pop(current_type_file, None)

        removed_type_files = [current_type_file] if current_type_file != target_type_file and not current_type_data.get("categories") else []
        touched_type_files = [target_type_file]
        if current_type_file != target_type_file and current_type_data.get("categories"):
            touched_type_files.append(current_type_file)
        library = PromptComposerStore.save_type_files(library, touched_type_files, removed_type_files=removed_type_files)
        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in save-category-settings: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/save-type-settings")
async def compose_save_type_settings(request):
    try:
        data = await request.json()
        requested_prompt_type = _normalize_optional_string(data.get("prompt_type"))
        requested_type_file = data.get("type_file")
        if not requested_prompt_type and not requested_type_file:
            return server.web.json_response({"success": False, "error": "Prompt group is required"})

        library = PromptComposerStore.load_canonical_prompts()
        target_type_file = _find_type_case_insensitive(library, requested_type_file)
        if not target_type_file:
            candidate_file = _resolve_target_type_file("", requested_prompt_type, fallback_category="misc")
            target_type_file = _find_type_case_insensitive(library, candidate_file) or candidate_file
        existing_types = library.get(CANONICAL_TYPES_KEY, {}) if isinstance(library, dict) else {}
        is_new_type = target_type_file not in existing_types
        if is_new_type:
            _ensure_dense_type_orders(library)

        target_type_data = _ensure_type(
            library,
            target_type_file,
            name=_normalize_optional_string(data.get("type_name")) or _default_type_name_from_file(target_type_file),
        )
        if is_new_type:
            target_type_data["order"] = len(library.get(CANONICAL_TYPES_KEY, {})) - 1

        if "subject_type" in data:
            target_type_data["subject_type"] = _normalize_subject_type(
                data.get("subject_type"),
                target_type_data.get("subject_type", _default_subject_type()),
            )
        if "subject_kind" in data:
            target_type_data["subject_kind"] = _normalize_subject_kind(
                data.get("subject_kind"),
                _default_subject_kind(target_type_data.get("subject_type", _default_subject_type())),
            )
        elif "subject_type" in data:
            target_type_data["subject_kind"] = _normalize_subject_kind(
                target_type_data.get("subject_kind"),
                _default_subject_kind(target_type_data.get("subject_type", _default_subject_type())),
            )

        prefix = str(data.get("prefix", "") or "")
        if prefix.strip():
            target_type_data["prefix"] = prefix
        else:
            target_type_data.pop("prefix", None)

        icon = _normalize_type_icon(data.get("icon"))
        if icon:
            target_type_data["icon"] = icon
        elif "icon" in data:
            target_type_data.pop("icon", None)

        base_prompt = str(data.get("base_prompt", "") or "")
        if base_prompt.strip():
            target_type_data["base_prompt"] = base_prompt
        else:
            target_type_data.pop("base_prompt", None)

        initial_category_name = str(data.get("initial_category_name", "") or "").strip()
        if initial_category_name:
            existing_category_location, _existing_category_error = _locate_category(
                library,
                initial_category_name,
                type_file=target_type_file,
            )
            if existing_category_location:
                return server.web.json_response({
                    "success": False,
                    "error": f"Category already exists as '{existing_category_location[2]}'",
                })
            initial_category_data = _ensure_type_category(target_type_data, initial_category_name)
            if _coerce_bool(data.get("initial_category_nsfw")):
                initial_category_data["nsfw"] = True

        library = PromptComposerStore.save_type_files(library, [target_type_file])
        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
            "type_file": target_type_file,
            "type_name": target_type_data.get("name") or _default_type_name_from_file(target_type_file),
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in save-type-settings: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/rename-category")
async def compose_rename_category(request):
    try:
        data = await request.json()
        old_category = str(data.get("old_category", "")).strip()
        new_category = str(data.get("new_category", "")).strip()
        if not old_category or not new_category:
            return server.web.json_response({"success": False, "error": "Both old and new category names are required"})

        library = PromptComposerStore.load_canonical_prompts()
        location, error = _locate_category(library, old_category, type_file=data.get("type_file"))
        if error:
            return server.web.json_response({"success": False, "error": error})
        source_type_file, source_type_data, canonical_category, category_data = location

        requested_target_type_file = data.get("new_type_file") or data.get("type_file") or source_type_file
        target_type_file = _find_type_case_insensitive(library, requested_target_type_file) or _normalize_type_file_name(requested_target_type_file)
        target_type_data = _ensure_type(
            library,
            target_type_file,
            name=_normalize_optional_string(data.get("target_type_name")) or _default_type_name_from_file(target_type_file),
        )

        conflicting_location, _conflict_error = _locate_category(library, new_category, type_file=target_type_file)
        if conflicting_location:
            same_category = (
                str(conflicting_location[0]).lower() == str(source_type_file).lower()
                and str(conflicting_location[2]).lower() == str(canonical_category).lower()
            )
            if same_category:
                conflicting_location = None

        if conflicting_location:
            return server.web.json_response({
                "success": False,
                "error": f"Category already exists as '{conflicting_location[2]}'",
            })

        source_categories = source_type_data.setdefault("categories", {})
        source_categories.pop(canonical_category, None)
        target_categories = target_type_data.setdefault("categories", {})
        target_categories[new_category] = category_data

        removed_type_files = [source_type_file] if source_type_file != target_type_file and not source_type_data.get("categories") else []
        touched_type_files = [target_type_file]
        if source_type_file != target_type_file and source_type_data.get("categories"):
            touched_type_files.append(source_type_file)
        library = PromptComposerStore.save_type_files(library, touched_type_files, removed_type_files=removed_type_files)
        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
            "new_category": new_category,
            "type_file": target_type_file,
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in rename-category: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/rename-type")
async def compose_rename_type(request):
    try:
        data = await request.json()
        target_type_file = _find_type_case_insensitive(library := PromptComposerStore.load_canonical_prompts(), data.get("type_file") or data.get("prompt_type"))
        new_name = str(data.get("new_name", "")).strip()
        if not target_type_file:
            return server.web.json_response({"success": False, "error": "Type not found"})
        if not new_name:
            return server.web.json_response({"success": False, "error": "New type name is required"})

        for existing_type_file, existing_type_data in _iter_type_items(library):
            if existing_type_file == target_type_file:
                continue
            existing_name = str(existing_type_data.get("name", "")).strip()
            if existing_name and existing_name.lower() == new_name.lower():
                return server.web.json_response({
                    "success": False,
                    "error": f"Type already exists as '{existing_name}'",
                })

        library[CANONICAL_TYPES_KEY][target_type_file]["name"] = new_name
        library = PromptComposerStore.save_type_files(library, [target_type_file])
        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
            "type_file": target_type_file,
            "type_name": new_name,
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in rename-type: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/delete-category")
async def compose_delete_category(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        if not category:
            return server.web.json_response({"success": False, "error": "Category name is required"})

        library = PromptComposerStore.load_canonical_prompts()
        location, error = _locate_category(library, category, type_file=data.get("type_file"))
        if not error:
            type_file, type_data, canonical_category, _category_data = location
            had_type_order = _type_orders_are_dense(library)
            ordered_type_files = _ordered_type_files(library) if had_type_order else []
            deleted_index = ordered_type_files.index(type_file) if had_type_order and type_file in ordered_type_files else -1
            type_data.setdefault("categories", {}).pop(canonical_category, None)
            if not type_data.get("categories"):
                library[CANONICAL_TYPES_KEY].pop(type_file, None)
                if had_type_order and deleted_index >= 0:
                    remaining_type_files = [current_type_file for current_type_file in ordered_type_files if current_type_file != type_file]
                    _assign_type_orders(library, remaining_type_files, deleted_index, len(remaining_type_files) - 1)
            PromptComposerStore.save_prompts(library)

        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in delete-category: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/delete-type")
async def compose_delete_type(request):
    try:
        data = await request.json()
        library = PromptComposerStore.load_canonical_prompts()
        target_type_file = _find_type_case_insensitive(library, data.get("type_file") or data.get("prompt_type"))
        if not target_type_file:
            return server.web.json_response({"success": False, "error": "Type not found"})

        had_type_order = _type_orders_are_dense(library)
        ordered_type_files = _ordered_type_files(library) if had_type_order else []
        deleted_index = ordered_type_files.index(target_type_file) if had_type_order and target_type_file in ordered_type_files else -1
        library.get(CANONICAL_TYPES_KEY, {}).pop(target_type_file, None)
        if had_type_order and deleted_index >= 0:
            remaining_type_files = [current_type_file for current_type_file in ordered_type_files if current_type_file != target_type_file]
            _assign_type_orders(library, remaining_type_files, deleted_index, len(remaining_type_files) - 1)
        PromptComposerStore.save_prompts(library)
        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in delete-type: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/reorder-types")
async def compose_reorder_types(request):
    try:
        data = await request.json()
        library = PromptComposerStore.load_canonical_prompts()
        source_type_file = _find_type_case_insensitive(library, data.get("source_type_file") or data.get("source_prompt_type"))
        target_type_file = _find_type_case_insensitive(library, data.get("target_type_file") or data.get("target_prompt_type"))
        position = str(data.get("position", "before") or "before").strip().lower()
        if position not in {"before", "after"}:
            position = "before"

        if not source_type_file or not target_type_file:
            return server.web.json_response({"success": False, "error": "Both source and target prompt groups are required"})
        if source_type_file == target_type_file:
            return server.web.json_response({
                "success": True,
                "library": library,
                "prompts": _flatten_canonical_library(library),
            })

        ordered_type_files = _ensure_dense_type_orders(library)
        if source_type_file not in ordered_type_files or target_type_file not in ordered_type_files:
            return server.web.json_response({"success": False, "error": "Prompt group order is unavailable"})

        source_index = ordered_type_files.index(source_type_file)
        reordered_type_files = [type_file for type_file in ordered_type_files if type_file != source_type_file]
        target_index = reordered_type_files.index(target_type_file)
        insertion_index = target_index + (1 if position == "after" else 0)
        reordered_type_files.insert(insertion_index, source_type_file)

        if reordered_type_files == ordered_type_files:
            return server.web.json_response({
                "success": True,
                "library": library,
                "prompts": _flatten_canonical_library(library),
            })

        affected_start = min(source_index, insertion_index)
        affected_end = max(source_index, insertion_index)
        _assign_type_orders(library, reordered_type_files, affected_start, affected_end)

        touched_type_files = []
        for type_file in reordered_type_files[affected_start:affected_end + 1]:
            if type_file not in touched_type_files:
                touched_type_files.append(type_file)
        library = PromptComposerStore.save_type_files(library, touched_type_files)
        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in reorder-types: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/import-prompts")
async def compose_import_prompts(request):
    try:
        content_type = str(request.headers.get("Content-Type", "") or "").lower()
        if "multipart/form-data" in content_type:
            post_data, _file_name, imported_library = await _read_uploaded_library_request(request)
            mode = str(post_data.get("mode", "skip_existing") or "skip_existing").strip().lower()
        else:
            data = await request.json()
            imported_library = _normalize_library_input(data.get("data", {}))
            mode = str(data.get("mode", "skip_existing") or "skip_existing").strip().lower()
        if mode not in {"skip_existing", "replace_existing"}:
            mode = "skip_existing"

        library = PromptComposerStore.load_canonical_prompts()
        imported_prompts = 0
        skipped_prompts = 0
        imported_category_settings = 0
        skipped_category_settings = 0
        created_categories = 0

        for imported_type_file, imported_type_data in _iter_type_items(imported_library):
            target_type_file = _find_type_case_insensitive(library, imported_type_file) or imported_type_file
            is_new_type = target_type_file not in library.get(CANONICAL_TYPES_KEY, {})
            target_type_data = _ensure_type(
                library,
                target_type_file,
                name=imported_type_data.get("name"),
                subject_type=imported_type_data.get("subject_type"),
                nsfw=imported_type_data.get("nsfw"),
                prefix=imported_type_data.get("prefix", ""),
                base_prompt=imported_type_data.get("base_prompt", ""),
            )

            if mode == "replace_existing" or is_new_type:
                if imported_type_data.get("prefix"):
                    target_type_data["prefix"] = imported_type_data["prefix"]
                if imported_type_data.get("base_prompt"):
                    target_type_data["base_prompt"] = imported_type_data["base_prompt"]
                if _type_nsfw(imported_type_data):
                    target_type_data["nsfw"] = True

            imported_categories = imported_type_data.get("categories", {}) if isinstance(imported_type_data, dict) else {}
            for imported_category_name, imported_category_data in imported_categories.items():
                existing_category = None
                for category_name in target_type_data.get("categories", {}).keys():
                    if str(category_name).strip().lower() == str(imported_category_name).strip().lower():
                        existing_category = category_name
                        break
                is_new_category = existing_category is None
                if is_new_category:
                    existing_category = imported_category_name
                    target_type_data.setdefault("categories", {})[existing_category] = {"_prompts_": {}}
                    created_categories += 1

                target_category_data = _ensure_type_category(target_type_data, existing_category)
                normalized_imported_category = _normalize_type_category_data(imported_category_name, imported_category_data)
                has_category_settings = bool(
                    str(normalized_imported_category.get("base_prompt") or "").strip()
                    or str(normalized_imported_category.get("prefix") or "").strip()
                    or _category_nsfw(normalized_imported_category)
                )
                if has_category_settings:
                    if mode == "replace_existing" or is_new_category:
                        if normalized_imported_category.get("base_prompt"):
                            target_category_data["base_prompt"] = normalized_imported_category["base_prompt"]
                        if normalized_imported_category.get("prefix"):
                            target_category_data["prefix"] = normalized_imported_category["prefix"]
                        if _category_nsfw(normalized_imported_category):
                            target_category_data["nsfw"] = True
                        imported_category_settings += 1
                    else:
                        skipped_category_settings += 1

                prompt_entries = _ensure_category_prompts_map(target_category_data)
                for prompt_name, imported_entry in _get_category_prompts_map(normalized_imported_category).items():
                    existing_entry, existing_name = _find_prompt_case_insensitive(target_category_data, prompt_name)
                    if existing_entry is not None and mode != "replace_existing":
                        skipped_prompts += 1
                        continue
                    if existing_name and existing_name != prompt_name:
                        prompt_entries.pop(existing_name, None)
                    prompt_entries[prompt_name] = _normalize_prompt_entry(imported_entry)
                    imported_prompts += 1

        PromptComposerStore.save_prompts(library)
        type_count = len(library.get(CANONICAL_TYPES_KEY, {})) if isinstance(library, dict) else 0
        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
            "type_count": type_count,
            "imported_prompts": imported_prompts,
            "skipped_prompts": skipped_prompts,
            "imported_category_settings": imported_category_settings,
            "skipped_category_settings": skipped_category_settings,
            "created_categories": created_categories,
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in import-prompts: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/replace-prompts")
async def compose_replace_prompts(request):
    try:
        content_type = str(request.headers.get("Content-Type", "") or "").lower()
        if "multipart/form-data" in content_type:
            post_data, _file_name, library = await _read_uploaded_library_request(request)
            backup_existing = _coerce_bool(post_data.get("backup_existing", False))
        else:
            data = await request.json()
            library = _normalize_library_input(data.get("data", {}))
            backup_existing = _coerce_bool(data.get("backup_existing", False))
        backup_dir = None
        if backup_existing:
            backup_dir = _backup_composer_storage(PromptComposerStore.get_storage_dir())
        PromptComposerStore.save_prompts(library)
        category_count, prompt_count = _count_prompt_totals(library)
        type_count = len(library.get(CANONICAL_TYPES_KEY, {})) if isinstance(library, dict) else 0
        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
            "type_count": type_count,
            "category_count": category_count,
            "prompt_count": prompt_count,
            "backup_dir": backup_dir,
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in replace-prompts: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/export-prompts-file")
async def compose_export_prompts_file(request):
    try:
        data = await request.json()
        export_path = _normalize_export_path(data.get("path", ""))
        exported_library = _normalize_library_input(data.get("data", {}))

        if not export_path:
            return server.web.json_response({"success": False, "error": "Export path is required"})

        parent_dir = os.path.dirname(export_path)
        if not parent_dir or not os.path.isdir(parent_dir):
            return server.web.json_response({"success": False, "error": "Target folder does not exist"})

        if not atomic_save(export_path, exported_library, "PromptComposerExport"):
            return server.web.json_response({"success": False, "error": "Failed to save export file"}, status=500)

        return server.web.json_response({"success": True, "path": export_path})
    except Exception as exc:
        print(f"[PromptComposerStore] Error in export-prompts-file: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/export-selected-zip")
async def compose_export_selected_zip(request):
    try:
        data = await request.json()
        requested_type_files = data.get("type_files") if isinstance(data, dict) else []
        if not isinstance(requested_type_files, list) or not requested_type_files:
            return server.web.json_response({"success": False, "error": "At least one prompt-group JSON is required"}, status=400)

        library = PromptComposerStore.load_canonical_prompts()
        ordered_selection = []
        seen = set()
        for requested in requested_type_files:
            type_file = _find_type_case_insensitive(library, requested)
            if not type_file or type_file in seen:
                continue
            seen.add(type_file)
            ordered_selection.append(type_file)

        if not ordered_selection:
            return server.web.json_response({"success": False, "error": "No matching prompt-group JSON files were found"}, status=404)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for type_file in ordered_selection:
                type_data = library.get(CANONICAL_TYPES_KEY, {}).get(type_file, {})
                payload = _serialize_type_data(type_file, type_data)
                archive.writestr(type_file, json.dumps(payload, indent=2, ensure_ascii=False))

        zip_bytes = zip_buffer.getvalue()
        filename = f"prompt_composer_jsons_{datetime.now().strftime('%y.%m.%d_%H.%M.%S')}.zip"
        return server.web.Response(
            body=zip_bytes,
            headers={
                "Content-Type": "application/zip",
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )
    except Exception as exc:
        print(f"[PromptComposerStore] Error in export-selected-zip: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/inspect-import-file")
async def compose_inspect_import_file(request):
    try:
        _post_data, file_name, imported_library = await _read_uploaded_library_request(request)
        category_count, prompt_count = _count_prompt_totals(imported_library)
        type_count = len(imported_library.get(CANONICAL_TYPES_KEY, {})) if isinstance(imported_library, dict) else 0
        return server.web.json_response({
            "success": True,
            "library": imported_library,
            "type_count": type_count,
            "category_count": category_count,
            "prompt_count": prompt_count,
            "file_name": file_name,
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in inspect-import-file: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/load-prompts-file")
async def compose_load_prompts_file(request):
    try:
        data = await request.json()
        file_path = _safe_abspath(data.get("path", ""))
        if not file_path:
            return server.web.json_response({"success": False, "error": "Path is required"})
        if not os.path.isfile(file_path):
            return server.web.json_response({"success": False, "error": "JSON file not found"}, status=404)

        with open(file_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            return server.web.json_response({"success": False, "error": "Invalid JSON data format"}, status=400)

        normalized = _normalize_library_input(loaded)
        return server.web.json_response({"success": True, "data": normalized, "path": file_path})
    except Exception as exc:
        print(f"[PromptComposerStore] Error in load-prompts-file: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/save-prompt")
async def compose_save_prompt(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        name = str(data.get("name", "")).strip()
        text = str(data.get("text", "") or "")
        old_category = str(data.get("old_category", "") or category).strip()
        old_name = str(data.get("old_name", "") or name).strip()
        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name are required"})

        library = PromptComposerStore.load_canonical_prompts()
        current_location, current_error = _locate_category(library, category, type_file=data.get("type_file"))
        if current_error:
            inferred_type_file = _normalize_type_file_name(data.get("prompt_type") or category or FALLBACK_TYPE_FILE)
            type_data = _ensure_type(library, inferred_type_file)
            category_data = _ensure_type_category(type_data, category)
            current_type_file = inferred_type_file
            canonical_category = category
        else:
            current_type_file, type_data, canonical_category, category_data = current_location

        prompt_entries = _ensure_category_prompts_map(category_data)
        source_location, _source_error = _locate_category(library, old_category, type_file=data.get("type_file"))
        if source_location:
            source_type_file, source_type_data, source_category_name, source_category_data = source_location
            source_prompt_entries = _ensure_category_prompts_map(source_category_data)
        else:
            source_type_file = current_type_file
            source_type_data = type_data
            source_category_name = canonical_category
            source_category_data = category_data
            source_prompt_entries = prompt_entries

        existing_old_name = next((entry_name for entry_name in source_prompt_entries.keys() if str(entry_name).lower() == old_name.lower()), None)
        existing_prompt = source_prompt_entries.get(existing_old_name, {}) if existing_old_name else {}

        existing_target_name = next((entry_name for entry_name in prompt_entries.keys() if str(entry_name).lower() == name.lower()), None)
        if existing_target_name:
            same_entry = (
                existing_old_name is not None
                and source_type_file == current_type_file
                and source_category_name == canonical_category
                and str(existing_target_name).lower() == str(existing_old_name).lower()
            )
            if not same_entry:
                return server.web.json_response({
                    "success": False,
                    "error": f"A prompt named '{existing_target_name}' already exists in '{canonical_category}'",
                })

        if existing_old_name and (source_type_file != current_type_file or source_category_name != canonical_category or existing_old_name != name):
            existing_prompt = source_prompt_entries.pop(existing_old_name, existing_prompt)

        entry = {"prompt": text}
        thumbnail = data.get("thumbnail")
        if thumbnail is not None:
            entry["thumbnail"] = thumbnail
        elif existing_prompt.get("thumbnail"):
            entry["thumbnail"] = existing_prompt["thumbnail"]

        lora = data.get("lora", None)
        lora_strength = data.get("lora_strength", None)
        lora_image = data.get("lora_image", None)
        lora_image_strength = data.get("lora_image_strength", None)
        lora_video = data.get("lora_video", None)
        lora_video_strength = data.get("lora_video_strength", None)
        refmod = data.get("refmod", None)
        refmod_weight = data.get("refmod_weight", None)

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
        type_data.setdefault("categories", {})[canonical_category] = category_data
        if source_location:
            source_type_data.setdefault("categories", {})[source_category_name] = source_category_data

        library = PromptComposerStore.save_type_files(library, [current_type_file, source_type_file])
        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in save-prompt: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/delete-prompt")
async def compose_delete_prompt(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        name = str(data.get("name", "")).strip()
        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name are required"})

        library = PromptComposerStore.load_canonical_prompts()
        location, error = _locate_category(library, category, type_file=data.get("type_file"))
        if not error:
            type_file, type_data, canonical_category, category_data = location
            prompt_entries = _ensure_category_prompts_map(category_data)
            entry, canonical_name = _find_prompt_case_insensitive(category_data, name)
            if canonical_name:
                prompt_entries.pop(canonical_name, None)
                type_data.setdefault("categories", {})[canonical_category] = category_data
                library = PromptComposerStore.save_type_files(library, [type_file])

        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in delete-prompt: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


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

        library = PromptComposerStore.load_canonical_prompts()
        old_location, old_error = _locate_category(library, old_category, type_file=data.get("type_file"))
        if old_error:
            return server.web.json_response({"success": False, "error": old_error})
        old_type_file, old_type_data, old_category_name, old_category_data = old_location
        old_prompt_entries = _ensure_category_prompts_map(old_category_data)
        existing_old_name = next((entry_name for entry_name in old_prompt_entries.keys() if str(entry_name).lower() == old_name.lower()), None)
        if not existing_old_name:
            return server.web.json_response({"success": False, "error": "Prompt not found"})

        new_location, _new_error = _locate_category(library, new_category, type_file=data.get("new_type_file") or data.get("type_file"))
        if new_location:
            new_type_file, new_type_data, new_category_name, new_category_data = new_location
        else:
            new_type_file = old_type_file
            new_type_data = old_type_data
            new_category_name = new_category
            new_category_data = _ensure_type_category(new_type_data, new_category_name)

        same_category = old_type_file == new_type_file and old_category_name == new_category_name
        new_prompt_entries = _ensure_category_prompts_map(new_category_data)
        existing_target_name = next((entry_name for entry_name in new_prompt_entries.keys() if str(entry_name).lower() == new_name.lower()), None)
        if existing_target_name and not (same_category and str(existing_target_name).lower() == str(existing_old_name).lower()):
            return server.web.json_response({
                "success": False,
                "error": f"A prompt named '{existing_target_name}' already exists in '{new_category_name}'",
            })

        entry = old_prompt_entries.pop(existing_old_name)
        new_prompt_entries[new_name] = entry
        old_type_data.setdefault("categories", {})[old_category_name] = old_category_data
        new_type_data.setdefault("categories", {})[new_category_name] = new_category_data
        library = PromptComposerStore.save_type_files(library, [old_type_file, new_type_file])
        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in rename-prompt: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/toggle-nsfw")
async def compose_toggle_nsfw(request):
    try:
        data = await request.json()
        toggle_type = str(data.get("type", "prompt") or "prompt").strip().lower()
        library = PromptComposerStore.load_canonical_prompts()

        if toggle_type == "type":
            target_type_file = _find_type_case_insensitive(library, data.get("type_file") or data.get("prompt_type"))
            if not target_type_file:
                return server.web.json_response({"success": False, "error": "Type not found"})
            type_data = library[CANONICAL_TYPES_KEY][target_type_file]
            type_data["nsfw"] = not _type_nsfw(type_data)
        elif toggle_type == "category":
            category = str(data.get("category", "")).strip()
            location, error = _locate_category(library, category, type_file=data.get("type_file"))
            if error:
                return server.web.json_response({"success": False, "error": error})
            type_file, type_data, canonical_category, category_data = location
            if _category_nsfw(category_data):
                category_data.pop("nsfw", None)
            else:
                category_data["nsfw"] = True
            type_data.setdefault("categories", {})[canonical_category] = category_data
        else:
            category = str(data.get("category", "")).strip()
            name = str(data.get("name", "")).strip()
            location, error = _locate_category(library, category, type_file=data.get("type_file"))
            if error:
                return server.web.json_response({"success": False, "error": error})
            _type_file, type_data, canonical_category, category_data = location
            prompt_entries = _ensure_category_prompts_map(category_data)
            entry, canonical_name = _find_prompt_case_insensitive(category_data, name)
            if not canonical_name:
                return server.web.json_response({"success": False, "error": "Prompt not found"})
            if isinstance(entry, dict):
                entry["nsfw"] = not _coerce_bool(entry.get("nsfw"))
            prompt_entries[canonical_name] = _normalize_prompt_entry(entry)
            type_data.setdefault("categories", {})[canonical_category] = category_data

        if toggle_type == "type":
            touched_type_files = [target_type_file]
        elif toggle_type == "category":
            touched_type_files = [type_file]
        else:
            touched_type_files = [_type_file]
        library = PromptComposerStore.save_type_files(library, touched_type_files)
        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in toggle-nsfw: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)


@server.PromptServer.instance.routes.post("/prompt-manager/compose/save-thumbnail")
async def compose_save_thumbnail(request):
    try:
        data = await request.json()
        category = str(data.get("category", "")).strip()
        name = str(data.get("name", "")).strip()
        thumbnail = data.get("thumbnail")
        if not category or not name:
            return server.web.json_response({"success": False, "error": "Category and name are required"})

        library = PromptComposerStore.load_canonical_prompts()
        location, error = _locate_category(library, category, type_file=data.get("type_file"))
        if error:
            inferred_type = _normalize_type_file_name(data.get("prompt_type") or category or FALLBACK_TYPE_FILE)
            type_data = _ensure_type(library, inferred_type)
            category_data = _ensure_type_category(type_data, category)
            type_file = inferred_type
            canonical_category = category
        else:
            type_file, type_data, canonical_category, category_data = location

        prompt_entries = _ensure_category_prompts_map(category_data)
        entry, canonical_name = _find_prompt_case_insensitive(category_data, name)
        if not canonical_name:
            canonical_name = name
            entry = {"prompt": ""}
        if not isinstance(entry, dict):
            entry = {"prompt": str(entry)}

        if thumbnail:
            entry["thumbnail"] = thumbnail
        else:
            entry.pop("thumbnail", None)

        prompt_entries[canonical_name] = entry
        type_data.setdefault("categories", {})[canonical_category] = category_data
        library = PromptComposerStore.save_type_files(library, [type_file])
        return server.web.json_response({
            "success": True,
            "library": library,
            "prompts": _flatten_canonical_library(library),
        })
    except Exception as exc:
        print(f"[PromptComposerStore] Error in save-thumbnail: {exc}")
        return server.web.json_response({"success": False, "error": str(exc)}, status=500)