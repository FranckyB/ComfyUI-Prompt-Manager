"""
Prompt Composer - build a prompt from multiple selectable prompt parts.

Each part stores category, selected prompt names (multi-select), and strength.
"""
import importlib
import importlib.util
import json
import math
import os
import random
import server
import sys
import time
from copy import copy
from pathlib import Path

from ..py.prompt_composer_store import (
    PromptComposerStore,
    _find_category_case_insensitive,
    _find_prompt_case_insensitive,
)
from ..py.workflow_data_utils import build_v2_recipe_data_from_prompt, ensure_v2_recipe_data
from ..py.lora_utils import get_lora_relative_path


def _normalize_strength(strength):
    try:
        value = float(strength)
    except (TypeError, ValueError):
        value = 1.0
    if not math.isfinite(value):
        value = 1.0
    return max(0.0, min(5.0, value))


def _format_fragment(text, strength=1.0):
    trimmed = str(text or "").strip()
    if not trimmed:
        return ""
    strength_value = _normalize_strength(strength)
    if strength_value == 1.0:
        return trimmed
    return f"({trimmed}:{strength_value:.15g})"


def _prefix_with_category(category, text):
    trimmed_text = str(text or "").strip()
    if not trimmed_text:
        return ""
    trimmed_category = str(category or "").strip()
    if not trimmed_category:
        return trimmed_text
    if trimmed_category.endswith(":"):
        return f"{trimmed_category} {trimmed_text}"
    return f"{trimmed_category}: {trimmed_text}"

OUTPUT_FORMAT_CHOICES = ["text", "json"]
COMPOSE_POSITION_CHOICES = ["after", "before"]
GENERATION_MODE_CHOICES = ["image", "video"]
RECIPE_SYNC_MODE_CHOICES = ["edit", "sync"]
INPUT_PROMPT_MODE_CHOICES = ["no_prompt", "use_prompt"]
INPUT_LORA_MODE_CHOICES = ["no_lora", "use_lora"]
REFMOD_MAX_WEIGHT = 10.0
SUBJECT_NONE = 0
SUBJECT_MIN = 1
SUBJECT_MAX = 16
_REFMOD_CORE_MODULE = None
_REFMOD_CORE_IMPORT_ERROR = None
_VISUAL_SUFFIXES = ("_visual", "_video")
_AUDIO_SUFFIXES = ("_audio",)
_VISUAL_FILE_SUFFIXES = ("_visual", "_Visual", "_video", "_Video")
_AUDIO_FILE_SUFFIXES = ("_audio", "_Audio")
PROMPT_COMPOSER_RECIPE_KEY = "prompt_composer"


def _resolve_prompt_type(category_data, fallback_category):
    if isinstance(category_data, dict):
        raw_type = category_data.get("_prompt_type_", "")
        prompt_type = str(raw_type or "").strip()
        if prompt_type:
            return prompt_type
    return str(fallback_category or "").strip()


def _resolve_prompt_prefix(category_data, fallback_category):
    if isinstance(category_data, dict):
        raw_prefix = category_data.get("_prompt_prefix_", "")
        prompt_prefix = str(raw_prefix or "").strip()
        if prompt_prefix:
            return prompt_prefix
    return ""


def _json_section_key(category, prompt_type):
    """Use the category prompt type, or fall back to the category name."""
    key = str(prompt_type or "").strip().rstrip(":")
    if key:
        return key
    return str(category or "").strip()


def _text_section_key(category):
    return str(category or "").strip()


def _normalize_prompt_ref(value, fallback_category=""):
    if not isinstance(value, dict):
        return None
    category = str(value.get("category") or fallback_category or "").strip()
    name = str(value.get("name") or value.get("prompt") or "").strip()
    if not name:
        return None
    return {
        "category": category,
        "name": name,
    }


def _derive_prompt_refs(category, prompts):
    fallback_category = str(category or "").strip()
    refs = []
    for name in prompts or []:
        normalized_name = str(name or "").strip()
        if not normalized_name:
            continue
        refs.append({
            "category": fallback_category,
            "name": normalized_name,
        })
    return refs


def _get_part_prompt_refs(part):
    fallback_category = str(part.get("category") or "").strip() if isinstance(part, dict) else ""
    explicit_refs = []
    if isinstance(part, dict) and isinstance(part.get("prompt_refs"), list):
        for item in part.get("prompt_refs") or []:
            normalized = _normalize_prompt_ref(item, fallback_category)
            if normalized:
                explicit_refs.append(normalized)
    if explicit_refs:
        return explicit_refs
    prompts = part.get("prompts") if isinstance(part, dict) else []
    return _derive_prompt_refs(fallback_category, prompts)


def _parse_parts(parts_data):
    try:
        parts = json.loads(parts_data or "[]")
    except Exception:
        parts = []

    if not isinstance(parts, list):
        return []

    normalized = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        category = str(part.get("category") or "").strip()
        prompts = part.get("prompts")
        if isinstance(prompts, list):
            names = [str(name).strip() for name in prompts if str(name).strip()]
        else:
            names = []
        if not names:
            single_name = str(part.get("name") or "").strip()
            if single_name:
                names = [single_name]
        prompt_refs = _get_part_prompt_refs({
            "category": category,
            "prompt_refs": part.get("prompt_refs"),
            "prompts": names,
        })
        if prompt_refs:
            names = [ref["name"] for ref in prompt_refs]
            if not category:
                category = str(prompt_refs[0].get("category") or "").strip()
        strength = _normalize_strength(part.get("strength", 1.0))
        if not names:
            continue
        normalized.append({
            "category": category,
            "prompts": names,
            "prompt_refs": prompt_refs,
            "strength": strength,
            "subject_number": _normalize_subject_number(part.get("subject_number", part.get("subject", SUBJECT_MIN)), default=SUBJECT_MIN),
            "subject_locked": bool(part.get("subject_locked", part.get("subject_manual", False))),
            "muted": bool(part.get("muted", False)),
        })
    return normalized


def _coerce_recipe_data_payload(recipe_data, source="PromptComposer"):
    if isinstance(recipe_data, dict):
        return ensure_v2_recipe_data(recipe_data, source=source)
    if isinstance(recipe_data, str) and recipe_data.strip():
        try:
            parsed = json.loads(recipe_data)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return ensure_v2_recipe_data(parsed, source=source)
    return None


def _extract_prompt_composer_recipe_state(recipe_data):
    if not isinstance(recipe_data, dict):
        return None

    payload = recipe_data.get(PROMPT_COMPOSER_RECIPE_KEY)
    if not isinstance(payload, dict):
        return None

    raw_parts_data = payload.get("parts_data")
    if not isinstance(raw_parts_data, str) or not raw_parts_data.strip():
        raw_parts = payload.get("parts")
        if isinstance(raw_parts, list):
            raw_parts_data = json.dumps(raw_parts, ensure_ascii=False)
        else:
            raw_parts_data = "[]"

    normalized_parts = _parse_parts(raw_parts_data)
    input_payload = payload.get("input_data") if isinstance(payload.get("input_data"), dict) else {}
    input_prompt = str(
        input_payload.get("prompt", payload.get("input_prompt", "")) or ""
    ).strip()
    input_lora_stack = _recipe_lora_entries_from_stack(
        input_payload.get("lora_stack", payload.get("input_lora_stack", []))
    )
    return {
        "parts": normalized_parts,
        "parts_data": json.dumps(normalized_parts, ensure_ascii=False),
        "output_format": str(payload.get("output_format") or "text").strip().lower() or "text",
        "compose_position": str(payload.get("compose_position") or "before").strip().lower() or "before",
        "generation_mode": str(payload.get("generation_mode") or "image").strip().lower() or "image",
        "input_prompt_mode": _normalize_input_prompt_mode(payload.get("input_prompt_mode", payload.get("input_mode", "no_input"))),
        "input_lora_mode": _normalize_input_lora_mode(payload.get("input_lora_mode", payload.get("input_mode", "no_input"))),
        "input_prompt": input_prompt,
        "input_lora_stack": input_lora_stack,
    }


def _serialize_prompt_composer_recipe_state(parts_data, output_format="text", compose_position="before", generation_mode="image", input_prompt_mode="no_prompt", input_lora_mode="no_lora", input_prompt="", input_lora_stack=None, prompt_lora_stack=None):
    normalized_parts = _parse_parts(parts_data)
    normalized_parts_data = json.dumps(normalized_parts, ensure_ascii=False)
    return {
        "version": 1,
        "parts": normalized_parts,
        "parts_data": normalized_parts_data,
        "output_format": str(output_format or "text").strip().lower() or "text",
        "compose_position": str(compose_position or "before").strip().lower() or "before",
        "generation_mode": str(generation_mode or "image").strip().lower() or "image",
        "input_prompt_mode": _normalize_input_prompt_mode(input_prompt_mode),
        "input_lora_mode": _normalize_input_lora_mode(input_lora_mode),
        "input_data": {
            "source": "upstream",
            "prompt": str(input_prompt or ""),
            "lora_stack": _recipe_lora_entries_from_stack(input_lora_stack, source_type="upstream"),
        },
        "prompt_data": {
            "source": "compose",
            "lora_stack": _recipe_lora_entries_from_stack(prompt_lora_stack, source_type="compose"),
        },
    }


def _is_recipe_sync_enabled(recipe_sync_mode):
    return str(recipe_sync_mode or "edit").strip().lower() == "sync"


def _normalize_input_prompt_mode(input_mode):
    value = str(input_mode or "no_prompt").strip().lower()
    if value in {"use_prompt", "prompt", "use_input", "use", "input", "on", "true", "1"}:
        return "use_prompt"
    return "no_prompt"


def _normalize_input_lora_mode(input_mode):
    value = str(input_mode or "no_lora").strip().lower()
    if value in {"use_lora", "lora", "use_input", "use", "input", "on", "true", "1"}:
        return "use_lora"
    return "no_lora"


def _should_use_input_prompt_mode(input_mode):
    return _normalize_input_prompt_mode(input_mode) == "use_prompt"


def _should_use_input_lora_mode(input_mode):
    return _normalize_input_lora_mode(input_mode) == "use_lora"


def _resolve_effective_parts_for_change(parts_data, recipe_sync_mode="edit", recipe_data=None):
    base_recipe_data = _coerce_recipe_data_payload(recipe_data, source="PromptComposer")
    composer_recipe_state = _extract_prompt_composer_recipe_state(base_recipe_data)
    parsed_parts = _parse_parts(parts_data)
    if _is_recipe_sync_enabled(recipe_sync_mode) and composer_recipe_state:
        return composer_recipe_state["parts"]
    return parsed_parts


def _build_prompt_library_signature(parts, prompts_data):
    if not isinstance(prompts_data, dict):
        return ""

    signature_rows = []
    for part in parts or []:
        if not isinstance(part, dict) or bool(part.get("muted", False)):
            continue
        for prompt_ref in _get_part_prompt_refs(part):
            raw_category = prompt_ref.get("category") or part.get("category") or ""
            prompt_name = prompt_ref.get("name") or ""
            resolved_category = _resolve_part_category(prompts_data, raw_category, prompt_name)
            category_data = prompts_data.get(resolved_category, {}) if isinstance(prompts_data.get(resolved_category), dict) else {}
            entry, canonical_name = _find_prompt_case_insensitive(category_data, prompt_name)
            signature_rows.append({
                "category": resolved_category,
                "prompt": canonical_name or str(prompt_name or "").strip(),
                "category_meta": {
                    "prompt_type": category_data.get("_prompt_type_"),
                    "prompt_prefix": category_data.get("_prompt_prefix_"),
                    "subject_type": category_data.get("_subject_type_"),
                    "subject_kind": category_data.get("_subject_kind_"),
                },
                "entry": {
                    "prompt": entry.get("prompt") if isinstance(entry, dict) else None,
                    "lora": entry.get("lora") if isinstance(entry, dict) else None,
                    "lora_strength": entry.get("lora_strength") if isinstance(entry, dict) else None,
                    "lora_image": entry.get("lora_image") if isinstance(entry, dict) else None,
                    "lora_image_strength": entry.get("lora_image_strength") if isinstance(entry, dict) else None,
                    "lora_video": entry.get("lora_video") if isinstance(entry, dict) else None,
                    "lora_video_strength": entry.get("lora_video_strength") if isinstance(entry, dict) else None,
                    "refmod": entry.get("refmod") if isinstance(entry, dict) else None,
                    "refmod_weight": entry.get("refmod_weight") if isinstance(entry, dict) else None,
                },
                "missing": not isinstance(entry, dict),
            })

    if not signature_rows:
        return ""

    return json.dumps(signature_rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _recipe_lora_entries_from_stack(lora_stack, source_type=None):
    entries = []
    for item in _coerce_lora_stack(lora_stack):
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            continue
        path = str(item[0] or "").strip()
        if not path:
            continue
        model_strength = _normalize_scalar(item[1] if len(item) >= 2 else 1.0, default=1.0)
        clip_strength = _normalize_scalar(item[2] if len(item) >= 3 else model_strength, default=model_strength)
        entry = {
            "name": path,
            "path": path,
            "strength": model_strength,
            "model_strength": model_strength,
            "clip_strength": clip_strength,
            "active": True,
            "available": True,
        }
        if source_type:
            entry["source"] = str(source_type)
        entries.append(entry)
    return entries


def _resolve_part_category(prompts_data, raw_category, prompt_name=""):
    if not isinstance(prompts_data, dict):
        return str(raw_category or "").strip()

    exact = _find_category_case_insensitive(prompts_data, raw_category)
    if exact:
        return exact

    target = str(raw_category or "").strip().lower()
    if not target:
        return ""

    prompt_target = str(prompt_name or "").strip()
    matches = []
    for existing_category, category_data in prompts_data.items():
        if existing_category == "__meta__" or not isinstance(category_data, dict):
            continue
        display_name = str(category_data.get("_category_name_") or existing_category).strip().lower()
        if display_name != target:
            continue
        if prompt_target:
            entry, _canonical_name = _find_prompt_case_insensitive(category_data, prompt_target)
            if isinstance(entry, dict):
                return existing_category
        matches.append(existing_category)

    if len(matches) == 1:
        return matches[0]
    return matches[0] if matches else str(raw_category or "").strip()


def _normalize_subject_number(value, default=SUBJECT_MIN):
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        numeric = int(default)
    return max(SUBJECT_NONE, min(SUBJECT_MAX, numeric))


def _resolve_subject_parts(parts):
    resolved = []
    current_subject = SUBJECT_MIN
    for part in parts:
        if not isinstance(part, dict):
            continue
        normalized = dict(part)
        if bool(normalized.get("muted", False)):
            normalized["subject_number"] = _normalize_subject_number(normalized.get("subject_number", SUBJECT_MIN), default=current_subject)
            normalized["subject_locked"] = bool(normalized.get("subject_locked", False))
            normalized["effective_subject_number"] = current_subject
            resolved.append(normalized)
            continue
        subject_number = _normalize_subject_number(normalized.get("subject_number", SUBJECT_MIN), default=current_subject)
        subject_locked = bool(normalized.get("subject_locked", False))
        if subject_locked and subject_number != SUBJECT_NONE:
            current_subject = subject_number
        if subject_locked and subject_number == SUBJECT_NONE:
            effective_subject_number = SUBJECT_NONE
        else:
            effective_subject_number = current_subject if not subject_locked else subject_number
            current_subject = effective_subject_number
        normalized["subject_number"] = subject_number
        normalized["subject_locked"] = subject_locked
        normalized["effective_subject_number"] = effective_subject_number
        resolved.append(normalized)
    return resolved


def _has_multi_part_selection(parts):
    for part in parts:
        if bool(part.get("muted", False)):
            continue
        prompt_refs = _get_part_prompt_refs(part)
        if len(prompt_refs) > 1:
            return True
    return False


def _resolve_run_seed(seed):
    try:
        seed_value = int(seed)
    except Exception:
        seed_value = 0
    if seed_value != 0:
        return seed_value
    return random.SystemRandom().randrange(0, 0xFFFFFFFFFFFFFFFF)


def _normalize_scalar(value, default=1.0, minimum=None, maximum=None):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    if not math.isfinite(numeric):
        numeric = float(default)
    if minimum is not None:
        numeric = max(float(minimum), numeric)
    if maximum is not None:
        numeric = min(float(maximum), numeric)
    return numeric


def _normalize_lora_path(value):
    candidate = str(value or "").strip()
    if not candidate:
        return ""

    for probe in (candidate, os.path.basename(candidate.replace("\\", "/")), os.path.splitext(candidate)[0]):
        rel_path, found = get_lora_relative_path(probe)
        if found and rel_path:
            return str(rel_path).replace("\\", "/")

    return candidate.replace("\\", "/")


def _coerce_lora_stack(raw_stack):
    if raw_stack is None:
        return []

    if isinstance(raw_stack, dict) and "__value__" in raw_stack:
        raw_stack = raw_stack.get("__value__")

    if not isinstance(raw_stack, list):
        return []

    out = []
    for item in raw_stack:
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            path = _normalize_lora_path(item[0])
            if not path:
                continue
            model_strength = _normalize_scalar(item[1] if len(item) >= 2 else 1.0, default=1.0)
            clip_strength = _normalize_scalar(item[2] if len(item) >= 3 else model_strength, default=model_strength)
            out.append((path, model_strength, clip_strength))
            continue

        if isinstance(item, dict):
            path = _normalize_lora_path(item.get("path") or item.get("name") or "")
            if not path:
                continue
            model_strength = _normalize_scalar(item.get("model_strength", item.get("strength", 1.0)), default=1.0)
            clip_strength = _normalize_scalar(item.get("clip_strength", model_strength), default=model_strength)
            out.append((path, model_strength, clip_strength))

    return out


def _lora_asset_key(path):
    normalized = str(path or "").replace("\\", "/").strip().lower()
    leaf = os.path.basename(normalized)
    stem, _ext = os.path.splitext(leaf)
    return stem or leaf


def _merge_lora_stacks(base_stack, additions):
    merged = []
    seen = set()
    for path, model_strength, clip_strength in _coerce_lora_stack(base_stack):
        key = _lora_asset_key(path)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append((path, model_strength, clip_strength))
    for path, model_strength, clip_strength in _coerce_lora_stack(additions):
        key = _lora_asset_key(path)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append((path, model_strength, clip_strength))
    return merged


def _normalize_refmod_name(value):
    normalized = str(value or "").strip().replace("\\", "/")
    if not normalized or normalized.lower() == "(none)":
        return ""
    if normalized.lower().endswith(".safetensors"):
        normalized = normalized[:-len(".safetensors")]
    return normalized.lstrip("/")


def _strip_known_refmod_suffix(path_no_ext):
    lower_name = os.path.basename(path_no_ext).lower()
    for suffix in _VISUAL_SUFFIXES:
        if lower_name.endswith(suffix):
            return (path_no_ext[:-len(suffix)], "visual", suffix)
    for suffix in _AUDIO_SUFFIXES:
        if lower_name.endswith(suffix):
            return (path_no_ext[:-len(suffix)], "audio", suffix)
    return (path_no_ext, None, None)


def _paired_refmod_path(path_no_ext, target_kind):
    base, current_kind, _suffix = _strip_known_refmod_suffix(path_no_ext)
    if current_kind is None:
        return None
    current_path = path_no_ext + ".safetensors"
    suffixes = _VISUAL_FILE_SUFFIXES if target_kind == "visual" else _AUDIO_FILE_SUFFIXES
    for suffix in suffixes:
        candidate = base + suffix + ".safetensors"
        if candidate != current_path and os.path.isfile(candidate):
            return candidate
    return None


def _resolve_refmod_path_no_ext(mod_name):
    normalized = _normalize_refmod_name(mod_name)
    if not normalized:
        return ""
    backend_kind, backend_module = _load_refmod_backend_module()
    if backend_kind == "official":
        return backend_module._find_mod_path(normalized)
    candidate = os.path.join(Path(__file__).resolve().parents[3], "models", "refmods", normalized)
    if os.path.isfile(candidate + ".safetensors"):
        return candidate
    raise FileNotFoundError(f"RefMod '{normalized}' not found in models/refmods")


def _load_python_package(package_dir, package_name):
    package_dir = Path(package_dir)
    init_path = package_dir / "__init__.py"
    if not init_path.is_file():
        raise RuntimeError(f"RefMod support unavailable: {init_path} not found")

    existing = sys.modules.get(package_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"RefMod support unavailable: could not load {init_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(package_name, None)
        raise
    return module


def _load_refmod_backend_module():
    global _REFMOD_CORE_MODULE, _REFMOD_CORE_IMPORT_ERROR
    if _REFMOD_CORE_MODULE is not None:
        return _REFMOD_CORE_MODULE
    if _REFMOD_CORE_IMPORT_ERROR is not None:
        raise RuntimeError(_REFMOD_CORE_IMPORT_ERROR)

    try:
        package_name = "prompt_composer_minimax_h3mod"
        package_dir = Path(__file__).resolve().parents[2] / "ComfyUI-MiniMaxH3Mod"
        _load_python_package(package_dir, package_name)
        module = importlib.import_module(f"{package_name}.nodes")
        _REFMOD_CORE_MODULE = ("official", module)
        return _REFMOD_CORE_MODULE
    except Exception as official_exc:
        try:
            core_path = Path(__file__).resolve().parents[2] / "ComfyUI-H3RefModPicker" / "py" / "refmod_core.py"
            if not core_path.is_file():
                raise RuntimeError(f"RefMod support unavailable: {core_path} not found")
            spec = importlib.util.spec_from_file_location("prompt_composer_refmod_core", core_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"RefMod support unavailable: could not load {core_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _REFMOD_CORE_MODULE = ("legacy", module)
            return _REFMOD_CORE_MODULE
        except Exception as legacy_exc:
            _REFMOD_CORE_IMPORT_ERROR = (
                "RefMod support unavailable: official ComfyUI-MiniMaxH3Mod import failed "
                f"({official_exc}); fallback import failed ({legacy_exc})"
            )
            raise RuntimeError(_REFMOD_CORE_IMPORT_ERROR)


def _refmod_asset_key(mod):
    if isinstance(mod, str):
        normalized = mod.replace("\\", "/").strip().lower()
        leaf = os.path.basename(normalized)
        stem, _ext = os.path.splitext(leaf)
        return stem or leaf
    name = getattr(mod, "name", None)
    if name is None and isinstance(mod, dict):
        name = mod.get("name") or mod.get("path")
    normalized = str(name or "").replace("\\", "/").strip().lower()
    leaf = os.path.basename(normalized)
    stem, _ext = os.path.splitext(leaf)
    return stem or leaf


def _merge_refmod_rows(base_rows, additions):
    merged = list(base_rows) if isinstance(base_rows, (list, tuple)) else []
    seen = {_refmod_asset_key(item[0]) for item in merged if isinstance(item, (list, tuple)) and item}
    for item in additions or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        key = _refmod_asset_key(item[0])
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _override_refmod_row_descriptions(rows, description):
    normalized_description = str(description or "").strip()
    if not normalized_description:
        return list(rows or [])

    overridden = []
    for item in rows or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        mod = item[0]
        strength = item[1]
        mod_copy = copy(mod)
        try:
            mod_copy.description = normalized_description
        except Exception:
            pass
        if len(item) == 2:
            overridden.append((mod_copy, strength))
        else:
            overridden.append((mod_copy, strength, *item[2:]))
    return overridden


def _subject_label(subject_number):
    if int(subject_number) <= 0:
        return "Not Subject"
    return f"Subject {int(subject_number)}"


def _ordinal_word(index):
    words = {
        1: "First",
        2: "Second",
        3: "Third",
        4: "Fourth",
        5: "Fifth",
        6: "Sixth",
        7: "Seventh",
        8: "Eighth",
        9: "Ninth",
        10: "Tenth",
        11: "Eleventh",
        12: "Twelfth",
        13: "Thirteenth",
        14: "Fourteenth",
        15: "Fifteenth",
        16: "Sixteenth",
    }
    try:
        normalized = int(index)
    except (TypeError, ValueError):
        normalized = 0
    return words.get(normalized, f"Subject {normalized}")


def _normalize_subject_kind(value, fallback="other"):
    normalized = str(value or "").strip().lower()
    if normalized == "person":
        return "character"
    if normalized in {"character", "animal", "environment", "other"}:
        return normalized
    return fallback


def _default_subject_kind(subject_type="subject"):
    return "character" if str(subject_type or "").strip().lower() == "new_subject" else "other"


def _first_labeled_text_bucket(sections):
    if not isinstance(sections, dict):
        return None
    for bucket in sections.values():
        if not isinstance(bucket, dict):
            continue
        label = str(bucket.get("label") or "").strip()
        descriptions = [str(value or "").strip() for value in bucket.get("descriptions", []) if str(value or "").strip()]
        if label and descriptions:
            return {
                "label": label,
                "descriptions": descriptions,
            }
    return None


def _image_subject_label(text_sections, subject_kind="other"):
    normalized_kind = _normalize_subject_kind(subject_kind, "other")
    if normalized_kind == "character":
        return "person"
    if normalized_kind == "animal":
        return "animal"
    bucket = _first_labeled_text_bucket(text_sections)
    if bucket:
        return bucket["label"]
    if normalized_kind == "environment":
        return "The environment is"
    return "subject"


def _image_subject_prefix(position, total, subject_label="Subject", subject_kind="other"):
    normalized_label = str(subject_label or "").strip() or "Subject"
    normalized_kind = _normalize_subject_kind(subject_kind, "other")
    if normalized_kind == "environment":
        return normalized_label
    if int(total or 0) <= 1:
        return ""
    return f"{_ordinal_word(position)} {normalized_label} is"


def _image_subject_name(position, total, subject_label="Subject", subject_kind="other"):
    normalized_kind = _normalize_subject_kind(subject_kind, "other")
    if normalized_kind == "character":
        normalized_label = "Person"
    elif normalized_kind == "animal":
        normalized_label = "Animal"
    elif normalized_kind == "environment":
        normalized_label = "Environment"
    else:
        normalized_label = "Subject"
    if normalized_kind == "environment":
        return normalized_label
    if int(total or 0) <= 1:
        return normalized_label
    return f"{_ordinal_word(position)} {normalized_label}"


def _render_image_subject_body(text_sections, subject_label):
    normalized_subject_label = str(subject_label or "").strip()
    fragments = []
    suppressed_subject_label = False
    for bucket in (text_sections or {}).values():
        descriptions = [str(value or "").strip() for value in bucket.get("descriptions", []) if str(value or "").strip()]
        if not descriptions:
            continue
        joined = _join_text_descriptions(descriptions)
        label = str(bucket.get("label") or "").strip()
        fragment_label = label
        if (
            normalized_subject_label
            and not suppressed_subject_label
            and label.lower() == normalized_subject_label.lower()
        ):
            fragment_label = ""
            suppressed_subject_label = True
        fragment_text = f"{fragment_label} {joined}".strip() if fragment_label else joined
        if fragment_text:
            fragments.append({
                "text": fragment_text,
                "has_label": bool(fragment_label),
            })
    if not fragments:
        return ""
    if len(fragments) == 1:
        return fragments[0]["text"]

    rendered = fragments[0]["text"]
    for index, fragment in enumerate(fragments[1:], start=1):
        separator = ", "
        rendered = f"{rendered}{separator}{fragment['text']}"
    return rendered


def _render_image_subject_groups(subject_groups):
    rendered_groups = []
    typed_groups = _prepare_image_subject_groups(subject_groups)

    for group in typed_groups:
        prefix = _image_subject_prefix(
            group["position"],
            group["total"],
            group.get("subject_label", "Subject"),
            group.get("subject_kind", "other"),
        )
        rendered_groups.append(
            f"{prefix} {group['body']}".strip() if prefix else group["body"]
        )

    return "\n\n".join(fragment for fragment in rendered_groups if fragment)


def _prepare_image_subject_groups(subject_groups):
    prepared_groups = []
    for group in subject_groups:
        text_sections = group.get("text_sections", {})
        subject_kind = _normalize_subject_kind(
            group.get("subject_kind"),
            _default_subject_kind(group.get("subject_type")),
        )
        subject_label = _image_subject_label(text_sections, subject_kind)
        body = _render_image_subject_body(text_sections, subject_label)
        if not body:
            continue
        prepared_groups.append({
            "body": body,
            "text_sections": text_sections,
            "sections": group.get("sections", {}),
            "subject_kind": subject_kind,
            "subject_type": group.get("subject_type"),
            "subject_label": subject_label,
        })

    kind_totals = {}
    for group in prepared_groups:
        subject_kind = group["subject_kind"]
        if subject_kind == "environment":
            continue
        kind_totals[subject_kind] = kind_totals.get(subject_kind, 0) + 1

    kind_positions = {}
    typed_groups = []
    for group in prepared_groups:
        subject_kind = group["subject_kind"]
        if subject_kind == "environment":
            position = 1
            total = 1
        else:
            next_position = kind_positions.get(subject_kind, 0) + 1
            kind_positions[subject_kind] = next_position
            position = next_position
            total = kind_totals.get(subject_kind, 1)
        typed_groups.append({
            **group,
            "position": position,
            "total": total,
        })

    return typed_groups


def _get_subject_group(subject_groups, subject_number):
    for group in subject_groups:
        if group["number"] == subject_number:
            return group
    group = {
        "number": subject_number,
        "subject_type": "subject",
        "subject_kind": "other",
        "text_sections": {},
        "sections": {},
    }
    subject_groups.append(group)
    return group


def _append_text_section(sections, section_key, section_label, text):
    key = str(section_key or "").strip()
    fragment_text = str(text or "").strip()
    if not fragment_text:
        return
    bucket = sections.get(key)
    if bucket is None:
        bucket = {
            "label": str(section_label or "").strip(),
            "descriptions": [],
        }
        sections[key] = bucket
    elif not bucket.get("label") and str(section_label or "").strip():
        bucket["label"] = str(section_label or "").strip()
    bucket["descriptions"].append(fragment_text)


def _join_text_descriptions(descriptions):
    items = [str(value or "").strip() for value in (descriptions or []) if str(value or "").strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _render_text_sections(sections, break_on_labeled_sections=False):
    fragments = []
    for bucket in sections.values():
        descriptions = [str(value or "").strip() for value in bucket.get("descriptions", []) if str(value or "").strip()]
        if not descriptions:
            continue
        joined = _join_text_descriptions(descriptions)
        label = str(bucket.get("label") or "").strip()
        fragment_text = f"{label} {joined}".strip() if label else joined
        if fragment_text:
            fragments.append({
                "text": fragment_text,
                "has_label": bool(label),
            })
    if not fragments:
        return ""
    if len(fragments) == 1:
        return fragments[0]["text"]

    rendered = fragments[0]["text"]
    for index, fragment in enumerate(fragments[1:], start=1):
        previous = fragments[index - 1]
        separator = "\n\n" if break_on_labeled_sections and (fragment["has_label"] or previous["has_label"]) else ", "
        rendered = f"{rendered}{separator}{fragment['text']}"
    return rendered


def _append_json_section(sections, section_key, text):
    key = str(section_key or "").strip()
    description = str(text or "").strip()
    if not key or not description:
        return
    sections.setdefault(key, []).append(description)


def _render_json_section_value(values):
    descriptions = [str(value or "").strip() for value in (values or []) if str(value or "").strip()]
    if not descriptions:
        return ""
    return ", ".join(descriptions)


def _format_json_description(text, strength, use_strength=True):
    if not use_strength:
        return str(text or "").strip()
    return _format_fragment(text, strength)


def _scale_prompt_asset_weight(base_weight, part_strength, minimum=0.0, maximum=None):
    normalized_base = _normalize_scalar(base_weight, default=1.0, minimum=minimum, maximum=maximum)
    normalized_part = _normalize_strength(part_strength)
    scaled = normalized_base * normalized_part
    if minimum is not None:
        scaled = max(float(minimum), scaled)
    if maximum is not None:
        scaled = min(float(maximum), scaled)
    return scaled


def _load_prompt_refmods(mod_name, weight):
    normalized_name = _normalize_refmod_name(mod_name)
    if not normalized_name:
        return []

    clipped_weight = _normalize_scalar(weight, default=1.0, minimum=0.0, maximum=REFMOD_MAX_WEIGHT)
    if clipped_weight <= 0.0:
        return []

    backend_kind, backend_module = _load_refmod_backend_module()
    path_no_ext = _resolve_refmod_path_no_ext(normalized_name)

    if backend_kind == "official":
        meta = backend_module.read_refmod_meta(path_no_ext)
        if isinstance(meta, dict) and meta.get("kind") == "bundle":
            return list(backend_module.load_bundle(path_no_ext, "All", clipped_weight, clipped_weight))

        loaded = [(backend_module._load_mod(normalized_name), clipped_weight)]
        if not any(getattr(mod, "kind", None) == "audio" for mod, _strength in loaded):
            paired_audio = _paired_refmod_path(path_no_ext, "audio")
            if paired_audio:
                loaded.append((backend_module.H3RefMod.load(paired_audio[:-len(".safetensors")], device="cpu"), clipped_weight))
        return loaded

    loaded = list(backend_module.load_refmods_from_file(path_no_ext, device="cpu"))
    if not any(getattr(mod, "kind", None) == "audio" for mod in loaded):
        paired_audio = _paired_refmod_path(path_no_ext, "audio")
        if paired_audio:
            loaded.extend(backend_module.load_refmods_from_file(paired_audio[:-len(".safetensors")], device="cpu"))
    return [(mod, clipped_weight) for mod in loaded]


class PromptComposer:
    """Compose prompt fragments from multiple parts in one node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parts_data": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "dynamicPrompts": False,
                    "tooltip": "Internal: JSON list of prompt composer parts",
                }),
                "output_format": (OUTPUT_FORMAT_CHOICES, {
                    "default": "text",
                    "tooltip": "Choose whether the node outputs plain text or structured JSON.",
                }),
                "compose_position": (COMPOSE_POSITION_CHOICES, {
                    "default": "before",
                    "tooltip": "Place composed prompt fragments before or after the incoming prompt.",
                }),
                "generation_mode": (GENERATION_MODE_CHOICES, {
                    "default": "image",
                    "tooltip": "Choose whether Prompt Composer emits Image or Video LoRAs for this run.",
                }),
                "recipe_sync_mode": (RECIPE_SYNC_MODE_CHOICES, {
                    "default": "edit",
                    "tooltip": "Choose whether Prompt Composer keeps local edits or follows connected compose_data on execute.",
                }),
                "input_prompt_mode": (INPUT_PROMPT_MODE_CHOICES, {
                    "default": "no_prompt",
                    "tooltip": "When enabled during compose_data sync, use the saved prompt input from compose_data instead of the live prompt input.",
                }),
                "input_lora_mode": (INPUT_LORA_MODE_CHOICES, {
                    "default": "no_lora",
                    "tooltip": "When enabled during compose_data sync, include the saved extra LoRA input data from compose_data in addition to the composed LoRAs.",
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Optional incoming prompt. Composed parts can be placed before or after it.",
                }),
                "compose_data": ("RECIPE_DATA,COMPOSE_DATA", {
                    "forceInput": True,
                    "tooltip": "Optional saved Prompt Composer payload. Connect recipe or compose data to restore Prompt Composer parts and reuse them.",
                }),
                "lora_stack": ("LORA_STACK", {
                    "forceInput": True,
                    "tooltip": "Optional incoming LoRA stack. Prompt Composer either uses this live input or the saved compose_data LoRA input, then appends per-prompt LoRAs.",
                }),
                "mods": ("H3_REF_MODS", {
                    "forceInput": True,
                    "tooltip": "Optional incoming RefMod bundle. Prompt Composer appends per-prompt RefMods to it.",
                }),

            },
            "hidden": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                }),
                "unique_id": "UNIQUE_ID",
            },
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Compose multiple prompt fragments with per-part strength in one node."
    RETURN_TYPES = ("STRING", "COMPOSE_DATA", "LORA_STACK", "H3_REF_MODS")
    RETURN_NAMES = ("Prompt", "compose_data", "lora_stack", "mods")
    FUNCTION = "compose"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, parts_data="[]", seed=0, prompt="", output_format="text", compose_position="before", generation_mode="image", recipe_sync_mode="edit", input_prompt_mode="no_prompt", input_lora_mode="no_lora", compose_data=None, lora_stack=None, mods=None, **kwargs):
        recipe_data = compose_data if compose_data is not None else kwargs.get("recipe_data")
        parts = _resolve_effective_parts_for_change(parts_data, recipe_sync_mode=recipe_sync_mode, recipe_data=recipe_data)
        prompts_data = PromptComposerStore.load_prompts()
        prompt_library_signature = _build_prompt_library_signature(parts, prompts_data)
        if _has_multi_part_selection(parts):
            dynamic_seed = time.time_ns()
            return (parts_data, dynamic_seed, prompt, output_format, compose_position, generation_mode, recipe_sync_mode, _normalize_input_prompt_mode(input_prompt_mode), _normalize_input_lora_mode(input_lora_mode), str(recipe_data) if recipe_data else None, prompt_library_signature, lora_stack, mods)
        return (parts_data, seed, prompt, output_format, compose_position, generation_mode, recipe_sync_mode, _normalize_input_prompt_mode(input_prompt_mode), _normalize_input_lora_mode(input_lora_mode), str(recipe_data) if recipe_data else None, prompt_library_signature, lora_stack, mods)

    def compose(self, parts_data="[]", seed=0, prompt="", output_format="text", compose_position="before", generation_mode="image", recipe_sync_mode="edit", input_prompt_mode="no_prompt", input_lora_mode="no_lora", compose_data=None, lora_stack=None, mods=None, unique_id=None, **kwargs):
        recipe_data = compose_data if compose_data is not None else kwargs.get("recipe_data")
        prompts_data = PromptComposerStore.load_prompts()
        base_recipe_data = _coerce_recipe_data_payload(recipe_data, source="PromptComposer")
        composer_recipe_state = _extract_prompt_composer_recipe_state(base_recipe_data)
        effective_parts_data = parts_data
        parsed_parts = _parse_parts(parts_data)
        effective_output_format = str(output_format or "text").strip().lower() or "text"
        effective_compose_position = str(compose_position or "before").strip().lower() or "before"
        effective_generation_mode = str(generation_mode or "image").strip().lower() or "image"
        effective_input_prompt_mode = _normalize_input_prompt_mode(input_prompt_mode)
        effective_input_lora_mode = _normalize_input_lora_mode(input_lora_mode)
        if _is_recipe_sync_enabled(recipe_sync_mode) and composer_recipe_state:
            effective_parts_data = composer_recipe_state["parts_data"]
            parsed_parts = composer_recipe_state["parts"]
        parts = _resolve_subject_parts(parsed_parts)
        selected_generation_mode = effective_generation_mode

        live_input_prompt = str(prompt or "").strip() if isinstance(prompt, str) else ""
        live_input_lora_stack = list(_coerce_lora_stack(lora_stack))
        saved_input_prompt = composer_recipe_state.get("input_prompt", "") if composer_recipe_state else ""
        saved_input_lora_stack = list(_coerce_lora_stack(composer_recipe_state.get("input_lora_stack", []))) if composer_recipe_state else []
        stored_input_prompt = live_input_prompt if live_input_prompt else saved_input_prompt
        stored_input_lora_stack = _merge_lora_stacks([], live_input_lora_stack if live_input_lora_stack else saved_input_lora_stack)
        effective_input_prompt = saved_input_prompt if _should_use_input_prompt_mode(effective_input_prompt_mode) and saved_input_prompt else live_input_prompt
        effective_input_lora_stack = saved_input_lora_stack if _should_use_input_lora_mode(effective_input_lora_mode) else live_input_lora_stack

        run_seed = _resolve_run_seed(seed)
        rng = random.Random(run_seed)

        subject_groups = []
        non_subject_text_sections = {}
        non_subject_sections = {}
        prompt_lora_stack = []
        prompt_mods = []
        for part in parts:
            if bool(part.get("muted", False)):
                continue
            prompt_refs = _get_part_prompt_refs(part)
            if not prompt_refs:
                continue

            chosen_ref = prompt_refs[0] if len(prompt_refs) == 1 else rng.choice(prompt_refs)
            raw_category = chosen_ref.get("category") or part.get("category") or ""
            category = _resolve_part_category(prompts_data, raw_category, chosen_ref.get("name") or "")
            category_data = prompts_data.get(category, {})
            chosen_name = chosen_ref.get("name") or ""
            entry, canonical_name = _find_prompt_case_insensitive(category_data, chosen_name)
            if not isinstance(entry, dict):
                continue

            text = entry.get("prompt", "") or ""
            prompt_type = _resolve_prompt_type(category_data, category)
            prompt_prefix = _resolve_prompt_prefix(category_data, category)
            text_section_key = _text_section_key(category)
            json_section_key = _json_section_key(category, prompt_type)
            use_strength = selected_generation_mode != "video"
            formatted_plain = _format_fragment(text, part.get("strength", 1.0)) if use_strength else str(text or "").strip()
            formatted_json = _format_json_description(text, part.get("strength", 1.0), use_strength=use_strength)
            subject_number = part.get("effective_subject_number", SUBJECT_MIN)
            subject_type = str(category_data.get("_subject_type_") or "subject").strip().lower() or "subject"
            subject_kind = _normalize_subject_kind(
                category_data.get("_subject_kind_"),
                _default_subject_kind(subject_type),
            )
            if subject_number == SUBJECT_NONE:
                if formatted_plain:
                    _append_text_section(
                        non_subject_text_sections,
                        text_section_key,
                        prompt_prefix,
                        formatted_plain,
                    )
                if formatted_json:
                    _append_json_section(
                        non_subject_sections,
                        json_section_key,
                        formatted_json,
                    )
            else:
                subject_group = _get_subject_group(subject_groups, subject_number)
                if subject_type == "new_subject" and not subject_group.get("subject_kind_locked"):
                    subject_group["subject_type"] = subject_type
                    subject_group["subject_kind"] = subject_kind
                    subject_group["subject_kind_locked"] = True
                if formatted_plain:
                    _append_text_section(
                        subject_group["text_sections"],
                        text_section_key,
                        prompt_prefix,
                        formatted_plain,
                    )
                key = json_section_key
                if key:
                    _append_json_section(subject_group["sections"], key, formatted_json)

            part_strength = part.get("strength", 1.0)
            if selected_generation_mode == "video":
                lora_name = _normalize_lora_path(entry.get("lora_video") or "")
                lora_strength = _scale_prompt_asset_weight(entry.get("lora_video_strength", 1.0), part_strength)
            else:
                lora_name = _normalize_lora_path(entry.get("lora_image") or entry.get("lora") or "")
                lora_strength = _scale_prompt_asset_weight(entry.get("lora_image_strength", entry.get("lora_strength", 1.0)), part_strength)
            if lora_name:
                prompt_lora_stack = _merge_lora_stacks(
                    prompt_lora_stack,
                    [(lora_name, lora_strength, lora_strength)],
                )

            refmod_name = _normalize_refmod_name(entry.get("refmod") or "")
            if refmod_name:
                refmod_weight = _scale_prompt_asset_weight(
                    entry.get("refmod_weight", 1.0),
                    part_strength,
                    minimum=0.0,
                    maximum=REFMOD_MAX_WEIGHT,
                )
                try:
                    loaded_prompt_mods = _load_prompt_refmods(refmod_name, refmod_weight)
                    loaded_prompt_mods = _override_refmod_row_descriptions(loaded_prompt_mods, text)
                    prompt_mods = _merge_refmod_rows(prompt_mods, loaded_prompt_mods)
                except Exception as exc:
                    print(f"[PromptComposer] Skipping RefMod '{refmod_name}': {exc}")

        base = effective_input_prompt
        position = effective_compose_position
        format_name = effective_output_format

        if format_name == "json":
            structured = {}
            if position != "before" and base:
                structured["scene"] = base
            structured["subjects"] = []
            for group in _prepare_image_subject_groups(subject_groups) if selected_generation_mode != "video" else subject_groups:
                if selected_generation_mode == "video":
                    subject_entry = {"name": _subject_label(group["number"])}
                    section_values = group["sections"]
                else:
                    subject_entry = {
                        "name": _image_subject_name(
                            group["position"],
                            group["total"],
                            group.get("subject_label", "Subject"),
                            group.get("subject_kind", "other"),
                        )
                    }
                    section_values = group["sections"]
                for key, values in section_values.items():
                    rendered_value = _render_json_section_value(values)
                    if rendered_value:
                        subject_entry[key] = rendered_value
                structured["subjects"].append(subject_entry)
            if non_subject_sections:
                note_entries = []
                for values in non_subject_sections.values():
                    rendered_value = _render_json_section_value(values)
                    if rendered_value:
                        note_entries.append(rendered_value)
                if note_entries:
                    structured["notes"] = ", ".join(note_entries)
            if position == "before" and base:
                structured["scene"] = base
            json_output = json.dumps(structured, indent=2, ensure_ascii=False) if structured else ""
            final_output = json_output
        else:
            if selected_generation_mode == "video":
                subject_blocks = []
                for group in subject_groups:
                    body = _render_text_sections(group.get("text_sections", {}))
                    if not body:
                        continue
                    subject_blocks.append(f"<{_subject_label(group['number'])}>\n{body}")
                sections = []
                if subject_blocks:
                    sections.append("subject_definitions:\n" + "\n\n".join(subject_blocks))
                non_subject_text = _render_text_sections(non_subject_text_sections, break_on_labeled_sections=True)
                if non_subject_text:
                    sections.append(non_subject_text)
                fragments_text = "\n\n".join(section for section in sections if section)
            else:
                sections = []
                subject_text = _render_image_subject_groups(subject_groups)
                if subject_text:
                    sections.append(subject_text)
                non_subject_text = _render_text_sections(non_subject_text_sections, break_on_labeled_sections=True)
                if non_subject_text:
                    sections.append(non_subject_text)
                fragments_text = "\n\n".join(section for section in sections if section)

            if base and fragments_text:
                if position == "before":
                    text_output = f"{fragments_text}\n{base}"
                else:
                    text_output = f"{base}\n{fragments_text}"
            elif base:
                text_output = base
            else:
                text_output = fragments_text
            final_output = text_output
        merged_lora_stack = _merge_lora_stacks(effective_input_lora_stack, prompt_lora_stack)
        merged_mods = _merge_refmod_rows(mods, prompt_mods)

        input_lora_entries = _recipe_lora_entries_from_stack(effective_input_lora_stack, source_type="upstream")
        prompt_lora_entries = _recipe_lora_entries_from_stack(prompt_lora_stack, source_type="compose")
        input_lora_keys = {
            _lora_asset_key(item.get("path") or item.get("name"))
            for item in input_lora_entries
            if item.get("path") or item.get("name")
        }
        merged_lora_entries = list(input_lora_entries)
        for item in prompt_lora_entries:
            key = _lora_asset_key(item.get("path") or item.get("name"))
            if not key or key in input_lora_keys:
                continue
            input_lora_keys.add(key)
            merged_lora_entries.append(item)

        out_recipe_data = build_v2_recipe_data_from_prompt(
            prompt_text=final_output,
            negative_prompt="",
            loras_a=merged_lora_entries,
            source="PromptComposer",
            base_recipe_data=base_recipe_data,
        )
        out_recipe_data[PROMPT_COMPOSER_RECIPE_KEY] = _serialize_prompt_composer_recipe_state(
            effective_parts_data,
            output_format=effective_output_format,
            compose_position=effective_compose_position,
            generation_mode=effective_generation_mode,
            input_prompt_mode=effective_input_prompt_mode,
            input_lora_mode=effective_input_lora_mode,
            input_prompt=stored_input_prompt,
            input_lora_stack=stored_input_lora_stack,
            prompt_lora_stack=prompt_lora_stack,
        )

        if unique_id is not None:
            server.PromptServer.instance.send_sync("prompt-composer-update", {
                "node_id": str(unique_id),
                "prompt_composer": out_recipe_data.get(PROMPT_COMPOSER_RECIPE_KEY),
            })

        return (final_output, out_recipe_data, merged_lora_stack, merged_mods)
