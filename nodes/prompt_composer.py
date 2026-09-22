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
import sys
import time
from copy import copy
from pathlib import Path

from ..py.prompt_composer_store import (
    PromptComposerStore,
    _find_category_case_insensitive,
    _find_prompt_case_insensitive,
)
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


PROMPT_TYPE_CHOICES = [
    "scene",
    "character",
    "animal",
    "accessory",
    "ambience",
    "attire",
    "background",
    "style",
    "lighting",
    "composition",
    "camera",
    "motion",
    "soundscape",
    "dialogue",
]

OUTPUT_FORMAT_CHOICES = ["text", "json"]
COMPOSE_POSITION_CHOICES = ["after", "before"]
GENERATION_MODE_CHOICES = ["image", "video"]
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


def _resolve_prompt_type(category_data, fallback_category):
    if isinstance(category_data, dict):
        raw_type = category_data.get("_prompt_type_", "")
        prompt_type = str(raw_type or "").strip()
        if prompt_type:
            return prompt_type
    return str(fallback_category or "").strip()


def _resolve_prompt_prefix(category_data, fallback_category):
    def _normalize_prefix(value):
        prefix = str(value or "").strip()
        if not prefix:
            return ""
        return prefix if prefix.endswith(":") else f"{prefix}:"

    if isinstance(category_data, dict):
        raw_type = category_data.get("_prompt_type_", "")
        prompt_type = str(raw_type or "").strip()
        if prompt_type:
            return _normalize_prefix(prompt_type)
    return _normalize_prefix(fallback_category)


def _json_section_key(category, prompt_prefix):
    """Use the explicit prompt prefix or category name as the JSON section key."""
    key = str(prompt_prefix or "").strip().rstrip(":")
    if key:
        return key
    return str(category or "").strip()


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
        strength = _normalize_strength(part.get("strength", 1.0))
        if not names:
            continue
        normalized.append({
            "category": category,
            "prompts": names,
            "strength": strength,
            "subject_number": _normalize_subject_number(part.get("subject_number", part.get("subject", SUBJECT_MIN)), default=SUBJECT_MIN),
            "subject_locked": bool(part.get("subject_locked", part.get("subject_manual", False))),
            "muted": bool(part.get("muted", False)),
        })
    return normalized


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
        names = part.get("prompts") or []
        if len(names) > 1:
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
    merged = list(_coerce_lora_stack(base_stack))
    seen = {_lora_asset_key(item[0]) for item in merged if item and item[0]}
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


def _get_subject_group(subject_groups, subject_number):
    for group in subject_groups:
        if group["number"] == subject_number:
            return group
    group = {
        "number": subject_number,
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
    bucket["descriptions"].append(fragment_text)


def _render_text_sections(sections):
    fragments = []
    for bucket in sections.values():
        descriptions = [str(value or "").strip() for value in bucket.get("descriptions", []) if str(value or "").strip()]
        if not descriptions:
            continue
        joined = ", ".join(descriptions)
        fragments.append(joined)
    return ", ".join(fragment for fragment in fragments if fragment)


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
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Optional incoming prompt. Composed parts can be placed before or after it.",
                }),
                "lora_stack": ("LORA_STACK", {
                    "forceInput": True,
                    "tooltip": "Optional incoming LoRA stack. Prompt Composer appends per-prompt LoRAs to it.",
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
            },
        }

    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Compose multiple prompt fragments with per-part strength in one node."
    RETURN_TYPES = ("STRING", "LORA_STACK", "H3_REF_MODS")
    RETURN_NAMES = ("Prompt", "lora_stack", "mods")
    FUNCTION = "compose"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, parts_data="[]", seed=0, prompt="", output_format="text", compose_position="before", generation_mode="image", lora_stack=None, mods=None, **kwargs):
        parts = _parse_parts(parts_data)
        if _has_multi_part_selection(parts):
            dynamic_seed = time.time_ns()
            return (parts_data, dynamic_seed, prompt, output_format, compose_position, generation_mode, lora_stack, mods)
        return (parts_data, seed, prompt, output_format, compose_position, generation_mode, lora_stack, mods)

    def compose(self, parts_data="[]", seed=0, prompt="", output_format="text", compose_position="before", generation_mode="image", lora_stack=None, mods=None):
        prompts_data = PromptComposerStore.load_prompts()
        parts = _resolve_subject_parts(_parse_parts(parts_data))
        selected_generation_mode = str(generation_mode or "image").strip().lower()

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
            raw_category = part.get("category") or ""
            category = _find_category_case_insensitive(prompts_data, raw_category) or raw_category
            category_data = prompts_data.get(category, {})

            names = part.get("prompts") or []
            if not names:
                continue

            chosen_name = names[0] if len(names) == 1 else rng.choice(names)
            entry, canonical_name = _find_prompt_case_insensitive(category_data, chosen_name)
            if not isinstance(entry, dict):
                continue

            text = entry.get("prompt", "") or ""
            prompt_prefix = _resolve_prompt_prefix(category_data, category)
            use_strength = selected_generation_mode != "video"
            formatted_plain = _format_fragment(text, part.get("strength", 1.0)) if use_strength else str(text or "").strip()
            formatted_json = _format_json_description(text, part.get("strength", 1.0), use_strength=use_strength)
            subject_number = part.get("effective_subject_number", SUBJECT_MIN)
            if subject_number == SUBJECT_NONE:
                if formatted_plain:
                    _append_text_section(
                        non_subject_text_sections,
                        _json_section_key(category, prompt_prefix),
                        prompt_prefix,
                        formatted_plain,
                    )
                if formatted_json:
                    _append_json_section(
                        non_subject_sections,
                        _json_section_key(category, prompt_prefix),
                        formatted_json,
                    )
            else:
                subject_group = _get_subject_group(subject_groups, subject_number)
                if formatted_plain:
                    _append_text_section(
                        subject_group["text_sections"],
                        _json_section_key(category, prompt_prefix),
                        prompt_prefix,
                        formatted_plain,
                    )
                key = _json_section_key(category, prompt_prefix)
                if key:
                    _append_json_section(subject_group["sections"], key, formatted_json)

            if selected_generation_mode == "video":
                lora_name = _normalize_lora_path(entry.get("lora_video") or "")
                lora_strength = _normalize_scalar(entry.get("lora_video_strength", 1.0), default=1.0)
            else:
                lora_name = _normalize_lora_path(entry.get("lora_image") or entry.get("lora") or "")
                lora_strength = _normalize_scalar(entry.get("lora_image_strength", entry.get("lora_strength", 1.0)), default=1.0)
            if lora_name:
                prompt_lora_stack = _merge_lora_stacks(
                    prompt_lora_stack,
                    [(lora_name, lora_strength, lora_strength)],
                )

            refmod_name = _normalize_refmod_name(entry.get("refmod") or "")
            if refmod_name:
                refmod_weight = _normalize_scalar(entry.get("refmod_weight", 1.0), default=1.0, minimum=0.0, maximum=REFMOD_MAX_WEIGHT)
                try:
                    loaded_prompt_mods = _load_prompt_refmods(refmod_name, refmod_weight)
                    loaded_prompt_mods = _override_refmod_row_descriptions(loaded_prompt_mods, text)
                    prompt_mods = _merge_refmod_rows(prompt_mods, loaded_prompt_mods)
                except Exception as exc:
                    print(f"[PromptComposer] Skipping RefMod '{refmod_name}': {exc}")

        base = str(prompt or "").strip() if isinstance(prompt, str) else ""
        position = str(compose_position or "after").strip().lower()
        format_name = str(output_format or "text").strip().lower()

        if format_name == "json":
            structured = {}
            if position != "before" and base:
                structured["scene"] = base
            structured["subjects"] = []
            for group in subject_groups:
                subject_entry = {"name": _subject_label(group["number"])}
                for key, values in group["sections"].items():
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
                non_subject_text = _render_text_sections(non_subject_text_sections)
                if non_subject_text:
                    sections.append(non_subject_text)
                fragments_text = "\n\n".join(section for section in sections if section)
            else:
                sections = []
                for group in subject_groups:
                    body = _render_text_sections(group.get("text_sections", {}))
                    if body:
                        sections.append(body)
                non_subject_text = _render_text_sections(non_subject_text_sections)
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
        merged_lora_stack = _merge_lora_stacks(lora_stack, prompt_lora_stack)
        merged_mods = _merge_refmod_rows(mods, prompt_mods)

        return (final_output, merged_lora_stack, merged_mods)
