"""
Shared LoRA resolution utilities for ComfyUI-Prompt-Manager.

Provides fuzzy matching logic for finding LoRA files on disk, handling
renamed LoRAs, WAN-related tokens, and partial name matches.

Used by: PromptManagerAdvanced, PromptExtractor, WorkflowBuilder, WorkflowRenderer.
"""
import os
import re
import folder_paths


# Known LoRA file extensions
LORA_EXTENSIONS = ['.safetensors', '.ckpt', '.pt', '.bin', '.pth']

# Tokens to remove for fuzzy matching (case-insensitive)
# Order matters: remove longer tokens first to avoid partial matches
WAN_TOKENS = ['wan_2_2', 'wan22', 'wan2.2', '20epoc', 'a14b', '14b', 'i2v', 't2v']


def get_available_loras():
    """Get all available LoRAs from ComfyUI's folder system."""
    return folder_paths.get_filename_list("loras")


def normalize_path_separators(path):
    """Normalize path separators based on OS - for basename extraction only."""
    if os.name == 'nt':  # Windows
        return path.replace('/', '\\')
    else:  # Linux/Mac
        return path.replace('\\', '/')


def strip_lora_extension(name):
    """Remove only known LoRA file extensions, not arbitrary dots in names."""
    name_lower = name.lower()
    for ext in LORA_EXTENSIONS:
        if name_lower.endswith(ext):
            return name[:-len(ext)]
    return name


def _normalize_name_for_fuzzy(name):
    """
    Remove WAN tokens from name, treating underscores and hyphens as separators.
    Also removes content in parentheses (e.g., "MyLora (1)" becomes "MyLora").
    Returns list of remaining non-empty parts in lowercase.
    """
    name_lower = name.lower()

    # Remove anything in parentheses (e.g., " (1)", "(copy)", etc.)
    name_lower = re.sub(r'\s*\([^)]*\)', '', name_lower)

    # Replace each token with a placeholder to preserve boundaries
    for token in WAN_TOKENS:
        # Use word boundary-aware replacement with _ or - as delimiter
        pattern = rf'(?:^|[_-]){re.escape(token)}(?:[_-]|$)'
        name_lower = re.sub(pattern, '_', name_lower)

    # Split by underscore or hyphen and filter out empty strings
    parts = [p for p in re.split(r'[_-]', name_lower) if p]
    return parts


def fuzzy_match_lora(lora_name, lora_files):
    """
    Attempt to find a matching LoRA using fuzzy matching.
    Handles renamed LoRAs by removing common WAN-related tokens.

    Example: "DR34LAY_HIGH_V2" can match "DR34LAY_I2V_14B_HIGH_V2"

    Returns (matched_file, True) if found, (None, False) otherwise.
    """
    # Normalize the search name
    search_parts = _normalize_name_for_fuzzy(strip_lora_extension(lora_name))
    search_set = set(search_parts)

    # If search_set is empty after normalization, we can't do fuzzy matching
    if not search_set:
        return None, False

    candidates = []
    for lora_file in lora_files:
        file_name_no_ext = strip_lora_extension(os.path.basename(lora_file))
        file_parts = _normalize_name_for_fuzzy(file_name_no_ext)
        file_set = set(file_parts)

        # Check if all search parts are present in the file parts
        if search_set.issubset(file_set):
            # Calculate how well this matches (prefer closer matches)
            extra_parts = len(file_set - search_set)
            candidates.append((lora_file, file_name_no_ext, extra_parts))

    if not candidates:
        return None, False

    # If only one match, return it
    if len(candidates) == 1:
        return candidates[0][0], True

    # Multiple matches - prefer ones that match i2v/t2v from original if present
    lora_name_lower = lora_name.lower()
    has_i2v = 'i2v' in lora_name_lower
    has_t2v = 't2v' in lora_name_lower

    if has_i2v:
        i2v_matches = [c for c in candidates if 'i2v' in c[1].lower()]
        if i2v_matches:
            candidates = i2v_matches
    elif has_t2v:
        t2v_matches = [c for c in candidates if 't2v' in c[1].lower()]
        if t2v_matches:
            candidates = t2v_matches

    # Return the match with fewest extra parts (closest match)
    candidates.sort(key=lambda x: x[2])
    return candidates[0][0], True


def get_lora_relative_path(lora_name):
    """
    Get the relative path for a LoRA that ComfyUI expects for loading.
    Returns (relative_path, found) tuple.
    Supports fuzzy matching for renamed LoRAs.
    """
    lora_files = get_available_loras()

    lora_name_lower = lora_name.lower()

    # Try exact match first
    for lora_file in lora_files:
        file_name_no_ext = strip_lora_extension(os.path.basename(lora_file))
        if file_name_no_ext.lower() == lora_name_lower:
            return lora_file, True

    # Try fuzzy match for renamed LoRAs
    fuzzy_match, found = fuzzy_match_lora(lora_name, lora_files)
    if found:
        return fuzzy_match, True

    # Not found
    return lora_name, False


def resolve_lora_path(lora_name):
    """
    Resolve a LoRA name to its full path using ComfyUI's folder system.
    Uses exact matching first, then fuzzy matching with WAN token handling.
    Returns (full_path_or_name, found) tuple.
    """
    lora_files = get_available_loras()

    # Try exact match first (with extension, as-is from workflow)
    for lora_file in lora_files:
        if lora_file == lora_name:
            return folder_paths.get_full_path("loras", lora_file), True

    # Try matching by name without extension
    lora_name_lower = lora_name.lower()
    for lora_file in lora_files:
        file_name_no_ext = strip_lora_extension(os.path.basename(lora_file))
        if file_name_no_ext.lower() == lora_name_lower:
            return folder_paths.get_full_path("loras", lora_file), True

    # Fuzzy match for renamed LoRAs
    fuzzy_match, found = fuzzy_match_lora(lora_name, lora_files)
    if found:
        return folder_paths.get_full_path("loras", fuzzy_match), True

    return None, False
