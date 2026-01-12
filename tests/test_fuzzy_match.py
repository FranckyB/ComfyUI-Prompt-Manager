"""
Test fuzzy LoRA matching logic
"""
import os
import re

def fuzzy_match_lora(lora_name, lora_files):
    """
    Attempt to find a matching LoRA using fuzzy matching.
    Handles renamed LoRAs by removing common WAN-related tokens.

    Example: "DR34LAY_HIGH_V2" can match "DR34LAY_I2V_14B_HIGH_V2"

    Returns (matched_file, True) if found, (None, False) otherwise.
    """
    # Tokens to remove for fuzzy matching (case-insensitive)
    # Order matters: remove longer tokens first to avoid partial matches
    wan_tokens = ['wan_2_2', 'wan22', 'wan2.2', '20epoc', 'a14b', '14b', 'i2v', 't2v']

    def normalize_name(name):
        """
        Remove WAN tokens from name, treating underscores and hyphens as separators.
        Also removes content in parentheses (e.g., "MyLora (1)" becomes "MyLora").
        Returns list of remaining non-empty parts in lowercase.
        """
        name_lower = name.lower()
        
        # Remove anything in parentheses (e.g., " (1)", "(copy)", etc.)
        name_lower = re.sub(r'\s*\([^)]*\)', '', name_lower)

        # Replace each token with a placeholder to preserve boundaries
        for token in wan_tokens:
            # Use word boundary-aware replacement with _ or - as delimiter
            # Replace [_-]token[_-], [_-]token, token[_-], or standalone token
            pattern = rf'(?:^|[_-]){re.escape(token)}(?:[_-]|$)'
            name_lower = re.sub(pattern, '_', name_lower)

        # Split by underscore or hyphen and filter out empty strings
        parts = [p for p in re.split(r'[_-]', name_lower) if p]
        return parts

    # Normalize the search name
    search_parts = normalize_name(os.path.splitext(lora_name)[0])
    search_set = set(search_parts)

    candidates = []
    for lora_file in lora_files:
        file_name_no_ext = os.path.splitext(os.path.basename(lora_file))[0]
        file_parts = normalize_name(file_name_no_ext)
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


def test_fuzzy_matching():
    """Test various fuzzy matching scenarios"""

    # Simulate available LoRA files
    available_loras = [
        "DR34LAY_I2V_14B_HIGH_V2.safetensors",
        "DR34LAY_T2V_14B_HIGH_V2.safetensors",
        "DR34LAY_I2V_A14B_LOW_V2.safetensors",
        "SomeOther_I2V_LoRA.safetensors",
        "ExactMatch.safetensors",
        "MyLora_WAN_2_2_I2V_Style.safetensors",
        "MyLora_Style.safetensors",
        "CoolStyle_I2V (1).safetensors",
        "CoolStyle_I2V.safetensors",
    ]

    test_cases = [
        ("DR34LAY_HIGH_V2", "DR34LAY_I2V_14B_HIGH_V2.safetensors", "Renamed WAN LoRA (missing I2V, 14B)"),
        ("DR34LAY_HIGH_V2.safetensors", "DR34LAY_I2V_14B_HIGH_V2.safetensors", "Renamed WAN LoRA with extension"),
        ("DR34LAY_LOW_V2", "DR34LAY_I2V_A14B_LOW_V2.safetensors", "Renamed WAN LoRA (missing I2V, A14B)"),
        ("DR34LAY_I2V_HIGH_V2", "DR34LAY_I2V_14B_HIGH_V2.safetensors", "Has I2V - should prefer I2V match"),
        ("DR34LAY_T2V_HIGH_V2", "DR34LAY_T2V_14B_HIGH_V2.safetensors", "Has T2V - should prefer T2V match"),
        ("MyLora_Style", "MyLora_WAN_2_2_I2V_Style.safetensors", "Should match with WAN_2_2 removed"),
        ("CoolStyle_I2V", "CoolStyle_I2V (1).safetensors", "Should match ignoring (1) suffix"),
        ("ExactMatch", "ExactMatch.safetensors", "Exact match case"),
        ("NonExistent_LoRA", None, "Non-existent LoRA"),
    ]

    print("=" * 80)
    print("FUZZY LORA MATCHING TESTS")
    print("=" * 80)

    all_passed = True
    for search_name, expected_match, description in test_cases:
        result, found = fuzzy_match_lora(search_name, available_loras)
        expected_found = expected_match is not None

        passed = (found == expected_found and (not found or result == expected_match))
        status = "[PASS]" if passed else "[FAIL]"

        print(f"\n{status} {description}")
        print(f"  Search: '{search_name}'")
        print(f"  Expected: {expected_match}")
        print(f"  Got: {result}")

        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("[SUCCESS] All tests passed!")
    else:
        print("[FAILURE] Some tests failed!")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    test_fuzzy_matching()
