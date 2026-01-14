"""
Test the specific workflow with disconnected LoRAs
"""
import json
import sys
import os
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Mock dependencies
sys.modules['folder_paths'] = Mock()
sys.modules['server'] = Mock()
sys.modules['torch'] = Mock()
sys.modules['PIL'] = Mock()

from prompt_extractor import parse_workflow_for_prompts

def test_unsaved_workflow_3():
    """
    Test the specific workflow that has disconnected LoRAs
    """
    workflow_path = r"D:\ComfyUI\user\default\workflows\Unsaved Workflow (3).json"

    if not os.path.exists(workflow_path):
        print(f"Workflow file not found: {workflow_path}")
        return

    with open(workflow_path, 'r', encoding='utf-8') as f:
        workflow_data = json.load(f)

    result = parse_workflow_for_prompts(None, workflow_data)

    # Filter to ACTIVE only (like the real PromptExtractor node does at lines 1565/1581)
    active_loras_a = [lora for lora in result['loras_a'] if lora.get('active', True)]
    active_loras_b = [lora for lora in result['loras_b'] if lora.get('active', True)]

    print("\n" + "=" * 80)
    print("TEST: Unsaved Workflow (3) - Should NOT include disconnected LoRAs")
    print("=" * 80)

    print(f"\nLoRAs Stack A (ALL): {len(result['loras_a'])} LoRAs")
    for lora in result['loras_a'][:10]:  # Show first 10
        active_status = " [ACTIVE]" if lora.get('active', True) else " [INACTIVE]"
        print(f"  - {lora['name']} (model: {lora['model_strength']}, clip: {lora['clip_strength']}){active_status}")
    if len(result['loras_a']) > 10:
        print(f"  ... and {len(result['loras_a']) - 10} more")

    print(f"\nLoRAs Stack A (ACTIVE ONLY): {len(active_loras_a)} LoRAs")
    for lora in active_loras_a:
        print(f"  - {lora['name']} (model: {lora['model_strength']}, clip: {lora['clip_strength']})")

    print(f"\nLoRAs Stack B (ALL): {len(result['loras_b'])} LoRAs")
    for lora in result['loras_b'][:10]:  # Show first 10
        active_status = " [ACTIVE]" if lora.get('active', True) else " [INACTIVE]"
        print(f"  - {lora['name']} (model: {lora['model_strength']}, clip: {lora['clip_strength']}){active_status}")
    if len(result['loras_b']) > 10:
        print(f"  ... and {len(result['loras_b']) - 10} more")

    print(f"\nLoRAs Stack B (ACTIVE ONLY): {len(active_loras_b)} LoRAs")
    for lora in active_loras_b:
        print(f"  - {lora['name']} (model: {lora['model_strength']}, clip: {lora['clip_strength']})")

    print("\n" + "=" * 80)

    # Check for the disconnected LoRAs in ACTIVE lists only
    active_lora_names_a = [lora['name'] for lora in active_loras_a]
    active_lora_names_b = [lora['name'] for lora in active_loras_b]

    has_23high = any('23High noise-Cumshot' in name for name in active_lora_names_a + active_lora_names_b)
    has_56low = any('56Low noise-Cumshot' in name for name in active_lora_names_a + active_lora_names_b)

    print("\nDisconnected LoRA check (in ACTIVE lists):")
    print(f"  '23High noise-Cumshot Aesthetics' found: {has_23high} (should be False)")
    print(f"  '56Low noise-Cumshot Aesthetics' found: {has_56low} (should be False)")
    print(f"\nTotal ACTIVE LoRAs extracted: {len(active_loras_a) + len(active_loras_b)} (expected: 8)")

    if has_23high or has_56low:
        print("\n[FAIL] Disconnected/inactive LoRAs are appearing in active lists!")
    elif len(active_loras_a) == 4 and len(active_loras_b) == 4:
        print("\n[PASS] Disconnected/inactive LoRAs correctly excluded!")
    else:
        print(f"\n[FAIL] Expected 4 active LoRAs per stack, got {len(active_loras_a)} and {len(active_loras_b)}")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    test_unsaved_workflow_3()
