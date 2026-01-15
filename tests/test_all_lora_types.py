"""
Test extraction for all LoRA loader types to ensure compatibility
"""
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

from prompt_extractor import (
    extract_power_lora_loader,
    extract_lora_manager_stacker,
    extract_wan_video_lora_select_multi,
    extract_standard_lora_loader,
    is_lora_node
)

def test_power_lora_loader():
    """Test Power Lora Loader (rgthree) extraction"""
    print("\n" + "=" * 80)
    print("TEST: Power Lora Loader (rgthree)")
    print("=" * 80)

    node = {
        'type': 'Power Lora Loader (rgthree)',
        'widgets_values': [
            {"on": True, "lora": "lora1.safetensors", "strength": 1.0, "strengthTwo": 0.8},
            {"on": False, "lora": "lora2.safetensors", "strength": 0.5, "strengthTwo": None},
            {"on": True, "lora": "lora3.safetensors", "strength": 1.2, "strengthTwo": 1.5}
        ]
    }

    loras = extract_power_lora_loader(node)

    print(f"Extracted {len(loras)} LoRAs:")
    for lora in loras:
        active_status = " [ACTIVE]" if lora.get('active', True) else " [INACTIVE]"
        print(f"  - {lora['name']} (model: {lora['model_strength']}, clip: {lora['clip_strength']}){active_status}")

    assert len(loras) == 3, f"Expected 3 LoRAs, got {len(loras)}"
    assert loras[0]['name'] == 'lora1'
    assert loras[0]['model_strength'] == 1.0
    assert loras[0]['clip_strength'] == 0.8
    assert loras[0]['active'] == True

    assert loras[1]['name'] == 'lora2'
    assert loras[1]['active'] == False

    assert loras[2]['name'] == 'lora3'
    assert loras[2]['model_strength'] == 1.2
    assert loras[2]['clip_strength'] == 1.5

    print("[PASS] Power Lora Loader extraction working correctly!")


def test_lora_manager_stacker():
    """Test Lora Stacker (LoraManager) extraction"""
    print("\n" + "=" * 80)
    print("TEST: Lora Stacker (LoraManager)")
    print("=" * 80)

    node = {
        'type': 'Lora Stacker (LoraManager)',
        'widgets_values': [
            None,  # First item is usually None
            [
                {"name": "stacker_lora1", "strength": 0.33, "active": True, "clipStrength": 0.5},
                {"name": "stacker_lora2", "strength": "0.8", "active": False, "clipStrength": "0.9"},
                {"name": "stacker_lora3", "strength": 1.0, "active": True, "clipStrength": 1.0}
            ]
        ]
    }

    loras = extract_lora_manager_stacker(node)

    print(f"Extracted {len(loras)} LoRAs:")
    for lora in loras:
        active_status = " [ACTIVE]" if lora.get('active', True) else " [INACTIVE]"
        print(f"  - {lora['name']} (model: {lora['model_strength']}, clip: {lora['clip_strength']}){active_status}")

    assert len(loras) == 3, f"Expected 3 LoRAs, got {len(loras)}"
    assert loras[0]['name'] == 'stacker_lora1'
    assert loras[0]['model_strength'] == 0.33
    assert loras[0]['clip_strength'] == 0.5

    assert loras[1]['name'] == 'stacker_lora2'
    assert loras[1]['model_strength'] == 0.8
    assert loras[1]['active'] == False

    print("[PASS] Lora Stacker extraction working correctly!")


def test_wan_video_lora_select_multi():
    """Test WanVideoLoraSelectMulti extraction"""
    print("\n" + "=" * 80)
    print("TEST: WanVideoLoraSelectMulti")
    print("=" * 80)

    node = {
        'type': 'WanVideoLoraSelectMulti',
        'widgets_values': [
            'WanAnimate_relight_lora_fp16.safetensors', 1,
            'lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors', 1.2,
            'NSFW-22-L-e8.safetensors', 1,
            'WAN-2.2-I2V-Handjob-LOW-v1.safetensors', 2,
            'none', 1,
            False, False
        ]
    }

    loras = extract_wan_video_lora_select_multi(node)

    print(f"Extracted {len(loras)} LoRAs:")
    for lora in loras:
        active_status = " [ACTIVE]" if lora.get('active', True) else " [INACTIVE]"
        print(f"  - {lora['name']} (model: {lora['model_strength']}, clip: {lora['clip_strength']}){active_status}")

    assert len(loras) == 4, f"Expected 4 LoRAs, got {len(loras)}"
    assert loras[0]['name'] == 'WanAnimate_relight_lora_fp16'
    assert loras[0]['model_strength'] == 1.0

    assert loras[1]['name'] == 'lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16'
    assert loras[1]['model_strength'] == 1.2

    assert loras[3]['name'] == 'WAN-2.2-I2V-Handjob-LOW-v1'
    assert loras[3]['model_strength'] == 2.0

    print("[PASS] WanVideoLoraSelectMulti extraction working correctly!")


def test_standard_lora_loader():
    """Test standard LoraLoader extraction"""
    print("\n" + "=" * 80)
    print("TEST: Standard LoraLoader")
    print("=" * 80)

    node = {
        'type': 'LoraLoader',
        'widgets_values': ['standard_lora.safetensors', 0.7, 0.9]
    }

    loras = extract_standard_lora_loader(node)

    print(f"Extracted {len(loras)} LoRAs:")
    for lora in loras:
        active_status = " [ACTIVE]" if lora.get('active', True) else " [INACTIVE]"
        print(f"  - {lora['name']} (model: {lora['model_strength']}, clip: {lora['clip_strength']}){active_status}")

    assert len(loras) == 1, f"Expected 1 LoRA, got {len(loras)}"
    assert loras[0]['name'] == 'standard_lora'
    assert loras[0]['model_strength'] == 0.7
    assert loras[0]['clip_strength'] == 0.9
    assert loras[0]['active'] == True

    print("[PASS] Standard LoraLoader extraction working correctly!")


def test_is_lora_node():
    """Test that all LoRA node types are recognized"""
    print("\n" + "=" * 80)
    print("TEST: is_lora_node recognition")
    print("=" * 80)

    lora_types = [
        'Power Lora Loader (rgthree)',
        'Lora Stacker (LoraManager)',
        'LoRA Stacker',
        'LoraLoader',
        'LoraLoaderModelOnly',
        'WanVideoLoraSelectMulti'
    ]

    non_lora_types = [
        'KSampler',
        'CLIPTextEncode',
        'VAEDecode',
        'WanVideoModelLoader'
    ]

    print("Testing LoRA node types:")
    for node_type in lora_types:
        result = is_lora_node(node_type)
        status = "✓" if result else "✗"
        print(f"  {status} {node_type}: {result}")
        assert result, f"{node_type} should be recognized as LoRA node"

    print("\nTesting non-LoRA node types:")
    for node_type in non_lora_types:
        result = is_lora_node(node_type)
        status = "✓" if not result else "✗"
        print(f"  {status} {node_type}: {result}")
        assert not result, f"{node_type} should NOT be recognized as LoRA node"

    print("[PASS] All node type recognition working correctly!")


if __name__ == "__main__":
    test_power_lora_loader()
    test_lora_manager_stacker()
    test_wan_video_lora_select_multi()
    test_standard_lora_loader()
    test_is_lora_node()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
