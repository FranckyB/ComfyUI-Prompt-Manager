"""
ComfyUI Prompt Manager - A simple custom node for saving and reusing prompts
A prompt management system for ComfyUI with LLM input support
"""
__version__ = "1.15.6"
__author__ = "François Beaudry"
__license__ = "MIT"

from .prompt_manager import PromptManager
from .prompt_manager_advanced import PromptManagerAdvanced
from .prompt_generator import PromptGenerator
from .prompt_generator_options import PromptGenOptions
from .prompt_apply_lora import PromptApplyLora
from .prompt_extractor import PromptExtractor
from .model_manager import get_local_models
from .save_video_h26x import SaveVideoH26x, LoadLatentFile, MonoToStereo, GetVideoComponentsPlus
from server import PromptServer

# Initialize latent preview hook (with VHS conflict detection)
from .latent_preview import install_latent_preview_hook
install_latent_preview_hook()

NODE_CLASS_MAPPINGS = {
    "PromptManager": PromptManager,
    "PromptManagerAdvanced": PromptManagerAdvanced,
    "PromptGenerator": PromptGenerator,
    "PromptGenOptions": PromptGenOptions,
    "PromptApplyLora": PromptApplyLora,
    "PromptExtractor": PromptExtractor,
    "SaveVideoH26x": SaveVideoH26x,
    "LoadLatentFile": LoadLatentFile,
    "AudioMonoToStereo": MonoToStereo,
    "GetVideoComponentsPlus": GetVideoComponentsPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptManager": "Prompt Manager",
    "PromptManagerAdvanced": "Prompt Manager Advanced",
    "PromptGenerator": "Prompt Generator",
    "PromptGenOptions": "Prompt Generator Options",
    "PromptApplyLora": "Prompt Apply LoRA",
    "PromptExtractor": "Prompt Extractor",
    "SaveVideoH26x": "Save Video H264/H265",
    "LoadLatentFile": "Load Latent File",
    "AudioMonoToStereo": "Audio Mono to Stereo",
    "GetVideoComponentsPlus": "Get Video Components+",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("[PromptManager] Node registered successfully")
