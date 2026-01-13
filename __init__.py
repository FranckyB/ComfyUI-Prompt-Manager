"""
ComfyUI Prompt Manager - A simple custom node for saving and reusing prompts
A prompt management system for ComfyUI with LLM input support
"""
__version__ = "1.12.7"
__author__ = "François Beaudry"
__license__ = "MIT"

from .prompt_manager import PromptManager
from .prompt_manager_advanced import PromptManagerAdvanced
from .prompt_generator import PromptGenerator
from .prompt_generator_options import PromptGenOptions
from .prompt_apply_lora import PromptApplyLora
from .prompt_extractor import PromptExtractor
from .model_manager import get_local_models
from server import PromptServer

NODE_CLASS_MAPPINGS = {
    "PromptManager": PromptManager,
    "PromptManagerAdvanced": PromptManagerAdvanced,
    "PromptGenerator": PromptGenerator,
    "PromptGenOptions": PromptGenOptions,
    "PromptApplyLora": PromptApplyLora,
    "PromptExtractor": PromptExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptManager": "Prompt Manager",
    "PromptManagerAdvanced": "Prompt Manager Advanced",
    "PromptGenerator": "Prompt Generator",
    "PromptGenOptions": "Prompt Generator Options",
    "PromptApplyLora": "Prompt Apply LoRA",
    "PromptExtractor": "Prompt Extractor"
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("[PromptManager] Node registered successfully")
