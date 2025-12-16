"""
ComfyUI Prompt Manager - A simple custom node for saving and reusing prompts
A prompt management system for ComfyUI with LLM input support
"""
__version__ = "1.8.2"
__author__ = "François Beaudry"
__license__ = "MIT"

from .prompt_manager import PromptManager
from .prompt_generator import PromptGenerator
from .prompt_generator_options import PromptGenOptions

NODE_CLASS_MAPPINGS = {
    "PromptManager": PromptManager,
    "PromptGenerator": PromptGenerator,
    "PromptGenOptions": PromptGenOptions
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptManager": "Prompt Manager",
    "PromptGenerator": "Prompt Generator",
    "PromptGenOptions": "Prompt Generator Options"
}


WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("[PromptManager] Node registered successfully")
