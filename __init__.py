"""
ComfyUI Prompt Manager - A simple custom node for saving and reusing prompts
A prompt management system for ComfyUI with LLM input support
"""
__version__ = "1.22.6"
__author__ = "François Beaudry"
__license__ = "MIT"

from .nodes.prompt_manager import PromptManager
from .nodes.prompt_manager_advanced import PromptManagerAdvanced
from .nodes.prompt_generator import PromptGenerator
from .nodes.prompt_generator_options import PromptGenOptions
from .nodes.prompt_extractor import PromptExtractor
from .nodes.prompt_model_loader import PromptModelLoader

NODE_CLASS_MAPPINGS = {
    "PromptManager": PromptManager,
    "PromptManagerAdvanced": PromptManagerAdvanced,
    "PromptGenerator": PromptGenerator,
    "PromptGenOptions": PromptGenOptions,
    "PromptExtractor": PromptExtractor,
    "PromptModelLoader": PromptModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptManager": "Prompt Manager",
    "PromptManagerAdvanced": "Prompt Manager Advanced",
    "PromptGenerator": "Prompt Generator",
    "PromptGenOptions": "Prompt Generator Options",
    "PromptExtractor": "Prompt Extractor",
    "PromptModelLoader": "Prompt Model Loader",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("[PromptManager] Node registered successfully")
