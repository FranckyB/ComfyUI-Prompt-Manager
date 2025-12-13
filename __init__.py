"""
ComfyUI Prompt Manager - A simple custom node for saving and reusing prompts
A prompt management system for ComfyUI with LLM input support
"""
__version__ = "1.5.1"
__author__ = "François Beaudry"
__license__ = "MIT"

from .prompt_manager import PromptManagerZ
from .prompt_generator import PromptGeneratorZ
from .prompt_generator_options import PromptGenOptionsZ

NODE_CLASS_MAPPINGS = {
    "PromptManagerZ": PromptManagerZ,
    "PromptGeneratorZ": PromptGeneratorZ,
    "PromptGenOptionsZ": PromptGenOptionsZ
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptManagerZ": "Prompt Manager",
    "PromptGeneratorZ": "Prompt Generator",
    "PromptGenOptionsZ": "Prompt Generator Options"
}


WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("[PromptManagerZ] Node registered successfully")
