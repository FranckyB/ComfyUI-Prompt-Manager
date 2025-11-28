"""
ComfyUI Prompt Manager - A simple custom node for saving and reusing prompts
A prompt management system for ComfyUI with LLM input support
"""
__version__ = "1.1.0"
__author__ = "François Beaudry"
__license__ = "MIT"

from .prompt_manager import PromptManagerNode

NODE_CLASS_MAPPINGS = {
    "PromptManager": PromptManagerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptManager": "Prompt Manager",
}


WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("[PromptManager] Node registered successfully")
