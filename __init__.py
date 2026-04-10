"""
ComfyUI Prompt Manager - A comprehensive prompt and workflow management system for ComfyUI.
Features: prompt management, LoRA stacks, workflow extraction, workflow generation.
"""
__version__ = "1.23.0"
__author__ = "François Beaudry"
__license__ = "MIT"

from .nodes.prompt_manager import PromptManager
from .nodes.prompt_manager_advanced import PromptManagerAdvanced
from .nodes.prompt_generator import PromptGenerator
from .nodes.prompt_generator_options import PromptGenOptions
from .nodes.prompt_extractor import PromptExtractor
from .nodes.prompt_model_loader import PromptModelLoader
from .nodes.workflow_extractor import WorkflowExtractor
from .nodes.workflow_manager import WorkflowManager
from .nodes.workflow_generator import WorkflowGenerator

NODE_CLASS_MAPPINGS = {
    "PromptManager":         PromptManager,
    "PromptManagerAdvanced": PromptManagerAdvanced,
    "PromptGenerator":       PromptGenerator,
    "PromptGenOptions":      PromptGenOptions,
    "PromptExtractor":       PromptExtractor,
    "PromptModelLoader":     PromptModelLoader,
    "WorkflowExtractor":     WorkflowExtractor,
    "WorkflowManager":       WorkflowManager,
    "WorkflowGenerator":     WorkflowGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptManager":         "Prompt Manager",
    "PromptManagerAdvanced": "Prompt Manager Advanced",
    "PromptGenerator":       "Prompt Generator",
    "PromptGenOptions":      "Prompt Generator Options",
    "PromptExtractor":       "Prompt Extractor",
    "PromptModelLoader":     "Prompt Model Loader",
    "WorkflowExtractor":     "Workflow Extractor",
    "WorkflowManager":       "Workflow Manager",
    "WorkflowGenerator":     "Workflow Generator",
}

# WORKFLOW_DICT is a custom ComfyUI type used to pass workflow configuration
# from WorkflowManager → WorkflowGenerator.
# It is a plain Python dict — ComfyUI passes it by reference, no special handling.

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("[PromptManager] Nodes registered: PromptManager, PromptManagerAdvanced, "
      "PromptGenerator, PromptGenOptions, PromptExtractor, PromptModelLoader, "
      "WorkflowExtractor, WorkflowManager, WorkflowGenerator")
