"""
ComfyUI Prompt Manager - A comprehensive prompt and workflow management system for ComfyUI.
Features: prompt management, LoRA stacks, workflow extraction, workflow generation.
"""
__version__ = "1.23.0"
__author__ = "François Beaudry"
__license__ = "MIT"

from .nodes.prompt_apply_lora import PromptApplyLora
from .nodes.prompt_manager import PromptManager
from .nodes.prompt_manager_advanced import PromptManagerAdvanced
from .nodes.prompt_generator import PromptGenerator
from .nodes.prompt_generator_options import PromptGenOptions
from .nodes.prompt_extractor import PromptExtractor
from .nodes.prompt_extractor import WorkflowExtractor
from .nodes.workflow_builder import WorkflowBuilder
from .nodes.workflow_renderer import WorkflowRenderer
from .nodes.workflow_context import WorkflowContext
from .nodes.workflow_model_loader import WorkflowModelLoader

NODE_CLASS_MAPPINGS = {
    "PromptApplyLora":       PromptApplyLora,
    "PromptManager":         PromptManager,
    "PromptManagerAdvanced": PromptManagerAdvanced,
    "PromptGenerator":       PromptGenerator,
    "PromptGenOptions":      PromptGenOptions,
    "PromptExtractor":       PromptExtractor,
    "WorkflowExtractor":     WorkflowExtractor,
    "WorkflowBuilder":       WorkflowBuilder,
    "WorkflowRenderer":      WorkflowRenderer,
    "WorkflowContext":       WorkflowContext,
    "WorkflowModelLoader":   WorkflowModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptApplyLora":       "Prompt Apply LoRA",
    "PromptManager":         "Prompt Manager",
    "PromptManagerAdvanced": "Prompt Manager Advanced",
    "PromptGenerator":       "Prompt Generator",
    "PromptGenOptions":      "Prompt Generator Options",
    "PromptExtractor":       "Prompt Extractor",
    "WorkflowExtractor":     "Workflow Extractor",
    "WorkflowBuilder":       "Workflow Builder",
    "WorkflowRenderer":      "Workflow Renderer",
    "WorkflowContext":       "Workflow Context",
    "WorkflowModelLoader":   "Workflow Model Loader",
}

# WORKFLOW_DICT is a custom ComfyUI type used to pass workflow configuration
# from WorkflowManager → WorkflowGenerator.
# It is a plain Python dict — ComfyUI passes it by reference, no special handling.

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("[PromptManager] Nodes registered: PromptApplyLora, PromptManagerAdvanced, PromptGenerator, PromptGenOptions, PromptExtractor, WorkflowExtractor, WorkflowBuilder, WorkflowRenderer, WorkflowContext, WorkflowModelLoader")
