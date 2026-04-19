"""
ComfyUI Prompt Manager - A comprehensive prompt and workflow management system for ComfyUI.
Features: prompt & workflow management, LoRA stacks, workflow extraction, workflow generation.
"""
__version__ = "1.25.0"
__author__ = "François Beaudry"
__license__ = "MIT"

from .nodes.apply_lora import ApplyLoraPlusPlus
from .nodes.prompt_manager_adv import PromptManagerAdvanced
from .nodes.prompt_manager_basic import PromptManager
from .nodes.prompt_generator import PromptGenerator
from .nodes.prompt_generator_options import PromptGenOptions
from .nodes.prompt_extractor import PromptExtractor
from .nodes.prompt_extractor import WorkflowExtractor
from .nodes.workflow_builder import WorkflowBuilder
from .nodes.workflow_renderer import WorkflowRenderer
from .nodes.workflow_bridge import WorkflowBridge
from .nodes.workflow_model_loader import WorkflowModelLoader
from .nodes.workflow_saver import WorkflowSaver
from .nodes.workflow_manager import WorkflowManager

NODE_CLASS_MAPPINGS = {
    "PromptApplyLora":       ApplyLoraPlusPlus,
    "PromptManagerAdvanced": PromptManagerAdvanced,
    "PromptManager":         PromptManager,
    "PromptGenerator":       PromptGenerator,
    "PromptGenOptions":      PromptGenOptions,
    "PromptExtractor":       PromptExtractor,
    "WorkflowExtractor":     WorkflowExtractor,
    "WorkflowBuilder":       WorkflowBuilder,
    "WorkflowRenderer":      WorkflowRenderer,
    "WorkflowBridge":        WorkflowBridge,
    "WorkflowModelLoader":   WorkflowModelLoader,
    "WorkflowSaver":         WorkflowSaver,
    "WorkflowManager":       WorkflowManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptApplyLora":       "Apply LoRA++",
    "PromptManagerAdvanced": "Prompt Manager",
    "PromptManager":         "Prompt Manager (Basic)",
    "PromptGenerator":       "Prompt Generator",
    "PromptGenOptions":      "Prompt Generator Options",
    "PromptExtractor":       "Prompt Extractor",
    "WorkflowExtractor":     "Workflow Extractor",
    "WorkflowBuilder":       "Workflow Builder",
    "WorkflowRenderer":      "Workflow Renderer",
    "WorkflowBridge":        "Workflow Bridge",
    "WorkflowModelLoader":   "Workflow Model Loader",
    "WorkflowSaver":         "Workflow Saver",
    "WorkflowManager":       "Workflow Manager",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("[PromptManager] Nodes registered: Apply LoRA++, Prompt Manager (Basic), Prompt Manager, Prompt Generator, Prompt Generator Options, Prompt Extractor, Workflow Extractor, Workflow Builder, Workflow Renderer, Workflow Bridge, Workflow Model Loader, Workflow Saver, Workflow Manager")
