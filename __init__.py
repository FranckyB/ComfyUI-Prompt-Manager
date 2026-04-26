"""
ComfyUI Prompt Manager - A comprehensive prompt and workflow management system for ComfyUI.
Features: prompt & workflow management, LoRA stacks, workflow extraction, workflow generation.
"""
__version__ = "2.2.0"
__author__ = "François Beaudry"
__license__ = "MIT"

from .nodes.apply_lora import ApplyLoraPlusPlus
from .nodes.prompt_manager_adv import PromptManagerAdvanced
from .nodes.prompt_manager_basic import PromptManager
from .nodes.prompt_generator import PromptGenerator
from .nodes.prompt_generator_options import PromptGenOptions
from .nodes.prompt_extractor import PromptExtractor
from .nodes.prompt_extractor import WorkflowExtractor as RecipeExtractor
from .nodes.recipe_builder import WorkflowBuilder as RecipeBuilder
from .nodes.recipe_builder import WorkflowBuilderWan as RecipeBuilderWan
from .nodes.recipe_builder import WorkflowBuilderMulti as RecipeBuilderMulti
from .nodes.recipe_renderer import WorkflowRenderer as RecipeRenderer
from .nodes.recipe_relay import WorkflowRelay as RecipeRelay
from .nodes.recipe_model_loader import WorkflowModelLoader as RecipeModelLoader
from .nodes.recipe_model_picker import RecipeModelPicker
from .nodes.recipe_manager import WorkflowManager as RecipeManager

NODE_CLASS_MAPPINGS = {
    "PromptApplyLora":       ApplyLoraPlusPlus,
    "PromptManagerAdvanced": PromptManagerAdvanced,
    "PromptManager":         PromptManager,
    "PromptGenerator":       PromptGenerator,
    "PromptGenOptions":      PromptGenOptions,
    "PromptExtractor":       PromptExtractor,
    "RecipeExtractor":       RecipeExtractor,
    "RecipeBuilder":         RecipeBuilder,
    "RecipeBuilderWan":      RecipeBuilderWan,
    "RecipeBuilderMulti":    RecipeBuilderMulti,
    "RecipeRenderer":        RecipeRenderer,
    "RecipeRelay":           RecipeRelay,
    "RecipeModelLoader":     RecipeModelLoader,
    "RecipeModelPicker":     RecipeModelPicker,
    "RecipeManager":         RecipeManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptApplyLora":       "Apply LoRA++",
    "PromptManagerAdvanced": "Prompt Manager",
    "PromptManager":         "Prompt Manager (Basic)",
    "PromptGenerator":       "Prompt Generator",
    "PromptGenOptions":      "Prompt Generator Options",
    "PromptExtractor":       "Prompt Extractor",
    "RecipeExtractor":       "Recipe Extractor",
    "RecipeBuilder":         "Recipe Builder",
    "RecipeBuilderWan":      "Recipe Builder (WAN)",
    "RecipeBuilderMulti":    "Recipe Builder (Multi Model)",
    "RecipeRenderer":        "Recipe Renderer",
    "RecipeRelay":           "Recipe Relay",
    "RecipeModelLoader":     "Recipe Model Loader",
    "RecipeModelPicker":     "Recipe Model Picker",
    "RecipeManager":         "Recipe Manager",
}

WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
print("[PromptManager] Nodes registered: Apply LoRA++, Prompt Manager (Basic), Prompt Manager, Prompt Generator, Prompt Generator Options, Prompt Extractor, Recipe Extractor, Recipe Builder, Recipe Builder (WAN), Recipe Merge, Recipe Renderer, Recipe Relay, Recipe Model Loader, Recipe Model Picker, Recipe Manager")

