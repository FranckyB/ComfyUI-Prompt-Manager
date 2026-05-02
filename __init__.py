"""
ComfyUI Prompt Manager - A comprehensive prompt and workflow management system for ComfyUI.
Features: prompt & workflow management, LoRA stacks, workflow extraction, workflow generation.
"""
__version__ = "2.5.1"
__author__ = "François Beaudry"
__license__ = "MIT"

from .nodes.apply_lora import ApplyLoraPlusPlus
from .nodes.prompt_manager_adv import PromptManagerAdvanced
from .nodes.prompt_manager_basic import PromptManager
from .nodes.prompt_generator import PromptGenerator
from .nodes.prompt_generator_options import PromptGenOptions
from .nodes.prompt_extractor import PromptExtractor
from .nodes.prompt_extractor import WorkflowExtractor as RecipeExtractor
from .nodes.recipe_builder import RecipeBuilder
from .nodes.recipe_renderer import WorkflowRenderer as RecipeRenderer
from .nodes.recipe_relay import WorkflowRelay as RecipeRelay
from .nodes.recipe_model_loader import WorkflowModelLoader as RecipeModelLoader
from .nodes.recipe_model_picker import RecipeModelPicker
from .nodes.recipe_manager import WorkflowManager as RecipeManager
from .nodes.multi_prompt import RecipeBuilderMultiPrompts
from .nodes.multi_lora_stacker import MultiLoraStackerLM, MultiLoraCombine

NODE_CLASS_MAPPINGS = {
    "PromptApplyLora":           ApplyLoraPlusPlus,
    "PromptManagerAdvanced":     PromptManagerAdvanced,
    "PromptManager":             PromptManager,
    "PromptGenerator":           PromptGenerator,
    "PromptGenOptions":          PromptGenOptions,
    "PromptExtractor":           PromptExtractor,
    "RecipeExtractor":           RecipeExtractor,
    "RecipeBuilder":             RecipeBuilder,
    "RecipeRenderer":            RecipeRenderer,
    "RecipeRelay":               RecipeRelay,
    "RecipeModelLoader":         RecipeModelLoader,
    "RecipeModelPicker":         RecipeModelPicker,
    "RecipeManager":             RecipeManager,
    "RecipeBuilderMultiPrompts": RecipeBuilderMultiPrompts,
    "MultiLoraStackerLM":        MultiLoraStackerLM,
    "MultiLoraCombine":          MultiLoraCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptApplyLora":            "Apply LoRA++",
    "PromptManagerAdvanced":      "Prompt Manager",
    "PromptManager":              "Prompt Manager (Basic)",
    "PromptGenerator":            "Prompt Generator",
    "PromptGenOptions":           "Prompt Generator Options",
    "PromptExtractor":            "Prompt Extractor",
    "RecipeExtractor":            "Recipe Extractor",
    "RecipeBuilder":              "Recipe Builder",
    "RecipeRenderer":             "Recipe Renderer",
    "RecipeRelay":                "Recipe Relay",
    "RecipeModelLoader":          "Recipe Model Loader",
    "RecipeModelPicker":          "Recipe Model Picker",
    "RecipeManager":              "Recipe Manager",
    "RecipeBuilderMultiPrompts":  "Multi Prompts",
    "MultiLoraStackerLM":         "Multi LoRA Stack",
    "MultiLoraCombine":           "Multi LoRA Combine",
}

WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
print("[PromptManager] Nodes registered: Apply LoRA++, Prompt Manager, Prompt Generator, Prompt Extractor, Recipe Extractor, Recipe Builder, Recipe Renderer, Recipe Relay, Recipe Manager")
