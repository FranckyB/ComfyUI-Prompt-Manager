# Feature Reference

This page contains the detailed feature overview that was moved out of the main README for faster scanning.

## Prompt Toolset

### Prompt Manager
- Category-based prompt organization
- Save/load prompts quickly
- NSFW tagging and filtering
- LLM input toggle
- Persistent storage in your ComfyUI user folder

### Prompt Manager Advanced
- All Prompt Manager features, plus LoRA and trigger-word integration
- Dual LoRA stack support for complex pipelines (for example Wan video)
- LoRA tags with inline strength editing
- Toggle active state per LoRA without rewiring
- Save LoRAs with prompt entries
- Override/merge behavior for connected vs saved LoRAs
- Optional workflow_data input integration
- Missing LoRA highlighting and LoRA Manager preview support
- Thumbnail generation from saved prompt entries

### Prompt Extractor
- Extract prompt/model/LoRA metadata from image, video, and JSON workflow files
- Supports ComfyUI, A1111/Forge, and WebP metadata formats
- First-frame extraction from videos
- Dual LoRA stack extraction for compatible workflows
- Input/output folder browsing
- Can also be used as an advanced media loader with metadata awareness

### Prompt Generator
- Local LLM generation with llama.cpp or Ollama
- Text enhancement and image analysis modes
- Optional JSON output mode
- Thinking mode support
- Multi-image support (up to 5 total with options node)
- Prompt Generator Options node for model and parameter control

## Recipe Toolset

The Recipe Toolset is designed to build, manage, and rerun reusable generation pipelines. You can author recipes from scratch in Builder, combine recipe logic with Prompt Manager/Prompt Manager Advanced to reuse saved prompts, or import existing metadata via Extractor when you want to bootstrap from prior outputs.

### Recipe Builder
- Central editing surface to build or modify recipes
- Primary authoring path for creating recipes from scratch
- Provides a recipe-focused workspace to inspect, adjust, and refine recipe fields without manually wiring every parameter node-by-node
- Can be used standalone or connected with other recipe/prompt nodes
- Works well with Prompt Manager so saved prompts can be reused as part of recipe-driven pipelines
- Ideal for iterative tuning: extract once, then keep adjusting recipe values and rerendering until the pipeline is where you want it
- Helps standardize presets so repeated jobs start from consistent known-good settings

### Recipe Extractor
- Reads and normalizes workflow metadata into reusable recipe data
- Converts raw generation metadata into a structured recipe format that is easier to edit and pass between nodes
- Helps preserve key settings (for example model family, sampler context, and recipe-specific parameters) when moving from ad-hoc runs to reusable setups
- Useful when you want to recreate a look/style from existing outputs and then iterate from a stable baseline

### Recipe Renderer
- Executes recipe-driven generation for fast reruns after setup
- Interprets recipe data and applies it to generation execution so you can reproduce prior results or batch variants with minimal manual changes
- Reduces setup overhead for repeated jobs by moving configuration logic into reusable recipe data
- Especially useful when you want to test multiple prompt/input variations while keeping the same core generation recipe intact

### Recipe Relay
- Relays recipe data and compatible passthrough context across nodes
- Acts as a clean handoff point for recipe payloads so larger graphs stay modular and easier to reason about
- Preserves compatible context as data moves through the pipeline, reducing brittle copy/paste style wiring patterns
- Helpful when splitting complex graphs into logical stages (extract/build/render/manage) without losing recipe continuity

### Recipe Model Loader
- Loads model/checkpoint assets based on recipe data
- Resolves and applies model selections encoded in recipes so execution follows the intended model path automatically
- Supports model slot selection and loader-specific behavior
- Makes recipe playback more reliable across sessions by tying loader behavior to saved recipe intent instead of manual recollection
- Useful for multi-model pipelines where model choice is part of the recipe contract, not an afterthought

### Recipe Manager
- Save, browse, and reuse recipe entries
- Provides persistent recipe storage so successful setups can be cataloged and recalled quickly
- Works with Builder to quickly review and update saved recipe data
- Encourages a library-style workflow: capture working recipes, name/tag them, and reapply them across projects
- Great for maintaining repeatable production presets while still allowing safe versioned edits over time

## Model Family Support

Built-in support includes:
- Flux 1
- Flux 2
- Ernie
- SDXL
- Wan
- Qwen
- Z-Image
- Wan Image
- Wan Video (I2V and T2V)
