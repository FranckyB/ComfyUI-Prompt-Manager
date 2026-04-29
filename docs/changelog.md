# Changelog

## version 2.5.0
- **Recipe Builder is now Multi-model**
  - Expanded Builder authoring around `recipe_data` with slot-aware model blocks (`model_a`/`model_b`/`model_c`/`model_d`).
  - Improved cross-slot behavior for prompts, LoRAs, sampler settings, and resolution handling in multi-model workflows.
- **Authoritative Builder extraction flow**
  - Extraction now preserves Builder-authored v2 model data as the source of truth when embedded Builder v2 metadata is present.
  - Added compatibility safeguards so legacy/non-v2 workflows continue to use fallback extraction logic.
- **Builder lock/reset and execution consistency**
  - Improved update and execute behavior so incoming recipe values, lock states, and current UI values resolve more predictably.
  - Strength/reset handling was tightened for better consistency across slot updates and reruns.
- **New LoRA utility node: MultiLoraCombine**
  - Added `MultiLoraCombine` to merge optional multi-stack input with optional stack A/B/C/D inputs into a single `MULTI_LORA_STACK` output.
  - Helps consolidate mixed LoRA sources for multi-model Builder and downstream recipe flows.
- **New feature: Multi LoRA Stacker with Lora-Manager compatibility**
  - Added a new Multi LoRA Stacker for slot-aware and multi-stack recipe flows.
  - Compatible with [Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager), including interoperability with managed LoRA selections and previews.

## version 2.2.0
- **Recipe_data v2 expansion and slot-aware workflows**
  - Expanded v2 compatibility across Builder, Relay, Renderer, Manager, and extraction utilities.
  - Improved model slot behavior for `model_a`/`model_b`/`model_c`/`model_d`, including WAN pair handling.
- **Recipe Builder Update flow and persistence improvements**
  - Update Recipe now respects Pull From selection consistently and supports expected clear behavior for empty/none selections.
  - Pull From / Send To state persistence improved across tab switches and workflow reloads.
  - Update Recipe visibility refined to avoid incomplete Builder-to-Builder pull paths.
- **LoRA synchronization and execution-order fixes**
  - Fixed connected LoRA add/remove timing so `recipe_data` reflects changes on first execute.
  - Improved LoRA state authority between connected inputs and persisted UI overrides.
- **Renderer validation hardening**
  - Renderer now fails fast with a clear error when the selected model slot is undefined.
  - Prevents silent renders from empty/non-existent selected model slots.
- **Prompt Generator GPU utilization and VRAM prep**
  - Ensured llama.cpp server launch uses `-ngl 100` so generation is pushed to GPU as much as possible.
  - Added more aggressive cache/VRAM flushing before generation to reduce memory pressure and improve run reliability.

## version 2.1.1
- Updated Documentation
- Bug fixes

## version 2.1.0
- **Naming Migration: Workflow -> Recipe**
  - Renamed node/toolset terminology to Recipe naming across the addon UI and docs.
  - Bug fixes.

## version 2.0.2
- Added Ernie family support in Workflow Builder/Renderer.
- Bug fixes.

## version 2.0.1
- **Added Denoise options to Workflow Pipeline**
  - Workflow data now contains denoise value (Extractor returns 1 by default).
  - Allows mid-run model handoff using KSampler denoise.
  - Bug fixes.

## version 2.0.0
- **Prompt Manager becomes Prompt + Workflow Manager**
  - Workflow Manager allows saving workflow and integrates with Prompt Manager save/reuse behavior.
  - Extractor/Builder/Manager workflow loop focused on simplified reusable workflow cores.
  - recipe_data-first persistence improvements.
  - Documentation updates.

## version 1.25.0
- **Workflow Tools Release Milestone**
  - Workflow Builder / Workflow Renderer / Workflow Bridge / Workflow Model Loader form a complete editable workflow pipeline.
  - Improved extractor-to-builder update flow.
  - Improved Builder dropdown stability and persistence.
  - Improved tab-switch persistence for Builder UI state.
  - Improved compatibility with Builder persistence extraction.
- **Wan i2v workflow quality and UX improvements**
- **Prompt Manager Advanced save + thumbnail improvements**

## version 1.22.6
- Prompt Extractor now embeds extracted data in workflow so it can extract itself.

## version 1.22.5
- Thumbnail generation in Prompt Manager Advanced.
- Improved model high/low detection.

## version 1.22.1
- Prompt Model Loader improvements, including GGUF support and persistence improvements.

## version 1.22.0
- New node: Prompt Model Loader.
- Expanded model/checkpoint extraction in Prompt Extractor.
- A1111/Forge image support improvements.
- WebP metadata support improvements.
- Video metadata parsing improvements.

## version 1.21.x to 1.0.0
Older historical entries were previously listed in README and can still be recovered from repository history.
