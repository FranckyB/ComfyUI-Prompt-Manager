# Changelog

## version 3.0.0
- **Breaking: one `MANAGER_DATA` type**
  - Prompt Manager, Prompt Composer, Prompt Layout Builder and the Recipe nodes now share `manager_data` (replaces `RECIPE_DATA` and `COMPOSE_DATA`). Older workflows need these links reconnected.
  - Recipe Renderer outputs `output_image`, `output_latent`, `manager_data`; Recipe Extractor outputs `image`, `manager_data`.
- **New Library Manager** (replaces Composition Manager and Recipe Manager)
  - Saves and reloads the `manager_data` of Prompt Manager, the Recipe nodes, Prompt Composer and Prompt Layout Builder, with thumbnails.
  - Filters for All / Prompt / Recipe / Compose / Layout.
- **New Prompt Library Hub** (sidebar **Prompts** button, can be hidden in Settings)
  - Browse & Edit, Import and Export for Saved Prompts (Prompt Manager / Prompt Composer / Prompt Layout / Prompt Recipe), Saved Snippets and Saved System Prompts.
  - Per-category export; Snippets and System Prompts can also be replaced from a file, and System Prompts can restore the built-in defaults.
- **New Prompt Layout Builder**
  - Draw Object, Character, Text and Panel boxes over a scene prompt, or load them from an LLM.
  - Outputs MOSS-Image layout JSON, Ideogram 4 JSON captions or Krea 2 grounding lines (needs [krea2-turbo-bbox](https://huggingface.co/jimmycarter/krea2-turbo-bbox)).
  - `import_json` reads all three formats, Create Bounding Boxes elements, and layouts saved with Library Manager.
  - Box descriptions can come from Prompt Composer snippets.
- **New Bbox Converter**: converts a layout JSON to bounding boxes and elements, or bounding boxes back to a layout JSON.
- **Prompt Generator**
  - The system prompt dropdown is replaced by a selector: arrows cycle through prompts, the name opens the Prompt Browser.
  - New **Bbox** category with **MOSS Layout**, **Ideogram Layout** and **Krea 2 Layout** system prompts (Create and From Image).
  - Fixed: the selected system prompt was reset to Enhance Prompt when reopening a workflow or switching tabs if the prompt was added or moved after the page loaded. Prompts moved to another category are now found by name.
- **Prompt Manager and editor**
  - Export only selected categories.
  - Preferred model settings, draft handling and clearer save/rename buttons in the prompt editor.
  - Improved NSFW handling in the browser.
- **Prompt Composer**
  - Prompt text tooltips, clear muted parts, RefMod rows.
  - Picker renamed to "Select Prompt Snippet".
- Customizable thumbnail generation steps.
- Nodes 2.0 compatibility.
- Larger text across the node UIs.
- New Krea layout example workflow.

## version 2.8.0
- **Prompt Composer v2**
  - New library format built around prompt groups, with a dialog to create groups and pick their category.
  - Output as text or JSON, before or after the selected prompt, or on its own.
  - Image and video generation modes.
  - Subject numbers and subject types, mute for prompt parts.
  - Export and import of composer libraries; new Generate Bulk Snippets system prompt.
- **Prompt Composer Manager** to save and reload Composer setups with thumbnails.
  - Input prompt and LoRA modes on Prompt Composer and Prompt Manager, so compose data passing through Prompt Manager can add LoRAs and be saved for reuse.
- **Prompt Browser**: double-click to select, Home button, more responsive dialogs.
- Anima family support for thumbnail rendering.
- Preference to hide the downloadable built-in models in Prompt Generator.
- llama-server GPU support detection.
- Trigger word fixes: only connected words are pruned, words saved with the prompt are kept.
- Fixed LoRA stack cards collapsing or stretching to viewport height (thanks Martlet-Tech).
- Prompt Composer example workflows.

## version 2.7.2
- LoRA blacklist keywords for MiniMax and distilled LoRAs.
- Prompt Generator code moved to its own `prompt_generator.js`.
- Converter for the older system prompt storage format.
- README and feature reference updates.

## version 2.7.1
- Prompt text widgets become read-only while an external prompt input is connected; the received prompt is shown after execution.
- Prompt Extractor accepts a video input.
- Video prompt writing guides for MiniMax.
- Prompt Generator input prompt handling and model management improvements.

## version 2.7.0
- **New Prompt Composer** (started as Prompt Mixer): build prompts from multiple reusable parts.
- **System prompts are now user data**
  - Stored in `prompt_generator_data.json` and editable in the Prompt Browser, grouped by category.
  - New default system prompts for image and video models.
- **Prompt Browser**: multi-select, category type filter, scrolling category bar, view modes, unsaved-changes confirmation, bulk prompt builder.
- **llama.cpp** (thanks sophosympatheia)
  - GPU device selection for multi-GPU systems.
  - Fixed the server port ignoring the saved setting.
  - Models and mmproj files are found in subfolders.
- Rotating backups for prompt data files.
- VRAM flush fixes when using a CLIP/text encoder.

## version 2.6.0
- Prompt Generator can use a connected CLIP/text encoder directly.
- Expression Selector: arrow navigation and strength slider.
- Multi LoRA Stacker: configurable stack count.
- Krea2 example workflow.

## version 2.5.4
- **Prompt Manager Advanced browser overhaul**
  - Larger non-compact prompt browser with bigger 3:4 thumbnails.
  - New browser view modes: Grid, Icon (small grid), and List.
  - View mode is persisted in the browser and no longer controlled by a preference default.
  - Compact browser size can still be restored via Preferences.
- **Krea2 thumbnail generation support**
  - Recipe Renderer now supports the Krea2 family for thumbnail renders.
- **New Expression Selector node**
  - Appends a saved Expression-category prompt to an input prompt.
  - Supports gender-aware pronoun substitution.
  - Add new expressions directly through Prompt Manager Advanced.
  - Select `(none)` to pass the input prompt through unchanged.

## version 2.5.3
- **Improved Prompt Extractor to better match FBNodes image loader+**
  - Can now browse any folder and can view in detail, normal and large Thumbnails.

## version 2.5.2
- **Improved LLM prompts in Prompt Generator**
  - New System prompts for Video and Audio Generation

## version 2.5.0
- **Recipe Builder is now Multi-model Only**
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
