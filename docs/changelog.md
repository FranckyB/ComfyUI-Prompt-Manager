# Changelog

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
