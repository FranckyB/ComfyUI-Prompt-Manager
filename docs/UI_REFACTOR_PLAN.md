# Plan: Unified "FB UI" visual language for FBnodes, Prompt-Manager, H3RefModPicker

## Context
The three packs grew independently and each node builds its UI differently. Some use canvas drawing in `onDrawForeground`, others plain-object "button" widgets spliced into `node.widgets`, DOM widgets with `widget.draw` `!important` hacks, or about 900 inline `cssText` assignments with hard-coded colours. There are no shared tokens, and Nodes 2.0 support is partial. The goal is the AusBoss look: a solid node body and title colour, one consistent card of label/value rows, pill toggles, scrub number fields with units, and buttons that look the same. The goal applies to all three packs, works in Classic and Nodes 2.0, and keeps every pack installable on its own. Start with FBnodes.

### Key findings
- **AusBoss pattern:**
  - `js/shared/appearance.mjs` sets `node.color` / `node.bgcolor` per pack.
  - `js/shared/widget_card.mjs` provides `mountWidgetCard(node,{rows})`: one `addDOMWidget` card per node, with native widgets hidden but kept.
  - `widget_visibility.mjs` (`hideWidget`), `scrub_input.mjs`, `panel_controls.mjs`, `panel_layout.mjs` (`fillNodeHeight` / `pinVuePanelHeight` for Vue).
  - Tokens live in `theme.mjs`. CSS is injected once per module, prefixed `ausboss-`.
  - We emulate this approach; we do not import it, because the packs must stay independent.
- **Workflow compatibility is safer than feared.** Installed frontend 1.59.11 serializes `widgets_values` compactly: widgets with `serialize===false` are skipped, and restoring walks only the serializable widgets. Removing or adding JS-only `serialize:false` widgets (Browse, Mask and Crop buttons, file combo, proxies) does **not** shift saved values. Only reordering, renaming or removing serialized widgets (the Python inputs, plus `_fbSaveVideoLastResult`) breaks old workflows. **Most nodes can be redone in place; no v2 keys are expected.**
- **FBnodes** (28 nodes, all V1, category `FBnodes`):
  - `file_browser.js` and `path_browser.js` are the only shared modules. Helpers are copied 3–6× (`hideWidget`, `isAbsolutePath`, title play-icon, `showConfirm`, wheel forwarding, Load+ loader scaffold, Clip/VACE Stitcher list UI, Save Image/Video result cache).
  - **Canvas-only UI that degrades in Vue:** title play icons and the mask overlay (load_image, load_video), `canvasOnly` previews, plain-object buttons, and latent_preview's `app.canvas._*_callback` forwarding.
- **Prompt-Manager** (21 nodes, V1):
  - `js/ui_palette.js` is a partial token file. Seven files use it, `prompt_browser.js` keeps a drifting copy, and the others hard-code colours.
  - Canvas title badges in PMA, Basic, Extractor and Recipe Builder. Polling inside `onDrawForeground` in the extractor. A custom canvas widget in `recipe_model_loader.js`.
  - Dead files: `prompt_browser_node.js` and `prompt_model_loader.js` target unregistered nodes.
  - `prompt_browser_edit.js` calls `/fbnodes/*` and `/h3refmods/*` without checking that those packs are installed.
- **H3RefModPicker:** its own browser modal, a canvas "?" badge, and the same `widget.draw` hack. No tokens.

## Decisions (defaults; change before Phase 0 if you disagree)
1. **Colour identity:** our own blue family, so these nodes are distinguishable from AusBoss: title `#1b3350`, body `#161b22`, accent `hsl(208 73% 57%)` (the existing PM accent). Every value is one token, so changing it later is a one-file edit.
2. **Dark-only**, like AusBoss. Tokens are emitted as CSS variables, so light mode can be added later.
3. **Pilot first:** build the kit, convert Load Image+, review in both renderers, then do the rest in batches.
4. Stay on the **V1 schema** for existing nodes. The UI work doesn't need V3, and a schema change risks compatibility.

## The shared kit: `js/fbui/` (vendored, identical in each pack)
- **Canonical copy:** `ComfyUI-FBnodes/js/fbui/`. A tiny `tools/sync_fbui.py <pack_dir>` copies it into the other packs. Files are `.js` with no side effects at import (ComfyUI auto-imports every `.js`, and `.mjs` gets no `no-store` header, so the browser served stale copies; see UI_RULES).
- **No collisions between packs:**
  - Each pack serves its own copy from its own URL.
  - `KIT_VERSION` is a constant. CSS is injected as `<style id="fbui-css-v{KIT_VERSION}">`, so identical copies inject once.
  - Classes are prefixed `fbui-`.
  - Setting ids are per pack (`FBnodes.Appearance.*`, `PromptManager.Appearance.*`), because duplicate setting ids throw.
- **Modules.** Start lean; add only when a converted node needs it.
  - `theme.mjs`: `TOKENS` (surfaces, ink, accent, danger, radii, `CONTROL_HEIGHT=26`, font). `ensureCss()` writes `:root`-scoped `--fbui-*` variables and the base classes.
  - `dom.mjs`: `el(tag, cls, props)`, `chainCallback(obj, name, fn)`, `notifyChange(node)`, `toast()` (wraps `app.extensionManager.toast`).
  - `widgets.mjs`: `hideWidget(w)` (the AusBoss recipe: `hidden`, `options.hidden`, `computeSize→[0,-4]`, `computeLayoutSize`, no-op `draw`; widget kept in `node.widgets`), `setWidgetValue(node, name, v)` (sets the value and fires the callback).
  - `controls.mjs`: `scrub({min,max,step,decimals,unit})`, `segment(options)`, `bool()` (off|on pill), `select(options)`, `textField()`, `button(label,{primary,danger,icon})`, `fold(label)`.
  - `card.mjs`: `mountCard(node, {rows, minWidth})` uses the AusBoss row grammar (`{widget,label,kind,suffix,when}`, `{pair:[a,b]}`, `{section}`, `{panel:(node)=>({element,sync}), grow}`, `{group,label}` fold). It builds one `addDOMWidget("fbui_card", …, {serialize:false, hideOnZoom:false})` and resyncs from widgets on `onConfigure`, link changes and callbacks.
  - `layout.mjs`: `fillNodeHeight(widget,{minWidth,minHeight})`, `pinHeight(widget)`, `isVueNodes()` (`LiteGraph.vueNodesMode`), with media placed `position:absolute` inside a stage.
  - `modal.mjs`: `openModal({title, body, footer})`, `confirm()`, `info()`. One z-index scale, one backdrop.
  - `appearance.mjs`: `installAppearance({pack, module, settingsPrefix})`. It colours every node whose `nodeData.python_module` is that pack (so nodes without JS get the colour too). It assigns `undefined` instead of using `delete`. The user's own colour menu choice wins over it. It adds a title-ink wrap for Classic.
  - `media.mjs`: the preview stage plus an "open in player" overlay button. This is a DOM button that replaces the canvas title play icons in both renderers.
- **Per pack:** a single `js/fbui_setup.js` (auto-loaded) calls `installAppearance` with that pack's ids.

## Rules (committed as `docs/UI_RULES.md`, identical in each pack)
1. Node faces are **one fbui card**. No visible native widget rows on a converted node, no plain-object widgets pushed into `node.widgets`, no `onDrawForeground`/`onDrawBackground` UI, no `widget.draw` overrides.
2. Every colour, radius and size comes from `fbui/theme.js` tokens. No hex literals in node files. Node colour is set only by `appearance.mjs`.
3. Controls:
   - INT and FLOAT values use `scrub` with the unit inside.
   - Booleans use an off|on pill.
   - Up to 4 short choices use `segment`; anything longer uses `select`.
   - Buttons use `button()` only.
4. **Compatibility contract:**
   - Never rename, reorder or remove Python inputs.
   - Widgets added in JS are `serialize:false`.
   - Hide widgets; never delete them.
   - Keep `_fbSaveVideoLastResult` as the last serialized widget.
   - Before and after each conversion, load a saved old workflow and confirm `widgets_values` round-trips unchanged.
   - Only if a node's Python inputs truly must change: keep the old class under the old key with `DEPRECATED = True` and display name "… (legacy)", and register the new class under a new key (`LoadImagePlusV2`) with the original display name.
5. **Nodes 2.0:**
   - DOM only.
   - Growing panels use `fillNodeHeight`; fixed panels use `pinHeight`.
   - Set `hideOutputImages` where we draw our own preview.
   - No reaching into `app.canvas._*` internals.
   - Check every node in both renderers.
6. Hooks go through `chainCallback`. Messages go through `toast`, never `alert`. Server calls go through `api.fetchApi`, never raw `fetch`.
7. Namespacing:
   - kit: `fbui-` classes, `fbui_` widget names;
   - pack: extension names `FBnodes.*`, `PromptManager.*`, `H3RefModPicker.*`;
   - routes stay pack-prefixed.
8. Cross-pack calls (for example, PM to `/fbnodes/...`) must handle a 404 gracefully and hide that feature.

## Progress (2026-11-23)
- **Done (FBnodes):**
  - Phase 0: kit, colours, rules and hygiene.
  - Phases 1–2: Load Image+, Load Video+, Load Audio+, Visual LoRA Picker+ and the file browser (Load+ source handling in `js/common/load_plus.js`).
  - Phase 3: Save Image+ (review), Save Video+, Video Review (was LTX Review; shares `js/common/review.js`), Crop Image+.
  - Phase 4: Clip Stitcher, VACE Stitcher and LoRA List+ share `js/common/item_list.js` (LoRA List+ uses the shared file browser); Show as Text, Switch Any and the small nodes (`js/simple_cards.js`) are cards; Load Checkpoint+ / Diffusion Model+ / LoRA+ are cards that keep real paths.
  - Phase 5: the remap summary is a kit modal; the latent preview no longer reaches into `app.canvas._*`.
  - Legacy: Crop Audio+, Load LTX Latent File, LTX Review Preview.
- **Left native on purpose:** Prompt Batcher (three multiline texts; the card has no multiline row), Get Video Components+ and If Any (no settings).
- **Next:** sync the kit into Prompt-Manager, then H3RefModPicker.

## FBnodes phases
**Phase 0: kit and hygiene**
- Add `js/fbui/*`, `js/fbui_setup.js`, `docs/UI_RULES.md`, `tools/sync_fbui.py`.
- Fix the version mismatch (`__init__` 1.5.0 vs `pyproject` 1.4.6).
- Rename the extension `"SwitchAny"` to `"FBnodes.SwitchAny"`.
- Remove the unused `NODE_DISPLAY_NAME_MAPPINGS` in `nodes/load_models_plus.py` and `nodes/visual_lora_picker.py`.
- Remove the stale `BetterImageLoader` match in `load_image.js`.
- Fix the `video_metadata.js` import depth.
- Replace raw `fetch` with `api.fetchApi`.

**Phase 1: pilot Load Image+ (`js/load_image.js`, 3.4k lines)**
- Card rows: Source (file select plus Browse button), a Mask | Crop action row, a preview `panel` (grow) holding the mask overlay as a DOM canvas instead of `onDrawForeground`, and a frame-position scrub that shows only for video.
- Remove the spliced plain-object buttons and the `widget.draw` hack. The play icon becomes `media.mjs`'s overlay button.
- Pull the Load+ scaffold (file combo, browse path, `classifySelection`, drag-drop upload) into `js/common/load_plus.js`, so that Video and Audio reuse it in Phase 2.

**Phase 2: Load Video+, Load Audio+, Visual LoRA Picker+**
- Reuse `load_plus.mjs` and `media.mjs`.
- Drop `canvasOnly`, the `canvas.canvas.title` tooltip hack and the wheel-forwarding copies.

**Phase 3: Save Image+, Save Video+, Crop Image+, Crop Audio+**
- Share the result-cache, restore and Controls fold logic in `js/common/save_result.js`.
- Save Image's compare view becomes a card `panel` hosting its existing canvas in both renderers (generalising the Vue path from commit 3246735), and the separate classic `onDrawForeground` path goes away.

**Phase 4: lists and the rest**
- Merge the Clip Stitcher and VACE Stitcher list UI into `js/common/clip_list.js`.
- LoRA List+ uses `file_browser.js` instead of its own second modal.
- Convert the Video Review (was LTX Review) modal to the Save Image+ review style, Show Text and Switch Any.
- Give the native-only nodes (Load Models+, VACE Options, Apply LoRA+) card faces for uniformity; the colour already applies from Phase 0.

**Phase 5: shared modals**
- `file_browser.js`, `repath_util.js` dialog and `showConfirm` copies are rebuilt on `modal.mjs` and tokens.
- Make latent_preview Vue-safe.

## Later: Prompt-Manager, then H3RefModPicker (same kit, same rules)
- **Prompt-Manager:**
  - Sync `fbui/`.
  - Replace `ui_palette.js` and the `prompt_browser.js` palette copy with tokens.
  - Delete `prompt_browser_node.js` and `prompt_model_loader.js`.
  - Turn the canvas title badges into card chips.
  - Remove the extractor's `onDrawForeground` polling (use widget callbacks).
  - Rebuild the `recipe_model_loader.js` canvas widget as a card.
  - Move the many dialog factories onto `modal.mjs`.
  - Guard cross-pack routes.
  - The big modals (`prompt_browser.js`, PMA) get a token swap first; a structural rewrite comes only where it removes duplication.
- **H3RefModPicker:** sync the kit, rebuild the Visual RefMod Picker and RefMod Create as cards, and turn the "?" badge into a card help chip.

## Verification (each phase)
- **Workflow compatibility:** never read the user's saved workflows, prompt data or anything else in ComfyUI's `user/` folder. Instead:
  - Build synthetic node fixtures in `test/`, taken from each node's Python input list, and assert that `widgets_values` survives configure and serialize unchanged.
  - The user loads their own old workflows in Classic and Nodes 2.0 and reports anything that changed.
- **Visual check:** start ComfyUI and screenshot each converted node in both renderers (Playwright browser tools), comparing against the AusBoss reference: solid body, aligned rows, no stray native widgets, resizing behaves, and zoom below 0.5 doesn't break.
- **Tests:** `node --test test/*.mjs` (update `save_image_frontend.test.mjs` / `latent_preview.test.mjs` as needed) and `pytest test/`.
- **Independence:** run with only FBnodes enabled, then with all three packs. Confirm one `fbui-css` style, no setting-id errors, no console errors.
