# FB UI rules

These rules apply to FBnodes, Prompt Manager and H3RefModPicker. Each pack ships an identical copy of `js/fbui/`; the canonical copy is in FBnodes, and `tools/sync_fbui.py` copies it into the other packs. Every pack still works on its own.

## Node faces
- A converted node shows **one card** (`mountCard` in `fbui/card.js`). No stock widget rows stay visible on it.
- Do not push plain-object widgets into `node.widgets`, do not draw UI in `onDrawForeground`/`onDrawBackground`, and do not override `widget.draw`. These do not render under Nodes 2.0.
- Previews, players and editors are card `panel` rows. A panel that should take the node's spare height uses `grow: true`, and media sits `position:absolute` inside a `.fbui-stage`.
- A stage may add a full-screen view or editor: a `.fbui-stage-tools` button opens it through the kit modal layer — `openViewer` for media, `openModal` from `fbui/modal.js` for a node's own DOM, as Save Image+ in preview mode and the Video Review node do. While it is open the card shows a placeholder in place of the panel, and the same button state closes it.

## Look
- Every colour, radius and size comes from `fbui/theme.js`. Node files contain no hex literals.
- Only `fbui/appearance.js` sets a node's colour. Individual nodes never set `color`/`bgcolor`.
- Controls:
  - INT/FLOAT values use `scrub`, with any unit inside the box.
  - Booleans use an off | on pill.
  - Up to 4 short choices use a segmented pill; longer lists use a select.
  - Buttons use `button()` from `fbui/controls.js`.
- A field that takes tokens or a syntax gets an (i) beside its label (`info:` on the card row) listing what it understands, instead of a long tooltip.
- Toolbars over a preview go in `.fbui-stage-tools`.
- **Media frame.** Pictures and videos sit in a `.fbui-stage`:
  - a neutral slate plate (`T.plate`) with a 1px trim (`T.plateEdge`) and rounded corners;
  - the media is fitted edge to edge, with the plate showing around it.
  - No dark well, no backdrop pattern.
- **Footnote bar.** Every node that shows a picture or video ends its preview with a footnote bar: a card `panel` row of height `CAPTION_HEIGHT` holding a `.fbui-caption`. It uses the same plate and trim as the media frame. This covers Load Image+, Load Video+, Save Image+ and Save Video+.
  - It is always the **last row** of the card, below any frame picker or fold-out section.
  - It states the one size the node outputs: the cropped size when a crop is on, not "before → after".
  - It shows `—` when there is nothing loaded.
  - Never draw these numbers on the picture itself.
- **Frame picker.** Choosing a video frame uses a rail with a white playhead (`rail()` in `fbui/controls.js`) above a line holding a `FRAME` tag, a frame-number scrub, and a readout (`6.88 s · of 240 frames`).
- **Spacing.** Keep the kit's spacing; don't squeeze rows. Buttons in a row with a label use icon-only squares (Browse, flips) so the main control keeps its width.
- Overlays drawn on a picture (crop outline, guides) are dotted while their tool is active.

## Review screens
A run that pauses for a choice (Save Image+ review; Video Review next) uses the same screen:
- It is full screen. The media fill the space as large as possible (`bestGrid` in `js/common/grid_math.js`).
- A small floating panel holds the node's name, the pick count, the timer pill with its bar, and two buttons: **Send** (the picked ones; none picked sends none) and **Skip**.
- A click picks. Space shows one item at full size (←/→ to step, Space or Esc back), and Enter sends.
- Picking pauses the timer. A queue cancel ends the wait at once.

## Workflow compatibility
- Never rename, reorder or remove Python inputs. Append new inputs at the end.
- Widgets added in JS must have `serialize = false`. The frontend skips them when saving `widgets_values`, so adding or removing them never shifts saved values.
- Hide widgets with `hideWidget`; never splice them out of `node.widgets`.
- Prove each conversion with a fixture in `test/` that asserts `widgets_values` survives configure and serialize unchanged. Real saved workflows are checked by the maintainer; tools and agents never read ComfyUI's `user/` folder.
- If a node's Python inputs really must change:
  - Keep the old class under its old key, with `DEPRECATED = True` and the display name `"… (legacy)"`.
  - Register the new class under a new key (such as `LoadImagePlusV2`), using the original display name.

## Nodes 2.0
- Build UI from DOM only.
- Growing panels use `fillNodeHeight`, and fixed panels use `pinHeight` (both in `fbui/layout.js`).
- Set `node.hideOutputImages = true` when the node draws its own preview.
- Do not reach into `app.canvas._*` internals.
- Check every change in both the Classic and Nodes 2.0 renderers.

## Code
- Shared modules (`js/fbui/`, `js/common/`) are `.js`, never `.mjs`. ComfyUI serves `.js` with `Cache-Control: no-store`, but sends `.mjs` with no cache header. The browser then keeps an old copy after an update, and a new module can import a name the stale one lacks, so the whole pack fails to load.
- ComfyUI imports every `.js` in the web folder, so shared modules must do nothing at import time: only exports, with no registration and no DOM work.
- Hook callbacks with `chainCallback`/`chainHandler`; never assign over an existing callback.
- Show messages with `toast()` from `fbui/comfy.js`, never `alert()`. After a user edit, call `notifyChange()` so undo and the dirty flag see it.
- Call server routes with `api.fetchApi`, never raw `fetch`.
- Naming:
  - kit classes start with `fbui-`;
  - kit widget names start with `fbui_`;
  - extension names start with the pack (`FBnodes.*`, `PromptManager.*`, `H3RefModPicker.*`);
  - routes stay prefixed with their pack.
- A call into another pack's routes must handle a 404 and hide the feature.
- Bump `KIT_VERSION` in `fbui/theme.js` whenever the kit CSS changes, then sync the kit to every pack.
