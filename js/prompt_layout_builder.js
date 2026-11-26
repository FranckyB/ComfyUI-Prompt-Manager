/**
 * Prompt Layout Builder: the MOSS-Image layout editor on one fbui card. The
 * output format and canvas size sit at the top, then the collapsible scene
 * group that holds for the whole image, then a stage that grows with its tools
 * along the top and the missing-fields bar along the bottom, then the rows that
 * change per box, with the output size as the last row.
 *
 * The stock widgets stay the source of truth: aspect_ratio, megapixels,
 * use_import_json, output_format and canvas are card rows, and prompt and
 * layout_data are hidden and written by the card. layout_data holds the boxes
 * as canvas fractions {x, y, w, h, type, text, desc, palette} plus the
 * format-specific scene fields, mirroring nodes/prompt_layout_builder.py.
 * Full screen puts all the fields in one column beside the canvas in a kit
 * modal, as Save Image+ does for a preview.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { mountCard } from "./fbui/card.js";
import { notifyChange } from "./fbui/comfy.js";
import { button, helpBadge, segmented, ICONS } from "./fbui/controls.js";
import { chainCallback, el, keepKeyInField } from "./fbui/dom.js";
import { confirm, openModal } from "./fbui/modal.js";
import { CAPTION_HEIGHT, FONT, RADIUS, RADIUS_CARD, RADIUS_XS, ROW_HEIGHT, T, Z_MENU } from "./fbui/theme.js";
import { makeScrub } from "./fbui/scrub.js";
import { findWidget, hideInputsInDef, hideWidget, isWidgetHidden, showWidget, suppressCorePreview, widgetLinked } from "./fbui/widgets.js";
import { DEFAULT_THUMBNAIL } from "./prompt_manager_advanced.js";
import { showThumbnailBrowser } from "./prompt_browser.js";
import { COMPOSER_ENDPOINT_PREFIX, getComposerEntry, loadComposerPrompts } from "./prompt_composer_common.js";

const NODE_NAME = "PromptLayoutBuilder";
const REF_RATIO = "Use Reference Ratio";
const SIZE_MULTIPLE = 16;
const BOX_PALETTE_MAX = 8;
const SUMMARY_PALETTE_MAX = 16;
const HISTORY_LIMIT = 50;
const MIN_NODE_WIDTH = 560;
const INITIAL_NODE_HEIGHT = 960;
const STAGE_MIN_HEIGHT = 280;
// The tools ride inside the top of the stage, so the image area starts below them.
const TOOLBAR_HEIGHT = ROW_HEIGHT + 8;
const TEXT_PANEL_HEIGHT = 58;
const DESC_PANEL_HEIGHT = 72;
const SCENE_PANEL_HEIGHT = 84;
const PALETTE_PANEL_HEIGHT = 30;
const HANDLE_SIZE = 8;
const MIN_BOX = 0.01;
const BG_PROP = "pm_layout_background";
const REF_SIZE_PROP = "pm_layout_ref_size";
const TEXT_SIZE_PROP = "pm_layout_text_size";
const TEXT_SIZES = { hidden: 0, small: 9, medium: 11, large: 14, larger: 18 };
const FULLSCREEN_TEXT_SIZES = { hidden: 0, small: 11, medium: 14, large: 18, larger: 23 };
const TEXT_SIZE_CHOICES = [["hidden", "Hidden"], ["small", "Small"], ["medium", "Medium"], ["large", "Large"], ["larger", "Largest"]];
const DISPLAY_FONT = '"Futura", "Century Gothic", "Avenir Next", "Trebuchet MS", sans-serif';
const BODY_FONT = '"Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif';
const CSS_ID = "pm-layout-css-v17";
const DEFAULT_COLOR = "#FFFFFF";
const HIDDEN_WIDGETS = ["aspect_ratio", "megapixels", "prompt", "meta", "canvas", "use_import_json", "layout_data", "output_format"];
const OUTPUT_TITLES = {
    MOSS: "MOSS-Image: layout JSON with boxes in canvas pixels.",
    Ideogram: "Ideogram: 0-1000 caption elements and a high-level description.",
    "Krea 2": "Krea 2: grounding lines with 0-1000 boxes, x first.",
};
const GUIDE = [
    "Drag on the canvas to draw a region, or press + Box. Drag a box to move it, its handles to resize it, Delete to remove it.",
    "Import takes the layout from import_json: the editor then shows the imported result instead of writing one.",
    "Text size and Grid only change what you see here, never the output.",
    "Ctrl/Cmd+Z undoes an edit, Ctrl/Cmd+Shift+Z redoes it, Ctrl/Cmd+D duplicates the selected box.",
    "Full screen moves the whole editor into a dialog; Esc brings it back to the node.",
];
const BOX_TYPES = [
    { value: "object", label: "Object", tag: "OBJ" },
    { value: "character", label: "Character", tag: "CHR" },
    { value: "text", label: "Text", tag: "TXT" },
    { value: "panel", label: "Panel", tag: "PNL" },
];
const KREA_TAGS = [["pe", "Person"], ["ac", "Anime character"], ["fc", "Furry / anthro"]];
const IDEOGRAM_STYLES = [["none", "None"], ["photo", "Photo"], ["art_style", "Art style"]];
const IDEOGRAM_DETAIL = {
    none: { placeholder: "Nothing extra beyond the caption" },
    photo: { placeholder: "35mm, f/1.4, shallow depth of field" },
    art_style: { placeholder: "flat vector illustration, bold outlines" },
};
const BOX_HUES = [205, 40, 92, 8, 248, 155, 332];
const HEX_COLOR = /^#[0-9a-f]{6}$/i;
const RESIZE_CURSORS = { nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize", n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize" };

const pad2 = (n) => String(n).padStart(2, "0");
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const boxHue = (index) => BOX_HUES[index % BOX_HUES.length];
const typeInfo = (type) => BOX_TYPES.find((t) => t.value === type) || BOX_TYPES[0];

// A box takes its first palette color, otherwise a Prompt Composer accent by index.
function boxRgb(box, index) {
    const color = box.palette?.[0];
    if (HEX_COLOR.test(color || "")) return [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16));
    const h = boxHue(index);
    const lightness = 0.62;
    const a = 0.85 * Math.min(lightness, 1 - lightness);
    const channel = (n) => {
        const k = (n + h / 30) % 12;
        return lightness - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    };
    return [0, 8, 4].map((n) => Math.round(channel(n) * 255));
}

const rgba = ([r, g, b], alpha = 1) => `rgba(${r}, ${g}, ${b}, ${alpha})`;

function hsvToHex(h, s, v) {
    const channel = (n) => {
        const k = (n + h / 60) % 6;
        return v - v * s * Math.max(0, Math.min(k, 4 - k, 1));
    };
    return `#${[5, 3, 1].map((n) => Math.round(channel(n) * 255).toString(16).padStart(2, "0")).join("")}`.toUpperCase();
}

function hexToHsv(hex) {
    const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
    const max = Math.max(r, g, b);
    const delta = max - Math.min(r, g, b);
    const hue = !delta ? 0 : max === r ? ((g - b) / delta + 6) % 6 : max === g ? (b - r) / delta + 2 : (r - g) / delta + 4;
    return [hue * 60, max ? delta / max : 0, max];
}

function formatColor(hex, format) {
    return format === "rgb" ? [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)).join(", ") : hex;
}

function parseColor(text) {
    const value = String(text || "").trim();
    const hex = value.startsWith("#") ? value : `#${value}`;
    if (HEX_COLOR.test(hex)) return hex.toUpperCase();
    const rgb = value.split(/[\s,]+/).map(Number);
    if (rgb.length !== 3 || !rgb.every((c) => Number.isInteger(c) && c >= 0 && c <= 255)) return null;
    return `#${rgb.map((c) => c.toString(16).padStart(2, "0")).join("")}`.toUpperCase();
}

// Mirrors _resolution() in nodes/prompt_layout_builder.py.
function canvasSize(node) {
    const ratioName = String(findWidget(node, "aspect_ratio")?.value || "");
    const ratio = ratioName === REF_RATIO
        ? (node.properties?.[REF_SIZE_PROP] || [1, 1])
        : (ratioName.match(/^(\d+):(\d+)/)?.slice(1).map(Number) || [1, 1]);
    const megapixels = Number(findWidget(node, "megapixels")?.value) || 1;
    const scale = Math.sqrt(megapixels * 1024 * 1024 / (ratio[0] * ratio[1]));
    return ratio.map((r) => Math.max(SIZE_MULTIPLE, Math.round(r * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE));
}

function readState(node) {
    try {
        const state = JSON.parse(findWidget(node, "layout_data")?.value || "{}");
        return {
            boxes: Array.isArray(state.boxes) ? state.boxes : [],
            palette: Array.isArray(state.palette) ? state.palette : [],
            ideogram: state.ideogram && typeof state.ideogram === "object" ? state.ideogram : {},
            krea: state.krea && typeof state.krea === "object" ? state.krea : {},
        };
    } catch {
        return { boxes: [], palette: [], ideogram: {}, krea: {} };
    }
}

function boxLabel(box, index) {
    const summary = String(box.type === "text" && box.text ? `"${box.text}"` : box.desc || "").replace(/\s+/g, " ").trim();
    const head = `${pad2(index + 1)} ${typeInfo(box.type).label}`;
    return summary ? `${head} — ${summary.length > 48 ? `${summary.slice(0, 47)}…` : summary}` : head;
}

// Only the pieces the kit has no class for: the notes, the swatches, the colour
// editor and the full-screen card. Every colour is a token.
function ensureCss() {
    if (document.getElementById(CSS_ID)) return;
    const style = el("style");
    style.id = CSS_ID;
    style.textContent = `
.pm-layout-line{display:flex;align-items:center;gap:6px;width:100%}
.pm-layout-line>*{min-width:0}
.pm-layout-bar{position:absolute;top:0;left:0;right:0;z-index:1;height:${TOOLBAR_HEIGHT}px;padding:4px 6px;background:${T.panel};border-bottom:1px solid ${T.plateEdge}}
.fbui-stage.pm-layout-stage>canvas.fbui-fit{top:${TOOLBAR_HEIGHT}px;height:calc(100% - ${TOOLBAR_HEIGHT + 1}px - var(--pm-layout-warning,0px))}
.fbui-stage.pm-layout-stage>.fbui-stage-empty{top:${TOOLBAR_HEIGHT}px;bottom:var(--pm-layout-warning,0px)}
.pm-layout-hint{position:absolute;left:0;right:0;bottom:0;z-index:1;box-sizing:border-box;height:${CAPTION_HEIGHT}px;display:flex;align-items:center;gap:6px;padding:0 8px;background:${T.dangerSoft};color:${T.dangerInk};font:600 11px ${FONT}}
.pm-layout-note{display:flex;flex-direction:column;gap:4px;height:100%;box-sizing:border-box}
.pm-layout-note>span{flex:none;color:${T.faint};font:600 10px/1 ${FONT};letter-spacing:.08em;text-transform:uppercase}
.pm-layout-note textarea.fbui-text{flex:1 1 auto;height:auto;min-height:0;resize:none;pointer-events:auto}
.pm-layout-desc{display:flex;gap:6px;flex:1 1 auto;min-height:0}
.pm-layout-thumb{flex:none;width:46px;padding:0;border:1px solid ${T.plateEdge};border-radius:${RADIUS_XS}px;background:${T.card};color:${T.faint};font:600 15px ${FONT};overflow:hidden;cursor:pointer}
.pm-layout-thumb:hover{border-color:${T.brand}}
.pm-layout-thumb img,.pm-layout-thumb video{display:block;width:100%;height:100%;object-fit:cover}
.pm-layout-palette{display:flex;align-items:center;gap:8px;height:100%}
.pm-layout-palette>span{flex:none;color:${T.faint};font:600 10px/1 ${FONT};letter-spacing:.08em;text-transform:uppercase}
.pm-layout-swatches{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.pm-layout-swatch{flex:none;width:22px;height:22px;padding:0;border:1px solid ${T.plateEdge};border-radius:${RADIUS_XS}px;background:${T.fill};color:${T.faint};font:600 13px ${FONT};cursor:pointer}
.pm-layout-swatch:hover{border-color:${T.brand}}
.pm-layout-swatch:disabled{opacity:.4;cursor:default}
.pm-layout-mini{flex:none;display:flex;align-items:center;color:${T.faint};font:600 10px ${FONT};letter-spacing:.06em;text-transform:uppercase}
.pm-layout-mini svg{display:block}
.fbui-row.pm-layout-locked{opacity:.5}
.fbui-row.pm-layout-locked>:not(.fbui-label){pointer-events:none}
.pm-layout-held{flex:1 1 auto;display:grid;place-items:center;border:1px dashed ${T.plateEdge};border-radius:${RADIUS_CARD}px;color:${T.faint};font:11px ${FONT}}
.pm-layout-modal .fbui-modal-body{display:flex;padding:12px}
.pm-layout-spread{display:grid;grid-template-columns:minmax(0,1fr) minmax(360px,30%);gap:12px;flex:1 1 auto;min-width:0;min-height:0}
.pm-layout-col{min-width:0;overflow:auto}
.pm-layout-fields{display:flex;flex-direction:column;gap:12px;min-width:0;min-height:0;overflow:auto}
.pm-layout-fields>.fbui-card{flex:none;height:auto;border:1px solid ${T.plateEdge}}
.fbui-card:has(.pm-layout-held)>.fbui-group{display:none}
.pm-layout-picker{position:fixed;z-index:${Z_MENU};width:236px;box-sizing:border-box;padding:8px;display:flex;flex-direction:column;gap:8px;border:1px solid ${T.plateEdge};border-radius:${RADIUS}px;background:${T.panel};font:12px ${FONT};color:${T.ink}}
.pm-layout-picker-area{position:relative;height:120px;border-radius:${RADIUS_XS}px;cursor:crosshair;touch-action:none}
.pm-layout-picker-hue{position:relative;height:10px;border-radius:999px;background:linear-gradient(to right,hsl(0 100% 50%),hsl(60 100% 50%),hsl(120 100% 50%),hsl(180 100% 50%),hsl(240 100% 50%),hsl(300 100% 50%),hsl(360 100% 50%));cursor:crosshair;touch-action:none}
.pm-layout-picker-thumb{position:absolute;width:10px;height:10px;margin:-5px;box-sizing:border-box;border:1px solid ${T.ink};border-radius:50%;pointer-events:none}
`;
    document.head.append(style);
}

// ------------------------------------------------------------------- state

const layout = (node) => node._layout;
const snapshot = (node) => JSON.stringify(layout(node).state);
const isLocked = (node) => !!findWidget(node, "use_import_json")?.value;
const selectedBox = (node) => layout(node).state.boxes[layout(node).selected];
const hasSelection = (node) => layout(node).selected >= 0 && !!selectedBox(node);

function outputFormat(node) {
    const widget = findWidget(node, "output_format");
    const value = String(widget?.value || "MOSS");
    if (!OUTPUT_TITLES[value]) {
        if (widget) widget.value = "MOSS";
        return "MOSS";
    }
    return value;
}

// Row predicates, evaluated against the card's widget values.
const isIdeogram = (values) => values.output_format === "Ideogram";
const isKrea = (values) => values.output_format === "Krea 2";
const hasBoxes = (values, node) => !values.use_import_json && node._layout.state.boxes.length > 0;
const boxPicked = (values, node) => !values.use_import_json && node._layout.selected >= 0 && !!node._layout.state.boxes[node._layout.selected];
const textPicked = (values, node) => boxPicked(values, node) && node._layout.state.boxes[node._layout.selected].type === "text";
const scenePaletteShown = (values) => values.output_format !== "Krea 2";
const kreaCharacter = (values, node) => boxPicked(values, node) && values.output_format === "Krea 2" && node._layout.state.boxes[node._layout.selected].type === "character";

function setWidgetValue(node, name, value) {
    const widget = findWidget(node, name);
    if (!widget) return;
    widget.value = value;
    node.graph?.setDirtyCanvas(true, true);
}

function persist(node) {
    setWidgetValue(node, "layout_data", JSON.stringify(layout(node).state));
}

function schedulePaint(node) {
    const state = layout(node);
    if (state.frame === null) state.frame = requestAnimationFrame(() => paint(node));
}

// import_json drives the layout and the scene text, so the rows that would edit
// them stay on show with the imported values and go inert, as they were greyed
// out in the stock editor. The imported boxes are not ours to change, so the
// selection drops with the lock.
function applyLock(node) {
    const state = layout(node);
    const locked = isLocked(node);
    if (locked) state.selected = -1;
    for (const entry of state.card.entries) entry.element.classList.toggle("pm-layout-locked", locked && Boolean(entry.row.lock));
}

// Rows come and go with the selected box and the output format, so the node
// follows the card the same way it does for a fold opened by hand.
function repaint(node) {
    applyLock(node);
    layout(node).card?.refreshByHand();
    schedulePaint(node);
}

function pushUndo(node, before) {
    const state = layout(node);
    state.undo.push(before);
    if (state.undo.length > HISTORY_LIMIT) state.undo.shift();
    state.redo = [];
}

function editState(node, mutate) {
    if (isLocked(node)) return;
    const before = snapshot(node);
    mutate();
    if (snapshot(node) === before) return;
    pushUndo(node, before);
    persist(node);
    repaint(node);
}

function travel(node, fromName, toName) {
    const state = layout(node);
    const from = state[fromName];
    if (!from.length || isLocked(node)) return;
    state[toName].push(snapshot(node));
    state.state = JSON.parse(from.pop());
    state.selected = Math.min(state.selected, state.state.boxes.length - 1);
    persist(node);
    repaint(node);
}

// Format-specific scene fields live in layout_data; typing does not add an undo
// step, as for the scene prompt.
function setSceneField(node, group, key, value) {
    if (isLocked(node)) return;
    const state = layout(node).state;
    state[group] = { ...(state[group] ?? {}), [key]: value };
    persist(node);
}

const textSizeKey = (node) => (node.properties?.[TEXT_SIZE_PROP] in TEXT_SIZES ? node.properties[TEXT_SIZE_PROP] : "medium");

// A field typed rather than picked: one undo step per focus session.
function bindLive(node, input, apply) {
    input.spellcheck = false;
    input.addEventListener("pointerdown", (event) => event.stopPropagation());
    input.addEventListener("keydown", keepKeyInField);
    input.addEventListener("focus", () => { layout(node).pendingSnapshot = snapshot(node); });
    input.addEventListener("input", () => {
        const state = layout(node);
        if (state.selected < 0 || isLocked(node)) return;
        if (state.pendingSnapshot !== null) {
            pushUndo(node, state.pendingSnapshot);
            state.pendingSnapshot = null;
        }
        apply(state.state.boxes[state.selected], input.value);
        persist(node);
        syncCaption(node);
        schedulePaint(node);
    });
    input.addEventListener("change", () => {
        notifyChange();
        repaint(node);
    });
}

function note(label, control) {
    const box = el("div", "pm-layout-note");
    const caption = el("span", "", label);
    box.append(caption, control);
    return { box, caption };
}

// Avoid resetting the caret of the field being typed in.
function setValue(input, value) {
    if (document.activeElement !== input && input.value !== value) input.value = value;
}

// -------------------------------------------------------------------- boxes

function newBox(rect) {
    return { ...rect, type: "object", text: "", desc: "", palette: [] };
}

function addBox(node) {
    editState(node, () => {
        const state = layout(node);
        const offset = (state.state.boxes.length % 6) * 0.025;
        state.state.boxes.push(newBox({ x: 0.3 + offset, y: 0.3 + offset, w: 0.3, h: 0.3 }));
        state.selected = state.state.boxes.length - 1;
    });
}

function clearBoxes(node) {
    editState(node, () => {
        layout(node).state.boxes = [];
        layout(node).selected = -1;
    });
}

function duplicateSelected(node) {
    if (!hasSelection(node)) return;
    editState(node, () => {
        const state = layout(node);
        const copy = structuredClone(state.state.boxes[state.selected]);
        copy.x = clamp(copy.x + 0.02, 0, 1 - copy.w);
        copy.y = clamp(copy.y + 0.02, 0, 1 - copy.h);
        state.state.boxes.splice(state.selected + 1, 0, copy);
        state.selected += 1;
    });
}

function deleteSelected(node) {
    if (!hasSelection(node)) return;
    editState(node, () => {
        const state = layout(node);
        state.state.boxes.splice(state.selected, 1);
        state.selected = Math.min(state.selected, state.state.boxes.length - 1);
    });
}

// index: 0 x, 1 y, 2 w, 3 h, reported in canvas pixels.
function applyCoord(node, index, value) {
    const box = selectedBox(node);
    if (!box || !Number.isFinite(value)) return;
    const [W, H] = canvasSize(node);
    const px = [box.x * W, box.y * H, box.w * W, box.h * H];
    px[index] = value;
    const w = clamp(px[2] / W, MIN_BOX, 1);
    const h = clamp(px[3] / H, MIN_BOX, 1);
    Object.assign(box, { x: clamp(px[0] / W, 0, 1 - w), y: clamp(px[1] / H, 0, 1 - h), w, h });
}

// A scrub reports every step, so the undo entry is taken when it settles.
function liveCoord(node, index, value) {
    const state = layout(node);
    if (state.selected < 0 || isLocked(node)) return;
    state.scrubBefore ??= snapshot(node);
    applyCoord(node, index, value);
    persist(node);
    syncCaption(node);
    schedulePaint(node);
}

function endCoord(node) {
    const state = layout(node);
    const before = state.scrubBefore;
    state.scrubBefore = null;
    if (before === null) return;
    if (before !== snapshot(node)) {
        pushUndo(node, before);
        notifyChange();
    }
    repaint(node);
}

async function pickComposerPrompt(node) {
    const state = layout(node);
    if (!hasSelection(node) || isLocked(node)) return;
    const index = state.selected;
    const ref = state.state.boxes[index].prompt_ref;
    state.composerLoad ??= loadComposerPrompts(node);
    await state.composerLoad;
    const category = ref?.category || Object.keys(node.prompts || {}).find((key) => key !== "__meta__") || "";
    const selection = await showThumbnailBrowser(node, category, ref?.name || "", {
        title: "Select Prompt Snippet for Box",
        endpointPrefix: COMPOSER_ENDPOINT_PREFIX,
        promptOnly: true,
        loadPromptsFn: loadComposerPrompts,
        preferenceScope: "composer",
        allowMultiSelect: false,
    });
    const entry = selection?.prompt ? getComposerEntry(node, selection.category, selection.prompt) : null;
    if (!entry || !state.state.boxes[index]) return;
    state.selected = index;
    editState(node, () => {
        const box = state.state.boxes[index];
        box.desc = String(entry.prompt || "");
        box.prompt_ref = { category: selection.category, name: selection.prompt };
    });
}

function renderThumb(node, box) {
    const state = layout(node);
    if (!state.thumb) return;
    const ref = box.prompt_ref;
    state.thumb.replaceChildren();
    if (!ref) {
        state.thumb.textContent = "+";
        state.thumb.title = "Use a Prompt Composer prompt as the description";
        return;
    }
    state.thumb.title = `${ref.name}: click to pick another prompt`;
    if (!node.prompts) {
        state.composerLoad ??= loadComposerPrompts(node);
        state.composerLoad.then(() => repaint(node));
        return;
    }
    const src = getComposerEntry(node, ref.category, ref.name)?.thumbnail || DEFAULT_THUMBNAIL;
    const media = el(String(src).startsWith("data:video") ? "video" : "img");
    media.muted = true;
    media.src = src;
    state.thumb.append(media);
}

// ------------------------------------------------------------------- colour

let activePicker = null;

function closeColorPicker() {
    if (!activePicker) return;
    const { element, cleanup, onClose } = activePicker;
    activePicker = null;
    cleanup();
    element.remove();
    onClose();
}

function trackPointer(element, onMove) {
    const move = (event) => {
        const rect = element.getBoundingClientRect();
        onMove(clamp((event.clientX - rect.left) / rect.width, 0, 1), clamp((event.clientY - rect.top) / rect.height, 0, 1));
    };
    element.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        element.setPointerCapture(event.pointerId);
        move(event);
    });
    element.addEventListener("pointermove", (event) => {
        if (element.hasPointerCapture(event.pointerId)) move(event);
    });
}

function openColorPicker(anchor, color, { onInput, onRemove, onClose }) {
    closeColorPicker();
    let [hue, saturation, value] = hexToHsv(color);
    const picker = el("div", "pm-layout-picker");
    document.body.append(picker);
    const area = el("div", "pm-layout-picker-area");
    const areaThumb = el("div", "pm-layout-picker-thumb", area);
    const hueBar = el("div", "pm-layout-picker-hue");
    const hueThumb = el("div", "pm-layout-picker-thumb", hueBar);
    const format = el("select", "fbui-select");
    format.style.cssText = "flex:none;width:70px";
    format.append(new Option("Hex", "hex"), new Option("RGB", "rgb"));
    const input = el("input", "fbui-text");
    input.spellcheck = false;
    input.style.flex = "1 1 0";
    const row = el("div", "pm-layout-line");
    row.append(format, input);
    const remove = button("Remove color", { title: "Remove this color from the palette", onClick: () => { onRemove(); closeColorPicker(); } });
    picker.append(area, hueBar, row, remove);

    const update = (emit) => {
        const hex = hsvToHex(hue, saturation, value);
        area.style.background = `linear-gradient(to top, black, transparent), linear-gradient(to right, white, transparent), hsl(${hue} 100% 50%)`;
        areaThumb.style.left = `${saturation * 100}%`;
        areaThumb.style.top = `${(1 - value) * 100}%`;
        areaThumb.style.background = hex;
        hueThumb.style.left = `${(hue / 360) * 100}%`;
        setValue(input, formatColor(hex, format.value));
        if (emit) onInput(hex);
    };
    trackPointer(area, (x, y) => { saturation = x; value = 1 - y; update(true); });
    trackPointer(hueBar, (x) => { hue = x * 360; update(true); });
    input.addEventListener("keydown", keepKeyInField);
    format.addEventListener("change", () => { input.value = formatColor(hsvToHex(hue, saturation, value), format.value); });
    input.addEventListener("change", () => {
        const parsed = parseColor(input.value);
        if (parsed) [hue, saturation, value] = hexToHsv(parsed);
        input.blur();
        update(!!parsed);
    });

    const rect = anchor.getBoundingClientRect();
    picker.style.left = `${clamp(rect.left, 8, window.innerWidth - 244)}px`;
    picker.style.top = `${rect.bottom + 240 < window.innerHeight ? rect.bottom + 6 : Math.max(8, rect.top - 236)}px`;

    const onPointer = (event) => { if (!picker.contains(event.target)) closeColorPicker(); };
    const onKey = (event) => {
        if (event.key !== "Escape") return;
        event.preventDefault();
        event.stopPropagation();
        closeColorPicker();
    };
    document.addEventListener("pointerdown", onPointer, true);
    window.addEventListener("keydown", onKey, true);
    activePicker = {
        element: picker,
        anchor,
        onClose,
        cleanup: () => {
            document.removeEventListener("pointerdown", onPointer, true);
            window.removeEventListener("keydown", onKey, true);
        },
    };
    update(false);
}

// A palette row: click a swatch to edit it, right-click to remove it. Edits are
// committed once, when the picker closes.
function createPaletteField(labelText, limit, onCommit) {
    const box = el("div", "pm-layout-palette");
    const caption = el("span", "", labelText);
    const swatches = el("div", "pm-layout-swatches");
    box.append(caption, swatches);
    let colors = [];

    const open = (index, target) => openColorPicker(target, colors[index], {
        onInput: (hex) => {
            colors[index] = hex;
            target.style.background = hex;
            target.title = hex;
        },
        onRemove: () => { colors.splice(index, 1); },
        onClose: () => onCommit([...colors]),
    });

    const draw = () => {
        const items = colors.map((color, index) => {
            const swatch = el("button", "pm-layout-swatch");
            swatch.type = "button";
            swatch.style.background = color;
            swatch.title = `${color} (right-click to remove)`;
            swatch.addEventListener("click", () => open(index, swatch));
            swatch.addEventListener("contextmenu", (event) => {
                event.preventDefault();
                event.stopPropagation();
                colors.splice(index, 1);
                onCommit([...colors]);
            });
            return swatch;
        });
        const add = el("button", "pm-layout-swatch");
        add.type = "button";
        add.textContent = "+";
        add.title = "Add a color";
        add.disabled = colors.length >= limit;
        add.addEventListener("click", () => {
            colors.push(DEFAULT_COLOR);
            draw();
            open(colors.length - 1, swatches.children[colors.length - 1]);
        });
        swatches.replaceChildren(...items, add);
    };

    return {
        element: box,
        set(next) {
            if (activePicker && swatches.contains(activePicker.anchor)) return;
            colors = [...next];
            draw();
        },
    };
}

// ------------------------------------------------------------------ controls

// A pill whose choices can change between refreshes: rebuilt when they do.
function choicesRow(node, choices, onPick, current) {
    let shown = null;
    let pill = null;
    const row = el("div", "pm-layout-line");
    return {
        element: row,
        sync() {
            const list = choices(node);
            const key = list.map((choice) => choice.value).join("\n");
            if (key !== shown) {
                shown = key;
                const next = segmented(list, { onPick: (value) => onPick(node, value) });
                if (pill) pill.root.replaceWith(next.root);
                else row.append(next.root);
                pill = next;
            }
            pill.set(current(node));
        },
    };
}

function navControl(node) {
    const state = layout(node);
    const select = el("select", "fbui-select");
    select.style.flex = "1 1 0";
    select.title = "Jump to a box";
    select.addEventListener("change", () => {
        state.selected = Number(select.value);
        repaint(node);
    });
    const step = (delta) => {
        state.selected = clamp(state.selected + delta, 0, state.state.boxes.length - 1);
        repaint(node);
    };
    const prev = button("", { icon: ICONS.chevronLeft, title: "Previous box", onClick: () => step(-1) });
    const next = button("", { icon: ICONS.chevronRight, title: "Next box", onClick: () => step(1) });
    prev.style.flex = next.style.flex = "none";
    const row = el("div", "pm-layout-line");
    row.append(select, prev, next);
    return {
        element: row,
        sync() {
            const boxes = state.state.boxes;
            if (document.activeElement !== select) {
                const options = boxes.map((box, index) => new Option(boxLabel(box, index), String(index)));
                options.unshift(new Option("00 No Selection", "-1"));
                select.replaceChildren(...options);
                select.value = String(state.selected);
            }
            // Ring the picker in the colour of the box it points at, so the row
            // you are editing is obvious from across the node.
            const tint = state.selected >= 0 ? rgba(boxRgb(boxes[state.selected], state.selected)) : "";
            select.style.borderColor = tint;
            select.style.boxShadow = tint ? `inset 0 0 0 1px ${tint}` : "";
            prev.disabled = state.selected < 1;
            next.disabled = state.selected >= boxes.length - 1;
        },
    };
}

function typeControl(node) {
    return choicesRow(node, () => BOX_TYPES
        // Panels only exist in Krea 2; elsewhere they export as objects.
        .filter((type) => type.value !== "panel" || outputFormat(node) === "Krea 2" || selectedBox(node)?.type === "panel")
        .map((type) => ({ value: type.value, label: type.label, title: `Label this box as ${type.value}` })),
    (node, value) => editState(node, () => { selectedBox(node).type = value; }),
    (node) => selectedBox(node)?.type);
}

// One line for the whole region: x y w h, then the two things that act on it.
function coordControl(node) {
    const state = layout(node);
    const names = ["x", "y", "w", "h"];
    const axes = names.map((name, index) => makeScrub({
        value: 0,
        min: 0,
        max: canvasSize(node)[index % 2],
        step: 1,
        decimals: 0,
        unit: "px",
        title: `${name} in canvas pixels`,
        onChange: (value) => liveCoord(node, index, value),
        onSettle: () => endCoord(node),
    }));
    for (const scrub of axes) scrub.root.style.flex = "1 1 0";
    const duplicate = button("", { icon: ICONS.tiles, title: "Duplicate this box (Ctrl/Cmd+D)", onClick: () => duplicateSelected(node) });
    const remove = button("", { icon: ICONS.trash, title: "Delete this box (Delete)", onClick: () => deleteSelected(node) });
    const line = el("div", "pm-layout-line");
    names.forEach((name, index) => line.append(el("span", "pm-layout-mini", name), axes[index].root));
    line.append(duplicate, remove);
    return {
        element: line,
        sync() {
            const size = canvasSize(node);
            const box = state.state.boxes[state.selected];
            const px = box ? [box.x * size[0], box.y * size[1], box.w * size[0], box.h * size[1]] : [0, 0, 0, 0];
            axes.forEach((scrub, index) => {
                scrub.setRange({ max: size[index % 2] });
                scrub.set(Math.round(px[index]));
            });
            duplicate.disabled = remove.disabled = !hasSelection(node);
        },
    };
}

function kreaTagControl(node) {
    const select = el("select", "fbui-select");
    select.style.cssText = "flex:none;width:46%";
    for (const [value, label] of KREA_TAGS) select.append(new Option(label, value));
    select.title = "Krea 2 character tag";
    select.addEventListener("change", () => editState(node, () => { selectedBox(node).krea_tag = select.value; }));
    const input = el("input", "fbui-text");
    input.placeholder = "1";
    input.title = "Reuse the same ID for the same character across panels";
    input.spellcheck = false;
    input.style.flex = "1 1 0";
    input.addEventListener("pointerdown", (event) => event.stopPropagation());
    input.addEventListener("keydown", keepKeyInField);
    input.addEventListener("change", () => {
        const value = input.value.trim().toLowerCase();
        editState(node, () => {
            const box = selectedBox(node);
            if (value) box.char_id = value;
            else delete box.char_id;
        });
    });
    const row = el("div", "pm-layout-line");
    row.append(select, input);
    return {
        element: row,
        sync() {
            const box = selectedBox(node);
            if (!box) return;
            select.value = KREA_TAGS.some(([value]) => value === box.krea_tag) ? box.krea_tag : "pe";
            setValue(input, String(box.char_id ?? ""));
        },
    };
}

// ------------------------------------------------------------------- panels

// The canvas tools ride along the top of the frame: what to add, what the view
// shows, and how to get the editor out of the way.
function toolStrip(node) {
    const state = layout(node);
    const add = button("+ Box", { title: "Add a box, or drag on the canvas to draw one", onClick: () => addBox(node) });
    const grid = button("Grid", { title: "Show a 10x10 grid over the canvas", onClick: () => { state.grid = !state.grid; schedulePaint(node); } });
    const sizes = el("select", "fbui-select");
    sizes.title = "Size of the text drawn inside the boxes";
    sizes.style.cssText = "min-width:0;width:96px";
    const pen = el("span", "pm-layout-mini");
    pen.title = "Text size";
    pen.innerHTML = ICONS.pencil;
    for (const [value, label] of TEXT_SIZE_CHOICES) sizes.append(new Option(label, value));
    sizes.addEventListener("change", () => {
        node.properties[TEXT_SIZE_PROP] = sizes.value;
        schedulePaint(node);
    });
    const undo = button("", { icon: ICONS.undo, title: "Undo (Ctrl/Cmd+Z)", onClick: () => travel(node, "undo", "redo") });
    const redo = button("", { icon: ICONS.redo, title: "Redo (Ctrl/Cmd+Shift+Z)", onClick: () => travel(node, "redo", "undo") });
    const clear = button("Clear", {
        title: "Remove every box (asks first)",
        onClick: async () => {
            if (!(await confirm({ title: "Clear all boxes", message: "Remove every box from the layout? Ctrl/Cmd+Z brings them back.", confirmText: "Clear all", danger: true }))) return;
            clearBoxes(node);
        },
    });
    const expand = button("Full screen", { icon: ICONS.expand, title: "Open the editor in full screen", onClick: () => (state.fullscreen ? closeFullscreen(node) : openFullscreen(node)) });
    expand.style.marginLeft = "auto";
    const line = el("div", "pm-layout-line pm-layout-bar");
    line.append(add, grid, pen, sizes, undo, redo, clear, expand);
    return {
        element: line,
        sync() {
            const locked = isLocked(node);
            add.disabled = clear.disabled = undo.disabled = redo.disabled = locked;
            clear.disabled ||= !state.state.boxes.length;
            undo.disabled ||= state.undo.length === 0;
            redo.disabled ||= state.redo.length === 0;
            grid.classList.toggle("on", state.grid);
            if (document.activeElement !== sizes) sizes.value = textSizeKey(node);
            expand.classList.toggle("on", Boolean(state.fullscreen));
            expand.title = state.fullscreen ? "Back to the node (Esc)" : "Open the editor in full screen";
        },
    };
}

function stagePanel(node) {
    const state = layout(node);
    const stage = el("div", "fbui-stage pm-layout-stage");
    stage.tabIndex = 0;
    const tools = toolStrip(node);
    const canvas = el("canvas", "fbui-fit");
    const empty = el("div", "fbui-stage-empty");
    const warning = el("div", "pm-layout-hint");
    // The tools and the missing-fields bar ride inside the frame, so the image
    // area is the space between them.
    stage.append(tools.element, canvas, empty, warning);
    state.canvas = canvas;
    bindStage(node, stage, canvas);
    bindShortcuts(node, stage);
    const observer = new ResizeObserver(() => schedulePaint(node));
    observer.observe(canvas);
    chainCallback(node, "onRemoved", () => observer.disconnect());
    return {
        element: stage,
        sync() {
            tools.sync();
            const missing = missingNote(node);
            warning.textContent = missing;
            warning.hidden = !missing;
            stage.style.setProperty("--pm-layout-warning", missing ? `${CAPTION_HEIGHT}px` : "0px");
            empty.hidden = Boolean(state.state.boxes.length);
            empty.textContent = isLocked(node)
                ? "Driven by import_json. Turn off Use import_json to edit the layout."
                : "Drag on the canvas to draw a box";
        },
    };
}

function syncCaption(node) {
    const state = layout(node);
    if (!state.caption) return;
    const count = state.state.boxes.length;
    const size = canvasSize(node).join(" × ");
    state.caption.textContent = `${size} · ${count} box${count === 1 ? "" : "es"}`;
}

function captionPanel(node) {
    const caption = el("div", "fbui-caption");
    layout(node).caption = caption;
    return { element: caption, sync: () => syncCaption(node) };
}

function boxTextPanel(node) {
    const area = el("textarea", "fbui-text");
    area.placeholder = "Exact text to render in this region";
    bindLive(node, area, (box, value) => { box.text = value; });
    return {
        element: note("Text content", area).box,
        sync: () => setValue(area, selectedBox(node)?.text ?? ""),
    };
}

function boxDescPanel(node) {
    const state = layout(node);
    const area = el("textarea", "fbui-text");
    area.placeholder = "Describe this region, or pick a Prompt Composer prompt";
    const thumb = el("button", "pm-layout-thumb");
    thumb.type = "button";
    thumb.addEventListener("click", () => pickComposerPrompt(node));
    const line = el("div", "pm-layout-desc");
    line.append(area, thumb);
    bindLive(node, area, (box, value) => {
        box.desc = value;
        if (box.prompt_ref) {
            delete box.prompt_ref;
            renderThumb(node, box);
        }
    });
    state.thumb = thumb;
    return {
        element: note("Description", line).box,
        sync() {
            const box = selectedBox(node);
            if (!box) return;
            setValue(area, box.desc ?? "");
            renderThumb(node, box);
        },
    };
}

function boxPalettePanel(node) {
    const field = createPaletteField("Palette", BOX_PALETTE_MAX, (colors) => {
        if (!hasSelection(node)) return;
        editState(node, () => { selectedBox(node).palette = colors; });
    });
    return { element: field.element, sync: () => field.set(selectedBox(node)?.palette ?? []) };
}

function scenePalettePanel(node) {
    const field = createPaletteField("Scene palette", SUMMARY_PALETTE_MAX, (colors) => {
        editState(node, () => { layout(node).state.palette = colors; });
    });
    return { element: field.element, sync: () => field.set(layout(node).state.palette) };
}

// One text field bound to a key of layout_data's scene groups.
function sceneField(node, group, key, placeholder) {
    const input = el("input", "fbui-text");
    input.spellcheck = false;
    input.placeholder = placeholder;
    input.addEventListener("pointerdown", (event) => event.stopPropagation());
    input.addEventListener("keydown", keepKeyInField);
    input.addEventListener("input", () => setSceneField(node, group, key, input.value));
    input.addEventListener("change", notifyChange);
    return {
        element: input,
        sync: () => setValue(input, String(layout(node).state[group]?.[key] ?? "")),
    };
}

function scenePair(node, group, first, second) {
    const a = sceneField(node, group, first[0], first[1]);
    const b = sceneField(node, group, second[0], second[1]);
    a.element.style.flex = b.element.style.flex = "1 1 0";
    const row = el("div", "pm-layout-line");
    row.append(a.element, b.element);
    return { element: row, sync: () => { a.sync(); b.sync(); } };
}

// Ideogram's style element decides what the detail line means, and has nothing
// to add when the style is none.
function ideoStyleControl(node) {
    const select = el("select", "fbui-select");
    select.style.cssText = "flex:none;width:40%";
    for (const [value, label] of IDEOGRAM_STYLES) select.append(new Option(label, value));
    select.title = "Ideogram style element";
    select.addEventListener("change", () => {
        setSceneField(node, "ideogram", "style", select.value);
        notifyChange();
        repaint(node);
    });
    const detail = el("input", "fbui-text");
    detail.spellcheck = false;
    detail.style.flex = "1 1 0";
    detail.addEventListener("pointerdown", (event) => event.stopPropagation());
    detail.addEventListener("keydown", keepKeyInField);
    detail.addEventListener("input", () => setSceneField(node, "ideogram", "style_detail", detail.value));
    detail.addEventListener("change", notifyChange);
    const row = el("div", "pm-layout-line");
    row.append(select, detail);
    return {
        element: row,
        sync() {
            const stored = layout(node).state.ideogram;
            const style = IDEOGRAM_STYLES.some(([value]) => value === stored?.style) ? stored.style : "none";
            select.value = style;
            detail.placeholder = IDEOGRAM_DETAIL[style].placeholder;
            detail.disabled = style === "none";
            setValue(detail, String(stored?.style_detail ?? ""));
        },
    };
}

function scenePanel(node) {
    const area = el("textarea", "fbui-text");
    area.placeholder = "One or two sentences summarizing the whole image";
    area.spellcheck = false;
    area.addEventListener("pointerdown", (event) => event.stopPropagation());
    area.addEventListener("keydown", keepKeyInField);
    area.addEventListener("input", () => setWidgetValue(node, "prompt", area.value));
    area.addEventListener("change", () => {
        notifyChange();
        repaint(node);
    });
    const field = note("Scene Summary", area);
    return {
        element: field.box,
        sync() {
            area.readOnly = isLocked(node) || widgetLinked(node, "prompt");
            setValue(area, String(findWidget(node, "prompt")?.value ?? ""));
        },
    };
}

// An undescribed box still claims its region, leaving the model to guess what
// goes there, so say what the prompt is missing.
function missingNote(node) {
    const format = outputFormat(node);
    const missing = [];
    if (format !== "MOSS" && !String(findWidget(node, "prompt")?.value || "").trim()) missing.push("a scene summary");
    const blank = layout(node).state.boxes
        .map((box, index) => {
            const labelled = String(box.desc || "").trim() || box.type === "text" && String(box.text || "").trim();
            return labelled ? null : pad2(index + 1);
        })
        .filter(Boolean);
    if (blank.length) missing.push(`a description for box ${blank.join(", ")}`);
    return missing.length ? `Missing: ${missing.join(", ")}.` : "";
}


// ----------------------------------------------------------------- painting

const toView = (node, box) => {
    const { x, y, w, h } = layout(node).view;
    return { x: x + box.x * w, y: y + box.y * h, w: box.w * w, h: box.h * h };
};

function handlesOf(r) {
    const xs = { w: r.x, "": r.x + r.w / 2, e: r.x + r.w };
    const ys = { n: r.y, "": r.y + r.h / 2, s: r.y + r.h };
    return ["nw", "n", "ne", "e", "se", "s", "sw", "w"].map((key) => ({
        key,
        x: xs[key.match(/[we]/)?.[0] || ""],
        y: ys[key.match(/[ns]/)?.[0] || ""],
    }));
}

// Returns the y of the next line.
function drawWrapped(ctx, text, x, y, maxWidth, maxY, lineHeight) {
    const drawLine = (line) => {
        // A second pass darkens the drop shadow.
        ctx.fillText(line, x, y);
        ctx.fillText(line, x, y);
    };
    for (const paragraph of text.split("\n")) {
        let line = "";
        for (const word of paragraph.split(/\s+/)) {
            const test = line ? `${line} ${word}` : word;
            if (line && ctx.measureText(test).width > maxWidth) {
                if (y > maxY) return y;
                drawLine(line);
                y += lineHeight;
                line = word;
            } else {
                line = test;
            }
        }
        if (y > maxY) return y;
        drawLine(line);
        y += lineHeight;
    }
    return y;
}

function drawBox(node, ctx, box, index, active, textSize) {
    const state = layout(node);
    const rgb = boxRgb(box, index);
    const r = toView(node, box);
    ctx.fillStyle = rgba(rgb, active ? 0.24 : 0.12);
    ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.strokeStyle = rgba(rgb, active ? 1 : 0.8);
    ctx.lineWidth = active ? 2 : 1.25;
    ctx.strokeRect(r.x, r.y, r.w, r.h);

    ctx.save();
    ctx.beginPath();
    ctx.rect(r.x, r.y, r.w, r.h);
    ctx.clip();
    const tag = `${pad2(index + 1)} ${typeInfo(box.type).tag}`;
    ctx.font = "bold 10px sans-serif";
    const tagWidth = ctx.measureText(tag).width + 8;
    ctx.fillStyle = rgba(rgb);
    ctx.fillRect(r.x, r.y, tagWidth, 15);
    ctx.fillStyle = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 140 ? T.card : T.ink;
    ctx.fillText(tag, r.x + 4, r.y + 11);
    (box.palette || []).slice(0, BOX_PALETTE_MAX).forEach((color, i) => {
        ctx.fillStyle = color;
        ctx.fillRect(r.x + tagWidth + 3 + i * 10, r.y + 3, 8, 8);
    });
    if (textSize && r.w > 30 && r.h > 30) {
        const maxY = r.y + r.h - 4;
        let y = r.y + 19 + textSize;
        // Shadow sizes ignore the canvas transform, so scale them to the backing resolution.
        ctx.shadowColor = T.overlay;
        ctx.shadowBlur = Math.max(3, textSize / 3) * state.paintResolution;
        ctx.shadowOffsetY = state.paintResolution;
        if (box.type === "text" && box.text) {
            ctx.font = `bold ${textSize + 3}px ${DISPLAY_FONT}`;
            ctx.fillStyle = T.ink;
            y = drawWrapped(ctx, `"${box.text}"`, r.x + 4, y + 2, r.w - 8, maxY, textSize + 6) + 3;
        }
        if (box.desc) {
            ctx.font = `500 ${textSize}px ${BODY_FONT}`;
            ctx.fillStyle = T.inkSoft;
            drawWrapped(ctx, String(box.desc), r.x + 4, y, r.w - 8, maxY, Math.round(textSize * 1.3));
        }
    }
    ctx.restore();

    if (active && !isLocked(node)) {
        ctx.fillStyle = T.ink;
        ctx.strokeStyle = rgba(rgb.map((c) => Math.round(c * 0.6)));
        ctx.lineWidth = 1;
        for (const handle of handlesOf(r)) {
            ctx.fillRect(handle.x - 3.5, handle.y - 3.5, 7, 7);
            ctx.strokeRect(handle.x - 3.5, handle.y - 3.5, 7, 7);
        }
    }
}

function paint(node) {
    const state = layout(node);
    state.frame = null;
    const canvas = state.canvas;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    const zoom = canvas.getBoundingClientRect().width / width || 1;
    const resolution = Math.min(4, (window.devicePixelRatio || 1) * Math.max(1, zoom));
    const pixelWidth = Math.round(width * resolution);
    const pixelHeight = Math.round(height * resolution);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(resolution, 0, 0, resolution, 0, 0);
    state.paintResolution = resolution;
    ctx.clearRect(0, 0, width, height);

    const [W, H] = canvasSize(node);
    const scale = Math.min((width - 16) / W, (height - 16) / H);
    const view = state.view = { x: (width - W * scale) / 2, y: (height - H * scale) / 2, w: W * scale, h: H * scale };
    ctx.fillStyle = T.card;
    ctx.fillRect(view.x, view.y, view.w, view.h);
    if (state.backgroundReady) {
        // Center-crop like the backend resize, in case the ratio changed since the last run.
        const fit = Math.max(view.w / state.background.naturalWidth, view.h / state.background.naturalHeight);
        const sw = view.w / fit;
        const sh = view.h / fit;
        ctx.drawImage(state.background, (state.background.naturalWidth - sw) / 2, (state.background.naturalHeight - sh) / 2, sw, sh, view.x, view.y, view.w, view.h);
    }
    ctx.lineWidth = 1;
    if (state.grid) {
        ctx.setLineDash([2, 4]);
        ctx.strokeStyle = T.ink;
        for (let i = 1; i < 10; i++) {
            ctx.globalAlpha = i === 5 ? 0.28 : 0.12;
            ctx.beginPath();
            ctx.moveTo(view.x + (view.w * i) / 10, view.y);
            ctx.lineTo(view.x + (view.w * i) / 10, view.y + view.h);
            ctx.moveTo(view.x, view.y + (view.h * i) / 10);
            ctx.lineTo(view.x + view.w, view.y + (view.h * i) / 10);
            ctx.stroke();
        }
        ctx.setLineDash([]);
    }
    ctx.globalAlpha = 0.5;
    ctx.strokeStyle = T.ink;
    ctx.strokeRect(view.x + 0.5, view.y + 0.5, view.w - 1, view.h - 1);
    ctx.globalAlpha = 1;

    const textSize = (state.fullscreen ? FULLSCREEN_TEXT_SIZES : TEXT_SIZES)[textSizeKey(node)];
    const boxes = state.state.boxes;
    boxes.forEach((box, index) => { if (index !== state.selected) drawBox(node, ctx, box, index, false, textSize); });
    if (boxes[state.selected]) drawBox(node, ctx, boxes[state.selected], state.selected, true, textSize);

    if (state.drag?.mode === "draw") {
        const drawn = toView(node, {
            x: Math.min(state.drag.start.x, state.drag.current.x),
            y: Math.min(state.drag.start.y, state.drag.current.y),
            w: Math.abs(state.drag.current.x - state.drag.start.x),
            h: Math.abs(state.drag.current.y - state.drag.start.y),
        });
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = T.brand;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(drawn.x, drawn.y, drawn.w, drawn.h);
        ctx.setLineDash([]);
    }
}

// -------------------------------------------------------------------- stage

function resizeBox(box, origin, handle, dx, dy) {
    let left = origin.x;
    let top = origin.y;
    let right = origin.x + origin.w;
    let bottom = origin.y + origin.h;
    if (handle.includes("w")) left = clamp(left + dx, 0, right - MIN_BOX);
    if (handle.includes("e")) right = clamp(right + dx, left + MIN_BOX, 1);
    if (handle.includes("n")) top = clamp(top + dy, 0, bottom - MIN_BOX);
    if (handle.includes("s")) bottom = clamp(bottom + dy, top + MIN_BOX, 1);
    Object.assign(box, { x: left, y: top, w: right - left, h: bottom - top });
}

function bindStage(node, stage, canvas) {
    const state = layout(node);
    // The canvas takes the gestures so the tool buttons over it keep their own
    // clicks; the stage only holds focus for the shortcuts.
    const localPoint = (event) => {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (event.clientX - rect.left) * (canvas.clientWidth / rect.width),
            y: (event.clientY - rect.top) * (canvas.clientHeight / rect.height),
        };
    };
    const toFraction = (p) => ({ x: (p.x - state.view.x) / state.view.w, y: (p.y - state.view.y) / state.view.h });
    const contains = (box, f) => f.x >= box.x && f.x <= box.x + box.w && f.y >= box.y && f.y <= box.y + box.h;

    function hitHandle(p) {
        if (!hasSelection(node) || isLocked(node)) return null;
        const handle = handlesOf(toView(node, selectedBox(node)))
            .find((h) => Math.abs(p.x - h.x) <= HANDLE_SIZE && Math.abs(p.y - h.y) <= HANDLE_SIZE);
        return handle?.key || null;
    }

    // The selected box wins so it stays draggable under overlapping boxes.
    function hitBox(p) {
        const f = toFraction(p);
        const boxes = state.state.boxes;
        if (state.selected >= 0 && contains(boxes[state.selected], f)) return state.selected;
        for (let i = boxes.length - 1; i >= 0; i--) {
            if (contains(boxes[i], f)) return i;
        }
        return -1;
    }

    canvas.addEventListener("pointerdown", (event) => {
        if (event.button !== 0 || isLocked(node)) return;
        event.preventDefault();
        event.stopPropagation();
        stage.focus({ preventScroll: true });
        const p = localPoint(event);
        const handle = hitHandle(p);
        const hit = event.shiftKey ? -1 : hitBox(p);
        const start = toFraction(p);
        if (handle) {
            state.drag = { mode: "resize", handle, start, origin: { ...selectedBox(node) }, before: snapshot(node) };
        } else if (hit >= 0) {
            state.selected = hit;
            state.drag = { mode: "move", start, origin: { ...state.state.boxes[hit] }, before: snapshot(node) };
            repaint(node);
        } else {
            const point = { x: clamp(start.x, 0, 1), y: clamp(start.y, 0, 1) };
            state.drag = { mode: "draw", start: point, current: point, before: snapshot(node) };
        }
        canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener("pointermove", (event) => {
        const p = localPoint(event);
        if (!state.drag) {
            const handle = hitHandle(p);
            canvas.style.cursor = isLocked(node) ? "default" : handle ? RESIZE_CURSORS[handle] : hitBox(p) >= 0 ? "move" : "crosshair";
            return;
        }
        const f = toFraction(p);
        const dx = f.x - state.drag.start.x;
        const dy = f.y - state.drag.start.y;
        if (state.drag.mode === "draw") {
            state.drag.current = { x: clamp(f.x, 0, 1), y: clamp(f.y, 0, 1) };
        } else {
            const box = state.state.boxes[state.selected];
            if (state.drag.mode === "move") {
                box.x = clamp(state.drag.origin.x + dx, 0, 1 - state.drag.origin.w);
                box.y = clamp(state.drag.origin.y + dy, 0, 1 - state.drag.origin.h);
            } else {
                resizeBox(box, state.drag.origin, state.drag.handle, dx, dy);
            }
        }
        schedulePaint(node);
    });

    function endDrag() {
        if (!state.drag) return;
        const finished = state.drag;
        state.drag = null;
        if (finished.mode === "draw") {
            const x = Math.min(finished.start.x, finished.current.x);
            const y = Math.min(finished.start.y, finished.current.y);
            const w = Math.abs(finished.current.x - finished.start.x);
            const h = Math.abs(finished.current.y - finished.start.y);
            if (w >= MIN_BOX && h >= MIN_BOX) {
                pushUndo(node, finished.before);
                state.state.boxes.push(newBox({ x, y, w, h }));
                state.selected = state.state.boxes.length - 1;
                persist(node);
            } else {
                state.selected = -1;
            }
        } else if (snapshot(node) !== finished.before) {
            pushUndo(node, finished.before);
            persist(node);
        }
        repaint(node);
    }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
}

// Deleting and undo act on the stage that has the focus, and stay out of the
// way while text is being typed.
function bindShortcuts(node, stage) {
    stage.addEventListener("keydown", (event) => {
        // The tools live inside the stage, so a focused button must not steal
        // Delete or the undo keys from the canvas.
        if (event.target.closest("input, textarea, select, button")) return;
        const key = event.key.toLowerCase();
        const mod = event.ctrlKey || event.metaKey;
        if (key === "delete" || key === "backspace") deleteSelected(node);
        else if (mod && key === "z") travel(node, event.shiftKey ? "redo" : "undo", event.shiftKey ? "undo" : "redo");
        else if (mod && key === "y") travel(node, "redo", "undo");
        else if (mod && key === "d") duplicateSelected(node);
        else return;
        event.preventDefault();
        event.stopPropagation();
    });
}

// ---------------------------------------------------------------- full screen

// Full screen gives the canvas the left of the dialog and stacks the fields on
// the right as three framed blocks: output, scene and box. The frames mark where
// one block ends and the next starts, so the scene header stays behind. The
// rows are the card's own elements, so their bindings survive the move and go
// back in order on close.
function openFullscreen(node) {
    const state = layout(node);
    if (state.fullscreen) return;
    const root = state.card.root;
    const rows = [...root.children];
    const held = el("div", "pm-layout-held", "Editing in full screen.");
    root.append(held);
    // The scene block is always on show here, even if it is folded on the node,
    // so the card is refreshed once before the rows move.
    const sceneFold = node.properties.fbui_open_scene;
    node.properties.fbui_open_scene = true;
    state.card.refresh();
    const middle = el("div", "fbui-card pm-layout-col");
    const fields = el("div", "pm-layout-fields");
    const labelWidth = root.style.getPropertyValue("--fbui-label-w");
    middle.style.setProperty("--fbui-label-w", labelWidth);
    const blocks = [[], [], []];
    let belowCanvas = false;
    for (const entry of state.card.entries) {
        // Hidden rows and the scene fold header stay on the node; the header is
        // off screen while the placeholder is up, and close puts it all back.
        if (entry.header || entry.element.classList.contains("hidden")) continue;
        if (entry.row.side === "center") {
            middle.append(entry.element);
            belowCanvas = true;
            continue;
        }
        blocks[entry.row.group === "scene" ? 1 : belowCanvas ? 2 : 0].push(entry.element);
    }
    for (const block of blocks) {
        if (!block.length) continue;
        const frame = el("div", "fbui-card pm-layout-col");
        frame.style.setProperty("--fbui-label-w", labelWidth);
        frame.append(...block);
        fields.append(frame);
    }
    const spread = el("div", "pm-layout-spread");
    spread.append(middle, fields);
    // The modal sits over the graph, so nothing in here reaches the canvas.
    spread.addEventListener("pointerdown", (event) => event.stopPropagation());
    const dialog = openModal({
        title: `Prompt Layout Builder — ${canvasSize(node).join(" × ")}`,
        width: "min(1600px, 92vw)",
        height: "90vh",
        body: spread,
        onClose: () => {
            held.remove();
            root.append(...rows);
            state.fullscreen = null;
            node.properties.fbui_open_scene = sceneFold;
            state.card.refresh();
            repaint(node);
        },
    });
    dialog.modal.classList.add("pm-layout-modal");
    state.fullscreen = dialog;
    // The stage strip reads the full screen state, so its button lights up now.
    state.card.refresh();
    repaint(node);
}

function closeFullscreen(node) {
    layout(node).fullscreen?.close();
}

// ------------------------------------------------------------- execution glue

function setBackground(node, info) {
    const state = layout(node);
    state.backgroundReady = false;
    if (info?.filename) {
        const params = new URLSearchParams({ filename: info.filename, subfolder: info.subfolder || "", type: info.type || "temp" });
        state.background.src = api.apiURL(`/view?${params}`);
    } else {
        state.background.removeAttribute("src");
    }
    schedulePaint(node);
}

function loadState(node) {
    const state = layout(node);
    state.state = readState(node);
    state.selected = state.state.boxes.length ? 0 : -1;
    state.undo = [];
    state.redo = [];
    setBackground(node, node.properties?.[BG_PROP]);
}

function applyExecution(node, message) {
    const state = layout(node);
    node.properties ??= {};
    const refSize = message?.pm_layout_ref_size;
    if (Array.isArray(refSize)) {
        if (refSize.length === 2) node.properties[REF_SIZE_PROP] = refSize;
        else delete node.properties[REF_SIZE_PROP];
    }
    const images = message?.pm_layout_background;
    if (Array.isArray(images)) {
        if (images[0]) node.properties[BG_PROP] = images[0];
        else delete node.properties[BG_PROP];
        setBackground(node, images[0]);
    }
    const imported = message?.pm_layout_import?.[0];
    if (imported) {
        const next = {
            boxes: imported.boxes || [],
            palette: imported.palette ?? state.state.palette ?? [],
            ideogram: imported.ideogram ?? state.state.ideogram ?? {},
            krea: imported.krea ?? state.state.krea ?? {},
        };
        if (JSON.stringify(next) !== JSON.stringify(state.state)) {
            pushUndo(node, snapshot(node));
            state.state = next;
            state.selected = next.boxes.length ? 0 : -1;
            persist(node);
        }
        setWidgetValue(node, "prompt", imported.prompt ?? "");
        setWidgetValue(node, "meta", imported.meta ?? "");
        setWidgetValue(node, "canvas", imported.canvas || "none");
    }
    state.card?.refresh();
    schedulePaint(node);
}

// ------------------------------------------------------------------ node face

// The rows in order for the node. The scene describes the whole image, so it
// folds up top; the box rows follow the canvas because they change with the
// selected box, and the output size closes the node. Full screen puts the canvas
// on the left and the rest on the right as three framed blocks, with the rows
// marked center in beside the canvas; the rows marked lock stay on show but go
// inert while import_json drives the node.
function cardRows(node) {
    return [
        { widget: "use_import_json", label: "Import" },
        { widget: "output_format", label: "Output", kind: "segment", titles: OUTPUT_TITLES },
        { pair: ["aspect_ratio", "megapixels"], label: "Canvas", suffix: "MP", title: "Target canvas size; the ratio splits the megapixels into width and height, rounded to multiples of 16" },
        { group: "scene", label: "Scene" },
        { panel: () => scenePanel(node), height: SCENE_PANEL_HEIGHT, group: "scene", lock: true },
        { widget: "meta", label: "Style tags", group: "scene", lock: true, when: (values) => values.output_format === "MOSS" },
        { widget: "canvas", label: "Composition", kind: "segment", group: "scene", lock: true, labels: { none: "None", single: "Single", multi_panel: "Panels", collage: "Collage" } },
        { panel: () => scenePalettePanel(node), height: PALETTE_PANEL_HEIGHT, group: "scene", lock: true, when: scenePaletteShown },
        { label: "Background", control: () => sceneField(node, "ideogram", "background", "Keep the background clean"), group: "scene", lock: true, when: isIdeogram },
        { label: "Style", control: () => ideoStyleControl(node), group: "scene", lock: true, when: isIdeogram },
        { label: "Look", control: () => scenePair(node, "ideogram", ["aesthetics", "cinematic lighting, moody"], ["lighting", "soft rim light"]), group: "scene", lock: true, when: isIdeogram },
        { label: "Medium", control: () => sceneField(node, "ideogram", "medium", "35mm photograph"), group: "scene", lock: true, when: isIdeogram },
        { label: "Style", control: () => sceneField(node, "krea", "style", "photorealistic cinematic"), group: "scene", lock: true, when: isKrea },
        { label: "Medium", control: () => sceneField(node, "krea", "medium", "35mm photograph"), group: "scene", lock: true, when: isKrea },
        { label: "Background", control: () => sceneField(node, "krea", "background", "Plain studio backdrop"), group: "scene", lock: true, when: isKrea },
        { panel: () => stagePanel(node), grow: true, height: STAGE_MIN_HEIGHT, side: "center" },
        { label: "Box", control: () => navControl(node), when: hasBoxes },
        { label: "Coords", control: () => coordControl(node), when: boxPicked },
        { label: "Kind", control: () => typeControl(node), when: boxPicked },
        { label: "Tag", control: () => kreaTagControl(node), when: kreaCharacter },
        { panel: () => boxTextPanel(node), height: TEXT_PANEL_HEIGHT, when: textPicked },
        { panel: () => boxDescPanel(node), height: DESC_PANEL_HEIGHT, when: boxPicked },
        { panel: () => boxPalettePanel(node), height: PALETTE_PANEL_HEIGHT, when: (values, node) => boxPicked(values, node) && !isKrea(values) },
        { panel: () => captionPanel(node), height: CAPTION_HEIGHT, side: "center" },
    ];
}

function setupNode(node) {
    ensureCss();
    const state = {
        state: { boxes: [], palette: [], ideogram: {}, krea: {} },
        selected: -1,
        undo: [],
        redo: [],
        grid: false,
        view: { x: 0, y: 0, w: 1, h: 1 },
        drag: null,
        frame: null,
        paintResolution: 1,
        background: new Image(),
        backgroundReady: false,
        composerLoad: null,
        pendingSnapshot: null,
        scrubBefore: null,
        fullscreen: null,
        card: null,
        stage: null,
        canvas: null,
        caption: null,
        thumb: null,
    };
    node._layout = state;
    state.background.onload = () => { state.backgroundReady = true; schedulePaint(node); };
    state.background.onerror = () => { state.backgroundReady = false; schedulePaint(node); };
    loadState(node);
    suppressCorePreview(node);
    // Not driven by a card row, so hide them here as well as in the definition.
    for (const name of ["prompt", "layout_data"]) hideWidget(findWidget(node, name));
    // The scene is the part people come to this node for, so its group starts open.
    node.properties ??= {};
    if (node.properties.fbui_open_scene === undefined) node.properties.fbui_open_scene = true;
    const card = mountCard(node, { minWidth: MIN_NODE_WIDTH, rows: cardRows(node) });
    state.card = card;
    helpBadge(node, GUIDE);
    // The canvas frame follows the size widgets, so a new ratio or megapixel
    // value redraws the stage instead of waiting for a click in it.
    for (const name of ["aspect_ratio", "megapixels"]) chainCallback(findWidget(node, name), "callback", () => schedulePaint(node));
    chainCallback(findWidget(node, "use_import_json"), "callback", () => applyLock(node));
    chainCallback(node, "onConfigure", () => queueMicrotask(() => {
        loadState(node);
        card.refresh();
        applyLock(node);
    }));
    chainCallback(node, "onExecuted", (message) => applyExecution(node, message));
    chainCallback(node, "onRemoved", () => {
        closeColorPicker();
        closeFullscreen(node);
    });
    card.refresh();
    schedulePaint(node);
}

app.registerExtension({
    name: "PromptManager.PromptLayoutBuilder",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        hideInputsInDef(nodeData, HIDDEN_WIDGETS);
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            setupNode(this);
            this.setSize([Math.max(MIN_NODE_WIDTH, this.size[0]), Math.max(INITIAL_NODE_HEIGHT, this.size[1])]);
        });
    },
});

// The source toggle only matters when both inputs are linked.
function syncBboxConverter(node) {
    const widget = findWidget(node, "use_bboxes");
    if (!widget) return;
    const linked = (name) => node.inputs?.find((input) => input.name === name)?.link != null;
    const show = linked("layout_json") && linked("bboxes");
    if (show === !isWidgetHidden(widget)) return;
    if (show) showWidget(widget);
    else hideWidget(widget);
    node.setSize([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "PromptManager.BboxConverter",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "BboxConverter") return;
        for (const method of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
            chainCallback(nodeType.prototype, method, function () {
                syncBboxConverter(this);
            });
        }
    },
});
