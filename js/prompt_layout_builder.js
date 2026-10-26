import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { PM_UI_PALETTE as UI } from "./ui_palette.js";
import { DEFAULT_THUMBNAIL } from "./prompt_manager_advanced.js";
import { showThumbnailBrowser } from "./prompt_browser.js";
import { COMPOSER_ENDPOINT_PREFIX, getComposerEntry, loadComposerPrompts } from "./prompt_composer_common.js";

// Boxes are canvas fractions {x, y, w, h, type, text, desc, palette, prompt_ref?}; mirrors nodes/prompt_layout_builder.py.
const NODE_NAME = "PromptLayoutBuilder";
const REF_IMAGE_RATIO = "Use Ref Image Ratio";
const SIZE_MULTIPLE = 16;
const BOX_PALETTE_MAX = 8;
const SUMMARY_PALETTE_MAX = 16;
const HISTORY_LIMIT = 50;
const MIN_NODE_WIDTH = 560;
const INITIAL_EDITOR_HEIGHT = 760;
const STAGE_MIN_HEIGHT = 240;
const HANDLE_SIZE = 8;
const MIN_BOX = 0.01;
const BG_PROP = "pm_layout_background";
const REF_SIZE_PROP = "pm_layout_ref_size";
const TEXT_SIZE_PROP = "pm_layout_text_size";
const TEXT_SIZES = { hidden: 0, small: 9, medium: 11, large: 14, larger: 18 };
const FULLSCREEN_TEXT_SIZES = { hidden: 0, small: 11, medium: 14, large: 18, larger: 23 };
const DISPLAY_FONT = `Futura, "Futura PT", "Century Gothic", "Avenir Next", "Trebuchet MS", sans-serif`;
const BODY_FONT = `Inter, "Segoe UI", "Noto Sans", Roboto, "Helvetica Neue", system-ui, sans-serif`;
const STYLE_ID = "pm-layout-style";
const DEFAULT_COLOR = "#FFFFFF";
// Backend inputs whose values are edited from the editor UI instead of native widgets.
const HIDDEN_WIDGETS = ["aspect_ratio", "megapixels", "prompt", "meta", "canvas", "use_import_json", "layout_data"];
const BOX_TYPES = [
    { value: "object", label: "Object", tag: "OBJ" },
    { value: "character", label: "Character", tag: "CHR" },
    { value: "text", label: "Text", tag: "TXT" },
];
// Prompt Composer subject accent hues.
const BOX_HUES = [205, 40, 92, 8, 248, 155, 332];
const HEX_COLOR = /^#[0-9a-f]{6}$/i;
const RESIZE_CURSORS = { nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize", n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize" };

const getWidget = (node, name) => node.widgets?.find((w) => w.name === name) || null;
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

function canvasSize(node) {
    const ratioName = String(getWidget(node, "aspect_ratio")?.value || "");
    const ratio = ratioName === REF_IMAGE_RATIO
        ? (node.properties?.[REF_SIZE_PROP] || [1, 1])
        : (ratioName.match(/^(\d+):(\d+)/)?.slice(1).map(Number) || [1, 1]);
    const megapixels = Number(getWidget(node, "megapixels")?.value) || 1;
    const scale = Math.sqrt(megapixels * 1024 * 1024 / (ratio[0] * ratio[1]));
    return ratio.map((r) => Math.max(SIZE_MULTIPLE, Math.round(r * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE));
}

function readState(node) {
    try {
        const state = JSON.parse(getWidget(node, "layout_data")?.value || "{}");
        return {
            boxes: Array.isArray(state.boxes) ? state.boxes : [],
            palette: Array.isArray(state.palette) ? state.palette : [],
        };
    } catch {
        return { boxes: [], palette: [] };
    }
}

function boxLabel(box, index) {
    const summary = String(box.type === "text" && box.text ? `"${box.text}"` : box.desc || "").replace(/\s+/g, " ").trim();
    const head = `${pad2(index + 1)} ${typeInfo(box.type).label}`;
    return summary ? `${head} — ${summary.length > 48 ? `${summary.slice(0, 47)}…` : summary}` : head;
}

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .pm-layout-host { width: 100%; height: 100%; }
        .pm-layout { display: flex; flex-direction: column; gap: 8px; width: 100%; height: 100%; box-sizing: border-box; padding: 8px;
            background: ${UI.panel}; border: 1px solid ${UI.panelBorder}; border-radius: 8px; color: ${UI.textPrimary};
            font: 12px/1.35 sans-serif; overflow: auto; }
        .pm-layout * { box-sizing: border-box; }
        .pm-layout-main { display: flex; flex-direction: column; gap: 8px; flex: 1 0 auto; }
        .pm-layout-side { display: flex; flex-direction: column; gap: 8px; flex: 0 0 auto; }
        .pm-layout-row { display: flex; gap: 6px; align-items: center; }
        .pm-layout-settings { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); gap: 6px 8px; }
        .pm-layout-toggle { display: flex; gap: 8px; align-items: center; cursor: pointer; color: ${UI.textPrimary}; }
        .pm-layout .pm-layout-toggle input { appearance: none; position: relative; flex: 0 0 32px; width: 32px; height: 18px; margin: 0; padding: 0;
            border: 0; border-radius: 9px; background: ${UI.inputBorder}; cursor: pointer; transition: background 0.15s; }
        .pm-layout .pm-layout-toggle input::before { content: ""; position: absolute; top: 2px; left: 2px; width: 14px; height: 14px;
            border-radius: 50%; background: #fff; transition: transform 0.15s; }
        .pm-layout .pm-layout-toggle input:checked { background: hsl(208 73% 57%); }
        .pm-layout .pm-layout-toggle input:checked::before { transform: translateX(14px); }
        .pm-layout .pm-layout-toggle input:focus-visible { outline: 2px solid ${UI.accentBorder}; outline-offset: 2px; }
        .pm-layout-target { grid-column: 2; color: ${UI.textHint}; text-align: right; white-space: nowrap; }
        .pm-layout-split { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); gap: 8px; }
        .pm-layout .pm-layout-tool-select { width: auto; height: 28px; padding: 0 6px; }
        .pm-layout-spacer { flex: 1; }
        .pm-layout button { height: 28px; padding: 0 10px; border-radius: 6px; border: 1px solid ${UI.inputBorder};
            background: ${UI.buttonBg}; color: ${UI.textPrimary}; font: inherit; white-space: nowrap; cursor: pointer; }
        .pm-layout button:hover:not(:disabled) { border-color: ${UI.accentBorder}; }
        .pm-layout button:disabled { opacity: 0.4; cursor: default; }
        .pm-layout button.active { background: ${UI.accentSoft}; border-color: ${UI.accentBorder}; color: #dbeafe; }
        .pm-layout input, .pm-layout textarea, .pm-layout select { width: 100%; min-width: 0; padding: 5px 8px; border-radius: 6px;
            border: 1px solid ${UI.inputBorder}; background: ${UI.inputBg}; color: ${UI.textPrimary}; font: inherit; }
        .pm-layout textarea { resize: vertical; min-height: 52px; }
        .pm-layout input:focus, .pm-layout textarea:focus, .pm-layout select:focus { outline: none; border-color: ${UI.accentBorder}; }
        .pm-layout input:disabled, .pm-layout textarea:disabled, .pm-layout select:disabled { opacity: 0.55; }
        .pm-layout-label { display: block; margin-bottom: 3px; color: ${UI.textMuted}; font-size: 11px; }
        .pm-layout-stage { position: relative; flex: 1 1 auto; min-height: ${STAGE_MIN_HEIGHT}px; border: 1px solid ${UI.cardBorder};
            border-radius: 6px; background: hsl(220 15% 8%); overflow: hidden; outline: none; touch-action: none; }
        .pm-layout-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
        .pm-layout-status { display: flex; justify-content: space-between; gap: 8px; color: ${UI.textHint}; font-size: 11px; }
        .pm-layout-banner { padding: 6px 8px; border-radius: 6px; border: 1px solid hsl(40 92% 60% / 0.6);
            background: hsl(40 92% 60% / 0.12); color: hsl(48 100% 90%); }
        .pm-layout-card { display: flex; flex-direction: column; gap: 8px; padding: 8px; border-radius: 8px;
            border: 1px solid ${UI.sectionBorder}; background: ${UI.cardBg}; }
        .pm-layout fieldset { display: contents; }
        .pm-layout-title { font-weight: 600; color: ${UI.textHeading}; }
        .pm-layout-coords { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; }
        .pm-layout-types { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }
        .pm-layout-desc { display: flex; gap: 6px; align-items: stretch; }
        .pm-layout .pm-layout-thumb { flex: 0 0 52px; width: 52px; height: 52px; padding: 0; overflow: hidden; font-size: 18px;
            display: flex; align-items: center; justify-content: center; color: ${UI.textHint}; }
        .pm-layout-thumb img, .pm-layout-thumb video { width: 100%; height: 100%; object-fit: cover; }
        .pm-layout-swatches { display: flex; flex-wrap: wrap; gap: 6px; }
        .pm-layout .pm-layout-swatch { width: 26px; height: 26px; padding: 0; border-radius: 6px; border-color: rgba(255, 255, 255, 0.28); }
        .pm-layout-empty { justify-content: center; color: ${UI.textHint}; text-align: center; padding: 10px; }
        .pm-layout-ghost { opacity: 0.45; }
        .pm-layout .pm-layout-ghost textarea:disabled { opacity: 1; }
        .pm-layout-picker { position: fixed; z-index: 10002; width: 236px; display: flex; flex-direction: column; gap: 10px; padding: 10px;
            border-radius: 8px; border: 1px solid ${UI.panelBorder}; background: ${UI.panel}; box-shadow: 0 10px 28px rgba(0, 0, 0, 0.55);
            color: ${UI.textPrimary}; font: 12px/1.35 sans-serif; }
        .pm-layout-picker * { box-sizing: border-box; }
        .pm-layout-picker-area { position: relative; height: 150px; border-radius: 6px; cursor: crosshair; touch-action: none; }
        .pm-layout-picker-hue { position: relative; height: 12px; border-radius: 6px; cursor: pointer; touch-action: none;
            background: linear-gradient(to right, #f00, #ff0, #0f0, #0ff, #00f, #f0f, #f00); }
        .pm-layout-picker-thumb { position: absolute; width: 14px; height: 14px; margin: -7px 0 0 -7px; border-radius: 50%;
            border: 2px solid #fff; box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.6); pointer-events: none; }
        .pm-layout-picker-hue .pm-layout-picker-thumb { top: 50%; background: transparent; }
        .pm-layout-picker-row { display: flex; gap: 6px; }
        .pm-layout-picker select, .pm-layout-picker input, .pm-layout-picker button { height: 28px; padding: 0 8px; border-radius: 6px;
            border: 1px solid ${UI.inputBorder}; background: ${UI.inputBg}; color: ${UI.textPrimary}; font: inherit; }
        .pm-layout-picker select { flex: 0 0 72px; }
        .pm-layout-picker input { flex: 1; min-width: 0; }
        .pm-layout-picker button { background: ${UI.buttonBg}; cursor: pointer; }
        .pm-layout-picker input:focus, .pm-layout-picker select:focus { outline: none; border-color: ${UI.accentBorder}; }
        .pm-layout-placeholder { display: none; height: 100%; align-items: center; justify-content: center; border-radius: 8px;
            border: 1px dashed ${UI.panelBorder}; color: ${UI.textHint}; font: 12px sans-serif; }
        .pm-layout-overlay { position: fixed; inset: 0; z-index: 9000; display: flex; padding: 24px; background: ${UI.overlay}; }
        .pm-layout.pm-layout-full { display: grid; grid-template-columns: minmax(0, 1fr) 400px; gap: 12px; padding: 12px; overflow: hidden; }
        .pm-layout-full .pm-layout-main { min-height: 0; }
        .pm-layout-full .pm-layout-side { min-height: 0; overflow: auto; }
        .pm-layout-full .pm-layout-toggle { display: none; }
        .pm-layout-footer { display: none; }
        .pm-layout-full .pm-layout-footer { display: flex; justify-content: flex-end; margin-top: auto; padding-top: 8px; }
        .pm-layout-full .pm-layout-open-full { display: none; }
        .pm-layout-full .pm-layout-scene textarea { min-height: 120px; }
    `;
    document.head.appendChild(style);
}

function el(tag, className, parent, text) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text != null) item.textContent = text;
    parent?.appendChild(item);
    return item;
}

function makeButton(parent, text, title, onClick) {
    const button = el("button", "", parent, text);
    button.type = "button";
    button.title = title;
    button.addEventListener("click", onClick);
    return button;
}

function labeled(parent, text) {
    const wrap = el("div", "", parent);
    const label = el("span", "pm-layout-label", wrap, text);
    return { wrap, label };
}

// Avoid resetting the caret of the field being typed in.
function setValue(input, value) {
    if (document.activeElement !== input && input.value !== value) input.value = value;
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

let activePicker = null;

function closeColorPicker() {
    if (!activePicker) return;
    const { element, cleanup, onClose } = activePicker;
    activePicker = null;
    cleanup();
    element.remove();
    onClose();
}

function openColorPicker(anchor, color, { onInput, onRemove, onClose }) {
    closeColorPicker();
    let [hue, saturation, value] = hexToHsv(color);
    const picker = el("div", "pm-layout-picker", document.body);
    const area = el("div", "pm-layout-picker-area", picker);
    const areaThumb = el("div", "pm-layout-picker-thumb", area);
    const hueBar = el("div", "pm-layout-picker-hue", picker);
    const hueThumb = el("div", "pm-layout-picker-thumb", hueBar);
    const row = el("div", "pm-layout-picker-row", picker);
    const format = el("select", "", row);
    format.append(new Option("Hex", "hex"), new Option("RGB", "rgb"));
    const input = el("input", "", row);
    input.spellcheck = false;
    makeButton(el("div", "pm-layout-picker-row", picker), "Remove color", "Remove this color from the palette", () => {
        onRemove();
        closeColorPicker();
    });

    const update = (emit) => {
        const hex = hsvToHex(hue, saturation, value);
        area.style.background = `linear-gradient(to top, #000, transparent), linear-gradient(to right, #fff, transparent), hsl(${hue} 100% 50%)`;
        areaThumb.style.left = `${saturation * 100}%`;
        areaThumb.style.top = `${(1 - value) * 100}%`;
        areaThumb.style.background = hex;
        hueThumb.style.left = `${(hue / 360) * 100}%`;
        setValue(input, formatColor(hex, format.value));
        if (emit) onInput(hex);
    };
    trackPointer(area, (x, y) => { saturation = x; value = 1 - y; update(true); });
    trackPointer(hueBar, (x) => { hue = x * 360; update(true); });
    format.addEventListener("change", () => { input.value = formatColor(hsvToHex(hue, saturation, value), format.value); });
    input.addEventListener("change", () => {
        const parsed = parseColor(input.value);
        if (parsed) [hue, saturation, value] = hexToHsv(parsed);
        input.blur();
        update(!!parsed);
    });

    const rect = anchor.getBoundingClientRect();
    picker.style.left = `${clamp(rect.left, 8, window.innerWidth - 244)}px`;
    picker.style.top = `${rect.bottom + 270 < window.innerHeight ? rect.bottom + 6 : Math.max(8, rect.top - 266)}px`;

    const onPointer = (event) => { if (!picker.contains(event.target)) closeColorPicker(); };
    // Window capture runs before the full-screen Escape handler on document.
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

// Edits stay local while the picker is open and are committed once when it closes.
function createPaletteField(parent, labelText, limit, onCommit) {
    const swatches = el("div", "pm-layout-swatches", labeled(parent, labelText).wrap);
    let colors = [];

    const edit = (index, button) => openColorPicker(button, colors[index], {
        onInput: (hex) => {
            colors[index] = hex;
            button.style.background = hex;
            button.title = hex;
        },
        onRemove: () => { colors.splice(index, 1); },
        onClose: () => onCommit([...colors]),
    });

    const draw = () => {
        const buttons = colors.map((color, index) => {
            const button = makeButton(null, "", `${color} (right-click to remove)`, () => edit(index, button));
            button.classList.add("pm-layout-swatch");
            button.style.background = color;
            button.addEventListener("contextmenu", (event) => {
                event.preventDefault();
                event.stopPropagation();
                colors.splice(index, 1);
                onCommit([...colors]);
            });
            return button;
        });
        const add = makeButton(null, "+", "Add a color", () => {
            colors.push(DEFAULT_COLOR);
            draw();
            edit(colors.length - 1, swatches.children[colors.length - 1]);
        });
        add.classList.add("pm-layout-swatch");
        add.disabled = colors.length >= limit;
        swatches.replaceChildren(...buttons, add);
    };

    return {
        set(next) {
            if (activePicker && swatches.contains(activePicker.anchor)) return;
            colors = [...next];
            draw();
        },
    };
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

function createLayoutEditor(node) {
    ensureStyle();
    let state = readState(node);
    let selected = state.boxes.length ? 0 : -1;
    let undo = [];
    let redo = [];
    let showGrid = false;
    let drag = null;
    let view = { x: 0, y: 0, w: 1, h: 1 };
    let paintResolution = 1;
    let frame = null;
    let fullscreen = null;
    let pendingSnapshot = null;
    let picking = false;
    let composerLoad = null;
    let backgroundReady = false;
    let minHeight = INITIAL_EDITOR_HEIGHT;
    let fitHostHeight = 0;
    const background = new Image();
    background.onload = () => { backgroundReady = true; schedulePaint(); };
    background.onerror = () => { backgroundReady = false; schedulePaint(); };

    const host = el("div", "pm-layout-host");
    const placeholder = el("div", "pm-layout-placeholder", host, "Editing in full screen…");
    const root = el("div", "pm-layout", host);
    root.addEventListener("pointerdown", (event) => event.stopPropagation());

    const main = el("div", "pm-layout-main", root);
    const settings = el("div", "pm-layout-settings", main);
    const aspectSelect = el("select", "", labeled(settings, "Aspect ratio").wrap);
    aspectSelect.addEventListener("change", () => { setWidgetValue("aspect_ratio", aspectSelect.value); render(); });
    const megapixelsInput = el("input", "", labeled(settings, "Megapixels").wrap);
    Object.assign(megapixelsInput, { type: "number", min: "0.1", max: "16", step: "0.1" });
    megapixelsInput.addEventListener("input", () => {
        const value = Number(megapixelsInput.value);
        if (!Number.isFinite(value) || value < 0.1 || value > 16) return;
        setWidgetValue("megapixels", Math.round(value * 100) / 100);
        render();
    });
    megapixelsInput.addEventListener("change", () => { megapixelsInput.blur(); render(); });
    const importSwitch = el("label", "pm-layout-toggle", settings);
    importSwitch.title = "Read the layout from import_json and lock the editor";
    const importToggle = el("input", "", importSwitch);
    importToggle.type = "checkbox";
    importToggle.setAttribute("role", "switch");
    el("span", "", importSwitch, "Use import_json");
    importToggle.addEventListener("change", () => { setWidgetValue("use_import_json", importToggle.checked); render(); });
    const targetLabel = el("span", "pm-layout-target", settings);

    const sceneCard = el("div", "pm-layout-card pm-layout-scene", main);
    const sceneFields = el("fieldset", "", sceneCard);
    el("span", "pm-layout-title", sceneFields, "Scene");
    const scenePrompt = el("textarea", "", labeled(sceneFields, "Prompt").wrap);
    scenePrompt.placeholder = "Scene description (summary caption)";
    const sceneRow = el("div", "pm-layout-split", sceneFields);
    const sceneMeta = el("input", "", labeled(sceneRow, "Meta (comma-separated style tags)").wrap);
    sceneMeta.placeholder = "poster, graphic design";
    const sceneCanvas = el("select", "", labeled(sceneRow, "Canvas").wrap);
    for (const [input, name] of [[scenePrompt, "prompt"], [sceneMeta, "meta"], [sceneCanvas, "canvas"]]) {
        input.addEventListener(input === sceneCanvas ? "change" : "input", () => setWidgetValue(name, input.value));
    }
    const scenePalette = createPaletteField(sceneFields, "Scene palette", SUMMARY_PALETTE_MAX,
        (colors) => edit(() => { state.palette = colors; }));

    const toolbar = el("div", "pm-layout-row", main);
    const addButton = makeButton(toolbar, "+ Box", "Add a box", () => addBox());
    const gridButton = makeButton(toolbar, "Grid", "Toggle grid overlay", () => { showGrid = !showGrid; render(); });
    const textSizeSelect = el("select", "pm-layout-tool-select", toolbar);
    textSizeSelect.title = "Size of the box text shown on the canvas";
    for (const [value, label] of [["hidden", "Text: Hidden"], ["small", "Text: Small"], ["medium", "Text: Medium"], ["large", "Text: Large"], ["larger", "Text: Larger"]]) {
        textSizeSelect.append(new Option(label, value));
    }
    textSizeSelect.addEventListener("change", () => {
        node.properties ??= {};
        node.properties[TEXT_SIZE_PROP] = textSizeSelect.value;
        schedulePaint();
    });
    const undoButton = makeButton(toolbar, "↶", "Undo (Ctrl/Cmd+Z)", () => travel(undo, redo));
    const redoButton = makeButton(toolbar, "↷", "Redo (Ctrl/Cmd+Shift+Z)", () => travel(redo, undo));
    const clearButton = makeButton(toolbar, "Clear", "Clear all boxes (undoable)", () => edit(() => { state.boxes = []; selected = -1; }));
    el("div", "pm-layout-spacer", toolbar);
    const fullButton = makeButton(toolbar, "⛶ Full screen", "Open the editor in full screen", () => openFullscreen());
    fullButton.classList.add("pm-layout-open-full");
    const banner = el("div", "pm-layout-banner", main, "Driven by import_json. Turn off Use import_json to edit the layout.");
    const stage = el("div", "pm-layout-stage", main);
    stage.tabIndex = 0;
    const canvas = el("canvas", "", stage);
    const status = el("div", "pm-layout-status", main);
    const statusSize = el("span", "", status);
    const statusHint = el("span", "", status);

    const side = el("div", "pm-layout-side", root);
    const navRow = el("div", "pm-layout-row", side);
    const navSelect = el("select", "", navRow);
    navSelect.addEventListener("change", () => { selected = Number(navSelect.value); render(); });
    const prevButton = makeButton(navRow, "←", "Previous box", () => { selected -= 1; render(); });
    const nextButton = makeButton(navRow, "→", "Next box", () => { selected += 1; render(); });

    const emptyHint = el("div", "pm-layout-card pm-layout-empty", side, "No box selected. Drag on the canvas or press + Box.");
    const boxCard = el("div", "pm-layout-card", side);
    const boxFields = el("fieldset", "", boxCard);
    const boxHeader = el("div", "pm-layout-row", boxFields);
    const boxTitle = el("span", "pm-layout-title", boxHeader);
    el("div", "pm-layout-spacer", boxHeader);
    makeButton(boxHeader, "Duplicate", "Duplicate this box (Ctrl/Cmd+D)", () => duplicateSelected());
    makeButton(boxHeader, "Delete", "Delete this box (Del)", () => deleteSelected());

    const coords = el("div", "pm-layout-coords", boxFields);
    const coordInputs = ["x", "y", "width", "height"].map((name, index) => {
        const input = el("input", "", labeled(coords, name).wrap);
        input.type = "number";
        input.step = "1";
        input.addEventListener("change", () => setCoord(index, Number(input.value)));
        return input;
    });

    const types = el("div", "pm-layout-types", boxFields);
    const typeButtons = BOX_TYPES.map((type) => makeButton(types, type.label, `Label this box as ${type.value}`, () => {
        edit(() => { state.boxes[selected].type = type.value; });
    }));

    const textField = labeled(boxFields, "Text content");
    const textInput = el("textarea", "", textField.wrap);
    textInput.placeholder = "Exact text to render in this region";
    bindLive(textInput, (box, value) => { box.text = value; });

    const descField = labeled(boxFields, "Description");
    const descRow = el("div", "pm-layout-desc", descField.wrap);
    const descInput = el("textarea", "", descRow);
    descInput.placeholder = "Describe this region…";
    bindLive(descInput, (box, value) => {
        box.desc = value;
        if (box.prompt_ref) {
            delete box.prompt_ref;
            renderThumb(box);
        }
    });
    const thumbButton = makeButton(descRow, "", "", () => pickComposerPrompt());
    thumbButton.classList.add("pm-layout-thumb");

    const boxPalette = createPaletteField(boxFields, "Palette", BOX_PALETTE_MAX,
        (colors) => edit(() => { state.boxes[selected].palette = colors; }));

    const exitButton = makeButton(el("div", "pm-layout-footer", side), "Exit full screen", "Back to the node (Esc)", () => closeFullscreen());
    exitButton.classList.add("active");

    const snapshot = () => JSON.stringify(state);
    const textSizeKey = () => (node.properties?.[TEXT_SIZE_PROP] in TEXT_SIZES ? node.properties[TEXT_SIZE_PROP] : "medium");
    const isLocked = () => !!getWidget(node, "use_import_json")?.value;

    function setWidgetValue(name, value) {
        const widget = getWidget(node, name);
        if (!widget) return;
        widget.value = value;
        app.graph?.setDirtyCanvas(true, true);
    }

    function persist() {
        setWidgetValue("layout_data", JSON.stringify(state));
    }

    function pushUndo(before) {
        undo.push(before);
        if (undo.length > HISTORY_LIMIT) undo.shift();
        redo = [];
    }

    function edit(mutate) {
        if (isLocked()) return;
        const before = snapshot();
        mutate();
        if (snapshot() === before) return;
        pushUndo(before);
        persist();
        render();
    }

    function travel(from, to) {
        if (!from.length || isLocked()) return;
        to.push(snapshot());
        state = JSON.parse(from.pop());
        selected = Math.min(selected, state.boxes.length - 1);
        persist();
        render();
    }

    // One undo step per focus session of a text field.
    function bindLive(input, apply) {
        input.addEventListener("focus", () => { pendingSnapshot = snapshot(); });
        input.addEventListener("input", () => {
            if (selected < 0 || isLocked()) return;
            if (pendingSnapshot !== null) {
                pushUndo(pendingSnapshot);
                pendingSnapshot = null;
            }
            apply(state.boxes[selected], input.value);
            persist();
            renderNav();
            renderToolbar();
            schedulePaint();
        });
    }

    function newBox(rect) {
        return { ...rect, type: "object", text: "", desc: "", palette: [] };
    }

    function addBox() {
        edit(() => {
            const offset = (state.boxes.length % 6) * 0.025;
            state.boxes.push(newBox({ x: 0.3 + offset, y: 0.3 + offset, w: 0.3, h: 0.3 }));
            selected = state.boxes.length - 1;
        });
    }

    function duplicateSelected() {
        if (selected < 0) return;
        edit(() => {
            const copy = structuredClone(state.boxes[selected]);
            copy.x = clamp(copy.x + 0.02, 0, 1 - copy.w);
            copy.y = clamp(copy.y + 0.02, 0, 1 - copy.h);
            state.boxes.splice(selected + 1, 0, copy);
            selected += 1;
        });
    }

    function deleteSelected() {
        if (selected < 0) return;
        edit(() => {
            state.boxes.splice(selected, 1);
            selected = Math.min(selected, state.boxes.length - 1);
        });
    }

    function setCoord(index, value) {
        if (selected < 0 || !Number.isFinite(value)) return render();
        const [W, H] = canvasSize(node);
        edit(() => {
            const box = state.boxes[selected];
            const px = [box.x * W, box.y * H, box.w * W, box.h * H];
            px[index] = value;
            const w = clamp(px[2] / W, MIN_BOX, 1);
            const h = clamp(px[3] / H, MIN_BOX, 1);
            Object.assign(box, { x: clamp(px[0] / W, 0, 1 - w), y: clamp(px[1] / H, 0, 1 - h), w, h });
        });
    }

    function ensureComposerPrompts() {
        composerLoad ??= loadComposerPrompts(node);
        return composerLoad;
    }

    async function pickComposerPrompt() {
        if (selected < 0 || isLocked()) return;
        const index = selected;
        const ref = state.boxes[index].prompt_ref;
        await ensureComposerPrompts();
        const category = ref?.category || Object.keys(node.prompts || {}).find((key) => key !== "__meta__") || "";
        picking = true;
        let selection;
        try {
            selection = await showThumbnailBrowser(node, category, ref?.name || "", {
                title: "Select Prompt for Box",
                endpointPrefix: COMPOSER_ENDPOINT_PREFIX,
                promptOnly: true,
                loadPromptsFn: loadComposerPrompts,
                preferenceScope: "composer",
                allowMultiSelect: false,
            });
        } finally {
            picking = false;
        }
        const entry = selection?.prompt ? getComposerEntry(node, selection.category, selection.prompt) : null;
        if (!entry || !state.boxes[index]) return;
        selected = index;
        edit(() => {
            const box = state.boxes[index];
            box.desc = String(entry.prompt || "");
            box.prompt_ref = { category: selection.category, name: selection.prompt };
        });
    }

    function renderThumb(box) {
        const ref = box.prompt_ref;
        thumbButton.replaceChildren();
        if (!ref) {
            thumbButton.textContent = "+";
            thumbButton.title = "Use a Prompt Composer prompt as the description";
            return;
        }
        thumbButton.title = `${ref.name}: click to pick another prompt`;
        if (!node.prompts) {
            ensureComposerPrompts().then(() => render());
            return;
        }
        const src = getComposerEntry(node, ref.category, ref.name)?.thumbnail || DEFAULT_THUMBNAIL;
        const media = el(String(src).startsWith("data:video") ? "video" : "img", "", thumbButton);
        media.muted = true;
        media.src = src;
    }

    function renderToolbar() {
        const locked = isLocked();
        addButton.disabled = locked;
        clearButton.disabled = locked || !state.boxes.length;
        undoButton.disabled = locked || !undo.length;
        redoButton.disabled = locked || !redo.length;
        gridButton.classList.toggle("active", showGrid);
        textSizeSelect.value = textSizeKey();
    }

    function renderNav() {
        const count = state.boxes.length;
        const options = state.boxes.map((box, index) => {
            const option = el("option", "", null, boxLabel(box, index));
            option.value = String(index);
            return option;
        });
        if (selected < 0) {
            const placeholderOption = el("option", "", null, count ? "No box selected" : "No boxes");
            placeholderOption.value = "-1";
            options.unshift(placeholderOption);
        }
        navSelect.replaceChildren(...options);
        navSelect.value = String(selected);
        navSelect.disabled = !count;
        navSelect.style.borderLeft = selected >= 0 ? `3px solid ${rgba(boxRgb(state.boxes[selected], selected))}` : "";
        prevButton.disabled = selected <= 0;
        nextButton.disabled = selected < 0 || selected >= count - 1;
    }

    function renderBox() {
        // Keep the panel height when the card hides so the canvas does not jump.
        if (selected < 0 && boxCard.offsetHeight) emptyHint.style.minHeight = `${boxCard.offsetHeight}px`;
        boxCard.style.display = selected >= 0 ? "" : "none";
        emptyHint.style.display = selected >= 0 || isLocked() ? "none" : "";
        if (selected < 0) return;
        const box = state.boxes[selected];
        const [W, H] = canvasSize(node);
        boxTitle.textContent = `Box ${pad2(selected + 1)}`;
        boxCard.style.borderColor = rgba(boxRgb(box, selected), 0.7);
        [box.x * W, box.y * H, box.w * W, box.h * H].forEach((value, index) => setValue(coordInputs[index], String(Math.round(value))));
        typeButtons.forEach((button, index) => button.classList.toggle("active", BOX_TYPES[index].value === box.type));
        textField.wrap.classList.toggle("pm-layout-ghost", box.type !== "text");
        textInput.disabled = box.type !== "text";
        setValue(textInput, box.text || "");
        setValue(descInput, box.desc || "");
        boxPalette.set(box.palette || []);
        renderThumb(box);
    }

    function syncOptions(select, values) {
        if (select.options.length !== values.length) select.replaceChildren(...values.map((value) => new Option(value, value)));
    }

    function renderSettings(W, H) {
        const aspectWidget = getWidget(node, "aspect_ratio");
        syncOptions(aspectSelect, aspectWidget?.options?.values || []);
        aspectSelect.value = String(aspectWidget?.value ?? "");
        setValue(megapixelsInput, String(getWidget(node, "megapixels")?.value ?? 1));
        importToggle.checked = !!getWidget(node, "use_import_json")?.value;
        targetLabel.textContent = `Target: ${W} × ${H}`;
    }

    function renderScene() {
        setValue(scenePrompt, String(getWidget(node, "prompt")?.value ?? ""));
        setValue(sceneMeta, String(getWidget(node, "meta")?.value ?? ""));
        const canvasWidget = getWidget(node, "canvas");
        syncOptions(sceneCanvas, canvasWidget?.options?.values || []);
        sceneCanvas.value = String(canvasWidget?.value ?? "none");
        scenePalette.set(state.palette);
    }

    function render() {
        const locked = isLocked();
        const [W, H] = canvasSize(node);
        selected = locked ? -1 : Math.min(selected, state.boxes.length - 1);
        banner.style.display = locked ? "" : "none";
        navRow.style.display = locked ? "none" : "";
        boxFields.disabled = locked;
        sceneFields.disabled = locked;
        statusSize.textContent = `${W} × ${H} · ${state.boxes.length} box${state.boxes.length === 1 ? "" : "es"}`;
        statusHint.textContent = locked ? "" : "Drag to draw a box";
        renderSettings(W, H);
        renderToolbar();
        renderNav();
        renderBox();
        renderScene();
        schedulePaint();
    }

    // Grow the node until the whole editor fits, with the canvas at its minimum height.
    function fitNode() {
        const hostHeight = host.offsetHeight;
        if (fullscreen || !hostHeight) return;
        minHeight = Math.ceil(root.scrollHeight + 2 - stage.offsetHeight + STAGE_MIN_HEIGHT);
        // Skip a retry when the last resize did not change the widget height.
        if (minHeight - hostHeight > 1 && hostHeight !== fitHostHeight) {
            fitHostHeight = hostHeight;
            node.setSize([node.size[0], node.size[1] + minHeight - hostHeight]);
            app.graph?.setDirtyCanvas(true, true);
        }
    }

    function schedulePaint() {
        if (frame === null) frame = requestAnimationFrame(paint);
    }

    const toView = (box) => ({ x: view.x + box.x * view.w, y: view.y + box.y * view.h, w: box.w * view.w, h: box.h * view.h });

    function handlesOf(r) {
        const xs = { w: r.x, "": r.x + r.w / 2, e: r.x + r.w };
        const ys = { n: r.y, "": r.y + r.h / 2, s: r.y + r.h };
        return ["nw", "n", "ne", "e", "se", "s", "sw", "w"].map((key) => ({
            key,
            x: xs[key.match(/[we]/)?.[0] || ""],
            y: ys[key.match(/[ns]/)?.[0] || ""],
        }));
    }

    function paint() {
        frame = null;
        fitNode();
        const width = stage.clientWidth;
        const height = stage.clientHeight;
        if (!width || !height) return;
        const zoom = stage.getBoundingClientRect().width / width || 1;
        const resolution = Math.min(4, (window.devicePixelRatio || 1) * Math.max(1, zoom));
        const pixelWidth = Math.round(width * resolution);
        const pixelHeight = Math.round(height * resolution);
        if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
            canvas.width = pixelWidth;
            canvas.height = pixelHeight;
        }
        const ctx = canvas.getContext("2d");
        ctx.setTransform(resolution, 0, 0, resolution, 0, 0);
        paintResolution = resolution;
        ctx.clearRect(0, 0, width, height);

        const [W, H] = canvasSize(node);
        const scale = Math.min((width - 16) / W, (height - 16) / H);
        view = { x: (width - W * scale) / 2, y: (height - H * scale) / 2, w: W * scale, h: H * scale };
        ctx.fillStyle = UI.cardBg;
        ctx.fillRect(view.x, view.y, view.w, view.h);
        if (backgroundReady) {
            // Center-crop like the backend resize, in case the ratio changed since the last run.
            const fit = Math.max(view.w / background.naturalWidth, view.h / background.naturalHeight);
            const sw = view.w / fit;
            const sh = view.h / fit;
            ctx.drawImage(background, (background.naturalWidth - sw) / 2, (background.naturalHeight - sh) / 2, sw, sh, view.x, view.y, view.w, view.h);
        }
        if (showGrid) {
            ctx.lineWidth = 1;
            for (let i = 1; i < 10; i++) {
                ctx.strokeStyle = i === 5 ? "rgba(255, 255, 255, 0.28)" : "rgba(255, 255, 255, 0.12)";
                ctx.beginPath();
                ctx.moveTo(view.x + view.w * i / 10, view.y);
                ctx.lineTo(view.x + view.w * i / 10, view.y + view.h);
                ctx.moveTo(view.x, view.y + view.h * i / 10);
                ctx.lineTo(view.x + view.w, view.y + view.h * i / 10);
                ctx.stroke();
            }
        }
        ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
        ctx.lineWidth = 1;
        ctx.strokeRect(view.x + 0.5, view.y + 0.5, view.w - 1, view.h - 1);
        state.boxes.forEach((box, index) => { if (index !== selected) drawBox(ctx, box, index, false); });
        if (selected >= 0) drawBox(ctx, state.boxes[selected], selected, true);
        if (drag?.mode === "draw") {
            const x = Math.min(drag.start.x, drag.current.x);
            const y = Math.min(drag.start.y, drag.current.y);
            const r = toView({ x, y, w: Math.abs(drag.current.x - drag.start.x), h: Math.abs(drag.current.y - drag.start.y) });
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = UI.accentBorder;
            ctx.lineWidth = 1.5;
            ctx.strokeRect(r.x, r.y, r.w, r.h);
            ctx.setLineDash([]);
        }
    }

    function drawBox(ctx, box, index, active) {
        const rgb = boxRgb(box, index);
        const r = toView(box);
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
        ctx.fillStyle = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 140 ? "#0b0f14" : "#ffffff";
        ctx.fillText(tag, r.x + 4, r.y + 11);
        (box.palette || []).slice(0, BOX_PALETTE_MAX).forEach((color, i) => {
            ctx.fillStyle = color;
            ctx.fillRect(r.x + tagWidth + 3 + i * 10, r.y + 3, 8, 8);
        });
        const size = (fullscreen ? FULLSCREEN_TEXT_SIZES : TEXT_SIZES)[textSizeKey()];
        if (size && r.w > 30 && r.h > 30) {
            const maxY = r.y + r.h - 4;
            let y = r.y + 19 + size;
            // Shadow sizes ignore the canvas transform, so scale them to the backing resolution.
            ctx.shadowColor = "rgba(0, 0, 0, 0.95)";
            ctx.shadowBlur = Math.max(3, size / 3) * paintResolution;
            ctx.shadowOffsetY = paintResolution;
            if (box.type === "text" && box.text) {
                ctx.font = `bold ${size + 3}px ${DISPLAY_FONT}`;
                ctx.fillStyle = "#ffffff";
                y = drawWrapped(ctx, `"${box.text}"`, r.x + 4, y + 2, r.w - 8, maxY, size + 6) + 3;
            }
            if (box.desc) {
                ctx.font = `500 ${size}px ${BODY_FONT}`;
                ctx.fillStyle = "#f1f5fb";
                drawWrapped(ctx, String(box.desc), r.x + 4, y, r.w - 8, maxY, Math.round(size * 1.3));
            }
        }
        ctx.restore();

        if (active && !isLocked()) {
            ctx.fillStyle = "#ffffff";
            ctx.strokeStyle = rgba(rgb.map((c) => Math.round(c * 0.6)));
            ctx.lineWidth = 1;
            for (const handle of handlesOf(r)) {
                ctx.fillRect(handle.x - 3.5, handle.y - 3.5, 7, 7);
                ctx.strokeRect(handle.x - 3.5, handle.y - 3.5, 7, 7);
            }
        }
    }

    function localPoint(event) {
        const rect = stage.getBoundingClientRect();
        return {
            x: (event.clientX - rect.left) * (stage.clientWidth / rect.width),
            y: (event.clientY - rect.top) * (stage.clientHeight / rect.height),
        };
    }

    const toFraction = (p) => ({ x: (p.x - view.x) / view.w, y: (p.y - view.y) / view.h });
    const contains = (box, f) => f.x >= box.x && f.x <= box.x + box.w && f.y >= box.y && f.y <= box.y + box.h;

    function hitHandle(p) {
        if (selected < 0 || isLocked()) return null;
        const handle = handlesOf(toView(state.boxes[selected]))
            .find((h) => Math.abs(p.x - h.x) <= HANDLE_SIZE && Math.abs(p.y - h.y) <= HANDLE_SIZE);
        return handle?.key || null;
    }

    // The selected box wins so it stays draggable under overlapping boxes.
    function hitBox(p) {
        const f = toFraction(p);
        if (selected >= 0 && contains(state.boxes[selected], f)) return selected;
        for (let i = state.boxes.length - 1; i >= 0; i--) {
            if (contains(state.boxes[i], f)) return i;
        }
        return -1;
    }

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

    stage.addEventListener("pointerdown", (event) => {
        if (event.button !== 0 || isLocked()) return;
        event.preventDefault();
        stage.focus({ preventScroll: true });
        const p = localPoint(event);
        const handle = hitHandle(p);
        const hit = event.shiftKey ? -1 : hitBox(p);
        const start = toFraction(p);
        if (handle) {
            drag = { mode: "resize", handle, start, origin: { ...state.boxes[selected] }, before: snapshot() };
        } else if (hit >= 0) {
            selected = hit;
            drag = { mode: "move", start, origin: { ...state.boxes[hit] }, before: snapshot() };
            render();
        } else {
            const point = { x: clamp(start.x, 0, 1), y: clamp(start.y, 0, 1) };
            drag = { mode: "draw", start: point, current: point, before: snapshot() };
        }
        stage.setPointerCapture(event.pointerId);
    });

    stage.addEventListener("pointermove", (event) => {
        const p = localPoint(event);
        if (!drag) {
            const handle = hitHandle(p);
            stage.style.cursor = isLocked() ? "default" : handle ? RESIZE_CURSORS[handle] : hitBox(p) >= 0 ? "move" : "crosshair";
            return;
        }
        const f = toFraction(p);
        const dx = f.x - drag.start.x;
        const dy = f.y - drag.start.y;
        if (drag.mode === "draw") {
            drag.current = { x: clamp(f.x, 0, 1), y: clamp(f.y, 0, 1) };
        } else {
            const box = state.boxes[selected];
            if (drag.mode === "move") {
                box.x = clamp(drag.origin.x + dx, 0, 1 - drag.origin.w);
                box.y = clamp(drag.origin.y + dy, 0, 1 - drag.origin.h);
            } else {
                resizeBox(box, drag.origin, drag.handle, dx, dy);
            }
            renderBox();
        }
        schedulePaint();
    });

    function endDrag() {
        if (!drag) return;
        const finished = drag;
        drag = null;
        if (finished.mode === "draw") {
            const x = Math.min(finished.start.x, finished.current.x);
            const y = Math.min(finished.start.y, finished.current.y);
            const w = Math.abs(finished.current.x - finished.start.x);
            const h = Math.abs(finished.current.y - finished.start.y);
            if (w >= MIN_BOX && h >= MIN_BOX) {
                pushUndo(finished.before);
                state.boxes.push(newBox({ x, y, w, h }));
                selected = state.boxes.length - 1;
                persist();
            } else {
                selected = -1;
            }
        } else if (snapshot() !== finished.before) {
            pushUndo(finished.before);
            persist();
        }
        render();
    }
    stage.addEventListener("pointerup", endDrag);
    stage.addEventListener("pointercancel", endDrag);

    root.addEventListener("keydown", (event) => {
        if (event.target.closest("input, textarea, select")) return;
        const key = event.key.toLowerCase();
        const mod = event.ctrlKey || event.metaKey;
        if (key === "delete" || key === "backspace") deleteSelected();
        else if (mod && key === "z") travel(event.shiftKey ? redo : undo, event.shiftKey ? undo : redo);
        else if (mod && key === "y") travel(redo, undo);
        else if (mod && key === "d") duplicateSelected();
        else return;
        event.preventDefault();
        event.stopPropagation();
    });

    function openFullscreen() {
        if (fullscreen) return;
        const overlay = el("div", "pm-layout-overlay", document.body);
        const onKey = (event) => {
            if (event.key !== "Escape" || picking) return;
            event.preventDefault();
            event.stopPropagation();
            closeFullscreen();
        };
        document.addEventListener("keydown", onKey, true);
        fullscreen = { overlay, onKey };
        overlay.appendChild(root);
        root.classList.add("pm-layout-full");
        side.prepend(settings, sceneCard);
        placeholder.style.display = "flex";
        render();
    }

    function closeFullscreen() {
        if (!fullscreen) return;
        document.removeEventListener("keydown", fullscreen.onKey, true);
        root.classList.remove("pm-layout-full");
        main.prepend(settings, sceneCard);
        host.appendChild(root);
        placeholder.style.display = "";
        fullscreen.overlay.remove();
        fullscreen = null;
        render();
    }

    function setBackground(info) {
        backgroundReady = false;
        if (info?.filename) {
            const params = new URLSearchParams({ filename: info.filename, subfolder: info.subfolder || "", type: info.type || "temp" });
            background.src = api.apiURL(`/view?${params}`);
        } else {
            background.removeAttribute("src");
        }
        schedulePaint();
    }

    const observer = new ResizeObserver(() => schedulePaint());
    for (const section of [stage, main, side]) observer.observe(section);

    return {
        host,
        render,
        minHeight: () => minHeight,
        load() {
            state = readState(node);
            selected = state.boxes.length ? 0 : -1;
            undo = [];
            redo = [];
            setBackground(node.properties?.[BG_PROP]);
            render();
        },
        applyExecution(message) {
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
                setBackground(images[0]);
            }
            const imported = message?.pm_layout_import?.[0];
            if (imported) {
                const next = { boxes: imported.boxes || [], palette: imported.palette || [] };
                if (JSON.stringify(next) !== snapshot()) {
                    pushUndo(snapshot());
                    state = next;
                    selected = state.boxes.length ? 0 : -1;
                    persist();
                }
                setWidgetValue("prompt", imported.prompt ?? "");
                setWidgetValue("meta", imported.meta ?? "");
                setWidgetValue("canvas", imported.canvas || "none");
            }
            render();
        },
        dispose() {
            closeColorPicker();
            closeFullscreen();
            observer.disconnect();
        },
    };
}

app.registerExtension({
    name: "PromptManager.PromptLayoutBuilder",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            for (const name of HIDDEN_WIDGETS) {
                const widget = getWidget(node, name);
                if (!widget) continue;
                widget.hidden = true;
                widget.computeSize = () => [0, -4];
                const element = widget.element || widget.inputEl;
                if (element) element.style.display = "none";
            }
            const editor = createLayoutEditor(node);
            node._layoutEditor = editor;
            node.addDOMWidget("layout_editor", "div", editor.host, {
                serialize: false,
                hideOnZoom: false,
                getMinHeight: () => editor.minHeight(),
            });
            const [minWidth, minHeight] = node.computeSize();
            node.setSize([Math.max(MIN_NODE_WIDTH, minWidth, node.size[0]), Math.max(minHeight, node.size[1])]);
            editor.load();
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            this._layoutEditor?.load();
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            this._layoutEditor?.render();
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            this._layoutEditor?.applyExecution(message);
            return result;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this._layoutEditor?.dispose();
            return onRemoved?.apply(this, arguments);
        };
    },
});
