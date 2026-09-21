import { app } from "../../scripts/app.js";
import { PM_UI_PALETTE as UI } from "./ui_palette.js";
import { DEFAULT_THUMBNAIL } from "./prompt_manager_advanced.js";
import { showThumbnailBrowser } from "./prompt_browser.js";
import { getPromptTypeChoices } from "./prompt_browser_edit.js";
import { loadComposerPrompts, getComposerEntry, COMPOSER_ENDPOINT_PREFIX } from "./prompt_composer_common.js";

const PARTS_PROP_KEY = "prompt_composer_parts";
const THUMB_ZOOM_PROP_KEY = "prompt_composer_thumb_zoom";
const OUTPUT_FORMAT_PROP_KEY = "prompt_composer_output_format";
const COMPOSE_POSITION_PROP_KEY = "prompt_composer_compose_position";
const GENERATION_MODE_PROP_KEY = "prompt_composer_generation_mode";
const PARTS_WIDGET_NAME = "parts_data";
const OUTPUT_FORMAT_WIDGET_NAME = "output_format";
const COMPOSE_POSITION_WIDGET_NAME = "compose_position";
const GENERATION_MODE_WIDGET_NAME = "generation_mode";
const MIN_NODE_WIDTH = 500;
const MIN_NODE_HEIGHT = 600;
const HOLD_TO_DRAG_MS = 140;
const COMPOSER_DRAG_STYLE_ID = "pm-composer-drag-style";
const THUMB_BASE_WIDTH = 128;
const DEFAULT_THUMB_ZOOM = 1.0;
const RESET_THUMB_ZOOM = 1.0;
const MIN_THUMB_ZOOM = 0.75;
const MAX_THUMB_ZOOM = 1.5;
const THUMB_ZOOM_STEPS = [0.75, 1.0, 1.25, 1.5];
const GRID_GAP = 8;
const CARD_META_HEIGHT = 62;
const CARD_META_HEIGHT_VIDEO = 36;
const NODE_CHROME_HEIGHT = 86;
const SCROLLER_PADDING_TOP = 8;
const SCROLLER_PADDING_BOTTOM = 24;
const SUBJECT_NONE = 0;
const SUBJECT_MIN = 1;
const SUBJECT_MAX = 16;
const NON_SUBJECT_PROMPT_TYPES = new Set([
    "style",
    "motion",
    "lighting",
    "ambience",
    "composition",
    "camera",
    "motion",
    "soundscape",
    "dialogue",
    "weather",
]);
const SUBJECT_ACCENTS = [
    { border: "hsla(8, 84%, 64%, 0.95)", soft: "hsla(8, 84%, 64%, 0.18)", strong: "hsla(8, 84%, 48%, 0.95)", text: "hsl(8, 100%, 96%)" },
    { border: "hsla(40, 92%, 60%, 0.95)", soft: "hsla(40, 92%, 60%, 0.18)", strong: "hsla(40, 92%, 44%, 0.95)", text: "hsl(48, 100%, 96%)" },
    { border: "hsla(92, 72%, 56%, 0.95)", soft: "hsla(92, 72%, 56%, 0.18)", strong: "hsla(92, 72%, 40%, 0.95)", text: "hsl(92, 100%, 96%)" },
    { border: "hsla(155, 72%, 48%, 0.95)", soft: "hsla(155, 72%, 48%, 0.18)", strong: "hsla(155, 72%, 34%, 0.95)", text: "hsl(155, 100%, 96%)" },
    { border: "hsla(205, 88%, 60%, 0.95)", soft: "hsla(205, 88%, 60%, 0.18)", strong: "hsla(205, 88%, 44%, 0.95)", text: "hsl(205, 100%, 96%)" },
    { border: "hsla(248, 80%, 68%, 0.95)", soft: "hsla(248, 80%, 68%, 0.18)", strong: "hsla(248, 80%, 52%, 0.95)", text: "hsl(248, 100%, 97%)" },
    { border: "hsla(294, 72%, 64%, 0.95)", soft: "hsla(294, 72%, 64%, 0.18)", strong: "hsla(294, 72%, 48%, 0.95)", text: "hsl(294, 100%, 97%)" },
    { border: "hsla(332, 78%, 62%, 0.95)", soft: "hsla(332, 78%, 62%, 0.18)", strong: "hsla(332, 78%, 46%, 0.95)", text: "hsl(332, 100%, 97%)" },
];

function getWidgetByName(node, name) {
    return node.widgets?.find((w) => w.name === name) || null;
}

function hideWidget(widget) {
    if (!widget) return;
    widget.type = "converted-widget";
    widget.computeSize = () => [0, -4];
    widget.hidden = true;
    widget.draw = function () {};
}

function showComposerTypePicker(anchorEvent, anchorElement = null) {
    return new Promise((resolve) => {
        const existing = document.querySelector('.pm-composer-type-picker');
        if (existing) existing.remove();

        const menu = document.createElement('div');
        menu.className = 'pm-composer-type-picker';
        menu.style.cssText = `
            position: fixed;
            background: ${UI.panel || "#1f2937"};
            border: 1px solid ${UI.inputBorder || "#445064"};
            border-radius: 8px;
            padding: 6px 0;
            z-index: 10001;
            min-width: 180px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.45);
        `;

        const addItem = (value, label) => {
            const item = document.createElement('div');
            item.textContent = label;
            item.style.cssText = `
                padding: 8px 14px;
                color: ${UI.textPrimary || "#d1d5db"};
                cursor: pointer;
                font-size: 13px;
                white-space: nowrap;
            `;
            item.onmouseover = () => {
                item.style.background = UI.accentSoft || 'rgba(56, 130, 246, 0.16)';
            };
            item.onmouseout = () => {
                item.style.background = 'transparent';
            };
            item.onclick = () => {
                cleanup();
                resolve(value);
            };
            menu.appendChild(item);
        };

        addItem('__all__', 'All');
        for (const choice of getPromptTypeChoices()) {
            if (!choice?.value) continue;
            addItem(choice.value, choice.label || choice.value);
        }

        const clickX = Number(anchorEvent?.clientX);
        const clickY = Number(anchorEvent?.clientY);
        const hasPointerPosition = Number.isFinite(clickX) && Number.isFinite(clickY);
        const rect = anchorElement?.getBoundingClientRect?.();
        const left = hasPointerPosition
            ? Math.min(clickX + 4, window.innerWidth - 200)
            : (rect ? Math.min(rect.left, window.innerWidth - 200) : Math.max(16, (window.innerWidth - 180) / 2));
        const top = hasPointerPosition
            ? Math.min(clickY + 4, window.innerHeight - 320)
            : (rect ? Math.min(rect.bottom + 6, window.innerHeight - 320) : Math.max(16, (window.innerHeight - 320) / 2));
        menu.style.left = `${Math.max(8, left)}px`;
        menu.style.top = `${Math.max(8, top)}px`;
        document.body.appendChild(menu);

        const onPointerDown = (event) => {
            if (!menu.contains(event.target)) {
                cleanup();
                resolve(null);
            }
        };

        const onKeyDown = (event) => {
            if (event.key === 'Escape') {
                cleanup();
                resolve(null);
            }
        };

        const cleanup = () => {
            document.removeEventListener('mousedown', onPointerDown, true);
            document.removeEventListener('keydown', onKeyDown, true);
            if (menu.parentNode) menu.parentNode.removeChild(menu);
        };

        setTimeout(() => {
            document.addEventListener('mousedown', onPointerDown, true);
            document.addEventListener('keydown', onKeyDown, true);
        }, 0);
    });
}

function readToggleValue(node, widgetName, propKey, fallbackValue) {
    const propValue = String(node.properties?.[propKey] ?? "").trim();
    if (propValue) return propValue;
    const widgetValue = String(getWidgetByName(node, widgetName)?.value ?? "").trim();
    return widgetValue || fallbackValue;
}

function writeToggleValue(node, widgetName, propKey, value) {
    const normalized = String(value || "").trim();
    const widget = getWidgetByName(node, widgetName);
    if (widget) {
        widget.value = normalized;
    }
    node.properties = node.properties || {};
    node.properties[propKey] = normalized;
    app.graph.setDirtyCanvas(true, true);
}

function readOutputFormat(node) {
    return readToggleValue(node, OUTPUT_FORMAT_WIDGET_NAME, OUTPUT_FORMAT_PROP_KEY, "text");
}

function writeOutputFormat(node, value) {
    writeToggleValue(node, OUTPUT_FORMAT_WIDGET_NAME, OUTPUT_FORMAT_PROP_KEY, value);
}

function readComposePosition(node) {
    return readToggleValue(node, COMPOSE_POSITION_WIDGET_NAME, COMPOSE_POSITION_PROP_KEY, "before");
}

function writeComposePosition(node, value) {
    writeToggleValue(node, COMPOSE_POSITION_WIDGET_NAME, COMPOSE_POSITION_PROP_KEY, value);
}

function readGenerationMode(node) {
    return readToggleValue(node, GENERATION_MODE_WIDGET_NAME, GENERATION_MODE_PROP_KEY, "image");
}

function writeGenerationMode(node, value) {
    writeToggleValue(node, GENERATION_MODE_WIDGET_NAME, GENERATION_MODE_PROP_KEY, value);
}

function snapThumbZoom(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return DEFAULT_THUMB_ZOOM;
    let nearest = THUMB_ZOOM_STEPS[0];
    let bestDistance = Math.abs(numeric - nearest);
    for (const step of THUMB_ZOOM_STEPS) {
        const distance = Math.abs(numeric - step);
        if (distance < bestDistance) {
            nearest = step;
            bestDistance = distance;
        }
    }
    return nearest;
}

function clampThumbZoom(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return DEFAULT_THUMB_ZOOM;
    const bounded = Math.max(MIN_THUMB_ZOOM, Math.min(MAX_THUMB_ZOOM, numeric));
    return snapThumbZoom(bounded);
}

function readThumbZoom(node) {
    const raw = node.properties?.[THUMB_ZOOM_PROP_KEY];
    return clampThumbZoom(raw ?? DEFAULT_THUMB_ZOOM);
}

function writeThumbZoom(node, zoom) {
    node.properties = node.properties || {};
    node.properties[THUMB_ZOOM_PROP_KEY] = clampThumbZoom(zoom);
    app.graph.setDirtyCanvas(true, true);
}

function ensureComposerDragStyles() {
    if (document.getElementById(COMPOSER_DRAG_STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = COMPOSER_DRAG_STYLE_ID;
    style.textContent = `
.pm-composer-card-drag-active {
    outline: 2px dotted rgba(255, 255, 255, 0.95);
    outline-offset: -2px;
    animation: pm-composer-drag-pulse 0.85s linear infinite;
}

@keyframes pm-composer-drag-pulse {
    0% {
        outline-color: rgba(255, 255, 255, 0.45);
    }
    50% {
        outline-color: rgba(255, 255, 255, 1);
    }
    100% {
        outline-color: rgba(255, 255, 255, 0.45);
    }
}

.pm-composer-zoom-row {
    position: absolute;
    right: 2px;
    bottom: 2px;
    display: flex;
    align-items: center;
    padding: 1px 3px;
    border: 1px solid rgba(116, 131, 154, 0.55);
    border-radius: 0;
    background: rgba(17, 22, 30, 0.82);
    box-sizing: border-box;
    flex-shrink: 0;
    z-index: 2;
}

.pm-composer-zoom-slider {
    width: 120px;
    margin: 0;
    appearance: none;
    -webkit-appearance: none;
    background: transparent;
    cursor: pointer;
}

.pm-composer-zoom-slider:focus {
    outline: none;
}

.pm-composer-zoom-slider::-webkit-slider-runnable-track {
    height: 1px;
    background: rgba(229, 231, 235, 0.9);
    border-radius: 0;
}

.pm-composer-zoom-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 7px;
    height: 7px;
    background: #e5e7eb;
    border: 1px solid rgba(15, 23, 42, 0.9);
    border-radius: 0;
    margin-top: -3px;
}

.pm-composer-zoom-slider::-moz-range-track {
    height: 1px;
    background: rgba(229, 231, 235, 0.9);
    border: none;
    border-radius: 0;
}

.pm-composer-zoom-slider::-moz-range-thumb {
    width: 7px;
    height: 7px;
    background: #e5e7eb;
    border: 1px solid rgba(15, 23, 42, 0.9);
    border-radius: 0;
}
`;
    document.head.appendChild(style);
}

function clampStrength(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return 1.0;
    return Math.max(0, Math.min(5, numeric));
}

function clampSubjectNumber(value, fallback = SUBJECT_MIN) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return fallback;
    return Math.max(SUBJECT_NONE, Math.min(SUBJECT_MAX, Math.round(numeric)));
}

function padSubjectNumber(value) {
    if (clampSubjectNumber(value) === SUBJECT_NONE) return "NS";
    return String(clampSubjectNumber(value)).padStart(2, "0");
}

function getSubjectAccent(subjectNumber) {
    const normalized = clampSubjectNumber(subjectNumber);
    if (normalized === SUBJECT_NONE) {
        return {
            border: "rgba(122, 131, 148, 0.88)",
            soft: "rgba(122, 131, 148, 0.12)",
            strong: "rgba(80, 88, 104, 0.95)",
            text: "#eef2f7",
        };
    }
    return SUBJECT_ACCENTS[(normalized - SUBJECT_MIN) % SUBJECT_ACCENTS.length];
}

function normalizePart(part) {
    const category = String(part?.category || "").trim();
    const promptsInput = Array.isArray(part?.prompts) ? part.prompts : [];
    const prompts = promptsInput
        .map((name) => String(name || "").trim())
        .filter((name) => name.length > 0);
    return {
        category,
        prompts,
        strength: clampStrength(part?.strength ?? 1.0),
        subject_number: clampSubjectNumber(part?.subject_number ?? part?.subject ?? SUBJECT_MIN),
        subject_locked: !!(part?.subject_locked ?? part?.subject_manual ?? false),
        muted: part?.muted === true,
    };
}

function resolveSubjectAssignments(parts) {
    const normalizedParts = Array.isArray(parts) ? parts.map((part) => normalizePart(part)) : [];
    let currentSubject = SUBJECT_MIN;
    return normalizedParts.map((part) => {
        if (part.muted) {
            return {
                ...part,
                effective_subject_number: clampSubjectNumber(currentSubject, SUBJECT_MIN),
            };
        }
        if (part.subject_locked && part.subject_number !== SUBJECT_NONE) {
            currentSubject = clampSubjectNumber(part.subject_number, currentSubject);
        }
        const effectiveSubjectNumber = part.subject_locked && part.subject_number === SUBJECT_NONE
            ? SUBJECT_NONE
            : clampSubjectNumber(
                part.subject_locked ? part.subject_number : currentSubject,
                currentSubject,
            );
        if (effectiveSubjectNumber !== SUBJECT_NONE) {
            currentSubject = effectiveSubjectNumber;
        }
        return {
            ...part,
            effective_subject_number: effectiveSubjectNumber,
        };
    });
}

function getInheritedSubjectDefaults(parts) {
    const resolved = resolveSubjectAssignments(parts);
    if (!resolved.length) {
        return { subject_number: SUBJECT_MIN, subject_locked: false };
    }
    return {
        subject_number: resolved[resolved.length - 1].effective_subject_number,
        subject_locked: false,
    };
}

function nextSubjectNumber(value, delta) {
    const current = clampSubjectNumber(value);
    const span = SUBJECT_MAX - SUBJECT_MIN + 1;
    const offset = ((current - SUBJECT_MIN + delta) % span + span) % span;
    return SUBJECT_MIN + offset;
}

function getCategoryPromptType(node, category) {
    const raw = node?.prompts?.[category]?._prompt_type_;
    return String(raw || "").trim().toLowerCase();
}

function categoryShouldBeNonSubject(node, category) {
    return NON_SUBJECT_PROMPT_TYPES.has(getCategoryPromptType(node, category));
}

function inferPartSubjectState(node, category, basePart = null, inheritedDefaults = null) {
    if (categoryShouldBeNonSubject(node, category)) {
        return {
            subject_number: SUBJECT_NONE,
            subject_locked: true,
        };
    }

    if (basePart && basePart.subject_locked && basePart.subject_number !== SUBJECT_NONE) {
        return {
            subject_number: clampSubjectNumber(basePart.subject_number),
            subject_locked: true,
        };
    }

    if (basePart && !basePart.subject_locked && basePart.subject_number !== SUBJECT_NONE) {
        return {
            subject_number: clampSubjectNumber(basePart.subject_number),
            subject_locked: false,
        };
    }

    const inherited = inheritedDefaults || { subject_number: SUBJECT_MIN, subject_locked: false };
    return {
        subject_number: clampSubjectNumber(inherited.subject_number ?? SUBJECT_MIN),
        subject_locked: false,
    };
}

function parseParts(raw) {
    try {
        const parsed = JSON.parse(raw || "[]");
        if (!Array.isArray(parsed)) return [];
        return parsed
            .map((part) => normalizePart(part))
            .filter((part) => part.prompts.length > 0 || part.category.length > 0);
    } catch {
        return [];
    }
}

function serializeParts(parts) {
    const normalized = (Array.isArray(parts) ? parts : [])
        .map((part) => normalizePart(part))
        .filter((part) => part.prompts.length > 0 || part.category.length > 0);
    return JSON.stringify(normalized);
}

function getPartsWidget(node) {
    return getWidgetByName(node, PARTS_WIDGET_NAME);
}

function readParts(node) {
    const widget = getPartsWidget(node);
    const propRaw = String(node.properties?.[PARTS_PROP_KEY] || "").trim();
    if (propRaw) return parseParts(propRaw);
    return parseParts(widget?.value || "[]");
}

function writeParts(node, parts) {
    const payload = serializeParts(parts);
    const widget = getPartsWidget(node);
    if (widget) {
        widget.value = payload;
    }
    node.properties = node.properties || {};
    node.properties[PARTS_PROP_KEY] = payload;
    app.graph.setDirtyCanvas(true, true);
}

function getSelectedPartIndices(node, partCount = null) {
    const raw = node?._composerSelectedPartIndices;
    const values = raw instanceof Set ? Array.from(raw) : (Array.isArray(raw) ? raw : []);
    return values
        .map((value) => Number(value))
        .filter((value) => Number.isInteger(value) && value >= 0 && (partCount === null || value < partCount))
        .sort((a, b) => a - b);
}

function setSelectedPartIndices(node, indices, partCount = null) {
    const next = new Set(
        (Array.isArray(indices) ? indices : [])
            .map((value) => Number(value))
            .filter((value) => Number.isInteger(value) && value >= 0 && (partCount === null || value < partCount))
    );
    node._composerSelectedPartIndices = next;
}

function clearSelectedPartIndices(node) {
    node._composerSelectedPartIndices = new Set();
}

function toggleSelectedPartIndex(node, index, partCount = null) {
    const next = new Set(getSelectedPartIndices(node, partCount));
    if (next.has(index)) {
        next.delete(index);
    } else {
        next.add(index);
    }
    node._composerSelectedPartIndices = next;
    return next;
}

function buildPartsFromBrowserSelection(node, selection, inheritedSubject, basePart = null, preferredCategory = "") {
    if (!selection || !Array.isArray(selection.prompts) || selection.prompts.length === 0) {
        return [];
    }

    const normalizedBasePart = basePart ? normalizePart(basePart) : null;
    const selectionMode = String(selection.selectionMode || "combine").trim().toLowerCase();
    const buildPart = (category, prompts) => {
        const subjectState = inferPartSubjectState(node, category, normalizedBasePart, inheritedSubject);
        return normalizePart({
            category,
            prompts,
            strength: normalizedBasePart?.strength ?? 1.0,
            muted: normalizedBasePart?.muted === true,
            subject_number: subjectState.subject_number,
            subject_locked: subjectState.subject_locked,
        });
    };

    if (selection.selectionsByCategory && Object.keys(selection.selectionsByCategory).length > 0) {
        const entries = Object.entries(selection.selectionsByCategory)
            .filter(([, prompts]) => Array.isArray(prompts) && prompts.length > 0)
            .sort((a, b) => {
                if (!preferredCategory) return 0;
                if (a[0] === preferredCategory) return -1;
                if (b[0] === preferredCategory) return 1;
                return 0;
            });
        if (selectionMode === "split") {
            return entries.flatMap(([category, prompts]) => prompts.map((promptName) => buildPart(category, [promptName])));
        }
        return entries.map(([category, prompts]) => buildPart(category, prompts));
    }

    const category = selection.category || normalizedBasePart?.category || "";
    const prompts = selection.prompts.filter((name) => String(name || "").trim());
    if (selectionMode === "split") {
        return prompts.map((promptName) => buildPart(category, [promptName]));
    }
    return [buildPart(category, prompts)];
}

function ensureHiddenComposerWidgets(node) {
    hideWidget(getPartsWidget(node));
    hideWidget(getWidgetByName(node, OUTPUT_FORMAT_WIDGET_NAME));
    hideWidget(getWidgetByName(node, COMPOSE_POSITION_WIDGET_NAME));
    hideWidget(getWidgetByName(node, GENERATION_MODE_WIDGET_NAME));
}

function ensureComposerUi(node) {
    if (node._composerUiAttached) return;
    ensureComposerDragStyles();

    const root = document.createElement("div");
    root.style.cssText = `
        display: flex;
        flex-direction: column;
        width: 100%;
        height: 100%;
        min-height: 0;
        box-sizing: border-box;
        margin-top: -6px;
        padding: 0;
        overflow: hidden;
        position: relative;
    `;

    const switchRow = document.createElement("div");
    switchRow.style.cssText = `
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        min-height: 24px;
        margin: 0 0 8px 0;
        padding: 0 10px;
        border: 1px solid rgba(78, 90, 108, 0.72);
        border-radius: 10px;
        background: rgba(34, 39, 48, 0.98);
        box-sizing: border-box;
        flex: 0 0 auto;
    `;

    const createInlineSwitch = ({ title, leftLabel, rightLabel, getValue, onToggle, isRightActive }) => {
        const group = document.createElement("div");
        group.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            min-width: 0;
            flex: 1 1 0;
        `;
        group.title = title;

        const left = document.createElement("span");
        left.textContent = leftLabel;
        left.style.cssText = "font-size: 12px; color: #b9c2ce; white-space: nowrap; user-select: none;";

        const button = document.createElement("button");
        button.type = "button";
        button.style.cssText = `
            position: relative;
            width: 34px;
            height: 18px;
            border: 1px solid rgba(116, 131, 154, 0.7);
            border-radius: 999px;
            background: transparent;
            cursor: pointer;
            padding: 0;
            flex: 0 0 auto;
        `;

        const knob = document.createElement("span");
        knob.style.cssText = `
            position: absolute;
            top: 1px;
            left: 1px;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #f3f4f6;
            transition: transform 0.16s ease, background 0.16s ease;
            pointer-events: none;
        `;
        button.appendChild(knob);

        const right = document.createElement("span");
        right.textContent = rightLabel;
        right.style.cssText = "font-size: 12px; color: #b9c2ce; white-space: nowrap; user-select: none;";

        const sync = () => {
            const current = getValue();
            const active = typeof isRightActive === "function" ? !!isRightActive(current) : false;
            button.dataset.active = active ? "1" : "0";
            button.style.background = active ? "#2f6f92" : "transparent";
            knob.style.transform = active ? "translateX(16px)" : "translateX(0)";
            left.style.color = active ? "#8d97a5" : "#f3f4f6";
            right.style.color = active ? "#f3f4f6" : "#8d97a5";
        };

        button.onclick = (evt) => {
            evt.preventDefault();
            evt.stopPropagation();
            onToggle(getValue());
            sync();
            node._composerUiRender?.();
        };

        group.appendChild(left);
        group.appendChild(button);
        group.appendChild(right);
        return { group, sync };
    };

    const formatSwitch = createInlineSwitch({
        title: "Switch Prompt output between text and JSON",
        leftLabel: "TXT",
        rightLabel: "JSON",
        getValue: () => readOutputFormat(node),
        onToggle: (current) => writeOutputFormat(node, current === "json" ? "text" : "json"),
        isRightActive: (current) => current === "json",
    });
    const positionSwitch = createInlineSwitch({
        title: "Switch whether composed parts go before or after the incoming prompt",
        leftLabel: "Before",
        rightLabel: "After",
        getValue: () => readComposePosition(node),
        onToggle: (current) => writeComposePosition(node, current === "after" ? "before" : "after"),
        isRightActive: (current) => current === "after",
    });
    const generationModeSwitch = createInlineSwitch({
        title: "Switch whether Prompt Composer uses Image or Video LoRAs",
        leftLabel: "Image",
        rightLabel: "Video",
        getValue: () => readGenerationMode(node),
        onToggle: (current) => writeGenerationMode(node, current === "video" ? "image" : "video"),
        isRightActive: (current) => current === "video",
    });

    switchRow.appendChild(formatSwitch.group);
    switchRow.appendChild(positionSwitch.group);
    switchRow.appendChild(generationModeSwitch.group);
    root.appendChild(switchRow);

    const scroller = document.createElement("div");
    scroller.style.cssText = `
        flex: 1;
        min-height: 0;
        overflow-y: auto;
        overflow-x: hidden;
        background: ${UI.inputBg || "#1a1d22"};
        border: 2px solid ${UI.accentBorder || "hsl(208 73% 57% / 0.65)"};
        padding: ${SCROLLER_PADDING_TOP}px 8px ${SCROLLER_PADDING_BOTTOM}px 8px;
        box-sizing: border-box;
        scrollbar-width: thin;
    `;

    const grid = document.createElement("div");
    grid.style.cssText = `
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(96px, 1fr));
        gap: ${GRID_GAP}px;
        justify-content: start;
        align-content: start;
    `;

    scroller.appendChild(grid);
    root.appendChild(scroller);

    const zoomRow = document.createElement("div");
    zoomRow.className = "pm-composer-zoom-row";

    const zoomSlider = document.createElement("input");
    zoomSlider.type = "range";
    zoomSlider.min = String(Math.round(MIN_THUMB_ZOOM * 100));
    zoomSlider.max = String(Math.round(MAX_THUMB_ZOOM * 100));
    zoomSlider.step = "25";
    zoomSlider.value = String(Math.round(readThumbZoom(node) * 100));
    zoomSlider.title = "Thumbnail zoom (right-click to reset)";
    zoomSlider.className = "pm-composer-zoom-slider";

    const syncZoomLabel = () => {
        zoomSlider.value = String(Math.round(clampThumbZoom(Number(zoomSlider.value) / 100) * 100));
    };
    syncZoomLabel();

    zoomSlider.addEventListener("input", () => {
        const zoom = clampThumbZoom(Number(zoomSlider.value) / 100);
        writeThumbZoom(node, zoom);
        syncZoomLabel();
        render();
    });

    zoomSlider.addEventListener("contextmenu", (evt) => {
        evt.preventDefault();
        const zoom = RESET_THUMB_ZOOM;
        zoomSlider.value = String(Math.round(zoom * 100));
        writeThumbZoom(node, zoom);
        render();
    });

    zoomRow.appendChild(zoomSlider);
    root.appendChild(zoomRow);

    const removeContextMenu = () => {
        if (node._composerContextMenu && node._composerContextMenu.parentNode) {
            node._composerContextMenu.parentNode.removeChild(node._composerContextMenu);
        }
        node._composerContextMenu = null;
    };

    const mergeSelectedPromptParts = (indices = null) => {
        const parts = readParts(node);
        const selectedIndices = Array.isArray(indices) && indices.length > 0
            ? indices.filter((value) => Number.isInteger(value) && value >= 0 && value < parts.length).sort((a, b) => a - b)
            : getSelectedPartIndices(node, parts.length);
        if (selectedIndices.length < 2) return false;

        const selectedParts = selectedIndices.map((index) => normalizePart(parts[index]));
        const category = selectedParts[0]?.category || "";
        if (!selectedParts.every((part) => part.category === category)) {
            return false;
        }

        const seenPrompts = new Set();
        const mergedPrompts = [];
        selectedParts.forEach((part) => {
            part.prompts.forEach((promptName) => {
                const key = String(promptName || "").trim().toLowerCase();
                if (!key || seenPrompts.has(key)) return;
                seenPrompts.add(key);
                mergedPrompts.push(promptName);
            });
        });
        if (!mergedPrompts.length) return false;

        const mergedPart = normalizePart({
            ...selectedParts[0],
            prompts: mergedPrompts,
        });

        const selectedIndexSet = new Set(selectedIndices);
        const insertionIndex = selectedIndices[0];
        const next = [];
        parts.forEach((part, index) => {
            if (index === insertionIndex) {
                next.push(mergedPart);
                return;
            }
            if (selectedIndexSet.has(index)) {
                return;
            }
            next.push(part);
        });

        writeParts(node, next);
        setSelectedPartIndices(node, [insertionIndex], next.length);
        render();
        return true;
    };

    const splitPromptPart = (partIndex) => {
        const parts = readParts(node);
        const part = normalizePart(parts[partIndex]);
        if (!part || part.prompts.length < 2) return false;
        const splitParts = part.prompts.map((promptName) => normalizePart({
            ...part,
            prompts: [promptName],
        }));
        const next = [...parts];
        next.splice(partIndex, 1, ...splitParts);
        writeParts(node, next);
        setSelectedPartIndices(node, splitParts.map((_, offset) => partIndex + offset), next.length);
        render();
        return true;
    };

    const showPartContextMenu = (evt, partIndex) => {
        evt.preventDefault();
        evt.stopPropagation();
        removeContextMenu();

        const menu = document.createElement("div");
        menu.style.cssText = `
            position: fixed;
            left: ${evt.clientX}px;
            top: ${evt.clientY}px;
            background: ${UI.panel || "#2a2a2a"};
            border: 1px solid ${UI.inputBorder || "#444"};
            border-radius: 6px;
            padding: 4px 0;
            z-index: 10050;
            min-width: 130px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.45);
        `;

        const addItem = (label, onClick, disabled = false) => {
            const item = document.createElement("div");
            item.textContent = label;
            item.style.cssText = `
                padding: 7px 12px;
                font-size: 12px;
                color: ${disabled ? "#666" : "#ddd"};
                cursor: ${disabled ? "default" : "pointer"};
                user-select: none;
            `;
            if (!disabled) {
                item.onmouseenter = () => {
                    item.style.background = UI.accentSoft || "rgba(56,130,246,0.2)";
                };
                item.onmouseleave = () => {
                    item.style.background = "transparent";
                };
                item.onclick = () => {
                    removeContextMenu();
                    onClick();
                };
            }
            menu.appendChild(item);
        };

        const parts = readParts(node);
        const resolvedParts = resolveSubjectAssignments(parts);
        const resolvedPart = resolvedParts[partIndex] || null;
        const selectedIndices = getSelectedPartIndices(node, parts.length);
        const contextIndices = selectedIndices.length > 1 && selectedIndices.includes(partIndex)
            ? selectedIndices
            : [partIndex];
        const canMergeContextParts = contextIndices.length > 1 && contextIndices.every((index) => {
            const current = normalizePart(parts[index]);
            return current.category === normalizePart(parts[contextIndices[0]]).category;
        });

        addItem(
            resolvedPart?.effective_subject_number === SUBJECT_NONE
                ? "Not Subject"
                : resolvedPart?.subject_locked
                ? `Subject #${padSubjectNumber(resolvedPart.subject_number)} (custom)`
                : `Subject #${padSubjectNumber(resolvedPart?.effective_subject_number ?? SUBJECT_MIN)} (auto)`,
            () => {},
            true,
        );
        addItem(resolvedPart?.muted ? "Unmute Prompt" : "Mute Prompt", () => {
            const next = [...parts];
            if (!next[partIndex]) return;
            next[partIndex] = normalizePart({
                ...next[partIndex],
                muted: !resolvedPart?.muted,
            });
            writeParts(node, next);
            render();
        });
        if (contextIndices.length > 1) {
            addItem(`Merge Prompts (${contextIndices.length})`, () => {
                mergeSelectedPromptParts(contextIndices);
            }, !canMergeContextParts);
        }
        addItem("Split Prompts", () => {
            splitPromptPart(partIndex);
        }, (resolvedPart?.prompts?.length || 0) < 2);
        addItem("Subject +1", () => {
            const next = [...parts];
            if (!next[partIndex]) return;
            const baseSubject = resolvedPart?.effective_subject_number ?? SUBJECT_MIN;
            next[partIndex] = normalizePart({
                ...next[partIndex],
                subject_number: nextSubjectNumber(baseSubject, 1),
                subject_locked: true,
            });
            writeParts(node, next);
            render();
        });
        addItem("Subject -1", () => {
            const next = [...parts];
            if (!next[partIndex]) return;
            const baseSubject = resolvedPart?.effective_subject_number ?? SUBJECT_MIN;
            next[partIndex] = normalizePart({
                ...next[partIndex],
                subject_number: nextSubjectNumber(baseSubject, -1),
                subject_locked: true,
            });
            writeParts(node, next);
            render();
        });
        addItem("Subject Auto", () => {
            const next = [...parts];
            if (!next[partIndex]) return;
            next[partIndex] = normalizePart({
                ...next[partIndex],
                subject_number: resolvedPart?.effective_subject_number === SUBJECT_NONE
                    ? SUBJECT_MIN
                    : resolvedPart?.effective_subject_number,
                subject_locked: false,
            });
            writeParts(node, next);
            render();
        }, !resolvedPart?.subject_locked || resolvedPart?.effective_subject_number === SUBJECT_NONE);
        addItem("Not Subject", () => {
            const next = [...parts];
            if (!next[partIndex]) return;
            next[partIndex] = normalizePart({
                ...next[partIndex],
                subject_number: SUBJECT_NONE,
                subject_locked: true,
            });
            writeParts(node, next);
            render();
        }, resolvedPart?.effective_subject_number === SUBJECT_NONE);
        addItem("Delete", () => {
            const next = parts.filter((_, idx) => idx !== partIndex);
            writeParts(node, next);
            render();
        });

        document.body.appendChild(menu);
        node._composerContextMenu = menu;

        const close = (e) => {
            if (!menu.contains(e.target)) {
                removeContextMenu();
                document.removeEventListener("mousedown", close, true);
                document.removeEventListener("contextmenu", close, true);
            }
        };

        setTimeout(() => {
            document.addEventListener("mousedown", close, true);
            document.addEventListener("contextmenu", close, true);
        }, 0);
    };

    const openBrowserForPart = async (index) => {
        const parts = readParts(node);
        const part = parts[index] || { category: "", prompts: [], strength: 1.0, subject_number: SUBJECT_MIN, subject_locked: false };
        const inheritedSubject = getInheritedSubjectDefaults(parts.slice(0, index));
        const currentPrompt = part.prompts[0] || "";
        const hasMultiSelection = Array.isArray(part.prompts) && part.prompts.length > 1;
        const initialCategoryTypeFilter = getCategoryPromptType(node, part.category) || "__none__";
        const selection = await showThumbnailBrowser(node, part.category || "", currentPrompt, {
            title: "Select Prompt Composer Part",
            multiSelect: hasMultiSelection,
            multiCategorySelect: hasMultiSelection,
            endpointPrefix: COMPOSER_ENDPOINT_PREFIX,
            promptOnly: true,
            selectedPrompts: part.prompts,
            loadPromptsFn: loadComposerPrompts,
            preferenceScope: "composer",
            initialCategoryTypeFilter,
            multiSelectActionMode: "composer-add",
        });

        if (!selection || !Array.isArray(selection.prompts) || selection.prompts.length === 0) return;

        const replacementParts = buildPartsFromBrowserSelection(node, selection, inheritedSubject, part, part.category || "");
        if (!replacementParts.length) return;

        const next = [...parts];
        next.splice(index, 1, ...replacementParts);

        clearSelectedPartIndices(node);
        writeParts(node, next);
        render();
    };

    const reorderPart = (fromIndex, toIndex) => {
        const parts = readParts(node);
        if (fromIndex === toIndex) return;
        if (fromIndex < 0 || fromIndex >= parts.length) return;
        if (toIndex < 0 || toIndex >= parts.length) return;
        const next = [...parts];
        const [moved] = next.splice(fromIndex, 1);
        next.splice(toIndex, 0, moved);
        writeParts(node, next);
        render();
    };

    const setPartStrength = (index, value) => {
        const next = readParts(node);
        if (!next[index]) return;
        next[index].strength = clampStrength(value);
        writeParts(node, next);
    };

    const render = () => {
        const parts = readParts(node);
        const resolvedParts = resolveSubjectAssignments(parts);
        const selectedPartIndexSet = new Set(getSelectedPartIndices(node, resolvedParts.length));
        const isVideoMode = readGenerationMode(node) === "video";
        const thumbZoom = readThumbZoom(node);
        zoomSlider.value = String(Math.round(thumbZoom * 100));
        syncZoomLabel();
        const minCardWidth = Math.round(THUMB_BASE_WIDTH * thumbZoom);
        const metaHeight = isVideoMode ? CARD_META_HEIGHT_VIDEO : CARD_META_HEIGHT;
        const tileMinHeight = Math.round(minCardWidth * (4 / 3)) + metaHeight;

        // Flexible tracks keep rows filled while min width controls scale steps.
        grid.style.gridTemplateColumns = `repeat(auto-fill, minmax(${minCardWidth}px, 1fr))`;
        grid.innerHTML = "";

        resolvedParts.forEach((part, index) => {
            const entry = part.prompts.length > 0
                ? getComposerEntry(node, part.category, part.prompts[0])
                : null;
            const thumb = entry?.thumbnail || DEFAULT_THUMBNAIL;
            const multiCount = part.prompts.length;
            const subjectAccent = getSubjectAccent(part.effective_subject_number);
            const isSubjectAnchor = !!part.subject_locked && part.effective_subject_number !== SUBJECT_NONE;
            const isMuted = part.muted === true;
            const isSelectedPart = selectedPartIndexSet.has(index);

            const card = document.createElement("div");
            const cardBorderColor = subjectAccent.border;
            const cardShadowParts = [];
            if (isSubjectAnchor) {
                cardShadowParts.push(`0 0 0 1px ${subjectAccent.soft}`);
            }
            if (isSelectedPart) {
                cardShadowParts.push(`0 0 0 2px ${UI.accentBorder || "hsl(208 73% 57% / 0.65)"} inset`);
            }
            card.style.cssText = `
                display: flex;
                flex-direction: column;
                gap: 4px;
                border: ${isSubjectAnchor ? 2 : 1}px solid ${cardBorderColor};
                border-radius: 6px;
                background: linear-gradient(180deg, ${subjectAccent.soft}, ${UI.cardBg || "#2b3340"} 42%);
                padding: 4px;
                box-sizing: border-box;
                min-height: ${tileMinHeight}px;
                box-shadow: ${cardShadowParts.length ? cardShadowParts.join(", ") : "none"};
                opacity: ${isMuted ? "0.5" : "1"};
                filter: ${isMuted ? "grayscale(0.45)" : "none"};
            `;
            card.oncontextmenu = (evt) => showPartContextMenu(evt, index);
            card.draggable = false;

            card.addEventListener("dragstart", (evt) => {
                if (node._composerDragSourceIndex !== index) {
                    evt.preventDefault();
                    return;
                }
                node._composerIsDragging = true;
                if (evt.dataTransfer) {
                    evt.dataTransfer.effectAllowed = "move";
                    evt.dataTransfer.setData("text/plain", String(index));
                }
                card.style.opacity = "0.65";
                card.style.cursor = "grabbing";
                card.classList.add("pm-composer-card-drag-active");
            });

            card.addEventListener("dragend", () => {
                card.draggable = false;
                card.style.opacity = "1";
                card.style.borderColor = cardBorderColor;
                card.style.cursor = "default";
                card.style.outline = "none";
                node._composerDragSourceIndex = null;
                node._composerIsDragging = false;
                card.classList.remove("pm-composer-card-drag-active");
            });

            card.addEventListener("dragover", (evt) => {
                if (node._composerDragSourceIndex === null || node._composerDragSourceIndex === index) return;
                evt.preventDefault();
                card.style.outline = `1px dashed ${UI.accentBorder || "hsl(208 73% 57% / 0.65)"}`;
            });

            card.addEventListener("dragleave", () => {
                card.style.outline = "none";
            });

            card.addEventListener("drop", (evt) => {
                evt.preventDefault();
                card.style.outline = "none";
                const fromIndex = Number(node._composerDragSourceIndex);
                if (!Number.isInteger(fromIndex)) return;
                reorderPart(fromIndex, index);
            });

            const thumbBtn = document.createElement("button");
            thumbBtn.type = "button";
            thumbBtn.style.cssText = `
                width: 100%;
                aspect-ratio: 3 / 4;
                border: 1px solid ${UI.inputBorder || "#445064"};
                border-radius: 4px;
                background-image: url(${thumb});
                background-size: contain;
                background-repeat: no-repeat;
                background-position: center;
                background-color: #1a1a1a;
                cursor: pointer;
                position: relative;
                display: block;
            `;
            thumbBtn.title = "Click to select prompt fragment(s)\nMiddle click to mute";

            const togglePartMuted = () => {
                const next = [...readParts(node)];
                if (!next[index]) return;
                next[index] = normalizePart({
                    ...next[index],
                    muted: !(next[index]?.muted === true),
                });
                writeParts(node, next);
                render();
            };

            const handleAuxClick = (evt) => {
                if (evt.button !== 1) return;
                evt.preventDefault();
                evt.stopPropagation();
                togglePartMuted();
            };
            card.addEventListener("auxclick", handleAuxClick);

            const subjectBadge = document.createElement("button");
            subjectBadge.type = "button";
            subjectBadge.textContent = `#${padSubjectNumber(part.effective_subject_number)}`;
            subjectBadge.title = part.effective_subject_number === SUBJECT_NONE
                ? "Not a subject. Click to assign Subject 01, right-click for options."
                : part.subject_locked
                ? "Custom subject. Click to advance, Shift-click to go back, right-click for auto mode."
                : "Auto subject. Click to create a custom subject, right-click for options.";
            subjectBadge.style.cssText = `
                position: absolute;
                left: 4px;
                top: 4px;
                min-width: 26px;
                height: 18px;
                border-radius: 9px;
                background: ${part.subject_locked ? subjectAccent.strong : "rgba(15,23,42,0.82)"};
                border: 1px solid ${subjectAccent.border};
                color: ${subjectAccent.text};
                font-size: 10px;
                line-height: 16px;
                text-align: center;
                font-weight: 700;
                padding: 0 5px;
                box-sizing: border-box;
                cursor: pointer;
            `;
            subjectBadge.addEventListener("mousedown", (evt) => {
                evt.stopPropagation();
            });
            subjectBadge.addEventListener("mouseup", (evt) => {
                evt.stopPropagation();
            });
            subjectBadge.addEventListener("click", (evt) => {
                evt.preventDefault();
                evt.stopPropagation();
                const delta = evt.shiftKey ? -1 : 1;
                const next = [...readParts(node)];
                if (!next[index]) return;
                const baseSubject = part.effective_subject_number === SUBJECT_NONE ? SUBJECT_MIN : part.effective_subject_number;
                next[index] = normalizePart({
                    ...next[index],
                    subject_number: nextSubjectNumber(baseSubject, delta),
                    subject_locked: true,
                });
                writeParts(node, next);
                render();
            });
            subjectBadge.addEventListener("contextmenu", (evt) => {
                evt.preventDefault();
                evt.stopPropagation();
                const next = [...readParts(node)];
                if (!next[index]) return;
                next[index] = normalizePart({
                    ...next[index],
                    subject_locked: false,
                });
                writeParts(node, next);
                render();
            });
            thumbBtn.appendChild(subjectBadge);

            if (isMuted) {
                const mutedBadge = document.createElement("div");
                mutedBadge.textContent = "MUTED";
                mutedBadge.style.cssText = `
                    position: absolute;
                    right: 4px;
                    bottom: 4px;
                    min-width: 40px;
                    height: 18px;
                    border-radius: 9px;
                    background: rgba(15,23,42,0.88);
                    border: 1px solid rgba(148, 163, 184, 0.75);
                    color: #e5e7eb;
                    font-size: 9px;
                    line-height: 16px;
                    text-align: center;
                    font-weight: 700;
                    padding: 0 6px;
                    box-sizing: border-box;
                    letter-spacing: 0.05em;
                `;
                thumbBtn.appendChild(mutedBadge);
            }

            if (multiCount > 1) {
                const badge = document.createElement("div");
                badge.textContent = `+${multiCount}`;
                badge.style.cssText = `
                    position: absolute;
                    right: 4px;
                    top: 4px;
                    min-width: 18px;
                    height: 18px;
                    border-radius: 9px;
                    background: rgba(15,23,42,0.85);
                    border: 1px solid ${UI.accentBorder || "hsl(208 73% 57% / 0.65)"};
                    color: #dbeafe;
                    font-size: 10px;
                    line-height: 16px;
                    text-align: center;
                    font-weight: bold;
                    padding: 0 4px;
                    box-sizing: border-box;
                `;
                thumbBtn.appendChild(badge);
            }

            const label = document.createElement("div");
            const primaryName = part.prompts[0] || "Select";
            label.textContent = multiCount > 1
                ? `${part.category || "Category"}: (Multi)`
                : `${part.category || "Category"}: ${primaryName}`;
            label.title = multiCount > 1
                ? `Subject #${padSubjectNumber(part.effective_subject_number)}\n${part.category || ""}\n${part.prompts.join("\n")}`
                : label.textContent;
            label.style.cssText = `
                font-size: 10px;
                color: ${isMuted ? (UI.textMuted || "#9ca3af") : (UI.textPrimary || "#d1d5db")};
                line-height: 1.2;
                text-align: center;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                cursor: pointer;
                text-decoration: ${isMuted ? "line-through" : "none"};
            `;
            const strengthRow = document.createElement("div");
            strengthRow.style.cssText = `
                display: ${isVideoMode ? "none" : "flex"};
                align-items: center;
                gap: 4px;
            `;

            const makeAdjustBtn = (labelText, delta) => {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.textContent = labelText;
                btn.style.cssText = `
                    width: 22px;
                    height: 22px;
                    border: 1px solid ${UI.inputBorder || "#445064"};
                    border-radius: 4px;
                    background: ${UI.buttonBg || "#232a36"};
                    color: ${UI.textPrimary || "#d1d5db"};
                    cursor: pointer;
                    font-size: 12px;
                    line-height: 1;
                    padding: 0;
                `;
                btn.onclick = (evt) => {
                    evt.stopPropagation();
                    const current = clampStrength(strengthInput.value);
                    const nextValue = clampStrength(current + delta);
                    strengthInput.value = nextValue.toFixed(2);
                    setPartStrength(index, nextValue);
                };
                return btn;
            };

            const strengthInput = document.createElement("input");
            strengthInput.type = "text";
            strengthInput.value = clampStrength(part.strength).toFixed(2);
            strengthInput.style.cssText = `
                flex: 1;
                min-width: 0;
                height: 22px;
                border: 1px solid ${UI.inputBorder || "#445064"};
                border-radius: 4px;
                background: ${UI.inputBg || "#181d25"};
                color: ${UI.textPrimary || "#d1d5db"};
                font-size: 11px;
                text-align: center;
                box-sizing: border-box;
                padding: 0 4px;
            `;

            const commitStrengthInput = () => {
                const nextValue = clampStrength(strengthInput.value);
                strengthInput.value = nextValue.toFixed(2);
                setPartStrength(index, nextValue);
            };

            strengthInput.addEventListener("keydown", (evt) => {
                if (evt.key === "Enter") {
                    evt.preventDefault();
                    commitStrengthInput();
                    strengthInput.blur();
                }
            });
            strengthInput.addEventListener("blur", commitStrengthInput);

            const decBtn = makeAdjustBtn("<", -0.1);
            const incBtn = makeAdjustBtn(">", 0.1);

            strengthRow.appendChild(decBtn);
            strengthRow.appendChild(strengthInput);
            strengthRow.appendChild(incBtn);

            let holdTimer = null;
            let dragArmed = false;

            const clearHoldTimer = () => {
                if (holdTimer) {
                    clearTimeout(holdTimer);
                    holdTimer = null;
                }
            };

            const disarmDrag = () => {
                dragArmed = false;
                card.draggable = false;
                card.style.borderColor = cardBorderColor;
                card.style.cursor = "default";
                card.classList.remove("pm-composer-card-drag-active");
                node._composerDragSourceIndex = null;
            };

            const armDrag = () => {
                dragArmed = true;
                node._composerDragSourceIndex = index;
                card.draggable = true;
                card.style.borderColor = UI.accentBorder || "hsl(208 73% 57% / 0.65)";
                card.style.cursor = "grab";
                card.classList.add("pm-composer-card-drag-active");
            };

            const onPressStart = (evt) => {
                if (evt.button === 1) {
                    evt.preventDefault();
                    evt.stopPropagation();
                    return;
                }
                if (evt.button !== 0) return;
                dragArmed = false;
                clearHoldTimer();
                window.addEventListener("mouseup", onGlobalMouseUp, true);
                holdTimer = setTimeout(armDrag, HOLD_TO_DRAG_MS);
            };

            const onPressCancel = () => {
                if (!dragArmed) {
                    clearHoldTimer();
                }
            };

            const onGlobalMouseUp = () => {
                clearHoldTimer();
                if (dragArmed && !node._composerIsDragging) {
                    disarmDrag();
                }
                window.removeEventListener("mouseup", onGlobalMouseUp, true);
            };

            const onPressEnd = async (evt) => {
                if (evt.button !== 0) return;
                clearHoldTimer();
                window.removeEventListener("mouseup", onGlobalMouseUp, true);
                if (!dragArmed) {
                    evt.stopPropagation();
                    if (evt.ctrlKey || evt.metaKey) {
                        toggleSelectedPartIndex(node, index, resolvedParts.length);
                        render();
                        return;
                    }
                    if (isMuted) {
                        clearSelectedPartIndices(node);
                        togglePartMuted();
                        return;
                    }
                    clearSelectedPartIndices(node);
                    await openBrowserForPart(index);
                } else if (!node._composerIsDragging) {
                    disarmDrag();
                }
            };

            [thumbBtn, label].forEach((el) => {
                el.addEventListener("mousedown", onPressStart);
                el.addEventListener("mouseleave", onPressCancel);
                el.addEventListener("mouseup", onPressEnd);
                el.addEventListener("auxclick", handleAuxClick);
            });

            card.appendChild(thumbBtn);
            card.appendChild(label);
            card.appendChild(strengthRow);
            grid.appendChild(card);
        });

        const addCard = document.createElement("button");
        addCard.type = "button";
        addCard.style.cssText = `
            min-height: ${tileMinHeight}px;
            border: 1px dashed ${UI.accentBorder || "hsl(208 73% 57% / 0.65)"};
            border-radius: 6px;
            background: ${UI.panel || "#1f2937"};
            color: ${UI.textMuted || "#9ca3af"};
            cursor: pointer;
            font-size: 24px;
            line-height: 1;
        `;
        addCard.textContent = "+";
        addCard.title = "Add prompt part";
        addCard.onclick = async (evt) => {
            const parts = readParts(node);
            const inheritedSubject = getInheritedSubjectDefaults(parts);
            const selectedType = await showComposerTypePicker(evt, evt.currentTarget);
            if (selectedType === null) {
                return;
            }
            const selection = await showThumbnailBrowser(node, "", "", {
                title: "Add Prompt Composer Part",
                multiSelect: true,
                multiCategorySelect: true,
                endpointPrefix: COMPOSER_ENDPOINT_PREFIX,
                promptOnly: true,
                selectedPrompts: [],
                loadPromptsFn: loadComposerPrompts,
                preferenceScope: "composer",
                initialCategoryTypeFilter: selectedType,
                multiSelectActionMode: "composer-add",
            });

            if (!selection || !Array.isArray(selection.prompts) || selection.prompts.length === 0) {
                return;
            }

            const next = [...parts];
            const addedParts = buildPartsFromBrowserSelection(node, selection, inheritedSubject, null, "");
            if (!addedParts.length) {
                return;
            }
            next.push(...addedParts);
            clearSelectedPartIndices(node);
            writeParts(node, next);
            render();
        };

        addCard.addEventListener("dragover", (evt) => {
            if (node._composerDragSourceIndex === null || node._composerDragSourceIndex === undefined) return;
            evt.preventDefault();
        });

        addCard.addEventListener("drop", (evt) => {
            evt.preventDefault();
            const fromIndex = Number(node._composerDragSourceIndex);
            if (!Number.isInteger(fromIndex)) return;
            const parts = readParts(node);
            if (fromIndex < 0 || fromIndex >= parts.length) return;
            const next = [...parts];
            const [moved] = next.splice(fromIndex, 1);
            next.push(moved);
            writeParts(node, next);
            render();
        });

        grid.appendChild(addCard);

        node._composerUiRefreshHeight?.();
    };

    const computeComposerHeight = () => {
        const nodeHeight = Number(node?.size?.[1]) || MIN_NODE_HEIGHT;
        return Math.max(180, nodeHeight - NODE_CHROME_HEIGHT);
    };

    const refreshComposerHeight = () => {
        const h = computeComposerHeight();
        root.style.setProperty("--comfy-widget-min-height", `${h}px`);
        root.style.setProperty("--comfy-widget-height", `${h}px`);
    };

    const widget = node.addDOMWidget("prompt_composer_ui", "div", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => computeComposerHeight(),
        getHeight: () => "100%",
    });

    node._composerUiAttached = true;
    node._composerUiRender = render;
    node._composerUiRefreshHeight = refreshComposerHeight;
    node._composerUiSyncSwitches = () => {
        formatSwitch.sync();
        positionSwitch.sync();
        generationModeSwitch.sync();
    };

    refreshComposerHeight();
    node._composerUiSyncSwitches();
    render();
}

app.registerExtension({
    name: "PromptComposer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PromptComposer") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            ensureHiddenComposerWidgets(node);
            if (!node.properties) node.properties = {};
            if (node.properties[PARTS_PROP_KEY] === undefined) {
                const existing = getPartsWidget(node)?.value || "[]";
                node.properties[PARTS_PROP_KEY] = String(existing || "[]");
            }
            if (node.properties[OUTPUT_FORMAT_PROP_KEY] === undefined) {
                node.properties[OUTPUT_FORMAT_PROP_KEY] = readOutputFormat(node);
            }
            if (node.properties[COMPOSE_POSITION_PROP_KEY] === undefined) {
                node.properties[COMPOSE_POSITION_PROP_KEY] = readComposePosition(node);
            }
            if (node.properties[GENERATION_MODE_PROP_KEY] === undefined) {
                node.properties[GENERATION_MODE_PROP_KEY] = readGenerationMode(node);
            }

            node.setSize([
                Math.max(MIN_NODE_WIDTH, node.size?.[0] || MIN_NODE_WIDTH),
                Math.max(MIN_NODE_HEIGHT, node.size?.[1] || MIN_NODE_HEIGHT),
            ]);

            ensureComposerUi(node);

            loadComposerPrompts(node).then(() => {
                node._composerUiSyncSwitches?.();
                node._composerUiRefreshHeight?.();
                node._composerUiRender?.();
                app.graph.setDirtyCanvas(true, true);
            });

            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = onConfigure?.apply(this, arguments);
            const node = this;

            ensureHiddenComposerWidgets(node);
            ensureComposerUi(node);

            const widget = getPartsWidget(node);
            if (widget && typeof widget.value === "string") {
                node.properties = node.properties || {};
                node.properties[PARTS_PROP_KEY] = widget.value;
            }
            node.properties = node.properties || {};
            node.properties[OUTPUT_FORMAT_PROP_KEY] = readOutputFormat(node);
            node.properties[COMPOSE_POSITION_PROP_KEY] = readComposePosition(node);
            node.properties[GENERATION_MODE_PROP_KEY] = readGenerationMode(node);

            node._composerUiSyncSwitches?.();
            node._composerUiRefreshHeight?.();
            node._composerUiRender?.();
            return result;
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            size[0] = Math.max(MIN_NODE_WIDTH, size[0]);
            size[1] = Math.max(MIN_NODE_HEIGHT, size[1]);
            const result = onResize ? onResize.apply(this, arguments) : size;
            this._composerUiRefreshHeight?.();
            this._composerUiRender?.();
            return result;
        };
    },
});
