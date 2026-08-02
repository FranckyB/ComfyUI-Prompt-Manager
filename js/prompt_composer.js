import { app } from "../../scripts/app.js";
import { PM_UI_PALETTE as UI } from "./ui_palette.js";
import { DEFAULT_THUMBNAIL } from "./prompt_manager_advanced.js";
import { showThumbnailBrowser } from "./prompt_browser.js";
import { loadComposerPrompts, getComposerEntry, COMPOSER_ENDPOINT_PREFIX } from "./prompt_composer_common.js";

const PARTS_PROP_KEY = "prompt_composer_parts";
const THUMB_ZOOM_PROP_KEY = "prompt_composer_thumb_zoom";
const PARTS_WIDGET_NAME = "parts_data";
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
const NODE_CHROME_HEIGHT = 86;
const SCROLLER_PADDING_TOP = 8;
const SCROLLER_PADDING_BOTTOM = 24;

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
    return node.widgets?.find((w) => w.name === PARTS_WIDGET_NAME) || null;
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

function ensureHiddenPartsWidget(node) {
    const widget = getPartsWidget(node);
    if (!widget) return;
    widget.type = "converted-widget";
    widget.computeSize = () => [0, -4];
    widget.hidden = true;
    widget.draw = function () {};
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
    let scroller = null;
    const absorbWheel = (evt) => {
        evt.preventDefault();
        evt.stopPropagation();
        if (scroller) {
            scroller.scrollTop += evt.deltaY;
        }
    };
    root.addEventListener("wheel", absorbWheel, { passive: false });

    scroller = document.createElement("div");
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
    scroller.addEventListener("wheel", absorbWheel, { passive: false });

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
        const part = parts[index] || { category: "", prompts: [], strength: 1.0 };
        const currentPrompt = part.prompts[0] || "";
        const selection = await showThumbnailBrowser(node, part.category || "", currentPrompt, {
            title: "Select Prompt Composer Part",
            multiSelect: true,
            multiCategorySelect: true,
            endpointPrefix: COMPOSER_ENDPOINT_PREFIX,
            promptOnly: true,
            selectedPrompts: part.prompts,
            loadPromptsFn: loadComposerPrompts,
            preferenceScope: "composer",
        });

        if (!selection || !Array.isArray(selection.prompts) || selection.prompts.length === 0) return;

        const next = [...parts];
        const originalCategory = part.category || "";

        if (selection.selectionsByCategory && Object.keys(selection.selectionsByCategory).length > 0) {
            const selectedCats = Object.keys(selection.selectionsByCategory);

            if (selection.selectionsByCategory[originalCategory]) {
                next[index] = normalizePart({
                    category: originalCategory,
                    prompts: selection.selectionsByCategory[originalCategory],
                    strength: part.strength,
                });
            } else {
                const firstCat = selectedCats[0];
                next[index] = normalizePart({
                    category: firstCat,
                    prompts: selection.selectionsByCategory[firstCat],
                    strength: part.strength,
                });
            }

            const usedCategory = next[index].category;
            for (const cat of selectedCats) {
                if (cat === usedCategory) continue;
                next.push(normalizePart({
                    category: cat,
                    prompts: selection.selectionsByCategory[cat],
                    strength: part.strength,
                }));
            }
        } else {
            next[index] = normalizePart({
                category: selection.category || part.category || "",
                prompts: selection.prompts,
                strength: part.strength,
            });
        }

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
        const thumbZoom = readThumbZoom(node);
        zoomSlider.value = String(Math.round(thumbZoom * 100));
        syncZoomLabel();
        const minCardWidth = Math.round(THUMB_BASE_WIDTH * thumbZoom);
        const tileMinHeight = Math.round(minCardWidth * (4 / 3)) + CARD_META_HEIGHT;

        // Flexible tracks keep rows filled while min width controls scale steps.
        grid.style.gridTemplateColumns = `repeat(auto-fill, minmax(${minCardWidth}px, 1fr))`;
        grid.innerHTML = "";

        parts.forEach((part, index) => {
            const entry = part.prompts.length > 0
                ? getComposerEntry(node, part.category, part.prompts[0])
                : null;
            const thumb = entry?.thumbnail || DEFAULT_THUMBNAIL;
            const multiCount = part.prompts.length;

            const card = document.createElement("div");
            const cardBorderColor = UI.inputBorder || "#3e495b";
            card.style.cssText = `
                display: flex;
                flex-direction: column;
                gap: 4px;
                border: 1px solid ${cardBorderColor};
                border-radius: 6px;
                background: ${UI.cardBg || "#2b3340"};
                padding: 4px;
                box-sizing: border-box;
                min-height: ${tileMinHeight}px;
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
                border: 1px solid ${UI.accentBorder || "hsl(208 73% 57% / 0.65)"};
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
            thumbBtn.title = "Click to select prompt fragment(s)";

            if (multiCount > 1) {
                const badge = document.createElement("div");
                badge.textContent = `+${multiCount - 1}`;
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
                ? `${part.category || ""}\n${part.prompts.join("\n")}`
                : label.textContent;
            label.style.cssText = `
                font-size: 10px;
                color: ${UI.textPrimary || "#d1d5db"};
                line-height: 1.2;
                text-align: center;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                cursor: pointer;
            `;
            const strengthRow = document.createElement("div");
            strengthRow.style.cssText = `
                display: flex;
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
                    await openBrowserForPart(index);
                } else if (!node._composerIsDragging) {
                    disarmDrag();
                }
            };

            [thumbBtn, label].forEach((el) => {
                el.addEventListener("mousedown", onPressStart);
                el.addEventListener("mouseleave", onPressCancel);
                el.addEventListener("mouseup", onPressEnd);
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
        addCard.onclick = async () => {
            const parts = readParts(node);
            const selection = await showThumbnailBrowser(node, "", "", {
                title: "Add Prompt Composer Part",
                multiSelect: true,
                multiCategorySelect: true,
                endpointPrefix: COMPOSER_ENDPOINT_PREFIX,
                promptOnly: true,
                selectedPrompts: [],
                loadPromptsFn: loadComposerPrompts,
                preferenceScope: "composer",
            });

            if (!selection || !Array.isArray(selection.prompts) || selection.prompts.length === 0) {
                return;
            }

            const next = [...parts];
            if (selection.selectionsByCategory && Object.keys(selection.selectionsByCategory).length > 0) {
                for (const [cat, catPrompts] of Object.entries(selection.selectionsByCategory)) {
                    next.push(normalizePart({
                        category: cat,
                        prompts: catPrompts,
                        strength: 1.0,
                    }));
                }
            } else {
                next.push(normalizePart({
                    category: selection.category || "",
                    prompts: selection.prompts,
                    strength: 1.0,
                }));
            }
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

    refreshComposerHeight();
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

            ensureHiddenPartsWidget(node);
            if (!node.properties) node.properties = {};
            if (node.properties[PARTS_PROP_KEY] === undefined) {
                const existing = getPartsWidget(node)?.value || "[]";
                node.properties[PARTS_PROP_KEY] = String(existing || "[]");
            }

            node.setSize([
                Math.max(MIN_NODE_WIDTH, node.size?.[0] || MIN_NODE_WIDTH),
                Math.max(MIN_NODE_HEIGHT, node.size?.[1] || MIN_NODE_HEIGHT),
            ]);

            ensureComposerUi(node);

            loadComposerPrompts(node).then(() => {
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

            ensureHiddenPartsWidget(node);
            ensureComposerUi(node);

            const widget = getPartsWidget(node);
            if (widget && typeof widget.value === "string") {
                node.properties = node.properties || {};
                node.properties[PARTS_PROP_KEY] = widget.value;
            }

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
