import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { PM_UI_PALETTE as UI } from "./ui_palette.js";
import {
    PMA_THEME,
    DEFAULT_THUMBNAIL,
    forwardWheelToCanvas,
} from "./prompt_manager_advanced.js";
import { showThumbnailBrowser } from "./prompt_browser.js";
import { loadMixerPrompts, getMixerCategories, getMixerNames, getMixerEntry, MIXER_ENDPOINT_PREFIX } from "./prompt_mixer_common.js";

const DEFAULT_THUMBNAIL_URL = DEFAULT_THUMBNAIL;
const MIXER_PREVIEW_BORDER_OPAQUE = "hsl(208 73% 57%)";

function ensureNameInOptions(node, widget, name) {
    if (!widget || !Array.isArray(widget.options?.values)) return;
    const values = widget.options.values;
    if (!values.includes(name)) {
        widget.options.values = [...values, name];
    }
}

function formatMixerTooltip(text) {
    if (!text) return "No prompt available";
    const words = String(text).trim().split(/\s+/);
    const lines = [];
    for (let i = 0; i < words.length; i += 10) {
        lines.push(words.slice(i, i + 10).join(" "));
    }
    return lines.join("\n");
}

function getOptimalTileColumns(count) {
    if (count <= 1) return 1;
    if (count <= 6) return 2;
    if (count <= 9) return 3;
    return 4;
}

function getFittingTileLayout(count, containerWidth, containerHeight) {
    const safeCount = Math.max(1, Number(count) || 1);
    const width = Math.max(1, Number(containerWidth) || 1);
    const height = Math.max(1, Number(containerHeight) || 1);
    const gap = 1;
    const columns = Math.max(1, Math.ceil(Math.sqrt(safeCount)));
    const rows = Math.max(1, Math.ceil(safeCount / columns));
    const tileWidth = Math.max(1, (width - gap * (columns - 1)) / columns);
    const tileHeight = Math.max(1, (height - gap * (rows - 1)) / rows);

    return {
        columns,
        rows,
        tileWidth,
        tileHeight,
        gap,
        area: tileWidth * tileHeight,
    };
}

function addMixerSelectorBar(node) {
    if (node._mixerSelectorBarAttached) return;

    const categoryWidget = node.widgets.find((w) => w.name === "category");
    const nameWidget = node.widgets.find((w) => w.name === "name");
    const selectedWidget = node.widgets.find((w) => w.name === "selected_prompts");
    if (!categoryWidget || !nameWidget) return;

    // Hide the native widgets that the DOM UI replaces.
    [categoryWidget, nameWidget, selectedWidget].forEach((w) => {
        if (!w) return;
        w.type = "converted-widget";
        w.computeSize = () => [0, -4];
        w.hidden = true;
        w.draw = function () {};
    });

    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const inputName = node.inputs[i].name;
        if (inputName === "category" || inputName === "name") {
            node.removeInput(i);
        }
    }

    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        align-items: center;
        gap: 0;
        background: #1a1a1a;
        overflow: visible;
        height: 26px;
        margin-top: -8px;
        box-sizing: border-box;
    `;
    forwardWheelToCanvas(container);

    const arrowStyle = `
        background: #2a2a2a;
        border: none;
        color: #888;
        padding: 0 10px;
        height: 100%;
        cursor: pointer;
        font-size: 10px;
        transition: all 0.15s ease;
    `;

    const makeArrow = (label) => {
        const btn = document.createElement("button");
        btn.textContent = label;
        btn.style.cssText = arrowStyle;
        btn.onmouseover = () => {
            btn.style.background = "#3a3a3a";
            btn.style.color = "#fff";
        };
        btn.onmouseout = () => {
            btn.style.background = "#2a2a2a";
            btn.style.color = "#888";
        };
        return btn;
    };

    const nameDisplay = document.createElement("div");
    nameDisplay.style.cssText = `
        flex: 1;
        text-align: center;
        color: #ddd;
        font-size: 13px;
        padding: 0 10px;
        cursor: pointer;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        background: #1a1a1a;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s ease;
    `;
    nameDisplay.onmouseover = () => (nameDisplay.style.background = "#252525");
    nameDisplay.onmouseout = () => (nameDisplay.style.background = "#1a1a1a");

    const leftArrow = makeArrow("◀");
    const rightArrow = makeArrow("▶");

    container.appendChild(leftArrow);
    container.appendChild(nameDisplay);
    container.appendChild(rightArrow);

    const updateDisplay = (overrideCategory, overrideName) => {
        const cat = overrideCategory || categoryWidget.value || "";
        const name = overrideName || nameWidget.value || "";
        const selected = getSelectedPrompts(node);
        if (selected.length > 1) {
            nameDisplay.textContent = cat ? `${cat} : (Multi)` : "(Multi)";
            nameDisplay.title = `Random pick from ${selected.length} fragments`;
        } else {
            nameDisplay.textContent = name ? `${cat} : ${name}` : "Select fragment...";
            nameDisplay.title = name ? formatMixerTooltip(getMixerEntry(node, cat, name)?.prompt || "") : nameDisplay.textContent;
        }
    };

    const getNames = () => getMixerNames(node, categoryWidget.value);

    const setCategory = async (newCat) => {
        if (!newCat) return;
        ensureNameInOptions(node, categoryWidget, newCat);
        categoryWidget.value = newCat;
        if (typeof categoryWidget.callback === "function") {
            await categoryWidget.callback(newCat);
        }
        const names = getNames();
        if (names.length > 0 && !names.includes(nameWidget.value)) {
            await setName(names[0]);
        } else {
            updateDisplay();
            updateMixerPreview(node);
        }
        app.graph.setDirtyCanvas(true, true);
    };

    const setName = async (newName) => {
        if (!newName) return;
        ensureNameInOptions(node, nameWidget, newName);
        nameWidget.value = newName;
        if (typeof nameWidget.callback === "function") {
            await nameWidget.callback(newName);
        }
        updateDisplay();
        updateMixerPreview(node);
        app.graph.setDirtyCanvas(true, true);
    };

    leftArrow.onclick = async (e) => {
        e.stopPropagation();
        const names = getNames();
        if (names.length === 0) return;
        const idx = names.indexOf(nameWidget.value);
        const newIdx = idx <= 0 ? names.length - 1 : idx - 1;
        await setName(names[newIdx]);
    };

    rightArrow.onclick = async (e) => {
        e.stopPropagation();
        const names = getNames();
        if (names.length === 0) return;
        const idx = names.indexOf(nameWidget.value);
        const newIdx = idx >= names.length - 1 ? 0 : idx + 1;
        await setName(names[newIdx]);
    };

    const openBrowser = async () => {
        const currentName = nameWidget.value || "";
        const currentCat = categoryWidget.value || "";
        const selectedPrompts = getSelectedPrompts(node);
        const selection = await showThumbnailBrowser(node, currentCat, currentName, {
            title: "Select Prompt Mixer Fragments",
            multiSelect: true,
            clearSelectionOnCategorySwitch: true,
            endpointPrefix: MIXER_ENDPOINT_PREFIX,
            promptOnly: true,
            selectedPrompts,
            loadPromptsFn: loadMixerPrompts,
        });
        if (selection && Array.isArray(selection.prompts)) {
            if (selection.category) {
                await setCategory(selection.category);
            }
            if (selection.prompt) {
                await setName(selection.prompt);
            }
            // Apply multi-selection after callbacks; callbacks can rebuild widget state.
            setSelectedPrompts(node, selection.prompts);
            updateMixerPreview(node);
            app.graph.setDirtyCanvas(true, true);
        }
    };

    nameDisplay.onclick = openBrowser;

    const widget = node.addDOMWidget("mixer_selector", "div", container, { hideOnZoom: false });
    widget.computeSize = (width) => [width, 20];

    node._mixerSelectorBarAttached = true;
    node._updateMixerSelectorDisplay = updateDisplay;

    updateDisplay();
    return widget;
}

function addMixerPreview(node) {
    if (node._mixerPreviewAttached) return;

    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 0;
        width: 100%;
        height: 100%;
        margin-top: -6px;
        box-sizing: border-box;
        overflow: hidden;
    `;

    const previewBox = document.createElement("div");
    previewBox.style.cssText = `
        position: relative;
        flex: 1;
        min-height: 80px;
        width: 100%;
        border-radius: 0;
        background: ${PMA_THEME.inputBg};
        border: 2px solid ${MIXER_PREVIEW_BORDER_OPAQUE};
        overflow: hidden;
        box-sizing: border-box;
    `;

    const image = document.createElement("img");
    image.draggable = false;
    image.style.cssText = `
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
        object-position: center center;
        display: none;
        cursor: pointer;
        user-select: none;
        pointer-events: none;
        -webkit-user-drag: none;
    `;

    const emptyLabel = document.createElement("div");
    emptyLabel.textContent = "No fragment selected";
    emptyLabel.style.cssText = `
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-size: 11px;
        color: ${PMA_THEME.textMuted};
        background: ${PMA_THEME.inputBg};
        border: 2px dashed ${PMA_THEME.accent};
        border-radius: 0;
        padding: 8px;
        box-sizing: border-box;
    `;

    const tilesContainer = document.createElement("div");
    tilesContainer.style.cssText = `
        position: absolute;
        inset: 0;
        display: none;
        padding: 0;
        box-sizing: border-box;
        overflow: hidden;
        gap: 0;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        grid-auto-rows: auto;
        align-content: center;
        justify-content: center;
        pointer-events: none;
    `;

    emptyLabel.style.pointerEvents = "none";

    previewBox.appendChild(image);
    previewBox.appendChild(emptyLabel);
    previewBox.appendChild(tilesContainer);

    previewBox.style.cursor = "pointer";
    previewBox.onclick = async (e) => {
        e.stopPropagation();
        try {
            const categoryWidget = node.widgets.find((w) => w.name === "category");
            const nameWidget = node.widgets.find((w) => w.name === "name");
            const selection = await showThumbnailBrowser(node, categoryWidget?.value || "", nameWidget?.value || "", {
                title: "Select Prompt Mixer Fragments",
                multiSelect: true,
                clearSelectionOnCategorySwitch: true,
                endpointPrefix: MIXER_ENDPOINT_PREFIX,
                promptOnly: true,
                selectedPrompts: getSelectedPrompts(node),
                loadPromptsFn: loadMixerPrompts,
            });
            if (selection && Array.isArray(selection.prompts)) {
                if (selection.category && categoryWidget) {
                    ensureNameInOptions(node, categoryWidget, selection.category);
                    categoryWidget.value = selection.category;
                    if (typeof categoryWidget.callback === "function") {
                        await categoryWidget.callback(selection.category);
                    }
                }
                if (selection.prompt && nameWidget) {
                    ensureNameInOptions(node, nameWidget, selection.prompt);
                    nameWidget.value = selection.prompt;
                    if (typeof nameWidget.callback === "function") {
                        await nameWidget.callback(selection.prompt);
                    }
                }
                // Apply multi-selection after callbacks; callbacks can rebuild widget state.
                setSelectedPrompts(node, selection.prompts);
                if (node._updateMixerSelectorDisplay) {
                    node._updateMixerSelectorDisplay();
                }
                updateMixerPreview(node);
                app.graph.setDirtyCanvas(true, true);
            }
        } catch (err) {
            console.error("[PromptMixer] Error opening browser:", err);
        }
    };

    previewBox.addEventListener("wheel", (e) => {
        const canvas = app.canvas?.canvas || document.querySelector("canvas.lgraphcanvas");
        if (!canvas) return;
        const newEvent = new WheelEvent("wheel", {
            bubbles: true,
            cancelable: true,
            clientX: e.clientX,
            clientY: e.clientY,
            deltaX: e.deltaX,
            deltaY: e.deltaY,
            deltaZ: e.deltaZ,
            deltaMode: e.deltaMode,
            ctrlKey: e.ctrlKey,
            shiftKey: e.shiftKey,
            altKey: e.altKey,
            metaKey: e.metaKey,
        });
        canvas.dispatchEvent(newEvent);
    }, { passive: true });

    container.appendChild(previewBox);

    const widget = node.addDOMWidget("mixer_preview", "div", container, { hideOnZoom: false });
    widget.getHeight = () => "100%";
    const origDraw = widget.draw;
    widget.draw = function (ctx, n, widgetWidth, y, H) {
        if (typeof origDraw === "function") origDraw.apply(this, arguments);
        if (!this.element || n.flags?.collapsed) return;
        this.element.style.setProperty("width", (n.size[0] - 18) + "px", "important");
        this.element.style.setProperty("left", "0px", "important");
        this.element.style.setProperty("top", "-6px", "important");
        this.element.style.setProperty("height", "calc(100% + 6px)", "important");
        this.element.style.setProperty("margin", "0px", "important");
        this.element.style.setProperty("padding", "0px", "important");
        this.element.style.setProperty("box-sizing", "border-box", "important");
        this.element.style.setProperty("overflow", "hidden", "important");
    };

    node._mixerPreview = { container, previewBox, image, emptyLabel, tilesContainer, widget };
    node._mixerPreviewAttached = true;

    updateMixerPreview(node);
    return widget;
}

function updateMixerPreview(node) {
    const ui = node._mixerPreview;
    if (!ui) return;

    const categoryWidget = node.widgets.find((w) => w.name === "category");
    const nameWidget = node.widgets.find((w) => w.name === "name");
    const category = categoryWidget?.value || "";
    const name = nameWidget?.value || "";
    const selected = getSelectedPrompts(node);

    let displayName = name;
    let displayCategory = category;

    if (selected.length > 1) {
        displayName = "(Multi)";
        displayCategory = category;

        ui.image.style.display = "none";
        ui.emptyLabel.style.display = "none";
        ui.tilesContainer.style.display = "grid";
        ui.tilesContainer.style.visibility = "hidden";
        ui.tilesContainer.innerHTML = "";

        // Make container visible first so measurements are stable on first render.
        const containerWidth = Math.max(1, ui.tilesContainer.clientWidth || ui.previewBox?.clientWidth || 0);
        const containerHeight = Math.max(1, ui.tilesContainer.clientHeight || ui.previewBox?.clientHeight || 0);
        const layout = getFittingTileLayout(selected.length, containerWidth, containerHeight);

        ui.tilesContainer.style.gap = `${layout.gap}px`;
        ui.tilesContainer.style.gridTemplateColumns = `repeat(${layout.columns}, ${Math.max(1, Math.floor(layout.tileWidth))}px)`;
        ui.tilesContainer.style.gridAutoRows = `${Math.max(1, Math.floor(layout.tileHeight))}px`;

        selected.forEach((promptName) => {
            const entry = getMixerEntry(node, category, promptName);
            const thumbnail = entry?.thumbnail || DEFAULT_THUMBNAIL_URL;
            const tile = document.createElement("div");
            tile.style.cssText = `
                width: 100%;
                height: 100%;
                background-image: url(${thumbnail});
                background-size: cover;
                background-position: center;
                background-color: #1a1a1a;
                border-radius: 2px;
                border: 1px solid ${MIXER_PREVIEW_BORDER_OPAQUE};
            `;
            tile.title = promptName;
            ui.tilesContainer.appendChild(tile);
        });

        ui.tilesContainer.style.visibility = "visible";
    } else {
        ui.tilesContainer.style.display = "none";
        ui.tilesContainer.style.visibility = "visible";
        ui.tilesContainer.innerHTML = "";

        let entry = null;
        if (name) {
            entry = getMixerEntry(node, category, name);
        }
        const thumbnail = entry?.thumbnail || DEFAULT_THUMBNAIL_URL;

        if (thumbnail && thumbnail !== DEFAULT_THUMBNAIL_URL) {
            ui.image.src = thumbnail;
            ui.image.style.display = "block";
            ui.emptyLabel.style.display = "none";
        } else {
            ui.image.removeAttribute("src");
            ui.image.style.display = "none";
            ui.emptyLabel.style.display = "flex";
            ui.emptyLabel.textContent = name ? "No thumbnail for selected fragment" : "No fragment selected";
        }
    }

    if (node._updateMixerSelectorDisplay) {
        node._updateMixerSelectorDisplay(displayCategory || category, displayName);
    }
}

function getSelectedPrompts(node) {
    const widget = node.widgets.find((w) => w.name === "selected_prompts");
    if (!widget) return [];
    try {
        const parsed = JSON.parse(widget.value || "[]");
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
}

function setSelectedPrompts(node, names) {
    const widget = node.widgets.find((w) => w.name === "selected_prompts");
    if (!widget) return;
    const list = Array.isArray(names) ? names : [];
    widget.value = JSON.stringify(list);
}

function resizeMixerNodeToContent(node) {
    if (!node || typeof node.computeSize !== "function") return;
    const computed = node.computeSize();
    const width = Math.max(240, node.size?.[0] || 240);
    const minHeight = Math.max(380, computed[1] + 20);
    if ((node.size?.[1] || 0) < minHeight) {
        node.setSize([width, minHeight]);
    }
    app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "PromptMixerSelector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "PromptMixerSelector") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            node.mixerPrompts = {};
            node.prompts = {};
            node._configuredFromWorkflow = false;
            node.setSize([240, 380]);

            const selectedWidget = node.widgets.find((w) => w.name === "selected_prompts");
            if (selectedWidget && (selectedWidget.value === undefined || selectedWidget.value === null || selectedWidget.value === "")) {
                selectedWidget.value = "[]";
            }

            addMixerSelectorBar(node);
            addMixerPreview(node);

            loadMixerPrompts(node).then(async () => {
                const categories = getMixerCategories(node);
                const categoryWidget = node.widgets.find((w) => w.name === "category");
                const nameWidget = node.widgets.find((w) => w.name === "name");

                if (categoryWidget) {
                    if (!categoryWidget.value && categories.length > 0) {
                        categoryWidget.value = categories[0];
                        if (typeof categoryWidget.callback === "function") {
                            categoryWidget.callback(categoryWidget.value);
                        }
                    }
                    ensureNameInOptions(node, categoryWidget, categoryWidget.value);
                }

                if (nameWidget) {
                    const names = getMixerNames(node, categoryWidget?.value);
                    if (!nameWidget.value && names.length > 0) {
                        nameWidget.value = names[0];
                        if (typeof nameWidget.callback === "function") {
                            nameWidget.callback(nameWidget.value);
                        }
                    }
                    ensureNameInOptions(node, nameWidget, nameWidget.value);
                }

                if (node._updateMixerSelectorDisplay) {
                    node._updateMixerSelectorDisplay();
                }
                updateMixerPreview(node);
                resizeMixerNodeToContent(node);
            });

            api.addEventListener("prompt-mixer-selector-update", (event) => {
                if (String(event.detail.node_id) === String(node.id)) {
                    updateMixerPreview(node);
                    app.graph.setDirtyCanvas(true, true);
                }
            });

            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = onConfigure?.apply(this, arguments);
            const node = this;
            node._configuredFromWorkflow = true;

            const selectedWidget = node.widgets.find((w) => w.name === "selected_prompts");
            if (selectedWidget) {
                try {
                    const parsed = JSON.parse(selectedWidget.value || "[]");
                    if (!Array.isArray(parsed)) selectedWidget.value = "[]";
                } catch {
                    selectedWidget.value = "[]";
                }
            }

            if (!node._mixerSelectorBarAttached) {
                addMixerSelectorBar(node);
            }
            if (!node._mixerPreviewAttached) {
                addMixerPreview(node);
            }

            const strengthWidget = node.widgets.find((w) => w.name === "strength");
            const storedValues = info?.widgets_values;
            const strengthIdx = node.widgets.findIndex((w) => w.name === "strength");
            const hasStoredStrength = Array.isArray(storedValues) && strengthIdx >= 0 && strengthIdx < storedValues.length;
            if (strengthWidget && (!hasStoredStrength || storedValues[strengthIdx] === undefined || storedValues[strengthIdx] === null || storedValues[strengthIdx] === "")) {
                strengthWidget.value = 1.0;
            }

            loadMixerPrompts(node).then(async () => {
                const categoryWidget = node.widgets.find((w) => w.name === "category");
                const nameWidget = node.widgets.find((w) => w.name === "name");
                if (categoryWidget && categoryWidget.value) {
                    ensureNameInOptions(node, categoryWidget, categoryWidget.value);
                }
                if (nameWidget && nameWidget.value) {
                    ensureNameInOptions(node, nameWidget, nameWidget.value);
                }
                if (node._updateMixerSelectorDisplay) {
                    node._updateMixerSelectorDisplay();
                }
                updateMixerPreview(node);
                resizeMixerNodeToContent(node);
                app.graph.setDirtyCanvas(true, true);
            });

            return result;
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            size[0] = Math.max(240, size[0]);
            size[1] = Math.max(380, size[1]);
            const result = onResize ? onResize.apply(this, arguments) : size;
            updateMixerPreview(this);
            return result;
        };
    },
});

console.log("[PromptMixerSelector] Extension loaded");
