import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { PM_UI_PALETTE as UI } from "./ui_palette.js";
import {
    PMA_THEME,
    DEFAULT_THUMBNAIL,
    loadPrompts,
    getPromptNamesForCategory,
    showThumbnailBrowser,
    forwardWheelToCanvas,
} from "./prompt_manager_advanced.js";

const EXPRESSION_CATEGORY = "Expressions";

function getExpressionCategory(node) {
    if (!node.prompts) return EXPRESSION_CATEGORY;
    for (const cat of Object.keys(node.prompts)) {
        if (cat.toLowerCase() === EXPRESSION_CATEGORY.toLowerCase()) {
            return cat;
        }
    }
    return EXPRESSION_CATEGORY;
}

function getExpressionNames(node) {
    const category = getExpressionCategory(node);
    return getPromptNamesForCategory(node, category, { hideNSFW: false, workflowOnly: false });
}

function getExpressionData(node, name) {
    const category = getExpressionCategory(node);
    return node.prompts?.[category]?.[name] || null;
}

function getExpressionThumbnail(entry) {
    if (!entry || typeof entry !== "object") return DEFAULT_THUMBNAIL;
    return entry.thumbnail || DEFAULT_THUMBNAIL;
}

function ensureNameInOptions(node, nameWidget, name) {
    if (!nameWidget || !Array.isArray(nameWidget.options?.values)) return;
    const values = nameWidget.options.values;
    if (!values.includes(name)) {
        nameWidget.options.values = [...values, name];
    }
}

function resizeExpressionNodeToContent(node, options = {}) {
    if (!node || typeof node.computeSize !== "function") return;
    const computed = node.computeSize();
    const width = Math.max(360, node.size?.[0] || 360);
    const minHeight = Math.max(360, computed[1] + 20);
    if (options.shrink || (node.size?.[1] || 0) < minHeight) {
        node.setSize([width, minHeight]);
    }
    app.graph.setDirtyCanvas(true, true);
}

/**
 * Create the < name > selector bar used to pick an expression.
 */
function addExpressionSelectorBar(node) {
    if (node._expressionSelectorBarAttached) return;

    const nameWidget = node.widgets.find((w) => w.name === "name");
    if (!nameWidget) return;

    // Hide the native name widget completely while keeping its value serialized.
    nameWidget.type = "converted-widget";
    nameWidget.computeSize = () => [0, -4];
    nameWidget.hidden = true;
    nameWidget.draw = function () {};

    for (let i = node.inputs.length - 1; i >= 0; i--) {
        if (node.inputs[i].name === "name") {
            node.removeInput(i);
        }
    }

    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        align-items: center;
        gap: 0;
        background: #1a1a1a;
        border-radius: 4px;
        overflow: visible;
        height: 26px;
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

    const leftArrow = document.createElement("button");
    leftArrow.textContent = "◀";
    leftArrow.style.cssText = arrowStyle;
    leftArrow.onmouseover = () => {
        leftArrow.style.background = "#3a3a3a";
        leftArrow.style.color = "#fff";
    };
    leftArrow.onmouseout = () => {
        leftArrow.style.background = '#2a2a2a';
        leftArrow.style.color = '#888';
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
    nameDisplay.onmouseover = () => nameDisplay.style.background = '#252525';
    nameDisplay.onmouseout = () => nameDisplay.style.background = '#1a1a1a';

    const rightArrow = document.createElement("button");
    rightArrow.textContent = "▶";
    rightArrow.style.cssText = arrowStyle;
    rightArrow.onmouseover = () => {
        rightArrow.style.background = "#3a3a3a";
        rightArrow.style.color = "#fff";
    };
    rightArrow.onmouseout = () => {
        rightArrow.style.background = '#2a2a2a';
        rightArrow.style.color = '#888';
    };

    container.appendChild(leftArrow);
    container.appendChild(nameDisplay);
    container.appendChild(rightArrow);

    const updateDisplay = () => {
        const name = nameWidget.value || "";
        nameDisplay.textContent = name ? `Expression : ${name}` : "Select expression...";
        nameDisplay.title = nameDisplay.textContent;
    };

    const getNames = () => getExpressionNames(node);
    const getCurrentIndex = (names) => names.findIndex((n) => n === nameWidget.value);

    const navigateTo = async (newName) => {
        if (!newName) return;
        ensureNameInOptions(node, nameWidget, newName);
        nameWidget.value = newName;
        if (typeof nameWidget.callback === "function") {
            await nameWidget.callback(newName);
        }
        updateDisplay();
        updateExpressionPreview(node);
        app.graph.setDirtyCanvas(true, true);
    };

    leftArrow.onclick = async (e) => {
        e.stopPropagation();
        const names = getNames();
        if (names.length === 0) return;
        const idx = getCurrentIndex(names);
        const newIdx = idx <= 0 ? names.length - 1 : idx - 1;
        await navigateTo(names[newIdx]);
    };

    rightArrow.onclick = async (e) => {
        e.stopPropagation();
        const names = getNames();
        if (names.length === 0) return;
        const idx = getCurrentIndex(names);
        const newIdx = idx >= names.length - 1 ? 0 : idx + 1;
        await navigateTo(names[newIdx]);
    };

    nameDisplay.onclick = async (e) => {
        e.stopPropagation();
        try {
            const currentName = nameWidget.value || "";
            const category = getExpressionCategory(node);
            const selection = await showThumbnailBrowser(node, category, currentName, {
                allowedCategories: [category],
                title: "Select Expression",
            });
            if (selection && selection.prompt) {
                await navigateTo(selection.prompt);
            }
        } catch (err) {
            console.error("[ExpressionSelector] Error opening browser:", err);
        }
    };

    const widget = node.addDOMWidget("expression_selector", "div", container, {
        hideOnZoom: false,
    });
    widget.computeSize = (width) => [width, 30];

    node._expressionSelectorBarAttached = true;
    node._updateExpressionSelectorDisplay = updateDisplay;

    updateDisplay();
    return widget;
}

/**
 * Create the expression preview widget (thumbnail + info button).
 */
function addExpressionPreview(node) {
    if (node._expressionPreviewAttached) return;

    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 6px;
        width: 100%;
        height: 100%;
        background: ${PMA_THEME.panel};
        border: 1px solid ${PMA_THEME.panelBorder};
        border-radius: 8px;
        padding: 8px;
        box-sizing: border-box;
        overflow: hidden;
    `;

    const previewBox = document.createElement("div");
    previewBox.style.cssText = `
        position: relative;
        flex: 1;
        min-height: 80px;
        width: 100%;
        border-radius: 6px;
        background: ${PMA_THEME.inputBg};
        border: 1px solid ${PMA_THEME.inputBorder};
        overflow: hidden;
        box-sizing: border-box;
    `;

    const image = document.createElement("img");
    image.style.cssText = `
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
        object-position: center center;
        display: none;
    `;

    const emptyLabel = document.createElement("div");
    emptyLabel.textContent = "No expression selected";
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
        border: 1px dashed ${PMA_THEME.inputBorder};
        border-radius: 5px;
        padding: 8px;
        box-sizing: border-box;
    `;

    previewBox.appendChild(image);
    previewBox.appendChild(emptyLabel);
    container.appendChild(previewBox);

    const widget = node.addDOMWidget("expression_preview", "div", container, {
        hideOnZoom: false,
    });
    // Let the DOM widget fill the remaining node height; do NOT override
    // computeSize with a fixed height or vertical resize will be locked.
    widget.getHeight = () => "100%";
    const origDraw = widget.draw;
    widget.draw = function (ctx, n, widgetWidth, y, H) {
        if (typeof origDraw === "function") origDraw.apply(this, arguments);
        if (!this.element || n.flags?.collapsed) return;
        this.element.style.setProperty("width", (n.size[0] - 18) + "px", "important");
        this.element.style.setProperty("left", "0px", "important");
        this.element.style.setProperty("margin", "0px", "important");
        this.element.style.setProperty("padding", "0px", "important");
        this.element.style.setProperty("box-sizing", "border-box", "important");
        this.element.style.setProperty("overflow", "hidden", "important");
    };

    node._expressionPreview = { container, image, emptyLabel, widget };
    node._expressionPreviewAttached = true;

    updateExpressionPreview(node);
    return widget;
}

function updateExpressionPreview(node) {
    const ui = node._expressionPreview;
    if (!ui) return;

    const nameWidget = node.widgets.find((w) => w.name === "name");
    const name = nameWidget?.value || "";
    const entry = name ? getExpressionData(node, name) : null;
    const thumbnail = getExpressionThumbnail(entry);

    if (thumbnail && thumbnail !== DEFAULT_THUMBNAIL) {
        ui.image.src = thumbnail;
        ui.image.style.display = "block";
        ui.emptyLabel.style.display = "none";
    } else {
        ui.image.removeAttribute("src");
        ui.image.style.display = "none";
        ui.emptyLabel.style.display = "flex";
        ui.emptyLabel.textContent = name ? "No thumbnail for selected expression" : "No expression selected";
    }
}

app.registerExtension({
    name: "ExpressionSelector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ExpressionSelector") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            node.prompts = {};
            node._configuredFromWorkflow = false;

            node.setSize([400, 460]);

            addExpressionSelectorBar(node);
            addExpressionPreview(node);

            loadPrompts(node).then(() => {
                const names = getExpressionNames(node);
                const nameWidget = node.widgets.find((w) => w.name === "name");
                if (nameWidget) {
                    if (!nameWidget.value && names.length > 0) {
                        nameWidget.value = names[0];
                        if (typeof nameWidget.callback === "function") {
                            nameWidget.callback(nameWidget.value);
                        }
                    }
                    ensureNameInOptions(node, nameWidget, nameWidget.value);
                }
                if (node._updateExpressionSelectorDisplay) {
                    node._updateExpressionSelectorDisplay();
                }
                updateExpressionPreview(node);
                resizeExpressionNodeToContent(node);
            });

            // Listen for backend execution updates
            api.addEventListener("expression-selector-update", (event) => {
                if (String(event.detail.node_id) === String(node.id)) {
                    updateExpressionPreview(node);
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

            if (!node._expressionSelectorBarAttached) {
                addExpressionSelectorBar(node);
            }
            if (!node._expressionPreviewAttached) {
                addExpressionPreview(node);
            }

            loadPrompts(node).then(() => {
                const nameWidget = node.widgets.find((w) => w.name === "name");
                if (nameWidget && nameWidget.value) {
                    ensureNameInOptions(node, nameWidget, nameWidget.value);
                }
                if (node._updateExpressionSelectorDisplay) {
                    node._updateExpressionSelectorDisplay();
                }
                updateExpressionPreview(node);
                resizeExpressionNodeToContent(node);
                app.graph.setDirtyCanvas(true, true);
            });

            return result;
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            size[0] = Math.max(360, size[0]);
            size[1] = Math.max(360, size[1]);
            return onResize ? onResize.apply(this, arguments) : size;
        };
    },
});

console.log("[ExpressionSelector] Extension loaded");
