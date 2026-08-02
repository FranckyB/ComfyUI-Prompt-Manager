import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { PM_UI_PALETTE as UI } from "./ui_palette.js";
import {
    showInfo,
    showConfirm,
} from "./prompt_manager_advanced.js";
import { showThumbnailBrowser } from "./prompt_browser.js";
import { loadComposerPrompts, getComposerCategories, getComposerNames, getComposerEntry, COMPOSER_ENDPOINT_PREFIX } from "./prompt_composer_common.js";

const PMA_THEME = {
    panel: UI.panel || "hsl(216 11% 15%)",
    panelBorder: UI.panelBorder || "hsl(216 20% 65% / 0.24)",
    sectionBorder: UI.sectionBorder || "hsl(216 20% 65% / 0.20)",
    inputBg: UI.inputBg || "hsl(220 15% 10%)",
    inputBorder: UI.inputBorder || "hsl(218 10% 41%)",
    buttonBg: UI.buttonBg || "hsl(219 16% 18%)",
    cardBg: UI.cardBg || "hsl(219 16% 18%)",
    textPrimary: UI.textPrimary || "hsl(0 0% 87%)",
    textMuted: UI.textMuted || "hsl(0 0% 67%)",
    textHint: UI.textHint || "hsl(216 15% 65%)",
    accent: UI.accent || "hsl(208 73% 57% / 0.9)",
    accentSoft: UI.accentSoft || "hsl(208 73% 57% / 0.16)",
    accentBorder: UI.accentBorder || "hsl(208 73% 57% / 0.65)",
};

async function saveComposerPrompt(node, category, name, text, thumbnail = null) {
    try {
        const resp = await fetch(`${COMPOSER_ENDPOINT_PREFIX}/save-prompt`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ category, name, text, thumbnail }),
        });
        const data = await resp.json();
        if (data.success) {
            node.composerPrompts = data.prompts;
            node.prompts = data.prompts;
        }
        return data;
    } catch (err) {
        console.error("[PromptComposerManager] Error saving prompt:", err);
        return { success: false, error: String(err) };
    }
}

async function deleteComposerPrompt(node, category, name) {
    try {
        const resp = await fetch(`${COMPOSER_ENDPOINT_PREFIX}/delete-prompt`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ category, name }),
        });
        const data = await resp.json();
        if (data.success) {
            node.composerPrompts = data.prompts;
            node.prompts = data.prompts;
        }
        return data;
    } catch (err) {
        console.error("[PromptComposerManager] Error deleting prompt:", err);
        return { success: false, error: String(err) };
    }
}

function createComposerButton(text, callback) {
    const button = document.createElement("button");
    button.textContent = text;
    button.style.flex = "1";
    button.style.minWidth = "70px";
    button.style.padding = "6px 8px";
    button.style.cursor = "pointer";
    button.style.backgroundColor = "#222";
    button.style.color = "#fff";
    button.style.border = "1px solid #444";
    button.style.borderRadius = "6px";
    button.style.fontSize = "11px";
    button.style.whiteSpace = "nowrap";
    button.style.overflow = "hidden";
    button.style.textOverflow = "ellipsis";
    button.style.height = "28px";
    button.style.display = "flex";
    button.style.alignItems = "center";
    button.style.justifyContent = "center";
    button.onclick = callback;
    return button;
}

function createComposerDropdownButton(text, items) {
    const container = document.createElement("div");
    container.style.position = "relative";
    container.style.flex = "1";
    container.style.minWidth = "70px";

    const button = document.createElement("button");
    button.textContent = text;
    button.style.width = "100%";
    button.style.padding = "6px 8px";
    button.style.cursor = "pointer";
    button.style.backgroundColor = "#222";
    button.style.color = "#fff";
    button.style.border = "1px solid #444";
    button.style.borderRadius = "6px";
    button.style.fontSize = "11px";
    button.style.whiteSpace = "nowrap";
    button.style.height = "28px";
    button.style.display = "flex";
    button.style.alignItems = "center";
    button.style.justifyContent = "center";

    const dropdown = document.createElement("div");
    dropdown.style.cssText = `
        position: fixed;
        background: ${PMA_THEME.panel};
        border: 1px solid ${PMA_THEME.inputBorder};
        border-radius: 6px;
        z-index: 999999;
        display: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        min-width: 140px;
    `;
    document.body.appendChild(dropdown);

    items.forEach((item) => {
        if (item.divider) {
            const divider = document.createElement("div");
            divider.style.cssText = `height: 1px; background: ${PMA_THEME.sectionBorder}; margin: 4px 0;`;
            dropdown.appendChild(divider);
        } else {
            const menuItem = document.createElement("div");
            menuItem.textContent = item.label;
            menuItem.style.cssText = `
                padding: 8px 12px;
                cursor: pointer;
                font-size: 11px;
                color: ${PMA_THEME.textPrimary};
                white-space: nowrap;
            `;
            menuItem.addEventListener("mouseenter", () => {
                menuItem.style.backgroundColor = PMA_THEME.accentSoft;
            });
            menuItem.addEventListener("mouseleave", () => {
                menuItem.style.backgroundColor = "transparent";
            });
            menuItem.addEventListener("click", (e) => {
                e.stopPropagation();
                dropdown.style.display = "none";
                item.action();
            });
            dropdown.appendChild(menuItem);
        }
    });

    button.addEventListener("click", (e) => {
        e.stopPropagation();
        const isVisible = dropdown.style.display === "block";
        if (isVisible) {
            dropdown.style.display = "none";
        } else {
            const rect = button.getBoundingClientRect();
            dropdown.style.left = rect.left + "px";
            dropdown.style.top = (rect.bottom + 2) + "px";
            dropdown.style.display = "block";
        }
    });

    document.addEventListener("click", (e) => {
        if (!dropdown.contains(e.target) && e.target !== button) {
            dropdown.style.display = "none";
        }
    });

    container.appendChild(button);
    return container;
}

function setupPromptComposerManager(nodeType, nodeData) {
    if (nodeData.name !== "PromptComposerManager") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated?.apply(this, arguments);
        const node = this;

        node.composerPrompts = {};
        node.prompts = {};
        node._configuredFromWorkflow = false;
        node.isNewUnsavedPrompt = false;
        node.newPromptCategory = null;
        node.newPromptName = null;

        // Match Prompt Manager Advanced footprint.
        node.setSize([440, 440]);

        // Hide native category/name widgets immediately; the custom selector bar replaces them.
        const categoryWidget = node.widgets.find((w) => w.name === "category");
        const nameWidget = node.widgets.find((w) => w.name === "name");
        if (categoryWidget) {
            categoryWidget.type = "converted-widget";
            categoryWidget.computeSize = () => [0, 0];
            categoryWidget.hidden = true;
        }
        if (nameWidget) {
            nameWidget.type = "converted-widget";
            nameWidget.computeSize = () => [0, 0];
            nameWidget.hidden = true;
        }
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const inp = node.inputs[i];
            if (inp.name === "category" || inp.name === "name") {
                node.removeInput(i);
            }
        }

        loadComposerPrompts(node).then(() => {
            buildComposerSelectorBar(node);
            buildComposerButtonBar(node);
            syncSelectorToData(node);
            updateComposerLastSavedState(node);
            refreshComposerPromptInputGhosting(node);
        });

        const textWidget = node.widgets.find((w) => w.name === "text");
        api.addEventListener("prompt-manager-update-text", (event) => {
            if (String(event.detail.node_id) !== String(node.id)) return;
            node._composerIncomingPrompt = event.detail.prompt || "";
            const usePromptInput = event.detail.use_prompt_input === true;
            if (textWidget && usePromptInput) {
                textWidget.value = node._composerIncomingPrompt;
            }
            refreshComposerPromptInputGhosting(node);
            app.graph.setDirtyCanvas(true, true);
        });

        const usePromptInputWidget = node.widgets.find((w) => w.name === "use_prompt_input");
        if (usePromptInputWidget && !usePromptInputWidget._composerWrapped) {
            const originalCallback = usePromptInputWidget.callback;
            usePromptInputWidget.callback = function () {
                if (typeof originalCallback === "function") {
                    originalCallback.apply(this, arguments);
                }
                refreshComposerPromptInputGhosting(node);
            };
            usePromptInputWidget._composerWrapped = true;
        }

        refreshComposerPromptInputGhosting(node);

        return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
        const result = onConfigure?.apply(this, arguments);
        const node = this;
        node._configuredFromWorkflow = true;

        const categoryWidget = node.widgets.find((w) => w.name === "category");
        const nameWidget = node.widgets.find((w) => w.name === "name");
        if (categoryWidget) {
            categoryWidget.type = "converted-widget";
            categoryWidget.computeSize = () => [0, 0];
            categoryWidget.hidden = true;
        }
        if (nameWidget) {
            nameWidget.type = "converted-widget";
            nameWidget.computeSize = () => [0, 0];
            nameWidget.hidden = true;
        }
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const inp = node.inputs[i];
            if (inp.name === "category" || inp.name === "name") {
                node.removeInput(i);
            }
        }

        loadComposerPrompts(node).then(() => {
            buildComposerSelectorBar(node);
            buildComposerButtonBar(node);
            syncSelectorToData(node);
            updateComposerLastSavedState(node);
            refreshComposerPromptInputGhosting(node);
        });

        refreshComposerPromptInputGhosting(node);

        return result;
    };
}

function syncSelectorToData(node) {
    if (typeof node.updateComposerSelectorDisplay === "function") {
        node.updateComposerSelectorDisplay();
    }
}

function shouldWarnComposerUnsavedChanges() {
    return app.ui.settings.getSettingValue("PromptManager.WarnUnsavedChanges") !== false;
}

function refreshComposerPromptInputGhosting(node) {
    const textWidget = node.widgets?.find((w) => w.name === "text");
    const usePromptInputWidget = node.widgets?.find((w) => w.name === "use_prompt_input");
    if (!textWidget || !usePromptInputWidget) return;

    const promptInputConnection = node.inputs?.find((inp) => inp.name === "prompt");
    const isPromptConnected = promptInputConnection && promptInputConnection.link != null;
    const usePromptInput = usePromptInputWidget.value === true;

    if (usePromptInput && isPromptConnected) {
        if (typeof node._composerIncomingPrompt === "string") {
            textWidget.value = node._composerIncomingPrompt;
        }
        textWidget.disabled = true;
        if (textWidget.inputEl) {
            textWidget.inputEl.style.pointerEvents = "auto";
            textWidget.inputEl.readOnly = true;
        }
    } else {
        textWidget.disabled = false;
        if (textWidget.inputEl) {
            textWidget.inputEl.readOnly = false;
        }
    }
}

function buildComposerSelectorBar(node) {
    if (node._composerSelectorBuilt) return;
    node._composerSelectorBuilt = true;

    const categoryWidget = node.widgets.find((w) => w.name === "category");
    const nameWidget = node.widgets.find((w) => w.name === "name");
    const textWidget = node.widgets.find((w) => w.name === "text");
    if (!categoryWidget || !nameWidget) return;

    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        align-items: center;
        gap: 0;
        background: #1a1a1a;
        border-radius: 4px;
        overflow: visible;
        height: 26px;
        margin: -8px 0 0 0;
        position: relative;
    `;

    const leftArrow = document.createElement("button");
    leftArrow.textContent = "◀";
    leftArrow.style.cssText = `
        background: #2a2a2a;
        border: none;
        color: #888;
        padding: 0 10px;
        height: 100%;
        cursor: pointer;
        font-size: 10px;
        transition: all 0.15s ease;
    `;
    leftArrow.onmouseover = () => { leftArrow.style.background = "#3a3a3a"; leftArrow.style.color = "#fff"; };
    leftArrow.onmouseout = () => { leftArrow.style.background = "#2a2a2a"; leftArrow.style.color = "#888"; };

    const nameDisplay = document.createElement("div");
    nameDisplay.style.cssText = `
        flex: 1;
        text-align: center;
        color: #dbeafe;
        font-size: 13px;
        padding: 0 10px;
        cursor: pointer;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        background: rgba(56, 130, 246, 0.22);
        border: 1px solid rgba(56, 130, 246, 0.85);
        border-radius: 4px;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s ease;
        box-sizing: border-box;
    `;
    nameDisplay.onmouseover = () => { nameDisplay.style.background = "rgba(56, 130, 246, 0.32)"; };
    nameDisplay.onmouseout = () => { nameDisplay.style.background = "rgba(56, 130, 246, 0.22)"; };

    // Thumbnail preview tooltip on hover (matches Prompt Manager Advanced).
    const thumbnailPreview = document.createElement("div");
    thumbnailPreview.style.cssText = `
        position: fixed;
        display: none;
        z-index: 10001;
        pointer-events: none;
    `;
    const thumbnailImg = document.createElement("img");
    thumbnailImg.style.cssText = `
        max-width: 300px;
        max-height: 300px;
        object-fit: contain;
        border-radius: 8px;
        display: block;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    `;
    thumbnailPreview.appendChild(thumbnailImg);
    document.body.appendChild(thumbnailPreview);

    let hoverTimeout = null;
    nameDisplay.addEventListener("mouseenter", () => {
        hoverTimeout = setTimeout(() => {
            const category = categoryWidget.value;
            const prompt = nameWidget.value;
            const promptData = node.composerPrompts?.[category]?.[prompt] || node.prompts?.[category]?.[prompt];
            const thumbnail = promptData?.thumbnail;
            if (!thumbnail) return;

            const tempImg = new Image();
            tempImg.onload = function() {
                const imgWidth = Math.min(this.naturalWidth, 300);
                const imgHeight = Math.min(this.naturalHeight, 300);
                thumbnailImg.style.width = imgWidth + "px";
                thumbnailImg.style.height = imgHeight + "px";
                thumbnailImg.src = thumbnail;

                const rect = nameDisplay.getBoundingClientRect();
                const margin = 8;
                let left = rect.left + (rect.width / 2) - (imgWidth / 2);
                let top = rect.top - imgHeight - margin;
                left = Math.max(5, Math.min(left, window.innerWidth - imgWidth - 5));
                top = Math.max(5, Math.min(top, window.innerHeight - imgHeight - 5));
                thumbnailPreview.style.left = left + "px";
                thumbnailPreview.style.top = top + "px";
                thumbnailPreview.style.display = "block";
            };
            tempImg.src = thumbnail;
        }, 300);
    });
    nameDisplay.addEventListener("mouseleave", () => {
        if (hoverTimeout) clearTimeout(hoverTimeout);
        thumbnailPreview.style.display = "none";
    });

    node.onRemoved = function() {
        if (thumbnailPreview && thumbnailPreview.parentNode) {
            thumbnailPreview.parentNode.removeChild(thumbnailPreview);
        }
    };

    const rightArrow = document.createElement("button");
    rightArrow.textContent = "▶";
    rightArrow.style.cssText = `
        background: #2a2a2a;
        border: none;
        color: #888;
        padding: 0 10px;
        height: 100%;
        cursor: pointer;
        font-size: 10px;
        transition: all 0.15s ease;
    `;
    rightArrow.onmouseover = () => { rightArrow.style.background = "#3a3a3a"; rightArrow.style.color = "#fff"; };
    rightArrow.onmouseout = () => { rightArrow.style.background = "#2a2a2a"; rightArrow.style.color = "#888"; };

    container.appendChild(leftArrow);
    container.appendChild(nameDisplay);
    container.appendChild(rightArrow);

    const getAllFlat = () => {
        const list = [];
        const data = node.composerPrompts || {};
        for (const cat of getComposerCategories(node)) {
            for (const name of getComposerNames(node, cat)) {
                list.push({ category: cat, prompt: name });
            }
        }
        return list;
    };

    const getCurrentIndex = (list) => {
        return list.findIndex((p) => p.category === categoryWidget.value && p.prompt === nameWidget.value);
    };

    const navigateTo = async (item, skipCheck = false) => {
        if (!skipCheck && shouldWarnComposerUnsavedChanges() && hasComposerUnsavedChanges(node)) {
            const confirmed = await showConfirm(
                "Unsaved Changes",
                "You have unsaved changes to the current fragment. Discard them and switch?",
                "Discard & Switch",
                "#f80"
            );
            if (!confirmed) return false;
        }

        node.isNewUnsavedPrompt = false;
        node.newPromptCategory = null;
        node.newPromptName = null;

        const categoryChanged = item.category !== categoryWidget.value;
        if (categoryChanged) {
            categoryWidget.value = item.category;
            if (typeof categoryWidget.callback === "function") {
                await categoryWidget.callback(item.category);
            }
        }

        nameWidget.value = item.prompt;
        if (typeof nameWidget.callback === "function") {
            await nameWidget.callback(item.prompt);
        }

        const usePromptInputWidget = node.widgets.find((w) => w.name === "use_prompt_input");
        if (textWidget && usePromptInputWidget?.value !== true) {
            const entry = getComposerEntry(node, item.category, item.prompt);
            textWidget.value = entry?.prompt || "";
        }

        updateComposerLastSavedState(node);
        refreshComposerPromptInputGhosting(node);
        updateDisplay();
        app.graph.setDirtyCanvas(true, true);
        return true;
    };

    leftArrow.onclick = async (e) => {
        e.stopPropagation();
        const list = getAllFlat();
        if (list.length === 0) return;
        const idx = getCurrentIndex(list);
        const newIdx = idx <= 0 ? list.length - 1 : idx - 1;
        await navigateTo(list[newIdx]);
    };

    rightArrow.onclick = async (e) => {
        e.stopPropagation();
        const list = getAllFlat();
        if (list.length === 0) return;
        const idx = getCurrentIndex(list);
        const newIdx = idx >= list.length - 1 ? 0 : idx + 1;
        await navigateTo(list[newIdx]);
    };

    nameDisplay.onclick = async (e) => {
        e.stopPropagation();
        if (shouldWarnComposerUnsavedChanges() && hasComposerUnsavedChanges(node)) {
            const confirmed = await showConfirm(
                "Unsaved Changes",
                "You have unsaved changes to the current fragment. Discard them and edit?",
                "Discard & Edit",
                "#f80"
            );
            if (!confirmed) return;
        }

        const selection = await showThumbnailBrowser(node, categoryWidget.value, nameWidget.value, {
            title: "Edit Prompt Composer Fragments",
            promptOnly: true,
            editMode: true,
            endpointPrefix: COMPOSER_ENDPOINT_PREFIX,
            loadPromptsFn: loadComposerPrompts,
            preferenceScope: "composer",
        });

        if (selection?.prompt) {
            categoryWidget.value = selection.category;
            if (typeof categoryWidget.callback === "function") {
                await categoryWidget.callback(selection.category);
            }
            nameWidget.value = selection.prompt;
            if (typeof nameWidget.callback === "function") {
                await nameWidget.callback(selection.prompt);
            }
            const entry = getComposerEntry(node, selection.category, selection.prompt);
            if (textWidget && entry) {
                textWidget.value = entry.prompt || "";
            }
            updateComposerLastSavedState(node);
            updateDisplay();
            app.graph.setDirtyCanvas(true, true);
        }
    };

    const updateDisplay = () => {
        const category = categoryWidget.value || "";
        const prompt = nameWidget.value || "new fragment";
        nameDisplay.textContent = `${category} : ${prompt}`;
        nameDisplay.title = `${category} : ${prompt}`;
    };

    updateDisplay();

    const widget = node.addDOMWidget("composer_selector", "div", container);
    widget.computeSize = function(width) {
        return [width, 24];
    };
    node._composerSelectorContainer = container;
    node.updateComposerSelectorDisplay = updateDisplay;
}

function buildComposerButtonBar(node) {
    if (node._composerButtonBarBuilt) return;
    node._composerButtonBarBuilt = true;

    const categoryWidget = node.widgets.find((w) => w.name === "category");
    const nameWidget = node.widgets.find((w) => w.name === "name");
    const textWidget = node.widgets.find((w) => w.name === "text");
    if (!categoryWidget || !nameWidget || !textWidget) return;

    const buttonContainer = document.createElement("div");
    buttonContainer.style.cssText = `
        display: flex;
        gap: 8px;
        padding: 2px 0 0 0;
        align-items: center;
        justify-content: space-between;
    `;

    const saveBtn = createComposerButton("Save Prompt", async () => {
        const currentCategory = String(categoryWidget.value || "").trim();
        const currentName = String(nameWidget.value || "").trim();

        const selection = await showThumbnailBrowser(node, currentCategory, currentName, {
            title: "Save Prompt Composer Fragment",
            endpointPrefix: COMPOSER_ENDPOINT_PREFIX,
            promptOnly: true,
            mode: "save",
            initialName: currentName,
            saveButtonText: "Save",
            onSave: async (category, name) => {
                const text = String(textWidget.value || "").trim();
                const existing = getComposerEntry(node, category, name);
                const result = await saveComposerPrompt(node, category, name, text, existing?.thumbnail || null);
                if (result?.success) {
                    categoryWidget.value = category;
                    if (typeof categoryWidget.callback === "function") {
                        await categoryWidget.callback(category);
                    }
                    nameWidget.value = name;
                    if (typeof nameWidget.callback === "function") {
                        await nameWidget.callback(name);
                    }
                    updateComposerLastSavedState(node);
                    if (typeof node.updateComposerSelectorDisplay === "function") {
                        node.updateComposerSelectorDisplay();
                    }
                    app.graph.setDirtyCanvas(true, true);
                }
                return result;
            },
            loadPromptsFn: loadComposerPrompts,
            preferenceScope: "composer",
        });

        if (selection?.prompt) {
            categoryWidget.value = selection.category;
            if (typeof categoryWidget.callback === "function") {
                await categoryWidget.callback(selection.category);
            }
            nameWidget.value = selection.prompt;
            if (typeof nameWidget.callback === "function") {
                await nameWidget.callback(selection.prompt);
            }
            const entry = getComposerEntry(node, selection.category, selection.prompt);
            if (textWidget && entry) {
                textWidget.value = entry.prompt || "";
            }
            updateComposerLastSavedState(node);
            if (typeof node.updateComposerSelectorDisplay === "function") {
                node.updateComposerSelectorDisplay();
            }
            app.graph.setDirtyCanvas(true, true);
        }
    });

    const newBtn = createComposerButton("New Prompt", async () => {
        if (shouldWarnComposerUnsavedChanges() && hasComposerUnsavedChanges(node)) {
            const confirmed = await showConfirm(
                "Unsaved Changes",
                "You have unsaved changes to the current fragment. Discard them and start fresh?",
                "Discard & Continue",
                "#f80"
            );
            if (!confirmed) return;
        }

        const currentCategory = categoryWidget.value;
        nameWidget.value = "";
        textWidget.value = "";

        node.isNewUnsavedPrompt = true;
        node.newPromptCategory = currentCategory;
        node.newPromptName = null;
        node.composerLastSavedState = null;

        if (typeof node.updateComposerSelectorDisplay === "function") {
            node.updateComposerSelectorDisplay();
        }
        app.graph.setDirtyCanvas(true, true);
    });

    const moreBtn = createComposerDropdownButton("More ▼", [
        {
            label: "Delete Prompt",
            action: async () => {
                const category = categoryWidget.value;
                const name = nameWidget.value;
                if (!name) {
                    await showInfo("Error", "No fragment selected to delete.");
                    return;
                }
                const confirmed = await showConfirm(
                    "Delete Fragment",
                    `Are you sure you want to delete "${name}" from "${category}"? This cannot be undone.`,
                    "Delete",
                    "#c00"
                );
                if (confirmed) {
                    await deleteComposerPrompt(node, category, name);
                    nameWidget.value = "";
                    textWidget.value = "";
                    if (typeof node.updateComposerSelectorDisplay === "function") {
                        node.updateComposerSelectorDisplay();
                    }
                    app.graph.setDirtyCanvas(true, true);
                }
            }
        },
        { divider: true },
        {
            label: "Export JSON",
            action: () => exportComposerJSON(node),
        },
        {
            label: "Import JSON",
            action: () => importComposerJSON(node),
        },
    ]);

    buttonContainer.appendChild(saveBtn);
    buttonContainer.appendChild(newBtn);
    buttonContainer.appendChild(moreBtn);

    const widget = node.addDOMWidget("composer_buttons", "div", buttonContainer);
    widget.computeSize = function(width) {
        return [width, 40];
    };
    node._composerButtonBarContainer = buttonContainer;
}

function hasComposerUnsavedChanges(node) {
    const categoryWidget = node.widgets.find((w) => w.name === "category");
    const nameWidget = node.widgets.find((w) => w.name === "name");
    const textWidget = node.widgets.find((w) => w.name === "text");
    const usePromptInputWidget = node.widgets.find((w) => w.name === "use_prompt_input");
    if (!categoryWidget || !nameWidget || !textWidget) return false;

    // Mirror Prompt Manager Advanced: input-driven text is read-only and not treated as dirty edits.
    if (usePromptInputWidget?.value === true) {
        return false;
    }

    const category = categoryWidget.value || "";
    const name = nameWidget.value || "";
    const text = textWidget.value || "";

    if (node.isNewUnsavedPrompt) {
        return text.trim().length > 0 || name.trim().length > 0;
    }

    if (!node.composerLastSavedState) {
        const entry = getComposerEntry(node, category, name);
        node.composerLastSavedState = entry ? { category, name, text: entry.prompt || "" } : null;
    }

    if (!node.composerLastSavedState) return false;
    return (
        node.composerLastSavedState.category !== category ||
        node.composerLastSavedState.name !== name ||
        node.composerLastSavedState.text !== text
    );
}

function updateComposerLastSavedState(node) {
    const categoryWidget = node.widgets.find((w) => w.name === "category");
    const nameWidget = node.widgets.find((w) => w.name === "name");
    const textWidget = node.widgets.find((w) => w.name === "text");
    if (!categoryWidget || !nameWidget || !textWidget) return;
    node.composerLastSavedState = {
        category: categoryWidget.value || "",
        name: nameWidget.value || "",
        text: textWidget.value || "",
    };
}

function exportComposerJSON(node) {
    const data = node.composerPrompts || {};
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "prompt_composer_data.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

async function importComposerJSON(node) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        try {
            const text = await file.text();
            const data = JSON.parse(text);
            if (!data || typeof data !== "object") {
                await showInfo("Import Failed", "Invalid JSON file.");
                return;
            }
            // Bulk import via category-by-category save.
            let imported = 0;
            for (const [category, entries] of Object.entries(data)) {
                if (category === "__meta__" || typeof entries !== "object") continue;
                for (const [name, entry] of Object.entries(entries)) {
                    if (name === "__meta__") continue;
                    const promptText = typeof entry === "string" ? entry : (entry?.prompt || "");
                    await saveComposerPrompt(node, category, name, promptText, entry?.thumbnail || null);
                    imported++;
                }
            }
            await showInfo("Import Complete", `Imported ${imported} fragments.`);
            if (typeof node.updateComposerSelectorDisplay === "function") {
                node.updateComposerSelectorDisplay();
            }
        } catch (err) {
            console.error("[PromptComposerManager] Import error:", err);
            await showInfo("Import Failed", err.message || "Unknown error");
        }
    };
    input.click();
}

app.registerExtension({
    name: "PromptComposerManager",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        setupPromptComposerManager(nodeType, nodeData);
    },
});
