import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "PromptManager",
    settings: [
        {
            id: "PromptManager.PreferredBaseModel",
            category: ["Prompt Manager", "Model Preferences", "Base Model (text enhancement)"],
            name: "Preferred Base Model",
            tooltip: "Default model for 'Enhance User Prompt' mode (leave empty to auto-select smallest)",
            type: "text",
            defaultValue: "",
            async onChange(value) {
                try {
                    await fetch("/prompt-manager/save-preference", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ key: "preferred_base_model", value: value })
                    });
                } catch (error) {
                    console.error("[PromptManager] Error saving base model preference:", error);
                }
            }
        },
        {
            id: "PromptManager.PreferredVisionModel",
            category: ["Prompt Manager", "Model Preferences", "Vision Model (image analysis)"],
            name: "Preferred Vision Model",
            tooltip: "Default model for 'Analyze Image' modes (leave empty to auto-select smallest)",
            type: "text",
            defaultValue: "",
            async onChange(value) {
                try {
                    await fetch("/prompt-manager/save-preference", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ key: "preferred_vision_model", value: value })
                    });
                } catch (error) {
                    console.error("[PromptManager] Error saving vision model preference:", error);
                }
            }
        },
        {
            id: "PromptManager.LlamaPath",
            category: ["Prompt Manager", "Llama Preferences", "Custom Llama Path"],
            name: "Custom Llama Path",
            tooltip: "Path to custom Llama installation (Can leave empty if it's defined in your system Path)",
            type: "text",
            defaultValue: "",
            async onChange(value) {
                try {
                    await fetch("/prompt-manager/save-preference", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ key: "custom_llama_path", value: value })
                    });
                } catch (error) {
                    console.error("[PromptManager] Error saving Llama path preference:", error);
                }
            }
        }
    ],
    async setup() {
        // Load settings from ComfyUI and sync to Python cache
        try {
            const baseModel = app.ui.settings.getSettingValue("PromptManager.PreferredBaseModel", "");
            const visionModel = app.ui.settings.getSettingValue("PromptManager.PreferredVisionModel", "");
            
            // Sync both settings to Python cache
            if (baseModel) {
                await fetch("/prompt-manager/save-preference", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ key: "preferred_base_model", value: baseModel })
                });
            }
            if (visionModel) {
                await fetch("/prompt-manager/save-preference", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ key: "preferred_vision_model", value: visionModel })
                });
            }
        } catch (error) {
            console.error("[PromptManager] Error syncing preferences:", error);
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PromptManager") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                const node = this;
                node.prompts = {};
                
                // Set initial size to be square (400x400)
                this.setSize([400, 400]);

                // Change widget labels
                const promptTextWidget = this.widgets.find(w => w.name === "text");
                if (promptTextWidget) {
                    promptTextWidget.label = "prompt";
                }

                const promptNameWidget = this.widgets.find(w => w.name === "name");
                if (promptNameWidget) {
                    promptNameWidget.label = "name";
                }

                const useExternalWidget = this.widgets.find(w => w.name === "use_external");
                if (useExternalWidget) {
                    useExternalWidget.label = "use llm input";
                }

                // Make text widget scrollable even when disabled
                if (promptTextWidget && promptTextWidget.inputEl) {
                    promptTextWidget.inputEl.style.pointerEvents = "auto";
                    promptTextWidget.inputEl.addEventListener("mousedown", function(e) {
                        if (this.disabled) {
                            // Allow scrolling but prevent editing
                            e.stopPropagation();
                        }
                    });
                }

                // Listen for text updates from backend (when connected inputs change or toggle changes)
                api.addEventListener("prompt-manager-update-text", (event) => {
                    if (String(event.detail.node_id) === String(this.id)) {
                        if (promptTextWidget) {
                            const useExternal = event.detail.use_external || false;
                            const llmInput = event.detail.llm_input || "";
                            
                            if (useExternal && llmInput) {
                                // When using external, display the LLM input text (grayed out)
                                promptTextWidget.value = llmInput;
                                promptTextWidget.disabled = true;
                                // Keep scrolling enabled
                                if (promptTextWidget.inputEl) {
                                    promptTextWidget.inputEl.style.pointerEvents = "auto";
                                }
                            } else {
                                // When using internal, enable the widget
                                promptTextWidget.disabled = false;
                            }
                            
                            this.serialize_widgets = true;
                            app.graph.setDirtyCanvas(true, true);
                        }
                    }
                });

                // Load prompts on creation
                loadPrompts(node).then(() => {
                    addButtonBar(node);
                    setupCategoryChangeHandler(node);
                    filterPromptDropdown(node);
                    setupUseExternalToggleHandler(node);

                    // After buttons are added, ensure height is sufficient
                    setTimeout(() => {
                        const computedSize = node.computeSize();
                        const minHeight = Math.max(400, computedSize[1] + 20);
                        
                        if (node.size[1] < minHeight) {
                            node.setSize([Math.max(400, node.size[0]), minHeight]);
                        }
                    }, 200);
                });

                return result;
            };

            // Handle node reconfiguration when ComfyUI refreshes
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                const result = onConfigure?.apply(this, arguments);

                const node = this;

                // Reload prompts from server and reapply filtering
                setTimeout(() => {
                    loadPrompts(node).then(() => {
                        filterPromptDropdown(node);

                        // Reattach button bar if needed
                        if (!node.buttonBarAttached) {
                            addButtonBar(node);
                            setupCategoryChangeHandler(node);
                            setupUseExternalToggleHandler(node);
                        }
                    });
                }, 100);

                return result;
            };

            // Enforce minimum node width
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                size[0] = Math.max(300, size[0]);
                return onResize ? onResize.apply(this, arguments) : size;
            };
        }

        // Make the Options node taller/wider by default so the system_prompt area is roomier
        if (nodeData.name === "PromptGenOptions") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                // Set a default size (width x height)
                try {
                    this.setSize([400, 420]);
                } catch (e) {
                    // ignore if method unavailable
                }
                return result;
            };
            // Enforce sensible minimums when the user resizes the options node
            const onResizeOpt = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                size[0] = Math.max(300, size[0]);
                return onResizeOpt ? onResizeOpt.apply(this, arguments) : size;
            };
        }

        // Enforce minimum size for ModelGenerator node (no default size)
        if (nodeData.name === "PromptGenerator") {
           const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                // Set a default size (width x height)
                try {
                    this.setSize([400, 300]);
                } catch (e) {
                    // ignore if method unavailable
                }
                return result;
            };
            const onResizeModel = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                size[0] = Math.max(300, size[0]);
                return onResizeModel ? onResizeModel.apply(this, arguments) : size;
            };
        }
    }
});

async function loadPrompts(node) {
    try {
        const response = await fetch("/prompt-manager/get-prompts");
        const data = await response.json();
        node.prompts = data;
        return data;
    } catch (error) {
        console.error("[PromptManager] Error loading prompts:", error);
        return {};
    }
}

function filterPromptDropdown(node) {
    // Filter name dropdown to only show prompts from current category
    const categoryWidget = node.widgets.find(w => w.name === "category");
    const promptWidget = node.widgets.find(w => w.name === "name");

    if (categoryWidget && promptWidget) {
        const currentCategory = categoryWidget.value;
        if (node.prompts[currentCategory]) {
            const promptNames = Object.keys(node.prompts[currentCategory]);
            if (promptNames.length === 0) {
                promptNames.push("");
            }
            promptWidget.options.values = promptNames;
        }
    }
}

function addButtonBar(node) {
    if (node.buttonBarAttached) {
        return;
    }

    const textWidget = node.widgets.find(w => w.name === "text");
    const categoryWidget = node.widgets.find(w => w.name === "category");
    const promptWidget = node.widgets.find(w => w.name === "name");

    // Create button container
    const buttonContainer = document.createElement("div");
    buttonContainer.style.display = "flex";
    buttonContainer.style.gap = "4px";
    buttonContainer.style.padding = "2px 4px 4px 4px";
    buttonContainer.style.flexWrap = "nowrap";
    buttonContainer.style.marginTop = "-10px";

    // Create action buttons
    const createCategoryBtn = createButton("New Category", async () => {
        const categoryName = await showTextPrompt("New Category", "Enter new category name:");
        
        if (categoryName && categoryName.trim()) {
            // Case-insensitive check for existing category - PREVENT overwriting
            let existingCategoryName = null;
            if (node.prompts) {
                const existingCategories = Object.keys(node.prompts);
                existingCategoryName = existingCategories.find(cat => cat.toLowerCase() === categoryName.trim().toLowerCase());
            }

            if (existingCategoryName) {
                await showInfo(
                    "Category Exists",
                    `Category already exists as "${existingCategoryName}". Cannot overwrite existing categories.`
                );
                return;
            }

            await createCategory(node, categoryName.trim());
        }
    });

    const savePromptBtn = createButton("Save Prompt", async () => {
        const promptName = await showTextPrompt("Save Prompt", "Enter prompt name:", promptWidget.value || "New Prompt");
        
        if (promptName && promptName.trim()) {
            const promptText = textWidget.value;

            if (promptText && promptText.trim()) {
                const currentCategory = categoryWidget.value;
                
                // Case-insensitive check for existing prompt
                let existingPromptName = null;
                if (node.prompts[currentCategory]) {
                    const existingNames = Object.keys(node.prompts[currentCategory]);
                    existingPromptName = existingNames.find(name => name.toLowerCase() === promptName.trim().toLowerCase());
                }

                // Ask for confirmation before overwriting existing prompt
                if (existingPromptName) {
                    const confirmed = await showConfirm(
                        "Overwrite Prompt",
                        `Prompt "${existingPromptName}" already exists in category "${currentCategory}". Do you want to overwrite it?`,
                        "Overwrite",
                        "#f80"
                    );

                    if (!confirmed) {
                        return;
                    }
                }

                await savePrompt(node, categoryWidget.value, promptName.trim(), promptText.trim());
            } else {
                await showInfo("Empty Prompt", "Prompt text cannot be empty");
            }
        }
    });

    const deleteCategoryBtn = createButton("Del Category", async () => {
        if (await showConfirm("Delete Category", `Are you sure you want to delete category "${categoryWidget.value}"?`)) {
            await deleteCategory(node, categoryWidget.value);
        }
    });

    const deletePromptBtn = createButton("Del Prompt", async () => {
        if (await showConfirm("Delete Prompt", `Are you sure you want to delete prompt "${promptWidget.value}"?`)) {
            await deletePrompt(node, categoryWidget.value, promptWidget.value);
        }
    });

    buttonContainer.appendChild(savePromptBtn);
    buttonContainer.appendChild(createCategoryBtn);
    buttonContainer.appendChild(deleteCategoryBtn);
    buttonContainer.appendChild(deletePromptBtn);

    // Add button bar to node
    const htmlWidget = node.addDOMWidget("buttons", "div", buttonContainer);
    htmlWidget.computeSize = function(width) {
        return [width, 34];
    };

    node.buttonBarAttached = true;
}

function setupCategoryChangeHandler(node) {
    const categoryWidget = node.widgets.find(w => w.name === "category");
    const promptWidget = node.widgets.find(w => w.name === "name");
    const textWidget = node.widgets.find(w => w.name === "text");
    const useExternalWidget = node.widgets.find(w => w.name === "use_external");

    if (!categoryWidget || !promptWidget || !textWidget) return;

    const originalCallback = categoryWidget.callback;

    // Update prompt dropdown when category changes
    categoryWidget.callback = function(value) {
        if (originalCallback) {
            originalCallback.apply(this, arguments);
        }

        const category = value;
        if (node.prompts && node.prompts[category]) {
            const promptNames = Object.keys(node.prompts[category]);
            promptWidget.options.values = promptNames;

            if (promptNames.length > 0) {
                promptWidget.value = promptNames[0];
                // Only update text widget if not using external
                const useExternal = useExternalWidget ? useExternalWidget.value : false;
                if (!useExternal) {
                    const promptData = node.prompts[category][promptNames[0]];
                    textWidget.value = promptData?.prompt || "";
                }
            } else {
                promptWidget.value = "";
                if (!useExternal) {
                    textWidget.value = "";
                }
            }

            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
        }
    };

    const originalPromptCallback = promptWidget.callback;

    // Update text widget when prompt selection changes
    promptWidget.callback = function(value) {
        if (originalPromptCallback) {
            originalPromptCallback.apply(this, arguments);
        }

        // Only update text if not using external
        const useExternal = useExternalWidget ? useExternalWidget.value : false;
        if (!useExternal) {
            const category = categoryWidget.value;
            if (node.prompts && node.prompts[category] && node.prompts[category][value]) {
                const promptData = node.prompts[category][value];
                textWidget.value = promptData?.prompt || "";
            }
        }

        node.serialize_widgets = true;
        app.graph.setDirtyCanvas(true, true);
    };

    // Prevent ComfyUI from setting all prompts instead of filtered category prompts
    Object.defineProperty(promptWidget.options, 'values', {
        get: function() {
            return this._values;
        },
        set: function(newValues) {
            this._values = newValues;

            // Re-filter if ComfyUI tries to set the master list of all prompts
            if (node.prompts && categoryWidget && newValues && newValues.length > 0) {
                const currentCategory = categoryWidget.value;
                if (node.prompts[currentCategory]) {
                    const categoryPrompts = Object.keys(node.prompts[currentCategory]);
                    const hasOtherCategoryPrompts = newValues.some(val =>
                        val !== "" && !categoryPrompts.includes(val)
                    );

                    if (hasOtherCategoryPrompts) {
                        setTimeout(() => {
                            filterPromptDropdown(node);
                        }, 50);
                    }
                }
            }
        },
        enumerable: true,
        configurable: true
    });

    promptWidget.options._values = promptWidget.options.values || [];
}

function setupUseExternalToggleHandler(node) {
    const textWidget = node.widgets?.find(w => w.name === "text");
    const useExternalWidget = node.widgets?.find(w => w.name === "use_external");
    const categoryWidget = node.widgets?.find(w => w.name === "category");
    const promptWidget = node.widgets?.find(w => w.name === "name");
    
    if (!textWidget || !useExternalWidget) return;

    // Apply initial state on load/reload
    const applyToggleState = (value) => {
        const llmInputConnection = node.inputs?.find(inp => inp.name === "llm_input");
        const isLlmConnected = llmInputConnection && llmInputConnection.link != null;

        if (value && isLlmConnected) {
            // Using LLM and it's connected - disable text widget immediately
            textWidget.disabled = true;
            // Keep scrolling enabled but prevent editing
            if (textWidget.inputEl) {
                textWidget.inputEl.style.pointerEvents = "auto";
                textWidget.inputEl.readOnly = true;
            }
            
            // Try to show LLM value if available
            const graph = app.graph;
            const link = graph.links[llmInputConnection.link];
            if (link) {
                const originNode = graph.getNodeById(link.origin_id);
                if (originNode) {
                    // Try to get the output value from the origin node
                    const outputData = originNode.getOutputData?.(link.origin_slot);
                    if (outputData !== undefined) {
                        textWidget.value = outputData;
                    } else if (originNode.widgets) {
                        // Fallback: try to find widget with matching output
                        const outputWidget = originNode.widgets.find(w => w.name === "text" || w.name === "STRING");
                        if (outputWidget) {
                            textWidget.value = outputWidget.value;
                        }
                    }
                }
            }
        } else {
            // Using internal text - enable widget but keep current text (which may be LLM output)
            textWidget.disabled = false;
            if (textWidget.inputEl) {
                textWidget.inputEl.readOnly = false;
            }
        }
    };

    // Apply initial state when the handler is set up
    applyToggleState(useExternalWidget.value);

    const originalCallback = useExternalWidget.callback;

    useExternalWidget.callback = function(value) {
        // Check if llm_input is connected
        const llmInputConnection = node.inputs?.find(inp => inp.name === "llm_input");
        const isLlmConnected = llmInputConnection && llmInputConnection.link != null;

        // Prevent turning on if nothing is connected
        if (value && !isLlmConnected) {
            useExternalWidget.value = false;
            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
            return;
        }

        if (originalCallback) {
            originalCallback.apply(this, arguments);
        }

        applyToggleState(value);

        node.serialize_widgets = true;
        app.graph.setDirtyCanvas(true, true);
    };
}

function createButton(text, callback) {
    const button = document.createElement("button");
    button.textContent = text;
    button.style.flex = "1";
    button.style.minWidth = "80px";
    button.style.padding = "6px 8px";
    button.style.cursor = "pointer";
    button.style.backgroundColor = "#222";
    button.style.color = "#fff";
    button.style.border = "1px solid #444";
    button.style.borderRadius = "3px";
    button.style.fontSize = "11px";
    button.style.fontWeight = "normal";
    button.style.whiteSpace = "nowrap";
    button.style.overflow = "hidden";
    button.style.textOverflow = "ellipsis";
    button.style.lineHeight = "normal";
    button.style.textAlign = "center";
    button.style.height = "28px";
    button.style.display = "flex";
    button.style.alignItems = "center";
    button.style.justifyContent = "center";
    button.onclick = callback;

    return button;
}

function showTextPrompt(title, message, defaultValue = "") {
    return new Promise((resolve) => {
        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #222;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 20px;
            z-index: 10000;
            min-width: 300px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        `;

        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 9999;
        `;

        dialog.innerHTML = `
            <div style="margin-bottom: 15px; font-size: 16px; font-weight: bold; color: #fff;">${title}</div>
            <div style="margin-bottom: 10px; color: #ccc;">${message}</div>
            <input type="text" value="${defaultValue}" style="width: 100%; padding: 8px; margin-bottom: 15px; background: #333; border: 1px solid #555; color: #fff; border-radius: 4px; font-size: 14px;" />
            <div style="display: flex; gap: 10px; justify-content: flex-end;">
                <button class="cancel-btn" style="padding: 8px 16px; background: #555; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
                <button class="ok-btn" style="padding: 8px 16px; background: #0a0; color: #fff; border: none; border-radius: 4px; cursor: pointer;">OK</button>
            </div>
        `;

        const input = dialog.querySelector("input");
        const okBtn = dialog.querySelector(".ok-btn");
        const cancelBtn = dialog.querySelector(".cancel-btn");

        const cleanup = () => {
            document.body.removeChild(overlay);
            document.body.removeChild(dialog);
        };

        const handleOk = () => {
            resolve(input.value);
            cleanup();
        };

        const handleCancel = () => {
            resolve(null);
            cleanup();
        };

        okBtn.onclick = handleOk;
        cancelBtn.onclick = handleCancel;
        overlay.onclick = handleCancel;

        input.onkeydown = (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                e.stopPropagation();
                handleOk();
            } else if (e.key === "Escape") {
                e.preventDefault();
                e.stopPropagation();
                handleCancel();
            }
        };

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        input.focus();
        input.select();
    });
}

function showConfirm(title, message, confirmText = "Delete", confirmColor = "#c00") {
    return new Promise((resolve) => {
        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #222;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 20px;
            z-index: 10000;
            min-width: 300px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        `;

        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 9999;
        `;

        dialog.innerHTML = `
            <div style="margin-bottom: 15px; font-size: 16px; font-weight: bold; color: #fff;">${title}</div>
            <div style="margin-bottom: 20px; color: #ccc;">${message}</div>
            <div style="display: flex; gap: 10px; justify-content: flex-end;">
                <button class="cancel-btn" style="padding: 8px 16px; background: #555; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
                <button class="ok-btn" style="padding: 8px 16px; background: ${confirmColor}; color: #fff; border: none; border-radius: 4px; cursor: pointer;">${confirmText}</button>
            </div>
        `;

        const okBtn = dialog.querySelector(".ok-btn");
        const cancelBtn = dialog.querySelector(".cancel-btn");

        const cleanup = () => {
            document.body.removeChild(overlay);
            document.body.removeChild(dialog);
        };

        okBtn.onclick = () => {
            resolve(true);
            cleanup();
        };

        cancelBtn.onclick = () => {
            resolve(false);
            cleanup();
        };

        overlay.onclick = () => {
            resolve(false);
            cleanup();
        };

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        okBtn.focus();
    });
}

function showInfo(title, message) {
    return new Promise((resolve) => {
        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #222;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 20px;
            z-index: 10000;
            min-width: 300px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        `;

        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 9999;
        `;

        dialog.innerHTML = `
            <div style="margin-bottom: 15px; font-size: 16px; font-weight: bold; color: #fff;">${title}</div>
            <div style="margin-bottom: 20px; color: #ccc;">${message}</div>
            <div style="display: flex; gap: 10px; justify-content: flex-end;">
                <button class="ok-btn" style="padding: 8px 16px; background: #0a0; color: #fff; border: none; border-radius: 4px; cursor: pointer;">OK</button>
            </div>
        `;

        const okBtn = dialog.querySelector(".ok-btn");

        const cleanup = () => {
            document.body.removeChild(overlay);
            document.body.removeChild(dialog);
        };

        okBtn.onclick = () => {
            resolve(true);
            cleanup();
        };

        overlay.onclick = () => {
            resolve(true);
            cleanup();
        };

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        okBtn.focus();
    });
}

function updateDropdowns(node) {
    const categoryWidget = node.widgets.find(w => w.name === "category");
    const promptWidget = node.widgets.find(w => w.name === "name");
    const textWidget = node.widgets.find(w => w.name === "text");

    if (!categoryWidget || !promptWidget || !textWidget) return;

    // Update category dropdown
    const categories = Object.keys(node.prompts);
    categoryWidget.options.values = categories;

    if (!node.prompts[categoryWidget.value] && categories.length > 0) {
        categoryWidget.value = categories[0];
    }

    // Update prompt dropdown for current category
    const currentCategory = categoryWidget.value;
    if (node.prompts[currentCategory]) {
        const promptNames = Object.keys(node.prompts[currentCategory]);

        if (promptNames.length === 0) {
            promptNames.push("");
        }

        promptWidget.options.values = promptNames;

        if (!node.prompts[currentCategory][promptWidget.value] && promptNames.length > 0) {
            promptWidget.value = promptNames[0];
        }

        if (promptWidget.value && node.prompts[currentCategory][promptWidget.value]) {
            const promptData = node.prompts[currentCategory][promptWidget.value];
            textWidget.value = promptData?.prompt || "";
        } else {
            textWidget.value = "";
        }
    } else {
        promptWidget.options.values = [""];
        promptWidget.value = "";
        textWidget.value = "";
    }
}

async function createCategory(node, categoryName) {
    try {
        const response = await fetch("/prompt-manager/save-category", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ category_name: categoryName })
        });

        const data = await response.json();

        if (data.success) {
            node.prompts = data.prompts;

            // Select the newly created category
            const categoryWidget = node.widgets.find(w => w.name === "category");
            const promptWidget = node.widgets.find(w => w.name === "name");
            const textWidget = node.widgets.find(w => w.name === "text");

            if (categoryWidget) {
                categoryWidget.value = categoryName;
            }

            if (promptWidget) {
                promptWidget.value = "";
            }
            if (textWidget) {
                textWidget.value = "";
            }

            updateDropdowns(node);

            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
        } else {
            await showInfo("Error", data.error);
        }
    } catch (error) {
        console.error("[PromptManager] Error creating category:", error);
        await showInfo("Error", "Error creating category");
    }
}

async function savePrompt(node, category, name, text) {
    try {
        const response = await fetch("/prompt-manager/save-prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                category: category,
                name: name,
                text: text
            })
        });

        const data = await response.json();

        if (data.success) {
            node.prompts = data.prompts;
            updateDropdowns(node);

            // Select the newly saved prompt
            const promptWidget = node.widgets.find(w => w.name === "name");
            if (promptWidget) {
                promptWidget.value = name;
                updateDropdowns(node);
            }

            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error("[PromptManager] Error saving prompt:", error);
        alert("Error saving prompt");
    }
}

async function deleteCategory(node, category) {
    try {
        const response = await fetch("/prompt-manager/delete-category", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ category: category })
        });

        const data = await response.json();

        if (data.success) {
            node.prompts = data.prompts;
            updateDropdowns(node);

            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error("[PromptManager] Error deleting category:", error);
        alert("Error deleting category");
    }
}

async function deletePrompt(node, category, name) {
    try {
        const response = await fetch("/prompt-manager/delete-prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                category: category,
                name: name
            })
        });

        const data = await response.json();

        if (data.success) {
            node.prompts = data.prompts;
            updateDropdowns(node);

            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error("[PromptManager] Error deleting prompt:", error);
        alert("Error deleting prompt");
    }
}

console.log("[PromptManager] Extension loaded");
