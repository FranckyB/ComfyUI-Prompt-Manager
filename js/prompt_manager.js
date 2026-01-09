import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "PromptManager",
    settings: [
        {
            id: "PromptManager.PreferredBaseModel",
            category: ["Prompt Manager", "1. Model Preferences", "Base Model (text enhancement)"],
            name: "Preferred Base Model",
            tooltip: "Filename of default model for 'Enhance User Prompt' mode (leave empty to auto-select model)",
            type: "text",
            defaultValue: "",
            onChange(value) {
                fetch("/prompt-manager/save-preference", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ key: "preferred_base_model", value: value })
                }).catch(error => {
                    console.error("[PromptManager] Error saving base model preference:", error);
                });
            }
        },
        {
            id: "PromptManager.PreferredVisionModel",
            category: ["Prompt Manager", "1. Model Preferences", "Vision Model (image analysis)"],
            name: "Preferred Vision Model",
            tooltip: "Filename of default model for 'Analyze Image' modes (leave empty to auto-select model)",
            type: "text",
            defaultValue: "",
            onChange(value) {
                fetch("/prompt-manager/save-preference", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ key: "preferred_vision_model", value: value })
                }).catch(error => {
                    console.error("[PromptManager] Error saving vision model preference:", error);
                });
            }
        },
        {
            id: "PromptManager.LlamaPath",
            category: ["Prompt Manager", "2. Llama Preferences", "Custom Llama Path"],
            name: "Custom Llama Path",
            tooltip: "Path to custom Llama installation (Can leave empty if it's defined in your system Path)",
            type: "text",
            defaultValue: "",
            onChange(value) {
                fetch("/prompt-manager/save-preference", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ key: "custom_llama_path", value: value })
                }).catch(error => {
                    console.error("[PromptManager] Error saving Llama path preference:", error);
                });
            }
        },
        {
            id: "PromptManager.ModelPath",
            category: ["Prompt Manager", "2. Llama Preferences", "Custom Model Path"],
            name: "Custom Model Path",
            tooltip: "Path to custom model location (If emtpty, defaults to the models/gguf folder)",
            type: "text",
            defaultValue: "",
            onChange(value) {
                fetch("/prompt-manager/save-preference", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ key: "custom_llama_model_path", value: value })
                }).catch(error => {
                    console.error("[PromptManager] Error saving Model path preference:", error);
                });
            }
        },
        {
            id: "PromptManager.Port",
            category: ["Prompt Manager", "2. Llama Preferences", "Port"],
            name: "Port",
            tooltip: "Port number for the Llama server (default 8080)",
            type: "text",
            defaultValue: "8080",
            onChange(value) {
                fetch("/prompt-manager/save-preference", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ key: "custom_llama_port", value: value })
                }).catch(error => {
                    console.error("[PromptManager] Error saving Port preference:", error);
                });
            }
        },
        {
            id: "PromptManager.CloseLlama",
            category: ["Prompt Manager", "3. Exit Preferences", "Close Llama on Exit"],
            name: "Close Llama on Exit",
            tooltip: "If enabled, will close Llama when ComfyUI exits",
            type: "boolean",
            defaultValue: true,
            onChange(value) {
                fetch("/prompt-manager/save-preference", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ key: "close_llama_on_exit", value: value })
                }).catch(error => {
                    console.error("[PromptManager] Error saving Close Llama preference:", error);
                });
            }
        }
    ],
    async setup() {
        // Load settings from ComfyUI and sync to Python cache
        try {
            // Sync current values to Python cache first
            const baseModel = app.ui.settings.getSettingValue("PromptManager.PreferredBaseModel", "");
            const visionModel = app.ui.settings.getSettingValue("PromptManager.PreferredVisionModel", "");
            const llamaPath = app.ui.settings.getSettingValue("PromptManager.LlamaPath", "");
            const modelPath = app.ui.settings.getSettingValue("PromptManager.ModelPath", "");
            const port = app.ui.settings.getSettingValue("PromptManager.Port", "8080");
            const CloseLlama = app.ui.settings.getSettingValue("PromptManager.CloseLlama", true);
            
            console.log("[PromptManager] Syncing preferences:", { baseModel, visionModel, llamaPath, modelPath, port, CloseLlama });
            await fetch("/prompt-manager/save-preference", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ key: "preferred_base_model", value: baseModel })
            });
            await fetch("/prompt-manager/save-preference", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ key: "preferred_vision_model", value: visionModel })
            });
            await fetch("/prompt-manager/save-preference", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ key: "custom_llama_path", value: llamaPath })
            });
            await fetch("/prompt-manager/save-preference", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ key: "custom_llama_model_path", value: modelPath })
            });
            await fetch("/prompt-manager/save-preference", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ key: "custom_llama_port", value: port })
            });
            await fetch("/prompt-manager/save-preference", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ key: "close_llama_on_exit", value: CloseLlama })
            });
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

                // IMPORTANT: Add DOM widgets SYNCHRONOUSLY during node creation
                // to ensure proper positioning and rendering within the node bounds
                addButtonBar(node);
                setupCategoryChangeHandler(node);
                setupUseExternalToggleHandler(node);

                // Load prompts asynchronously (data only, not widgets)
                loadPrompts(node).then(() => {
                    filterPromptDropdown(node);

                    // Ensure height is sufficient after data is loaded
                    setTimeout(() => {
                        const computedSize = node.computeSize();
                        const minHeight = Math.max(400, computedSize[1] + 20);
                        
                        if (node.size[1] < minHeight) {
                            node.setSize([Math.max(400, node.size[0]), minHeight]);
                        }
                        app.graph.setDirtyCanvas(true, true);
                    }, 100);
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

/**
 * Check if there are unsaved changes in the prompt
 */
function hasUnsavedChanges(node) {
    // New unsaved prompt always has unsaved changes
    if (node.isNewUnsavedPrompt) {
        return true;
    }
    
    // For basic PromptManager, check if text differs from saved
    const textWidget = node.widgets?.find(w => w.name === "text");
    const categoryWidget = node.widgets?.find(w => w.name === "category");
    const promptWidget = node.widgets?.find(w => w.name === "name");
    
    if (!textWidget || !categoryWidget || !promptWidget) {
        return false;
    }
    
    const currentText = textWidget.value || "";
    const category = categoryWidget.value;
    const promptName = promptWidget.value;
    
    // If no saved prompt exists, no unsaved changes (unless it's a new prompt)
    if (!node.prompts || !node.prompts[category] || !node.prompts[category][promptName]) {
        return false;
    }
    
    const savedPrompt = node.prompts[category][promptName];
    const savedText = savedPrompt?.prompt || "";
    
    return currentText !== savedText;
}

function filterPromptDropdown(node) {
    // Filter name dropdown to only show prompts from current category
    const categoryWidget = node.widgets.find(w => w.name === "category");
    const promptWidget = node.widgets.find(w => w.name === "name");

    if (categoryWidget && promptWidget) {
        const currentCategory = categoryWidget.value;
        if (node.prompts[currentCategory]) {
            const promptNames = Object.keys(node.prompts[currentCategory]).sort((a, b) => a.localeCompare(b));
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

    // Save Prompt button with category selection
    const savePromptBtn = createButton("Save Prompt", async () => {
        const categories = Object.keys(node.prompts || {}).sort((a, b) => a.localeCompare(b));
        const currentCategory = categoryWidget.value;
        
        const result = await showPromptWithCategory(
            "Save Prompt",
            "Enter prompt name:",
            promptWidget.value || "New Prompt",
            categories,
            currentCategory
        );
        
        if (result && result.name && result.name.trim()) {
            const promptName = result.name.trim();
            const targetCategory = result.category;
            const promptText = textWidget.value;

            if (!promptText || !promptText.trim()) {
                await showInfo("Empty Prompt", "Prompt text cannot be empty");
                return;
            }
            
            // Case-insensitive check for existing prompt
            let existingPromptName = null;
            if (node.prompts[targetCategory]) {
                const existingNames = Object.keys(node.prompts[targetCategory]);
                existingPromptName = existingNames.find(name => name.toLowerCase() === promptName.toLowerCase());
            }

            // Ask for confirmation before overwriting existing prompt
            if (existingPromptName) {
                const confirmed = await showConfirm(
                    "Overwrite Prompt",
                    `Prompt "${existingPromptName}" already exists in category "${targetCategory}". Do you want to overwrite it?`,
                    "Overwrite",
                    "#f80"
                );

                if (!confirmed) {
                    return;
                }
            }

            await savePrompt(node, targetCategory, promptName, promptText.trim());
            
            // Clear new prompt flag since it's now saved
            node.isNewUnsavedPrompt = false;
            node.newPromptCategory = null;
            node.newPromptName = null;
            
            // Update UI to show the saved prompt
            categoryWidget.value = targetCategory;
            filterPromptDropdown(node);
            promptWidget.value = promptName;
        }
    });

    // New Prompt button with category selection
    const newPromptBtn = createButton("New Prompt", async () => {
        // Check for unsaved changes before creating new prompt
        const hasUnsaved = hasUnsavedChanges(node);
        const warnEnabled = app.ui.settings.getSettingValue("PromptManager.warnUnsavedChanges", true);
        
        if (hasUnsaved && warnEnabled) {
            const confirmed = await showConfirm(
                "Unsaved Changes",
                "You have unsaved changes to the current prompt. Do you want to discard them and create a new prompt?",
                "Discard & Continue",
                "#f80"
            );
            if (!confirmed) {
                return;
            }
        }
        
        const categories = Object.keys(node.prompts || {}).sort((a, b) => a.localeCompare(b));
        const currentCategory = categoryWidget.value;
        
        const result = await showPromptWithCategory(
            "New Prompt",
            "Enter new prompt name:",
            "",
            categories,
            currentCategory
        );
        
        if (result && result.name && result.name.trim()) {
            const promptName = result.name.trim();
            const targetCategory = result.category;
            
            // Check for existing prompt
            let existingPromptName = null;
            if (node.prompts[targetCategory]) {
                const existingNames = Object.keys(node.prompts[targetCategory]);
                existingPromptName = existingNames.find(name => name.toLowerCase() === promptName.toLowerCase());
            }

            if (existingPromptName) {
                await showInfo(
                    "Prompt Exists",
                    `Prompt "${existingPromptName}" already exists in category "${targetCategory}".`
                );
                return;
            }

            // Set up UI for new prompt (temporary, not saved yet)
            // User must click "Save Prompt" to persist it
            categoryWidget.value = targetCategory;
            
            // Add prompt name to dropdown temporarily so it can be selected
            if (!promptWidget.options.values.includes(promptName)) {
                promptWidget.options.values = [...promptWidget.options.values, promptName].sort((a, b) => a.localeCompare(b));
            }
            promptWidget.value = promptName;
            textWidget.value = "";
            
            // Mark as unsaved/new prompt
            node.isNewUnsavedPrompt = true;
            node.newPromptCategory = targetCategory;
            node.newPromptName = promptName;
            
            // Update previous values so cancel/revert works correctly
            node._previousCategory = targetCategory;
            node._previousPrompt = promptName;
            
            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
        }
    });

    // More dropdown button
    const moreBtn = createDropdownButton("More ▼", [
        {
            label: "New Category",
            action: async () => {
                const categoryName = await showTextPrompt("New Category", "Enter new category name:");
                
                if (categoryName && categoryName.trim()) {
                    let existingCategoryName = null;
                    if (node.prompts) {
                        const existingCategories = Object.keys(node.prompts);
                        existingCategoryName = existingCategories.find(cat => cat.toLowerCase() === categoryName.trim().toLowerCase());
                    }

                    if (existingCategoryName) {
                        await showInfo(
                            "Category Exists",
                            `Category already exists as "${existingCategoryName}".`
                        );
                        return;
                    }

                    await createCategory(node, categoryName.trim());
                }
            }
        },
        {
            label: "Delete Category",
            action: async () => {
                if (await showConfirm("Delete Category", `Are you sure you want to delete category "${categoryWidget.value}" and all its prompts?`)) {
                    await deleteCategory(node, categoryWidget.value);
                }
            }
        },
        {
            label: "Delete Prompt",
            action: async () => {
                if (await showConfirm("Delete Prompt", `Are you sure you want to delete prompt "${promptWidget.value}"?`)) {
                    await deletePrompt(node, categoryWidget.value, promptWidget.value);
                }
            }
        },
        { divider: true },
        {
            label: "Export JSON",
            action: async () => {
                await exportPromptsJSON(node);
            }
        },
        {
            label: "Import JSON",
            action: async () => {
                await importPromptsJSON(node);
            }
        }
    ]);

    buttonContainer.appendChild(savePromptBtn);
    buttonContainer.appendChild(newPromptBtn);
    buttonContainer.appendChild(moreBtn);

    // Add button bar to node
    const htmlWidget = node.addDOMWidget("buttons", "div", buttonContainer);
    htmlWidget.computeSize = function(width) {
        return [width, 36];
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
    categoryWidget.callback = async function(value) {
        if (originalCallback) {
            originalCallback.apply(this, arguments);
        }

        // Reload prompts from server to get latest changes from other tabs
        await loadPrompts(node);

        const category = value;
        if (node.prompts && node.prompts[category]) {
            const promptNames = Object.keys(node.prompts[category]).sort((a, b) => a.localeCompare(b));
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
    promptWidget.callback = async function(value) {
        if (originalPromptCallback) {
            originalPromptCallback.apply(this, arguments);
        }

        // Reload prompts from server to get latest changes from other tabs
        await loadPrompts(node);

        const category = categoryWidget.value;
        
        // Update prompt dropdown options in case prompts were deleted/added in another tab
        if (node.prompts && node.prompts[category]) {
            const promptNames = Object.keys(node.prompts[category]).sort((a, b) => a.localeCompare(b));
            promptWidget.options.values = promptNames.length > 0 ? promptNames : [""];
            
            // Check if selected prompt still exists after reload
            if (!promptNames.includes(value)) {
                // Prompt was deleted in another tab, switch to first available
                value = promptNames.length > 0 ? promptNames[0] : "";
                promptWidget.value = value;
            }
        }

        // Only update text if not using external
        const useExternal = useExternalWidget ? useExternalWidget.value : false;
        if (!useExternal) {
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
    button.style.borderRadius = "6px";
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

function createDropdownButton(text, items) {
    const container = document.createElement("div");
    container.style.position = "relative";
    container.style.flex = "1";
    container.style.minWidth = "80px";

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
    button.style.fontWeight = "normal";
    button.style.whiteSpace = "nowrap";
    button.style.height = "28px";
    button.style.display = "flex";
    button.style.alignItems = "center";
    button.style.justifyContent = "center";

    // Create dropdown and append to body to escape stacking context issues
    const dropdown = document.createElement("div");
    dropdown.style.cssText = `
        position: fixed;
        background: #2a2a2a;
        border: 1px solid #555;
        border-radius: 6px;
        z-index: 999999;
        display: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        min-width: 140px;
    `;
    document.body.appendChild(dropdown);

    items.forEach(item => {
        if (item.divider) {
            const divider = document.createElement("div");
            divider.style.cssText = "height: 1px; background: #444; margin: 4px 0;";
            dropdown.appendChild(divider);
        } else {
            const menuItem = document.createElement("div");
            menuItem.textContent = item.label;
            menuItem.style.cssText = `
                padding: 8px 12px;
                cursor: pointer;
                font-size: 11px;
                color: #fff;
                white-space: nowrap;
            `;
            menuItem.addEventListener("mouseenter", () => {
                menuItem.style.backgroundColor = "#4a4a4a";
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
            // Position dropdown below the button
            const rect = button.getBoundingClientRect();
            dropdown.style.left = rect.left + "px";
            dropdown.style.top = (rect.bottom + 2) + "px";
            dropdown.style.display = "block";
        }
    });

    // Close dropdown when clicking elsewhere
    document.addEventListener("click", (e) => {
        if (!dropdown.contains(e.target) && e.target !== button) {
            dropdown.style.display = "none";
        }
    });

    container.appendChild(button);

    return container;
}

function showPromptWithCategory(title, message, defaultName, categories, defaultCategory) {
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
            min-width: 320px;
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

        // Build category options
        const categoryOptions = categories.map(cat => 
            `<option value="${cat}" ${cat === defaultCategory ? 'selected' : ''}>${cat}</option>`
        ).join('');

        dialog.innerHTML = `
            <div style="margin-bottom: 15px; font-size: 16px; font-weight: bold; color: #fff;">${title}</div>
            <div style="margin-bottom: 6px; color: #aaa; font-size: 12px;">Category:</div>
            <select style="width: 100%; padding: 8px; margin-bottom: 12px; background: #333; border: 1px solid #555; color: #fff; border-radius: 4px; font-size: 14px;">
                ${categoryOptions}
            </select>
            <div style="margin-bottom: 6px; color: #aaa; font-size: 12px;">${message}</div>
            <input type="text" value="${defaultName}" style="width: 100%; padding: 8px; margin-bottom: 15px; background: #333; border: 1px solid #555; color: #fff; border-radius: 4px; font-size: 14px; box-sizing: border-box;" />
            <div style="display: flex; gap: 10px; justify-content: flex-end;">
                <button class="cancel-btn" style="padding: 8px 16px; background: #555; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
                <button class="ok-btn" style="padding: 8px 16px; background: #0a0; color: #fff; border: none; border-radius: 4px; cursor: pointer;">OK</button>
            </div>
        `;

        const selectEl = dialog.querySelector("select");
        const input = dialog.querySelector("input");
        const okBtn = dialog.querySelector(".ok-btn");
        const cancelBtn = dialog.querySelector(".cancel-btn");

        const cleanup = () => {
            document.body.removeChild(overlay);
            document.body.removeChild(dialog);
        };

        const handleOk = () => {
            resolve({ name: input.value, category: selectEl.value });
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

async function exportPromptsJSON(node) {
    try {
        const response = await fetch("/prompt-manager/get-prompts");
        const data = await response.json();
        
        const jsonStr = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonStr], { type: "application/json" });
        
        // Try File System Access API first (works on localhost and https)
        if (window.showSaveFilePicker) {
            try {
                const handle = await window.showSaveFilePicker({
                    suggestedName: "prompt_manager_data.json",
                    types: [{
                        description: "JSON Files",
                        accept: { "application/json": [".json"] }
                    }]
                });
                const writable = await handle.createWritable();
                await writable.write(blob);
                await writable.close();
                await showInfo("Export Complete", "Prompts exported successfully!");
                return;
            } catch (err) {
                // User cancelled the dialog
                if (err.name === "AbortError") {
                    return;
                }
                // Fall back to download method if API fails
                console.log("[PromptManager] Save picker failed, falling back to download:", err);
            }
        }
        
        // Fallback: Prompt user for filename, then download
        const filename = await showTextPrompt(
            "Export Prompts",
            "Enter filename for export:",
            "prompt_manager_data.json"
        );
        
        if (!filename) {
            return; // User cancelled
        }
        
        const finalFilename = filename.endsWith(".json") ? filename : filename + ".json";
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = finalFilename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        await showInfo("Export Complete", "Prompts exported successfully!");
    } catch (error) {
        console.error("[PromptManager] Error exporting prompts:", error);
        await showInfo("Error", "Failed to export prompts");
    }
}

async function importPromptsJSON(node) {
    return new Promise((resolve) => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".json";
        
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) {
                resolve(false);
                return;
            }
            
            try {
                const text = await file.text();
                const importedData = JSON.parse(text);
                
                // Validate structure
                if (typeof importedData !== 'object' || Array.isArray(importedData)) {
                    await showInfo("Error", "Invalid JSON structure. Expected an object with categories.");
                    resolve(false);
                    return;
                }
                
                // Ask user how to handle import
                const importMode = await showImportOptions();
                
                if (importMode === null) {
                    // User cancelled
                    resolve(false);
                    return;
                }
                
                // Send to backend
                const response = await fetch("/prompt-manager/import-prompts", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        data: importedData,
                        mode: importMode
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    node.prompts = result.prompts;
                    updateDropdowns(node);
                    
                    // Refresh the current prompt text
                    const categoryWidget = node.widgets.find(w => w.name === "category");
                    const promptWidget = node.widgets.find(w => w.name === "name");
                    const textWidget = node.widgets.find(w => w.name === "text");
                    
                    if (categoryWidget && promptWidget && textWidget) {
                        const category = categoryWidget.value;
                        const promptName = promptWidget.value;
                        if (node.prompts[category] && node.prompts[category][promptName]) {
                            textWidget.value = node.prompts[category][promptName].prompt || "";
                        }
                    }
                    
                    node.serialize_widgets = true;
                    app.graph.setDirtyCanvas(true, true);
                    
                    await showInfo("Import Complete", `Successfully imported prompts!`);
                    resolve(true);
                } else {
                    await showInfo("Error", result.error || "Failed to import prompts");
                    resolve(false);
                }
            } catch (error) {
                console.error("[PromptManager] Error importing prompts:", error);
                await showInfo("Error", "Failed to parse JSON file");
                resolve(false);
            }
        };
        
        input.click();
    });
}

function showImportOptions() {
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
            min-width: 320px;
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
            <div style="margin-bottom: 15px; font-size: 16px; font-weight: bold; color: #fff;">Import Prompts</div>
            <div style="margin-bottom: 20px; color: #ccc;">How do you want to import?<br><br>
                <strong>Merge:</strong> Keep existing prompts, add/update from import<br>
                <strong>Replace:</strong> Delete all existing, use import only
            </div>
            <div style="display: flex; gap: 10px; justify-content: flex-end;">
                <button class="cancel-btn" style="padding: 8px 16px; background: #555; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
                <button class="replace-btn" style="padding: 8px 16px; background: #c00; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Replace</button>
                <button class="merge-btn" style="padding: 8px 16px; background: #0a0; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Merge</button>
            </div>
        `;

        const mergeBtn = dialog.querySelector(".merge-btn");
        const replaceBtn = dialog.querySelector(".replace-btn");
        const cancelBtn = dialog.querySelector(".cancel-btn");

        const cleanup = () => {
            document.body.removeChild(overlay);
            document.body.removeChild(dialog);
        };

        mergeBtn.onclick = () => {
            resolve("merge");
            cleanup();
        };

        replaceBtn.onclick = () => {
            resolve("replace");
            cleanup();
        };

        cancelBtn.onclick = () => {
            resolve(null);
            cleanup();
        };

        overlay.onclick = () => {
            resolve(null);
            cleanup();
        };

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        mergeBtn.focus();
    });
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

    // Update category dropdown (sorted alphabetically)
    const categories = Object.keys(node.prompts).sort((a, b) => a.localeCompare(b));
    categoryWidget.options.values = categories;

    if (!node.prompts[categoryWidget.value] && categories.length > 0) {
        categoryWidget.value = categories[0];
    }

    // Update prompt dropdown for current category (sorted alphabetically)
    const currentCategory = categoryWidget.value;
    if (node.prompts[currentCategory]) {
        const promptNames = Object.keys(node.prompts[currentCategory]).sort((a, b) => a.localeCompare(b));

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
            await showInfo("Error", data.error || "Failed to save prompt");
        }
    } catch (error) {
        console.error("[PromptManager] Error saving prompt:", error);
        await showInfo("Error", "Failed to save prompt");
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
            await showInfo("Error", data.error || "Failed to delete category");
        }
    } catch (error) {
        console.error("[PromptManager] Error deleting category:", error);
        await showInfo("Error", "Failed to delete category");
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
            await showInfo("Error", data.error || "Failed to delete prompt");
        }
    } catch (error) {
        console.error("[PromptManager] Error deleting prompt:", error);
        await showInfo("Error", "Failed to delete prompt");
    }
}

console.log("[PromptManager] Extension loaded");
