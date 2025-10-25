import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "PromptManager",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PromptManager") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                
                const node = this;
                node.prompts = {};
                
                // Change widget labels
                const promptTextWidget = this.widgets.find(w => w.name === "prompt_text");
                if (promptTextWidget) {
                    promptTextWidget.label = "prompt";
                }
                
                const promptNameWidget = this.widgets.find(w => w.name === "prompt_name");
                if (promptNameWidget) {
                    promptNameWidget.label = "prompt name";
                }
                
                // Load prompts on creation
                loadPrompts(node).then(() => {
                    addButtonBar(node);
                    setupCategoryChangeHandler(node);
                    filterPromptDropdown(node);
                    setupInputConnectionHandler(node);
                    
                    setTimeout(() => {
                        const computedSize = node.computeSize();
                        const minHeight = computedSize[1] + 20;
                        
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
                            setupInputConnectionHandler(node);
                        }
                    });
                }, 100);
                
                return result;
            };
            
            // Enforce minimum node width
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                size[0] = Math.max(400, size[0]);
                return onResize ? onResize.apply(this, arguments) : size;
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
    // Filter prompt_name dropdown to only show prompts from current category
    const categoryWidget = node.widgets.find(w => w.name === "category");
    const promptWidget = node.widgets.find(w => w.name === "prompt_name");
    
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
    
    const textWidget = node.widgets.find(w => w.name === "prompt_text");
    const categoryWidget = node.widgets.find(w => w.name === "category");
    const promptWidget = node.widgets.find(w => w.name === "prompt_name");
    
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
            await createCategory(node, categoryName.trim(), textWidget);
        }
    });
    
    const savePromptBtn = createButton("Save Prompt", async () => {
        const promptName = await showTextPrompt("Save Prompt", "Enter prompt name:", promptWidget.value || "New Prompt");
        if (promptName && promptName.trim()) {
            const promptText = textWidget.value;
            
            if (promptText && promptText.trim()) {
                const currentCategory = categoryWidget.value;
                const promptExists = node.prompts[currentCategory] && node.prompts[currentCategory][promptName.trim()];
                
                // Ask for confirmation before overwriting existing prompt
                if (promptExists) {
                    const confirmed = await showConfirm(
                        "Overwrite Prompt", 
                        `Prompt "${promptName.trim()}" already exists in category "${currentCategory}". Do you want to overwrite it?`,
                        "Overwrite",
                        "#f80"
                    );
                    
                    if (!confirmed) {
                        return;
                    }
                }
                
                await savePrompt(node, categoryWidget.value, promptName.trim(), promptText.trim(), textWidget);
            } else {
                alert("Prompt text cannot be empty");
            }
        }
    });
    
    const deleteCategoryBtn = createButton("Del Category", async () => {
        if (await showConfirm("Delete Category", `Are you sure you want to delete category "${categoryWidget.value}"?`)) {
            await deleteCategory(node, categoryWidget.value, textWidget);
        }
    });
    
    const deletePromptBtn = createButton("Del Prompt", async () => {
        if (await showConfirm("Delete Prompt", `Are you sure you want to delete prompt "${promptWidget.value}"?`)) {
            await deletePrompt(node, categoryWidget.value, promptWidget.value, textWidget);
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
    const promptWidget = node.widgets.find(w => w.name === "prompt_name");
    const textWidget = node.widgets.find(w => w.name === "prompt_text");
    
    if (!categoryWidget || !promptWidget || !textWidget) return;
    
    // Check if prompt_text input is connected from another node
    const isTextInputConnected = () => {
        const promptTextInput = node.inputs?.find(inp => inp.name === "prompt_text");
        return promptTextInput && promptTextInput.link != null;
    };
    
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
                // Only update text widget if not connected to another node
                if (!isTextInputConnected()) {
                    textWidget.value = node.prompts[category][promptNames[0]] || "";
                }
            } else {
                promptWidget.value = "";
                if (!isTextInputConnected()) {
                    textWidget.value = "";
                }
            }
        }
    };
    
    const originalPromptCallback = promptWidget.callback;
    
    // Update text widget when prompt selection changes
    promptWidget.callback = function(value) {
        if (originalPromptCallback) {
            originalPromptCallback.apply(this, arguments);
        }
        
        // Only update text if not connected to another node
        if (!isTextInputConnected()) {
            const category = categoryWidget.value;
            if (node.prompts && node.prompts[category] && node.prompts[category][value]) {
                textWidget.value = node.prompts[category][value];
            }
        }
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

function setupInputConnectionHandler(node) {
    // Handle displaying value from connected input nodes
    const textWidget = node.widgets?.find(w => w.name === "prompt_text");
    if (!textWidget) return;
    
    const originalGetInputOrProperty = node.getInputOrProperty;
    if (originalGetInputOrProperty) {
        node.getInputOrProperty = function(name) {
            const value = originalGetInputOrProperty.apply(this, arguments);
            
            if (name === "prompt_text" && value !== undefined) {
                textWidget.value = value;
            }
            
            return value;
        };
    }
    
    const graphCanvas = app.graph;
    if (graphCanvas) {
        const originalBeforeChange = graphCanvas.beforeChange;
        graphCanvas.beforeChange = function() {
            if (originalBeforeChange) {
                originalBeforeChange.apply(this, arguments);
            }
            
            const promptTextInput = node.inputs?.find(inp => inp.name === "prompt_text");
            if (promptTextInput && promptTextInput.link != null) {
                const link = graphCanvas.links[promptTextInput.link];
                if (link) {
                    const originNode = graphCanvas.getNodeById(link.origin_id);
                    if (originNode) {
                        const outputValue = originNode.getOutputData?.(link.origin_slot);
                        if (outputValue !== undefined) {
                            textWidget.value = outputValue;
                        }
                    }
                }
            }
        };
    }
    
    const api = app.api;
    if (api) {
        const originalQueuePrompt = api.queuePrompt;
        api.queuePrompt = async function(number, workflow) {
            const promptTextInput = node.inputs?.find(inp => inp.name === "prompt_text");
            if (promptTextInput && promptTextInput.link != null) {
                const graph = app.graph;
                const link = graph.links[promptTextInput.link];
                if (link) {
                    const originNode = graph.getNodeById(link.origin_id);
                    if (originNode && originNode.widgets) {
                        const outputWidget = originNode.widgets.find(w => w.name === link.origin_slot || w.name === "text" || w.name === "STRING");
                        if (outputWidget) {
                            textWidget.value = outputWidget.value;
                        }
                    }
                }
            }
            
            return originalQueuePrompt.apply(this, arguments);
        };
    }
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
        
        okBtn.onclick = () => {
            resolve(input.value);
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
        
        input.onkeydown = (e) => {
            if (e.key === "Enter") {
                resolve(input.value);
                cleanup();
            } else if (e.key === "Escape") {
                resolve(null);
                cleanup();
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

function updateDropdowns(node) {
    const categoryWidget = node.widgets.find(w => w.name === "category");
    const promptWidget = node.widgets.find(w => w.name === "prompt_name");
    const textWidget = node.widgets.find(w => w.name === "prompt_text");
    
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
            textWidget.value = node.prompts[currentCategory][promptWidget.value];
        } else {
            textWidget.value = "";
        }
    } else {
        promptWidget.options.values = [""];
        promptWidget.value = "";
        textWidget.value = "";
    }
}

async function createCategory(node, categoryName, textWidget) {
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
            const promptWidget = node.widgets.find(w => w.name === "prompt_name");
            const textWidget = node.widgets.find(w => w.name === "prompt_text");
            
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
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error("[PromptManager] Error creating category:", error);
        alert("Error creating category");
    }
}

async function savePrompt(node, category, promptName, promptText, textWidget) {
    try {
        const response = await fetch("/prompt-manager/save-prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                category: category,
                prompt_name: promptName,
                prompt_text: promptText
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            node.prompts = data.prompts;
            updateDropdowns(node);
            
            // Select the newly saved prompt
            const promptWidget = node.widgets.find(w => w.name === "prompt_name");
            if (promptWidget) {
                promptWidget.value = promptName;
                updateDropdowns(node);
            }
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error("[PromptManager] Error saving prompt:", error);
        alert("Error saving prompt");
    }
}

async function deleteCategory(node, category, textWidget) {
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
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error("[PromptManager] Error deleting category:", error);
        alert("Error deleting category");
    }
}

async function deletePrompt(node, category, promptName, textWidget) {
    try {
        const response = await fetch("/prompt-manager/delete-prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                category: category,
                prompt_name: promptName
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            node.prompts = data.prompts;
            updateDropdowns(node);
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error("[PromptManager] Error deleting prompt:", error);
        alert("Error deleting prompt");
    }
}

console.log("[PromptManager] Extension loaded");