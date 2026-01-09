import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * PromptManagerAdvanced Extension for ComfyUI
 * Extends prompt management with LoRA stack support and toggleable LoRA displays
 */

// ========================
// LoRA Manager Integration
// ========================

/**
 * Lazy-loaded PreviewTooltip from LoRA Manager (if installed)
 * This provides preview images on hover for LoRA tags
 */
let loraManagerPreviewTooltip = null;
let loraManagerCheckDone = false;
let loraManagerAvailable = false;

/**
 * Attempt to initialize the LoRA Manager preview tooltip integration
 * Only tries once, then caches the result
 */
async function getLoraManagerPreviewTooltip() {
    if (loraManagerCheckDone) {
        return loraManagerPreviewTooltip;
    }
    
    loraManagerCheckDone = true;
    
    try {
        // Try to dynamically import the PreviewTooltip from LoRA Manager
        // ComfyUI serves extension files from /extensions/<folder_name>/
        const previewModule = await import("/extensions/ComfyUI-Lora-Manager/preview_tooltip.js");
        
        if (previewModule && previewModule.PreviewTooltip) {
            loraManagerPreviewTooltip = new previewModule.PreviewTooltip({
                modelType: "loras"
            });
            loraManagerAvailable = true;
            console.log("[PromptManagerAdvanced] LoRA Manager preview integration enabled");
        }
    } catch (error) {
        // LoRA Manager not installed or preview_tooltip.js not found - this is expected
        console.log("[PromptManagerAdvanced] LoRA Manager preview not available:", error.message);
        loraManagerAvailable = false;
    }
    
    return loraManagerPreviewTooltip;
}

// Initialize on load (non-blocking)
getLoraManagerPreviewTooltip();

app.registerExtension({
    name: "PromptManagerAdvanced",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PromptManagerAdvanced") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                const node = this;
                node.prompts = {};
                node.currentLorasA = [];
                node.currentLorasB = [];
                node.savedLorasA = [];
                node.savedLorasB = [];
                
                // Set initial size - taller to accommodate lora displays
                this.setSize([450, 600]);

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

                // Listen for updates from backend
                api.addEventListener("prompt-manager-advanced-update", (event) => {
                    if (String(event.detail.node_id) === String(this.id)) {
                        const newLorasA = event.detail.loras_a || [];
                        const newLorasB = event.detail.loras_b || [];
                        
                        // Only update lora displays if the data actually changed
                        const lorasAChanged = JSON.stringify(newLorasA) !== JSON.stringify(this.currentLorasA);
                        const lorasBChanged = JSON.stringify(newLorasB) !== JSON.stringify(this.currentLorasB);
                        
                        if (lorasAChanged || lorasBChanged) {
                            // Update current loras from connected inputs
                            this.currentLorasA = newLorasA;
                            this.currentLorasB = newLorasB;
                            
                            // Update the lora display widgets
                            updateLoraDisplays(this);
                        }

                        // Handle use_external toggle state for text widget
                        const promptTextWidget = this.widgets.find(w => w.name === "text");
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
                // to ensure proper positioning within the node bounds
                addButtonBar(node);
                addLoraDisplays(node);
                setupCategoryChangeHandler(node);
                setupUseExternalToggleHandler(node);

                // Load prompts asynchronously (data only, not widgets)
                loadPrompts(node).then(() => {
                    filterPromptDropdown(node);

                    // Ensure height is sufficient after data is loaded
                    setTimeout(() => {
                        const computedSize = node.computeSize();
                        const minHeight = Math.max(600, computedSize[1] + 20);
                        
                        if (node.size[1] < minHeight) {
                            node.setSize([Math.max(450, node.size[0]), minHeight]);
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

                // Restore saved lora toggle states if they exist
                if (info.widgets_values) {
                    // Find the lora toggle widget indices and restore their values
                    const lorasAIndex = node.widgets?.findIndex(w => w.name === "loras_a_toggle");
                    const lorasBIndex = node.widgets?.findIndex(w => w.name === "loras_b_toggle");
                    
                    if (lorasAIndex >= 0 && info.widgets_values[lorasAIndex]) {
                        try {
                            node.savedLorasA = JSON.parse(info.widgets_values[lorasAIndex]);
                        } catch (e) {
                            node.savedLorasA = [];
                        }
                    }
                    if (lorasBIndex >= 0 && info.widgets_values[lorasBIndex]) {
                        try {
                            node.savedLorasB = JSON.parse(info.widgets_values[lorasBIndex]);
                        } catch (e) {
                            node.savedLorasB = [];
                        }
                    }
                }

                // IMPORTANT: Reattach DOM widgets SYNCHRONOUSLY during configure
                // to ensure proper positioning within the node bounds
                if (!node.buttonBarAttached) {
                    addButtonBar(node);
                    setupCategoryChangeHandler(node);
                }
                if (!node.loraDisplaysAttached) {
                    addLoraDisplays(node);
                }
                setupUseExternalToggleHandler(node);

                // Load prompts data asynchronously (data only, widgets already added)
                loadPrompts(node).then(() => {
                    filterPromptDropdown(node);
                    updateLoraDisplays(node);
                    app.graph.setDirtyCanvas(true, true);
                });

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

// ========================
// Data Loading Functions
// ========================

async function loadPrompts(node) {
    try {
        const response = await fetch("/prompt-manager-advanced/get-prompts");
        const data = await response.json();
        node.prompts = data;
        return data;
    } catch (error) {
        console.error("[PromptManagerAdvanced] Error loading prompts:", error);
        return {};
    }
}

/**
 * Query connected LoRA stacker nodes to get current LoRA configurations
 * This allows saving without executing the workflow first
 */
function getLorasFromConnectedNodes(node, inputName) {
    const loras = [];
    
    // Find the input slot
    const inputIndex = node.inputs?.findIndex(input => input.name === inputName);
    if (inputIndex === -1 || inputIndex === undefined) {
        return loras;
    }
    
    const input = node.inputs[inputIndex];
    if (!input || !input.link) {
        return loras;
    }
    
    // Get the link and find the source node
    const link = app.graph.links[input.link];
    if (!link) {
        return loras;
    }
    
    const sourceNode = app.graph.getNodeById(link.origin_id);
    if (!sourceNode) {
        return loras;
    }
    
    // Try to extract LoRA data based on node type
    return extractLorasFromNode(sourceNode);
}

/**
 * Extract LoRA data from various node types
 */
function extractLorasFromNode(sourceNode) {
    const loras = [];
    
    // Check for LoRA Manager's loras widget (most reliable source)
    const lorasWidget = sourceNode.widgets?.find(w => w.name === "loras");
    if (lorasWidget && lorasWidget.value) {
        const loraValues = Array.isArray(lorasWidget.value) ? lorasWidget.value : [];
        for (const lora of loraValues) {
            // Only extract ACTIVE loras - inactive ones are not in the LORA_STACK output
            if (lora && lora.name && lora.active !== false) {
                loras.push({
                    name: lora.name,
                    path: lora.path || "",
                    model_strength: lora.strength ?? lora.model_strength ?? 1.0,
                    clip_strength: lora.clipStrength ?? lora.clip_strength ?? lora.strength ?? 1.0,
                    active: true,
                    strength: lora.strength ?? lora.model_strength ?? 1.0
                });
            }
        }
        return loras;
    }
    
    // Check for text widget with LoRA syntax (e.g., <lora:name:strength>)
    const textWidget = sourceNode.widgets?.find(w => w.name === "text");
    if (textWidget && textWidget.value) {
        const loraMatches = textWidget.value.matchAll(/<lora:([^:>]+):([^:>]+)(?::([^>]+))?>/g);
        for (const match of loraMatches) {
            const name = match[1].trim();
            const modelStrength = parseFloat(match[2]) || 1.0;
            const clipStrength = match[3] ? parseFloat(match[3]) : modelStrength;
            
            loras.push({
                name: name,
                path: "",
                model_strength: modelStrength,
                clip_strength: clipStrength,
                active: true,
                strength: modelStrength
            });
        }
        return loras;
    }
    
    // Check for lora_name and strength widgets (individual LoRA loader nodes)
    const loraNameWidget = sourceNode.widgets?.find(w => w.name === "lora_name");
    const strengthWidget = sourceNode.widgets?.find(w => 
        w.name === "strength" || w.name === "strength_model" || w.name === "model_strength"
    );
    const clipStrengthWidget = sourceNode.widgets?.find(w => 
        w.name === "strength_clip" || w.name === "clip_strength"
    );
    
    if (loraNameWidget && loraNameWidget.value) {
        const modelStrength = strengthWidget ? parseFloat(strengthWidget.value) || 1.0 : 1.0;
        const clipStrength = clipStrengthWidget ? parseFloat(clipStrengthWidget.value) || modelStrength : modelStrength;
        
        loras.push({
            name: loraNameWidget.value.replace(/\.[^/.]+$/, ""), // Remove file extension
            path: loraNameWidget.value,
            model_strength: modelStrength,
            clip_strength: clipStrength,
            active: true,
            strength: modelStrength
        });
    }
    
    return loras;
}

/**
 * Recursively collect LoRAs from a node and any connected upstream LoRA stackers
 */
function collectAllLorasFromChain(node, inputName, visited = new Set()) {
    const allLoras = [];
    
    // Find the input slot
    const inputIndex = node.inputs?.findIndex(input => input.name === inputName);
    if (inputIndex === -1 || inputIndex === undefined) {
        return allLoras;
    }
    
    const input = node.inputs[inputIndex];
    if (!input || !input.link) {
        return allLoras;
    }
    
    // Get the link and find the source node
    const link = app.graph.links[input.link];
    if (!link) {
        return allLoras;
    }
    
    const sourceNode = app.graph.getNodeById(link.origin_id);
    if (!sourceNode || visited.has(sourceNode.id)) {
        return allLoras;
    }
    
    visited.add(sourceNode.id);
    
    // Extract LoRAs from this node
    const nodeLoras = extractLorasFromNode(sourceNode);
    allLoras.push(...nodeLoras);
    
    // Check if this node has a lora_stack input (for chained stackers)
    const loraStackInput = sourceNode.inputs?.find(inp => 
        inp.name === "lora_stack" || inp.name === "loras"
    );
    if (loraStackInput && loraStackInput.link) {
        const upstreamLoras = collectAllLorasFromChain(sourceNode, loraStackInput.name, visited);
        // Prepend upstream loras (they come first in the chain)
        allLoras.unshift(...upstreamLoras);
    }
    
    return allLoras;
}

function filterPromptDropdown(node) {
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

// ========================
// LoRA Display Functions
// ========================

function addLoraDisplays(node) {
    if (node.loraDisplaysAttached) {
        return;
    }

    // Create LoRA A display section
    const loraAContainer = createLoraDisplayContainer("LoRAs Stack A", "a", node);
    const loraAWidget = node.addDOMWidget("loras_a_display", "div", loraAContainer);
    loraAWidget.computeSize = function(width) {
        // Check if override_lora is enabled
        const overrideWidget = node.widgets?.find(w => w.name === "override_lora");
        const overrideLora = overrideWidget?.value === true;
        
        // Count loras based on override mode
        let tagCount;
        if (overrideLora) {
            // Override ON: only saved loras
            tagCount = (node.savedLorasA || []).length;
        } else {
            // Override OFF: unique loras from merged list
            const seen = new Set();
            (node.currentLorasA || []).forEach(l => seen.add(l.name));
            (node.savedLorasA || []).forEach(l => seen.add(l.name));
            tagCount = seen.size;
        }
        
        // Use node's actual width, not the passed width which may be stale
        // Subtract ~32px for container padding (8px) + widget margins (~24px)
        const actualWidth = node.size?.[0] || width || 400;
        const tagsPerRow = Math.max(1, Math.floor((actualWidth - 32) / 204));
        const rows = Math.max(1, Math.ceil(tagCount / tagsPerRow));
        const height = 58 + rows * 28;
        return [width, height];
    };
    node.loraAWidget = loraAWidget;
    node.loraAContainer = loraAContainer;

    // Create LoRA B display section
    const loraBContainer = createLoraDisplayContainer("LoRAs Stack B", "b", node);
    const loraBWidget = node.addDOMWidget("loras_b_display", "div", loraBContainer);
    loraBWidget.computeSize = function(width) {
        // Check if override_lora is enabled
        const overrideWidget = node.widgets?.find(w => w.name === "override_lora");
        const overrideLora = overrideWidget?.value === true;
        
        // Count loras based on override mode
        let tagCount;
        if (overrideLora) {
            // Override ON: only saved loras
            tagCount = (node.savedLorasB || []).length;
        } else {
            // Override OFF: unique loras from merged list
            const seen = new Set();
            (node.currentLorasB || []).forEach(l => seen.add(l.name));
            (node.savedLorasB || []).forEach(l => seen.add(l.name));
            tagCount = seen.size;
        }
        
        // Use node's actual width, not the passed width which may be stale
        // Subtract ~32px for container padding (8px) + widget margins (~24px)
        const actualWidth = node.size?.[0] || width || 400;
        const tagsPerRow = Math.max(1, Math.floor((actualWidth - 32) / 204));
        const rows = Math.max(1, Math.ceil(tagCount / tagsPerRow));
        const height = 58 + rows * 28;
        return [width, height];
    };
    node.loraBWidget = loraBWidget;
    node.loraBContainer = loraBContainer;

    // Add hidden widgets to store toggle states for serialization
    const lorasAToggleWidget = node.addWidget('text', 'loras_a_toggle', '[]');
    lorasAToggleWidget.type = "converted-widget";
    lorasAToggleWidget.hidden = true;
    lorasAToggleWidget.computeSize = () => [0, -4];
    node.lorasAToggleWidget = lorasAToggleWidget;

    const lorasBToggleWidget = node.addWidget('text', 'loras_b_toggle', '[]');
    lorasBToggleWidget.type = "converted-widget";
    lorasBToggleWidget.hidden = true;
    lorasBToggleWidget.computeSize = () => [0, -4];
    node.lorasBToggleWidget = lorasBToggleWidget;

    node.loraDisplaysAttached = true;
}

function createLoraDisplayContainer(title, stackId, node) {
    const container = document.createElement("div");
    container.className = `lora-display-container lora-display-${stackId}`;
    Object.assign(container.style, {
        display: "flex",
        flexDirection: "column",
        gap: "4px",
        padding: "8px",
        backgroundColor: "rgba(40, 44, 52, 0.6)",
        borderRadius: "6px",
        width: "100%",
        boxSizing: "border-box",
        marginTop: "4px"
    });

    // Title bar with label
    const titleBar = document.createElement("div");
    Object.assign(titleBar.style, {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "4px",
        paddingBottom: "4px",
        borderBottom: "1px solid rgba(255,255,255,0.1)"
    });

    const titleLabel = document.createElement("span");
    titleLabel.textContent = title;
    Object.assign(titleLabel.style, {
        fontSize: "12px",
        fontWeight: "bold",
        color: "#aaa"
    });
    titleBar.appendChild(titleLabel);

    // Toggle All button
    const toggleAllBtn = document.createElement("button");
    toggleAllBtn.textContent = "Toggle All";
    Object.assign(toggleAllBtn.style, {
        fontSize: "10px",
        padding: "2px 8px",
        backgroundColor: "#333",
        color: "#ccc",
        border: "1px solid #555",
        borderRadius: "6px",
        cursor: "pointer"
    });
    toggleAllBtn.onclick = () => toggleAllLoras(node, stackId);
    titleBar.appendChild(toggleAllBtn);

    container.appendChild(titleBar);

    // Tags container
    const tagsContainer = document.createElement("div");
    tagsContainer.className = "lora-tags-container";
    Object.assign(tagsContainer.style, {
        display: "flex",
        flexWrap: "wrap",
        gap: "4px",
        minHeight: "30px"
    });
    container.appendChild(tagsContainer);

    return container;
}

function updateLoraDisplays(node) {
    if (!node.loraAContainer || !node.loraBContainer) return;

    // Check if override_lora is enabled
    const overrideWidget = node.widgets?.find(w => w.name === "override_lora");
    const overrideLora = overrideWidget?.value === true;

    let lorasA, lorasB;
    if (overrideLora) {
        // Override ON: Only show saved loras from the prompt (filter by saved names)
        const savedNamesA = new Set((node.savedLorasA || []).map(l => l.name));
        const savedNamesB = new Set((node.savedLorasB || []).map(l => l.name));
        
        lorasA = (node.savedLorasA || []).map(l => ({ ...l, source: 'saved' }));
        lorasB = (node.savedLorasB || []).map(l => ({ ...l, source: 'saved' }));
    } else {
        // Override OFF: Merge current (from input) and saved (from prompt) loras
        lorasA = mergeLoraLists(node.currentLorasA, node.savedLorasA);
        lorasB = mergeLoraLists(node.currentLorasB, node.savedLorasB);
    }

    // Update display A
    const tagsContainerA = node.loraAContainer.querySelector(".lora-tags-container");
    if (tagsContainerA) {
        renderLoraTags(tagsContainerA, lorasA, "a", node);
    }

    // Update display B
    const tagsContainerB = node.loraBContainer.querySelector(".lora-tags-container");
    if (tagsContainerB) {
        renderLoraTags(tagsContainerB, lorasB, "b", node);
    }

    // Update hidden widgets for serialization
    updateToggleWidgets(node);

    // Just redraw canvas - the computeSize functions already handle correct sizing
    // using node.size[0] for accurate width calculation
    app.graph.setDirtyCanvas(true, true);
}

function mergeLoraLists(currentLoras, savedLoras) {
    // Create a map of saved loras to preserve user toggle states
    const savedMap = new Map();
    (savedLoras || []).forEach(lora => {
        savedMap.set(lora.name, lora);
    });

    // Merge loras, preserving toggle states from saved when available
    const merged = [];
    const seen = new Set();

    // First add all current loras, but preserve toggle state from saved if exists
    (currentLoras || []).forEach(lora => {
        const savedLora = savedMap.get(lora.name);
        if (savedLora) {
            // Lora exists in both - mark as 'saved' since user has it in their preset
            // This ensures modifications are persisted to savedLoras, not currentLoras
            merged.push({ 
                ...lora, 
                active: savedLora.active,
                strength: savedLora.strength ?? savedLora.model_strength ?? lora.strength ?? 1.0,
                source: 'saved'  // Important: mark as saved so modifications persist correctly
            });
        } else {
            merged.push({ ...lora, source: 'current' });
        }
        seen.add(lora.name);
    });

    // Then add saved loras that aren't in current
    (savedLoras || []).forEach(lora => {
        if (!seen.has(lora.name)) {
            merged.push({ ...lora, source: 'saved' });
            seen.add(lora.name);
        }
    });

    // Sort alphabetically by name to maintain stable order when toggling
    merged.sort((a, b) => a.name.localeCompare(b.name));

    return merged;
}

function renderLoraTags(container, loras, stackId, node) {
    // Clear existing tags
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }

    if (!loras || loras.length === 0) {
        const emptyMessage = document.createElement("div");
        emptyMessage.textContent = "No LoRAs connected";
        Object.assign(emptyMessage.style, {
            color: "rgba(200, 200, 200, 0.5)",
            fontStyle: "italic",
            fontSize: "11px",
            padding: "8px",
            width: "100%",
            textAlign: "center"
        });
        container.appendChild(emptyMessage);
        return;
    }

    loras.forEach((lora, index) => {
        const tag = createLoraTag(lora, index, stackId, node);
        container.appendChild(tag);
    });
}

function createLoraTag(lora, index, stackId, node) {
    const tag = document.createElement("div");
    tag.className = "lora-tag";
    tag.dataset.loraName = lora.name;
    tag.dataset.loraIndex = index;
    tag.dataset.stackId = stackId;

    const isActive = lora.active !== false;
    const isAvailable = lora.available !== false;
    const strength = lora.strength ?? lora.model_strength ?? 1.0;

    // Determine colors based on active and available status
    let bgColor, textColor, borderColor;
    if (!isAvailable) {
        // Missing LoRA - red/orange warning colors
        bgColor = isActive ? "rgba(220, 53, 69, 0.9)" : "rgba(220, 53, 69, 0.4)";
        textColor = isActive ? "white" : "rgba(255, 200, 200, 0.8)";
        borderColor = "rgba(220, 53, 69, 0.9)";
    } else if (isActive) {
        // Available and active - blue
        bgColor = "rgba(66, 153, 225, 0.9)";
        textColor = "white";
        borderColor = "rgba(66, 153, 225, 0.9)";
    } else {
        // Available but inactive - gray
        bgColor = "rgba(45, 55, 72, 0.7)";
        textColor = "rgba(226, 232, 240, 0.6)";
        borderColor = "rgba(226, 232, 240, 0.2)";
    }

    Object.assign(tag.style, {
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        padding: "4px 10px",
        borderRadius: "6px",
        fontSize: "12px",
        cursor: "pointer",
        transition: "background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease",
        backgroundColor: bgColor,
        color: textColor,
        border: `1px solid ${borderColor}`,
        width: "200px",
        height: "24px",
        boxSizing: "border-box",
        userSelect: "none",
        flexShrink: "0"
    });

    // Warning icon for missing LoRAs
    if (!isAvailable) {
        const warningIcon = document.createElement("span");
        warningIcon.textContent = "⚠️";
        warningIcon.style.fontSize = "11px";
        warningIcon.style.marginRight = "-2px";
        tag.appendChild(warningIcon);
    }

    // LoRA name
    const nameSpan = document.createElement("span");
    nameSpan.textContent = lora.name;
    Object.assign(nameSpan.style, {
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        flexGrow: "1",
        textDecoration: !isAvailable ? "line-through" : "none",
        opacity: !isAvailable ? "0.8" : "1"
    });
    tag.appendChild(nameSpan);

    // Editable strength input
    const strengthInput = document.createElement("input");
    strengthInput.type = "text";
    strengthInput.className = "strength-input";
    strengthInput.value = strength.toFixed(2);
    Object.assign(strengthInput.style, {
        fontSize: "10px",
        fontWeight: "600",
        padding: "1px 4px",
        borderRadius: "999px",
        backgroundColor: "rgba(255,255,255,0.15)",
        color: "rgba(255,255,255,0.9)",
        width: "38px",
        textAlign: "center",
        border: "1px solid transparent",
        outline: "none",
        cursor: "text"
    });
    
    // Handle focus - select all text
    strengthInput.addEventListener("focus", (e) => {
        e.stopPropagation();
        strengthInput.select();
        strengthInput.style.backgroundColor = "rgba(255,255,255,0.25)";
        strengthInput.style.border = "1px solid rgba(66, 153, 225, 0.6)";
    });
    
    // Handle blur - apply value
    strengthInput.addEventListener("blur", () => {
        strengthInput.style.backgroundColor = "rgba(255,255,255,0.15)";
        strengthInput.style.border = "1px solid transparent";
        
        const newValue = parseFloat(strengthInput.value);
        if (!isNaN(newValue) && newValue >= 0) {
            setLoraStrength(node, stackId, index, newValue);
        } else {
            // Reset to current value if invalid
            strengthInput.value = strength.toFixed(2);
        }
    });
    
    // Handle Enter key
    strengthInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            strengthInput.blur();
        } else if (e.key === "Escape") {
            e.preventDefault();
            strengthInput.value = strength.toFixed(2);
            strengthInput.blur();
        }
        e.stopPropagation();
    });
    
    // Prevent click from toggling the tag
    strengthInput.addEventListener("click", (e) => {
        e.stopPropagation();
    });
    
    tag.appendChild(strengthInput);

    // Click handler to toggle active state (but not when clicking on input)
    tag.addEventListener("click", (e) => {
        if (e.target !== strengthInput) {
            e.stopPropagation();
            toggleLoraActive(node, stackId, index);
        }
    });

    // Hover effects and LoRA Manager preview integration
    let hoverTimeout = null;
    tag.addEventListener("mouseenter", (e) => {
        tag.style.transform = "translateY(-1px)";
        tag.style.boxShadow = "0 2px 4px rgba(0,0,0,0.2)";
        
        // Show LoRA Manager preview tooltip if available (with small delay)
        // Position it to the right/below to not cover the tag
        if (loraManagerAvailable && isAvailable) {
            const tagRect = tag.getBoundingClientRect();
            hoverTimeout = setTimeout(async () => {
                const tooltip = await getLoraManagerPreviewTooltip();
                if (tooltip) {
                    // Position tooltip to the right of the tag, or below if no space
                    const tooltipX = tagRect.right + 10;
                    const tooltipY = tagRect.top;
                    tooltip.show(lora.name, tooltipX, tooltipY);
                }
            }, 300); // 300ms delay before showing preview
        }
    });
    tag.addEventListener("mouseleave", () => {
        tag.style.transform = "translateY(0)";
        tag.style.boxShadow = "none";
        
        // Clear hover timeout and hide tooltip
        if (hoverTimeout) {
            clearTimeout(hoverTimeout);
            hoverTimeout = null;
        }
        if (loraManagerPreviewTooltip) {
            loraManagerPreviewTooltip.hide();
        }
    });

    // Add title for full name on hover
    const availabilityStatus = isAvailable ? "" : "\n⚠️ NOT FOUND - This LoRA is missing from your system";
    const previewHint = loraManagerAvailable && isAvailable ? "\nHover to preview" : "";
    tag.title = `${lora.name}\nStrength: ${strength.toFixed(2)}${availabilityStatus}${previewHint}\nClick to toggle on/off`;

    return tag;
}

function toggleLoraActive(node, stackId, index) {
    // Check if override_lora is enabled
    const overrideWidget = node.widgets?.find(w => w.name === "override_lora");
    const overrideLora = overrideWidget?.value === true;
    
    // Get the list that matches what's currently displayed
    let loraList;
    if (overrideLora) {
        // Override ON: only saved loras are displayed
        loraList = stackId === "a" ? [...(node.savedLorasA || [])] : [...(node.savedLorasB || [])];
    } else {
        // Override OFF: merged list is displayed
        loraList = stackId === "a" ? 
            mergeLoraLists(node.currentLorasA, node.savedLorasA) : 
            mergeLoraLists(node.currentLorasB, node.savedLorasB);
    }
    
    if (loraList[index]) {
        const lora = loraList[index];
        
        // Check if this is a connected lora (source: 'current') - can't toggle these off
        if (lora.source === 'current' && lora.active !== false) {
            // User tried to turn off a connected lora - flash light blue to indicate it can't be toggled
            const container = stackId === "a" ? node.loraAContainer : node.loraBContainer;
            const tags = container?.querySelectorAll('.lora-tag');
            if (tags && tags[index]) {
                const tag = tags[index];
                const originalBg = tag.style.backgroundColor;
                tag.style.transition = 'background-color 0.1s';
                tag.style.backgroundColor = 'rgba(100, 180, 255, 0.9)';
                setTimeout(() => {
                    tag.style.backgroundColor = originalBg;
                }, 200);
            }
            return;
        }
        
        loraList[index].active = !loraList[index].active;
        
        // Update the appropriate list
        if (overrideLora) {
            // Only update saved loras when override is on
            if (stackId === "a") {
                node.savedLorasA = loraList;
            } else {
                node.savedLorasB = loraList;
            }
        } else {
            if (stackId === "a") {
                updateLoraListFromMerged(node, "a", loraList);
            } else {
                updateLoraListFromMerged(node, "b", loraList);
            }
        }
        
        updateLoraDisplays(node);
        app.graph.setDirtyCanvas(true, true);
    }
}
/**
 * Set LoRA strength to a specific value
 */
function setLoraStrength(node, stackId, index, newStrength) {
    // Check if override_lora is enabled
    const overrideWidget = node.widgets?.find(w => w.name === "override_lora");
    const overrideLora = overrideWidget?.value === true;
    
    // Get the list that matches what's currently displayed
    let loraList;
    if (overrideLora) {
        // Override ON: only saved loras are displayed
        loraList = stackId === "a" ? [...(node.savedLorasA || [])] : [...(node.savedLorasB || [])];
    } else {
        // Override OFF: merged list is displayed
        loraList = stackId === "a" ? 
            mergeLoraLists(node.currentLorasA, node.savedLorasA) : 
            mergeLoraLists(node.currentLorasB, node.savedLorasB);
    }
    
    if (loraList[index]) {
        loraList[index].strength = newStrength;
        
        if (overrideLora) {
            // Only update saved loras when override is on
            if (stackId === "a") {
                node.savedLorasA = loraList;
            } else {
                node.savedLorasB = loraList;
            }
        } else {
            if (stackId === "a") {
                updateLoraListFromMerged(node, "a", loraList);
            } else {
                updateLoraListFromMerged(node, "b", loraList);
            }
        }
        
        updateLoraDisplays(node);
        app.graph.setDirtyCanvas(true, true);
    }
}

function toggleAllLoras(node, stackId) {
    // Check if override_lora is enabled
    const overrideWidget = node.widgets?.find(w => w.name === "override_lora");
    const overrideLora = overrideWidget?.value === true;
    
    // Get the list that matches what's currently displayed
    let loraList;
    if (overrideLora) {
        // Override ON: only saved loras are displayed
        loraList = stackId === "a" ? [...(node.savedLorasA || [])] : [...(node.savedLorasB || [])];
    } else {
        // Override OFF: merged list is displayed
        loraList = stackId === "a" ? 
            mergeLoraLists(node.currentLorasA, node.savedLorasA) : 
            mergeLoraLists(node.currentLorasB, node.savedLorasB);
    }
    
    // Determine if we should turn all on or all off
    const allActive = loraList.every(lora => lora.active !== false);
    const newState = !allActive;
    
    loraList.forEach(lora => {
        lora.active = newState;
    });
    
    if (overrideLora) {
        // Only update saved loras when override is on
        if (stackId === "a") {
            node.savedLorasA = loraList;
        } else {
            node.savedLorasB = loraList;
        }
    } else {
        if (stackId === "a") {
            updateLoraListFromMerged(node, "a", loraList);
        } else {
            updateLoraListFromMerged(node, "b", loraList);
        }
    }
    
    updateLoraDisplays(node);
    app.graph.setDirtyCanvas(true, true);
}

function updateLoraListFromMerged(node, stackId, mergedList) {
    // Separate back into current and saved lists based on source
    const currentLoras = [];
    const savedLoras = [];
    
    mergedList.forEach(lora => {
        if (lora.source === 'current') {
            currentLoras.push(lora);
        } else {
            savedLoras.push(lora);
        }
    });
    
    if (stackId === "a") {
        node.currentLorasA = currentLoras;
        node.savedLorasA = savedLoras;
    } else {
        node.currentLorasB = currentLoras;
        node.savedLorasB = savedLoras;
    }
}

function updateToggleWidgets(node) {
    // Format lora data for backend processing
    // IMPORTANT: Only store SAVED loras (preset loras), not connected input loras
    // Connected loras are managed by their source nodes and should not be persisted here
    const formatForBackend = (loraList) => {
        return loraList.map(lora => ({
            name: lora.name,
            active: lora.active !== false,
            strength: lora.strength ?? lora.model_strength ?? 1.0
        }));
    };
    
    // Only serialize saved loras (not current/connected ones)
    const lorasA = node.savedLorasA || [];
    const lorasB = node.savedLorasB || [];
    
    if (node.lorasAToggleWidget) {
        node.lorasAToggleWidget.value = JSON.stringify(formatForBackend(lorasA));
    }
    if (node.lorasBToggleWidget) {
        node.lorasBToggleWidget.value = JSON.stringify(formatForBackend(lorasB));
    }
}

// ========================
// Button Bar Functions
// ========================

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
    buttonContainer.style.padding = "2px 4px 8px 4px";
    buttonContainer.style.flexWrap = "wrap";
    buttonContainer.style.marginTop = "-10px";

    // Create action buttons
    const createCategoryBtn = createButton("New Category", async () => {
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
            const currentCategory = categoryWidget.value;
            
            // Check for existing prompt
            let existingPromptName = null;
            if (node.prompts[currentCategory]) {
                const existingNames = Object.keys(node.prompts[currentCategory]);
                existingPromptName = existingNames.find(name => name.toLowerCase() === promptName.trim().toLowerCase());
            }

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

            // Query connected LoRA stacker nodes to get current configurations
            // Note: extractLorasFromNode only returns ACTIVE loras
            const connectedLorasA = collectAllLorasFromChain(node, "lora_stack_a");
            const connectedLorasB = collectAllLorasFromChain(node, "lora_stack_b");
            
            // Check if override_lora is enabled
            const overrideWidget = node.widgets?.find(w => w.name === "override_lora");
            const overrideLora = overrideWidget?.value === true;
            
            // Get loras to save - same logic as display/execution
            let allLorasA, allLorasB;
            
            if (overrideLora) {
                // Override ON: only save the saved/preset loras (ignore connected inputs)
                allLorasA = [...(node.savedLorasA || [])].filter(l => l.active !== false);
                allLorasB = [...(node.savedLorasB || [])].filter(l => l.active !== false);
            } else {
                // Override OFF: merge connected + saved, then filter active ones
                // Use mergeLoraLists to get the same merged view as display
                const mergedA = mergeLoraLists(
                    connectedLorasA.map(l => ({ ...l, source: 'current' })),
                    node.savedLorasA || []
                );
                const mergedB = mergeLoraLists(
                    connectedLorasB.map(l => ({ ...l, source: 'current' })),
                    node.savedLorasB || []
                );
                
                // Filter to only active loras
                allLorasA = mergedA.filter(l => l.active !== false);
                allLorasB = mergedB.filter(l => l.active !== false);
            }

            await savePrompt(node, currentCategory, promptName.trim(), promptText, allLorasA, allLorasB);
            
            // Update the saved loras to what was just saved
            node.savedLorasA = allLorasA;
            node.savedLorasB = allLorasB;
            updateLoraDisplays(node);
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
        // Allow wrapping - estimate height based on content
        // Buttons wrap at ~380px width
        const rows = width < 380 ? 2 : 1;
        return [width, rows * 28 + 8];
    };



    node.buttonBarAttached = true;
}

function setupCategoryChangeHandler(node) {
    const categoryWidget = node.widgets.find(w => w.name === "category");
    const promptWidget = node.widgets.find(w => w.name === "name");
    const textWidget = node.widgets.find(w => w.name === "text");

    if (!categoryWidget || !promptWidget || !textWidget) return;

    const originalCallback = categoryWidget.callback;

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
                loadPromptData(node, category, promptNames[0]);
            } else {
                promptWidget.value = "";
                textWidget.value = "";
                // Clear ALL loras when no prompts in category
                node.savedLorasA = [];
                node.savedLorasB = [];
                node.currentLorasA = [];
                node.currentLorasB = [];
                updateLoraDisplays(node);
            }

            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
        }
    };

    const originalPromptCallback = promptWidget.callback;

    promptWidget.callback = function(value) {
        if (originalPromptCallback) {
            originalPromptCallback.apply(this, arguments);
        }

        const category = categoryWidget.value;
        loadPromptData(node, category, value);

        node.serialize_widgets = true;
        app.graph.setDirtyCanvas(true, true);
    };

    // Prevent ComfyUI from setting all prompts
    Object.defineProperty(promptWidget.options, 'values', {
        get: function() {
            return this._values;
        },
        set: function(newValues) {
            this._values = newValues;

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
    const overrideLoraWidget = node.widgets?.find(w => w.name === "override_lora");
    const categoryWidget = node.widgets?.find(w => w.name === "category");
    const promptWidget = node.widgets?.find(w => w.name === "name");
    
    if (!textWidget || !useExternalWidget) return;

    // Setup override_lora toggle handler to update lora display
    if (overrideLoraWidget) {
        const originalOverrideCallback = overrideLoraWidget.callback;
        overrideLoraWidget.callback = function(value) {
            if (originalOverrideCallback) {
                originalOverrideCallback.apply(this, arguments);
            }
            // Update lora displays when override changes
            updateLoraDisplays(node);
        };
    }

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

    // Store original callback and add our handler
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

async function loadPromptData(node, category, promptName) {
    const textWidget = node.widgets.find(w => w.name === "text");
    
    // Always clear ALL loras when switching prompts (both saved and current)
    // This ensures clean state - current loras will be repopulated on next execution
    node.savedLorasA = [];
    node.savedLorasB = [];
    node.currentLorasA = [];
    node.currentLorasB = [];
    
    if (!node.prompts || !node.prompts[category] || !node.prompts[category][promptName]) {
        if (textWidget) textWidget.value = "";
        updateLoraDisplays(node);
        return;
    }

    const promptData = node.prompts[category][promptName];
    
    if (textWidget) {
        textWidget.value = promptData.prompt || "";
    }
    
    // Load saved loras - all saved loras are active (inactive ones weren't saved)
    const lorasA = (promptData.loras_a || []).map(lora => ({
        ...lora,
        active: true, // All saved loras are active
        strength: lora.strength ?? lora.model_strength ?? 1.0,
        available: true // Will be updated after check
    }));
    const lorasB = (promptData.loras_b || []).map(lora => ({
        ...lora,
        active: true, // All saved loras are active
        strength: lora.strength ?? lora.model_strength ?? 1.0,
        available: true
    }));
    
    // Check availability of all loras
    const allLoraNames = [
        ...lorasA.map(l => l.name),
        ...lorasB.map(l => l.name)
    ].filter(name => name);
    
    if (allLoraNames.length > 0) {
        try {
            const response = await fetch("/prompt-manager-advanced/check-loras", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ lora_names: allLoraNames })
            });
            const data = await response.json();
            
            if (data.success && data.results) {
                // Update availability status
                lorasA.forEach(lora => {
                    lora.available = data.results[lora.name] !== false;
                });
                lorasB.forEach(lora => {
                    lora.available = data.results[lora.name] !== false;
                });
            }
        } catch (error) {
            console.error("[PromptManagerAdvanced] Error checking LoRA availability:", error);
        }
    }
    
    node.savedLorasA = lorasA;
    node.savedLorasB = lorasB;
    
    updateLoraDisplays(node);
}

// ========================
// API Functions
// ========================

async function createCategory(node, categoryName) {
    try {
        const response = await fetch("/prompt-manager-advanced/save-category", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ category_name: categoryName })
        });

        const data = await response.json();

        if (data.success) {
            node.prompts = data.prompts;

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

            node.savedLorasA = [];
            node.savedLorasB = [];
            updateLoraDisplays(node);
            updateDropdowns(node);

            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
        } else {
            await showInfo("Error", data.error);
        }
    } catch (error) {
        console.error("[PromptManagerAdvanced] Error creating category:", error);
        await showInfo("Error", "Error creating category");
    }
}

async function savePrompt(node, category, name, text, lorasA, lorasB) {
    try {
        const response = await fetch("/prompt-manager-advanced/save-prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                category: category,
                name: name,
                text: text,
                // Only save name and strength - all saved loras are active by definition
                loras_a: lorasA.map(l => ({
                    name: l.name,
                    strength: l.strength ?? l.model_strength ?? 1.0,
                    clip_strength: l.clip_strength || l.strength || 1.0
                })),
                loras_b: lorasB.map(l => ({
                    name: l.name,
                    strength: l.strength ?? l.model_strength ?? 1.0,
                    clip_strength: l.clip_strength || l.strength || 1.0
                }))
            })
        });

        const data = await response.json();

        if (data.success) {
            node.prompts = data.prompts;
            updateDropdowns(node);

            const promptWidget = node.widgets.find(w => w.name === "name");
            if (promptWidget) {
                promptWidget.value = name;
            }

            // Update saved loras with what was just saved
            node.savedLorasA = lorasA;
            node.savedLorasB = lorasB;
            updateLoraDisplays(node);

            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
            
            await showInfo("Success", `Prompt "${name}" saved successfully!`);
        } else {
            await showInfo("Error", data.error);
        }
    } catch (error) {
        console.error("[PromptManagerAdvanced] Error saving prompt:", error);
        await showInfo("Error", "Error saving prompt");
    }
}

async function deleteCategory(node, category) {
    try {
        const response = await fetch("/prompt-manager-advanced/delete-category", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ category: category })
        });

        const data = await response.json();

        if (data.success) {
            node.prompts = data.prompts;
            node.savedLorasA = [];
            node.savedLorasB = [];
            updateDropdowns(node);
            updateLoraDisplays(node);

            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
        } else {
            await showInfo("Error", data.error);
        }
    } catch (error) {
        console.error("[PromptManagerAdvanced] Error deleting category:", error);
        await showInfo("Error", "Error deleting category");
    }
}

async function deletePrompt(node, category, name) {
    try {
        const response = await fetch("/prompt-manager-advanced/delete-prompt", {
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
            node.savedLorasA = [];
            node.savedLorasB = [];
            updateDropdowns(node);
            updateLoraDisplays(node);

            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
        } else {
            await showInfo("Error", data.error);
        }
    } catch (error) {
        console.error("[PromptManagerAdvanced] Error deleting prompt:", error);
        await showInfo("Error", "Error deleting prompt");
    }
}

function updateDropdowns(node) {
    const categoryWidget = node.widgets.find(w => w.name === "category");
    const promptWidget = node.widgets.find(w => w.name === "name");
    const textWidget = node.widgets.find(w => w.name === "text");

    if (!categoryWidget || !promptWidget || !textWidget) return;

    // Update category dropdown (sorted alphabetically)
    const categories = Object.keys(node.prompts).sort((a, b) => a.localeCompare(b));
    categoryWidget.options.values = categories.length > 0 ? categories : ["Default"];

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

// ========================
// UI Helper Functions
// ========================

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

console.log("[PromptManagerAdvanced] Extension loaded");
