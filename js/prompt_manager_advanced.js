import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * PromptManagerAdvanced Extension for ComfyUI
 * Extends prompt management with LoRA stack support and toggleable LoRA displays
 */

// ========================
// Canvas Wheel Event Helper
// ========================

/**
 * Forward wheel events from a DOM element to the LiteGraph canvas for zooming
 * This allows users to zoom the canvas even when hovering over node UI elements
 */
function forwardWheelToCanvas(element) {
    element.addEventListener("wheel", (e) => {
        // Get the LiteGraph canvas element
        const canvas = app.canvas?.canvas || document.querySelector("canvas.lgraphcanvas");
        if (canvas) {
            // Create and dispatch a new wheel event to the canvas
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
                metaKey: e.metaKey
            });
            canvas.dispatchEvent(newEvent);
        }
    }, { passive: true });
}

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

// Register setting for unsaved changes warning
app.registerExtension({
    name: "PromptManagerAdvanced.Settings",
    async setup() {
        app.ui.settings.addSetting({
            id: "PromptManagerAdvanced.warnUnsavedChanges",
            name: "Warn about unsaved prompt changes",
            type: "boolean",
            defaultValue: true,
            tooltip: "Show a warning when switching prompts if there are unsaved changes to the current prompt text, LoRAs, or trigger words."
        });
    }
});

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
                node.originalLorasA = [];  // Original strengths for reset
                node.originalLorasB = [];  // Original strengths for reset
                node.currentTriggerWords = [];  // From connected input
                node.savedTriggerWords = [];    // From saved prompt
                node.connectedThumbnail = null; // Thumbnail from connected image (set during execution)

                // Track the last known state of inputs and prompt to detect changes
                // When any of these change, we clear toggle states and reload fresh
                node.lastKnownPromptKey = "";  // "category|promptName" 
                node.lastKnownInputLorasA = "";  // Signature of input loras
                node.lastKnownInputLorasB = "";  // Signature of input loras
                node.lastKnownPromptInput = "";  // Text from connected prompt input (extractor)

                // Track last saved state for unsaved changes detection
                node.lastSavedState = {
                    text: "",
                    lorasA: "[]",
                    lorasB: "[]",
                    triggerWords: "[]"
                };

                // Set initial size - taller to accommodate lora displays
                this.setSize([440, 700]);

                // Change widget labels
                const promptTextWidget = this.widgets.find(w => w.name === "text");
                if (promptTextWidget) {
                    promptTextWidget.label = "prompt";
                }

                const promptNameWidget = this.widgets.find(w => w.name === "name");
                if (promptNameWidget) {
                    promptNameWidget.label = "name";
                }

                const useExternalWidget = this.widgets.find(w => w.name === "use_prompt_input");
                if (useExternalWidget) {
                    useExternalWidget.label = "use prompt input";
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
                        const newTriggerWords = event.detail.trigger_words || [];
                        const inputLorasA = event.detail.input_loras_a || [];
                        const inputLorasB = event.detail.input_loras_b || [];
                        // Explicit list of unavailable lora names from Python
                        const unavailableLorasA = new Set((event.detail.unavailable_loras_a || []).map(n => n.toLowerCase()));
                        const unavailableLorasB = new Set((event.detail.unavailable_loras_b || []).map(n => n.toLowerCase()));

                        // Store connected thumbnail for use when saving
                        this.connectedThumbnail = event.detail.connected_thumbnail || null;

                        // Get current prompt key from widgets
                        const categoryWidget = this.widgets?.find(w => w.name === "category");
                        const promptWidget = this.widgets?.find(w => w.name === "name");
                        const currentPromptKey = `${categoryWidget?.value || ""}|${promptWidget?.value || ""}`;

                        // Create signatures of input loras (just names, sorted)
                        const inputLorasASignature = inputLorasA.map(l => l.name).sort().join('|');
                        const inputLorasBSignature = inputLorasB.map(l => l.name).sort().join('|');

                        // Get the prompt input text (from extractor)
                        const promptInputText = event.detail.prompt_input || "";

                        // Check what has changed
                        const promptChanged = currentPromptKey !== this.lastKnownPromptKey;
                        const inputLorasChanged = inputLorasASignature !== this.lastKnownInputLorasA || 
                                                  inputLorasBSignature !== this.lastKnownInputLorasB;
                        const promptInputChanged = promptInputText !== this.lastKnownPromptInput;

                        // Determine if we should reset:
                        // 1. Prompt dropdown changed → reset
                        // 2. BOTH prompt input text AND loras changed → new extraction, reset
                        // 3. ONLY loras changed → user modified strengths, don't reset
                        const isNewExtraction = promptInputChanged && inputLorasChanged;
                        const shouldReset = promptChanged || isNewExtraction;

                        // DEBUG: Always log the state
                        console.log("[PromptManagerAdvanced] Update received:", {
                            promptChanged,
                            inputLorasChanged,
                            promptInputChanged,
                            isNewExtraction,
                            shouldReset,
                            promptInputText: promptInputText.substring(0, 50) + "...",
                            lastKnownPromptInput: this.lastKnownPromptInput.substring(0, 50) + "...",
                            inputLorasASignature,
                            lastKnownInputLorasA: this.lastKnownInputLorasA
                        });

                        // Always update tracking signatures
                        this.lastKnownInputLorasA = inputLorasASignature;
                        this.lastKnownInputLorasB = inputLorasBSignature;
                        this.lastKnownPromptInput = promptInputText;

                        // Clear and reload when needed
                        if (shouldReset) {
                            console.log("[PromptManagerAdvanced] Resetting - Prompt changed:", promptChanged, "New extraction:", isNewExtraction);

                            // Update prompt tracking
                            this.lastKnownPromptKey = currentPromptKey;

                            // Clear ALL loras - we'll reload fresh
                            this.currentLorasA = [];
                            this.currentLorasB = [];
                            this.savedLorasA = [];
                            this.savedLorasB = [];
                            this.originalLorasA = [];  // Clear original strengths too
                            this.originalLorasB = [];

                            // Reload saved loras from prompt to get clean toggle states
                            // Use Python's unavailable list directly - it's the source of truth
                            if (this.prompts && categoryWidget && promptWidget) {
                                const promptData = this.prompts[categoryWidget.value]?.[promptWidget.value];
                                if (promptData) {
                                    this.savedLorasA = (promptData.loras_a || []).map(lora => ({
                                        ...lora,
                                        active: lora.active !== false,
                                        strength: lora.strength ?? lora.model_strength ?? 1.0,
                                        available: !unavailableLorasA.has(lora.name.toLowerCase())
                                    }));
                                    this.savedLorasB = (promptData.loras_b || []).map(lora => ({
                                        ...lora,
                                        active: lora.active !== false,
                                        strength: lora.strength ?? lora.model_strength ?? 1.0,
                                        available: !unavailableLorasB.has(lora.name.toLowerCase())
                                    }));
                                    // Store original strengths for reset functionality
                                    this.originalLorasA = (promptData.loras_a || []).map(lora => ({
                                        name: lora.name,
                                        strength: lora.strength ?? lora.model_strength ?? 1.0
                                    }));
                                    this.originalLorasB = (promptData.loras_b || []).map(lora => ({
                                        name: lora.name,
                                        strength: lora.strength ?? lora.model_strength ?? 1.0
                                    }));
                                }
                            }

                            // Set current loras from input (these are the loras from connected nodes)
                            this.currentLorasA = inputLorasA.map(l => ({ ...l, source: 'current' }));
                            this.currentLorasB = inputLorasB.map(l => ({ ...l, source: 'current' }));

                            // Also store input lora original strengths for reset
                            inputLorasA.forEach(l => {
                                if (!this.originalLorasA.find(o => o.name.toLowerCase() === l.name.toLowerCase())) {
                                    this.originalLorasA.push({ name: l.name, strength: l.strength ?? l.model_strength ?? 1.0 });
                                }
                            });
                            inputLorasB.forEach(l => {
                                if (!this.originalLorasB.find(o => o.name.toLowerCase() === l.name.toLowerCase())) {
                                    this.originalLorasB.push({ name: l.name, strength: l.strength ?? l.model_strength ?? 1.0 });
                                }
                            });

                            // Update connected trigger words
                            const newConnectedTriggers = newTriggerWords.filter(t => t.source === 'connected');
                            this.currentTriggerWords = newConnectedTriggers;

                            // Refresh displays
                            updateLoraDisplays(this);
                            updateTriggerWordsDisplay(this);
                        } else {
                            // No major change - just update current loras if they differ
                            const lorasAChanged = JSON.stringify(inputLorasA) !== JSON.stringify(this.currentLorasA.map(l => ({ name: l.name, strength: l.strength })));
                            const lorasBChanged = JSON.stringify(inputLorasB) !== JSON.stringify(this.currentLorasB.map(l => ({ name: l.name, strength: l.strength })));
                            const newConnectedTriggers = newTriggerWords.filter(t => t.source === 'connected');
                            const triggerWordsChanged = JSON.stringify(newConnectedTriggers) !== JSON.stringify(this.currentTriggerWords);

                            if (lorasAChanged || lorasBChanged || triggerWordsChanged) {
                                this.currentLorasA = inputLorasA.map(l => ({ ...l, source: 'current' }));
                                this.currentLorasB = inputLorasB.map(l => ({ ...l, source: 'current' }));
                                this.currentTriggerWords = newConnectedTriggers;

                                updateLoraDisplays(this);
                                updateTriggerWordsDisplay(this);
                            }
                        }

                        // Handle use_prompt_input toggle state for text widget
                        const promptTextWidget = this.widgets.find(w => w.name === "text");
                        if (promptTextWidget) {
                            const useExternal = event.detail.use_prompt_input || false;
                            const llmInput = event.detail.prompt_input || "";

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
                createPromptSelectorWidget(node);  // Custom thumbnail selector (before buttons)
                addButtonBar(node);
                addLoraDisplays(node);
                addTriggerWordsDisplay(node);
                setupCategoryChangeHandler(node);
                setupUseExternalToggleHandler(node);

                // Load prompts asynchronously (data only, not widgets)
                loadPrompts(node).then(() => {
                    filterPromptDropdown(node);

                    // Update custom prompt selector display
                    if (node.updatePromptSelectorDisplay) {
                        node.updatePromptSelectorDisplay();
                    }

                    // Load initial prompt data (LoRAs and trigger words)
                    const categoryWidget = node.widgets.find(w => w.name === "category");
                    const promptWidget = node.widgets.find(w => w.name === "name");
                    if (categoryWidget && promptWidget && promptWidget.value) {
                        loadPromptData(node, categoryWidget.value, promptWidget.value);
                    }

                    // Ensure height is sufficient after data is loaded
                    setTimeout(() => {
                        const computedSize = node.computeSize();
                        const minHeight = Math.max(600, computedSize[1] + 20);

                        if (node.size[1] < minHeight) {
                            node.setSize([Math.max(440, node.size[0]), minHeight]);
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

                // Detect if this is a fresh workflow load (page refresh) vs tab switch
                // If widgets_values doesn't have current_loras_a or it's a fresh session, clear currentLoras
                const isFreshLoad = !info.widgets_values || info.widgets_values.every(v => v === null || v === undefined);

                // Restore saved lora toggle states if they exist
                if (info.widgets_values) {
                    // Find the lora toggle widget indices and restore their values
                    const lorasAIndex = node.widgets?.findIndex(w => w.name === "loras_a_toggle");
                    const lorasBIndex = node.widgets?.findIndex(w => w.name === "loras_b_toggle");
                    const triggerWordsIndex = node.widgets?.findIndex(w => w.name === "trigger_words_toggle");
                    const currentLorasAIndex = node.widgets?.findIndex(w => w.name === "current_loras_a");
                    const currentLorasBIndex = node.widgets?.findIndex(w => w.name === "current_loras_b");

                    if (lorasAIndex >= 0 && info.widgets_values[lorasAIndex]) {
                        try {
                            node.savedLorasA = JSON.parse(info.widgets_values[lorasAIndex]);
                            // Store original strengths for reset functionality
                            node.originalLorasA = node.savedLorasA.map(lora => ({
                                name: lora.name,
                                strength: lora.strength ?? lora.model_strength ?? 1.0
                            }));
                        } catch (e) {
                            node.savedLorasA = [];
                            node.originalLorasA = [];
                        }
                    }
                    if (lorasBIndex >= 0 && info.widgets_values[lorasBIndex]) {
                        try {
                            node.savedLorasB = JSON.parse(info.widgets_values[lorasBIndex]);
                            // Store original strengths for reset functionality
                            node.originalLorasB = node.savedLorasB.map(lora => ({
                                name: lora.name,
                                strength: lora.strength ?? lora.model_strength ?? 1.0
                            }));
                        } catch (e) {
                            node.savedLorasB = [];
                            node.originalLorasB = [];
                        }
                    }
                    if (triggerWordsIndex >= 0 && info.widgets_values[triggerWordsIndex]) {
                        try {
                            node.savedTriggerWords = JSON.parse(info.widgets_values[triggerWordsIndex]);
                        } catch (e) {
                            node.savedTriggerWords = [];
                        }
                    }

                    // Restore current loras for tab-switch persistence, but clear on fresh load
                    if (!isFreshLoad) {
                        if (currentLorasAIndex >= 0 && info.widgets_values[currentLorasAIndex]) {
                            try {
                                node.currentLorasA = JSON.parse(info.widgets_values[currentLorasAIndex]);
                            } catch (e) {
                                node.currentLorasA = [];
                            }
                        }
                        if (currentLorasBIndex >= 0 && info.widgets_values[currentLorasBIndex]) {
                            try {
                                node.currentLorasB = JSON.parse(info.widgets_values[currentLorasBIndex]);
                            } catch (e) {
                                node.currentLorasB = [];
                            }
                        }
                    } else {
                        // Fresh load - clear current loras and reset tracking
                        node.currentLorasA = [];
                        node.currentLorasB = [];
                        node.lastKnownPromptKey = "";
                        node.lastKnownInputLorasA = "";
                        node.lastKnownInputLorasB = "";
                        // Note: originalLoras are preserved from savedLoras restoration above
                    }
                }

                // IMPORTANT: Reattach DOM widgets SYNCHRONOUSLY during configure
                // to ensure proper positioning within the node bounds
                if (!node.promptSelectorWidget) {
                    createPromptSelectorWidget(node);
                }
                if (!node.buttonBarAttached) {
                    addButtonBar(node);
                    setupCategoryChangeHandler(node);
                }
                if (!node.loraDisplaysAttached) {
                    addLoraDisplays(node);
                }
                if (!node.triggerWordsDisplayAttached) {
                    addTriggerWordsDisplay(node);
                }
                setupUseExternalToggleHandler(node);

                // Load prompts data asynchronously (data only, widgets already added)
                loadPrompts(node).then(() => {
                    filterPromptDropdown(node);
                    updateLoraDisplays(node);
                    updateTriggerWordsDisplay(node);

                    // Update custom prompt selector display
                    if (node.updatePromptSelectorDisplay) {
                        node.updatePromptSelectorDisplay();
                    }

                    app.graph.setDirtyCanvas(true, true);
                });

                return result;
            };

            // Enforce minimum node size
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                size[0] = Math.max(440, size[0]);
                size[1] = Math.max(600, size[1]);
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
    const loraAWidget = node.addDOMWidget("loras_a_display", "div", loraAContainer, {
        hideOnZoom: true,
        serialize: false
    });
    loraAWidget.computeSize = function(width) {
        // Check if use_lora_input is disabled (use only saved loras)
        const useLoraInputWidget = node.widgets?.find(w => w.name === "use_lora_input");
        const useLoraInput = useLoraInputWidget?.value !== false;

        // Count loras based on mode
        let tagCount;
        if (!useLoraInput) {
            // use_lora_input OFF: only saved loras
            tagCount = (node.savedLorasA || []).length;
        } else {
            // use_lora_input ON: unique loras from merged list
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
        // Check if use_lora_input is disabled
        const useLoraInputWidget = node.widgets?.find(w => w.name === "use_lora_input");
        const useLoraInput = useLoraInputWidget?.value !== false;

        // Count loras based on mode
        let tagCount;
        if (!useLoraInput) {
            // use_lora_input OFF: only saved loras
            tagCount = (node.savedLorasB || []).length;
        } else {
            // use_lora_input ON: unique loras from merged list
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

    // Add hidden widgets to store current (connected) loras for tab-switch persistence
    const currentLorasAWidget = node.addWidget('text', 'current_loras_a', '[]');
    currentLorasAWidget.type = "converted-widget";
    currentLorasAWidget.hidden = true;
    currentLorasAWidget.computeSize = () => [0, -4];
    node.currentLorasAWidget = currentLorasAWidget;

    const currentLorasBWidget = node.addWidget('text', 'current_loras_b', '[]');
    currentLorasBWidget.type = "converted-widget";
    currentLorasBWidget.hidden = true;
    currentLorasBWidget.computeSize = () => [0, -4];
    node.currentLorasBWidget = currentLorasBWidget;

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

    // Prevent default context menu on container (LoRA tags have their own)
    container.addEventListener("contextmenu", (e) => e.preventDefault());

    // Forward wheel events to canvas for zooming
    forwardWheelToCanvas(container);

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

    // Button container for Reset and Toggle All
    const btnContainer = document.createElement("div");
    Object.assign(btnContainer.style, {
        display: "flex",
        gap: "4px"
    });

    // Reset button to restore original strength values
    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset Strength";
    resetBtn.title = "Reset all strength values to their original values";
    Object.assign(resetBtn.style, {
        fontSize: "10px",
        padding: "2px 8px",
        backgroundColor: "#333",
        color: "#ccc",
        border: "1px solid #555",
        borderRadius: "6px",
        cursor: "pointer"
    });
    resetBtn.onclick = () => resetLoraStrengths(node, stackId);
    btnContainer.appendChild(resetBtn);

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
    btnContainer.appendChild(toggleAllBtn);

    titleBar.appendChild(btnContainer);

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

    // Check if use_lora_input is disabled
    const useLoraInputWidget = node.widgets?.find(w => w.name === "use_lora_input");
    const useLoraInput = useLoraInputWidget?.value !== false;

    let lorasA, lorasB;
    if (!useLoraInput) {
        // use_lora_input OFF: Only show saved loras from the prompt
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
    // Create a map of saved loras to preserve user toggle states (case-insensitive)
    const savedMap = new Map();
    (savedLoras || []).forEach(lora => {
        savedMap.set(lora.name.toLowerCase(), lora);
    });

    // Merge loras, preserving toggle states from saved when available
    const merged = [];
    const seen = new Set();

    // First add all current loras, but preserve toggle state from saved if exists
    (currentLoras || []).forEach(lora => {
        const loraNameLower = lora.name.toLowerCase();
        const savedLora = savedMap.get(loraNameLower);
        if (savedLora) {
            // Lora exists in both - mark as 'saved' since user has it in their preset
            // This ensures modifications are persisted to savedLoras, not currentLoras
            // Preserve fromInput flag if it was already set (meaning it was moved from current to saved)
            merged.push({
                ...lora,
                active: savedLora.active,
                strength: savedLora.strength ?? savedLora.model_strength ?? lora.strength ?? 1.0,
                source: 'saved',  // Important: mark as saved so modifications persist correctly
                fromInput: savedLora.fromInput || false  // Preserve the fromInput flag
            });
        } else {
            merged.push({ ...lora, source: 'current', fromInput: true });
        }
        seen.add(loraNameLower);
    });

    // Then add saved loras that aren't in current (case-insensitive)
    (savedLoras || []).forEach(lora => {
        if (!seen.has(lora.name.toLowerCase())) {
            merged.push({ ...lora, source: 'saved' });
            seen.add(lora.name.toLowerCase());
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
    const isAvailable = lora.available === true;
    const isFromInput = lora.fromInput === true || lora.source === 'current';  // From connected input
    const strength = parseFloat(lora.strength ?? lora.model_strength ?? 1.0) || 1.0;

    // Determine colors based on active and available status
    let bgColor, textColor, borderColor;
    if (!isAvailable) {
        // Missing LoRA - red/orange warning colors
        bgColor = isActive ? "rgba(220, 53, 69, 0.9)" : "rgba(220, 53, 69, 0.4)";
        textColor = isActive ? "white" : "rgba(255, 200, 200, 0.8)";
        borderColor = "rgba(220, 53, 69, 0.9)";
    } else if (isFromInput) {
        // From connected input - purple tint
        if (isActive) {
        bgColor = "rgba(50, 112, 163, 0.9)";
        textColor = "white";
        borderColor = "rgba(66, 153, 225, 0.9)";
        } else {
            bgColor = "rgba(45, 55, 72, 0.7)";
            textColor = "rgba(226, 232, 240, 0.6)";
            borderColor = "rgba(226, 232, 240, 0.2)";
        }
    } else if (isActive) {
        // Available and active - blue (saved in prompt)
        bgColor = "rgba(66, 153, 225, 0.9)";
        textColor = "white";
        borderColor = "rgba(122, 188, 243, 0.9)";
    } else {
        // Available but inactive - gray (saved in prompt)
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
        // Position it centered below the tag
        if (loraManagerAvailable && isAvailable) {
            const tagRect = tag.getBoundingClientRect();
            hoverTimeout = setTimeout(async () => {
                const tooltip = await getLoraManagerPreviewTooltip();
                if (tooltip) {
                    // Center horizontally under the tag
                    const tooltipX = tagRect.right;
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

    // Right-click handler to show context menu
    tag.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        e.stopPropagation();
        showLoraContextMenu(e, node, stackId, index, lora.name, isAvailable);
    });

    // Only add title if LoRA Manager preview is not available (to not conflict with preview tooltip)
    if (!loraManagerAvailable || !isAvailable) {
        if (!isAvailable) {
            // Missing LoRA tooltip
            tag.title = `${lora.name}\n⚠️ NOT FOUND - This LoRA is missing from your system\nRight-click for options`;
        } else {
            // LoRA Manager not available, show normal tooltip
            tag.title = `${lora.name}\nStrength: ${strength.toFixed(2)}\nClick to toggle on/off\nRight-click for options`;
        }
    }

    return tag;
}

function toggleLoraActive(node, stackId, index) {
    // Check if use_lora_input is disabled
    const useLoraInputWidget = node.widgets?.find(w => w.name === "use_lora_input");
    const useLoraInput = useLoraInputWidget?.value !== false;

    // Get the list that matches what's currently displayed
    let loraList;
    if (!useLoraInput) {
        // use_lora_input OFF: only saved loras are displayed
        loraList = stackId === "a" ? [...(node.savedLorasA || [])] : [...(node.savedLorasB || [])];
    } else {
        // use_lora_input ON: merged list is displayed
        loraList = stackId === "a" ?
            mergeLoraLists(node.currentLorasA, node.savedLorasA) :
            mergeLoraLists(node.currentLorasB, node.savedLorasB);
    }

    if (loraList[index]) {
        const lora = loraList[index];

        // Toggle the active state
        loraList[index].active = !loraList[index].active;

        // If this was a connected lora (source: 'current'), move it to saved
        // so that the toggle state persists across workflow executions
        if (lora.source === 'current') {
            loraList[index].source = 'saved';
            loraList[index].fromInput = true;  // Preserve visual indicator that it came from input
        }

        // Update the appropriate list
        if (!useLoraInput) {
            // Only update saved loras when use_lora_input is off
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

function showLoraContextMenu(e, node, stackId, index, loraName, isAvailable = true) {
    // Remove any existing context menu
    const existingMenu = document.querySelector(".lora-context-menu");
    if (existingMenu) {
        existingMenu.remove();
    }

    const menu = document.createElement("div");
    menu.className = "lora-context-menu";
    menu.style.cssText = `
        position: fixed;
        left: ${e.clientX}px;
        top: ${e.clientY}px;
        background: #2a2a2a;
        border: 1px solid #555;
        border-radius: 6px;
        z-index: 999999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        min-width: 120px;
        padding: 4px 0;
    `;

    // Title showing full LoRA name
    const titleItem = document.createElement("div");
    titleItem.textContent = loraName;
    titleItem.style.cssText = `
        padding: 8px 12px;
        font-size: 12px;
        color: #aaa;
        font-weight: bold;
        border-bottom: 1px solid #444;
        margin-bottom: 4px;
        white-space: nowrap;
    `;
    menu.appendChild(titleItem);

    // Search on CivitAI option (only for missing LoRAs)
    if (!isAvailable) {
        const searchItem = document.createElement("div");
        searchItem.textContent = "🔍 Search on CivitAI";
        searchItem.style.cssText = `
            padding: 8px 12px;
            cursor: pointer;
            font-size: 12px;
            color: #4da6ff;
            white-space: nowrap;
        `;
        searchItem.addEventListener("mouseenter", () => {
            searchItem.style.backgroundColor = "#3a3a3a";
        });
        searchItem.addEventListener("mouseleave", () => {
            searchItem.style.backgroundColor = "transparent";
        });
        searchItem.addEventListener("click", (evt) => {
            evt.stopPropagation();
            menu.remove();
            // Open CivitAI search in new tab
            const searchQuery = encodeURIComponent(loraName);
            window.open(`https://civitai.com/search/models?sortBy=models_v9&query=${searchQuery}&modelType=LORA`, "_blank");
        });
        menu.appendChild(searchItem);
    }

    const deleteItem = document.createElement("div");
    deleteItem.textContent = "Delete";
    deleteItem.style.cssText = `
        padding: 8px 12px;
        cursor: pointer;
        font-size: 12px;
        color: #ff6b6b;
        white-space: nowrap;
    `;
    deleteItem.addEventListener("mouseenter", () => {
        deleteItem.style.backgroundColor = "#3a3a3a";
    });
    deleteItem.addEventListener("mouseleave", () => {
        deleteItem.style.backgroundColor = "transparent";
    });
    deleteItem.addEventListener("click", (evt) => {
        evt.stopPropagation();
        menu.remove();
        removeLora(node, stackId, index);
    });
    menu.appendChild(deleteItem);

    document.body.appendChild(menu);

    // Close menu when clicking elsewhere
    const closeMenu = (evt) => {
        if (!menu.contains(evt.target)) {
            menu.remove();
            document.removeEventListener("click", closeMenu);
            document.removeEventListener("contextmenu", closeMenu);
        }
    };

    // Delay adding listener to prevent immediate close
    setTimeout(() => {
        document.addEventListener("click", closeMenu);
        document.addEventListener("contextmenu", closeMenu);
    }, 0);
}

function removeLora(node, stackId, index) {
    // Check if use_lora_input is disabled
    const useLoraInputWidget = node.widgets?.find(w => w.name === "use_lora_input");
    const useLoraInput = useLoraInputWidget?.value !== false;

    // Get the list that matches what's currently displayed
    let loraList;
    if (!useLoraInput) {
        loraList = stackId === "a" ? [...(node.savedLorasA || [])] : [...(node.savedLorasB || [])];
    } else {
        loraList = stackId === "a" ?
            mergeLoraLists(node.currentLorasA, node.savedLorasA) :
            mergeLoraLists(node.currentLorasB, node.savedLorasB);
    }

    if (loraList[index]) {
        const lora = loraList[index];
        const loraNameLower = lora.name.toLowerCase();

        // Remove from saved loras
        if (stackId === "a") {
            node.savedLorasA = (node.savedLorasA || []).filter(
                l => l.name.toLowerCase() !== loraNameLower
            );
        } else {
            node.savedLorasB = (node.savedLorasB || []).filter(
                l => l.name.toLowerCase() !== loraNameLower
            );
        }

        // Also remove from current loras if present
        if (stackId === "a") {
            node.currentLorasA = (node.currentLorasA || []).filter(
                l => l.name.toLowerCase() !== loraNameLower
            );
        } else {
            node.currentLorasB = (node.currentLorasB || []).filter(
                l => l.name.toLowerCase() !== loraNameLower
            );
        }

        updateLoraDisplays(node);
        app.graph.setDirtyCanvas(true, true);
    }
}

/**
 * Set LoRA strength to a specific value
 */
function setLoraStrength(node, stackId, index, newStrength) {
    // Check if use_lora_input is disabled
    const useLoraInputWidget = node.widgets?.find(w => w.name === "use_lora_input");
    const useLoraInput = useLoraInputWidget?.value !== false;

    // Get the list that matches what's currently displayed
    let loraList;
    if (!useLoraInput) {
        // use_lora_input OFF: only saved loras are displayed
        loraList = stackId === "a" ? [...(node.savedLorasA || [])] : [...(node.savedLorasB || [])];
    } else {
        // use_lora_input ON: merged list is displayed
        loraList = stackId === "a" ?
            mergeLoraLists(node.currentLorasA, node.savedLorasA) :
            mergeLoraLists(node.currentLorasB, node.savedLorasB);
    }

    if (loraList[index]) {
        loraList[index].strength = newStrength;
        // Move to saved so the strength persists (connected loras need to become saved to persist changes)
        if (loraList[index].source === 'current') {
            loraList[index].source = 'saved';
        }

        if (!useLoraInput) {
            // Only update saved loras when use_lora_input is off
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
    // Check if use_lora_input is disabled
    const useLoraInputWidget = node.widgets?.find(w => w.name === "use_lora_input");
    const useLoraInput = useLoraInputWidget?.value !== false;

    // Get the list that matches what's currently displayed
    let loraList;
    if (!useLoraInput) {
        // use_lora_input OFF: only saved loras are displayed
        loraList = stackId === "a" ? [...(node.savedLorasA || [])] : [...(node.savedLorasB || [])];
    } else {
        // use_lora_input ON: merged list is displayed
        loraList = stackId === "a" ?
            mergeLoraLists(node.currentLorasA, node.savedLorasA) :
            mergeLoraLists(node.currentLorasB, node.savedLorasB);
    }

    // Determine if we should turn all on or all off
    const allActive = loraList.every(lora => lora.active !== false);
    const newState = !allActive;

    loraList.forEach(lora => {
        lora.active = newState;
        // Move connected loras to saved so toggle state persists
        if (lora.source === 'current') {
            lora.source = 'saved';
        }
    });

    if (!useLoraInput) {
        // Only update saved loras when use_lora_input is off
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

function resetLoraStrengths(node, stackId) {
    // Reset all lora strengths to their original values
    const useLoraInputWidget = node.widgets?.find(w => w.name === "use_lora_input");
    const useLoraInput = useLoraInputWidget?.value !== false;

    // Get original strengths
    const originalLoras = stackId === "a" ? (node.originalLorasA || []) : (node.originalLorasB || []);

    // Build a map of original strengths by name (case-insensitive)
    const originalMap = new Map();
    originalLoras.forEach(lora => {
        originalMap.set(lora.name.toLowerCase(), lora.strength);
    });

    // Get the list that matches what's currently displayed
    let loraList;
    if (!useLoraInput) {
        loraList = stackId === "a" ? [...(node.savedLorasA || [])] : [...(node.savedLorasB || [])];
    } else {
        loraList = stackId === "a" ?
            mergeLoraLists(node.currentLorasA, node.savedLorasA) :
            mergeLoraLists(node.currentLorasB, node.savedLorasB);
    }

    // Reset strengths to original values
    loraList.forEach(lora => {
        const originalStrength = originalMap.get(lora.name.toLowerCase());
        if (originalStrength !== undefined) {
            lora.strength = originalStrength;
        }
    });

    // Update the lists
    if (!useLoraInput) {
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

    // Store current loras for tab-switch persistence
    if (node.currentLorasAWidget) {
        node.currentLorasAWidget.value = JSON.stringify(node.currentLorasA || []);
    }
    if (node.currentLorasBWidget) {
        node.currentLorasBWidget.value = JSON.stringify(node.currentLorasB || []);
    }

    // Format and store trigger words
    const triggerWords = node.savedTriggerWords || [];
    if (node.triggerWordsToggleWidget) {
        node.triggerWordsToggleWidget.value = JSON.stringify(triggerWords.map(tw => ({
            text: tw.text,
            active: tw.active !== false
        })));
    }
}

// ========================
// Trigger Words Display Functions
// ========================

function addTriggerWordsDisplay(node) {
    if (node.triggerWordsDisplayAttached) {
        return;
    }

    // Create Trigger Words display section
    const triggerWordsContainer = createTriggerWordsDisplayContainer("Trigger Words", node);
    const triggerWordsWidget = node.addDOMWidget("trigger_words_display", "div", triggerWordsContainer);
    triggerWordsWidget.computeSize = function(width) {
        // Try to get actual height from rendered DOM
        const tagsContainer = triggerWordsContainer.querySelector(".trigger-words-tags-container");
        if (tagsContainer && tagsContainer.children.length > 0) {
            // Use actual scrollHeight of tags container + header height (58px)
            const tagsHeight = tagsContainer.scrollHeight || tagsContainer.offsetHeight;
            const height = 58 + Math.max(28, tagsHeight + 8);
            return [width, height];
        }

        // Fallback: estimate based on tag count
        const merged = mergeTriggerWordLists(node.currentTriggerWords || [], node.savedTriggerWords || []);

        if (merged.length === 0) {
            return [width, 58 + 28];  // Base height + one row for empty message
        }

        // Simple estimate: assume average ~80px per tag
        const actualWidth = node.size?.[0] || width || 400;
        const availableWidth = actualWidth - 24;
        const avgTagWidth = 80;
        const tagsPerRow = Math.max(1, Math.floor(availableWidth / avgTagWidth));
        const rows = Math.max(1, Math.ceil(merged.length / tagsPerRow));

        const height = 58 + rows * 28;
        return [width, height];
    };
    node.triggerWordsWidget = triggerWordsWidget;
    node.triggerWordsContainer = triggerWordsContainer;

    // Add hidden widget to store toggle states for serialization
    const triggerWordsToggleWidget = node.addWidget('text', 'trigger_words_toggle', '[]');
    triggerWordsToggleWidget.type = "converted-widget";
    triggerWordsToggleWidget.hidden = true;
    triggerWordsToggleWidget.computeSize = () => [0, -4];
    node.triggerWordsToggleWidget = triggerWordsToggleWidget;

    node.triggerWordsDisplayAttached = true;
}

function createTriggerWordsDisplayContainer(title, node) {
    const container = document.createElement("div");
    container.className = "trigger-words-display-container";
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

    // Prevent default context menu on container (trigger word tags have their own)
    container.addEventListener("contextmenu", (e) => e.preventDefault());

    // Forward wheel events to canvas for zooming
    forwardWheelToCanvas(container);

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

    // Button container for right side
    const buttonContainer = document.createElement("div");
    Object.assign(buttonContainer.style, {
        display: "flex",
        gap: "4px"
    });

    // Add button
    const addBtn = document.createElement("button");
    addBtn.textContent = "+ Add";
    Object.assign(addBtn.style, {
        fontSize: "10px",
        padding: "2px 8px",
        backgroundColor: "#2a5a3a",
        color: "#ccc",
        border: "1px solid #3a7a4a",
        borderRadius: "6px",
        cursor: "pointer"
    });
    addBtn.onclick = () => addTriggerWordPrompt(node);
    buttonContainer.appendChild(addBtn);

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
    toggleAllBtn.onclick = () => toggleAllTriggerWords(node);
    buttonContainer.appendChild(toggleAllBtn);

    titleBar.appendChild(buttonContainer);

    container.appendChild(titleBar);

    // Tags container
    const tagsContainer = document.createElement("div");
    tagsContainer.className = "trigger-words-tags-container";
    Object.assign(tagsContainer.style, {
        display: "flex",
        flexWrap: "wrap",
        gap: "4px",
        minHeight: "30px"
    });
    container.appendChild(tagsContainer);

    return container;
}

function updateTriggerWordsDisplay(node) {
    if (!node.triggerWordsContainer) return;

    // Merge current (from connected input) and saved (from prompt) trigger words
    const triggerWords = mergeTriggerWordLists(
        node.currentTriggerWords || [],
        node.savedTriggerWords || []
    );

    // Update display
    const tagsContainer = node.triggerWordsContainer.querySelector(".trigger-words-tags-container");
    if (tagsContainer) {
        renderTriggerWordTags(tagsContainer, triggerWords, node);
    }

    // Update hidden widget for serialization
    updateToggleWidgets(node);

    app.graph.setDirtyCanvas(true, true);
}

function mergeTriggerWordLists(currentWords, savedWords) {
    // Create a map of saved words to preserve user toggle states
    const savedMap = new Map();
    (savedWords || []).forEach(word => {
        savedMap.set(word.text.toLowerCase(), word);
    });

    const merged = [];
    const seen = new Set();

    // First add all current words, but preserve toggle state from saved if exists
    (currentWords || []).forEach(word => {
        const wordLower = word.text.toLowerCase();
        const savedWord = savedMap.get(wordLower);
        if (savedWord) {
            // Word exists in both - use saved state
            merged.push({
                text: savedWord.text,
                active: savedWord.active,
                source: 'saved'
            });
        } else {
            merged.push({ ...word, source: 'current' });
        }
        seen.add(wordLower);
    });

    // Then add saved words that aren't in current
    (savedWords || []).forEach(word => {
        const wordLower = word.text.toLowerCase();
        if (!seen.has(wordLower)) {
            merged.push({ ...word, source: 'saved' });
            seen.add(wordLower);
        }
    });

    // Sort alphabetically by text to maintain stable order when toggling
    merged.sort((a, b) => a.text.localeCompare(b.text));

    return merged;
}

function renderTriggerWordTags(container, triggerWords, node) {
    // Clear existing tags
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }

    if (!triggerWords || triggerWords.length === 0) {
        const emptyMessage = document.createElement("div");
        emptyMessage.textContent = "No trigger words - click '+ Add' to add some";
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

    triggerWords.forEach((word, index) => {
        const tag = createTriggerWordTag(word, index, node);
        container.appendChild(tag);
    });
}

function createTriggerWordTag(word, index, node) {
    const tag = document.createElement("div");
    tag.className = "trigger-word-tag";
    tag.dataset.wordIndex = index;

    const isActive = word.active !== false;

    // Determine colors based on active status
    let bgColor, textColor, borderColor;
    if (isActive) {
        // Active - green/teal
        bgColor = "rgba(72, 187, 120, 0.9)";
        textColor = "white";
        borderColor = "rgba(72, 187, 120, 0.9)";
    } else {
        // Inactive - gray
        bgColor = "rgba(45, 55, 72, 0.7)";
        textColor = "rgba(226, 232, 240, 0.6)";
        borderColor = "rgba(226, 232, 240, 0.2)";
    }

    Object.assign(tag.style, {
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "4px 10px",
        borderRadius: "6px",
        fontSize: "12px",
        cursor: "pointer",
        transition: "background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease",
        backgroundColor: bgColor,
        color: textColor,
        border: `1px solid ${borderColor}`,
        height: "24px",
        boxSizing: "border-box",
        userSelect: "none",
        whiteSpace: "nowrap",
        maxWidth: "100%",
        overflow: "hidden"
    });

    // Trigger word text
    const textSpan = document.createElement("span");
    textSpan.textContent = word.text;
    Object.assign(textSpan.style, {
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap"
    });
    tag.appendChild(textSpan);

    // Click handler to toggle active state
    tag.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleTriggerWordActive(node, index);
    });

    // Right-click handler to show context menu
    tag.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        e.stopPropagation();
        showTriggerWordContextMenu(e, node, index, word.text);
    });

    // Hover effects
    tag.addEventListener("mouseenter", () => {
        tag.style.transform = "translateY(-1px)";
        tag.style.boxShadow = "0 2px 4px rgba(0,0,0,0.2)";
    });
    tag.addEventListener("mouseleave", () => {
        tag.style.transform = "translateY(0)";
        tag.style.boxShadow = "none";
    });

    // Add title for full text on hover
    tag.title = `${word.text}\nClick to toggle on/off\nRight-click for options`;

    return tag;
}

function toggleTriggerWordActive(node, index) {
    // Get the merged list that's currently displayed
    const triggerWords = mergeTriggerWordLists(
        node.currentTriggerWords || [],
        node.savedTriggerWords || []
    );

    if (triggerWords[index]) {
        const word = triggerWords[index];

        // Toggle the active state
        triggerWords[index].active = !triggerWords[index].active;

        // Update savedTriggerWords to preserve the state
        // Find and update or add to saved list
        const savedIndex = (node.savedTriggerWords || []).findIndex(
            w => w.text.toLowerCase() === word.text.toLowerCase()
        );

        if (savedIndex >= 0) {
            node.savedTriggerWords[savedIndex].active = triggerWords[index].active;
        } else {
            // Add to saved with current state
            if (!node.savedTriggerWords) node.savedTriggerWords = [];
            node.savedTriggerWords.push({
                text: word.text,
                active: triggerWords[index].active,
                source: 'saved'
            });
        }

        updateTriggerWordsDisplay(node);
        app.graph.setDirtyCanvas(true, true);
    }
}

function toggleAllTriggerWords(node) {
    // Get the merged list
    const triggerWords = mergeTriggerWordLists(
        node.currentTriggerWords || [],
        node.savedTriggerWords || []
    );

    // Determine if we should turn all on or all off
    const allActive = triggerWords.every(word => word.active !== false);
    const newState = !allActive;

    // Update all trigger words
    triggerWords.forEach(word => {
        word.active = newState;

        // Update or add to saved list
        const savedIndex = (node.savedTriggerWords || []).findIndex(
            w => w.text.toLowerCase() === word.text.toLowerCase()
        );

        if (savedIndex >= 0) {
            node.savedTriggerWords[savedIndex].active = newState;
        } else {
            if (!node.savedTriggerWords) node.savedTriggerWords = [];
            node.savedTriggerWords.push({
                text: word.text,
                active: newState,
                source: 'saved'
            });
        }
    });

    updateTriggerWordsDisplay(node);
    app.graph.setDirtyCanvas(true, true);
}

async function addTriggerWordPrompt(node) {
    const input = await showTextPrompt(
        "Add Trigger Words",
        "Enter trigger word(s) separated by commas:",
        ""
    );

    if (input && input.trim()) {
        // Parse comma-separated words
        const newWords = input.split(',')
            .map(w => w.trim())
            .filter(w => w.length > 0);

        if (newWords.length === 0) return;

        // Add each word to savedTriggerWords if not already present
        if (!node.savedTriggerWords) node.savedTriggerWords = [];

        const existingLower = new Set(
            node.savedTriggerWords.map(w => w.text.toLowerCase())
        );
        const currentLower = new Set(
            (node.currentTriggerWords || []).map(w => w.text.toLowerCase())
        );

        for (const word of newWords) {
            const wordLower = word.toLowerCase();
            // Only add if not already in saved or current
            if (!existingLower.has(wordLower) && !currentLower.has(wordLower)) {
                node.savedTriggerWords.push({
                    text: word,
                    active: true,
                    source: 'saved'
                });
                existingLower.add(wordLower);
            }
        }

        updateTriggerWordsDisplay(node);
        app.graph.setDirtyCanvas(true, true);
    }
}

function showTriggerWordContextMenu(e, node, index, wordText) {
    // Remove any existing context menu
    const existingMenu = document.querySelector(".trigger-word-context-menu");
    if (existingMenu) {
        existingMenu.remove();
    }

    const menu = document.createElement("div");
    menu.className = "trigger-word-context-menu";
    menu.style.cssText = `
        position: fixed;
        left: ${e.clientX}px;
        top: ${e.clientY}px;
        background: #2a2a2a;
        border: 1px solid #555;
        border-radius: 6px;
        z-index: 999999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        min-width: 120px;
        padding: 4px 0;
    `;

    const deleteItem = document.createElement("div");
    deleteItem.textContent = "Delete";
    deleteItem.style.cssText = `
        padding: 8px 12px;
        cursor: pointer;
        font-size: 12px;
        color: #ff6b6b;
        white-space: nowrap;
    `;
    deleteItem.addEventListener("mouseenter", () => {
        deleteItem.style.backgroundColor = "#3a3a3a";
    });
    deleteItem.addEventListener("mouseleave", () => {
        deleteItem.style.backgroundColor = "transparent";
    });
    deleteItem.addEventListener("click", (evt) => {
        evt.stopPropagation();
        menu.remove();
        removeTriggerWord(node, index);
    });
    menu.appendChild(deleteItem);

    document.body.appendChild(menu);

    // Close menu when clicking elsewhere
    const closeMenu = (evt) => {
        if (!menu.contains(evt.target)) {
            menu.remove();
            document.removeEventListener("click", closeMenu);
            document.removeEventListener("contextmenu", closeMenu);
        }
    };

    // Delay adding listener to prevent immediate close
    setTimeout(() => {
        document.addEventListener("click", closeMenu);
        document.addEventListener("contextmenu", closeMenu);
    }, 0);
}

function removeTriggerWord(node, index) {
    // Get the merged list that's currently displayed
    const triggerWords = mergeTriggerWordLists(
        node.currentTriggerWords || [],
        node.savedTriggerWords || []
    );

    if (triggerWords[index]) {
        const word = triggerWords[index];
        const wordLower = word.text.toLowerCase();

        // Remove from savedTriggerWords
        if (node.savedTriggerWords) {
            node.savedTriggerWords = node.savedTriggerWords.filter(
                w => w.text.toLowerCase() !== wordLower
            );
        }

        // Remove from currentTriggerWords (if it came from connected input)
        if (node.currentTriggerWords) {
            node.currentTriggerWords = node.currentTriggerWords.filter(
                w => w.text.toLowerCase() !== wordLower
            );
        }

        updateTriggerWordsDisplay(node);
        app.graph.setDirtyCanvas(true, true);
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
    buttonContainer.style.padding = "4px 4px 8px 4px";
    buttonContainer.style.flexWrap = "nowrap";
    buttonContainer.style.marginTop = "0";

    // Prevent default context menu on button bar
    buttonContainer.addEventListener("contextmenu", (e) => e.preventDefault());

    // Forward wheel events to canvas for zooming
    forwardWheelToCanvas(buttonContainer);

    // Save Prompt button
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

            // Check for existing prompt in target category
            let existingPromptName = null;
            if (node.prompts[targetCategory]) {
                const existingNames = Object.keys(node.prompts[targetCategory]);
                existingPromptName = existingNames.find(name => name.toLowerCase() === promptName.toLowerCase());
            }

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

            // Query connected LoRA stacker nodes to get current configurations
            // Also use node.currentLorasA/B which are populated from backend (works for PromptExtractor)
            const chainLorasA = collectAllLorasFromChain(node, "lora_stack_a");
            const chainLorasB = collectAllLorasFromChain(node, "lora_stack_b");

            // Combine chain loras with currentLoras (from backend update)
            // This handles both widget-based stackers and output-only nodes like PromptExtractor
            const connectedLorasA = chainLorasA.length > 0 ? chainLorasA : (node.currentLorasA || []);
            const connectedLorasB = chainLorasB.length > 0 ? chainLorasB : (node.currentLorasB || []);

            // Check if use_lora_input is disabled
            const useLoraInputWidget = node.widgets?.find(w => w.name === "use_lora_input");
            const useLoraInput = useLoraInputWidget?.value !== false;

            // Get loras to save
            let allLorasA, allLorasB;

            if (!useLoraInput) {
                allLorasA = [...(node.savedLorasA || [])];
                allLorasB = [...(node.savedLorasB || [])];
            } else {
                const mergedA = mergeLoraLists(
                    connectedLorasA.map(l => ({ ...l, source: 'current' })),
                    node.savedLorasA || []
                );
                const mergedB = mergeLoraLists(
                    connectedLorasB.map(l => ({ ...l, source: 'current' })),
                    node.savedLorasB || []
                );
                allLorasA = [...mergedA];
                allLorasB = [...mergedB];
            }

            // Get all trigger words (merged, with their states)
            const allTriggerWords = mergeTriggerWordLists(
                node.currentTriggerWords || [],
                node.savedTriggerWords || []
            );

            // Use connected thumbnail if available
            const thumbnail = node.connectedThumbnail || null;

            await savePrompt(node, targetCategory, promptName, promptText, allLorasA, allLorasB, allTriggerWords, thumbnail);

            // Clear new prompt flag since it's now saved
            node.isNewUnsavedPrompt = false;
            node.newPromptCategory = null;
            node.newPromptName = null;

            // Update UI to show the saved prompt
            categoryWidget.value = targetCategory;
            filterPromptDropdown(node);
            promptWidget.value = promptName;

            // Update previous values tracking
            node._previousCategory = targetCategory;
            node._previousPrompt = promptName;

            node.savedLorasA = allLorasA;
            node.savedLorasB = allLorasB;
            node.savedTriggerWords = allTriggerWords;
            updateLoraDisplays(node);
            updateTriggerWordsDisplay(node);

            // Update custom prompt selector display
            if (node.updatePromptSelectorDisplay) {
                node.updatePromptSelectorDisplay();
            }

            // Update last saved state after successful save
            updateLastSavedState(node);
        }
    });

    // New Prompt button - simply clears fields for a fresh start
    const newPromptBtn = createButton("New Prompt", async () => {
        // Check for unsaved changes before creating new prompt
        const hasUnsaved = hasUnsavedChanges(node);
        const warnEnabled = app.ui.settings.getSettingValue("PromptManagerAdvanced.warnUnsavedChanges", true);

        if (hasUnsaved && warnEnabled) {
            const confirmed = await showConfirm(
                "Unsaved Changes",
                "You have unsaved changes to the current prompt. Do you want to discard them and start fresh?",
                "Discard & Continue",
                "#f80"
            );
            if (!confirmed) {
                return;
            }
        }

        // Keep the current category, just clear the prompt selection and content
        const currentCategory = categoryWidget.value;

        // Clear prompt selection (set to empty/placeholder)
        promptWidget.value = "";
        textWidget.value = "";

        // Update previous values for cancel/revert
        node._previousCategory = currentCategory;
        node._previousPrompt = "";

        // Clear all loras and trigger words for the new prompt
        node.savedLorasA = [];
        node.savedLorasB = [];
        node.currentLorasA = [];
        node.currentLorasB = [];
        node.originalLorasA = [];  // Clear original strengths too
        node.originalLorasB = [];
        node.savedTriggerWords = [];
        node.currentTriggerWords = [];

        updateLoraDisplays(node);
        updateTriggerWordsDisplay(node);

        // Update custom prompt selector display
        if (node.updatePromptSelectorDisplay) {
            node.updatePromptSelectorDisplay();
        }

        // Mark as new unsaved prompt state
        node.isNewUnsavedPrompt = true;
        node.newPromptCategory = currentCategory;
        node.newPromptName = null;

        // Clear the last saved state since this is a brand new prompt
        node.lastSavedState = null;

        node.serialize_widgets = true;
        app.graph.setDirtyCanvas(true, true);
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

    if (!categoryWidget || !promptWidget || !textWidget) return;

    // Track previous values for cancel/revert
    node._previousCategory = categoryWidget.value;
    node._previousPrompt = promptWidget.value;

    const originalCallback = categoryWidget.callback;

    categoryWidget.callback = async function(value) {
        const previousCategory = node._previousCategory;
        const previousPrompt = node._previousPrompt;

        // Check for unsaved changes before switching (skip if navigating via custom selector)
        if (!node._skipUnsavedCheck) {
            const hasUnsaved = hasUnsavedChanges(node);
            const warnEnabled = app.ui.settings.getSettingValue("PromptManagerAdvanced.warnUnsavedChanges", true);

            if (hasUnsaved && warnEnabled) {
                const confirmed = await showConfirm(
                    "Unsaved Changes",
                    "You have unsaved changes to the current prompt. Do you want to discard them and switch?",
                    "Discard & Switch",
                    "#f80"
                );
                if (!confirmed) {
                    // Revert the category dropdown to previous value
                    categoryWidget.value = previousCategory;
                    // Ensure the previous prompt name is in the dropdown options (for unsaved prompts)
                    if (previousPrompt && !promptWidget.options.values.includes(previousPrompt)) {
                        promptWidget.options.values = [...promptWidget.options.values, previousPrompt].sort((a, b) => a.localeCompare(b));
                    }
                    promptWidget.value = previousPrompt;
                    app.graph.setDirtyCanvas(true, true);
                    return;
                }
            }

            // Clear new prompt flag when switching away
            node.isNewUnsavedPrompt = false;
            node.newPromptCategory = null;
            node.newPromptName = null;
        }

        if (originalCallback) {
            originalCallback.apply(this, arguments);
        }

        // Clear all lora and trigger word data BEFORE loading new prompt
        // This ensures stale toggle states don't persist (same as "New Prompt" button)
        node.savedLorasA = [];
        node.savedLorasB = [];
        node.currentLorasA = [];
        node.currentLorasB = [];
        node.originalLorasA = [];  // Clear original strengths too
        node.originalLorasB = [];
        node.savedTriggerWords = [];
        node.currentTriggerWords = [];
        node.lastKnownPromptKey = "";
        node.lastKnownInputLorasA = "";
        node.lastKnownInputLorasB = "";
        updateLoraDisplays(node);
        updateTriggerWordsDisplay(node);

        // Reload prompts from server to get latest changes from other tabs
        await loadPrompts(node);

        const category = value;
        if (node.prompts && node.prompts[category]) {
            const promptNames = Object.keys(node.prompts[category]);
            promptWidget.options.values = promptNames;

            if (promptNames.length > 0) {
                promptWidget.value = promptNames[0];
                await loadPromptData(node, category, promptNames[0]);
            } else {
                promptWidget.value = "";
                textWidget.value = "";
                // Clear ALL loras and trigger words when no prompts in category
                node.savedLorasA = [];
                node.savedLorasB = [];
                node.currentLorasA = [];
                node.currentLorasB = [];
                node.originalLorasA = [];
                node.originalLorasB = [];
                node.savedTriggerWords = [];
                node.currentTriggerWords = [];
                updateLoraDisplays(node);
                updateTriggerWordsDisplay(node);
                // Update last saved state
                updateLastSavedState(node);
            }

            // Update previous values after successful switch
            node._previousCategory = category;
            node._previousPrompt = promptWidget.value;

            // Update custom prompt selector display
            if (node.updatePromptSelectorDisplay) {
                node.updatePromptSelectorDisplay();
            }

            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
        }
    };

    const originalPromptCallback = promptWidget.callback;

    promptWidget.callback = async function(value) {
        const previousCategory = node._previousCategory;
        const previousPrompt = node._previousPrompt;

        // Check for unsaved changes before switching (skip if navigating via custom selector)
        if (!node._skipUnsavedCheck) {
            const hasUnsaved = hasUnsavedChanges(node);
            const warnEnabled = app.ui.settings.getSettingValue("PromptManagerAdvanced.warnUnsavedChanges", true);

            if (hasUnsaved && warnEnabled) {
                const confirmed = await showConfirm(
                    "Unsaved Changes",
                    "You have unsaved changes to the current prompt. Do you want to discard them and switch?",
                    "Discard & Switch",
                    "#f80"
                );
                if (!confirmed) {
                    // Ensure the previous prompt name is in the dropdown options (for unsaved prompts)
                    if (previousPrompt && !promptWidget.options.values.includes(previousPrompt)) {
                        promptWidget.options.values = [...promptWidget.options.values, previousPrompt].sort((a, b) => a.localeCompare(b));
                    }
                    // Revert the dropdown to previous value
                    promptWidget.value = previousPrompt;
                    app.graph.setDirtyCanvas(true, true);
                    return;
                }
            }

            // Clear new prompt flag when switching away
            node.isNewUnsavedPrompt = false;
            node.newPromptCategory = null;
            node.newPromptName = null;
        }

        if (originalPromptCallback) {
            originalPromptCallback.apply(this, arguments);
        }

        // Clear all lora and trigger word data BEFORE loading new prompt
        // This ensures stale toggle states don't persist (same as "New Prompt" button)
        node.savedLorasA = [];
        node.savedLorasB = [];
        node.currentLorasA = [];
        node.currentLorasB = [];
        node.originalLorasA = [];  // Clear original strengths too
        node.originalLorasB = [];
        node.savedTriggerWords = [];
        node.currentTriggerWords = [];
        node.lastKnownPromptKey = "";
        node.lastKnownInputLorasA = "";
        node.lastKnownInputLorasB = "";
        updateLoraDisplays(node);
        updateTriggerWordsDisplay(node);

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

        await loadPromptData(node, category, value);

        // Update previous values after successful switch
        node._previousCategory = category;
        node._previousPrompt = value;

        // Update custom prompt selector display
        if (node.updatePromptSelectorDisplay) {
            node.updatePromptSelectorDisplay();
        }

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
    const useExternalWidget = node.widgets?.find(w => w.name === "use_prompt_input");
    const useLoraInputWidget = node.widgets?.find(w => w.name === "use_lora_input");
    const categoryWidget = node.widgets?.find(w => w.name === "category");
    const promptWidget = node.widgets?.find(w => w.name === "name");

    if (!textWidget || !useExternalWidget) return;

    // Setup use_lora_input toggle handler to update lora display
    if (useLoraInputWidget) {
        const originalLoraInputCallback = useLoraInputWidget.callback;
        useLoraInputWidget.callback = async function(value) {
            if (originalLoraInputCallback) {
                originalLoraInputCallback.apply(this, arguments);
            }

            // Clear current (connected) loras and reload saved from prompt
            // This ensures toggled-off connected loras don't persist
            node.currentLorasA = [];
            node.currentLorasB = [];
            node.currentTriggerWords = [];

            // Reload saved loras from the current prompt to get clean state
            if (node.prompts) {
                const promptData = node.prompts[categoryWidget?.value]?.[promptWidget?.value];
                if (promptData) {
                    const lorasA = (promptData.loras_a || []).map(lora => ({
                        ...lora,
                        active: lora.active !== false,
                        strength: lora.strength ?? lora.model_strength ?? 1.0,
                        source: 'saved',
                        available: true  // Will be updated after check
                    }));
                    const lorasB = (promptData.loras_b || []).map(lora => ({
                        ...lora,
                        active: lora.active !== false,
                        strength: lora.strength ?? lora.model_strength ?? 1.0,
                        source: 'saved',
                        available: true  // Will be updated after check
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
                    // Store original strengths for reset functionality
                    node.originalLorasA = lorasA.map(lora => ({
                        name: lora.name,
                        strength: lora.strength ?? lora.model_strength ?? 1.0
                    }));
                    node.originalLorasB = lorasB.map(lora => ({
                        name: lora.name,
                        strength: lora.strength ?? lora.model_strength ?? 1.0
                    }));
                } else {
                    node.savedLorasA = [];
                    node.savedLorasB = [];
                    node.originalLorasA = [];
                    node.originalLorasB = [];
                }
            }

            // Update lora displays when toggle changes
            updateLoraDisplays(node);
        };
    }

    // Apply initial state on load/reload
    const applyToggleState = (value) => {
        const promptInputConnection = node.inputs?.find(inp => inp.name === "prompt_input");
        const isLlmConnected = promptInputConnection && promptInputConnection.link != null;

        if (value && isLlmConnected) {
            // Using LLM and it's connected - disable text widget immediately
            textWidget.disabled = true;
            // Keep scrolling enabled but prevent editing
            if (textWidget.inputEl) {
                textWidget.inputEl.style.pointerEvents = "auto";
                textWidget.inputEl.readOnly = true;
            }

            // Try to show prompt input value if available
            const graph = app.graph;
            const link = graph.links[promptInputConnection.link];
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
        // Check if prompt_input is connected
        const promptInputConnection = node.inputs?.find(inp => inp.name === "prompt_input");
        const isLlmConnected = promptInputConnection && promptInputConnection.link != null;

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

    // Always clear ALL loras and trigger words when switching prompts
    // This ensures clean state - current items will be repopulated on next execution
    node.savedLorasA = [];
    node.savedLorasB = [];
    node.currentLorasA = [];
    node.currentLorasB = [];
    node.originalLorasA = [];  // Clear original strengths too
    node.originalLorasB = [];
    node.savedTriggerWords = [];
    node.currentTriggerWords = [];

    // Reset tracking - forces a full reload on next execution
    node.lastKnownPromptKey = "";
    node.lastKnownInputLorasA = "";
    node.lastKnownInputLorasB = "";

    if (!node.prompts || !node.prompts[category] || !node.prompts[category][promptName]) {
        if (textWidget) textWidget.value = "";
        updateLoraDisplays(node);
        updateTriggerWordsDisplay(node);
        return;
    }

    const promptData = node.prompts[category][promptName];

    if (textWidget) {
        textWidget.value = promptData.prompt || "";
    }

    // Load saved loras - preserve active state (default true for backward compatibility)
    const lorasA = (promptData.loras_a || []).map(lora => ({
        ...lora,
        active: lora.active !== false, // Default to true if not specified
        strength: lora.strength ?? lora.model_strength ?? 1.0,
        available: true // Will be updated after check
    }));
    const lorasB = (promptData.loras_b || []).map(lora => ({
        ...lora,
        active: lora.active !== false, // Default to true if not specified
        strength: lora.strength ?? lora.model_strength ?? 1.0,
        available: true
    }));

    // Load saved trigger words - preserve their active state
    const triggerWords = (promptData.trigger_words || []).map(word => ({
        text: word.text,
        active: word.active !== false,
        source: 'saved'
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
                // Use explicit check - if not found in results OR false, mark unavailable
                lorasA.forEach(lora => {
                    lora.available = data.results[lora.name] === true;
                });
                lorasB.forEach(lora => {
                    lora.available = data.results[lora.name] === true;
                });
            }
        } catch (error) {
            console.error("[PromptManagerAdvanced] Error checking LoRA availability:", error);
        }
    }

    node.savedLorasA = lorasA;
    node.savedLorasB = lorasB;
    node.savedTriggerWords = triggerWords;

    // Store original strengths for reset functionality
    node.originalLorasA = lorasA.map(lora => ({
        name: lora.name,
        strength: lora.strength ?? lora.model_strength ?? 1.0
    }));
    node.originalLorasB = lorasB.map(lora => ({
        name: lora.name,
        strength: lora.strength ?? lora.model_strength ?? 1.0
    }));

    // Update last saved state for change detection
    updateLastSavedState(node);

    updateLoraDisplays(node);
    updateTriggerWordsDisplay(node);
}

// ========================
// Unsaved Changes Detection
// ========================

/**
 * Serialize the current state for comparison
 */
function getCurrentStateSnapshot(node) {
    const textWidget = node.widgets?.find(w => w.name === "text");
    const text = textWidget?.value || "";

    // Get all loras with their states
    const lorasA = (node.savedLorasA || [])
        .map(l => ({ name: l.name, strength: l.strength, active: l.active !== false }))
        .sort((a, b) => a.name.localeCompare(b.name));

    const lorasB = (node.savedLorasB || [])
        .map(l => ({ name: l.name, strength: l.strength, active: l.active !== false }))
        .sort((a, b) => a.name.localeCompare(b.name));

    // Get all trigger words with their states
    const triggerWords = (node.savedTriggerWords || [])
        .map(tw => ({ text: tw.text, active: tw.active !== false }))
        .sort((a, b) => a.text.localeCompare(b.text));

    return {
        text: text,
        lorasA: JSON.stringify(lorasA),
        lorasB: JSON.stringify(lorasB),
        triggerWords: JSON.stringify(triggerWords)
    };
}

/**
 * Update the last saved state (call after save or load)
 */
function updateLastSavedState(node) {
    node.lastSavedState = getCurrentStateSnapshot(node);
}

/**
 * Check if there are unsaved changes
 */
function hasUnsavedChanges(node) {
    // New unsaved prompt always has unsaved changes
    if (node.isNewUnsavedPrompt) {
        return true;
    }

    if (!node.lastSavedState) {
        return false;
    }

    const current = getCurrentStateSnapshot(node);

    return (
        current.text !== node.lastSavedState.text ||
        current.lorasA !== node.lastSavedState.lorasA ||
        current.lorasB !== node.lastSavedState.lorasB ||
        current.triggerWords !== node.lastSavedState.triggerWords
    );
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
            node.originalLorasA = [];
            node.originalLorasB = [];
            node.savedTriggerWords = [];
            node.currentTriggerWords = [];
            updateLoraDisplays(node);
            updateTriggerWordsDisplay(node);
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

async function savePrompt(node, category, name, text, lorasA, lorasB, triggerWords, thumbnail = null) {
    try {
        const requestBody = {
            category: category,
            name: name,
            text: text,
            // Save loras with their active state
            loras_a: lorasA.map(l => ({
                name: l.name,
                strength: l.strength ?? l.model_strength ?? 1.0,
                clip_strength: l.clip_strength || l.strength || 1.0,
                active: l.active !== false
            })),
            loras_b: lorasB.map(l => ({
                name: l.name,
                strength: l.strength ?? l.model_strength ?? 1.0,
                clip_strength: l.clip_strength || l.strength || 1.0,
                active: l.active !== false
            })),
            // Save all trigger words with their active states
            trigger_words: (triggerWords || []).map(tw => ({
                text: tw.text,
                active: tw.active !== false
            }))
        };

        // Include thumbnail if provided
        if (thumbnail) {
            requestBody.thumbnail = thumbnail;
        }

        const response = await fetch("/prompt-manager-advanced/save-prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody)
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
            node.savedTriggerWords = triggerWords || [];

            // Update original strengths to match saved values (Reset now resets to saved state)
            node.originalLorasA = lorasA.map(l => ({
                name: l.name,
                strength: l.strength ?? l.model_strength ?? 1.0
            }));
            node.originalLorasB = lorasB.map(l => ({
                name: l.name,
                strength: l.strength ?? l.model_strength ?? 1.0
            }));

            updateLoraDisplays(node);
            updateTriggerWordsDisplay(node);

            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
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
            node.originalLorasA = [];
            node.originalLorasB = [];
            node.savedTriggerWords = [];
            node.currentTriggerWords = [];
            updateDropdowns(node);
            updateLoraDisplays(node);
            updateTriggerWordsDisplay(node);

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
            node.originalLorasA = [];
            node.originalLorasB = [];
            node.savedTriggerWords = [];
            node.currentTriggerWords = [];
            updateDropdowns(node);
            updateLoraDisplays(node);
            updateTriggerWordsDisplay(node);

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
        const response = await fetch("/prompt-manager-advanced/get-prompts");
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
                console.log("[PromptManagerAdvanced] Save picker failed, falling back to download:", err);
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
        console.error("[PromptManagerAdvanced] Error exporting prompts:", error);
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
                const response = await fetch("/prompt-manager-advanced/import-prompts", {
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

                    // Reload the current prompt's data (loras and trigger words)
                    const categoryWidget = node.widgets.find(w => w.name === "category");
                    const promptWidget = node.widgets.find(w => w.name === "name");

                    if (categoryWidget && promptWidget && promptWidget.value) {
                        await loadPromptData(node, categoryWidget.value, promptWidget.value);
                    } else {
                        // Clear current state if no prompt selected
                        node.savedLorasA = [];
                        node.savedLorasB = [];
                        node.originalLorasA = [];
                        node.originalLorasB = [];
                        node.savedTriggerWords = [];
                        node.currentTriggerWords = [];
                        updateLoraDisplays(node);
                        updateTriggerWordsDisplay(node);
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
                console.error("[PromptManagerAdvanced] Error importing prompts:", error);
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

console.log("[PromptManagerAdvanced] Extension loaded");

// ========================
// Thumbnail Browser Functions
// ========================

// Default placeholder image - loaded from shared static PNG file
const DEFAULT_THUMBNAIL = new URL("./placeholder.png", import.meta.url).href;

/**
 * Resize an image to fit within maxSize while maintaining aspect ratio
 * Returns a base64 data URL
 */
function resizeImageToThumbnail(file, maxSize = 128) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Calculate new dimensions maintaining aspect ratio
                let width = img.width;
                let height = img.height;

                if (width > height) {
                    if (width > maxSize) {
                        height = Math.round((height * maxSize) / width);
                        width = maxSize;
                    }
                } else {
                    if (height > maxSize) {
                        width = Math.round((width * maxSize) / height);
                        height = maxSize;
                    }
                }

                // Create canvas and draw resized image
                const canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                // Convert to JPEG for smaller file size
                resolve(canvas.toDataURL('image/jpeg', 0.85));
            };
            img.onerror = reject;
            img.src = e.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

/**
 * Show thumbnail browser popup for selecting prompts
 * Returns { category, prompt } or null if cancelled
 */
function showThumbnailBrowser(node, currentCategory, currentPrompt) {
    return new Promise((resolve) => {
        let selectedCategory = currentCategory;

        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 9999;
        `;

        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #1a1a1a;
            border: 2px solid #444;
            border-radius: 12px;
            padding: 16px;
            z-index: 10000;
            width: 634px;
            max-height: 80vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        `;

        // Prevent default context menu on dialog (thumbnail cards have their own)
        dialog.addEventListener("contextmenu", (e) => e.preventDefault());

        // Header with close button
        const header = document.createElement("div");
        header.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #444;
        `;

        const title = document.createElement("div");
        title.textContent = "Select Prompt";
        title.style.cssText = `
            font-size: 18px;
            font-weight: bold;
            color: #fff;
        `;

        const closeBtn = document.createElement("button");
        closeBtn.textContent = "✕";
        closeBtn.style.cssText = `
            background: transparent;
            border: none;
            color: #888;
            font-size: 20px;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
        `;
        closeBtn.onmouseover = () => closeBtn.style.color = "#fff";
        closeBtn.onmouseout = () => closeBtn.style.color = "#888";

        header.appendChild(title);
        header.appendChild(closeBtn);

        // Category selector
        const categoryContainer = document.createElement("div");
        categoryContainer.style.cssText = `
            display: flex;
            gap: 6px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        `;

        const categories = Object.keys(node.prompts || {}).sort((a, b) => a.localeCompare(b));
        const categoryButtons = [];

        const updateCategoryButtons = () => {
            categoryButtons.forEach(btn => {
                const isSelected = btn.dataset.category === selectedCategory;
                btn.style.background = isSelected ? '#4a8ad4' : '#2a2a2a';
                btn.style.borderColor = isSelected ? '#5a9ae4' : '#444';
                btn.style.color = isSelected ? '#fff' : '#aaa';
            });
        };

        categories.forEach(cat => {
            const btn = document.createElement("button");
            btn.textContent = cat;
            btn.dataset.category = cat;
            btn.style.cssText = `
                padding: 6px 14px;
                border-radius: 6px;
                border: 1px solid #444;
                background: #2a2a2a;
                color: #aaa;
                cursor: pointer;
                font-size: 13px;
                transition: all 0.15s ease;
            `;
            btn.onclick = () => {
                selectedCategory = cat;
                updateCategoryButtons();
                createThumbnailCards(searchInput.value);
            };
            categoryButtons.push(btn);
            categoryContainer.appendChild(btn);
        });
        updateCategoryButtons();

        // Search bar
        const searchContainer = document.createElement("div");
        searchContainer.style.cssText = `
            margin-bottom: 12px;
        `;

        const searchInput = document.createElement("input");
        searchInput.type = "text";
        searchInput.placeholder = "Search prompts...";
        searchInput.style.cssText = `
            width: 100%;
            padding: 10px 12px;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 6px;
            color: #fff;
            font-size: 14px;
            box-sizing: border-box;
            outline: none;
        `;
        searchInput.onfocus = () => searchInput.style.borderColor = "#666";
        searchInput.onblur = () => searchInput.style.borderColor = "#444";

        searchContainer.appendChild(searchInput);

        // Thumbnails grid container - fixed size for 4x5 grid
        const gridContainer = document.createElement("div");
        gridContainer.className = "thumbnail-grid-container";
        gridContainer.style.cssText = `
            overflow-y: auto;
            height: 680px;
            scrollbar-width: none;
            -ms-overflow-style: none;
        `;
        // Hide scrollbar for webkit browsers
        const style = document.createElement("style");
        style.textContent = `.thumbnail-grid-container::-webkit-scrollbar { display: none; }`;
        document.head.appendChild(style);

        const grid = document.createElement("div");
        grid.style.cssText = `
            display: grid;
            grid-template-columns: repeat(4, 140px);
            gap: 12px;
            padding: 4px 0;
        `;

        // Create thumbnail cards for selected category
        const createThumbnailCards = (filter = "") => {
            grid.innerHTML = "";

            // Get prompts for selected category
            const categoryPrompts = node.prompts[selectedCategory] || {};
            const promptNames = Object.keys(categoryPrompts).sort((a, b) => a.localeCompare(b));

            const filteredPrompts = filter
                ? promptNames.filter(name => name.toLowerCase().includes(filter.toLowerCase()))
                : promptNames;

            if (filteredPrompts.length === 0) {
                const emptyMsg = document.createElement("div");
                emptyMsg.textContent = filter ? "No matching prompts found" : "No prompts in this category";
                emptyMsg.style.cssText = `
                    grid-column: 1 / -1;
                    text-align: center;
                    color: #666;
                    padding: 40px;
                    font-style: italic;
                `;
                grid.appendChild(emptyMsg);
                return;
            }

            filteredPrompts.forEach(promptName => {
                const promptData = categoryPrompts[promptName];
                const thumbnail = promptData?.thumbnail || DEFAULT_THUMBNAIL;
                const isSelected = promptName === currentPrompt;

                const card = document.createElement("div");
                card.style.cssText = `
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 8px;
                    background: ${isSelected ? '#2a4a6a' : '#2a2a2a'};
                    border: 2px solid ${isSelected ? '#4a8ad4' : '#3a3a3a'};
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.15s ease;
                `;

                card.onmouseover = () => {
                    if (!isSelected) {
                        card.style.background = '#3a3a3a';
                        card.style.borderColor = '#555';
                    }
                };
                card.onmouseout = () => {
                    if (!isSelected) {
                        card.style.background = '#2a2a2a';
                        card.style.borderColor = '#3a3a3a';
                    }
                };

                // Thumbnail image
                const img = document.createElement("img");
                img.src = thumbnail;
                img.style.cssText = `
                    width: 100px;
                    height: 100px;
                    object-fit: cover;
                    border-radius: 6px;
                    background: #1a1a1a;
                `;

                // Prompt name
                const nameLabel = document.createElement("div");
                nameLabel.textContent = promptName;
                nameLabel.title = promptName;
                nameLabel.style.cssText = `
                    margin-top: 8px;
                    font-size: 12px;
                    color: #ccc;
                    text-align: center;
                    width: 100%;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                `;

                card.appendChild(img);
                card.appendChild(nameLabel);

                // Click to select - return both category and prompt
                card.onclick = () => {
                    resolve({ category: selectedCategory, prompt: promptName });
                    cleanup();
                };

                // Right-click context menu for setting thumbnail
                card.oncontextmenu = (e) => {
                    e.preventDefault();
                    showThumbnailContextMenu(e, node, selectedCategory, promptName, () => {
                        // Refresh the grid after thumbnail change
                        createThumbnailCards(searchInput.value);
                    });
                };

                grid.appendChild(card);
            });
        };

        // Initial render
        createThumbnailCards();

        // Search filtering
        searchInput.oninput = () => {
            createThumbnailCards(searchInput.value);
        };

        gridContainer.appendChild(grid);

        // Footer with hint
        const footer = document.createElement("div");
        footer.style.cssText = `
            margin-top: 4px;
            margin-bottom: -8px;
            padding-top: 6px;
            border-top: 1px solid #444;
            font-size: 11px;
            color: #666;
            text-align: center;
        `;
        footer.textContent = "Right-click a prompt to set or remove its thumbnail";

        dialog.appendChild(header);
        dialog.appendChild(categoryContainer);
        dialog.appendChild(searchContainer);
        dialog.appendChild(gridContainer);
        dialog.appendChild(footer);

        const cleanup = () => {
            document.body.removeChild(overlay);
            document.body.removeChild(dialog);
        };

        closeBtn.onclick = () => {
            resolve(null);
            cleanup();
        };

        overlay.onclick = () => {
            resolve(null);
            cleanup();
        };

        // Prevent dialog click from closing
        dialog.onclick = (e) => e.stopPropagation();

        // Keyboard shortcuts
        dialog.onkeydown = (e) => {
            if (e.key === "Escape") {
                resolve(null);
                cleanup();
            }
        };

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        searchInput.focus();
    });
}

/**
 * Show context menu for thumbnail operations
 */
function showThumbnailContextMenu(event, node, category, promptName, onUpdate) {
    // Remove any existing context menu
    const existing = document.querySelector('.thumbnail-context-menu');
    if (existing) existing.remove();

    const menu = document.createElement("div");
    menu.className = "thumbnail-context-menu";
    menu.style.cssText = `
        position: fixed;
        left: ${event.clientX}px;
        top: ${event.clientY}px;
        background: #2a2a2a;
        border: 1px solid #444;
        border-radius: 6px;
        padding: 4px 0;
        z-index: 10001;
        min-width: 150px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    `;

    const createMenuItem = (label, onClick) => {
        const item = document.createElement("div");
        item.textContent = label;
        item.style.cssText = `
            padding: 8px 16px;
            color: #ccc;
            cursor: pointer;
            font-size: 13px;
        `;
        item.onmouseover = () => item.style.background = '#3a3a3a';
        item.onmouseout = () => item.style.background = 'transparent';
        item.onclick = () => {
            menu.remove();
            onClick();
        };
        return item;
    };

    // Set thumbnail from file
    menu.appendChild(createMenuItem("📁 Set Thumbnail from File...", async () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (file) {
                try {
                    const thumbnail = await resizeImageToThumbnail(file, 128);
                    await saveThumbnail(node, category, promptName, thumbnail);
                    onUpdate();
                } catch (error) {
                    console.error("[PromptManagerAdvanced] Error setting thumbnail:", error);
                    await showInfo("Error", "Failed to set thumbnail");
                }
            }
        };
        input.click();
    }));

    // Set thumbnail from clipboard
    menu.appendChild(createMenuItem("📋 Set Thumbnail from Clipboard", async () => {
        try {
            const clipboardItems = await navigator.clipboard.read();
            for (const item of clipboardItems) {
                const imageType = item.types.find(type => type.startsWith('image/'));
                if (imageType) {
                    const blob = await item.getType(imageType);
                    const file = new File([blob], 'clipboard.png', { type: imageType });
                    const thumbnail = await resizeImageToThumbnail(file, 128);
                    await saveThumbnail(node, category, promptName, thumbnail);
                    onUpdate();
                    return;
                }
            }
            await showInfo("No Image", "No image found in clipboard");
        } catch (error) {
            console.error("[PromptManagerAdvanced] Error reading clipboard:", error);
            await showInfo("Error", "Failed to read clipboard. Make sure you have an image copied.");
        }
    }));

    // Remove thumbnail (only show if there is one)
    const promptData = node.prompts[category]?.[promptName];
    if (promptData?.thumbnail) {
        const divider = document.createElement("div");
        divider.style.cssText = `
            height: 1px;
            background: #444;
            margin: 4px 0;
        `;
        menu.appendChild(divider);

        menu.appendChild(createMenuItem("🗑️ Remove Thumbnail", async () => {
            await saveThumbnail(node, category, promptName, null);
            onUpdate();
        }));
    }

    // Close menu when clicking outside of it
    const closeMenu = (e) => {
        // Only close if clicking outside the menu
        if (!menu.contains(e.target)) {
            menu.remove();
            document.removeEventListener('mousedown', closeMenu, true);
        }
    };
    // Use mousedown with capture to catch clicks before they propagate
    setTimeout(() => {
        document.addEventListener('mousedown', closeMenu, true);
    }, 10);

    document.body.appendChild(menu);
}

/**
 * Save thumbnail to prompt data
 */
async function saveThumbnail(node, category, promptName, thumbnail) {
    try {
        const response = await fetch("/prompt-manager-advanced/save-thumbnail", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                category: category,
                name: promptName,
                thumbnail: thumbnail
            })
        });

        const data = await response.json();

        if (data.success) {
            // Update local data
            if (node.prompts[category] && node.prompts[category][promptName]) {
                if (thumbnail) {
                    node.prompts[category][promptName].thumbnail = thumbnail;
                } else {
                    delete node.prompts[category][promptName].thumbnail;
                }
            }
        } else {
            await showInfo("Error", data.error || "Failed to save thumbnail");
        }
    } catch (error) {
        console.error("[PromptManagerAdvanced] Error saving thumbnail:", error);
        await showInfo("Error", "Failed to save thumbnail");
    }
}

/**
 * Create custom prompt selector widget with arrows and thumbnail browser
 */
function createPromptSelectorWidget(node) {
    const categoryWidget = node.widgets.find(w => w.name === "category");
    const promptWidget = node.widgets.find(w => w.name === "name");

    if (!categoryWidget || !promptWidget) return;

    // Hide the original category and prompt widgets visually but keep them functional
    categoryWidget.type = "converted-widget";
    categoryWidget.computeSize = () => [0, -4];
    promptWidget.type = "converted-widget";
    promptWidget.computeSize = () => [0, -4];

    // Create custom selector container
    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        align-items: center;
        gap: 0;
        background: #1a1a1a;
        border-radius: 4px;
        overflow: visible;
        height: 26px;
        margin: 0;
        position: relative;
    `;

    // Prevent default context menu on prompt selector
    container.addEventListener("contextmenu", (e) => e.preventDefault());

    // Forward wheel events to canvas for zooming
    forwardWheelToCanvas(container);

    // Left arrow button
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
    leftArrow.onmouseover = () => {
        leftArrow.style.background = '#3a3a3a';
        leftArrow.style.color = '#fff';
    };
    leftArrow.onmouseout = () => {
        leftArrow.style.background = '#2a2a2a';
        leftArrow.style.color = '#888';
    };

    // Center name display (clickable to open browser)
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

    // Thumbnail preview tooltip
    const thumbnailPreview = document.createElement("div");
    thumbnailPreview.style.cssText = `
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: #2a2a2a;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 8px;
        display: none;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        margin-bottom: 8px;
    `;
    const thumbnailImg = document.createElement("img");
    thumbnailImg.style.cssText = `
        width: 128px;
        height: 128px;
        object-fit: cover;
        border-radius: 4px;
        display: block;
    `;
    thumbnailPreview.appendChild(thumbnailImg);
    container.appendChild(thumbnailPreview);

    // Show/hide thumbnail on hover
    let hoverTimeout = null;
    nameDisplay.addEventListener("mouseenter", () => {
        hoverTimeout = setTimeout(() => {
            const category = categoryWidget.value;
            const prompt = promptWidget.value;
            const promptData = node.prompts?.[category]?.[prompt];
            const thumbnail = promptData?.thumbnail || DEFAULT_THUMBNAIL;
            thumbnailImg.src = thumbnail;
            thumbnailPreview.style.display = "block";
        }, 300);
    });
    nameDisplay.addEventListener("mouseleave", () => {
        if (hoverTimeout) clearTimeout(hoverTimeout);
        thumbnailPreview.style.display = "none";
    });

    // Right arrow button
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
    rightArrow.onmouseover = () => {
        rightArrow.style.background = '#3a3a3a';
        rightArrow.style.color = '#fff';
    };
    rightArrow.onmouseout = () => {
        rightArrow.style.background = '#2a2a2a';
        rightArrow.style.color = '#888';
    };

    container.appendChild(leftArrow);
    container.appendChild(nameDisplay);
    container.appendChild(rightArrow);

    // Update display function - show CATEGORY : PROMPT
    const updateDisplay = () => {
        const category = categoryWidget.value || "";
        const prompt = promptWidget.value || "new prompt";
        nameDisplay.textContent = `${category} : ${prompt}`;
        nameDisplay.title = `${category} : ${prompt}`;
    };

    // Get flattened list of all prompts across all categories for navigation
    const getAllPromptsFlat = () => {
        const allPrompts = [];
        if (!node.prompts) return allPrompts;

        const categories = Object.keys(node.prompts).sort((a, b) => a.localeCompare(b));
        for (const cat of categories) {
            const prompts = Object.keys(node.prompts[cat]).sort((a, b) => a.localeCompare(b));
            for (const prompt of prompts) {
                allPrompts.push({ category: cat, prompt: prompt });
            }
        }
        return allPrompts;
    };

    // Find current position in flattened list
    const getCurrentIndex = (allPrompts) => {
        return allPrompts.findIndex(p =>
            p.category === categoryWidget.value && p.prompt === promptWidget.value
        );
    };

    // Navigate to a specific prompt (handles category change)
    const navigateTo = async (item, skipUnsavedCheck = false) => {
        // Check for unsaved changes before switching (unless skipped)
        if (!skipUnsavedCheck) {
            const hasUnsaved = hasUnsavedChanges(node);
            const warnEnabled = app.ui.settings.getSettingValue("PromptManagerAdvanced.warnUnsavedChanges", true);

            if (hasUnsaved && warnEnabled) {
                const confirmed = await showConfirm(
                    "Unsaved Changes",
                    "You have unsaved changes to the current prompt. Do you want to discard them and switch?",
                    "Discard & Switch",
                    "#f80"
                );
                if (!confirmed) {
                    return false;  // User cancelled, don't navigate
                }
            }

            // Clear new prompt flag when switching away
            node.isNewUnsavedPrompt = false;
            node.newPromptCategory = null;
            node.newPromptName = null;
        }

        // Set flag to skip unsaved check in the widget callbacks (we already checked above)
        node._skipUnsavedCheck = true;

        try {
            const categoryChanged = item.category !== categoryWidget.value;

            if (categoryChanged) {
                // Change category first
                categoryWidget.value = item.category;
                if (categoryWidget.callback) {
                    await categoryWidget.callback(item.category);
                }
            }

            // Then change prompt
            if (promptWidget.callback) {
                await promptWidget.callback(item.prompt);
            }
            promptWidget.value = item.prompt;
            updateDisplay();
            app.graph.setDirtyCanvas(true, true);
        } finally {
            // Clear the skip flag
            node._skipUnsavedCheck = false;
        }

        return true;
    };

    // Navigate to previous prompt (wraps across categories)
    leftArrow.onclick = async (e) => {
        e.stopPropagation();
        const allPrompts = getAllPromptsFlat();
        if (allPrompts.length === 0) return;

        const currentIndex = getCurrentIndex(allPrompts);
        const newIndex = currentIndex <= 0 ? allPrompts.length - 1 : currentIndex - 1;
        await navigateTo(allPrompts[newIndex]);
    };

    // Navigate to next prompt (wraps across categories)
    rightArrow.onclick = async (e) => {
        e.stopPropagation();
        const allPrompts = getAllPromptsFlat();
        if (allPrompts.length === 0) return;

        const currentIndex = getCurrentIndex(allPrompts);
        const newIndex = currentIndex >= allPrompts.length - 1 ? 0 : currentIndex + 1;
        await navigateTo(allPrompts[newIndex]);
    };

    // Open thumbnail browser on click
    nameDisplay.onclick = async (e) => {
        e.stopPropagation();

        // Check for unsaved changes before opening browser
        const hasUnsaved = hasUnsavedChanges(node);
        const warnEnabled = app.ui.settings.getSettingValue("PromptManagerAdvanced.warnUnsavedChanges", true);

        if (hasUnsaved && warnEnabled) {
            const confirmed = await showConfirm(
                "Unsaved Changes",
                "You have unsaved changes to the current prompt. Do you want to discard them and browse prompts?",
                "Discard & Browse",
                "#f80"
            );
            if (!confirmed) {
                return;  // User cancelled, don't open browser
            }
        }

        const category = categoryWidget.value;
        const currentPrompt = promptWidget.value;

        const selection = await showThumbnailBrowser(node, category, currentPrompt);

        if (selection) {
            // Navigate to the selected category/prompt (skip unsaved check since we already confirmed)
            await navigateTo(selection, true);

            // Clear new prompt flag after successful navigation
            node.isNewUnsavedPrompt = false;
            node.newPromptCategory = null;
            node.newPromptName = null;
        }
    };

    // Initial display update
    updateDisplay();

    // Add DOM widget
    const widget = node.addDOMWidget("prompt_selector", "div", container);
    widget.computeSize = function(width) {
        return [width, 28];
    };

    // Store reference for updates
    node.promptSelectorWidget = widget;
    node.updatePromptSelectorDisplay = updateDisplay;

    return widget;
}
