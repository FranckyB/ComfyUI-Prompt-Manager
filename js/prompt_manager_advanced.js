import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { PM_UI_PALETTE as UI } from "./ui_palette.js";

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

// ========================
// Session State for NSFW & View Mode & Thumbnail Preview
// ========================
// These persist during a working session but reset to preferences on ComfyUI restart
let sessionHideNSFW = null;   // null = use preference default
let sessionViewMode = null;   // null = use preference default
let sessionEnableThumbnailPreview = null;  // null = use localStorage default

function getHideNSFW() {
    if (sessionHideNSFW !== null) return sessionHideNSFW;
    return app.ui.settings.getSettingValue("PromptManager.DefaultHideNSFW");
}

function getViewMode() {
    if (sessionViewMode !== null) return sessionViewMode;
    return app.ui.settings.getSettingValue("PromptManager.DefaultViewMode");
}

function getThumbnailPreviewEnabled() {
    if (sessionEnableThumbnailPreview !== null) return sessionEnableThumbnailPreview;
    // Try settings API first, fall back to localStorage for backward compatibility
    const settingValue = app.ui.settings.getSettingValue("PromptManager.EnableThumbnailPreview");
    if (settingValue !== undefined) return settingValue;
    const stored = localStorage.getItem("PromptManager.EnableThumbnailPreview");
    return stored !== null ? stored === "true" : true; // Default to true
}

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
        if (nodeData.name === "PromptManagerAdvanced" || nodeData.name === "RecipeManager") {
            const isWorkflowManagerNode = nodeData.name === "RecipeManager";
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                const node = this;
                node._isWorkflowManager = isWorkflowManagerNode;
                enforceWorkflowManagerCompactMode(node);
                node.prompts = {};
                node.currentLorasA = [];
                node.currentLorasB = [];
                node.savedLorasA = [];
                node.savedLorasB = [];
                node.originalStrengthsA = {};  // Map of lora_name -> original_strength (from Python)
                node.originalStrengthsB = {};  // Map of lora_name -> original_strength (from Python)
                node.currentTriggerWords = [];  // From connected input
                node.savedTriggerWords = [];    // From saved prompt
                node.connectedThumbnail = null; // Thumbnail from connected image (set during execution)

                // Track last saved state for unsaved changes detection
                node.lastSavedState = {
                    text: "",
                    lorasA: "[]",
                    lorasB: "[]",
                    triggerWords: "[]"
                };

                // Set initial size.
                // Workflow Manager is compact-only, so keep a smaller default height.
                this.setSize([440, node._isWorkflowManager ? 420 : 700]);

                // Change widget labels
                const promptTextWidget = this.widgets.find(w => w.name === "text");
                if (promptTextWidget) {
                    promptTextWidget.label = "prompt";
                    if (node._isWorkflowManager) {
                        promptTextWidget.type = "converted-widget";
                        promptTextWidget.computeSize = () => [0, -4];
                    }
                }

                const promptNameWidget = this.widgets.find(w => w.name === "name");
                if (promptNameWidget) {
                    promptNameWidget.label = "name";
                }

                const useExternalWidget = this.widgets.find(w => w.name === "use_prompt_input");
                if (useExternalWidget) {
                    useExternalWidget.label = "use prompt input";
                }

                const useWorkflowWidget = this.widgets.find(w => w.name === "use_workflow_data");
                if (useWorkflowWidget) {
                    useWorkflowWidget.label = "use workflow input";
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
                api.addEventListener("prompt-manager-advanced-update", async (event) => {
                    if (String(event.detail.node_id) === String(this.id)) {
                        const normalizeLoraForSig = (lora) => ({
                            name: String(lora?.name || ""),
                            strength: Number(lora?.strength ?? lora?.model_strength ?? 1.0) || 1.0,
                            model_strength: Number(lora?.model_strength ?? lora?.strength ?? 1.0) || 1.0,
                            clip_strength: Number(lora?.clip_strength ?? lora?.strength ?? lora?.model_strength ?? 1.0) || 1.0,
                            active: lora?.active !== false,
                            available: lora?.available !== false,
                            found: lora?.found !== false,
                        });
                        const normalizedLoraListSig = (list) => JSON.stringify(
                            (Array.isArray(list) ? list : [])
                                .map(normalizeLoraForSig)
                                .sort((a, b) => a.name.localeCompare(b.name))
                        );

                        let newLorasA = event.detail.loras_a || [];
                        let newLorasB = event.detail.loras_b || [];
                        const newTriggerWords = event.detail.trigger_words || [];
                        const inputLorasA = event.detail.input_loras_a || [];
                        const inputLorasB = event.detail.input_loras_b || [];
                        const wfDataEvent = event.detail.workflow_data || null;
                        const useWorkflowEvent = event.detail.use_workflow_data === true;
                        const workflowInput = this.inputs?.find((inp) => inp.name === "recipe_data");
                        const hasWorkflowInputConnected = workflowInput?.link != null;
                        const shouldIngestWorkflowExecution = useWorkflowEvent && !this._isWorkflowManager;

                        if (shouldIngestWorkflowExecution) {
                            newLorasA = await applyLoraFoundState(newLorasA);
                            newLorasB = await applyLoraFoundState(newLorasB);
                        }

                        // Skip expensive state churn and canvas dirtying when payload is unchanged.
                        // Include found/available in the signature so availability refreshes still apply.
                        const incomingSig = JSON.stringify({
                            use_prompt_input: event.detail.use_prompt_input === true,
                            use_workflow_data: shouldIngestWorkflowExecution,
                            use_lora_input: (this.widgets?.find(w => w.name === "use_lora_input")?.value !== false),
                            prompt_input: String(event.detail.prompt_input || ""),
                            workflow_prompt: String(wfDataEvent?.positive_prompt || ""),
                            connected_thumbnail_sig: (() => {
                                const t = String(event.detail.connected_thumbnail || "");
                                return t ? `${t.length}:${t.slice(0, 64)}` : "";
                            })(),
                            workflow_data_sig: wfDataEvent ? JSON.stringify({
                                positive_prompt: String(wfDataEvent.positive_prompt || ""),
                                loras_a: (Array.isArray(wfDataEvent.loras_a) ? wfDataEvent.loras_a : []).map(normalizeLoraForSig).sort((a, b) => a.name.localeCompare(b.name)),
                                loras_b: (Array.isArray(wfDataEvent.loras_b) ? wfDataEvent.loras_b : []).map(normalizeLoraForSig).sort((a, b) => a.name.localeCompare(b.name)),
                            }) : null,
                            loras_a_sig: normalizedLoraListSig(newLorasA),
                            loras_b_sig: normalizedLoraListSig(newLorasB),
                            input_loras_a_sig: normalizedLoraListSig(inputLorasA),
                            input_loras_b_sig: normalizedLoraListSig(inputLorasB),
                            trigger_words_sig: JSON.stringify(newTriggerWords || []),
                            unavailable_a_sig: JSON.stringify((event.detail.unavailable_loras_a || []).slice().sort()),
                            unavailable_b_sig: JSON.stringify((event.detail.unavailable_loras_b || []).slice().sort()),
                        });

                        if (this._lastExecutionUpdateSig === incomingSig) {
                            return;
                        }
                        this._lastExecutionUpdateSig = incomingSig;

                        const effectiveInputLorasA = shouldIngestWorkflowExecution ? newLorasA : inputLorasA;
                        const effectiveInputLorasB = shouldIngestWorkflowExecution ? newLorasB : inputLorasB;
                        // Explicit list of unavailable lora names from Python
                        const unavailableLorasA = new Set((event.detail.unavailable_loras_a || []).map(n => n.toLowerCase()));
                        const unavailableLorasB = new Set((event.detail.unavailable_loras_b || []).map(n => n.toLowerCase()));
                        
                        // Python decides if we should reset - JavaScript just obeys
                        let shouldReset = event.detail.should_reset || false;
                        
                        // CRITICAL: Verify JavaScript state matches Python's truth
                        // If what we're displaying doesn't match what Python sent, force a reset
                        if (!shouldReset) {
                            // Check if use_lora_input is enabled to know what should be displayed
                            const useLoraInputWidget = this.widgets?.find(w => w.name === "use_lora_input");
                            const useLoraInput = useLoraInputWidget?.value !== false;
                            const currentSigA = useLoraInput
                                ? normalizedLoraListSig(mergeLoraLists(this.currentLorasA, this.savedLorasA))
                                : normalizedLoraListSig(this.savedLorasA || []);
                            const currentSigB = useLoraInput
                                ? normalizedLoraListSig(mergeLoraLists(this.currentLorasB, this.savedLorasB))
                                : normalizedLoraListSig(this.savedLorasB || []);
                            const pythonSigA = normalizedLoraListSig(newLorasA || []);
                            const pythonSigB = normalizedLoraListSig(newLorasB || []);
                            
                            // Get what JavaScript would currently display
                            let currentDisplayA, currentDisplayB;
                            if (!useLoraInput) {
                                // Only saved loras
                                currentDisplayA = (this.savedLorasA || []).map(l => l.name.toLowerCase()).sort();
                                currentDisplayB = (this.savedLorasB || []).map(l => l.name.toLowerCase()).sort();
                            } else {
                                // Merge of current + saved
                                const mergedA = mergeLoraLists(this.currentLorasA, this.savedLorasA);
                                const mergedB = mergeLoraLists(this.currentLorasB, this.savedLorasB);
                                currentDisplayA = mergedA.map(l => l.name.toLowerCase()).sort();
                                currentDisplayB = mergedB.map(l => l.name.toLowerCase()).sort();
                            }
                            
                            // Get what Python says should be displayed
                            const pythonLorasA = newLorasA.map(l => l.name.toLowerCase()).sort();
                            const pythonLorasB = newLorasB.map(l => l.name.toLowerCase()).sort();
                            
                            // If they don't match, we're out of sync - force reset
                            const lorasOutOfSyncA = JSON.stringify(currentDisplayA) !== JSON.stringify(pythonLorasA);
                            const lorasOutOfSyncB = JSON.stringify(currentDisplayB) !== JSON.stringify(pythonLorasB);
                            const lorasStateOutOfSyncA = currentSigA !== pythonSigA;
                            const lorasStateOutOfSyncB = currentSigB !== pythonSigB;
                            
                            if (lorasOutOfSyncA || lorasOutOfSyncB || lorasStateOutOfSyncA || lorasStateOutOfSyncB) {
                                console.log("[PromptManagerAdvanced] JavaScript state out of sync with Python - forcing reset");
                                console.log("  Current A:", currentDisplayA);
                                console.log("  Python A:", pythonLorasA);
                                console.log("  Current B:", currentDisplayB);
                                console.log("  Python B:", pythonLorasB);
                                shouldReset = true;
                            }
                        }
                        
                        // Store original strengths from Python (for Reset Strength button)
                        this.originalStrengthsA = event.detail.original_strengths_a || {};
                        this.originalStrengthsB = event.detail.original_strengths_b || {};

                        // Store connected thumbnail for use when saving
                        this.connectedThumbnail = event.detail.connected_thumbnail || null;

                        // Get current prompt key from widgets
                        const categoryWidget = this.widgets?.find(w => w.name === "category");
                        const promptWidget = this.widgets?.find(w => w.name === "name");

                        console.log("[PromptManagerAdvanced] Update received, shouldReset:", shouldReset);


                        // Workflow mode: execution payload is authoritative (single ingest path).
                        if (shouldIngestWorkflowExecution) {
                            syncWorkflowLorasForDisplay(this, wfDataEvent, newLorasA, newLorasB, { preserveUserState: false });
                            this.currentTriggerWords = [];
                            updateLoraDisplays(this);
                            updateTriggerWordsDisplay(this);
                        }
                        // Clear and reload when Python tells us to (or when we detect desync)
                        else if (shouldReset) {
                            console.log("[PromptManagerAdvanced] Resetting toggles as instructed by Python");

                            // Clear ALL loras and trigger words - we'll reload fresh
                            this.currentLorasA = [];
                            this.currentLorasB = [];
                            this.savedLorasA = [];
                            this.savedLorasB = [];
                            this.currentTriggerWords = [];
                            this.savedTriggerWords = [];

                            // Restore saved state from serialized widgets first.
                            // These are updated immediately when the user toggles items.
                            this.savedLorasA = getSerializedSavedLoras(this, "a", unavailableLorasA);
                            this.savedLorasB = getSerializedSavedLoras(this, "b", unavailableLorasB);
                            this.savedTriggerWords = getSerializedSavedTriggerWords(this);

                            // Filter out saved LoRAs that came from input but are no longer in the input.
                            // This prevents LoRAs removed from connected stacker from lingering in saved state.
                            const inputLoraSetA = new Set(effectiveInputLorasA.map(l => l.name.toLowerCase()));
                            const inputLoraSetB = new Set(effectiveInputLorasB.map(l => l.name.toLowerCase()));
                            this.savedLorasA = this.savedLorasA.filter(lora => 
                                !lora.fromInput || inputLoraSetA.has(lora.name.toLowerCase())
                            );
                            this.savedLorasB = this.savedLorasB.filter(lora => 
                                !lora.fromInput || inputLoraSetB.has(lora.name.toLowerCase())
                            );

                            // Fall back to cached prompt data only when there is no serialized state.
                            // Use Python's unavailable list directly - it's the source of truth.
                            if (
                                this.savedLorasA.length === 0 &&
                                this.savedLorasB.length === 0 &&
                                this.savedTriggerWords.length === 0 &&
                                this.prompts &&
                                categoryWidget &&
                                promptWidget
                            ) {
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
                                    // Reload saved trigger words with their saved active states
                                    this.savedTriggerWords = (promptData.trigger_words || []).map(word => ({
                                        text: word.text,
                                        active: word.active !== false,
                                        source: 'saved'
                                    }));
                                }
                            }

                            // Set current loras from input (these are the loras from connected nodes)
                            this.currentLorasA = effectiveInputLorasA.map(l => ({ ...l, source: shouldIngestWorkflowExecution ? 'workflow' : 'current' }));
                            this.currentLorasB = effectiveInputLorasB.map(l => ({ ...l, source: shouldIngestWorkflowExecution ? 'workflow' : 'current' }));

                            // Set current trigger words from connected input
                            const newConnectedTriggers = newTriggerWords.filter(t => t.source === 'connected');
                            this.currentTriggerWords = newConnectedTriggers;

                            // Refresh displays
                            updateLoraDisplays(this);
                            updateTriggerWordsDisplay(this);
                        } else {
                            // No major change - just update current loras if they differ
                            const lorasAChanged = normalizedLoraListSig(effectiveInputLorasA) !== normalizedLoraListSig(this.currentLorasA || []);
                            const lorasBChanged = normalizedLoraListSig(effectiveInputLorasB) !== normalizedLoraListSig(this.currentLorasB || []);
                            const newConnectedTriggers = newTriggerWords.filter(t => t.source === 'connected');
                            const triggerWordsChanged = JSON.stringify(newConnectedTriggers) !== JSON.stringify(this.currentTriggerWords);

                            if (lorasAChanged || lorasBChanged || triggerWordsChanged) {
                                this.currentLorasA = effectiveInputLorasA.map(l => ({ ...l, source: shouldIngestWorkflowExecution ? 'workflow' : 'current' }));
                                this.currentLorasB = effectiveInputLorasB.map(l => ({ ...l, source: shouldIngestWorkflowExecution ? 'workflow' : 'current' }));
                                this.currentTriggerWords = newConnectedTriggers;

                                // Filter out saved LoRAs that came from input but are no longer present
                                const inputLoraSetA = new Set(inputLorasA.map(l => l.name.toLowerCase()));
                                const inputLoraSetB = new Set(inputLorasB.map(l => l.name.toLowerCase()));
                                this.savedLorasA = (this.savedLorasA || []).filter(lora => 
                                    !lora.fromInput || inputLoraSetA.has(lora.name.toLowerCase())
                                );
                                this.savedLorasB = (this.savedLorasB || []).filter(lora => 
                                    !lora.fromInput || inputLoraSetB.has(lora.name.toLowerCase())
                                );

                                updateLoraDisplays(this);
                                updateTriggerWordsDisplay(this);
                            }
                        }

                        // Handle use_prompt_input / use_workflow_data toggle state for text widget
                        const promptTextWidget = this.widgets.find(w => w.name === "text");
                        if (promptTextWidget) {
                            const useExternal = event.detail.use_prompt_input || false;
                            const useWorkflow = shouldIngestWorkflowExecution;
                            const llmInput = event.detail.prompt_input || "";
                            const wfData = event.detail.workflow_data || null;

                            // Store workflow_data on node for saving
                            this.lastWorkflowData = wfData;
                            // For Workflow Manager in input-connected mode, keep incoming payload
                            // for preview/save UX but don't overwrite local serialized workflow state.
                            const isPmaWorkflowConnected = (!this._isWorkflowManager) && hasWorkflowInputConnected && useWorkflowEvent;
                            if (!(this._isWorkflowManager && hasWorkflowInputConnected) && !isPmaWorkflowConnected) {
                                syncSavedWorkflowDataWidget(this);
                            }
                            if (this._isWorkflowManager) {
                                updateWorkflowManagerPreview(this);
                                if (this.updatePromptSelectorDisplay) {
                                    this.updatePromptSelectorDisplay();
                                }
                            }

                            if (useExternal && llmInput) {
                                // Using external prompt — display the input text (grayed out)
                                promptTextWidget.value = llmInput;
                                promptTextWidget.disabled = true;
                                if (promptTextWidget.inputEl) {
                                    promptTextWidget.inputEl.style.pointerEvents = "auto";
                                    promptTextWidget.inputEl.readOnly = true;
                                }
                            } else if (useWorkflow && wfData && wfData.positive_prompt) {
                                // Using workflow_data prompt — display it (grayed out)
                                promptTextWidget.value = wfData.positive_prompt;
                                promptTextWidget.disabled = true;
                                if (promptTextWidget.inputEl) {
                                    promptTextWidget.inputEl.style.pointerEvents = "auto";
                                    promptTextWidget.inputEl.readOnly = true;
                                }
                            } else {
                                // Using internal text — enable the widget
                                promptTextWidget.disabled = false;
                                if (promptTextWidget.inputEl) {
                                    promptTextWidget.inputEl.readOnly = false;
                                }
                            }

                            this.serialize_widgets = true;
                            app.graph.setDirtyCanvas(true, true);
                        }
                    }
                });

                // IMPORTANT: Add DOM widgets SYNCHRONOUSLY during node creation
                // to ensure proper positioning within the node bounds
                if (node._isWorkflowManager) {
                    addWorkflowManagerPreview(node);
                }
                createPromptSelectorWidget(node);  // Custom thumbnail selector (before buttons)
                addButtonBar(node);
                if (!node._isWorkflowManager) {
                    addLoraDisplays(node);
                }
                if (!node._isWorkflowManager) {
                    addTriggerWordsDisplay(node);
                }
                setupCategoryChangeHandler(node);
                if (!node._isWorkflowManager) {
                    setupUseExternalToggleHandler(node);
                    setupUseWorkflowToggleHandler(node);
                }
                setupWorkflowLivePickupHandler(node);

                // Load prompts asynchronously (data only, not widgets)
                loadPrompts(node).then(() => {
                    // Preserve serialized selection during initial async hydrate so
                    // archived workflow names are not replaced by filtered library values.
                    filterPromptDropdown(node, { preserveDanglingSelection: true });

                    // Update custom prompt selector display
                    if (node.updatePromptSelectorDisplay) {
                        node.updatePromptSelectorDisplay();
                    }

                    // Only load prompt data from file if this is a fresh node creation.
                    // If onConfigure has already run (workflow restore / page reload / tab switch),
                    // the node state was restored from serialized widgets - don't overwrite it.
                    if (!node._configuredFromWorkflow) {
                        // Load initial prompt data (LoRAs and trigger words)
                        const categoryWidget = node.widgets.find(w => w.name === "category");
                        const promptWidget = node.widgets.find(w => w.name === "name");
                        if (categoryWidget && promptWidget && promptWidget.value) {
                            loadPromptData(node, categoryWidget.value, promptWidget.value);
                        }
                    }

                    // Ensure height is sufficient after data is loaded
                    setTimeout(() => {
                        const computedSize = node.computeSize();
                        const baseMinHeight = node._isWorkflowManager ? 420 : 600;
                        const minHeight = Math.max(baseMinHeight, computedSize[1] + 20);

                        if (node._isWorkflowManager) {
                            // Auto-shrink legacy oversized Workflow Manager nodes created before compact sizing.
                            if (node.size[1] > 560 || node.size[1] < minHeight) {
                                node.setSize([Math.max(440, node.size[0]), minHeight]);
                            }
                        } else if (node.size[1] < minHeight) {
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
                enforceWorkflowManagerCompactMode(node);

                // Flag that this node is being restored from a workflow,
                // so onNodeCreated's async loadPromptData won't overwrite state
                node._configuredFromWorkflow = true;

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
                    }
                }

                // IMPORTANT: Reattach DOM widgets SYNCHRONOUSLY during configure
                // to ensure proper positioning within the node bounds
                if (node._isWorkflowManager && !node.workflowManagerPreviewAttached) {
                    addWorkflowManagerPreview(node);
                }
                if (!node.promptSelectorWidget) {
                    createPromptSelectorWidget(node);
                }
                for (let i = node.inputs.length - 1; i >= 0; i--) {
                    const inp = node.inputs[i];
                    if (inp.name === "category" || inp.name === "name" || inp.name === "prompt_input") {
                        node.removeInput(i);
                    }
                }
                if (!node.buttonBarAttached) {
                    addButtonBar(node);
                    setupCategoryChangeHandler(node);
                }
                if (!node._isWorkflowManager && !node.loraDisplaysAttached) {
                    addLoraDisplays(node);
                }
                if (!node._isWorkflowManager && !node.triggerWordsDisplayAttached) {
                    addTriggerWordsDisplay(node);
                }
                if (!node._isWorkflowManager) {
                    setupUseExternalToggleHandler(node);
                }

                // Load prompts data asynchronously (data only, widgets already added)
                node._restoringFromWorkflow = true;
                loadPrompts(node).then(async () => {
                    try {
                        filterPromptDropdown(node, { preserveDanglingSelection: true });

                        // Archival-safe restore: never auto-load prompt-library data during
                        // onConfigure. The serialized workflow payload is authoritative and
                        // must not be overwritten even if a same-named prompt exists.

                        // Re-check LoRA availability after restoring from serialized state.
                        // The 'available' field is not serialized in toggle widgets, so after
                        // a tab switch all restored loras would show as "not found" without this.
                        await recheckLoraAvailability(node);

                        updateLoraDisplays(node);
                        updateTriggerWordsDisplay(node);

                        // Snapshot restored state so unsaved-changes detection
                        // doesn't falsely trigger after a reload
                        updateLastSavedState(node);

                        // Update custom prompt selector display
                        if (node.updatePromptSelectorDisplay) {
                            node.updatePromptSelectorDisplay();
                        }

                        if (node._isWorkflowManager) {
                            const computedSize = node.computeSize();
                            const minHeight = Math.max(420, computedSize[1] + 20);
                            if (node.size[1] > 560 || node.size[1] < minHeight) {
                                node.setSize([Math.max(440, node.size[0]), minHeight]);
                            }
                        }

                        app.graph.setDirtyCanvas(true, true);
                    } finally {
                        node._restoringFromWorkflow = false;
                    }
                });

                return result;
            };

            // Enforce minimum node size
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                size[0] = Math.max(440, size[0]);
                size[1] = Math.max(this?._isWorkflowManager ? 420 : 600, size[1]);
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
 * Re-check availability of all loras currently on the node.
 * Called after restoring from serialized state (tab switch / workflow load)
 * because the 'available' flag is not persisted in toggle widgets.
 */
async function recheckLoraAvailability(node) {
    const allLoras = [
        ...(node.savedLorasA || []),
        ...(node.savedLorasB || []),
        ...(node.currentLorasA || []),
        ...(node.currentLorasB || [])
    ];
    const allNames = [...new Set(allLoras.map(l => l.name).filter(Boolean))];
    if (allNames.length === 0) return;

    try {
        const response = await fetch("/prompt-manager-advanced/check-loras", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lora_names: allNames })
        });
        const data = await response.json();
        if (data.success && data.results) {
            const updateList = (list) => {
                (list || []).forEach(lora => {
                    if (lora.name) {
                        const found = data.results[lora.name] === true;
                        lora.available = found;
                        lora.found = found;
                    }
                });
            };
            updateList(node.savedLorasA);
            updateList(node.savedLorasB);
            updateList(node.currentLorasA);
            updateList(node.currentLorasB);
        }
    } catch (error) {
        console.error("[PromptManagerAdvanced] Error re-checking LoRA availability:", error);
        // Default to available on error so they don't all show as missing
        allLoras.forEach(l => {
            if (l.available === undefined) l.available = true;
            if (l.found === undefined) l.found = l.available !== false;
        });
    }
}

function parseSerializedWidgetValue(widget, fallback = []) {
    if (!widget || !widget.value) {
        return fallback;
    }

    try {
        const parsed = JSON.parse(widget.value);
        return Array.isArray(parsed) ? parsed : fallback;
    } catch (error) {
        return fallback;
    }
}

function getSerializedSavedLoras(node, stackId, unavailableLoras = new Set()) {
    const widget = stackId === "a" ? node.lorasAToggleWidget : node.lorasBToggleWidget;
    return parseSerializedWidgetValue(widget, []).map(lora => ({
        ...lora,
        active: lora.active !== false,
        strength: lora.strength ?? lora.model_strength ?? 1.0,
        available: !unavailableLoras.has((lora.name || "").toLowerCase())
    }));
}

function getSerializedSavedTriggerWords(node) {
    return parseSerializedWidgetValue(node.triggerWordsToggleWidget, []).map(word => ({
        text: word.text,
        active: word.active !== false,
        source: 'saved'
    }));
}

function enforceWorkflowManagerCompactMode(node) {
    if (!node?._isWorkflowManager) return;

    const hiddenNames = new Set(["category", "name", "text", "use_prompt_input", "use_lora_input", "use_workflow_data"]);
    for (const widget of (node.widgets || [])) {
        if (!hiddenNames.has(String(widget?.name || ""))) continue;
        widget.type = "converted-widget";
        widget.hidden = true;
        widget.computeSize = () => [0, -4];
        if (widget.inputEl) {
            widget.inputEl.style.display = "none";
        }
    }

    // Cleanup stale sockets from pre-change serialized nodes.
    if (Array.isArray(node.inputs) && typeof node.removeInput === "function") {
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const n = String(node.inputs[i]?.name || "").toLowerCase();
            if (n === "thumbnail_image" || n === "image") {
                try {
                    node.removeInput(i);
                } catch {
                    // Best-effort only.
                }
            }
        }
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
            const parsedModelStrength = parseFloat(match[2]);
            const modelStrength = Number.isNaN(parsedModelStrength) ? 1.0 : parsedModelStrength;
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
        const parsedModelStrength = strengthWidget ? parseFloat(strengthWidget.value) : NaN;
        const parsedClipStrength = clipStrengthWidget ? parseFloat(clipStrengthWidget.value) : NaN;
        const modelStrength = Number.isNaN(parsedModelStrength) ? 1.0 : parsedModelStrength;
        const clipStrength = Number.isNaN(parsedClipStrength) ? modelStrength : parsedClipStrength;

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

function hasWorkflowDataPayload(rawWorkflowData) {
    return (
        (typeof rawWorkflowData === "string" && rawWorkflowData.trim().length > 0) ||
        (rawWorkflowData && typeof rawWorkflowData === "object" && Object.keys(rawWorkflowData).length > 0)
    );
}

function hasMeaningfulWorkflowData(rawWorkflowData) {
    const wf = parseJsonObjectSafe(rawWorkflowData, null);
    if (!wf || typeof wf !== "object") return false;

    if ((Number(wf.version || 0) >= 2) && wf.models && typeof wf.models === "object") {
        for (const slot of ["model_a", "model_b", "model_c", "model_d"]) {
            const block = wf.models[slot];
            if (!block || typeof block !== "object") continue;
            const positivePrompt = String(block.positive_prompt || "").trim();
            const modelName = String(block.model || "").trim();
            const hasLoras = Array.isArray(block.loras) && block.loras.some((lora) => String(lora?.name || "").trim().length > 0);
            const hasRuntimeCarrier = ["MODEL", "CLIP", "VAE", "POSITIVE", "NEGATIVE"]
                .some((key) => block[key] != null);
            if (positivePrompt.length > 0 || modelName.length > 0 || hasLoras || hasRuntimeCarrier) {
                return true;
            }
        }

        if (["LATENT", "IMAGE", "MASK"].some((key) => wf[key] != null)) {
            return true;
        }
        return false;
    }

    // Renderer passthrough payloads may carry runtime objects but omit textual
    // fields. Treat these as meaningful for save-time extraction.
    const hasRuntimeCarrier = ["MODEL", "MODEL_A", "CLIP", "VAE", "POSITIVE", "NEGATIVE", "LATENT", "IMAGE"]
        .some((key) => wf[key] != null);

    const positivePrompt = String(wf.positive_prompt || "").trim();
    const modelA = String(wf.model_a || "").trim();
    const modelB = String(wf.model_b || "").trim();
    const hasLorasA = Array.isArray(wf.loras_a) && wf.loras_a.some((lora) => String(lora?.name || "").trim().length > 0);
    const hasLorasB = Array.isArray(wf.loras_b) && wf.loras_b.some((lora) => String(lora?.name || "").trim().length > 0);

    return hasRuntimeCarrier || positivePrompt.length > 0 || modelA.length > 0 || modelB.length > 0 || hasLorasA || hasLorasB;
}

function hasConnectedWorkflowInput(node) {
    const wfInput = node?.inputs?.find((inp) => inp?.name === "recipe_data");
    return wfInput?.link != null;
}

function getPromptNamesForCategory(node, category, options = {}) {
    const hideNSFW = options.hideNSFW === true;
    const workflowOnly = options.workflowOnly === true;
    const categoryPrompts = node?.prompts?.[category];
    if (!categoryPrompts || typeof categoryPrompts !== "object") return [];

    let promptNames = Object.keys(categoryPrompts)
        .filter((k) => k !== "__meta__")
        .sort((a, b) => a.localeCompare(b));

    if (hideNSFW) {
        promptNames = promptNames.filter((name) => categoryPrompts?.[name]?.nsfw !== true);
    }

    if (workflowOnly) {
        promptNames = promptNames.filter((name) => hasWorkflowDataPayload(categoryPrompts?.[name]?.workflow_data));
    }

    return promptNames;
}

function getVisibleCategories(node, options = {}) {
    const categories = Object.keys(node?.prompts || {})
        .filter((c) => c !== "__meta__")
        .sort((a, b) => a.localeCompare(b));

    // Keep empty categories visible so users can still save prompts/workflows into them.
    return categories;
}

function filterPromptDropdown(node, options = {}) {
    const preserveDanglingSelection = options?.preserveDanglingSelection === true;
    const categoryWidget = node.widgets.find(w => w.name === "category");
    const promptWidget = node.widgets.find(w => w.name === "name");

    if (categoryWidget && promptWidget) {
        const hideNSFW = app.ui.settings.getSettingValue("PromptManager.DefaultHideNSFW") === true;
        const workflowOnly = node?._isWorkflowManager === true;
        const visibleCategories = getVisibleCategories(node, { hideNSFW, workflowOnly });

        const currentCategoryValue = String(categoryWidget.value || "");
        let categoryOptions = visibleCategories.length > 0 ? [...visibleCategories] : ["Default"];
        if (preserveDanglingSelection && currentCategoryValue && !categoryOptions.includes(currentCategoryValue)) {
            categoryOptions = [currentCategoryValue, ...categoryOptions];
        }
        categoryWidget.options.values = categoryOptions;

        if (!categoryOptions.includes(categoryWidget.value)) {
            categoryWidget.value = visibleCategories[0] || "Default";
        }

        const currentCategory = categoryWidget.value;
        const promptNames = getPromptNamesForCategory(node, currentCategory, { hideNSFW, workflowOnly });

        // Preserve explicit empty selection (New Prompt) across tab-switch restore.
        const currentPromptValue = String(promptWidget.value || "");
        const wantsEmptySelection = currentPromptValue === "";
        let promptOptions = promptNames.length > 0 ? [...promptNames] : [""];
        if (wantsEmptySelection && !promptOptions.includes("")) {
            promptOptions = ["", ...promptOptions];
        }
        if (preserveDanglingSelection && currentPromptValue && !promptOptions.includes(currentPromptValue)) {
            promptOptions = [currentPromptValue, ...promptOptions];
        }

        promptWidget.options.values = promptOptions;
        if (!promptOptions.includes(promptWidget.value)) {
            promptWidget.value = wantsEmptySelection ? "" : (promptOptions[0] || "");
        }
    }
}

function updateWorkflowManagerPreview(node) {
    const ui = node?._workflowManagerPreview;
    if (!ui) return;

    const categoryWidget = node.widgets?.find(w => w.name === "category");
    const promptWidget = node.widgets?.find(w => w.name === "name");
    const category = categoryWidget?.value || "";
    const prompt = promptWidget?.value || "";
    const promptData = node.prompts?.[category]?.[prompt] || null;
    const liveWorkflow = (node.lastWorkflowData && typeof node.lastWorkflowData === "object") ? node.lastWorkflowData : null;
    const workflowInputConnected = hasConnectedWorkflowInput(node);

    // Keep info button in the preview corner, active only when recipe data is available.
    const rawInfoData = workflowInputConnected
        ? liveWorkflow
        : (promptData?.workflow_data || liveWorkflow || null);
    const infoData = parseJsonObjectSafe(rawInfoData, null);
    const infoSummary = _summarizeWorkflowDataDiscovery(infoData);
    const infoEnabled = infoSummary.modelCount > 0;
    if (ui.infoBtn) {
        ui.infoBtn.disabled = !infoEnabled;
        ui.infoBtn.style.background = infoEnabled ? "rgba(55, 142, 255, 0.92)" : "rgba(120, 130, 140, 0.35)";
        ui.infoBtn.style.color = infoEnabled ? "#ffffff" : "rgba(220, 226, 234, 0.75)";
        ui.infoBtn.style.cursor = infoEnabled ? "pointer" : "default";
        ui.infoBtn.style.opacity = infoEnabled ? "1" : "0.8";
        ui.infoBtn.title = infoEnabled ? "Show workflow summary" : "No workflow metadata";
    }

    // Input-connected mode should not show saved/random prompt thumbnails while waiting.
    // Use the placeholder image until live workflow_data arrives.
    if (workflowInputConnected && !liveWorkflow) {
        ui.image.src = DEFAULT_THUMBNAIL;
        ui.image.style.display = "block";
        ui.emptyLabel.style.display = "none";
        return;
    }

    const pickThumbnail = (obj) => {
        if (!obj || typeof obj !== "object") return "";
        const candidates = [
            obj.thumbnail,
            obj.connected_thumbnail,
            obj.thumbnail_image,
            obj.preview,
            obj.image,
        ];
        for (const value of candidates) {
            if (typeof value === "string" && value.trim()) {
                return value;
            }
        }
        return "";
    };

    const liveThumbnail = pickThumbnail(liveWorkflow) || (typeof node.connectedThumbnail === "string" ? node.connectedThumbnail : "");
    const savedThumbnail = pickThumbnail(promptData);

    // In input-connected mode, never fall back to saved prompt thumbnails.
    // If incoming workflow_data has no thumbnail, keep placeholder/empty state
    // instead of showing stale image from a previously selected recipe.
    const thumbnail = workflowInputConnected
        ? (liveThumbnail || DEFAULT_THUMBNAIL)
        : (liveThumbnail || savedThumbnail || DEFAULT_THUMBNAIL);

    if (thumbnail && thumbnail !== DEFAULT_THUMBNAIL) {
        ui.image.src = thumbnail;
        ui.image.style.display = "block";
        ui.emptyLabel.style.display = "none";
    } else {
        ui.image.removeAttribute("src");
        ui.image.style.display = "none";
        ui.emptyLabel.style.display = "flex";
        if (liveWorkflow) {
            ui.emptyLabel.textContent = "Connected workflow loaded (no thumbnail)";
        } else if (workflowInputConnected) {
            ui.emptyLabel.textContent = "Waiting for workflow_data input...";
        } else {
            ui.emptyLabel.textContent = "No thumbnail for selected workflow";
        }
    }
}

function addWorkflowManagerPreview(node) {
    if (node.workflowManagerPreviewAttached) return;

    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 6px;
        background: #1e1e1e;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        padding: 8px;
        box-sizing: border-box;
        overflow: hidden;
    `;

    const title = document.createElement("div");
    title.textContent = "Workflow Preview";
    title.style.cssText = `
        font-size: 11px;
        color: #9aa;
        font-weight: 600;
        letter-spacing: 0.2px;
    `;

    const previewBox = document.createElement("div");
    previewBox.style.cssText = `
        position: relative;
        width: 100%;
        aspect-ratio: 1 / 1;
        border-radius: 6px;
        background: #111;
        border: 1px solid #333;
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
    emptyLabel.textContent = "No thumbnail for selected workflow";
    emptyLabel.style.cssText = `
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-size: 11px;
        color: #777;
        background: #141414;
        border: 1px dashed rgba(110, 110, 110, 0.5);
        border-radius: 5px;
        padding: 8px;
        box-sizing: border-box;
    `;

    const infoBtn = document.createElement("button");
    const infoBtnLabel = document.createElement("span");
    infoBtnLabel.textContent = "info";
    infoBtnLabel.style.cssText = "display:block; ";
    infoBtn.appendChild(infoBtnLabel);
    infoBtn.style.cssText = `
        position: absolute;
        right: 4px;
        bottom: 4px;
        width: 28px;
        height: 14px;
        border-radius: 999px;
        border: 1px solid rgba(240, 248, 255, 0.95);
        background: rgba(120, 130, 140, 0.35);
        color: rgba(220, 226, 234, 0.75);
        font-size: 8px;
        font-weight: 700;
        letter-spacing: 0.2px;
        font-family: inherit;
        appearance: none;
        display: flex;
        align-items: center;
        justify-content: center;
        line-height: 1;
        cursor: default;
        z-index: 2;
        opacity: 0.8;
        padding: 0;
    `;
    infoBtn.onclick = async (evt) => {
        evt.preventDefault();
        evt.stopPropagation();
        if (infoBtn.disabled) return;
        await showWorkflowDiscoverySummary(node);
    };

    container.appendChild(title);
    previewBox.appendChild(image);
    previewBox.appendChild(emptyLabel);
    previewBox.appendChild(infoBtn);
    container.appendChild(previewBox);

    const widget = node.addDOMWidget("workflow_manager_preview", "div", container, {
        hideOnZoom: false,
    });
    // Keep width behavior unchanged; make preview area ~20% taller.
    widget.computeSize = (width) => [Math.max(width || 260, 260), 320];

    node._workflowManagerPreview = { container, image, emptyLabel, infoBtn, widget };
    node.workflowManagerPreviewAttached = true;
    updateWorkflowManagerPreview(node);
}

function _normalizeModelSlotLabel(slotKey) {
    const key = String(slotKey || "").toLowerCase();
    if (key === "model_a") return "A";
    if (key === "model_b") return "B";
    if (key === "model_c") return "C";
    if (key === "model_d") return "D";
    return String(slotKey || "?");
}

function _modelNameForSummary(pathLike) {
    const raw = String(pathLike || "").trim();
    if (!raw) return "";
    const parts = raw.replace(/\\/g, "/").split("/");
    const base = parts[parts.length - 1] || raw;
    return base.replace(/\.(safetensors|ckpt|pt|pth|bin|gguf|sft)$/i, "");
}

function _normalizeSummaryLoras(list) {
    if (!Array.isArray(list)) return [];
    return list
        .map((item) => {
            if (!item || typeof item !== "object") return null;
            const name = String(item.name || "").trim();
            if (!name) return null;
            const strengthRaw = Number(item.model_strength ?? item.strength ?? 1.0);
            const strength = Number.isFinite(strengthRaw) ? strengthRaw : 1.0;
            const available = item.available !== false && item.found !== false;
            return {
                name,
                strength,
                active: item.active !== false,
                available,
                found: item.found !== false,
            };
        })
        .filter(Boolean)
        .sort(compareLorasMissingLast);
}

function _summarizeWorkflowDataDiscovery(rawWorkflowData) {
    const empty = {
        modelCount: 0,
        modelDetails: [],
        modelSlots: [],
        loraSlots: [],
        positivePromptPresent: false,
        source: "",
    };

    if (!rawWorkflowData || typeof rawWorkflowData !== "object") return empty;

    const slotOrder = ["model_a", "model_b", "model_c", "model_d"];
    const modelSlots = [];
    const loraSlots = [];
    const modelDetails = [];
    const source = String(rawWorkflowData._source || "");

    const models = (rawWorkflowData.models && typeof rawWorkflowData.models === "object")
        ? rawWorkflowData.models
        : null;

    if (models) {
        for (const slot of slotOrder) {
            const block = models[slot];
            if (!block || typeof block !== "object") continue;
            const modelName = String(block.model || "").trim();
            const loras = _normalizeSummaryLoras(block.loras);
            if (modelName) modelSlots.push(slot);
            if (loras.length > 0) loraSlots.push(slot);
            if (modelName || loras.length > 0) {
                modelDetails.push({
                    slot,
                    slotLabel: _normalizeModelSlotLabel(slot),
                    model: modelName,
                    modelDisplay: _modelNameForSummary(modelName),
                    family: String(block.family || ""),
                    loras,
                    promptPresent: String(block.positive_prompt || "").trim().length > 0,
                });
            }
        }
    } else {
        const modelA = String(rawWorkflowData.model_a || "").trim();
        const modelB = String(rawWorkflowData.model_b || "").trim();
        if (modelA) modelSlots.push("model_a");
        if (modelB) modelSlots.push("model_b");
        const lorasA = _normalizeSummaryLoras(rawWorkflowData.loras_a);
        const lorasB = _normalizeSummaryLoras(rawWorkflowData.loras_b);
        if (lorasA.length > 0) loraSlots.push("model_a");
        if (lorasB.length > 0) loraSlots.push("model_b");

        if (modelA || lorasA.length > 0) {
            modelDetails.push({
                slot: "model_a",
                slotLabel: "A",
                model: modelA,
                modelDisplay: _modelNameForSummary(modelA),
                family: String(rawWorkflowData.family || ""),
                loras: lorasA,
                promptPresent: String(rawWorkflowData.positive_prompt || "").trim().length > 0,
            });
        }
        if (modelB || lorasB.length > 0) {
            modelDetails.push({
                slot: "model_b",
                slotLabel: "B",
                model: modelB,
                modelDisplay: _modelNameForSummary(modelB),
                family: String(rawWorkflowData.family || ""),
                loras: lorasB,
                promptPresent: String(rawWorkflowData.negative_prompt || "").trim().length > 0,
            });
        }
    }

    const positivePromptPresent = models
        ? modelSlots.some((slot) => String((models[slot] || {}).positive_prompt || "").trim().length > 0)
        : String(rawWorkflowData.positive_prompt || "").trim().length > 0;

    return {
        modelCount: modelSlots.length,
        modelDetails,
        modelSlots,
        loraSlots,
        positivePromptPresent,
        source,
    };
}

function _createSummaryLoraTag(lora) {
    const tag = document.createElement("div");
    const isActive = lora?.active !== false;
    const isAvailable = lora?.available !== false && lora?.found !== false;
    const strength = Number(lora?.strength ?? 1.0);

    let bgColor, textColor, borderColor;
    if (!isAvailable) {
        bgColor = isActive ? "rgba(220, 53, 69, 0.9)" : "rgba(220, 53, 69, 0.4)";
        textColor = isActive ? "white" : "rgba(255, 200, 200, 0.8)";
        borderColor = "rgba(220, 53, 69, 0.9)";
    } else if (isActive) {
        bgColor = "rgba(66, 153, 225, 0.9)";
        textColor = "white";
        borderColor = "rgba(122, 188, 243, 0.9)";
    } else {
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
        transition: "background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease",
        backgroundColor: bgColor,
        color: textColor,
        border: `1px solid ${borderColor}`,
        width: "200px",
        height: "24px",
        boxSizing: "border-box",
        userSelect: "none",
        cursor: "default",
        flexShrink: "0",
    });

    if (!isAvailable) {
        const warningIcon = document.createElement("span");
        warningIcon.textContent = "⚠️";
        warningIcon.style.fontSize = "11px";
        warningIcon.style.marginRight = "-2px";
        tag.appendChild(warningIcon);
    }

    const nameSpan = document.createElement("span");
    nameSpan.textContent = String(lora?.name || "");
    Object.assign(nameSpan.style, {
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        flexGrow: "1",
        textDecoration: !isAvailable ? "line-through" : "none",
        opacity: !isAvailable ? "0.8" : "1",
    });
    tag.appendChild(nameSpan);

    const strengthBadge = document.createElement("span");
    strengthBadge.textContent = Number.isFinite(strength) ? strength.toFixed(2) : "1.00";
    Object.assign(strengthBadge.style, {
        fontSize: "10px",
        fontWeight: "600",
        padding: "1px 6px",
        borderRadius: "999px",
        backgroundColor: "rgba(255,255,255,0.15)",
        color: "rgba(255,255,255,0.9)",
        border: "1px solid transparent",
        flexShrink: "0",
    });
    tag.appendChild(strengthBadge);

    let hoverTimeout = null;
    tag.addEventListener("mouseenter", () => {
        tag.style.transform = "translateY(-1px)";
        tag.style.boxShadow = "0 2px 4px rgba(0,0,0,0.2)";
        if (loraManagerAvailable && isAvailable) {
            const tagRect = tag.getBoundingClientRect();
            hoverTimeout = setTimeout(async () => {
                const tooltip = await getLoraManagerPreviewTooltip();
                if (tooltip) {
                    tooltip.show(String(lora?.name || ""), tagRect.right, tagRect.top);
                }
            }, 250);
        }
    });
    tag.addEventListener("mouseleave", () => {
        tag.style.transform = "translateY(0)";
        tag.style.boxShadow = "none";
        if (hoverTimeout) {
            clearTimeout(hoverTimeout);
            hoverTimeout = null;
        }
        if (loraManagerPreviewTooltip) {
            loraManagerPreviewTooltip.hide();
        }
    });

    if (!loraManagerAvailable || !isAvailable) {
        tag.title = !isAvailable
            ? `${lora?.name || ""}\nNot found`
            : `${lora?.name || ""}\nStrength: ${Number.isFinite(strength) ? strength.toFixed(2) : "1.00"}`;
    }

    return tag;
}

function _showWorkflowSummaryDialog(summary) {
    return new Promise((resolve) => {
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

        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: min(442px, 92vw);
            max-height: 78vh;
            overflow: auto;
            background: #232930;
            border: 1px solid #3c4b59;
            border-radius: 10px;
            padding: 12px;
            z-index: 10000;
            box-shadow: 0 8px 28px rgba(0,0,0,0.55);
        `;

        const title = document.createElement("div");
        title.textContent = "Summary";
        title.style.cssText = "margin-bottom: 8px; font-size: 17px; font-weight: 700; color: #e7eef6;";
        dialog.appendChild(title);

        const meta = document.createElement("div");
        meta.style.cssText = "margin-bottom: 12px; color: #9eb2c6; font-size: 12px; padding-bottom: 10px; border-bottom: 1px solid rgba(115, 144, 170, 0.35);";
        meta.innerHTML = `<b style=\"color:#c5d8ea;\">Models found:</b> ${summary.modelCount}`;
        dialog.appendChild(meta);

        const modelsSection = document.createElement("div");
        modelsSection.style.cssText = "margin-bottom: 14px;";
        modelsSection.innerHTML = `<div style=\"color:#8fb8dc; font-weight:700; margin-bottom:6px;\">Models</div>`;
        const modelsList = document.createElement("div");
        modelsList.style.cssText = "display:flex; flex-direction:column; gap:6px;";
        if (!summary.modelDetails.length) {
            const none = document.createElement("div");
            none.textContent = "No models found.";
            none.style.cssText = "color:#9aa8b7; font-size:12px;";
            modelsList.appendChild(none);
        } else {
            for (const item of summary.modelDetails) {
                const row = document.createElement("div");
                const familyText = item.family ? ` (${item.family})` : "";
                row.textContent = `${item.slotLabel}: ${item.modelDisplay || "(none)"}${familyText}`;
                row.style.cssText = "color:#d9e6f3; font-size:13px; background: rgba(37, 48, 60, 0.55); border: 1px solid rgba(96, 124, 150, 0.35); border-radius: 6px; padding: 6px 8px;";
                modelsList.appendChild(row);
            }
        }
        modelsSection.appendChild(modelsList);
        dialog.appendChild(modelsSection);

        const loraDivider = document.createElement("div");
        loraDivider.style.cssText = "height: 1px; background: rgba(115, 144, 170, 0.35); margin: 2px 0 12px 0;";
        dialog.appendChild(loraDivider);

        const loraSection = document.createElement("div");
        loraSection.innerHTML = `<div style=\"color:#8fb8dc; font-weight:700; margin-bottom:8px;\">LoRAs</div>`;
        if (!summary.modelDetails.some((m) => Array.isArray(m.loras) && m.loras.length > 0)) {
            const none = document.createElement("div");
            none.textContent = "No LoRAs found.";
            none.style.cssText = "color:#9aa8b7; font-size:12px; margin-bottom:8px;";
            loraSection.appendChild(none);
        } else {
            for (const item of summary.modelDetails) {
                if (!Array.isArray(item.loras) || item.loras.length === 0) continue;
                const stackCard = document.createElement("div");
                stackCard.style.cssText = "background: rgba(30, 40, 52, 0.55); border: 1px solid rgba(96, 124, 150, 0.35); border-radius: 8px; padding: 6px; margin-bottom: 6px;";

                const stackTitle = document.createElement("div");
                stackTitle.textContent = `Stack ${item.slotLabel}:`;
                stackTitle.style.cssText = "color:#c5d8ea; font-size:12px; margin:0 0 6px 0;";
                stackCard.appendChild(stackTitle);

                const stackTags = document.createElement("div");
                stackTags.style.cssText = "display:grid; grid-template-columns: 200px 200px; justify-content:start; gap:2px;";
                for (const lora of item.loras) {
                    stackTags.appendChild(_createSummaryLoraTag(lora));
                }
                stackCard.appendChild(stackTags);
                loraSection.appendChild(stackCard);
            }
        }
        dialog.appendChild(loraSection);

        const actions = document.createElement("div");
        actions.style.cssText = "display:flex; justify-content:flex-end; margin-top:14px;";
        const okBtn = document.createElement("button");
        okBtn.textContent = "OK";
        okBtn.style.cssText = "padding:8px 16px; background:#2f7dbf; color:#fff; border:none; border-radius:6px; cursor:pointer;";
        actions.appendChild(okBtn);
        dialog.appendChild(actions);

        const cleanup = () => {
            if (loraManagerPreviewTooltip) {
                loraManagerPreviewTooltip.hide();
            }
            if (overlay.parentNode) document.body.removeChild(overlay);
            if (dialog.parentNode) document.body.removeChild(dialog);
        };

        okBtn.onclick = () => {
            cleanup();
            resolve(true);
        };
        overlay.onclick = () => {
            cleanup();
            resolve(true);
        };

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        okBtn.focus();
    });
}

async function showWorkflowDiscoverySummary(node) {
    const hasWorkflowInput = hasConnectedWorkflowInput(node);
    const liveWorkflowData = hasWorkflowInput ? await resolveWorkflowDataForLive(node) : null;
    const workflowData = (liveWorkflowData && typeof liveWorkflowData === "object")
        ? liveWorkflowData
        : resolveWorkflowDataForSave(node);

    if (!workflowData || typeof workflowData !== "object") {
        await showInfo("Summary", "No recipe_data available yet. Execute upstream or select a saved workflow prompt.");
        return;
    }

    const summary = _summarizeWorkflowDataDiscovery(workflowData);
    await _showWorkflowSummaryDialog(summary);
}

function attachWorkflowCornerInfoHandlers(node) {
    if (!node?._isWorkflowManager || node._workflowCornerInfoAttached) return;

    const previousDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function(ctx) {
        const result = previousDrawForeground ? previousDrawForeground.apply(this, arguments) : undefined;

        if (this.flags?.collapsed) {
            this._workflowInfoBounds = null;
            return result;
        }

        const summary = _summarizeWorkflowDataDiscovery(this.lastWorkflowData);
        const infoEnabled = summary.modelCount > 0;
        const titleHeight = LiteGraph.NODE_TITLE_HEIGHT || 30;
        const infoWidth = 30;
        const infoHeight = 14;
        const infoX = this.size[0] - infoWidth - 8;
        const infoY = (titleHeight / 2) - 37;
        const cornerRadius = 7;

        const drawRoundRect = (x, y, w, h, r) => {
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.lineTo(x + w - r, y);
            ctx.quadraticCurveTo(x + w, y, x + w, y + r);
            ctx.lineTo(x + w, y + h - r);
            ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
            ctx.lineTo(x + r, y + h);
            ctx.quadraticCurveTo(x, y + h, x, y + h - r);
            ctx.lineTo(x, y + r);
            ctx.quadraticCurveTo(x, y, x + r, y);
            ctx.closePath();
        };

        ctx.save();
        drawRoundRect(infoX, infoY, infoWidth, infoHeight, cornerRadius);
        ctx.fillStyle = infoEnabled ? 'rgba(55, 142, 255, 0.92)' : 'rgba(120, 130, 140, 0.35)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(240, 248, 255, 0.95)';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = infoEnabled ? '#ffffff' : 'rgba(220, 226, 234, 0.75)';
        ctx.font = 'bold 8px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('info', infoX + (infoWidth / 2), infoY + (infoHeight / 2) + 0.5);
        ctx.restore();

        this._workflowInfoBounds = {
            x: infoX,
            y: infoY,
            width: infoWidth,
            height: infoHeight,
            enabled: infoEnabled,
        };
        return result;
    };

    const previousMouseMove = node.onMouseMove;
    node.onMouseMove = function(e, localPos, canvas) {
        const result = previousMouseMove ? previousMouseMove.apply(this, arguments) : undefined;
        const b = this._workflowInfoBounds;
        if (!b) return result;

        const hovering = localPos[0] >= b.x && localPos[0] <= b.x + b.width &&
            localPos[1] >= b.y && localPos[1] <= b.y + b.height;
        if (hovering) {
            canvas.canvas.title = b.enabled ? 'Workflow metadata found (info available)' : 'No workflow metadata';
            canvas.canvas.style.cursor = b.enabled ? 'pointer' : '';
        }
        return result;
    };

    const previousMouseDown = node.onMouseDown;
    node.onMouseDown = function(e, localPos, canvas) {
        const b = this._workflowInfoBounds;
        if (b) {
            const hit = localPos[0] >= b.x && localPos[0] <= b.x + b.width &&
                localPos[1] >= b.y && localPos[1] <= b.y + b.height;
            if (hit && b.enabled) {
                void showWorkflowDiscoverySummary(this);
                return true;
            }
        }
        return previousMouseDown ? previousMouseDown.apply(this, arguments) : undefined;
    };

    node._workflowCornerInfoAttached = true;
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

    // Keep available LoRAs first, then missing ones, each group alphabetical.
    merged.sort(compareLorasMissingLast);

    return merged;
}

function isLoraAvailableForSort(lora) {
    if (!lora) return true;
    if (lora.available === false) return false;
    if (lora.found === false) return false;
    return true;
}

function compareLorasMissingLast(a, b) {
    const aAvailable = isLoraAvailableForSort(a);
    const bAvailable = isLoraAvailableForSort(b);
    if (aAvailable !== bAvailable) {
        return aAvailable ? -1 : 1;
    }
    return String(a?.name || "").localeCompare(String(b?.name || ""));
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

    const orderedLoras = (Array.isArray(loras) ? [...loras] : []).sort(compareLorasMissingLast);
    orderedLoras.forEach((lora, index) => {
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
    const parsedStrength = parseFloat(lora.strength ?? lora.model_strength ?? 1.0);
    const strength = Number.isNaN(parsedStrength) ? 1.0 : parsedStrength;

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
        if (!isNaN(newValue)) {
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
            window.open(`https://civitai.red/search/models?sortBy=models_v9&query=${searchQuery}&modelType=LORA`, "_blank");
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
    // Reset all lora strengths to their original values (from Python)
    const useLoraInputWidget = node.widgets?.find(w => w.name === "use_lora_input");
    const useLoraInput = useLoraInputWidget?.value !== false;

    // Get original strengths from Python (simple map of name -> strength)
    const originalStrengths = stackId === "a" ? (node.originalStrengthsA || {}) : (node.originalStrengthsB || {});

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
        const originalStrength = originalStrengths[lora.name];
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
            strength: lora.strength ?? lora.model_strength ?? 1.0,
            fromInput: lora.fromInput || false  // Preserve flag indicating it came from connected input
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
            // New trigger word from connected input — default to OFF
            merged.push({ ...word, source: 'current', active: false });
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

function buildWorkflowDefaultName() {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const y = now.getFullYear();
    const m = pad(now.getMonth() + 1);
    const d = pad(now.getDate());
    const hh = pad(now.getHours());
    const mm = pad(now.getMinutes());
    const ss = pad(now.getSeconds());
    return `Workflow ${y}-${m}-${d} ${hh}-${mm}-${ss}`;
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
    buttonContainer.style.gap = "2px";
    buttonContainer.style.padding = "4px 4px 8px 4px";
    buttonContainer.style.flexWrap = "wrap";
    buttonContainer.style.marginTop = "0";

    // Prevent default context menu on button bar
    buttonContainer.addEventListener("contextmenu", (e) => e.preventDefault());

    // Forward wheel events to canvas for zooming
    forwardWheelToCanvas(buttonContainer);

    const saveButtonLabel = node._isWorkflowManager ? "Save Workflow" : "Save Prompt";
    const clearButtonLabel = node._isWorkflowManager ? "Clear Workflow" : "New Prompt";

    // Save Prompt/Workflow button
    const savePromptBtn = createButton(saveButtonLabel, async () => {
        const currentCategory = categoryWidget.value;
        const currentName = (promptWidget.value || "").trim();
        const useWorkflowDateDefault = node._isWorkflowManager && (
            hasConnectedWorkflowInput(node) ||
            !currentName ||
            currentName.toLowerCase() === "new prompt" ||
            node.isNewUnsavedPrompt === true
        );
        const initialName = useWorkflowDateDefault
            ? buildWorkflowDefaultName()
            : (currentName || "New Prompt");

        await openPromptBrowserForSave({
            node,
            currentCategory,
            currentPrompt: promptWidget.value || "",
            title: node._isWorkflowManager ? "Save Workflow" : "Save Prompt",
            saveButtonText: "Save",
            namePlaceholder: "Prompt name",
            initialName,
            workflowOnly: node?._isWorkflowManager === true,
            onSave: async ({ category, name, overwrite }) => {
                const promptName = String(name || "").trim();
                const targetCategory = String(category || "").trim();
                const promptText = textWidget.value;
                if (!promptName || !targetCategory) {
                    return { success: false, error: "Category and prompt name are required." };
                }

                try {
                    // Use the last-executed state from Python as the authoritative source for connected loras.
                    // node.currentLorasA/B is populated from the backend after each execution (input_loras_a/b),
                    // so it already reflects only what actually ran.
                    const connectedLorasA = node.currentLorasA || [];
                    const connectedLorasB = node.currentLorasB || [];

                    const useLoraInputWidget = node.widgets?.find(w => w.name === "use_lora_input");
                    const useLoraInput = useLoraInputWidget?.value !== false;

                    let allLorasA, allLorasB;
                    if (!useLoraInput) {
                        allLorasA = [...(node.savedLorasA || [])];
                        allLorasB = [...(node.savedLorasB || [])];
                    } else {
                        const mergedA = mergeLoraLists(
                            connectedLorasA.map(l => ({ ...l, source: "current" })),
                            node.savedLorasA || []
                        );
                        const mergedB = mergeLoraLists(
                            connectedLorasB.map(l => ({ ...l, source: "current" })),
                            node.savedLorasB || []
                        );
                        allLorasA = [...mergedA];
                        allLorasB = [...mergedB];
                    }

                    const allTriggerWords = mergeTriggerWordLists(
                        node.currentTriggerWords || [],
                        node.savedTriggerWords || []
                    );

                    const thumbnail = node.connectedThumbnail || null;

                    // Preserve existing NSFW status when overwriting from the browser-save flow.
                    let preservedNsfw = false;
                    const categoryPrompts = node.prompts?.[targetCategory];
                    if (categoryPrompts && typeof categoryPrompts === "object") {
                        const existingName = Object.keys(categoryPrompts)
                            .filter((k) => k !== "__meta__")
                            .find((k) => k.toLowerCase() === promptName.toLowerCase());
                        if (existingName && categoryPrompts[existingName]?.nsfw === true) {
                            preservedNsfw = true;
                        }
                    }

                    await savePrompt(node, targetCategory, promptName, promptText, allLorasA, allLorasB, allTriggerWords, thumbnail, preservedNsfw);

                    const useWorkflowWidget = node.widgets?.find(w => w.name === "use_workflow_data");
                    if (useWorkflowWidget?.value === true) {
                        const clone = (v, fb) => {
                            try {
                                return JSON.parse(JSON.stringify(v ?? fb));
                            } catch {
                                return fb;
                            }
                        };

                        node._preWorkflowModeState = {
                            text: promptText || "",
                            savedLorasA: clone(allLorasA, []),
                            savedLorasB: clone(allLorasB, []),
                            currentLorasA: [],
                            currentLorasB: [],
                            savedTriggerWords: clone(allTriggerWords, []),
                            currentTriggerWords: [],
                            lastWorkflowData: clone(node.lastWorkflowData, null),
                        };

                        useWorkflowWidget.value = false;
                        if (typeof useWorkflowWidget.callback === "function") {
                            await useWorkflowWidget.callback(false);
                        }
                    }

                    node._skipCallbackReload = true;
                    categoryWidget.value = targetCategory;
                    filterPromptDropdown(node);
                    promptWidget.value = promptName;
                    textWidget.value = promptText;
                    node._skipCallbackReload = false;

                    node._previousCategory = targetCategory;
                    node._previousPrompt = promptName;

                    node.savedLorasA = allLorasA;
                    node.savedLorasB = allLorasB;
                    node.savedTriggerWords = allTriggerWords;
                    updateLoraDisplays(node);
                    updateTriggerWordsDisplay(node);

                    if (node.updatePromptSelectorDisplay) {
                        node.updatePromptSelectorDisplay();
                    }

                    updateLastSavedState(node);

                    return {
                        success: true,
                        category: targetCategory,
                        name: promptName,
                        overwritten: overwrite === true,
                    };
                } catch (err) {
                    console.error("[PromptManagerAdvanced] Error during save:", err);
                    return { success: false, error: err?.message || "Error during save" };
                } finally {
                    node.isNewUnsavedPrompt = false;
                    node.newPromptCategory = null;
                    node.newPromptName = null;
                }
            },
        });
    });

    // New/Clear button - clears current editable state
    const newPromptBtn = createButton(clearButtonLabel, async () => {
        // Check for unsaved changes before creating new prompt
        const hasUnsaved = hasUnsavedChanges(node);
        const warnEnabled = app.ui.settings.getSettingValue("PromptManager.WarnUnsavedChanges");

        if (hasUnsaved && warnEnabled) {
            const confirmed = await showConfirm(
                "Unsaved Changes",
                node._isWorkflowManager
                    ? "You have unsaved workflow changes. Do you want to discard them and clear the current workflow selection?"
                    : "You have unsaved changes to the current prompt. Do you want to discard them and start fresh?",
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
        node.originalStrengthsA = {};  // Clear original strengths too
        node.originalStrengthsB = {};
        node.savedTriggerWords = [];
        node.currentTriggerWords = [];
        node.lastWorkflowData = null;
        syncSavedWorkflowDataWidget(node);

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

        // Force next backend update to be applied even when connected inputs
        // produce the same payload as before this New Prompt reset.
        node._lastExecutionUpdateSig = null;
        node._lastLiveWorkflowPickupSig = null;

        node.serialize_widgets = true;
        app.graph.setDirtyCanvas(true, true);
    });

    // More dropdown button
    const moreBtn = createDropdownButton("More ▼", [
        {
            label: "Delete Prompt",
            action: async () => {
                const currentCategory = categoryWidget.value;
                const currentPrompt = promptWidget.value;

                if (!currentPrompt) {
                    await showInfo("Error", "No prompt selected to delete.");
                    return;
                }

                const confirmed = await showConfirm(
                    "Delete Prompt",
                    `Are you sure you want to delete "${currentPrompt}" from category "${currentCategory}"? This cannot be undone.`,
                    "Delete",
                    "#c00"
                );

                if (confirmed) {
                    await deletePrompt(node, currentCategory, currentPrompt);
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
        return [width, 40];
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
        if (node._restoringFromWorkflow) {
            if (originalCallback) {
                originalCallback.apply(this, arguments);
            }
            return;
        }

        // Skip callback reload during save operations
        if (node._skipCallbackReload) {
            if (originalCallback) {
                originalCallback.apply(this, arguments);
            }
            return;
        }

        const previousCategory = node._previousCategory;
        const previousPrompt = node._previousPrompt;

        // Check for unsaved changes before switching (skip if navigating via custom selector)
        if (!node._skipUnsavedCheck) {
            const hasUnsaved = hasUnsavedChanges(node);
            const warnEnabled = app.ui.settings.getSettingValue("PromptManager.WarnUnsavedChanges");

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
            const promptNames = Object.keys(node.prompts[category]).filter(k => k !== '__meta__');
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
                node.originalStrengthsA = {};
                node.originalStrengthsB = {};
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
        if (node._restoringFromWorkflow) {
            if (originalPromptCallback) {
                originalPromptCallback.apply(this, arguments);
            }
            return;
        }

        // Skip callback reload during save operations
        if (node._skipCallbackReload) {
            if (originalPromptCallback) {
                originalPromptCallback.apply(this, arguments);
            }
            return;
        }

        const previousCategory = node._previousCategory;
        const previousPrompt = node._previousPrompt;

        // Check for unsaved changes before switching (skip if navigating via custom selector)
        if (!node._skipUnsavedCheck) {
            const hasUnsaved = hasUnsavedChanges(node);
            const warnEnabled = app.ui.settings.getSettingValue("PromptManager.WarnUnsavedChanges");

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
            const promptNames = Object.keys(node.prompts[category]).filter(k => k !== '__meta__').sort((a, b) => a.localeCompare(b));
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
                    const categoryPrompts = Object.keys(node.prompts[currentCategory]).filter(k => k !== '__meta__');
                    const hasOtherCategoryPrompts = newValues.some(val =>
                        val !== "" && !categoryPrompts.includes(val)
                    );

                    if (hasOtherCategoryPrompts) {
                        setTimeout(() => {
                            filterPromptDropdown(node, { preserveDanglingSelection: node._restoringFromWorkflow === true });
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

function syncSavedWorkflowDataWidget(node) {
    const w = node.widgets?.find((x) => x.name === "saved_workflow_data");
    if (!w) return;

    const wf = node.lastWorkflowData;
    if (!wf) {
        w.value = "";
        return;
    }

    if (typeof wf === "string") {
        w.value = wf;
        return;
    }

    try {
        w.value = JSON.stringify(wf);
    } catch {
        w.value = "";
    }
}

function mapWorkflowLorasToUi(list) {
    if (!Array.isArray(list)) return [];
    return list
        .map((lora) => {
            const name = String(lora?.name || "").trim();
            if (!name) return null;
            const strength = Number(lora?.model_strength ?? lora?.strength ?? 1.0);
            const clipStrength = Number(lora?.clip_strength ?? lora?.strength ?? strength);
            const available = lora?.available !== false;
            const found = (lora?.found !== undefined) ? (lora.found !== false) : available;
            return {
                name,
                strength: Number.isFinite(strength) ? strength : 1.0,
                model_strength: Number.isFinite(strength) ? strength : 1.0,
                clip_strength: Number.isFinite(clipStrength) ? clipStrength : (Number.isFinite(strength) ? strength : 1.0),
                active: lora?.active !== false,
                available,
                found,
                source: "workflow",
            };
        })
        .filter(Boolean);
}

function syncWorkflowLorasForDisplay(node, workflowData, fallbackLorasA = null, fallbackLorasB = null, options = null) {
    const preserveLocal = options?.preserveUserState !== false;
    const mappedPrimaryA = Array.isArray(workflowData?.loras_a) ? mapWorkflowLorasToUi(workflowData.loras_a) : [];
    const mappedPrimaryB = Array.isArray(workflowData?.loras_b) ? mapWorkflowLorasToUi(workflowData.loras_b) : [];
    const mappedFallbackA = mapWorkflowLorasToUi(fallbackLorasA || []);
    const mappedFallbackB = mapWorkflowLorasToUi(fallbackLorasB || []);

    const mergeFlagsFromFallback = (primary, fallback) => {
        const fallbackMap = new Map((fallback || []).map((l) => [String(l.name || "").toLowerCase(), l]));
        return (primary || []).map((lora) => {
            const fb = fallbackMap.get(String(lora.name || "").toLowerCase());
            return {
                ...lora,
                active: (lora.active !== undefined) ? (lora.active !== false) : (fb ? (fb.active !== false) : true),
                // Prefer freshly checked fallback flags when available.
                // workflow_data can carry stale availability from prior runs.
                available: fb
                    ? (fb.available !== false)
                    : ((lora.available !== undefined) ? (lora.available !== false) : true),
                found: fb
                    ? (fb.found !== false)
                    : ((lora.found !== undefined) ? (lora.found !== false) : true),
            };
        });
    };

    const wfLorasA = mappedPrimaryA.length > 0
        ? mergeFlagsFromFallback(mappedPrimaryA, mappedFallbackA)
        : mappedFallbackA;
    const wfLorasB = mappedPrimaryB.length > 0
        ? mergeFlagsFromFallback(mappedPrimaryB, mappedFallbackB)
        : mappedFallbackB;

    const preserveUserState = (incoming, existing) => {
        const existingMap = new Map((existing || []).map((l) => [String(l.name || "").toLowerCase(), l]));
        return incoming.map((lora) => {
            const prev = existingMap.get(String(lora.name || "").toLowerCase());
            const strength = Number(prev?.strength ?? prev?.model_strength ?? lora.strength ?? lora.model_strength ?? 1.0);
            const clipStrength = Number(prev?.clip_strength ?? lora.clip_strength ?? strength);
            return {
                ...lora,
                active: prev ? (prev.active !== false) : (lora.active !== false),
                strength: Number.isFinite(strength) ? strength : 1.0,
                model_strength: Number.isFinite(strength) ? strength : 1.0,
                clip_strength: Number.isFinite(clipStrength) ? clipStrength : (Number.isFinite(strength) ? strength : 1.0),
                source: "saved",
                fromWorkflow: true,
            };
        });
    };

    const toSavedWorkflow = (list) => (list || []).map((lora) => {
        const strength = Number(lora?.strength ?? lora?.model_strength ?? 1.0);
        const clipStrength = Number(lora?.clip_strength ?? lora?.strength ?? lora?.model_strength ?? strength);
        return {
            ...lora,
            active: lora?.active !== false,
            strength: Number.isFinite(strength) ? strength : 1.0,
            model_strength: Number.isFinite(strength) ? strength : 1.0,
            clip_strength: Number.isFinite(clipStrength) ? clipStrength : (Number.isFinite(strength) ? strength : 1.0),
            source: "saved",
            fromWorkflow: true,
        };
    });

    node.savedLorasA = preserveLocal ? preserveUserState(wfLorasA, node.savedLorasA) : toSavedWorkflow(wfLorasA);
    node.savedLorasB = preserveLocal ? preserveUserState(wfLorasB, node.savedLorasB) : toSavedWorkflow(wfLorasB);
    node.currentLorasA = wfLorasA.map((l) => ({ ...l, source: "current", fromWorkflow: true }));
    node.currentLorasB = wfLorasB.map((l) => ({ ...l, source: "current", fromWorkflow: true }));
}

function buildLiveWorkflowData(baseWorkflowData, promptText, lorasA, lorasB) {
    const base = (baseWorkflowData && typeof baseWorkflowData === "object" && !Array.isArray(baseWorkflowData))
        ? JSON.parse(JSON.stringify(baseWorkflowData))
        : {};

    base.positive_prompt = String(promptText || "");
    base.loras_a = (lorasA || []).map((l) => ({
        name: l.name,
        model_strength: Number(l.strength ?? l.model_strength ?? 1.0) || 1.0,
        clip_strength: Number(l.clip_strength ?? l.strength ?? l.model_strength ?? 1.0) || 1.0,
        active: l.active !== false,
        available: l.available !== false,
        found: l.found !== false,
    }));
    base.loras_b = (lorasB || []).map((l) => ({
        name: l.name,
        model_strength: Number(l.strength ?? l.model_strength ?? 1.0) || 1.0,
        clip_strength: Number(l.clip_strength ?? l.strength ?? l.model_strength ?? 1.0) || 1.0,
        active: l.active !== false,
        available: l.available !== false,
        found: l.found !== false,
    }));
    base._source = "PromptManagerAdvanced";
    return base;
}

async function applyLoraFoundState(loras) {
    const list = Array.isArray(loras) ? loras : [];
    const names = [...new Set(list.map((l) => String(l?.name || "").trim()).filter(Boolean))];
    if (names.length === 0) return list;

    try {
        const response = await fetch("/prompt-manager-advanced/check-loras", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lora_names: names }),
        });
        const data = await response.json();
        if (!data?.success || !data?.results) return list;

        return list.map((lora) => {
            const key = String(lora?.name || "").trim();
            const found = data.results[key] === true;
            return {
                ...lora,
                available: found,
                found,
            };
        });
    } catch {
        return list;
    }
}

async function pullWorkflowIntoNode(node) {
    const wfInput = node.inputs?.find((inp) => inp.name === "recipe_data");
    if (!wfInput || wfInput.link == null) {
        await showInfo("Workflow Data", "No workflow_data is connected.");
        return;
    }

    const wfData = resolveWorkflowDataForSave(node);
    if (!wfData || typeof wfData !== "object") {
        await showInfo("Workflow Data", "Could not resolve workflow_data from the connected source.");
        return;
    }

    if (!node._workflowPullSnapshot) {
        const textWidgetSnapshot = node.widgets?.find((w) => w.name === "text");
        node._workflowPullSnapshot = {
            text: textWidgetSnapshot?.value ?? "",
            savedLorasA: JSON.parse(JSON.stringify(node.savedLorasA || [])),
            savedLorasB: JSON.parse(JSON.stringify(node.savedLorasB || [])),
            currentLorasA: JSON.parse(JSON.stringify(node.currentLorasA || [])),
            currentLorasB: JSON.parse(JSON.stringify(node.currentLorasB || [])),
            lastWorkflowData: node.lastWorkflowData ? JSON.parse(JSON.stringify(node.lastWorkflowData)) : null,
        };
    }

    const textWidget = node.widgets?.find((w) => w.name === "text");
    const usePromptInputWidget = node.widgets?.find((w) => w.name === "use_prompt_input");
    const usePromptInput = usePromptInputWidget?.value === true;
    if (!usePromptInput && textWidget && wfData.positive_prompt) {
        textWidget.value = String(wfData.positive_prompt);
    }

    node.lastWorkflowData = wfData;
    syncSavedWorkflowDataWidget(node);

    const useLoraInputWidget = node.widgets?.find((w) => w.name === "use_lora_input");
    const useLoraInput = useLoraInputWidget?.value !== false;

    // Pull Workflow should only replace fields that are NOT controlled by live inputs.
    // If use_lora_input is ON, connected stacks control LoRAs at execution; keep PMA stacks unchanged.
    if (!useLoraInput) {
        let pulledA = mapWorkflowLorasToUi(wfData.loras_a || []).map((l) => ({ ...l, source: "saved", fromWorkflowPulled: true }));
        let pulledB = mapWorkflowLorasToUi(wfData.loras_b || []).map((l) => ({ ...l, source: "saved", fromWorkflowPulled: true }));
        pulledA = await applyLoraFoundState(pulledA);
        pulledB = await applyLoraFoundState(pulledB);

        const mergedSavedA = mergeLoraLists(node.savedLorasA || [], pulledA).map((l) => ({ ...l, source: "saved" }));
        const mergedSavedB = mergeLoraLists(node.savedLorasB || [], pulledB).map((l) => ({ ...l, source: "saved" }));

        node.savedLorasA = await applyLoraFoundState(mergedSavedA);
        node.savedLorasB = await applyLoraFoundState(mergedSavedB);
    }

    updateLoraDisplays(node);
    node.serialize_widgets = true;
    app.graph.setDirtyCanvas(true, true);
}

function clearPulledWorkflowFromNode(node) {
    if (node._workflowPullSnapshot) {
        const textWidget = node.widgets?.find((w) => w.name === "text");
        const usePromptInputWidget = node.widgets?.find((w) => w.name === "use_prompt_input");
        const usePromptInput = usePromptInputWidget?.value === true;

        if (!usePromptInput && textWidget) {
            textWidget.value = node._workflowPullSnapshot.text ?? "";
        }

        node.savedLorasA = node._workflowPullSnapshot.savedLorasA || [];
        node.savedLorasB = node._workflowPullSnapshot.savedLorasB || [];
        node.currentLorasA = node._workflowPullSnapshot.currentLorasA || [];
        node.currentLorasB = node._workflowPullSnapshot.currentLorasB || [];
        node.lastWorkflowData = node._workflowPullSnapshot.lastWorkflowData || null;
        delete node._workflowPullSnapshot;
    } else {
        node.lastWorkflowData = null;
    }

    syncSavedWorkflowDataWidget(node);

    updateLoraDisplays(node);
    node.serialize_widgets = true;
    app.graph.setDirtyCanvas(true, true);
}

function refreshPmaPromptGhosting(node) {
    const textWidget = node.widgets?.find((w) => w.name === "text");
    const useExternalWidget = node.widgets?.find((w) => w.name === "use_prompt_input");
    const useWorkflowWidget = node.widgets?.find((w) => w.name === "use_workflow_data");
    if (!textWidget || !useExternalWidget) return;

    const promptInputConnection = node.inputs?.find((inp) => inp.name === "prompt");
    const isLlmConnected = promptInputConnection && promptInputConnection.link != null;
    const workflowConnection = node.inputs?.find((inp) => inp.name === "recipe_data");
    const isWorkflowConnected = workflowConnection && workflowConnection.link != null;
    const useWorkflow = useWorkflowWidget?.value === true && isWorkflowConnected;

    if (useExternalWidget.value && isLlmConnected) {
        textWidget.disabled = true;
        if (textWidget.inputEl) {
            textWidget.inputEl.style.pointerEvents = "auto";
            textWidget.inputEl.readOnly = true;
        }
    } else if (useWorkflow) {
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

function getWorkflowDataLiveSig(workflowData) {
    if (!workflowData || typeof workflowData !== "object") return "";
    const normalize = (lora) => ({
        name: String(lora?.name || ""),
        model_strength: Number(lora?.model_strength ?? lora?.strength ?? 1.0) || 1.0,
        clip_strength: Number(lora?.clip_strength ?? lora?.strength ?? lora?.model_strength ?? 1.0) || 1.0,
        active: lora?.active !== false,
        available: lora?.available !== false,
        found: lora?.found !== false,
    });
    return JSON.stringify({
        positive_prompt: String(workflowData.positive_prompt || ""),
        loras_a: (Array.isArray(workflowData.loras_a) ? workflowData.loras_a : [])
            .map(normalize)
            .sort((a, b) => a.name.localeCompare(b.name)),
        loras_b: (Array.isArray(workflowData.loras_b) ? workflowData.loras_b : [])
            .map(normalize)
            .sort((a, b) => a.name.localeCompare(b.name)),
    });
}

async function tryLiveWorkflowPickup(node, { force = false } = {}) {
    const useWorkflowWidget = node.widgets?.find((w) => w.name === "use_workflow_data");
    const workflowConnection = node.inputs?.find((inp) => inp.name === "recipe_data");
    if (!useWorkflowWidget?.value || workflowConnection?.link == null) return false;

    const wfData = (await resolveWorkflowDataForLive(node)) || resolveWorkflowDataForSave(node);
    if (!wfData || typeof wfData !== "object") return false;

    const liveSig = getWorkflowDataLiveSig(wfData);
    if (!force && liveSig && node._lastLiveWorkflowPickupSig === liveSig) {
        return false;
    }

    let liveA = mapWorkflowLorasToUi(wfData.loras_a || []);
    let liveB = mapWorkflowLorasToUi(wfData.loras_b || []);
    liveA = await applyLoraFoundState(liveA);
    liveB = await applyLoraFoundState(liveB);

    syncWorkflowLorasForDisplay(node, wfData, liveA, liveB, { preserveUserState: false });
    node.lastWorkflowData = wfData;
    syncSavedWorkflowDataWidget(node);

    const useExternalWidget = node.widgets?.find((w) => w.name === "use_prompt_input");
    const textWidget = node.widgets?.find((w) => w.name === "text");
    if (textWidget && !(useExternalWidget?.value === true) && wfData.positive_prompt != null) {
        textWidget.value = String(wfData.positive_prompt);
    }

    node.currentTriggerWords = [];
    updateLoraDisplays(node);
    updateTriggerWordsDisplay(node);
    refreshPmaPromptGhosting(node);

    node._lastLiveWorkflowPickupSig = liveSig || null;
    node.serialize_widgets = true;
    app.graph.setDirtyCanvas(true, true);
    return true;
}

function setupWorkflowLivePickupHandler(node) {
    if (node._workflowLivePickupHandlerSetup) return;
    node._workflowLivePickupHandlerSetup = true;

    const getWorkflowInputLink = (n) => {
        const wfInput = n.inputs?.find((inp) => inp.name === "recipe_data");
        return wfInput?.link ?? null;
    };
    node._lastWorkflowInputLink = getWorkflowInputLink(node);

    const originalOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function () {
        if (originalOnConnectionsChange) {
            originalOnConnectionsChange.apply(this, arguments);
        }

        refreshPmaPromptGhosting(this);
        if (this._isWorkflowManager) {
            if (this.updatePromptSelectorDisplay) {
                this.updatePromptSelectorDisplay();
            }
            updateWorkflowManagerPreview(this);
        }

        // Only trigger live workflow pickup when the workflow_data input link changes.
        // Unrelated connection edits (e.g. output preview nodes) must not overwrite
        // user tweaks by re-ingesting upstream workflow payload.
        const currentWorkflowLink = getWorkflowInputLink(this);
        const workflowLinkChanged = currentWorkflowLink !== this._lastWorkflowInputLink;
        this._lastWorkflowInputLink = currentWorkflowLink;
        if (workflowLinkChanged) {
            this._lastLiveWorkflowPickupSig = null;

            if (this._isWorkflowManager && currentWorkflowLink != null) {
                // New input connection: clear local cached workflow state so
                // upstream workflow_data becomes authoritative.
                this.lastWorkflowData = null;
                this.savedLorasA = [];
                this.savedLorasB = [];
                this.currentLorasA = [];
                this.currentLorasB = [];
                this.savedTriggerWords = [];
                this.currentTriggerWords = [];

                const textWidget = this.widgets?.find((w) => w.name === "text");
                if (textWidget) textWidget.value = "";

                syncSavedWorkflowDataWidget(this);
                updateLoraDisplays(this);
                updateTriggerWordsDisplay(this);
                updateWorkflowManagerPreview(this);
            }

            void tryLiveWorkflowPickup(this);
        }
    };
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
        useLoraInputWidget.callback = function(value) {
            if (originalLoraInputCallback) {
                originalLoraInputCallback.apply(this, arguments);
            }

            if (!value) {
                // Switching OFF: clear current (connected) loras AND 
                // remove any saved loras that came from input (fromInput: true)
                node.currentLorasA = [];
                node.currentLorasB = [];
                node.currentTriggerWords = [];
                
                // Keep only saved loras that are from presets (not from input)
                node.savedLorasA = (node.savedLorasA || []).filter(lora => !lora.fromInput);
                node.savedLorasB = (node.savedLorasB || []).filter(lora => !lora.fromInput);
            }
            // When switching ON, keep everything - the update event will sync with input

            // Update display to reflect the new state
            updateLoraDisplays(node);
            updateTriggerWordsDisplay(node);
        };
    }

    // Apply initial state on load/reload
    const applyToggleState = (value) => {
        const promptInputConnection = node.inputs?.find(inp => inp.name === "prompt");
        const isLlmConnected = promptInputConnection && promptInputConnection.link != null;

        // Also check use_workflow_data toggle
        const useWorkflowWidget = node.widgets?.find(w => w.name === "use_workflow_data");
        const workflowConnection = node.inputs?.find(inp => inp.name === "recipe_data");
        const isWorkflowConnected = workflowConnection && workflowConnection.link != null;
        const useWorkflow = useWorkflowWidget?.value && isWorkflowConnected;

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
        } else if (useWorkflow) {
            // Using workflow_data — ghost the text widget
            textWidget.disabled = true;
            if (textWidget.inputEl) {
                textWidget.inputEl.style.pointerEvents = "auto";
                textWidget.inputEl.readOnly = true;
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
        const promptInputConnection = node.inputs?.find(inp => inp.name === "prompt");
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

function setupUseWorkflowToggleHandler(node) {
    const useWorkflowWidget = node.widgets?.find(w => w.name === "use_workflow_data");
    const textWidget = node.widgets?.find(w => w.name === "text");
    if (!useWorkflowWidget || !textWidget) return;

    const cloneJson = (value, fallback) => {
        try {
            return JSON.parse(JSON.stringify(value ?? fallback));
        } catch {
            return fallback;
        }
    };

    const snapshotPreWorkflowState = () => {
        node._preWorkflowModeState = {
            text: textWidget?.value ?? "",
            savedLorasA: cloneJson(node.savedLorasA, []),
            savedLorasB: cloneJson(node.savedLorasB, []),
            currentLorasA: cloneJson(node.currentLorasA, []),
            currentLorasB: cloneJson(node.currentLorasB, []),
            savedTriggerWords: cloneJson(node.savedTriggerWords, []),
            currentTriggerWords: cloneJson(node.currentTriggerWords, []),
            lastWorkflowData: cloneJson(node.lastWorkflowData, null),
        };
    };

    const restorePromptInputText = () => {
        const promptInputConnection = node.inputs?.find(inp => inp.name === "prompt");
        const isLlmConnected = promptInputConnection && promptInputConnection.link != null;
        if (!isLlmConnected || !textWidget) return;

        const graph = app.graph;
        const link = graph.links[promptInputConnection.link];
        if (!link) return;

        const originNode = graph.getNodeById(link.origin_id);
        if (!originNode) return;

        const outputData = originNode.getOutputData?.(link.origin_slot);
        if (outputData !== undefined) {
            textWidget.value = outputData;
            return;
        }

        if (originNode.widgets) {
            const outputWidget = originNode.widgets.find(w => w.name === "text" || w.name === "STRING");
            if (outputWidget) {
                textWidget.value = outputWidget.value;
            }
        }
    };

    // Re-use the existing applyToggleState from use_prompt_input handler
    // by triggering a refresh through that widget's state
    const refreshGhosting = () => {
        refreshPmaPromptGhosting(node);
    };

    // Apply initial state
    refreshGhosting();

    const originalCallback = useWorkflowWidget.callback;
    useWorkflowWidget.callback = async function(value) {
        // Prevent turning on if workflow_data is not connected
        const workflowConnection = node.inputs?.find(inp => inp.name === "recipe_data");
        const isConnected = workflowConnection && workflowConnection.link != null;

        if (value && !isConnected) {
            useWorkflowWidget.value = false;
            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);
            return;
        }

        if (originalCallback) {
            originalCallback.apply(this, arguments);
        }

        // Force next execution update to run even if payload content is unchanged.
        // This is required when toggling workflow mode OFF then ON, otherwise the
        // signature guard can incorrectly skip re-applying found/available flags.
        node._lastExecutionUpdateSig = null;
        node._lastLiveWorkflowPickupSig = null;

        if (value && !node._preWorkflowModeState) {
            snapshotPreWorkflowState();
        }

        if (!value) {
            // Restore pre-workflow state first, then re-apply active toggle behavior.
            node.lastWorkflowData = null;
            syncSavedWorkflowDataWidget(node);

            const useExternalWidget = node.widgets?.find((w) => w.name === "use_prompt_input");
            const useLoraInputWidget = node.widgets?.find((w) => w.name === "use_lora_input");
            const usePromptInput = useExternalWidget?.value === true;
            const useLoraInput = useLoraInputWidget?.value !== false;

            const pre = node._preWorkflowModeState;
            if (pre) {
                node.savedLorasA = cloneJson(pre.savedLorasA, []);
                node.savedLorasB = cloneJson(pre.savedLorasB, []);
                node.savedTriggerWords = cloneJson(pre.savedTriggerWords, []);

                if (useLoraInput) {
                    // Restore prompt stacks + connected input stacks when lora input is enabled.
                    node.currentLorasA = cloneJson(pre.currentLorasA, []);
                    node.currentLorasB = cloneJson(pre.currentLorasB, []);
                    node.currentTriggerWords = cloneJson(pre.currentTriggerWords, []);
                } else {
                    node.currentLorasA = [];
                    node.currentLorasB = [];
                    node.currentTriggerWords = [];
                }

                // Intentionally do not overwrite text when use_prompt_input is OFF.
                // Keep the current workflow text so it remains editable after un-ghosting.
            }

            const categoryWidget = node.widgets?.find((w) => w.name === "category");
            const promptWidget = node.widgets?.find((w) => w.name === "name");
            if (!pre) {
                // Fallback when no prompt is selected: only remove workflow-derived entries.
                node.currentLorasA = (node.currentLorasA || []).filter((l) => !l.fromWorkflow);
                node.currentLorasB = (node.currentLorasB || []).filter((l) => !l.fromWorkflow);
                node.savedLorasA = (node.savedLorasA || []).filter((l) => !l.fromWorkflow);
                node.savedLorasB = (node.savedLorasB || []).filter((l) => !l.fromWorkflow);
            }

            if (usePromptInput) {
                restorePromptInputText();
            }

            delete node._preWorkflowModeState;
            updateLoraDisplays(node);
            updateTriggerWordsDisplay(node);
        } else {
            // Live pickup with the same ingest path used for execution data.
            void tryLiveWorkflowPickup(node, { force: true });
        }

        refreshGhosting();

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
        node.lastWorkflowData = null;
        syncSavedWorkflowDataWidget(node);
        updateLoraDisplays(node);
        updateTriggerWordsDisplay(node);
        return;
    }

    const promptData = node.prompts[category][promptName];

    if (textWidget) {
        textWidget.value = promptData.prompt || "";
    }

    // Restore saved workflow_data if present
    node.lastWorkflowData = promptData.workflow_data || null;
    syncSavedWorkflowDataWidget(node);

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
    if (node?._isWorkflowManager) {
        return false;
    }

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

async function createCategory(node, categoryName, nsfw = false) {
    try {
        const response = await fetch("/prompt-manager-advanced/save-category", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ category_name: categoryName, nsfw: nsfw })
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

async function renameCategory(node, oldCategory, newCategory) {
    try {
        const response = await fetch("/prompt-manager-advanced/rename-category", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ old_category: oldCategory, new_category: newCategory })
        });

        const data = await response.json();

        if (data.success) {
            node.prompts = data.prompts;

            const categoryWidget = node.widgets.find(w => w.name === "category");
            const promptWidget = node.widgets.find(w => w.name === "name");

            // Update to the renamed category
            if (categoryWidget) {
                categoryWidget.value = data.new_category;
            }

            // Keep current prompt if it exists
            const currentPrompt = promptWidget?.value || "";
            const categoryPrompts = node.prompts[data.new_category] || {};
            if (currentPrompt && currentPrompt in categoryPrompts) {
                if (promptWidget) {
                    promptWidget.value = currentPrompt;
                }
            }

            updateDropdowns(node);

            // Update custom prompt selector display
            if (node.updatePromptSelectorDisplay) {
                node.updatePromptSelectorDisplay();
            }

            node.serialize_widgets = true;
            app.graph.setDirtyCanvas(true, true);

            await showInfo("Success", `Category renamed from "${oldCategory}" to "${newCategory}"`);
        } else {
            await showInfo("Error", data.error);
        }
    } catch (error) {
        console.error("[PromptManagerAdvanced] Error renaming category:", error);
        await showInfo("Error", "Error renaming category");
    }
}

function parseJsonObjectSafe(value, fallback = null) {
    if (!value) return fallback;
    if (typeof value === "object") return value;
    if (typeof value !== "string") return fallback;
    const t = value.trim();
    if (!t) return fallback;
    try {
        const parsed = JSON.parse(t);
        return (parsed && typeof parsed === "object") ? parsed : fallback;
    } catch {
        return fallback;
    }
}

function resolveUpstreamNodeThroughReroutes(graph, startLinkId, maxHops = 24) {
    if (!graph || startLinkId == null) return null;
    let linkId = startLinkId;
    const seen = new Set();

    const isRerouteNode = (n) => {
        if (!n) return false;
        const cc = String(n.comfyClass || "").toLowerCase();
        const ty = String(n.type || "").toLowerCase();
        return cc.includes("reroute") || ty.includes("reroute");
    };

    for (let hop = 0; hop < maxHops; hop++) {
        if (linkId == null || seen.has(linkId)) break;
        seen.add(linkId);

        const linkInfo = graph.links?.[linkId];
        if (!linkInfo) break;
        const srcNode = graph.getNodeById?.(linkInfo.origin_id);
        if (!srcNode) break;

        if (!isRerouteNode(srcNode)) return srcNode;
        const in0 = srcNode.inputs?.[0];
        linkId = in0?.link ?? null;
    }
    return null;
}

function normalizeBuilderLoraList(list) {
    if (!Array.isArray(list)) return [];
    return list
        .map((item) => {
            const name = String(item?.name || "").trim();
            if (!name) return null;
            const modelStrength = Number(item?.model_strength ?? item?.strength ?? 1.0);
            const clipStrength = Number(item?.clip_strength ?? item?.strength ?? modelStrength);
            return {
                name,
                model_strength: Number.isFinite(modelStrength) ? modelStrength : 1.0,
                clip_strength: Number.isFinite(clipStrength) ? clipStrength : 1.0,
                active: item?.active !== false,
            };
        })
        .filter(Boolean);
}

function buildWorkflowDataFromBuilderNode(builderNode) {
    if (!builderNode) return null;

    const overrideWidget = builderNode.widgets?.find((w) => w.name === "override_data");
    const ov =
        parseJsonObjectSafe(overrideWidget?.value, null) ||
        parseJsonObjectSafe(builderNode.properties?.we_ui_state, null) ||
        parseJsonObjectSafe(builderNode.properties?.we_override_data, null) ||
        null;

    if (!ov || typeof ov !== "object") return null;

    const clipNames = Array.isArray(ov.clip_names)
        ? ov.clip_names.filter((x) => !!x)
        : (ov.clip_names ? [ov.clip_names] : []);

    const workflowData = {
        _source: "RecipeBuilder",
        family: ov._family || "sdxl",
        model_a: ov.model_a || "",
        model_b: ov.model_b || "",
        positive_prompt: ov.positive_prompt || "",
        negative_prompt: ov.negative_prompt || "",
        loras_a: normalizeBuilderLoraList(ov.loras_a),
        loras_b: normalizeBuilderLoraList(ov.loras_b),
        vae: ov.vae || "",
        clip: clipNames,
        clip_type: ov.clip_type || "",
        loader_type: ov.loader_type || "",
        sampler: {
            steps_a: ov.steps_a ?? 20,
            steps_b: ov.steps_b ?? null,
            cfg: ov.cfg ?? 5,
            seed_a: ov.seed_a ?? 0,
            sampler_name: ov.sampler_name || "euler",
            scheduler: ov.scheduler || "simple",
            seed_b: ov.seed_b ?? null,
        },
        resolution: {
            width: ov.width ?? 768,
            height: ov.height ?? 1280,
            batch_size: ov.batch_size ?? 1,
            length: ov.length ?? null,
        },
    };

    return workflowData;
}

function buildWorkflowDataFromExtractorNode(extractorNode) {
    if (!extractorNode) return null;

    const extracted = parseJsonObjectSafe(extractorNode.properties?.pe_extracted_data, null);
    if (!extracted || typeof extracted !== "object") return null;

    const clipNames = Array.isArray(extracted.clip?.names)
        ? extracted.clip.names.filter((x) => !!x)
        : (Array.isArray(extracted.clip) ? extracted.clip.filter((x) => !!x) : []);

    return {
        _source: extractorNode.comfyClass || extractorNode.type || "PromptExtractor",
        family: extracted.model_family || extracted.family || "sdxl",
        model_a: extracted.model_a || "",
        model_b: extracted.model_b || "",
        positive_prompt: extracted.positive_prompt || "",
        negative_prompt: extracted.negative_prompt || "",
        loras_a: normalizeBuilderLoraList(extracted.loras_a),
        loras_b: normalizeBuilderLoraList(extracted.loras_b),
        vae: extracted.vae?.name || extracted.vae || "",
        clip: clipNames,
        clip_type: extracted.clip?.type || extracted.clip_type || "",
        loader_type: extracted.loader_type || "",
        sampler: extracted.sampler || {
            steps_a: 20,
            steps_b: null,
            cfg: 5,
            seed_a: 0,
            sampler_name: "euler",
            scheduler: "simple",
            seed_b: null,
        },
        resolution: extracted.resolution || {
            width: 768,
            height: 1280,
            batch_size: 1,
            length: null,
        },
    };
}

function resolveWorkflowDataForSave(node) {
    const wfInput = node.inputs?.find((inp) => inp.name === "recipe_data");
    if (wfInput?.link != null) {
        const upstream = resolveUpstreamNodeThroughReroutes(node.graph, wfInput.link);
        const sourceClass = upstream?.comfyClass || upstream?.type || "";
        if (sourceClass === "RecipeBuilder" || sourceClass === "RecipeBuilderWan") {
            const wfOutIdx = upstream?.outputs?.findIndex((o) => o.name === "recipe_data");
            if (wfOutIdx >= 0) {
                const out = upstream.outputs[wfOutIdx];
                const data = out?._data ?? out?.value ?? null;
                if (hasMeaningfulWorkflowData(data)) return data;
                if (typeof data === "string") {
                    const parsed = parseJsonObjectSafe(data, null);
                    if (hasMeaningfulWorkflowData(parsed)) return parsed;
                }
            }

            const fromBuilder = buildWorkflowDataFromBuilderNode(upstream);
            if (hasMeaningfulWorkflowData(fromBuilder)) return fromBuilder;
        }

        if (sourceClass === "PromptExtractor" || sourceClass === "RecipeExtractor") {
            const fromExtractor = buildWorkflowDataFromExtractorNode(upstream);
            if (hasMeaningfulWorkflowData(fromExtractor)) return fromExtractor;
        }

        const wfOutIdx = upstream?.outputs?.findIndex((o) => o.name === "recipe_data");
        if (wfOutIdx >= 0) {
            const out = upstream.outputs[wfOutIdx];
            const data = out?._data ?? out?.value ?? null;
            if (hasMeaningfulWorkflowData(data)) return data;
            if (typeof data === "string") {
                const parsed = parseJsonObjectSafe(data, null);
                if (hasMeaningfulWorkflowData(parsed)) return parsed;
            }
        }

        // Connected input but no fresh upstream workflow_data: for Recipe Manager,
        // allow fallback to the latest live payload captured from execution updates.
        if (node?._isWorkflowManager && hasMeaningfulWorkflowData(node.lastWorkflowData)) {
            return node.lastWorkflowData;
        }

        // For non-RecipeManager nodes, avoid saving stale local cache from
        // a previously selected prompt while connected.
        return null;
    }

    return hasMeaningfulWorkflowData(node.lastWorkflowData) ? node.lastWorkflowData : null;
}

async function resolveWorkflowDataForLive(node) {
    const wfInput = node.inputs?.find((inp) => inp.name === "recipe_data");
    if (wfInput?.link == null) return null;

    const upstream = resolveUpstreamNodeThroughReroutes(node.graph, wfInput.link);
    if (!upstream) return null;

    const sourceClass = String(upstream?.comfyClass || upstream?.type || "");
    const sourceClassLower = sourceClass.toLowerCase();

    if (sourceClass === "RecipeBuilder" || sourceClass === "RecipeBuilderWan") {
        const wfOutIdx = upstream?.outputs?.findIndex((o) => o.name === "recipe_data");
        if (wfOutIdx >= 0) {
            const out = upstream.outputs[wfOutIdx];
            const data = out?._data ?? out?.value ?? null;
            if (hasMeaningfulWorkflowData(data)) return data;
            if (typeof data === "string") {
                const parsed = parseJsonObjectSafe(data, null);
                if (hasMeaningfulWorkflowData(parsed)) return parsed;
            }
        }

        return buildWorkflowDataFromBuilderNode(upstream);
    }

    if (sourceClassLower === "promptextractor" || sourceClassLower === "recipeextractor") {
        const fromExtractorCache = buildWorkflowDataFromExtractorNode(upstream);
        if (fromExtractorCache) return fromExtractorCache;

        const imageWidget = upstream.widgets?.find((w) => w.name === "image");
        const sourceWidget = upstream.widgets?.find((w) => w.name === "source_folder");
        const filename = String(imageWidget?.value || "").trim();
        const source = String(sourceWidget?.value || "input").trim() || "input";
        if (filename && filename !== "(none)") {
            try {
                const resp = await fetch(
                    `/prompt-extractor/extract-preview?filename=${encodeURIComponent(filename)}&source=${encodeURIComponent(source)}`
                );
                if (resp.ok) {
                    const data = await resp.json();
                    const extracted = data?.extracted || null;
                    if (extracted && typeof extracted === "object") {
                        upstream.properties = upstream.properties || {};
                        upstream.properties.pe_extracted_data = JSON.stringify(extracted);
                        const fromPreview = buildWorkflowDataFromExtractorNode(upstream);
                        if (fromPreview) return fromPreview;
                    }
                }
            } catch {
                // Keep silent in live mode; execution update remains authoritative fallback.
            }
        }
    }

    const wfOutIdx = upstream?.outputs?.findIndex((o) => o.name === "recipe_data");
    if (wfOutIdx >= 0) {
        const out = upstream.outputs[wfOutIdx];
        const data = out?._data ?? out?.value ?? null;
        if (data && typeof data === "object") return data;
        if (typeof data === "string") {
            const parsed = parseJsonObjectSafe(data, null);
            if (parsed) return parsed;
        }
    }

    return null;
}

async function savePrompt(node, category, name, text, lorasA, lorasB, triggerWords, thumbnail = null, nsfw = false) {
    try {
        const requestBody = {
            category: category,
            name: name,
            text: text,
            nsfw: nsfw,
            // Save loras with their active state
            loras_a: lorasA.map(l => ({
                name: l.name,
                strength: l.strength ?? l.model_strength ?? 1.0,
                clip_strength: l.clip_strength ?? l.strength ?? 1.0,
                active: l.active !== false
            })),
            loras_b: lorasB.map(l => ({
                name: l.name,
                strength: l.strength ?? l.model_strength ?? 1.0,
                clip_strength: l.clip_strength ?? l.strength ?? 1.0,
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

        // Include workflow_data when available.
        // If use_workflow_data is enabled and workflow_data comes from a connected
        // RecipeBuilder / RecipeBuilderWan, pull it directly from Builder UI state so saving works
        // even before executing the graph.
        const hasWorkflowInput = hasConnectedWorkflowInput(node);
        const liveWorkflowData = hasWorkflowInput ? await resolveWorkflowDataForLive(node) : null;
        const workflowDataForSave = hasMeaningfulWorkflowData(liveWorkflowData)
            ? liveWorkflowData
            : resolveWorkflowDataForSave(node);

        if (node?._isWorkflowManager && hasWorkflowInput && !hasMeaningfulWorkflowData(workflowDataForSave)) {
            await showInfo("Workflow Data", "No live workflow_data available to save yet. Execute upstream nodes and try again.");
            return;
        }
        if (workflowDataForSave) {
            const effectivePromptText = (
                node?._isWorkflowManager &&
                hasWorkflowInput &&
                typeof workflowDataForSave?.positive_prompt === "string" &&
                workflowDataForSave.positive_prompt.trim().length > 0
            )
                ? workflowDataForSave.positive_prompt
                : text;

            const liveWorkflowData = buildLiveWorkflowData(workflowDataForSave, effectivePromptText, lorasA, lorasB);
            if (node?._isWorkflowManager) {
                liveWorkflowData._source = "RecipeManager";
            }
            node.lastWorkflowData = liveWorkflowData;
            syncSavedWorkflowDataWidget(node);
            requestBody.workflow_data = liveWorkflowData;
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
            
            // Clear current lora and trigger word data
            node.savedLorasA = [];
            node.savedLorasB = [];
            node.currentLorasA = [];
            node.currentLorasB = [];
            node.originalStrengthsA = {};
            node.originalStrengthsB = {};
            node.savedTriggerWords = [];
            node.currentTriggerWords = [];
            
            const categoryWidget = node.widgets.find(w => w.name === "category");
            const promptWidget = node.widgets.find(w => w.name === "name");
            const textWidget = node.widgets.find(w => w.name === "text");

            // Update dropdowns to reflect deletion
            updateDropdowns(node);

            // Load the new prompt data (first prompt in first category, or empty)
            const currentCategory = categoryWidget?.value;
            const newPrompt = promptWidget?.value || "";
            
            if (currentCategory && newPrompt && node.prompts[currentCategory]?.[newPrompt]) {
                // Load the new category's first prompt
                await loadPromptData(node, currentCategory, newPrompt);
            } else {
                // No categories/prompts left - clear everything
                if (textWidget) textWidget.value = "";
                updateLoraDisplays(node);
                updateTriggerWordsDisplay(node);
            }

            // Update previous values tracking
            node._previousCategory = currentCategory;
            node._previousPrompt = newPrompt;

            // Update custom prompt selector display
            if (node.updatePromptSelectorDisplay) {
                node.updatePromptSelectorDisplay();
            }

            // Update last saved state
            updateLastSavedState(node);

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
            
            // Clear current lora and trigger word data
            node.savedLorasA = [];
            node.savedLorasB = [];
            node.currentLorasA = [];
            node.currentLorasB = [];
            node.originalStrengthsA = {};
            node.originalStrengthsB = {};
            node.savedTriggerWords = [];
            node.currentTriggerWords = [];
            
            const categoryWidget = node.widgets.find(w => w.name === "category");
            const promptWidget = node.widgets.find(w => w.name === "name");
            const textWidget = node.widgets.find(w => w.name === "text");

            // Update dropdowns to reflect deletion
            updateDropdowns(node);

            // Load the new prompt data (first prompt in category, or empty)
            const currentCategory = categoryWidget?.value;
            const newPrompt = promptWidget?.value || "";
            
            if (currentCategory && newPrompt && node.prompts[currentCategory]?.[newPrompt]) {
                // Load the new prompt's data
                await loadPromptData(node, currentCategory, newPrompt);
            } else {
                // No prompts left in category - clear everything
                if (textWidget) textWidget.value = "";
                updateLoraDisplays(node);
                updateTriggerWordsDisplay(node);
            }

            // Update previous values tracking
            node._previousCategory = currentCategory;
            node._previousPrompt = newPrompt;

            // Update custom prompt selector display
            if (node.updatePromptSelectorDisplay) {
                node.updatePromptSelectorDisplay();
            }

            // Update last saved state
            updateLastSavedState(node);

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
        const promptNames = Object.keys(node.prompts[currentCategory]).filter(k => k !== '__meta__').sort((a, b) => a.localeCompare(b));

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

function showRenameCategoryDialog(title, message, categories, defaultCategory) {
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
            <div style="margin-bottom: 6px; color: #aaa; font-size: 12px;">Category to rename:</div>
            <select style="width: 100%; padding: 8px; margin-bottom: 12px; background: #333; border: 1px solid #555; color: #fff; border-radius: 4px; font-size: 14px;">
                ${categoryOptions}
            </select>
            <div style="margin-bottom: 6px; color: #aaa; font-size: 12px;">${message}</div>
            <input type="text" value="${defaultCategory}" style="width: 100%; padding: 8px; margin-bottom: 15px; background: #333; border: 1px solid #555; color: #fff; border-radius: 4px; font-size: 14px; box-sizing: border-box;" />
            <div style="display: flex; gap: 10px; justify-content: flex-end;">
                <button class="cancel-btn" style="padding: 8px 16px; background: #555; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
                <button class="ok-btn" style="padding: 8px 16px; background: #0a0; color: #fff; border: none; border-radius: 4px; cursor: pointer;">OK</button>
            </div>
        `;

        const selectEl = dialog.querySelector("select");
        const input = dialog.querySelector("input");
        const okBtn = dialog.querySelector(".ok-btn");
        const cancelBtn = dialog.querySelector(".cancel-btn");

        // Update input when category selection changes
        selectEl.onchange = () => {
            input.value = selectEl.value;
            input.select();
        };

        const cleanup = () => {
            document.body.removeChild(overlay);
            document.body.removeChild(dialog);
        };

        const handleOk = () => {
            resolve({ oldCategory: selectEl.value, newCategory: input.value });
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

function showPromptWithCategory(title, message, defaultName, categories, defaultCategory, defaultNsfw = false) {
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

        // Build category options — sanitize for safe HTML
        const categoryOptions = categories.map(cat => {
            const opt = document.createElement("option");
            opt.value = cat;
            opt.textContent = cat;
            return opt.outerHTML;
        }).join('');

        dialog.innerHTML = `
            <div style="margin-bottom: 15px; font-size: 16px; font-weight: bold; color: #fff;">${title}</div>
            <div style="margin-bottom: 6px; color: #aaa; font-size: 12px;">Category:</div>
            <select style="width: 100%; padding: 8px; margin-bottom: 12px; background: #333; border: 1px solid #555; color: #fff; border-radius: 4px; font-size: 14px;">
                ${categoryOptions}
            </select>
            <div style="margin-bottom: 6px; color: #aaa; font-size: 12px;">${message}</div>
            <input type="text" style="width: 100%; padding: 8px; margin-bottom: 12px; background: #333; border: 1px solid #555; color: #fff; border-radius: 4px; font-size: 14px; box-sizing: border-box;" />
            <label style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px; color: #aaa; font-size: 13px; cursor: pointer; user-select: none;">
                <input type="checkbox" class="nsfw-cb" style="cursor: pointer; accent-color: #c44;" />
                Mark as NSFW
            </label>
            <div style="display: flex; gap: 10px; justify-content: flex-end;">
                <button class="cancel-btn" style="padding: 8px 16px; background: #555; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
                <button class="ok-btn" style="padding: 8px 16px; background: #0a0; color: #fff; border: none; border-radius: 4px; cursor: pointer;">OK</button>
            </div>
        `;

        const selectEl = dialog.querySelector("select");
        const input = dialog.querySelector("input[type='text']");
        const nsfwCb = dialog.querySelector(".nsfw-cb");
        const okBtn = dialog.querySelector(".ok-btn");
        const cancelBtn = dialog.querySelector(".cancel-btn");

        // Set defaults after DOM is built
        selectEl.value = defaultCategory;
        input.value = defaultName;
        nsfwCb.checked = defaultNsfw;

        const cleanup = () => {
            document.body.removeChild(overlay);
            document.body.removeChild(dialog);
        };

        const handleOk = () => {
            resolve({ name: input.value, category: selectEl.value, nsfw: nsfwCb.checked });
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

function showNewCategoryDialog() {
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
            <div style="margin-bottom: 15px; font-size: 16px; font-weight: bold; color: #fff;">New Category</div>
            <div style="margin-bottom: 10px; color: #ccc;">Enter new category name:</div>
            <input type="text" value="" style="width: 100%; padding: 8px; margin-bottom: 12px; background: #333; border: 1px solid #555; color: #fff; border-radius: 4px; font-size: 14px; box-sizing: border-box;" />
            <label style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px; color: #aaa; font-size: 13px; cursor: pointer; user-select: none;">
                <input type="checkbox" class="nsfw-cb" style="cursor: pointer; accent-color: #c44;" />
                Mark as NSFW
            </label>
            <div style="display: flex; gap: 10px; justify-content: flex-end;">
                <button class="cancel-btn" style="padding: 8px 16px; background: #555; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
                <button class="ok-btn" style="padding: 8px 16px; background: #0a0; color: #fff; border: none; border-radius: 4px; cursor: pointer;">OK</button>
            </div>
        `;

        const input = dialog.querySelector("input[type='text']");
        const nsfwCb = dialog.querySelector(".nsfw-cb");
        const okBtn = dialog.querySelector(".ok-btn");
        const cancelBtn = dialog.querySelector(".cancel-btn");

        const cleanup = () => {
            document.body.removeChild(overlay);
            document.body.removeChild(dialog);
        };

        const handleOk = () => {
            resolve({ name: input.value, nsfw: nsfwCb.checked });
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
    });
}

function showConfirm(title, message, confirmText = "Delete", confirmColor = "#c00", useOverlay = true) {
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
            if (overlay.parentNode) {
                document.body.removeChild(overlay);
            }
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

        if (useOverlay) {
            document.body.appendChild(overlay);
        }
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
function resizeImageToThumbnail(file, minSize = 200) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Calculate new dimensions maintaining aspect ratio
                // Smallest dimension should be minSize (200px)
                let width = img.width;
                let height = img.height;
                const minDim = Math.min(width, height);

                if (minDim > minSize) {
                    // Scale down so smallest dimension = minSize
                    const scale = minSize / minDim;
                    width = Math.round(width * scale);
                    height = Math.round(height * scale);
                }
                // Note: We don't scale UP if image is smaller than minSize

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
async function showThumbnailBrowser(node, currentCategory, currentPrompt, options = {}) {
    // Reload prompts to ensure we have the latest data
    await loadPrompts(node);

    const mode = options?.mode === "save" ? "save" : "select";
    const onSave = typeof options?.onSave === "function" ? options.onSave : null;
    const saveButtonText = options?.saveButtonText || "Save";
    const saveTitle = options?.title || "Save Workflow";
    const saveNamePlaceholder = options?.namePlaceholder || "Prompt name";
    const initialSaveName = typeof options?.initialName === "string" ? options.initialName : "";
    const workflowOnly = options?.workflowOnly === true || node?._isWorkflowManager === true;
    
    // Check if thumbnail preview is enabled from user preferences
    const previewEnabled = getThumbnailPreviewEnabled();
    
    return new Promise((resolve) => {
        // Clean up any stale preview elements from previous modal openings
        const stalePreviews = document.querySelectorAll('[data-pm-thumbnail-preview]');
        stalePreviews.forEach(preview => {
            if (preview.parentNode) {
                preview.parentNode.removeChild(preview);
            }
        });
        
        let selectedCategory = currentCategory;

        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: ${UI.overlay};
            z-index: 9999;
        `;

        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: ${UI.panel};
            border: 1px solid ${UI.panelBorder};
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
            border-bottom: 1px solid ${UI.sectionBorder};
        `;

        const title = document.createElement("div");
        title.textContent = mode === "save"
            ? saveTitle
            : (workflowOnly ? "Select Recipe" : "Select Prompt");
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

        // Controls bar: Search + NSFW button + View Mode button
        const controlsBar = document.createElement("div");
        controlsBar.style.cssText = `
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid ${UI.sectionBorder};
        `;

        // Search input (fills remaining space)
        const searchInput = document.createElement("input");
        searchInput.type = "text";
        searchInput.placeholder = workflowOnly ? "Search workflows..." : "Search prompts...";
        searchInput.style.cssText = `
            flex: 1;
            min-width: 0;
            padding: 6px 10px;
            background: ${UI.inputBg};
            border: 1px solid ${UI.inputBorder};
            border-radius: 4px;
            color: #fff;
            font-size: 13px;
            box-sizing: border-box;
            outline: none;
        `;
        searchInput.onfocus = () => searchInput.style.borderColor = UI.accent;
        searchInput.onblur = () => searchInput.style.borderColor = UI.inputBorder;

        // Wrap search input in a container with clear button
        const searchWrapper = document.createElement("div");
        searchWrapper.style.cssText = `
            flex: 1;
            min-width: 0;
            position: relative;
            display: flex;
            align-items: center;
        `;
        searchInput.style.flex = "1";
        searchInput.style.paddingRight = "28px";
        const clearBtn = document.createElement("span");
        clearBtn.textContent = "\u00d7";
        clearBtn.title = "Clear search";
        clearBtn.style.cssText = `
            position: absolute;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
            color: #666;
            font-size: 16px;
            cursor: pointer;
            line-height: 1;
            display: none;
            user-select: none;
        `;
        clearBtn.onmouseover = () => clearBtn.style.color = "#fff";
        clearBtn.onmouseout = () => clearBtn.style.color = "#666";
        clearBtn.onclick = () => {
            searchInput.value = "";
            clearBtn.style.display = "none";
            searchInput.focus();
            renderContent("");
        };
        const origOninput = searchInput.oninput;
        searchInput.addEventListener("input", () => {
            clearBtn.style.display = searchInput.value ? "" : "none";
        });
        searchWrapper.appendChild(searchInput);
        searchWrapper.appendChild(clearBtn);

        // NSFW toggle button
        let hideNSFWState = getHideNSFW();
        const nsfwBtn = document.createElement("button");
        const btnStyle = `
            background: ${UI.buttonBg};
            border: 1px solid ${UI.inputBorder};
            color: #aaa;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            white-space: nowrap;
        `;
        const updateNsfwBtn = () => {
            if (hideNSFWState) {
                nsfwBtn.textContent = "NSFW: Hidden";
                nsfwBtn.style.cssText = btnStyle + `background: #453339; border-color: #9b727b; color: #d2a8b0;`;
            } else {
                nsfwBtn.textContent = "NSFW: Visible";
                nsfwBtn.style.cssText = btnStyle;
            }
            nsfwBtn.title = hideNSFWState ? "NSFW content is hidden — click to show" : "NSFW content is visible — click to hide";
        };
        updateNsfwBtn();
        nsfwBtn.onmouseover = () => { if (!hideNSFWState) { nsfwBtn.style.background = '#38414c'; nsfwBtn.style.color = '#fff'; } };
        nsfwBtn.onmouseout = () => { if (!hideNSFWState) { nsfwBtn.style.background = '#313843'; nsfwBtn.style.color = '#aaa'; } };

        // View mode toggle button
        let currentViewMode = getViewMode();
        const viewModeBtn = document.createElement("button");
        const updateViewModeBtn = () => {
            viewModeBtn.textContent = currentViewMode === "thumbnails" ? "⊞ Grid" : "☰ List";
            viewModeBtn.title = currentViewMode === "thumbnails" ? "Switch to list view" : "Switch to grid view";
        };
        viewModeBtn.style.cssText = btnStyle;
        viewModeBtn.onmouseover = () => { viewModeBtn.style.background = '#38414c'; viewModeBtn.style.color = '#fff'; };
        viewModeBtn.onmouseout = () => { viewModeBtn.style.background = '#313843'; viewModeBtn.style.color = '#aaa'; };
        updateViewModeBtn();

        controlsBar.appendChild(searchWrapper);
        controlsBar.appendChild(nsfwBtn);
        controlsBar.appendChild(viewModeBtn);

        // Category selector
        const categoryContainer = document.createElement("div");
        categoryContainer.style.cssText = `
            display: flex;
            gap: 6px;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid ${UI.sectionBorder};
            flex-wrap: wrap;
        `;

        let categories = getVisibleCategories(node, {
            hideNSFW: hideNSFWState,
            workflowOnly,
            keepCategory: mode === "save" ? selectedCategory : "",
        });
        let selectedSaveName = initialSaveName;
        let categoryButtons = [];

        const isCategoryNSFW = (cat) => {
            return node.prompts?.[cat]?.["__meta__"]?.nsfw === true;
        };

        const ensureSelectedCategory = () => {
            categories = getVisibleCategories(node, {
                hideNSFW: hideNSFWState,
                workflowOnly,
                keepCategory: mode === "save" ? selectedCategory : "",
            });

            if (!Array.isArray(categories) || categories.length === 0) {
                selectedCategory = "";
                return "";
            }

            if (!selectedCategory || !categories.includes(selectedCategory)) {
                selectedCategory = categories[0];
            }

            if (hideNSFWState && isCategoryNSFW(selectedCategory)) {
                const firstVisible = categories.find(c => !isCategoryNSFW(c));
                if (firstVisible) {
                    selectedCategory = firstVisible;
                }
            }

            return selectedCategory;
        };

        ensureSelectedCategory();

        const updateCategoryButtons = () => {
            ensureSelectedCategory();

            categoryButtons.forEach(btn => {
                const cat = btn.dataset.category;
                const isSelected = cat === selectedCategory;
                const isNSFW = isCategoryNSFW(cat);

                // Hide NSFW categories when filter is active
                if (hideNSFWState && isNSFW) {
                    btn.style.display = "none";
                    return;
                }
                btn.style.display = "";

                btn.style.background = isSelected ? '#4a8ad4' : '#2a2a2a';
                btn.style.background = isSelected ? UI.accentSoft : UI.buttonBg;
                btn.style.color = isSelected ? '#fff' : '#aaa';

                // NSFW categories get a red border, otherwise normal
                if (isNSFW && !isSelected) {
                    btn.style.borderColor = '#944';
                } else {
                    btn.style.borderColor = isSelected ? '#5a9ae4' : '#444';
                }
            });

            // If selected category is now hidden/empty under current filters, switch.
            ensureSelectedCategory();
        };

        // Category context menu for NSFW toggle
        const showCategoryContextMenu = (event, cat) => {
            const existing = document.querySelector('.category-context-menu');
            if (existing) existing.remove();

            const menu = document.createElement("div");
            menu.className = "category-context-menu";
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

            const isNSFW = isCategoryNSFW(cat);
            const item = document.createElement("div");
            item.textContent = isNSFW ? "✓ NSFW" : "Mark as NSFW";
            item.style.cssText = `
                padding: 8px 16px;
                color: ${isNSFW ? '#f66' : '#ccc'};
                cursor: pointer;
                font-size: 13px;
            `;
            item.onmouseover = () => item.style.background = '#3a3a3a';
            item.onmouseout = () => item.style.background = 'transparent';
            item.onclick = async () => {
                menu.remove();
                try {
                    const resp = await fetch("/prompt-manager-advanced/toggle-nsfw", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ type: "category", category: cat })
                    });
                    const result = await resp.json();
                    if (result.success) {
                        node.prompts = result.prompts;
                        updateCategoryButtons();
                        renderContent(searchInput.value);
                    }
                } catch (err) {
                    console.error("[PromptManagerAdvanced] Error toggling category NSFW:", err);
                }
            };
            menu.appendChild(item);

            // Rename Category
            const renameDivider = document.createElement("div");
            renameDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
            menu.appendChild(renameDivider);

            const renameItem = document.createElement("div");
            renameItem.textContent = "✏️ Rename";
            renameItem.style.cssText = `
                padding: 8px 16px;
                color: #ccc;
                cursor: pointer;
                font-size: 13px;
            `;
            renameItem.onmouseover = () => renameItem.style.background = '#3a3a3a';
            renameItem.onmouseout = () => renameItem.style.background = 'transparent';
            renameItem.onclick = async () => {
                menu.remove();
                const result = await showRenameCategoryDialog(
                    "Rename Category",
                    "Enter new category name:",
                    [cat],
                    cat
                );
                if (result && result.newCategory && result.newCategory.trim()) {
                    const newCat = result.newCategory.trim();
                    if (newCat === cat) return;
                    try {
                        const resp = await fetch("/prompt-manager-advanced/rename-category", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ old_category: cat, new_category: newCat })
                        });
                        const data = await resp.json();
                        if (data.success) {
                            node.prompts = data.prompts;
                            if (selectedCategory === cat) {
                                selectedCategory = data.new_category;
                            }
                            rebuildCategoryList();
                            renderContent(searchInput.value);
                        } else {
                            await showInfo("Error", data.error);
                        }
                    } catch (err) {
                        console.error("[PromptManagerAdvanced] Error renaming category:", err);
                    }
                }
            };
            menu.appendChild(renameItem);

            // Delete Category
            const deleteDivider = document.createElement("div");
            deleteDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
            menu.appendChild(deleteDivider);

            const deleteItem = document.createElement("div");
            deleteItem.textContent = "🗑️ Delete Category";
            deleteItem.style.cssText = `
                padding: 8px 16px;
                color: #f66;
                cursor: pointer;
                font-size: 13px;
            `;
            deleteItem.onmouseover = () => deleteItem.style.background = '#3a3a3a';
            deleteItem.onmouseout = () => deleteItem.style.background = 'transparent';
            deleteItem.onclick = async () => {
                menu.remove();
                if (await showConfirm("Delete Category", `Are you sure you want to delete category "${cat}" and all its prompts?`)) {
                    try {
                        const resp = await fetch("/prompt-manager-advanced/delete-category", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ category: cat })
                        });
                        const data = await resp.json();
                        if (data.success) {
                            node.prompts = data.prompts;
                            if (selectedCategory === cat) {
                                const cats = Object.keys(node.prompts).filter(c => c !== "__meta__");
                                selectedCategory = cats[0] || "";
                            }
                            rebuildCategoryList();
                            renderContent(searchInput.value);
                        } else {
                            await showInfo("Error", data.error);
                        }
                    } catch (err) {
                        console.error("[PromptManagerAdvanced] Error deleting category:", err);
                    }
                }
            };
            menu.appendChild(deleteItem);

            // Generate Missing Thumbnails
            const thumbDivider = document.createElement("div");
            thumbDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
            menu.appendChild(thumbDivider);

            const thumbItem = document.createElement("div");
            thumbItem.textContent = "🎨 Generate Missing Thumbnails";
            thumbItem.style.cssText = `
                padding: 8px 16px;
                color: #ccc;
                cursor: pointer;
                font-size: 13px;
            `;
            thumbItem.onmouseover = () => thumbItem.style.background = '#3a3a3a';
            thumbItem.onmouseout = () => thumbItem.style.background = 'transparent';
            thumbItem.onclick = async () => {
                menu.remove();
                const catPrompts = node.prompts[cat];
                if (!catPrompts) return;

                // Collect prompts without thumbnails
                const missing = Object.keys(catPrompts).filter(name => {
                    const data = catPrompts[name];
                    return data && typeof data === "object" && !data.thumbnail;
                });

                if (missing.length === 0) {
                    await showInfo("No Missing Thumbnails", `All prompts in "${cat}" already have thumbnails.`);
                    return;
                }

                if (!await showConfirm("Generate Missing Thumbnails", `Generate thumbnails for ${missing.length} prompt(s) in "${cat}"?`, "Generate", "#4CAF50")) {
                    return;
                }

                // Ensure renderer selection is ready for prompts without saved workflow_data.
                const renderSelection = await ensureThumbnailRenderSelection();
                if (!renderSelection) return;

                // Resolve family-compatible defaults once for the batch.
                const fallbackBase = await resolveThumbnailFallbackBase(renderSelection);

                // Progress indicator
                const progress = document.createElement("div");
                progress.style.cssText = `
                    position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                    background: #222; border: 2px solid #4CAF50; border-radius: 8px;
                    padding: 20px 30px; z-index: 10000; color: #fff; font-size: 14px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.5); min-width: 280px;
                `;
                const progressText = document.createElement("div");
                progressText.style.cssText = `display: flex; align-items: center; gap: 12px;`;
                progressText.innerHTML = `
                    <div style="width: 20px; height: 20px; border: 3px solid #4CAF50; border-top-color: transparent; border-radius: 50%; animation: thumb-spin 1s linear infinite;"></div>
                    <span></span>
                `;
                progress.appendChild(progressText);
                const styleEl = document.createElement("style");
                styleEl.textContent = `@keyframes thumb-spin { to { transform: rotate(360deg); } }`;
                progress.appendChild(styleEl);
                document.body.appendChild(progress);

                let generated = 0;
                let failed = 0;
                for (let i = 0; i < missing.length; i++) {
                    const pName = missing[i];
                    progressText.querySelector("span").textContent = `Generating ${i + 1} / ${missing.length}: ${pName}`;
                    try {
                        await generateThumbnailForPrompt(node, cat, pName, () => {
                            renderContent(searchInput.value);
                        }, {
                            silent: true,
                            renderSelection,
                            fallbackBase,
                        });
                        generated++;
                    } catch (e) {
                        console.error(`[ThumbnailGen] Failed for "${pName}":`, e);
                        failed++;
                    }
                }

                if (progress.parentNode) progress.remove();
                await showInfo("Batch Complete", `Generated: ${generated}, Failed: ${failed}`);
            };
            menu.appendChild(thumbItem);

            document.body.appendChild(menu);
            const closeMenu = (e) => {
                if (!menu.contains(e.target)) {
                    menu.remove();
                    document.removeEventListener("mousedown", closeMenu, true);
                    document.removeEventListener("contextmenu", closeMenu, true);
                }
            };
            setTimeout(() => {
                document.addEventListener("mousedown", closeMenu, true);
                document.addEventListener("contextmenu", closeMenu, true);
            }, 0);
        };

        const rebuildCategoryList = () => {
            categories = getVisibleCategories(node, {
                hideNSFW: hideNSFWState,
                workflowOnly,
                keepCategory: mode === "save" ? selectedCategory : "",
            });
            ensureSelectedCategory();
            categoryButtons = [];
            categoryContainer.innerHTML = "";
            categories.forEach(cat => {
                const btn = document.createElement("button");
                btn.dataset.category = cat;
                btn.style.cssText = `
                    padding: 6px 14px;
                    border-radius: 6px;
                    border: 1px solid ${UI.inputBorder};
                    background: ${UI.buttonBg};
                    color: #aaa;
                    cursor: pointer;
                    font-size: 13px;
                    transition: all 0.15s ease;
                    position: relative;
                `;

                btn.textContent = cat;

                btn.onclick = () => {
                    selectedCategory = cat;
                    updateCategoryButtons();
                    renderContent(searchInput.value);
                };
                btn.oncontextmenu = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    showCategoryContextMenu(e, cat);
                };
                categoryButtons.push(btn);
                categoryContainer.appendChild(btn);
            });

            // Add "+" button to create a new category
            const addBtn = document.createElement("button");
            addBtn.textContent = "+";
            addBtn.title = "New Category";
            addBtn.style.cssText = `
                padding: 6px 14px;
                border-radius: 6px;
                border: 1px solid #5f6773;
                background: #313843;
                color: #aaa;
                cursor: pointer;
                font-size: 13px;
                transition: all 0.15s ease;
            `;
            addBtn.onmouseover = () => { addBtn.style.background = '#38414c'; addBtn.style.color = '#fff'; };
            addBtn.onmouseout = () => { addBtn.style.background = '#313843'; addBtn.style.color = '#aaa'; };
            addBtn.onclick = async () => {
                const result = await showNewCategoryDialog();
                if (result && result.name && result.name.trim()) {
                    const categoryName = result.name.trim();
                    const existingCategories = Object.keys(node.prompts || {});
                    const existingCategoryName = existingCategories.find(cat => cat.toLowerCase() === categoryName.toLowerCase());
                    if (existingCategoryName) {
                        await showInfo("Category Exists", `Category already exists as "${existingCategoryName}".`);
                        return;
                    }
                    try {
                        const resp = await fetch("/prompt-manager-advanced/save-category", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ category_name: categoryName, nsfw: result.nsfw })
                        });
                        const data = await resp.json();
                        if (data.success) {
                            node.prompts = data.prompts;
                            selectedCategory = categoryName;
                            rebuildCategoryList();
                            renderContent(searchInput.value);
                        } else {
                            await showInfo("Error", data.error);
                        }
                    } catch (err) {
                        console.error("[PromptManagerAdvanced] Error creating category:", err);
                    }
                }
            };
            categoryContainer.appendChild(addBtn);

            updateCategoryButtons();
        };
        rebuildCategoryList();

        // Content container - fixed size
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

        // Helper: get filtered prompt names for the selected category
        const getFilteredPrompts = (filter = "") => {
            let promptNames = getPromptNamesForCategory(node, selectedCategory, {
                hideNSFW: hideNSFWState,
                workflowOnly,
            });

            // Filter by search
            if (filter) {
                promptNames = promptNames.filter(name => name.toLowerCase().includes(filter.toLowerCase()));
            }
            return promptNames;
        };

        // Shared right-click handler for prompt items (works in both grid and list view)
        const promptContextMenu = (e, promptName) => {
            e.preventDefault();
            showThumbnailContextMenu(e, node, selectedCategory, promptName, () => {
                renderContent(searchInput.value);
            });
        };

        // ---- Grid (Thumbnail) View ----
        const renderGridView = (filter = "") => {
            grid.style.cssText = `
                display: grid;
                grid-template-columns: repeat(4, 140px);
                gap: 12px;
                padding: 4px 0;
            `;
            grid.innerHTML = "";

            const categoryPrompts = node.prompts[selectedCategory] || {};
            const filteredPrompts = getFilteredPrompts(filter);

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
                const isNSFW = promptData?.nsfw === true || isCategoryNSFW(selectedCategory);
                const rawWorkflowData = promptData?.workflow_data;
                const hasWorkflowData = (
                    (typeof rawWorkflowData === "string" && rawWorkflowData.trim().length > 0) ||
                    (rawWorkflowData && typeof rawWorkflowData === "object" && Object.keys(rawWorkflowData).length > 0)
                );

                const card = document.createElement("div");
                if (isSelected) {
                    card.dataset.selectedPrompt = "true";
                }
                card.style.cssText = `
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 8px;
                    background: ${isSelected ? UI.accentSoft : UI.cardBg};
                    border: 2px solid ${isSelected ? UI.accentBorder : UI.cardBorder};
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.15s ease;
                    position: relative;
                `;

                card.onmouseenter = () => {
                    if (!isSelected) {
                        card.style.background = UI.accentSoft;
                        card.style.borderColor = UI.accentBorder;
                    }
                };
                card.onmouseleave = () => {
                    if (!isSelected) {
                        card.style.background = UI.cardBg;
                        card.style.borderColor = UI.cardBorder;
                    }
                };

                // Top-right badge stack: NSFW first, workflow badge under it.
                const showWorkflowBadge = !workflowOnly && hasWorkflowData;
                if (isNSFW || showWorkflowBadge) {
                    const badgeStack = document.createElement("div");
                    badgeStack.style.cssText = `
                        position: absolute;
                        top: 4px;
                        right: 4px;
                        display: flex;
                        flex-direction: column;
                        align-items: flex-end;
                        gap: 2px;
                        z-index: 1;
                    `;

                    if (isNSFW) {
                        const badge = document.createElement("div");
                        badge.textContent = "NSFW";
                        badge.style.cssText = `
                            background: rgba(204, 0, 0, 0.85);
                            color: #fff;
                            font-size: 8px;
                            font-weight: bold;
                            padding: 1px 4px;
                            border-radius: 3px;
                            line-height: 1.2;
                        `;
                        badgeStack.appendChild(badge);
                    }

                    if (showWorkflowBadge) {
                        const workflowBadge = document.createElement("div");
                        workflowBadge.textContent = "R";
                        workflowBadge.title = "Has Recipe Data";
                        workflowBadge.style.cssText = `
                            width: 14px;
                            height: 14px;
                            border-radius: 50%;
                            background: rgba(235, 140, 35, 0.95);
                            color: #fff;
                            font-size: 9px;
                            font-weight: bold;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            line-height: 1;
                        `;
                        badgeStack.appendChild(workflowBadge);
                    }

                    card.appendChild(badgeStack);
                }

                // Thumbnail (using div with background-image to avoid browser extension interference)
                const thumbDiv = document.createElement("div");
                thumbDiv.style.cssText = `
                    width: 100px;
                    height: 100px;
                    background-image: url(${thumbnail});
                    background-size: cover;
                    background-position: center;
                    border-radius: 6px;
                    background-color: #1a1a1a;
                    flex-shrink: 0;
                    cursor: pointer;
                `;
                
                // Add hover preview with proper event handling (if enabled and not placeholder)
                if (previewEnabled && thumbnail !== DEFAULT_THUMBNAIL) {
                    thumbDiv.addEventListener("mouseenter", (e) => {
                        e.stopPropagation();
                        showPreviewWithDelay(thumbnail, thumbDiv);
                    });
                    thumbDiv.addEventListener("mouseleave", (e) => {
                        e.stopPropagation();
                        scheduleHidePreview();
                    });
                }

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

                card.appendChild(thumbDiv);
                card.appendChild(nameLabel);

                card.onclick = async () => {
                    if (mode === "save") {
                        selectedSaveName = promptName;
                        if (saveNameInput) {
                            saveNameInput.value = promptName;
                            saveNameInput.focus();
                            saveNameInput.select();
                        }
                        currentPrompt = promptName;
                        renderContent(searchInput.value);
                        return;
                    }

                    resolve({ category: selectedCategory, prompt: promptName });
                    cleanup();
                };

                if (mode === "save") {
                    card.ondblclick = async () => {
                        if (!onSave) return;
                        const overwriteOk = await showConfirm(
                            "Overwrite Prompt",
                            `Prompt "${promptName}" already exists in category "${selectedCategory}". Do you want to replace it?`,
                            "Replace",
                            "#c44"
                        );
                        if (!overwriteOk) return;

                        const saveResult = await onSave({
                            category: selectedCategory,
                            name: promptName,
                            overwrite: true,
                        });

                        if (saveResult?.success) {
                            resolve(saveResult);
                            cleanup();
                        } else {
                            await showInfo("Save Failed", saveResult?.error || "Failed to save workflow.");
                        }
                    };
                }

                card.oncontextmenu = (e) => promptContextMenu(e, promptName);

                grid.appendChild(card);
            });
        };

        // ---- List View ----
        const renderListView = (filter = "") => {
            grid.style.cssText = `
                display: flex;
                flex-direction: column;
                gap: 0;
                padding: 4px 0;
            `;
            grid.innerHTML = "";

            const categoryPrompts = node.prompts[selectedCategory] || {};
            const filteredPrompts = getFilteredPrompts(filter);

            if (filteredPrompts.length === 0) {
                const emptyMsg = document.createElement("div");
                emptyMsg.textContent = filter ? "No matching prompts found" : "No prompts in this category";
                emptyMsg.style.cssText = `
                    text-align: center;
                    color: #666;
                    padding: 40px;
                    font-style: italic;
                `;
                grid.appendChild(emptyMsg);
                return;
            }

            // Column headers
            const headerRow = document.createElement("div");
            headerRow.style.cssText = `
                display: grid;
                grid-template-columns: 44px 1fr 70px 70px 70px;
                gap: 8px;
                padding: 4px 8px 6px 8px;
                    border-bottom: 1px solid rgba(148,163,184,0.2);
                margin-bottom: 4px;
                align-items: center;
            `;
            const headers = ["", "Name", "LoRAs A", "LoRAs B", "Triggers"];
            headers.forEach((h, i) => {
                const hDiv = document.createElement("div");
                hDiv.textContent = h;
                hDiv.style.cssText = `
                    font-size: 10px;
                    color: #888;
                    font-weight: bold;
                    text-transform: uppercase;
                    text-align: ${i === 0 ? 'center' : i >= 2 ? 'center' : 'left'};
                `;
                headerRow.appendChild(hDiv);
            });
            grid.appendChild(headerRow);

            filteredPrompts.forEach(promptName => {
                const promptData = categoryPrompts[promptName];
                const thumbnail = promptData?.thumbnail || DEFAULT_THUMBNAIL;
                const isSelected = promptName === currentPrompt;
                const isNSFW = promptData?.nsfw === true || isCategoryNSFW(selectedCategory);
                const rawWorkflowData = promptData?.workflow_data;
                const hasWorkflowData = (
                    (typeof rawWorkflowData === "string" && rawWorkflowData.trim().length > 0) ||
                    (rawWorkflowData && typeof rawWorkflowData === "object" && Object.keys(rawWorkflowData).length > 0)
                );
                const lorasACount = (promptData?.loras_a || []).length;
                const lorasBCount = (promptData?.loras_b || []).length;
                const triggerCount = (promptData?.trigger_words || []).length;

                const row = document.createElement("div");
                if (isSelected) {
                    row.dataset.selectedPrompt = "true";
                }
                row.style.cssText = `
                    display: grid;
                    grid-template-columns: 44px 1fr 70px 70px 70px;
                    gap: 8px;
                    padding: 4px 8px;
                    background: ${isSelected ? UI.accentSoft : 'transparent'};
                    border-radius: 4px;
                    cursor: pointer;
                    align-items: center;
                    transition: background 0.1s ease;
                `;
                row.onmouseenter = () => { if (!isSelected) row.style.background = '#2a2a2a'; };
                row.onmouseleave = () => { if (!isSelected) row.style.background = 'transparent'; };

                // Thumbnail icon (using div with background-image to avoid browser extension interference)
                const thumbWrap = document.createElement("div");
                thumbWrap.style.cssText = `
                    width: 36px;
                    height: 36px;
                    position: relative;
                    flex-shrink: 0;
                `;

                const thumbDiv = document.createElement("div");
                thumbDiv.style.cssText = `
                    width: 36px;
                    height: 36px;
                    background-image: url(${thumbnail});
                    background-size: cover;
                    background-position: center;
                    border-radius: 4px;
                    background-color: #1a1a1a;
                    cursor: pointer;
                `;
                
                // Add hover preview with proper event handling (if enabled and not placeholder)
                if (previewEnabled && thumbnail !== DEFAULT_THUMBNAIL) {
                    thumbDiv.addEventListener("mouseenter", (e) => {
                        e.stopPropagation();
                        showPreviewWithDelay(thumbnail, thumbDiv);
                    });
                    thumbDiv.addEventListener("mouseleave", (e) => {
                        e.stopPropagation();
                        scheduleHidePreview();
                    });
                }

                thumbWrap.appendChild(thumbDiv);

                const showWorkflowBadge = !workflowOnly && hasWorkflowData;
                if (showWorkflowBadge) {
                    const workflowBadge = document.createElement("div");
                    workflowBadge.textContent = "R";
                    workflowBadge.title = "Has Recipe Data";
                    workflowBadge.style.cssText = `
                        position: absolute;
                        right: -2px;
                        bottom: -2px;
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        background: rgba(235, 140, 35, 0.95);
                        color: #fff;
                        font-size: 8px;
                        font-weight: bold;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        line-height: 1;
                        border: 1px solid rgba(0, 0, 0, 0.4);
                        z-index: 1;
                    `;
                    thumbWrap.appendChild(workflowBadge);
                }

                // Name + optional NSFW badge
                const nameDiv = document.createElement("div");
                nameDiv.style.cssText = `
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    overflow: hidden;
                `;
                const nameSpan = document.createElement("span");
                nameSpan.textContent = promptName;
                nameSpan.title = promptName;
                nameSpan.style.cssText = `
                    font-size: 13px;
                    color: #ddd;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                `;
                nameDiv.appendChild(nameSpan);
                if (isNSFW) {
                    const badge = document.createElement("span");
                    badge.textContent = "NSFW";
                    badge.style.cssText = `
                        background: rgba(204, 0, 0, 0.85);
                        color: #fff;
                        font-size: 8px;
                        font-weight: bold;
                        padding: 1px 4px;
                        border-radius: 3px;
                        flex-shrink: 0;
                    `;
                    nameDiv.appendChild(badge);
                }

                // Counts
                const makeCount = (n) => {
                    const d = document.createElement("div");
                    d.textContent = n > 0 ? n : "—";
                    d.style.cssText = `
                        font-size: 12px;
                        color: ${n > 0 ? '#aaa' : '#555'};
                        text-align: center;
                    `;
                    return d;
                };

                row.appendChild(thumbWrap);
                row.appendChild(nameDiv);
                row.appendChild(makeCount(lorasACount));
                row.appendChild(makeCount(lorasBCount));
                row.appendChild(makeCount(triggerCount));

                row.onclick = async () => {
                    if (mode === "save") {
                        selectedSaveName = promptName;
                        if (saveNameInput) {
                            saveNameInput.value = promptName;
                            saveNameInput.focus();
                            saveNameInput.select();
                        }
                        currentPrompt = promptName;
                        renderContent(searchInput.value);
                        return;
                    }

                    resolve({ category: selectedCategory, prompt: promptName });
                    cleanup();
                };

                if (mode === "save") {
                    row.ondblclick = async () => {
                        if (!onSave) return;
                        const overwriteOk = await showConfirm(
                            "Overwrite Prompt",
                            `Prompt "${promptName}" already exists in category "${selectedCategory}". Do you want to replace it?`,
                            "Replace",
                            "#c44"
                        );
                        if (!overwriteOk) return;

                        const saveResult = await onSave({
                            category: selectedCategory,
                            name: promptName,
                            overwrite: true,
                        });

                        if (saveResult?.success) {
                            resolve(saveResult);
                            cleanup();
                        } else {
                            await showInfo("Save Failed", saveResult?.error || "Failed to save workflow.");
                        }
                    };
                }

                row.oncontextmenu = (e) => promptContextMenu(e, promptName);

                grid.appendChild(row);
            });
        };

        // Thumbnail hover preview system
        let hoverPreview = null;
        let hoverTimer = null;
        let hideTimer = null;
        let resetTimer = null;
        let previewActivated = false;  // Tracks if preview system is "warmed up"
        let currentMouseX = 0;
        let currentMouseY = 0;
        let currentThumbnail = "";
        let previewWidth = 0;
        let previewHeight = 0;

        const createHoverPreview = () => {
            if (!hoverPreview) {
                hoverPreview = document.createElement("div");
                hoverPreview.setAttribute('data-pm-thumbnail-preview', 'true');
                hoverPreview.style.cssText = `
                    position: fixed;
                    pointer-events: none;
                    z-index: 10001;
                    display: none;
                    border: 2px solid ${UI.inputBorder};
                    border-radius: 8px;
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.8);
                    background-color: #1a1a1a;
                    background-size: contain;
                    background-position: center;
                    background-repeat: no-repeat;
                `;
                
                document.body.appendChild(hoverPreview);
            }
            return hoverPreview;
        };

        const updatePreviewPosition = () => {
            if (!hoverPreview || !previewWidth || !previewHeight) return;
            
            // Center on thumbnail position, keeping on screen
            let left = currentMouseX - previewWidth / 2;
            let top = currentMouseY - previewHeight / 2;
            
            // Keep on screen with padding
            left = Math.max(5, Math.min(left, window.innerWidth - previewWidth - 9));
            top = Math.max(5, Math.min(top, window.innerHeight - previewHeight - 9));
            
            hoverPreview.style.left = left + "px";
            hoverPreview.style.top = top + "px";
        };

        const showPreviewWithDelay = (thumbnailSrc, thumbnailElement) => {
            // Cancel any pending hide operation and reset timer
            clearTimeout(hideTimer);
            clearTimeout(resetTimer);
            hideTimer = null;
            resetTimer = null;
            
            const rect = thumbnailElement.getBoundingClientRect();
            currentMouseX = rect.left + rect.width / 2;
            currentMouseY = rect.top + rect.height / 2;
            currentThumbnail = thumbnailSrc;
            
            // Intelligent delay: 1000ms initial, 10ms once activated
            const delay = previewActivated ? 10 : 1000;
            
            clearTimeout(hoverTimer);
            hoverTimer = setTimeout(() => {
                previewActivated = true;  // Activate fast mode after first preview
                const preview = createHoverPreview();
                
                // Load image in memory (not in DOM) to get dimensions
                const tempImg = new Image();
                tempImg.onload = function() {
                    const naturalWidth = this.naturalWidth;
                    const naturalHeight = this.naturalHeight;
                    
                    // Scale logic:
                    // - Thumbnails <= 200px in both dimensions: 2x scale (200px -> 400px)
                    // - Thumbnails > 200px in either dimension: 1x scale (keep original size)
                    const scale = (naturalWidth > 200 || naturalHeight > 200) ? 1 : 2;
                    
                    previewWidth = naturalWidth * scale;
                    previewHeight = naturalHeight * scale;
                    
                    // Update preview div with background image and dimensions
                    preview.style.width = previewWidth + 'px';
                    preview.style.height = previewHeight + 'px';
                    preview.style.backgroundImage = `url(${thumbnailSrc})`;
                    
                    updatePreviewPosition();
                    preview.style.display = "block";
                };
                
                // Load the image
                tempImg.src = thumbnailSrc;
            }, delay);
        };

        const hidePreview = () => {
            clearTimeout(hoverTimer);
            clearTimeout(hideTimer);
            hideTimer = null;
            
            if (hoverPreview) {
                hoverPreview.style.display = "none";
                previewWidth = 0;
                previewHeight = 0;
            }
        };

        const scheduleHidePreview = () => {
            clearTimeout(hideTimer);
            hideTimer = setTimeout(() => {
                hidePreview();
                
                // Start reset timer: after 2 seconds of no hovering, reset to slow mode
                clearTimeout(resetTimer);
                resetTimer = setTimeout(() => {
                    previewActivated = false;
                }, 2000);
            }, 100);
        };

        // Unified render function: picks grid vs list based on currentViewMode
        const renderContent = (filter = "") => {
            if (currentViewMode === "list") {
                renderListView(filter);
            } else {
                renderGridView(filter);
            }
        };

        // Event handlers for controls
        nsfwBtn.onclick = () => {
            hideNSFWState = !hideNSFWState;
            sessionHideNSFW = hideNSFWState;
            updateNsfwBtn();
            rebuildCategoryList();
            renderContent(searchInput.value);
        };

        viewModeBtn.onclick = () => {
            currentViewMode = currentViewMode === "thumbnails" ? "list" : "thumbnails";
            sessionViewMode = currentViewMode;
            updateViewModeBtn();
            renderContent(searchInput.value);
        };

        let saveNameInput = null;
        let saveActionButton = null;
        let cancelSaveButton = null;

        const handleSaveAction = async () => {
            if (mode !== "save" || !onSave || !saveNameInput) {
                return;
            }

            const category = ensureSelectedCategory();
            if (!category) {
                await showInfo("Missing Category", "Please create or select a category before saving.");
                return;
            }

            const name = (saveNameInput.value || "").trim();
            if (!name) {
                await showInfo("Missing Name", "Please enter a name before saving.");
                saveNameInput.focus();
                return;
            }

            const existing = node.prompts?.[category]?.[name];
            let overwrite = false;
            if (existing) {
                overwrite = await showConfirm(
                    "Overwrite Prompt",
                    `Prompt "${name}" already exists in category "${category}". Do you want to replace it?`,
                    "Replace",
                    "#c44",
                    false
                );
                if (!overwrite) {
                    return;
                }
            }

            const saveResult = await onSave({ category, name, overwrite });
            if (saveResult?.success) {
                resolve(saveResult);
                cleanup();
            } else {
                await showInfo("Save Failed", saveResult?.error || "Failed to save workflow.");
            }
        };

        // Initial render
        renderContent();

        // Scroll to selected prompt after rendering
        setTimeout(() => {
            const selectedElement = gridContainer.querySelector('[data-selected-prompt="true"]');
            if (selectedElement) {
                selectedElement.scrollIntoView({ behavior: 'instant', block: 'center' });
            }
        }, 50);

        // Search filtering
        searchInput.oninput = () => {
            renderContent(searchInput.value);
        };

        gridContainer.appendChild(grid);

        // Footer with hint
        const footer = document.createElement("div");
        footer.style.cssText = `
            margin-top: 4px;
            margin-bottom: -8px;
            padding-top: 6px;
            border-top: 1px solid ${UI.sectionBorder};
            font-size: 11px;
            color: #666;
            text-align: center;
        `;
        footer.textContent = mode === "save"
            ? "Right-click a prompt or category for more options (thumbnails, NSFW, delete). Single-click fills name; double-click replaces."
            : "Right-click a prompt or category for more options (thumbnails, NSFW, delete)";

        const saveBar = document.createElement("div");
        if (mode === "save") {
            saveBar.style.cssText = `
                display: flex;
                gap: 8px;
                align-items: center;
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid ${UI.sectionBorder};
            `;

            saveNameInput = document.createElement("input");
            saveNameInput.type = "text";
            saveNameInput.value = selectedSaveName;
            saveNameInput.placeholder = saveNamePlaceholder;
            saveNameInput.style.cssText = `
                flex: 1;
                min-width: 0;
                padding: 7px 10px;
                background: ${UI.inputBg};
                border: 1px solid ${UI.inputBorder};
                border-radius: 4px;
                color: #fff;
                font-size: 13px;
                box-sizing: border-box;
                outline: none;
            `;
            saveNameInput.onfocus = () => saveNameInput.style.borderColor = UI.accent;
            saveNameInput.onblur = () => saveNameInput.style.borderColor = UI.inputBorder;
            saveNameInput.onkeydown = async (e) => {
                if (e.key === "Enter") {
                    e.preventDefault();
                    e.stopPropagation();
                    await handleSaveAction();
                }
            };

            saveActionButton = document.createElement("button");
            saveActionButton.textContent = saveButtonText;
            saveActionButton.style.cssText = `
                background: #2b6d3a;
                border: 1px solid #4a9158;
                color: #fff;
                padding: 7px 14px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                white-space: nowrap;
            `;
            saveActionButton.onclick = async () => {
                await handleSaveAction();
            };

            cancelSaveButton = document.createElement("button");
            cancelSaveButton.textContent = "Cancel";
            cancelSaveButton.style.cssText = `
                background: #313843;
                border: 1px solid #5f6773;
                color: #ccc;
                padding: 7px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                white-space: nowrap;
            `;
            cancelSaveButton.onclick = () => {
                resolve(null);
                cleanup();
            };

            saveBar.appendChild(saveNameInput);
            saveBar.appendChild(cancelSaveButton);
            saveBar.appendChild(saveActionButton);
        }

        dialog.appendChild(header);
        dialog.appendChild(controlsBar);
        dialog.appendChild(categoryContainer);
        dialog.appendChild(gridContainer);
        dialog.appendChild(footer);
        if (mode === "save") {
            dialog.appendChild(saveBar);
        }

        const cleanup = () => {
            clearTimeout(hoverTimer);
            clearTimeout(hideTimer);
            clearTimeout(resetTimer);
            hidePreview();
            if (hoverPreview && hoverPreview.parentNode) {
                document.body.removeChild(hoverPreview);
            }
            hoverPreview = null;
            // Clean up any stale preview elements
            const allPreviews = document.querySelectorAll('[data-pm-thumbnail-preview]');
            allPreviews.forEach(p => {
                if (p.parentNode) p.parentNode.removeChild(p);
            });
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
        dialog.onkeydown = async (e) => {
            if (e.key === "Escape") {
                resolve(null);
                cleanup();
                return;
            }
            if (mode === "save" && e.key === "Enter") {
                const target = e.target;
                if (target !== searchInput) {
                    e.preventDefault();
                    e.stopPropagation();
                    await handleSaveAction();
                }
            }
        };

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        searchInput.focus();
    });
}

export async function openPromptBrowserForSave(options = {}) {
    const browserNode = options.node || {};
    const currentCategory = options.currentCategory || "Default";
    const currentPrompt = options.currentPrompt || "";
    return showThumbnailBrowser(browserNode, currentCategory, currentPrompt, {
        mode: "save",
        onSave: options.onSave,
        title: options.title || "Save Workflow",
        saveButtonText: options.saveButtonText || "Save",
        namePlaceholder: options.namePlaceholder || "Prompt name",
        initialName: options.initialName || "",
        workflowOnly: options.workflowOnly === true || browserNode?._isWorkflowManager === true,
    });
}

// ========================
// Thumbnail Generation System
// ========================

const THUMB_RENDER_WIDTH = 768;
const THUMB_RENDER_HEIGHT = 1024;
const THUMB_RENDER_BATCH = 1;
const THUMB_RENDER_LENGTH = 1;
const THUMB_DEFAULT_NEGATIVE = "blurry, bad quality, worst quality, low resolution, watermark, text, logo, deformed, ugly, disfigured";

let _thumbnailRenderFamily = null;
let _thumbnailRenderModel = null;
let _thumbnailLegacyModel = null;
let _thumbnailShowAllModels = false;

try {
    const savedFamily = app.ui.settings.getSettingValue("PromptManager.ThumbnailRenderFamily");
    const savedModel = app.ui.settings.getSettingValue("PromptManager.ThumbnailRenderModel");
    if (typeof savedFamily === "string" && savedFamily.trim()) {
        _thumbnailRenderFamily = savedFamily;
    }
    if (typeof savedModel === "string" && savedModel.trim()) {
        _thumbnailRenderModel = savedModel;
    }
    _thumbnailShowAllModels = app.ui.settings.getSettingValue("PromptManager.ThumbnailShowAllModels") === true;

    // Backward compatibility with old checkpoint-only preference:
    // keep it only as an initial picker prefill, do not auto-select family/model.
    if (!_thumbnailRenderModel) {
        const legacyCheckpoint = app.ui.settings.getSettingValue("PromptManager.ThumbnailCheckpoint");
        if (typeof legacyCheckpoint === "string" && legacyCheckpoint.trim()) {
            _thumbnailLegacyModel = legacyCheckpoint;
        }
    }
} catch (e) { /* settings not ready yet, will be loaded on first use */ }

function withThumbnailModelFilteringPreference(url, showAllModels) {
    if (!showAllModels) return url;
    return `${url}${url.includes("?") ? "&" : "?"}show_all=1`;
}

function _thumbnailLeafName(modelPath) {
    const p = String(modelPath || "");
    if (!p) return "";
    const idx = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
    const leaf = idx >= 0 ? p.substring(idx + 1) : p;
    return leaf.replace(/\.safetensors$/i, "");
}

function _thumbnailDisplayName(modelPath, maxLength = 68) {
    const leaf = _thumbnailLeafName(modelPath);
    if (leaf.length <= maxLength) return leaf;
    return `${leaf.slice(0, Math.max(0, maxLength - 1))}…`;
}

function applyThumbnailResolution(workflowData) {
    const wfForThumb = JSON.parse(JSON.stringify(workflowData || {}));
    const baseResolution = (wfForThumb.resolution && typeof wfForThumb.resolution === "object")
        ? wfForThumb.resolution
        : {};
    wfForThumb.resolution = {
        ...baseResolution,
        width: THUMB_RENDER_WIDTH,
        height: THUMB_RENDER_HEIGHT,
        batch_size: THUMB_RENDER_BATCH,
        length: THUMB_RENDER_LENGTH,
    };
    return wfForThumb;
}

function normalizeLorasForRenderer(loraList) {
    if (!Array.isArray(loraList)) return [];
    return loraList
        .filter((lora) => lora && typeof lora === "object" && lora.active !== false)
        .map((lora) => {
            const modelStrength = Number(lora.model_strength ?? lora.strength ?? 1.0);
            const clipStrength = Number(lora.clip_strength ?? lora.strength ?? modelStrength);
            return {
                name: String(lora.name || lora.path || "").trim(),
                model_strength: Number.isFinite(modelStrength) ? modelStrength : 1.0,
                clip_strength: Number.isFinite(clipStrength) ? clipStrength : (Number.isFinite(modelStrength) ? modelStrength : 1.0),
                active: true,
            };
        })
        .filter((lora) => lora.name.length > 0);
}

function getThumbnailFamilySamplerDefaults(familyKey) {
    const key = String(familyKey || "").trim().toLowerCase();
    const byFamily = {
        sdxl: { steps_a: 20, steps_b: null, cfg: 5, sampler_name: "dpmpp_2m_sde", scheduler: "karras" },
        flux1: { steps_a: 20, steps_b: null, cfg: 1, sampler_name: "euler", scheduler: "simple" },
        flux2: { steps_a: 4, steps_b: null, cfg: 1, sampler_name: "euler", scheduler: "simple" },
        zimage: { steps_a: 9, steps_b: null, cfg: 1, sampler_name: "euler", scheduler: "simple" },
        ltxv: { steps_a: 8, steps_b: null, cfg: 1, sampler_name: "euler", scheduler: "simple" },
        wan_image: { steps_a: 8, steps_b: null, cfg: 1, sampler_name: "lcm", scheduler: "simple" },
        wan_video_t2v: { steps_a: 2, steps_b: 2, cfg: 1, sampler_name: "lcm", scheduler: "simple" },
        wan_video_i2v: { steps_a: 2, steps_b: 2, cfg: 1, sampler_name: "lcm", scheduler: "simple" },
        qwen_image: { steps_a: 10, steps_b: null, cfg: 1, sampler_name: "euler", scheduler: "simple" },
    };
    const d = byFamily[key] || byFamily.sdxl;
    return {
        steps_a: d.steps_a,
        steps_b: d.steps_b,
        cfg: d.cfg,
        seed_a: 0,
        sampler_name: d.sampler_name,
        scheduler: d.scheduler,
        seed_b: null,
    };
}

async function buildRendererFallbackWorkflowData(promptText, promptData, renderSelection) {
    const familySamplerDefaults = getThumbnailFamilySamplerDefaults(renderSelection.family);
    const baseRaw = {
        model_family: renderSelection.family,
        model_a: renderSelection.model,
        model_b: "",
        positive_prompt: String(promptText || ""),
        negative_prompt: String(promptData?.negative_prompt || THUMB_DEFAULT_NEGATIVE),
        loras_a: normalizeLorasForRenderer(promptData?.loras_a),
        loras_b: normalizeLorasForRenderer(promptData?.loras_b),
        vae: { name: "", source: "unknown" },
        clip: { names: [], type: "", source: "unknown" },
        sampler: familySamplerDefaults,
        resolution: {
            width: THUMB_RENDER_WIDTH,
            height: THUMB_RENDER_HEIGHT,
            batch_size: THUMB_RENDER_BATCH,
            length: THUMB_RENDER_LENGTH,
        },
        is_video: false,
    };

    let normalized = null;
    try {
        const processResp = await fetch("/workflow-builder/process-extracted", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                extracted: baseRaw,
                family_override: renderSelection.family,
            }),
        });
        const processData = await processResp.json();
        if (processResp.ok && processData?.extracted && typeof processData.extracted === "object") {
            normalized = processData.extracted;
        }
    } catch (e) {
        console.warn("[ThumbnailGen] Failed to normalize fallback workflow_data via RecipeBuilder:", e);
    }

    const clipNames = Array.isArray(normalized?.clip?.names)
        ? normalized.clip.names.filter((x) => !!x)
        : [];

    const base = {
        family: normalized?.model_family || renderSelection.family,
        model_a: normalized?.model_a || renderSelection.model,
        model_b: normalized?.model_b || "",
        positive_prompt: String(normalized?.positive_prompt ?? baseRaw.positive_prompt),
        negative_prompt: String(normalized?.negative_prompt ?? baseRaw.negative_prompt),
        loras_a: Array.isArray(normalized?.loras_a) ? normalized.loras_a : baseRaw.loras_a,
        loras_b: Array.isArray(normalized?.loras_b) ? normalized.loras_b : baseRaw.loras_b,
        vae: String(normalized?.vae?.name || ""),
        clip: clipNames,
        clip_type: String(normalized?.clip?.type || ""),
        loader_type: String(normalized?.loader_type || ""),
        sampler: normalized?.sampler && typeof normalized.sampler === "object"
            ? normalized.sampler
            : baseRaw.sampler,
        _source: "PromptManagerAdvancedThumbnail",
    };
    return applyThumbnailResolution(base);
}

async function resolveThumbnailFallbackBase(renderSelection) {
    const familySamplerDefaults = getThumbnailFamilySamplerDefaults(renderSelection.family);
    const baseRaw = {
        model_family: renderSelection.family,
        model_a: renderSelection.model,
        model_b: "",
        positive_prompt: "",
        negative_prompt: THUMB_DEFAULT_NEGATIVE,
        loras_a: [],
        loras_b: [],
        vae: { name: "", source: "unknown" },
        clip: { names: [], type: "", source: "unknown" },
        sampler: familySamplerDefaults,
        resolution: {
            width: THUMB_RENDER_WIDTH,
            height: THUMB_RENDER_HEIGHT,
            batch_size: THUMB_RENDER_BATCH,
            length: THUMB_RENDER_LENGTH,
        },
        is_video: false,
    };

    let normalized = null;
    try {
        const processResp = await fetch("/workflow-builder/process-extracted", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                extracted: baseRaw,
                family_override: renderSelection.family,
            }),
        });
        const processData = await processResp.json();
        if (processResp.ok && processData?.extracted && typeof processData.extracted === "object") {
            normalized = processData.extracted;
        }
    } catch (e) {
        console.warn("[ThumbnailGen] Failed to pre-resolve batch fallback base:", e);
    }

    return { normalized, renderSelection };
}

async function buildRendererFallbackWorkflowDataWithBase(promptText, promptData, renderSelection, fallbackBase = null) {
    if (!fallbackBase?.normalized) {
        return buildRendererFallbackWorkflowData(promptText, promptData, renderSelection);
    }

    const normalized = fallbackBase.normalized;
    const clipNames = Array.isArray(normalized?.clip?.names)
        ? normalized.clip.names.filter((x) => !!x)
        : [];

    const wf = {
        family: normalized?.model_family || renderSelection.family,
        model_a: normalized?.model_a || renderSelection.model,
        model_b: normalized?.model_b || "",
        positive_prompt: String(promptText || ""),
        negative_prompt: String(promptData?.negative_prompt || THUMB_DEFAULT_NEGATIVE),
        loras_a: normalizeLorasForRenderer(promptData?.loras_a),
        loras_b: normalizeLorasForRenderer(promptData?.loras_b),
        vae: String(normalized?.vae?.name || ""),
        clip: clipNames,
        clip_type: String(normalized?.clip?.type || ""),
        loader_type: String(normalized?.loader_type || ""),
        sampler: normalized?.sampler && typeof normalized.sampler === "object"
            ? normalized.sampler
            : getThumbnailFamilySamplerDefaults(renderSelection.family),
        _source: "PromptManagerAdvancedThumbnail",
    };
    return applyThumbnailResolution(wf);
}

async function fetchRendererFamilies() {
    const resp = await fetch("/workflow-extractor/list-families");
    const data = await resp.json();
    const familiesObj = data?.families && typeof data.families === "object" ? data.families : {};
    return Object.entries(familiesObj)
        .map(([key, label]) => ({ key, label: String(label || key) }))
        .sort((a, b) => {
            if (a.key === "sdxl") return -1;
            if (b.key === "sdxl") return 1;
            return a.label.localeCompare(b.label);
        });
}

async function fetchRendererModelsForFamily(familyKey, showAllModels = false) {
    const baseUrl = familyKey
        ? `/workflow-extractor/list-models?family=${encodeURIComponent(familyKey)}`
        : `/workflow-extractor/list-models`;
    const url = withThumbnailModelFilteringPreference(baseUrl, showAllModels);
    const resp = await fetch(url);
    const data = await resp.json();
    if (!Array.isArray(data?.models)) return [];
    return [...data.models].sort((a, b) => a.localeCompare(b));
}

function saveThumbnailRenderSelection(selection) {
    if (!selection?.family || !selection?.model) return;
    _thumbnailRenderFamily = selection.family;
    _thumbnailRenderModel = selection.model;
    _thumbnailLegacyModel = null;
    try {
        app.ui.settings.setSettingValue("PromptManager.ThumbnailRenderFamily", selection.family);
        app.ui.settings.setSettingValue("PromptManager.ThumbnailRenderModel", selection.model);
    } catch (e) {
        console.warn("[ThumbnailGen] Could not persist thumbnail render selection:", e);
    }
}

async function showThumbnailRenderPicker(preselectedFamily = null, preselectedModel = null) {
    let families = [];
    try {
        families = await fetchRendererFamilies();
    } catch (e) {
        console.error("[ThumbnailGen] Failed loading model families:", e);
        await showInfo("Model Picker Error", "Could not load renderer families.");
        return null;
    }

    if (!families.length) {
        await showInfo("No Families", "No model families were found for Recipe Renderer.");
        return null;
    }

    const validFamily = families.some((f) => f.key === preselectedFamily)
        ? preselectedFamily
        : families[0].key;

    return new Promise((resolve) => {
        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
            background: ${UI.panel}; border: 1px solid ${UI.panelBorder}; border-radius: 10px;
            padding: 18px; z-index: 10000; width: 634px; min-width: 634px; max-width: 634px;
            font-family: inherit; font-size: 13px;
            box-sizing: border-box; overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        `;

        dialog.innerHTML = `
            <div style="margin-bottom: 12px; font-size: 18px; font-weight: 600; color: #ddd;">Select Family + Model</div>
            <div style="display: grid; grid-template-columns: 64px 1fr 34px; gap: 8px; align-items: center; margin-bottom: 8px;">
                <label style="color: #c4ccd6; font-size: 13px; font-weight: 600;">Type</label>
                <select class="family-select" style="padding: 7px; min-width: 0; background: ${UI.inputBg}; color: #e5e7eb; border: 1px solid ${UI.inputBorder}; border-radius: 8px; font-family: inherit; font-size: 13px;"></select>
                <button class="filter-toggle" title="Model filtering ON (by family). Click to show all models." style="height: 32px; background: ${UI.buttonBg}; color: #ddd; border: 1px solid ${UI.inputBorder}; border-radius: 8px; cursor: pointer; font-size: 14px;">👁</button>
            </div>
            <div style="display: grid; grid-template-columns: 64px 1fr; gap: 8px; align-items: center; margin-bottom: 8px;">
                <label style="color: #c4ccd6; font-size: 13px;">Model</label>
                <select class="model-select" size="13" style="width: 100%; min-width: 0; padding: 4px; background: ${UI.inputBg}; color: #e5e7eb; border: 1px solid ${UI.inputBorder}; border-radius: 8px; box-sizing: border-box; font-family: inherit; font-size: 13px;"></select>
            </div>
            <div class="model-hint" style="min-height: 16px; margin-bottom: 12px; color: #9aa5b4; font-size: 12px;"></div>
            <div style="display: flex; gap: 10px; justify-content: flex-end;">
                <button class="cancel-btn" style="padding: 8px 16px; background: ${UI.buttonBg}; color: #e5e7eb; border: 1px solid ${UI.inputBorder}; border-radius: 8px; cursor: pointer; font-family: inherit; font-size: 13px;">Cancel</button>
                <button class="ok-btn" style="padding: 8px 16px; background: ${UI.accentSoft}; color: #fff; border: 1px solid ${UI.accentBorder}; border-radius: 8px; cursor: pointer; font-family: inherit; font-size: 13px;">Select</button>
            </div>
        `;

        const familySel = dialog.querySelector(".family-select");
        const filterToggle = dialog.querySelector(".filter-toggle");
        const modelSel = dialog.querySelector(".model-select");
        const modelHint = dialog.querySelector(".model-hint");
        const okBtn = dialog.querySelector('.ok-btn');
        const cancelBtn = dialog.querySelector('.cancel-btn');

        let familyModels = [];
        let showAllModels = _thumbnailShowAllModels === true;

        const updateFilterToggleUi = () => {
            filterToggle.style.color = showAllModels ? "#ffb0b0" : "#e6e6e6";
            filterToggle.style.background = showAllModels ? "#453339" : UI.buttonBg;
            filterToggle.style.borderColor = showAllModels ? "#9b727b" : UI.inputBorder;
            filterToggle.title = showAllModels
                ? "Model filtering OFF (showing all models). Click to filter by family."
                : "Model filtering ON (by family). Click to show all models.";
        };
        updateFilterToggleUi();

        const setHint = (text, color = "#888") => {
            modelHint.textContent = text;
            modelHint.style.color = color;
        };

        const renderModelOptions = (preferredValue = "") => {
            const filtered = familyModels;

            modelSel.innerHTML = "";
            filtered.forEach((modelPath) => {
                const opt = document.createElement("option");
                opt.value = modelPath;
                opt.textContent = _thumbnailDisplayName(modelPath);
                opt.title = modelPath;
                modelSel.appendChild(opt);
            });

            if (!filtered.length) {
                setHint("No compatible models for this family.", "#c66");
                return;
            }

            const selectValue = filtered.find((m) => m === preferredValue)
                || filtered.find((m) => m === preselectedModel)
                || filtered[0];
            if (selectValue) {
                modelSel.value = selectValue;
            }
            const famText = familySel.selectedOptions?.[0]?.textContent || familySel.value || "selected family";
            const modeText = showAllModels
                ? `Showing all models on disk. Family (${famText}) still controls renderer pipeline.`
                : `Showing models filtered for family: ${famText}.`;
            setHint(`${filtered.length} model(s) available. ${modeText}`);
        };

        const loadFamilyModels = async (familyKey, preferredValue = "") => {
            setHint("Loading models...");
            modelSel.innerHTML = "";
            try {
                familyModels = await fetchRendererModelsForFamily(familyKey, showAllModels);
                renderModelOptions(preferredValue);
            } catch (e) {
                familyModels = [];
                setHint("Failed to load models for selected family.", "#c66");
                console.error("[ThumbnailGen] Error loading family models:", e);
            }
        };

        familySel.innerHTML = families
            .map((f) => `<option value="${f.key}">${f.label}</option>`)
            .join("");
        familySel.value = validFamily;

        const cleanup = () => {
            if (dialog.parentNode) document.body.removeChild(dialog);
        };

        familySel.onchange = async () => {
            await loadFamilyModels(familySel.value, "");
        };
        filterToggle.onclick = async () => {
            showAllModels = !showAllModels;
            _thumbnailShowAllModels = showAllModels;
            try {
                app.ui.settings.setSettingValue("PromptManager.ThumbnailShowAllModels", showAllModels);
            } catch (e) {
                console.warn("[ThumbnailGen] Could not persist show-all setting:", e);
            }
            updateFilterToggleUi();
            await loadFamilyModels(familySel.value, modelSel.value);
        };

        okBtn.onclick = () => {
            const family = familySel.value;
            const model = modelSel.value;
            if (!family || !model) {
                setHint("Please select a model before continuing.", "#c66");
                return;
            }
            resolve({ family, model });
            cleanup();
        };
        cancelBtn.onclick = () => { resolve(null); cleanup(); };

        dialog.onkeydown = (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                okBtn.click();
            } else if (e.key === "Escape") {
                e.preventDefault();
                cancelBtn.click();
            }
        };

        modelSel.ondblclick = () => okBtn.click();

        document.body.appendChild(dialog);
        loadFamilyModels(validFamily, preselectedModel || "");
        modelSel.focus();
    });
}

async function ensureThumbnailRenderSelection({ force = false } = {}) {
    if (!force && _thumbnailRenderFamily && _thumbnailRenderModel) {
        return { family: _thumbnailRenderFamily, model: _thumbnailRenderModel };
    }
    const picked = await showThumbnailRenderPicker(
        _thumbnailRenderFamily,
        _thumbnailRenderModel || _thumbnailLegacyModel || null
    );
    if (!picked) return null;
    saveThumbnailRenderSelection(picked);
    return picked;
}

/**
 * Build and queue a thumbnail generation workflow using RecipeRenderer.
 * Falls back to the basic thumbnail workflow when this path cannot run.
 * @param {Object} workflowData - Saved workflow_data payload
 * @returns {Promise<string|null>} Base64 thumbnail data URL or null
 */
async function generateThumbnailWorkflowFromWorkflowData(workflowData) {
    if (!workflowData || typeof workflowData !== "object") {
        return null;
    }

    // Enforce thumbnail-safe render sizing before queueing.
    const wfForThumb = applyThumbnailResolution(workflowData);

    const workflow = {};
    let nodeId = 1;

    const rendererId = String(nodeId++);
    workflow[rendererId] = {
        class_type: "RecipeRenderer",
        inputs: {
            recipe_data: wfForThumb,
            clear_cache_after_render: false,
        }
    };

    const saveId = String(nodeId++);
    workflow[saveId] = {
        class_type: "SaveImage",
        inputs: {
            filename_prefix: "_thumb_wf_",
            images: [rendererId, 1]
        }
    };

    // Quiet mode: this path is an enhancement and should silently fall back.
    const promptId = await queueThumbnailPrompt(workflow, { quiet: true });
    if (!promptId) return null;
    return await waitForThumbnailResult(promptId, saveId);
}

/**
 * Queue a workflow via ComfyUI's /prompt API
 * @returns {Promise<string|null>} prompt_id or null
 */
async function queueThumbnailPrompt(workflow, options = {}) {
    const quiet = options?.quiet === true;
    try {
        const body = {
            prompt: workflow,
            client_id: api.clientId
        };
        const resp = await fetch("/prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });
        const data = await resp.json();
        if (data.prompt_id) {
            return data.prompt_id;
        }
        console.error("[ThumbnailGen] Queue failed:", data);
        if (!quiet && data.error) {
            const detail = data.node_errors ? Object.values(data.node_errors).map(e => e.errors?.map(err => err.message).join(', ')).join('; ') : data.error;
            await showInfo("Generation Failed", `Could not queue thumbnail: ${detail}`);
        }
        return null;
    } catch (e) {
        console.error("[ThumbnailGen] Error queueing:", e);
        return null;
    }
}

/**
 * Wait for a queued prompt to complete and return the generated image as base64 thumbnail
 * @param {string} promptId - The prompt_id from queueing
 * @param {string} saveNodeId - The SaveImage node ID in the workflow
 * @returns {Promise<string|null>} Base64 thumbnail data URL or null
 */
function waitForThumbnailResult(promptId, saveNodeId) {
    return new Promise((resolve) => {
        let timeout = null;

        const onExecuted = (event) => {
            const detail = event.detail;
            if (detail?.prompt_id !== promptId) return;

            // Check for our save node's output
            const output = detail?.output;
            if (output?.images?.[0]) {
                cleanup();
                const img = output.images[0];
                fetchThumbnailImage(img.filename, img.subfolder, img.type).then(resolve);
            }
        };

        const onError = (event) => {
            if (event.detail?.prompt_id !== promptId) return;
            cleanup();
            console.error("[ThumbnailGen] Execution error:", event.detail);
            resolve(null);
        };

        const cleanup = () => {
            clearTimeout(timeout);
            api.removeEventListener("executed", onExecuted);
            api.removeEventListener("execution_error", onError);
        };

        // Timeout after 120 seconds
        timeout = setTimeout(() => {
            cleanup();
            console.warn("[ThumbnailGen] Timed out waiting for result");
            resolve(null);
        }, 120000);

        api.addEventListener("executed", onExecuted);
        api.addEventListener("execution_error", onError);
    });
}

/**
 * Fetch a generated image from ComfyUI and convert to base64 thumbnail
 * @param {string} filename
 * @param {string} subfolder
 * @param {string} type
 * @returns {Promise<string|null>} Base64 data URL
 */
async function fetchThumbnailImage(filename, subfolder, type) {
    try {
        const params = new URLSearchParams({ filename, subfolder: subfolder || '', type: type || 'output' });
        const resp = await fetch(`/view?${params}`);
        if (!resp.ok) return null;

        const blob = await resp.blob();

        // Resize to thumbnail using canvas
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const minSize = 200;
                let w = img.width, h = img.height;
                const minDim = Math.min(w, h);
                if (minDim > minSize) {
                    const scale = minSize / minDim;
                    w = Math.round(w * scale);
                    h = Math.round(h * scale);
                }
                const canvas = document.createElement('canvas');
                canvas.width = w;
                canvas.height = h;
                canvas.getContext('2d').drawImage(img, 0, 0, w, h);
                resolve(canvas.toDataURL('image/jpeg', 0.85));
            };
            img.onerror = () => resolve(null);
            img.src = URL.createObjectURL(blob);
        });
    } catch (e) {
        console.error("[ThumbnailGen] Error fetching image:", e);
        return null;
    }
}

/**
 * Main entry point: generate a thumbnail for a prompt
 * Uses RecipeRenderer from saved workflow_data when available, otherwise
 * builds a renderer-compatible workflow_data payload from prompt + selected model.
 */
async function generateThumbnailForPrompt(node, category, promptName, onUpdate, options = {}) {
    const { silent = false, renderSelection: providedRenderSelection = null, fallbackBase = null } = options || {};
    const promptData = node.prompts[category]?.[promptName];
    if (!promptData) {
        if (!silent) await showInfo("Error", "Prompt data not found.");
        return;
    }

    const promptText = promptData.prompt || promptName;

    // Show generating indicator (only in non-silent/single mode)
    let indicator = null;
    if (!silent) {
        indicator = document.createElement("div");
        indicator.style.cssText = `
            position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
            background: #222; border: 2px solid #4CAF50; border-radius: 8px;
            padding: 20px 30px; z-index: 10000; color: #fff; font-size: 14px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        `;
        indicator.innerHTML = `
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 20px; height: 20px; border: 3px solid #4CAF50; border-top-color: transparent; border-radius: 50%; animation: thumb-spin 1s linear infinite;"></div>
                <span>Generating thumbnail...</span>
            </div>
            <style>@keyframes thumb-spin { to { transform: rotate(360deg); } }</style>
        `;
        document.body.appendChild(indicator);
    }

    try {
        let thumbnail = null;

        // Preferred path: render from saved workflow_data using RecipeRenderer.
        const rawWorkflowData = promptData.workflow_data;
        let parsedWorkflowData = null;
        if (rawWorkflowData && typeof rawWorkflowData === "object") {
            parsedWorkflowData = rawWorkflowData;
        } else if (typeof rawWorkflowData === "string" && rawWorkflowData.trim()) {
            try {
                const maybeObj = JSON.parse(rawWorkflowData);
                if (maybeObj && typeof maybeObj === "object") {
                    parsedWorkflowData = maybeObj;
                }
            } catch {
                parsedWorkflowData = null;
            }
        }

        if (parsedWorkflowData) {
            try {
                thumbnail = await generateThumbnailWorkflowFromWorkflowData(parsedWorkflowData);
            } catch (e) {
                console.warn("[ThumbnailGen] RecipeRenderer thumbnail path failed, falling back:", e);
                thumbnail = null;
            }
        }

        // Fallback path: build renderer workflow_data from prompt + selected family/model.
        if (!thumbnail) {
            const renderSelection = providedRenderSelection || await ensureThumbnailRenderSelection();
            if (!renderSelection) return; // User cancelled

            const fallbackWorkflowData = await buildRendererFallbackWorkflowDataWithBase(
                promptText,
                promptData,
                renderSelection,
                fallbackBase
            );
            thumbnail = await generateThumbnailWorkflowFromWorkflowData(fallbackWorkflowData);
        }

        if (thumbnail) {
            await saveThumbnail(node, category, promptName, thumbnail);
            onUpdate();
        } else {
            if (!silent) await showInfo("Generation Failed", "Could not generate thumbnail. The model may be incompatible or an error occurred.");
        }
    } catch (e) {
        console.error("[ThumbnailGen] Error:", e);
        if (!silent) await showInfo("Error", `Thumbnail generation failed: ${e.message}`);
        else throw e;
    } finally {
        if (indicator?.parentNode) {
            document.body.removeChild(indicator);
        }
    }
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
                    const thumbnail = await resizeImageToThumbnail(file, 200);
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
                    const thumbnail = await resizeImageToThumbnail(file, 200);
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

    // Generate thumbnail divider
    const genDivider = document.createElement("div");
    genDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
    menu.appendChild(genDivider);

    // Generate Thumbnail
    menu.appendChild(createMenuItem("🎨 Generate Thumbnail", async () => {
        await generateThumbnailForPrompt(node, category, promptName, onUpdate);
    }));

    // Change Thumbnail Family/Model
    const modelLabel = (_thumbnailRenderFamily && _thumbnailRenderModel)
        ? `🔧 ${_thumbnailRenderFamily} : ${_thumbnailLeafName(_thumbnailRenderModel)}`
        : "🔧 Select Thumbnail Family + Model";
    menu.appendChild(createMenuItem(modelLabel, async () => {
        const picked = await showThumbnailRenderPicker(_thumbnailRenderFamily, _thumbnailRenderModel);
        if (picked) {
            saveThumbnailRenderSelection(picked);
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

    // NSFW toggle divider + option
    const nsfwDivider = document.createElement("div");
    nsfwDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
    menu.appendChild(nsfwDivider);

    const isNSFW = promptData?.nsfw === true;
    const nsfwItem = createMenuItem(isNSFW ? "✓ NSFW" : "Mark as NSFW", async () => {
        try {
            const resp = await fetch("/prompt-manager-advanced/toggle-nsfw", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ type: "prompt", category: category, name: promptName })
            });
            const result = await resp.json();
            if (result.success) {
                node.prompts = result.prompts;
                onUpdate();
            }
        } catch (err) {
            console.error("[PromptManagerAdvanced] Error toggling prompt NSFW:", err);
        }
    });
    nsfwItem.style.color = isNSFW ? '#f66' : '#ccc';
    menu.appendChild(nsfwItem);

    // Rename / Move Prompt
    const renameDivider = document.createElement("div");
    renameDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
    menu.appendChild(renameDivider);

    menu.appendChild(createMenuItem("✏️ Rename / Move", async () => {
        const allCategories = Object.keys(node.prompts).filter(c => c !== "__meta__").sort((a, b) => a.localeCompare(b));
        const result = await showPromptWithCategory(
            "Rename / Move Prompt",
            "Prompt name:",
            promptName,
            allCategories,
            category,
            promptData?.nsfw || false
        );
        if (result && result.name && result.name.trim()) {
            const newName = result.name.trim();
            const newCat = result.category;
            if (newName === promptName && newCat === category) return;
            try {
                const resp = await fetch("/prompt-manager-advanced/rename-prompt", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ category: category, old_name: promptName, new_name: newName, new_category: newCat })
                });
                const data = await resp.json();
                if (data.success) {
                    node.prompts = data.prompts;
                    onUpdate();
                } else {
                    await showInfo("Error", data.error);
                }
            } catch (err) {
                console.error("[PromptManagerAdvanced] Error renaming prompt:", err);
            }
        }
    }));

    // Delete Prompt
    const deleteDivider = document.createElement("div");
    deleteDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
    menu.appendChild(deleteDivider);

    menu.appendChild(createMenuItem("🗑️ Delete Prompt", async () => {
        if (await showConfirm("Delete Prompt", `Are you sure you want to delete prompt "${promptName}"?`)) {
            await deletePrompt(node, category, promptName);
            onUpdate();
        }
    }));
    // Style the delete item red
    menu.lastChild.style.color = '#f66';

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

    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const inp = node.inputs[i];
        if (inp.name === "category" || inp.name === "name" || inp.name === "prompt_input") {
            node.removeInput(i);
        }
    }

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

    // Thumbnail preview tooltip (screen-fixed, not affected by canvas zoom)
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

    // Show/hide thumbnail on hover
    let hoverTimeout = null;
    nameDisplay.addEventListener("mouseenter", () => {
        if (node._isWorkflowManager) {
            return;
        }
        hoverTimeout = setTimeout(() => {
            const category = categoryWidget.value;
            const prompt = promptWidget.value;
            const promptData = node.prompts?.[category]?.[prompt];
            const thumbnail = promptData?.thumbnail || DEFAULT_THUMBNAIL;
            
            // Don't show preview for placeholder/default thumbnail
            if (thumbnail === DEFAULT_THUMBNAIL) {
                return;
            }
            
            // Load image to get natural dimensions and display at 1x scale
            const tempImg = new Image();
            tempImg.onload = function() {
                const imgWidth = this.naturalWidth;
                const imgHeight = this.naturalHeight;
                
                // Display at actual thumbnail size (1x scale) with max constraints
                thumbnailImg.style.width = imgWidth + 'px';
                thumbnailImg.style.height = imgHeight + 'px';
                thumbnailImg.src = thumbnail;
                
                // Position using screen coordinates (immune to canvas zoom)
                const rect = nameDisplay.getBoundingClientRect();
                const margin = 8; // Margin between button and preview
                
                // Calculate preview dimensions (no padding, just image)
                const previewWidth = Math.min(imgWidth, 300);
                const previewHeight = Math.min(imgHeight, 300);
                
                // Center horizontally above the button
                let left = rect.left + (rect.width / 2) - (previewWidth / 2);
                let top = rect.top - previewHeight - margin;
                
                // Keep on screen with padding
                left = Math.max(5, Math.min(left, window.innerWidth - previewWidth - 5));
                top = Math.max(5, Math.min(top, window.innerHeight - previewHeight - 5));
                
                thumbnailPreview.style.left = left + 'px';
                thumbnailPreview.style.top = top + 'px';
                thumbnailPreview.style.display = "block";
            };
            tempImg.src = thumbnail;
        }, 300);
    });
    nameDisplay.addEventListener("mouseleave", () => {
        if (node._isWorkflowManager) {
            return;
        }
        if (hoverTimeout) clearTimeout(hoverTimeout);
        thumbnailPreview.style.display = "none";
    });
    
    // Clean up preview when node is removed
    node.onRemoved = function() {
        if (thumbnailPreview && thumbnailPreview.parentNode) {
            thumbnailPreview.parentNode.removeChild(thumbnailPreview);
        }
    };

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

    // Update display function - show CATEGORY : PROMPT with optional NSFW label
    const updateDisplay = () => {
        const category = categoryWidget.value || "";
        const prompt = promptWidget.value || "new prompt";

        const wmInputMode = node._isWorkflowManager && hasConnectedWorkflowInput(node);
        const hasLiveWorkflow = !!(node.lastWorkflowData && typeof node.lastWorkflowData === "object");

        if (wmInputMode) {
            nameDisplay.textContent = hasLiveWorkflow
                ? "Input connected: save incoming workflow"
                : "Waiting for workflow_data input...";
            nameDisplay.title = nameDisplay.textContent;
            nameDisplay.style.cursor = "default";
            leftArrow.style.opacity = "0.35";
            rightArrow.style.opacity = "0.35";
            leftArrow.style.cursor = "default";
            rightArrow.style.cursor = "default";
        } else {
            nameDisplay.textContent = `${category} : ${prompt}`;
            nameDisplay.title = `${category} : ${prompt}`;
            nameDisplay.style.cursor = "pointer";
            leftArrow.style.opacity = "1";
            rightArrow.style.opacity = "1";
            leftArrow.style.cursor = "pointer";
            rightArrow.style.cursor = "pointer";
        }

        if (node._isWorkflowManager) {
            updateWorkflowManagerPreview(node);
        }

        // Remove existing badges if any
        const existingNsfwLabel = nameDisplay.querySelector('.nsfw-selector-label');
        if (existingNsfwLabel) existingNsfwLabel.remove();
        const existingWorkflowLabel = nameDisplay.querySelector('.workflow-selector-label');
        if (existingWorkflowLabel) existingWorkflowLabel.remove();
        const existingFoundLabel = nameDisplay.querySelector('.workflow-found-label');
        if (existingFoundLabel) existingFoundLabel.remove();
        const existingInfoLabel = nameDisplay.querySelector('.workflow-info-label');
        if (existingInfoLabel) existingInfoLabel.remove();

        const createBadge = (className, text, styleBlock, onClick = null) => {
            const label = document.createElement("span");
            label.className = className;
            label.textContent = text;
            label.style.cssText = styleBlock;
            if (typeof onClick === "function") {
                label.style.pointerEvents = "auto";
                label.onclick = (evt) => {
                    evt.stopPropagation();
                    onClick(evt);
                };
            }
            nameDisplay.appendChild(label);
            return label;
        };

        // Show red NSFW badge for NSFW prompts (or prompts in NSFW categories) — always visible.
        // Also show orange WORKFLOW badge when selected prompt contains workflow_data.
        if (!wmInputMode && node.prompts && category) {
            const promptData = node.prompts[category]?.[prompt] || null;
            const catIsNSFW = node.prompts[category]?.["__meta__"]?.nsfw === true;
            const promptIsNSFW = promptData?.nsfw === true;
            const rawWorkflowData = promptData?.workflow_data;
            const hasWorkflowData = (
                (typeof rawWorkflowData === "string" && rawWorkflowData.trim().length > 0) ||
                (rawWorkflowData && typeof rawWorkflowData === "object" && Object.keys(rawWorkflowData).length > 0)
            );

            if (catIsNSFW || promptIsNSFW) {
                createBadge("nsfw-selector-label", "NSFW", `
                    background: rgba(204, 50, 50, 0.85);
                    color: #fff;
                    font-size: 8px;
                    font-weight: bold;
                    padding: 0px 4px;
                    border-radius: 2px;
                    margin-left: 6px;
                    flex-shrink: 0;
                    line-height: 12px;
                `);
            }

            if (hasWorkflowData && !node?._isWorkflowManager) {
                createBadge("workflow-selector-label", "R", `
                    width: 14px;
                    height: 14px;
                    border-radius: 50%;
                    background: rgba(235, 140, 35, 0.95);
                    color: #fff;
                    font-size: 9px;
                    font-weight: bold;
                    margin-left: 6px;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    flex-shrink: 0;
                    line-height: 1;
                `);
            }
        }

        let infoEnabled = false;
        if (node._isWorkflowManager) {
            const summary = _summarizeWorkflowDataDiscovery(node.lastWorkflowData);
            infoEnabled = summary.modelCount > 0;
        } else if (!wmInputMode && node.prompts && category) {
            const promptData = node.prompts[category]?.[prompt] || null;
            infoEnabled = hasMeaningfulWorkflowData(promptData?.workflow_data);
        }

        if (!node._isWorkflowManager) {
            createBadge("workflow-info-label", "info", `
                margin-left: 6px;
                width: 28px;
                height: 14px;
                border-radius: 999px;
                border: 1px solid rgba(235, 245, 255, 0.95);
                background: ${infoEnabled ? "rgba(55, 142, 255, 0.92)" : "rgba(120, 130, 140, 0.35)"};
                color: ${infoEnabled ? "#ffffff" : "rgba(220, 226, 234, 0.75)"};
                font-size: 8px;
                font-weight: 700;
                letter-spacing: 0.2px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                line-height: 1;
                flex-shrink: 0;
                cursor: ${infoEnabled ? "pointer" : "default"};
                opacity: ${infoEnabled ? "1" : "0.75"};
            `, infoEnabled ? () => {
                void showWorkflowDiscoverySummary(node);
            } : null);
        }
    };

    // Get flattened list of all prompts across all categories for navigation
    const getAllPromptsFlat = () => {
        const allPrompts = [];
        if (!node.prompts) return allPrompts;

        const hideNSFW = app.ui.settings.getSettingValue("PromptManager.DefaultHideNSFW");
        const workflowOnly = node?._isWorkflowManager === true;
        const categories = getVisibleCategories(node, {
            hideNSFW: hideNSFW === true,
            workflowOnly,
        });
        for (const cat of categories) {
            // Skip NSFW categories when preference is set to hide
            if (hideNSFW && node.prompts[cat]?.["__meta__"]?.nsfw === true) continue;

            const prompts = getPromptNamesForCategory(node, cat, {
                hideNSFW: hideNSFW === true,
                workflowOnly,
            });
            for (const prompt of prompts) {
                // Skip NSFW prompts when preference is set to hide
                if (hideNSFW && node.prompts[cat][prompt]?.nsfw === true) continue;
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
            const warnEnabled = app.ui.settings.getSettingValue("PromptManager.WarnUnsavedChanges");

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
        if (node._isWorkflowManager && hasConnectedWorkflowInput(node)) return;
        const allPrompts = getAllPromptsFlat();
        if (allPrompts.length === 0) return;

        const currentIndex = getCurrentIndex(allPrompts);
        const newIndex = currentIndex <= 0 ? allPrompts.length - 1 : currentIndex - 1;
        await navigateTo(allPrompts[newIndex]);
    };

    // Navigate to next prompt (wraps across categories)
    rightArrow.onclick = async (e) => {
        e.stopPropagation();
        if (node._isWorkflowManager && hasConnectedWorkflowInput(node)) return;
        const allPrompts = getAllPromptsFlat();
        if (allPrompts.length === 0) return;

        const currentIndex = getCurrentIndex(allPrompts);
        const newIndex = currentIndex >= allPrompts.length - 1 ? 0 : currentIndex + 1;
        await navigateTo(allPrompts[newIndex]);
    };

    // Open thumbnail browser on click
    nameDisplay.onclick = async (e) => {
        e.stopPropagation();

        if (node._isWorkflowManager && hasConnectedWorkflowInput(node)) {
            return;
        }

        try {
            // Check for unsaved changes before opening browser
            const hasUnsaved = hasUnsavedChanges(node);
            const warnEnabled = app.ui.settings.getSettingValue("PromptManager.WarnUnsavedChanges");

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

            const selection = await showThumbnailBrowser(node, category, currentPrompt, {
                workflowOnly: node?._isWorkflowManager === true,
            });

            if (selection) {
                // Navigate to the selected category/prompt (skip unsaved check since we already confirmed)
                await navigateTo(selection, true);

                // Clear new prompt flag after successful navigation
                node.isNewUnsavedPrompt = false;
                node.newPromptCategory = null;
                node.newPromptName = null;
            }
        } catch (err) {
            console.error("[PromptManagerAdvanced] Error opening prompt browser:", err);
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
    node._promptSelectorContainer = container;
    node.updatePromptSelectorDisplay = updateDisplay;

    return widget;
}
