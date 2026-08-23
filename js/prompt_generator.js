import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function migrateLegacyPromptGenOptionsWidgets(node) {
    if (!node?.widgets || node.widgets.length === 0) {
        return false;
    }

    const byName = (name) => node.widgets.find(w => w.name === name);
    const modeWidget = byName("system_prompt_mode");
    const systemPromptWidget = byName("system_prompt");
    const useDefaultWidget = byName("use_model_default_sampling");
    const temperatureWidget = byName("temperature");
    const topKWidget = byName("top_k");
    const topPWidget = byName("top_p");
    const minPWidget = byName("min_p");
    const repeatPenaltyWidget = byName("repeat_penalty");
    const contextSizeWidget = byName("context_size");
    const showConsoleWidget = byName("show_everything_in_console");

    if (!modeWidget || !systemPromptWidget || !useDefaultWidget || !temperatureWidget ||
        !topKWidget || !topPWidget || !minPWidget || !repeatPenaltyWidget ||
        !contextSizeWidget || !showConsoleWidget) {
        return false;
    }

    const modeIsValid = modeWidget.value === "replace" || modeWidget.value === "append";
    const looksLegacyShape = typeof systemPromptWidget.value === "boolean" && typeof useDefaultWidget.value === "number";

    if (modeIsValid && !looksLegacyShape) {
        return false;
    }

    const oldSystemPrompt = modeWidget.value;
    const oldUseDefaultSampling = systemPromptWidget.value;
    const oldTemperature = useDefaultWidget.value;
    const oldTopK = temperatureWidget.value;
    const oldTopP = topPWidget.value;
    const oldMinP = topPWidget.value;
    const oldRepeatPenalty = repeatPenaltyWidget.value;
    const oldContextSize = contextSizeWidget.value;
    const oldShowInConsole = showConsoleWidget.value;

    modeWidget.value = "replace";
    systemPromptWidget.value = typeof oldSystemPrompt === "string" ? oldSystemPrompt : "";
    useDefaultWidget.value = Boolean(oldUseDefaultSampling);
    temperatureWidget.value = Number(oldTemperature);
    topKWidget.value = Number.isFinite(Number(oldTopK)) ? Math.round(Number(oldTopK)) : oldTopK;
    topPWidget.value = Number(oldTopP);
    minPWidget.value = Number(oldMinP);
    repeatPenaltyWidget.value = Number(oldRepeatPenalty);
    contextSizeWidget.value = Number.isFinite(Number(oldContextSize)) ? Math.round(Number(oldContextSize)) : oldContextSize;
    showConsoleWidget.value = Boolean(oldShowInConsole);

    node.serialize_widgets = true;
    app.graph.setDirtyCanvas(true, true);
    console.log("[PromptManager] Migrated legacy PromptGenOptions widget order for workflow compatibility.");
    return true;
}

app.registerExtension({
    name: "comfyui-prompt-manager.generator",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Make the Options node taller/wider by default so the system_prompt area is roomier
        if (nodeData.name === "PromptGenOptions") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                migrateLegacyPromptGenOptionsWidgets(this);
                // Model selection now lives on the Prompt Generator base node. The
                // Options node's model widget is kept (hidden) purely so old workflows
                // still load and pass their value through as a fallback.
                const modelWidget = this.widgets?.find(w => w.name === "model");
                if (modelWidget) {
                    modelWidget.hidden = true;
                    modelWidget.computeSize = () => [0, -4];
                    if (modelWidget.inputEl) modelWidget.inputEl.style.display = "none";
                }
                // Set a default size (width x height)
                try {
                    this.setSize([400, 420]);
                } catch (e) {
                    // ignore if method unavailable
                }
                return result;
            };
            const onConfigureOpt = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                const result = onConfigureOpt?.apply(this, arguments);
                migrateLegacyPromptGenOptionsWidgets(this);
                // Re-hide the legacy model widget on restore (tab switch / reload).
                const modelWidget = this.widgets?.find(w => w.name === "model");
                if (modelWidget) {
                    modelWidget.hidden = true;
                    modelWidget.computeSize = () => [0, -4];
                    if (modelWidget.inputEl) modelWidget.inputEl.style.display = "none";
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

        // Enforce minimum size for PromptGenerator node (no default size)
        if (nodeData.name === "PromptGenerator") {
           const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                // Set a default size (width x height) - taller to fit the UI comfortably
                try {
                    this.setSize([400, 600]);
                } catch (e) {
                    // ignore if method unavailable
                }

                // Darken + make the prompt widget read-only while a `prompt_input`
                // link is connected (mirrors PromptManager's use_prompt_input UI).
                const node = this;
                const promptWidget = node.widgets?.find(w => w.name === "prompt");
                const promptInputSlot = () => node.inputs?.find(i => i && i.name === "prompt_input");
                const isPromptInputConnected = () => {
                    const inp = promptInputSlot();
                    return !!(inp && inp.link != null);
                };

                const applyPromptInputState = () => {
                    if (!promptWidget || !promptWidget.inputEl) return;
                    const connected = isPromptInputConnected();
                    promptWidget.inputEl.readOnly = connected;
                    promptWidget.inputEl.style.opacity = connected ? "0.5" : "";
                };
                // Python echoes back the value it actually received on prompt_input
                // (only when that input was connected) in the "executed" event.
                // Display it here; when the key is absent, never touch the widget.
                const prevOnConnectionsChange = node.onConnectionsChange;
                node.onConnectionsChange = function () {
                    const r = prevOnConnectionsChange?.apply(this, arguments);
                    applyPromptInputState();
                    return r;
                };
                const prevOnConfigure = node.onConfigure;
                node.onConfigure = function () {
                    const r = prevOnConfigure?.apply(this, arguments);
                    applyPromptInputState();
                    return r;
                };
                const prevOnExecuted = node.onExecuted;
                node.onExecuted = function (output) {
                    const r = prevOnExecuted?.apply(this, arguments);
                    applyPromptInputState();
                    const echoed = output && output.ui ? output.ui.prompt : null;
                    if (Array.isArray(echoed) && typeof echoed[0] === "string" && echoed[0].length > 0) {
                        promptWidget.value = echoed[0];
                        if (promptWidget.inputEl) {
                            promptWidget.inputEl.value = echoed[0];
                        }
                        node.setDirtyCanvas(true, true);
                    }
                    return r;
                };

                // Listen for incoming prompt_input text updates from the backend.
                // This fires at the very start of execution, before the LLM call.
                api.addEventListener("prompt-generator-update-text", (event) => {
                    if (String(event.detail.node_id) === String(node.id)) {
                        const incoming = event.detail.prompt || "";
                        if (event.detail.has_prompt_input) {
                            if (promptWidget) {
                                promptWidget.value = incoming;
                                if (promptWidget.inputEl) {
                                    promptWidget.inputEl.value = incoming;
                                    promptWidget.inputEl.readOnly = true;
                                    promptWidget.inputEl.style.opacity = "0.5";
                                }
                            }
                        } else if (promptWidget && promptWidget.inputEl) {
                            promptWidget.inputEl.readOnly = false;
                            promptWidget.inputEl.style.opacity = "";
                        }
                        node.serialize_widgets = true;
                        app.graph.setDirtyCanvas(true, true);
                    }
                });

                // Initial state.
                applyPromptInputState();

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

console.log("[PromptManager] PromptGenerator extension loaded");
