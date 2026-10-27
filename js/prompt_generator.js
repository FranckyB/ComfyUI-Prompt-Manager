import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { showThumbnailBrowser } from "./prompt_browser.js";
import { SYSTEM_PROMPTS_ENDPOINT_PREFIX, getCategoryPromptEntries } from "./prompt_store_adapters.js";
import { setPromptTextLocked } from "./prompt_text_lock.js";

function setupSystemPromptSelector(node) {
    const widget = node.widgets?.find(w => w.name === "system_prompt");
    if (!widget || widget.type === "system_prompt_selector") return;

    const selectPrompt = async (value) => {
        widget.value = value;
        await widget.callback?.call(widget, value, app.canvas, node);
        updateDisplay();
        node.setDirtyCanvas(true, true);
    };

    const loadSystemPrompts = async () => {
        const response = await api.fetchApi(`${SYSTEM_PROMPTS_ENDPOINT_PREFIX}/get-prompts`, { cache: "no-store" });
        if (!response.ok) throw new Error(`Could not load system prompts (${response.status})`);
        const prompts = await response.json();
        node.prompts = prompts;
        return prompts;
    };

    const getChoices = (prompts) => Object.entries(prompts)
        .filter(([category, entries]) => !category.startsWith("_") && entries && typeof entries === "object")
        .flatMap(([category, entries]) => Object.entries(getCategoryPromptEntries(entries, "generator"))
            .filter(([name, entry]) => {
                const text = entry && typeof entry === "object" ? entry.prompt : entry;
                return !name.startsWith("_") && String(text ?? "").trim();
            })
            .map(([name]) => `${category} / ${name}`))
        .sort((left, right) => left.toLowerCase().localeCompare(right.toLowerCase()));

    let busy = false;
    const activate = async (direction) => {
        if (busy) return;
        busy = true;
        try {
            if (direction) {
                const choices = getChoices(await loadSystemPrompts());
                if (!choices.length) return;
                const index = choices.indexOf(widget.value);
                const nextIndex = index < 0
                    ? (direction > 0 ? 0 : choices.length - 1)
                    : (index + direction + choices.length) % choices.length;
                await selectPrompt(choices[nextIndex]);
            } else {
                const value = String(widget.value ?? "");
                const separator = value.indexOf(" / ");
                const category = separator < 0 ? "" : value.slice(0, separator);
                const name = separator < 0 ? value : value.slice(separator + 3);
                const selection = await showThumbnailBrowser(node, category, name, {
                    title: "Select System Prompt",
                    endpointPrefix: SYSTEM_PROMPTS_ENDPOINT_PREFIX,
                    loadPromptsFn: loadSystemPrompts,
                    preferenceScope: "system",
                    promptOnly: true,
                    allowMultiSelect: false,
                });
                if (selection?.prompt) {
                    await selectPrompt(`${selection.category} / ${selection.prompt}`);
                }
            }
        } catch (error) {
            console.error("[PromptGenerator] System prompt selection failed:", error);
            app.extensionManager.toast.add({
                severity: "error",
                summary: "System prompts unavailable",
                detail: String(error.message || error),
            });
        } finally {
            busy = false;
        }
    };

    const container = document.createElement("div");
    // Mirrors Prompt Manager's selector; the dark backing is what keeps the translucent name button dark.
    container.style.cssText = `
        display: flex;
        align-items: center;
        gap: 0;
        background: #1a1a1a;
        border-radius: 6px;
        width: 100%;
        min-width: 0;
        height: 26px;
        box-sizing: border-box;
    `;
    for (const eventName of ["pointerdown", "mousedown", "dblclick"]) {
        container.addEventListener(eventName, event => event.stopPropagation());
    }

    const makeButton = (label, title, direction) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.title = title;
        button.setAttribute("aria-label", title);
        button.style.cssText = `
            background: #2a2a2a;
            border: none;
            color: #888;
            padding: 0 10px;
            height: 100%;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.15s ease;
        `;
        button.addEventListener("mouseover", () => {
            button.style.background = direction ? "#3a3a3a" : "rgba(56, 130, 246, 0.32)";
            if (direction) button.style.color = "#fff";
        });
        button.addEventListener("mouseout", () => {
            button.style.background = direction ? "#2a2a2a" : "rgba(56, 130, 246, 0.22)";
            if (direction) button.style.color = "#888";
        });
        button.addEventListener("click", event => {
            event.stopPropagation();
            void activate(direction);
        });
        container.appendChild(button);
        return button;
    };

    makeButton("\u25c0", "Previous system prompt", -1);
    const nameButton = makeButton("", "Browse System Prompts", 0);
    nameButton.style.cssText = `
        flex: 1;
        min-width: 0;
        text-align: center;
        color: #dbeafe;
        font-family: inherit;
        font-size: 15px;
        padding: 0 10px;
        cursor: pointer;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        background: rgba(56, 130, 246, 0.22);
        border: 1px solid rgba(56, 130, 246, 0.85);
        border-radius: 4px;
        height: 100%;
        transition: background 0.15s ease;
        box-sizing: border-box;
    `;
    makeButton("\u25b6", "Next system prompt", 1);
    const updateDisplay = () => {
        nameButton.textContent = String(widget.value ?? "").replace(" / ", " : ");
        nameButton.title = `Browse System Prompts: ${widget.value ?? ""}`;
    };
    const index = node.widgets.indexOf(widget);
    const selector = node.addDOMWidget("system_prompt", "system_prompt_selector", container, {
        getMinHeight: () => 26,
        getMaxHeight: () => 26,
        getValue: () => widget.value,
        setValue: value => {
            widget.value = value;
            updateDisplay();
        },
    });
    selector.computeSize = width => [width, 28];
    // Nodes 2.0 stretches DOM widgets that expose computeLayoutSize; keep this row fixed-height.
    selector.computeLayoutSize = undefined;
    selector.serializeValue = () => widget.value;
    node.widgets.splice(node.widgets.indexOf(selector), 1);
    node.widgets.splice(index, 1, selector);
    const onConfigure = node.onConfigure;
    node.onConfigure = function() {
        const result = onConfigure?.apply(this, arguments);
        const choices = typeof widget.options?.values === "function"
            ? widget.options.values()
            : widget.options?.values;
        if (!Array.isArray(choices) || !choices.length) return result;
        const previous = String(widget.value ?? "").trim();
        const normalized = previous.toLowerCase();
        const entries = choices.map(value => {
            const separator = value.indexOf(" / ");
            return { value, category: value.slice(0, separator), name: value.slice(separator + 3) };
        });
        const exact = entries.find(entry => entry.value.toLowerCase() === normalized);
        const legacy = entries.find(entry => `${entry.name} (${entry.category})`.toLowerCase() === normalized);
        const names = entries.filter(entry => entry.name.toLowerCase() === normalized);
        const fallback = entries.find(entry => entry.value.toLowerCase() === "image / enhance prompt") || entries[0];
        const match = exact || legacy || (names.length === 1 ? names[0] : null) || fallback;
        if (widget.value !== match.value) {
            selector.value = match.value;
            console.log(`[PromptGenerator] Migrated system prompt '${previous}' to '${match.value}'.`);
            node.setDirtyCanvas(true, true);
        }
        return result;
    };
    updateDisplay();
}

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
                setupSystemPromptSelector(this);
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
                    setPromptTextLocked(promptWidget, isPromptInputConnected());
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
                                }
                                setPromptTextLocked(promptWidget, true);
                            }
                        } else {
                            setPromptTextLocked(promptWidget, false);
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
