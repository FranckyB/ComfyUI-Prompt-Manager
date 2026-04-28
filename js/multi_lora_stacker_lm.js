import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { PM_UI_PALETTE } from "./ui_palette.js";

const NODE_CLASS = "MultiLoraStackerLM";
const LM_PROVIDER_CLASS = "Lora Stacker (LoraManager)";
const STYLE_ID = "pm-multi-lm-style";
const MIN_NODE_WIDTH = 920;
// Height constants mirroring LM's loras_widget_utils.js
const LM_LORA_ENTRY_H = 40;
const LM_HEADER_H = 32;
const LM_CONTAINER_PAD = 12;
const LM_EMPTY_H = 100;
// Per-column chrome: title(22) + search textarea(50) + col padding(16)
const COL_CHROME_H = 88;

let loraCodeListenerAttached = false;

const SLOT_DEFS = [
    { key: "model_a", label: "Model A", short: "A", state: "loras_state_a" },
    { key: "model_b", label: "Model B", short: "B", state: "loras_state_b" },
    { key: "model_c", label: "Model C", short: "C", state: "loras_state_c" },
    { key: "model_d", label: "Model D", short: "D", state: "loras_state_d" },
];

let lmBridgePromise = null;

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const p = PM_UI_PALETTE || {};
    const panel = p.panel || "hsl(216 11% 15%)";
    const panelBorder = p.panelBorder || "hsl(216 20% 65% / 0.24)";
    const inputBg = p.inputBg || "hsl(220 15% 10%)";
    const inputBorder = p.inputBorder || "hsl(218 10% 41%)";
    const textHeading = p.textHeading || "hsl(220 13% 85%)";
    const textPrimary = p.textPrimary || "hsl(0 0% 87%)";
    const textHint = p.textHint || "hsl(216 15% 65%)";
    const cardBg = p.cardBg || "hsl(219 16% 18%)";
    const accentBorder = p.accentBorder || "hsl(208 73% 57% / 0.65)";
    const accentSoft = p.accentSoft || "hsl(208 73% 57% / 0.16)";

    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.pm-multi-lm-root {
    width: 100%;
    height: 100%;
    box-sizing: border-box;
    padding: 6px 6px 6px 6px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
.pm-multi-lm-root,
.pm-multi-lm-root * {
    scrollbar-width: none;
    -ms-overflow-style: none;
}
.pm-multi-lm-root::-webkit-scrollbar,
.pm-multi-lm-root *::-webkit-scrollbar {
    width: 0;
    height: 0;
    display: none;
}
.pm-multi-lm-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(200px, 1fr));
    gap: 8px;
    flex: 1;
    min-height: 0;
    align-items: stretch;
}
.pm-multi-lm-col {
    border: 1px solid ${panelBorder};
    border-radius: 8px;
    background: ${panel};
    padding: 8px;
    min-width: 0;
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    min-height: 0;
    height: 100%;
}
.pm-multi-lm-col.active-slot {
    border-color: ${accentBorder};
    box-shadow: 0 0 0 1px ${accentSoft};
}
.pm-multi-lm-title {
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.02em;
    color: ${textHeading};
    margin-bottom: 6px;
    flex-shrink: 0;
}
.pm-multi-lm-search {
    width: 100%;
    min-height: 38px;
    resize: vertical;
    box-sizing: border-box;
    border: 1px solid ${inputBorder};
    border-radius: 6px;
    padding: 6px 8px;
    background: ${inputBg};
    color: ${textPrimary};
    margin-bottom: 6px;
    font-size: 12px;
    flex-shrink: 0;
}
.pm-multi-lm-search::placeholder {
    color: ${textHint};
}
.pm-multi-lm-list-host {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-width: none;
    -ms-overflow-style: none;
}
.pm-multi-lm-list-host .lm-loras-container {
    height: 100%;
    max-height: none;
    background: ${cardBg};
    scrollbar-width: none;
    -ms-overflow-style: none;
}
.pm-multi-lm-list-host::-webkit-scrollbar,
.pm-multi-lm-list-host .lm-loras-container::-webkit-scrollbar {
    width: 0;
    height: 0;
    display: none;
}
`;
    document.head.appendChild(style);
}

// ---------------------------------------------------------------------------
// LM bridge loader
// ---------------------------------------------------------------------------

async function loadLmBridge() {
    if (lmBridgePromise) return lmBridgePromise;
    lmBridgePromise = Promise.all([
        import("/extensions/ComfyUI-Lora-Manager/autocomplete.js"),
        import("/extensions/ComfyUI-Lora-Manager/loras_widget.js"),
        import("/extensions/ComfyUI-Lora-Manager/utils.js"),
        import("/extensions/ComfyUI-Lora-Manager/lora_syntax_utils.js"),
    ]).then(([autocompleteMod, lorasWidgetMod, utilsMod, syntaxMod]) => ({
        AutoComplete: autocompleteMod.AutoComplete,
        addLorasWidget: lorasWidgetMod.addLorasWidget,
        mergeLoras: utilsMod.mergeLoras,
        applyLoraValuesToText: syntaxMod.applyLoraValuesToText,
    })).catch((err) => {
        console.warn("[PromptManager] Failed to load Lora-Manager bridge modules", err);
        lmBridgePromise = null;
        return null;
    });
    return lmBridgePromise;
}

// ---------------------------------------------------------------------------
// Widget helpers
// ---------------------------------------------------------------------------

function getWidgetByName(node, name) {
    if (!node || !Array.isArray(node.widgets)) return null;
    return node.widgets.find((w) => w && w.name === name) ?? null;
}

function ensureStateWidget(node, name) {
    let widget = getWidgetByName(node, name);
    if (widget) return widget;
    if (!node || typeof node.addWidget !== "function") return null;
    return node.addWidget("text", name, "[]", null, { multiline: true });
}

function hideWidget(widget) {
    if (!widget || widget.__pm_hidden) return;
    widget.__pm_hidden = true;
    // Keep original widget type (STRING) so graphToPrompt includes state values.
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = function () {};
    if (widget.element) widget.element.style.display = "none";
}

function hideStateWidgets(node) {
    if (!node || !Array.isArray(node.widgets)) return;
    for (const widget of node.widgets) {
        if (!widget || widget.name === "multi_lora_ui") continue;
        hideWidget(widget);
    }
}

function parseLorasState(value) {
    if (Array.isArray(value)) return value;
    if (value && typeof value === "object" && Array.isArray(value.__value__)) {
        return value.__value__;
    }
    if (value && typeof value === "object" && typeof value.__value__ === "string") {
        try {
            const parsed = JSON.parse(value.__value__);
            return Array.isArray(parsed) ? parsed : [];
        } catch (_e) {
            return [];
        }
    }
    if (typeof value !== "string") return [];
    const raw = value.trim();
    if (!raw || raw === "[]") return [];
    try {
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed : [];
    } catch (_e) {
        return [];
    }
}

function writeLorasState(widget, listValue) {
    if (!widget) return;
    widget.value = JSON.stringify(Array.isArray(listValue) ? listValue : []);
}

function getSlotByKey(node, slotKey) {
    if (!node || !Array.isArray(node.__pmMultiLmSlots)) return null;
    return node.__pmMultiLmSlots.find((s) => s && s.key === slotKey) ?? null;
}

// ---------------------------------------------------------------------------
// Apply incoming lora code to a slot
// ---------------------------------------------------------------------------

function applyIncomingLoraCode(node, slotKey, code, mode) {
    if (!node || typeof code !== "string") return;
    const slot = getSlotByKey(node, slotKey);
    if (!slot || !slot.searchInput) return;
    const current = String(slot.searchInput.value || "");
    slot.searchInput.value = (mode === "replace" || !current) ? code : `${current}\n${code}`;
    slot.searchInput.dispatchEvent(new Event("input", { bubbles: true }));
}

// ---------------------------------------------------------------------------
// lora_code_update listener  (attached once, synchronously)
// ---------------------------------------------------------------------------

function attachLoraCodeListener() {
    if (loraCodeListenerAttached) return;
    loraCodeListenerAttached = true;

    api.addEventListener("lora_code_update", (event) => {
        const detail = event?.detail;
        if (!detail) return;

        const nodeId = Number(detail.id);
        if (!Number.isFinite(nodeId) || nodeId < 0) return;

        const loraCode = typeof detail.lora_code === "string" ? detail.lora_code : "";
        const mode = typeof detail.mode === "string" ? detail.mode : "append";

        // Find our node by integer ID
        const nodes = app.graph?._nodes;
        if (!Array.isArray(nodes)) return;
        const targetNode = nodes.find((n) => Number(n.id) === nodeId && n.type === NODE_CLASS);
        if (!targetNode) return;

        // Route to the currently-selected active slot
        const activeSlot = targetNode.properties?.pmActiveSlot ?? "model_a";
        applyIncomingLoraCode(targetNode, activeSlot, loraCode, mode);
    });
}

// ---------------------------------------------------------------------------
// Active-slot selection (click a section to target LM sends)
// ---------------------------------------------------------------------------

function setActiveSlot(node, slotKey) {
    if (!node) return;
    if (!node.properties) node.properties = {};
    node.properties.pmActiveSlot = slotKey;
    updateActiveColumnHighlight(node);
}

function updateActiveColumnHighlight(node) {
    if (!Array.isArray(node.__pmMultiLmSlots)) return;
    const activeSlot = node.properties?.pmActiveSlot ?? "model_a";
    for (const slot of node.__pmMultiLmSlots) {
        if (slot?.col) slot.col.classList.toggle("active-slot", slot.key === activeSlot);
    }
}

// ---------------------------------------------------------------------------
// Embedded loras widget via fakeNode pattern
// ---------------------------------------------------------------------------

function createEmbeddedLorasWidget(node, lm, widgetName, initialValue, onChange) {
    const host = document.createElement("div");
    host.className = "pm-multi-lm-list-host";

    const fakeNode = {
        addDOMWidget(name, type, container, options) {
            return {
                name,
                type,
                callback: null,
                element: container,
                get value() {
                    return typeof options?.getValue === "function" ? options.getValue() : [];
                },
                set value(v) {
                    if (typeof options?.setValue === "function") options.setValue(v);
                },
            };
        },
        setDirtyCanvas(fg, bg) { node?.setDirtyCanvas?.(fg, bg); },
    };

    const out = lm.addLorasWidget(fakeNode, widgetName, {}, onChange);
    if (out?.widget?.element) host.appendChild(out.widget.element);
    if (out?.widget && Array.isArray(initialValue) && initialValue.length > 0) {
        out.widget.value = initialValue;
    }

    return { host, widget: out?.widget ?? null };
}

// ---------------------------------------------------------------------------
// Height management — mirrors LM's updateWidgetHeight approach:
// set CSS vars on the root element so ComfyUI's layout system resizes the node,
// never calling node.setSize() (which would prevent user resize).
// ---------------------------------------------------------------------------

function computeContentHeight(node) {
    let maxListH = LM_EMPTY_H;
    if (Array.isArray(node.__pmMultiLmSlots)) {
        for (const slot of node.__pmMultiLmSlots) {
            const count = Array.isArray(slot.lorasWidget?.value) ? slot.lorasWidget.value.length : 0;
            const h = count === 0
                ? LM_EMPTY_H
                : LM_CONTAINER_PAD + LM_HEADER_H + Math.min(count, 12) * LM_LORA_ENTRY_H;
            if (h > maxListH) maxListH = h;
        }
    }
    return COL_CHROME_H + maxListH + 20;
}

function notifyHeightChange(node) {
    const root = node?.__pmMultiLmRoot;
    if (!root) return;
    const h = computeContentHeight(node);
    root.style.setProperty('--comfy-widget-min-height', `${h}px`);
    root.style.setProperty('--comfy-widget-height', `${h}px`);
    setTimeout(() => { node?.setDirtyCanvas?.(true, true); }, 10);
}

// ---------------------------------------------------------------------------
// Per-slot column builder
// ---------------------------------------------------------------------------

function buildSlotColumn(node, lm, slotDef) {
    const stateWidget = ensureStateWidget(node, slotDef.state);
    hideWidget(stateWidget);

    const col = document.createElement("div");
    col.className = "pm-multi-lm-col";
    col.title = `Click to make ${slotDef.label} the LoRA Manager target`;
    col.addEventListener("pointerdown", () => setActiveSlot(node, slotDef.key));

    const title = document.createElement("div");
    title.className = "pm-multi-lm-title";
    title.textContent = slotDef.label;

    const searchInput = document.createElement("textarea");
    searchInput.className = "pm-multi-lm-search";
    searchInput.placeholder = "Search LoRAs to add\u2026";
    searchInput.rows = 1;
    searchInput.addEventListener("focus", () => setActiveSlot(node, slotDef.key));

    // Attach LM autocomplete to the search textarea
    new lm.AutoComplete(searchInput, "loras", { showPreview: false });

    const initialState = parseLorasState(stateWidget?.value ?? "[]");

    let isSyncing = false;

    const embedded = createEmbeddedLorasWidget(
        node, lm, `${slotDef.key}_loras`, initialState,
        (value) => {
            if (isSyncing) return;
            isSyncing = true;
            try {
                const safe = Array.isArray(value) ? value : [];
                writeLorasState(stateWidget, safe);
                const nextText = lm.applyLoraValuesToText(searchInput.value || "", safe);
                if (searchInput.value !== nextText) searchInput.value = nextText;
            } finally {
                isSyncing = false;
                notifyHeightChange(node);
            }
        },
    );

    // When user types or autocomplete fires, merge into the loras list
    searchInput.addEventListener("input", () => {
        if (isSyncing) return;
        isSyncing = true;
        try {
            const existing = Array.isArray(embedded.widget?.value) ? embedded.widget.value : [];
            const merged = lm.mergeLoras(searchInput.value || "", existing);
            if (embedded.widget) embedded.widget.value = merged;
            writeLorasState(stateWidget, merged);
        } finally {
            isSyncing = false;
            notifyHeightChange(node);
        }
    });

    col.appendChild(title);
    col.appendChild(searchInput);
    col.appendChild(embedded.host);
    embedded.host.addEventListener("pointerdown", () => setActiveSlot(node, slotDef.key));

    return { key: slotDef.key, stateWidget, searchInput, lorasWidget: embedded.widget, col };
}

// ---------------------------------------------------------------------------
// Restore widget state from persisted JSON
// ---------------------------------------------------------------------------

function refreshFromStoredValues(node) {
    if (!node?.__pmMultiLmSlots || !node.__pmMultiLmBridge) return;
    for (const slot of node.__pmMultiLmSlots) {
        const state = parseLorasState(slot.stateWidget?.value ?? "[]");

        // Restore from persisted JSON directly. Do NOT call mergeLoras here:
        // mergeLoras requires syntax in text input, and would drop entries
        // during tab-switch/workflow-load when textarea is initially empty.
        if (slot.lorasWidget) slot.lorasWidget.value = state;
        writeLorasState(slot.stateWidget, state);

        // Keep syntax textarea consistent with restored list.
        const nextText = node.__pmMultiLmBridge.applyLoraValuesToText(slot.searchInput.value || "", state);
        if (slot.searchInput.value !== nextText) slot.searchInput.value = nextText;
    }
    notifyHeightChange(node);
}

// ---------------------------------------------------------------------------
// Main async UI setup
// ---------------------------------------------------------------------------

async function setupNodeUi(node) {
    if (!node || node.__pmMultiLmReady) return;

    ensureStyles();
    const lm = await loadLmBridge();
    if (!lm || node.__pmMultiLmReady) return;

    hideStateWidgets(node);

    const root = document.createElement("div");
    root.className = "pm-multi-lm-root";

    // 4-column LoRA grid
    const grid = document.createElement("div");
    grid.className = "pm-multi-lm-grid";
    root.appendChild(grid);

    const slots = SLOT_DEFS.map((slotDef) => {
        const slot = buildSlotColumn(node, lm, slotDef);
        grid.appendChild(slot.col);
        return slot;
    });

    const uiWidget = node.addDOMWidget("multi_lora_ui", "div", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => computeContentHeight(node),
        getHeight: () => "100%",
    });

    node.__pmMultiLmBridge = lm;
    node.__pmMultiLmSlots = slots;
    node.__pmMultiLmRoot = root;
    node.__pmMultiLmReady = true;
    node.__pmMultiLmRefresh = () => {
        refreshFromStoredValues(node);
        updateActiveColumnHighlight(node);
    };

    // comfyClass already set synchronously in onNodeCreated — no registry tricks needed
    // Restore any lora data that onConfigure may have written before async completed
    refreshFromStoredValues(node);
    hideStateWidgets(node);
    updateActiveColumnHighlight(node);
    notifyHeightChange(node);
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "PromptManager.MultiLoraStackerLM",

    async beforeRegisterNodeDef(nodeType) {
        if (nodeType.comfyClass !== NODE_CLASS) return;

        // Attach the send-to-node listener once, immediately (synchronous)
        attachLoraCodeListener();

        // ── onNodeCreated ──────────────────────────────────────────────────
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            this.serialize_widgets = true;

            // Set comfyClass SYNCHRONOUSLY so LM WorkflowRegistry includes us
            // on the next lora_registry_refresh sweep
            this.comfyClass = LM_PROVIDER_CLASS;

            if (!this.properties) this.properties = {};
            if (!this.properties.pmActiveSlot) this.properties.pmActiveSlot = "model_a";

            void setupNodeUi(this);
            return result;
        };

        // ── onConfigure  (workflow load / tab switch) ──────────────────────
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (config) {
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;

            // Re-assert comfyClass in case ComfyUI reset it during configure
            this.comfyClass = LM_PROVIDER_CLASS;

            hideStateWidgets(this);

            if (typeof this.__pmMultiLmRefresh === "function") {
                this.__pmMultiLmRefresh();
            }

            return result;
        };

        const onModeChange = nodeType.prototype.onModeChange;
        nodeType.prototype.onModeChange = function () {
            const result = onModeChange ? onModeChange.apply(this, arguments) : undefined;
            hideStateWidgets(this);
            if (typeof this.__pmMultiLmRefresh === "function") {
                this.__pmMultiLmRefresh();
            }
            return result;
        };

        // ── onRemoved ──────────────────────────────────────────────────────
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.__pmMultiLmSlots = null;
            this.__pmMultiLmRoot = null;
            this.__pmMultiLmBridge = null;
            this.__pmMultiLmReady = false;
            this.__pmMultiLmRefresh = null;
            return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };
    },
});
