import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "MultiLoraStackerLM";
const LM_PROVIDER_CLASS = "Lora Stacker (LoraManager)";
const STYLE_ID = "pm-multi-lm-style";
const MIN_NODE_WIDTH = 920;
const MIN_NODE_HEIGHT = 540;
const MAX_NODE_HEIGHT = 860;
const ROOT_PADDING = 22;

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
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.pm-multi-lm-root {
    width: 100%;
    box-sizing: border-box;
    padding: 6px 6px 2px 6px;
    overflow-y: auto;
    overflow-x: hidden;
}
.pm-multi-lm-slot-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
    padding: 4px 2px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    flex-shrink: 0;
}
.pm-multi-lm-slot-label {
    font-size: 11px;
    color: rgba(180,188,200,0.75);
    white-space: nowrap;
    flex-shrink: 0;
}
.pm-multi-lm-slot-btn {
    padding: 3px 10px;
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 4px;
    background: rgba(30,35,45,0.8);
    color: #c0c8d8;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
}
.pm-multi-lm-slot-btn:hover {
    background: rgba(60,70,90,0.9);
    border-color: rgba(120,150,200,0.5);
}
.pm-multi-lm-slot-btn.active {
    background: rgba(50,100,200,0.45);
    border-color: rgba(100,160,255,0.7);
    color: #e8efff;
}
.pm-multi-lm-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(200px, 1fr));
    gap: 8px;
}
.pm-multi-lm-col {
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px;
    background: rgba(18, 22, 28, 0.9);
    padding: 8px;
    min-width: 0;
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
}
.pm-multi-lm-col.active-slot {
    border-color: rgba(100,160,255,0.55);
    box-shadow: 0 0 0 1px rgba(100,160,255,0.2);
}
.pm-multi-lm-title {
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.02em;
    color: #d7dee8;
    margin-bottom: 6px;
    flex-shrink: 0;
}
.pm-multi-lm-search {
    width: 100%;
    min-height: 38px;
    resize: vertical;
    box-sizing: border-box;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 6px;
    padding: 6px 8px;
    background: rgba(8, 10, 14, 0.95);
    color: #dde4ee;
    margin-bottom: 6px;
    font-size: 12px;
    flex-shrink: 0;
}
.pm-multi-lm-search::placeholder {
    color: rgba(180,188,200,0.5);
}
.pm-multi-lm-list-host {
    flex: 1;
    min-height: 160px;
    overflow: hidden;
}
.pm-multi-lm-list-host .lm-loras-container {
    max-height: 420px;
    overflow: auto;
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
    widget.computeSize = () => [0, -4];
    widget.type = "converted-widget";
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
// Slot selector header row
// ---------------------------------------------------------------------------

function buildSlotSelector(node) {
    const row = document.createElement("div");
    row.className = "pm-multi-lm-slot-row";

    const label = document.createElement("span");
    label.className = "pm-multi-lm-slot-label";
    label.textContent = "LM sends to:";
    row.appendChild(label);

    const buttons = {};
    const activeSlot = node.properties?.pmActiveSlot ?? "model_a";

    for (const slotDef of SLOT_DEFS) {
        const btn = document.createElement("button");
        btn.className = "pm-multi-lm-slot-btn" + (slotDef.key === activeSlot ? " active" : "");
        btn.textContent = slotDef.short;
        btn.title = `LoRA Manager sends LoRAs to ${slotDef.label}`;

        btn.addEventListener("click", () => {
            if (!node.properties) node.properties = {};
            node.properties.pmActiveSlot = slotDef.key;
            for (const [k, b] of Object.entries(buttons)) {
                b.classList.toggle("active", k === slotDef.key);
            }
            updateActiveColumnHighlight(node);
        });

        buttons[slotDef.key] = btn;
        row.appendChild(btn);
    }

    return { row, buttons };
}

function updateActiveColumnHighlight(node) {
    if (!Array.isArray(node.__pmMultiLmSlots)) return;
    const activeSlot = node.properties?.pmActiveSlot ?? "model_a";
    for (const slot of node.__pmMultiLmSlots) {
        if (slot?.col) slot.col.classList.toggle("active-slot", slot.key === activeSlot);
    }
}

function syncSelectorButtons(node) {
    if (!node.__pmSlotButtons) return;
    const activeSlot = node.properties?.pmActiveSlot ?? "model_a";
    for (const [k, b] of Object.entries(node.__pmSlotButtons)) {
        b.classList.toggle("active", k === activeSlot);
    }
    updateActiveColumnHighlight(node);
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
// Node height
// ---------------------------------------------------------------------------

function recalcNodeHeight(node, rootEl) {
    if (!node || !rootEl) return;
    const w = Math.max(MIN_NODE_WIDTH, Math.round(node.size?.[0] ?? MIN_NODE_WIDTH));
    const rawH = Math.ceil(rootEl.scrollHeight + ROOT_PADDING);
    const h = Math.max(MIN_NODE_HEIGHT, Math.min(MAX_NODE_HEIGHT, rawH));
    if (Math.abs((node.size?.[0] ?? 0) - w) > 1 || Math.abs((node.size?.[1] ?? 0) - h) > 1) {
        node.setSize([w, h]);
    }
    node.setDirtyCanvas(true, true);
}

// ---------------------------------------------------------------------------
// Per-slot column builder
// ---------------------------------------------------------------------------

function buildSlotColumn(node, lm, slotDef) {
    const stateWidget = ensureStateWidget(node, slotDef.state);
    hideWidget(stateWidget);

    const col = document.createElement("div");
    col.className = "pm-multi-lm-col";

    const title = document.createElement("div");
    title.className = "pm-multi-lm-title";
    title.textContent = slotDef.label;

    const searchInput = document.createElement("textarea");
    searchInput.className = "pm-multi-lm-search";
    searchInput.placeholder = "Search LoRAs to add\u2026";
    searchInput.rows = 1;

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
                node.setDirtyCanvas?.(true, true);
                if (node.__pmMultiLmRoot) requestAnimationFrame(() => recalcNodeHeight(node, node.__pmMultiLmRoot));
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
            node.setDirtyCanvas?.(true, true);
            if (node.__pmMultiLmRoot) requestAnimationFrame(() => recalcNodeHeight(node, node.__pmMultiLmRoot));
        }
    });

    col.appendChild(title);
    col.appendChild(searchInput);
    col.appendChild(embedded.host);

    return { key: slotDef.key, stateWidget, searchInput, lorasWidget: embedded.widget, col };
}

// ---------------------------------------------------------------------------
// Restore widget state from persisted JSON
// ---------------------------------------------------------------------------

function refreshFromStoredValues(node) {
    if (!node?.__pmMultiLmSlots || !node.__pmMultiLmBridge) return;
    for (const slot of node.__pmMultiLmSlots) {
        const state = parseLorasState(slot.stateWidget?.value ?? "[]");
        if (state.length === 0) continue;
        const merged = node.__pmMultiLmBridge.mergeLoras(slot.searchInput.value || "", state);
        if (slot.lorasWidget) slot.lorasWidget.value = merged;
        writeLorasState(slot.stateWidget, merged);
    }
    if (node.__pmMultiLmRoot) recalcNodeHeight(node, node.__pmMultiLmRoot);
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

    // Slot selector row (LM sends to: A B C D)
    const { row: selectorRow, buttons: slotButtons } = buildSlotSelector(node);
    root.appendChild(selectorRow);

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
    });

    uiWidget.computeSize = (width) => {
        const w = Math.max(MIN_NODE_WIDTH, Math.round(width || MIN_NODE_WIDTH));
        const rawH = Math.ceil(root.scrollHeight + ROOT_PADDING);
        return [w, Math.max(MIN_NODE_HEIGHT, Math.min(MAX_NODE_HEIGHT, rawH))];
    };

    node.__pmMultiLmBridge = lm;
    node.__pmMultiLmSlots = slots;
    node.__pmMultiLmRoot = root;
    node.__pmSlotButtons = slotButtons;
    node.__pmMultiLmReady = true;
    node.__pmMultiLmRefresh = () => {
        refreshFromStoredValues(node);
        syncSelectorButtons(node);
    };

    // comfyClass already set synchronously in onNodeCreated — no registry tricks needed
    // Restore any lora data that onConfigure may have written before async completed
    refreshFromStoredValues(node);
    hideStateWidgets(node);
    updateActiveColumnHighlight(node);
    requestAnimationFrame(() => recalcNodeHeight(node, root));
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

        // ── onRemoved ──────────────────────────────────────────────────────
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.__pmMultiLmSlots = null;
            this.__pmMultiLmRoot = null;
            this.__pmMultiLmBridge = null;
            this.__pmMultiLmReady = false;
            this.__pmMultiLmRefresh = null;
            this.__pmSlotButtons = null;
            return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };
    },
});
