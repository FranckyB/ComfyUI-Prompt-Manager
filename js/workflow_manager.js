/**
 * Workflow Manager JS Extension
 *
 * Extends Prompt Manager Advanced's UI with an additional "Workflow Config"
 * section that lets users store and view model family / model A / model B /
 * VAE / CLIP / sampler / resolution alongside their prompt.
 *
 * State persistence strategy (same pattern as workflow_extractor.js):
 *   - Workflow config is stored in node.properties.wm_workflow_config
 *   - syncHidden() mirrors it to a hidden STRING widget AND node.properties
 *   - onConfigure() reads from node.properties first (reliable across tab switches)
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ─── Colour palette (matches workflow_extractor) ──────────────────────────────
const C = {
    bgDark:    "rgba(40, 44, 52, 0.6)",
    bgCard:    "rgba(40, 44, 52, 0.8)",
    bgInput:   "#1a1a1a",
    accent:    "rgba(66, 153, 225, 0.9)",
    accentDim: "rgba(66, 153, 225, 0.7)",
    text:      "#ccc",
    textMuted: "#999",
    border:    "rgba(226, 232, 240, 0.15)",
    success:   "#6c6",
    warning:   "#da3",
    error:     "rgba(220, 53, 69, 0.9)",
};

function makeEl(tag, style, text) {
    const el = document.createElement(tag);
    if (style) Object.assign(el.style, style);
    if (text !== undefined) el.textContent = text;
    return el;
}

function makeBtn(label, onclick, extraStyle) {
    const b = makeEl("button", {
        padding: "4px 10px", border: `1px solid ${C.border}`, borderRadius: "6px",
        background: C.bgInput, color: C.text, cursor: "pointer",
        fontSize: "11px", fontWeight: "600",
        ...extraStyle,
    }, label);
    b.onmouseenter = () => b.style.background = "#2a2a2a";
    b.onmouseleave = () => b.style.background = extraStyle?.background || C.bgInput;
    b.onclick = onclick;
    return b;
}

const ROW_STYLE = { display: "flex", alignItems: "center", padding: "2px 0", fontSize: "11px", gap: "4px" };
const LABEL_W = "36%";
const INPUT_W = "60%";
const INPUT_STYLE = {
    background: C.bgInput, color: C.text, border: `1px solid ${C.border}`,
    borderRadius: "5px", padding: "3px 6px", fontSize: "11px",
    width: INPUT_W, boxSizing: "border-box",
};

function makeRow(labelText, children) {
    const row = makeEl("div", { ...ROW_STYLE });
    row.appendChild(makeEl("span", { color: C.textMuted, width: LABEL_W, flexShrink: "0" }, labelText));
    if (Array.isArray(children)) children.forEach(c => row.appendChild(c));
    else row.appendChild(children);
    return row;
}

function makeSelect(options, value, onchange) {
    const sel = document.createElement("select");
    Object.assign(sel.style, { ...INPUT_STYLE });
    for (const [val, lbl] of options) {
        const o = document.createElement("option");
        o.value = val; o.textContent = lbl;
        sel.appendChild(o);
    }
    sel.value = value || "";
    sel.onchange = () => onchange(sel.value);
    return sel;
}

function makeModelSelect(value, family, lazyFetchFn, onchange) {
    const sel = document.createElement("select");
    Object.assign(sel.style, { ...INPUT_STYLE, flexGrow: "1" });
    if (value) {
        const o = document.createElement("option"); o.value = value;
        o.textContent = _cleanName(value); sel.appendChild(o); sel.value = value;
    }
    let _loaded = false;
    sel.onfocus = async () => {
        if (_loaded) return; _loaded = true;
        const models = await lazyFetchFn(family);
        const cur = sel.value;
        sel.innerHTML = "";
        const og0 = document.createElement("optgroup"); og0.label = "— select model —";
        const blank = document.createElement("option"); blank.value = ""; blank.textContent = "(none)";
        og0.appendChild(blank); sel.appendChild(og0);
        // Group by folder
        const groups = new Map(); const ungrouped = [];
        for (const m of models) {
            const norm = m.replace(/\\/g, "/");
            const slash = norm.lastIndexOf("/");
            if (slash < 0) { ungrouped.push(m); continue; }
            const g = norm.substring(0, slash).replace(/\//g, " - ");
            if (!groups.has(g)) groups.set(g, []);
            groups.get(g).push(m);
        }
        ungrouped.forEach(m => { const o = document.createElement("option"); o.value = m; o.textContent = _cleanName(m); sel.appendChild(o); });
        for (const [g, items] of groups) {
            const og = document.createElement("optgroup"); og.label = g;
            items.forEach(m => { const o = document.createElement("option"); o.value = m; o.textContent = _cleanName(m); og.appendChild(o); });
            sel.appendChild(og);
        }
        sel.value = cur || "";
    };
    sel.onchange = () => onchange(sel.value);
    return { sel, reset: (v, fam) => { _loaded = false; sel.innerHTML = ""; if (v) { const o = document.createElement("option"); o.value = v; o.textContent = _cleanName(v); sel.appendChild(o); sel.value = v; } family = fam; } };
}

function _cleanName(p) {
    if (!p) return "";
    let n = p.replace(/\\/g, "/"); n = n.substring(n.lastIndexOf("/") + 1);
    const dot = n.lastIndexOf("."); if (dot > 0) n = n.substring(0, dot);
    return n;
}

// ─── WorkflowConfig panel builder ────────────────────────────────────────────

async function loadFamilies() {
    try {
        const r = await fetch("/workflow-manager/families");
        const d = await r.json();
        return d.families || {};
    } catch { return {}; }
}

async function fetchModels(family) {
    try {
        const url = family
            ? `/workflow-manager/list-models?family=${encodeURIComponent(family)}`
            : "/workflow-manager/list-models";
        const r = await fetch(url); const d = await r.json();
        return d.models || [];
    } catch { return []; }
}

async function fetchVaes(family) {
    try {
        const r = await fetch(`/workflow-manager/list-vaes${family ? "?family=" + encodeURIComponent(family) : ""}`);
        const d = await r.json(); return d.vaes || [];
    } catch { return []; }
}

async function fetchClips(family) {
    try {
        const r = await fetch(`/workflow-manager/list-clips${family ? "?family=" + encodeURIComponent(family) : ""}`);
        const d = await r.json(); return d.clips || [];
    } catch { return []; }
}

/**
 * Build the workflow config DOM panel and attach it to the node.
 * Returns a { getConfig, setConfig } interface.
 */
function buildWorkflowConfigPanel(node, container) {
    let _config = {};
    let _family = "";
    let _families = {};

    const wrap = makeEl("div", {
        borderRadius: "6px", background: C.bgCard,
        border: `1px solid ${C.border}`, padding: "6px 8px",
        fontSize: "11px", marginTop: "4px",
    });

    // ── Header ────────────────────────────────────────────────────────────
    const header = makeEl("div", {
        fontWeight: "700", color: C.accent, fontSize: "11px",
        marginBottom: "6px", letterSpacing: "0.5px",
    }, "WORKFLOW CONFIG");
    wrap.appendChild(header);

    // ── Family selector ───────────────────────────────────────────────────
    const familyRow = makeEl("div", { ...ROW_STYLE });
    familyRow.appendChild(makeEl("span", { color: C.accent, width: LABEL_W, flexShrink: "0", fontWeight: "bold" }, "Family"));
    const familySel = document.createElement("select");
    Object.assign(familySel.style, { ...INPUT_STYLE, color: C.accent, fontWeight: "bold" });
    let _familiesLoaded = false;
    familySel.onfocus = async () => {
        if (_familiesLoaded) return; _familiesLoaded = true;
        _families = await loadFamilies();
        const cur = familySel.value;
        familySel.innerHTML = "";
        const blank = document.createElement("option"); blank.value = ""; blank.textContent = "— auto-detect —";
        familySel.appendChild(blank);
        for (const [k, lbl] of Object.entries(_families)) {
            const o = document.createElement("option"); o.value = k; o.textContent = lbl;
            familySel.appendChild(o);
        }
        familySel.value = cur;
    };
    familySel.onchange = async () => {
        _family = familySel.value;
        _config.family = _family;
        // Reload model/vae/clip dropdowns for the new family
        modelACtrl.reset(_config.model_a, _family);
        modelBCtrl.reset(_config.model_b, _family);
        vaeCtrl.reset(_config.vae, _family);
        clipCtrl.reset(_config.clip, _family);
        syncConfig();
    };
    familyRow.appendChild(familySel);
    wrap.appendChild(familyRow);

    // ── Model A ───────────────────────────────────────────────────────────
    const modelACtrl = _buildModelRow(wrap, "Model A",
        () => _config.model_a || "",
        () => _family,
        fetchModels,
        v => { _config.model_a = v; syncConfig(); }
    );

    // ── Model B ───────────────────────────────────────────────────────────
    const modelBCtrl = _buildModelRow(wrap, "Model B",
        () => _config.model_b || "",
        () => _family,
        fetchModels,
        v => { _config.model_b = v; syncConfig(); }
    );

    // ── VAE ───────────────────────────────────────────────────────────────
    const vaeCtrl = _buildModelRow(wrap, "VAE",
        () => _config.vae || "",
        () => _family,
        fetchVaes,
        v => { _config.vae = v; syncConfig(); }
    );

    // ── CLIP ──────────────────────────────────────────────────────────────
    const clipCtrl = _buildModelRow(wrap, "CLIP",
        () => (Array.isArray(_config.clip) ? _config.clip[0] : _config.clip) || "",
        () => _family,
        fetchClips,
        v => { _config.clip = v ? [v] : []; syncConfig(); }
    );

    // ── Sampler params (collapsed) ────────────────────────────────────────
    const sampToggle = makeEl("div", {
        cursor: "pointer", color: C.textMuted, fontSize: "10px",
        marginTop: "4px", userSelect: "none",
    }, "▶ Sampler overrides");
    const sampPanel = makeEl("div", { display: "none" });
    sampToggle.onclick = () => {
        const open = sampPanel.style.display !== "none";
        sampPanel.style.display = open ? "none" : "block";
        sampToggle.textContent = (open ? "▶" : "▼") + " Sampler overrides";
    };
    wrap.appendChild(sampToggle);

    const _sampRows = {};
    const _makeSampRow = (key, label, type, def, attrs) => {
        const r = makeEl("div", { ...ROW_STYLE });
        r.appendChild(makeEl("span", { color: C.textMuted, width: LABEL_W, flexShrink: "0" }, label));
        let inp;
        if (type === "select") {
            inp = document.createElement("select");
            Object.assign(inp.style, { ...INPUT_STYLE });
            for (const o of (attrs?.options || [])) {
                const opt = document.createElement("option"); opt.value = o; opt.textContent = o;
                inp.appendChild(opt);
            }
        } else {
            inp = document.createElement("input");
            inp.type = type; inp.value = def;
            if (attrs?.min !== undefined) inp.min = attrs.min;
            if (attrs?.max !== undefined) inp.max = attrs.max;
            if (attrs?.step)             inp.step = attrs.step;
            Object.assign(inp.style, { ...INPUT_STYLE });
        }
        inp.onchange = inp.oninput = () => {
            if (!_config.sampler) _config.sampler = {};
            _config.sampler[key] = type === "number" ? parseFloat(inp.value) : inp.value;
            syncConfig();
        };
        r.appendChild(inp);
        sampPanel.appendChild(r);
        return inp;
    };
    _sampRows.steps     = _makeSampRow("steps",        "Steps",     "number", 20,       { min: 1, max: 200, step: 1 });
    _sampRows.cfg       = _makeSampRow("cfg",          "CFG",      "number", 7.0,      { min: 0, max: 30,  step: 0.5 });
    _sampRows.seed      = _makeSampRow("seed",         "Seed",     "number", 0,        { min: 0, step: 1 });
    _sampRows.sampler   = _makeSampRow("sampler_name", "Sampler",  "select", "euler",  { options: ["euler","dpm_2","dpm_2_ancestral","heun","dpm_fast","dpm_adaptive","lms","dpmpp_2s_ancestral","dpmpp_sde","dpmpp_2m","dpmpp_2m_sde","ddim","uni_pc"] });
    _sampRows.scheduler = _makeSampRow("scheduler",    "Scheduler","select", "normal", { options: ["normal","karras","exponential","sgm_uniform","simple","ddim_uniform","beta"] });
    _sampRows.denoise   = _makeSampRow("denoise",      "Denoise",  "number", 1.0,      { min: 0, max: 1, step: 0.05 });
    wrap.appendChild(sampPanel);

    // ── Resolution ────────────────────────────────────────────────────────
    const resToggle = makeEl("div", {
        cursor: "pointer", color: C.textMuted, fontSize: "10px",
        marginTop: "4px", userSelect: "none",
    }, "▶ Resolution overrides");
    const resPanel = makeEl("div", { display: "none" });
    resToggle.onclick = () => {
        const open = resPanel.style.display !== "none";
        resPanel.style.display = open ? "none" : "block";
        resToggle.textContent = (open ? "▶" : "▼") + " Resolution overrides";
    };
    wrap.appendChild(resToggle);

    const _resRows = {};
    const _makeResRow = (key, label, def, max) => {
        const r = makeEl("div", { ...ROW_STYLE });
        r.appendChild(makeEl("span", { color: C.textMuted, width: LABEL_W, flexShrink: "0" }, label));
        const inp = document.createElement("input");
        inp.type = "number"; inp.value = def; inp.min = 1; inp.max = max; inp.step = 8;
        Object.assign(inp.style, { ...INPUT_STYLE });
        inp.onchange = inp.oninput = () => {
            if (!_config.resolution) _config.resolution = {};
            _config.resolution[key] = parseInt(inp.value) || def;
            syncConfig();
        };
        r.appendChild(inp);
        resPanel.appendChild(r);
        return inp;
    };
    _resRows.width  = _makeResRow("width",  "Width",  512,  8192);
    _resRows.height = _makeResRow("height", "Height", 512,  8192);
    _resRows.batch  = _makeResRow("batch_size",  "Batch",  1,    128);
    _resRows.length = _makeResRow("length", "Length", 81,   1000);
    wrap.appendChild(resPanel);

    // ── Save button ───────────────────────────────────────────────────────
    const btnRow = makeEl("div", { display: "flex", gap: "6px", marginTop: "6px" });
    btnRow.appendChild(makeBtn("💾 Save Config", async () => {
        // Trigger save to the prompt store from the current WM node state
        const catW = node.widgets?.find(w => w.name === "category");
        const namW = node.widgets?.find(w => w.name === "name");
        const txtW = node.widgets?.find(w => w.name === "text");
        if (!catW?.value || !namW?.value) {
            alert("Select a category and name before saving workflow config.");
            return;
        }
        try {
            const resp = await fetch("/workflow-manager/save-prompt", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    category:        catW.value,
                    name:            namW.value,
                    text:            txtW?.value || "",
                    workflow_config: _config,
                }),
            });
            const d = await resp.json();
            if (d.success) {
                statusEl.textContent = "✓ Saved";
                statusEl.style.color = C.success;
                setTimeout(() => { statusEl.textContent = ""; }, 2000);
            } else {
                statusEl.textContent = `❌ ${d.error}`;
                statusEl.style.color = C.error;
            }
        } catch (e) {
            statusEl.textContent = `❌ ${e.message}`;
            statusEl.style.color = C.error;
        }
    }));
    btnRow.appendChild(makeBtn("🗑 Clear Config", () => {
        _config = {};
        _family = "";
        familySel.value = "";
        modelACtrl.reset("", "");
        modelBCtrl.reset("", "");
        vaeCtrl.reset("", "");
        clipCtrl.reset("", "");
        syncConfig();
    }, { background: "rgba(220,53,69,0.2)" }));
    wrap.appendChild(btnRow);

    const statusEl = makeEl("div", {
        fontSize: "10px", color: C.textMuted, marginTop: "2px", minHeight: "14px",
    }, "");
    wrap.appendChild(statusEl);

    container.appendChild(wrap);

    // ── Sync helpers ──────────────────────────────────────────────────────
    function syncConfig() {
        // Write to hidden widget
        const w = node.widgets?.find(x => x.name === "wm_workflow_config");
        if (w) w.value = JSON.stringify(_config);
        // Also to properties
        node.properties = node.properties || {};
        node.properties.wm_workflow_config = JSON.stringify(_config);
    }

    function setConfig(cfg) {
        if (!cfg || typeof cfg !== "object") return;
        _config = { ...cfg };
        _family = cfg.family || "";
        familySel.value = _family;
        modelACtrl.reset(cfg.model_a || "", _family);
        modelBCtrl.reset(cfg.model_b || "", _family);
        vaeCtrl.reset(cfg.vae || "", _family);
        const clipVal = Array.isArray(cfg.clip) ? cfg.clip[0] : (cfg.clip || "");
        clipCtrl.reset(clipVal, _family);
        // Restore sampler
        if (cfg.sampler) {
            if (_sampRows.steps && cfg.sampler.steps != null) _sampRows.steps.value = cfg.sampler.steps;
            if (_sampRows.cfg   && cfg.sampler.cfg   != null) _sampRows.cfg.value   = cfg.sampler.cfg;
            if (_sampRows.seed  && cfg.sampler.seed  != null) _sampRows.seed.value  = cfg.sampler.seed;
            if (_sampRows.sampler   && cfg.sampler.sampler_name) _sampRows.sampler.value   = cfg.sampler.sampler_name;
            if (_sampRows.scheduler && cfg.sampler.scheduler)    _sampRows.scheduler.value = cfg.sampler.scheduler;
            if (_sampRows.denoise   && cfg.sampler.denoise != null) _sampRows.denoise.value = cfg.sampler.denoise;
        }
        // Restore resolution
        if (cfg.resolution) {
            if (_resRows.width  && cfg.resolution.width)      _resRows.width.value  = cfg.resolution.width;
            if (_resRows.height && cfg.resolution.height)     _resRows.height.value = cfg.resolution.height;
            if (_resRows.batch  && cfg.resolution.batch_size) _resRows.batch.value  = cfg.resolution.batch_size;
            if (_resRows.length && cfg.resolution.length)     _resRows.length.value = cfg.resolution.length;
        }
    }

    function getConfig() { return _config; }

    return { getConfig, setConfig, syncConfig };
}

// ── Generic lazy model select row ────────────────────────────────────────────
function _buildModelRow(container, label, getValue, getFamily, fetchFn, onChange) {
    const row = makeEl("div", { ...ROW_STYLE });
    row.appendChild(makeEl("span", { color: C.textMuted, width: LABEL_W, flexShrink: "0" }, label));
    const sel = document.createElement("select");
    Object.assign(sel.style, { ...INPUT_STYLE });
    const initVal = getValue();
    if (initVal) {
        const o = document.createElement("option"); o.value = initVal;
        o.textContent = _cleanName(initVal); sel.appendChild(o); sel.value = initVal;
    } else {
        const o = document.createElement("option"); o.value = ""; o.textContent = "(none)";
        sel.appendChild(o);
    }
    let _loaded = false;
    let _family = getFamily();
    sel.onfocus = async () => {
        if (_loaded) return; _loaded = true;
        const items = await fetchFn(_family);
        const cur   = sel.value;
        sel.innerHTML = "";
        const blank = document.createElement("option"); blank.value = ""; blank.textContent = "(none)";
        sel.appendChild(blank);
        // Group by folder
        const groups = new Map(); const ungrouped = [];
        for (const m of items) {
            const norm = m.replace(/\\/g, "/");
            const slash = norm.lastIndexOf("/");
            if (slash < 0) { ungrouped.push(m); continue; }
            const g = norm.substring(0, slash).replace(/\//g, " - ");
            if (!groups.has(g)) groups.set(g, []);
            groups.get(g).push(m);
        }
        ungrouped.forEach(m => { const o = document.createElement("option"); o.value = m; o.textContent = _cleanName(m); sel.appendChild(o); });
        for (const [g, items2] of groups) {
            const og = document.createElement("optgroup"); og.label = g;
            items2.forEach(m => { const o = document.createElement("option"); o.value = m; o.textContent = _cleanName(m); og.appendChild(o); });
            sel.appendChild(og);
        }
        sel.value = cur || "";
    };
    sel.onchange = () => onChange(sel.value);
    row.appendChild(sel);
    container.appendChild(row);

    return {
        reset: (v, fam) => {
            _loaded = false;
            _family = fam || "";
            sel.innerHTML = "";
            if (v) {
                const o = document.createElement("option"); o.value = v;
                o.textContent = _cleanName(v); sel.appendChild(o); sel.value = v;
            } else {
                const o = document.createElement("option"); o.value = ""; o.textContent = "(none)";
                sel.appendChild(o);
            }
        },
    };
}

// ─── Extension registration ───────────────────────────────────────────────────

app.registerExtension({
    name: "FBnodes.WorkflowManager",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WorkflowManager") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated?.apply(this, arguments);
            const node = this;
            this.serialize_widgets = true;

            // Inject hidden state widget for workflow config persistence
            const hidden = {
                type: "STRING",
                name: "wm_workflow_config",
                value: "{}",
                serialize: true,
                hidden: true,
            };
            // Only add if not already present (node is being restored)
            if (!node.widgets?.find(w => w.name === "wm_workflow_config")) {
                node.addWidget("text", "wm_workflow_config", "{}", () => {}, { hidden: true, serialize: true });
                // Hide it from the canvas
                const hw = node.widgets.find(w => w.name === "wm_workflow_config");
                if (hw) {
                    hw.computeSize = () => [0, -4];
                    hw.type = "converted-widget";
                    hw.hidden = true;
                    hw.draw = () => {};
                }
            }

            // Build the workflow config panel as a DOM widget below the existing node UI
            const container = document.createElement("div");
            Object.assign(container.style, {
                padding: "2px 6px 6px", boxSizing: "border-box",
                fontFamily: "Inter, system-ui, -apple-system, sans-serif",
                width: "100%",
            });

            const panel = buildWorkflowConfigPanel(node, container);
            node._wmPanel = panel;

            const domW = node.addDOMWidget("wm_config_ui", "div", container, {
                hideOnZoom: false,
                serialize: false,
            });

            domW.computeSize = function (w) {
                return [w, 280]; // fixed approximate height
            };

            return r;
        };

        // ── Restore state on tab switch / workflow load ────────────────────
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            origConfigure?.apply(this, arguments);
            const node = this;

            // Read from properties (reliable across tab switches)
            const savedJson = node.properties?.wm_workflow_config
                || node.widgets?.find(w => w.name === "wm_workflow_config")?.value
                || "{}";

            let cfg = {};
            try { cfg = JSON.parse(savedJson); } catch {}

            if (node._wmPanel && Object.keys(cfg).length > 0) {
                node._wmPanel.setConfig(cfg);
            }
        };
    },
});
