/**
 * Workflow Context — DOM-based family / model / VAE / CLIP dropdowns.
 *
 * Mirrors the pattern used in workflow_builder.js:
 * - Dropdowns lazy-fetch from the same /workflow-extractor/* endpoints
 * - Family change reloads model/vae/clip lists (filtered by family)
 * - All selections are serialised into the hidden "override_data" widget
 *   which Python's unpack() reads as JSON.
 * - State persists across tab-switches via node.properties.
 * - On first connection, reads workflow_data from upstream to populate defaults.
 */
import { app } from "../../scripts/app.js";

// ── Colour palette (exact match to workflow_builder.js) ─────────────
const C = {
    bgDark:    "rgba(40, 44, 52, 0.6)",
    bgCard:    "rgba(40, 44, 52, 0.6)",
    bgInput:   "#1a1a1a",
    accent:    "rgba(66, 153, 225, 0.9)",
    text:      "#ccc",
    textMuted: "#aaa",
    border:    "rgba(226, 232, 240, 0.15)",
    error:     "rgba(220, 53, 69, 0.9)",
};

// ── Layout constants (exact match to workflow_builder.js) ───────────
const LABEL_W  = "35%";
const INPUT_W  = "58%";
const ROW_STYLE = {
    display: "flex", alignItems: "center", padding: "2px 0", fontSize: "11px", gap: "2px",
};
const INPUT_STYLE = {
    background: C.bgInput, color: C.text, border: `1px solid ${C.border}`,
    borderRadius: "6px", padding: "3px 6px", fontSize: "11px",
    width: INPUT_W, boxSizing: "border-box", textAlign: "right",
};
const RESET_STYLE = {
    background: "none", border: "none", color: C.textMuted, cursor: "pointer",
    fontSize: "11px", padding: "0 2px", lineHeight: "1", flexShrink: "0",
    visibility: "hidden", width: "14px", textAlign: "center",
};

// ── Helpers ──────────────────────────────────────────────────────────
function makeEl(tag, style, text) {
    const el = document.createElement(tag);
    if (style) Object.assign(el.style, style);
    if (text !== undefined) el.textContent = text;
    return el;
}

function makeResetBtn(onReset) {
    const btn = makeEl("span", { ...RESET_STYLE }, "\u21BA");
    btn.title = "Reset to workflow_data value";
    btn.onclick = (e) => { e.stopPropagation(); onReset(); };
    return btn;
}

function cleanModelName(fullPath) {
    let name = fullPath;
    const extIdx = name.lastIndexOf(".");
    if (extIdx > 0) name = name.substring(0, extIdx);
    const slashIdx = Math.max(name.lastIndexOf("/"), name.lastIndexOf("\\"));
    if (slashIdx >= 0) name = name.substring(slashIdx + 1);
    return name;
}

function groupModelOptions(models) {
    const groups = new Map();
    const ungrouped = [];
    for (const m of models) {
        const normalized = m.replace(/\\/g, "/");
        const lastSlash = normalized.lastIndexOf("/");
        if (lastSlash < 0) {
            ungrouped.push({ value: m, display: cleanModelName(m) });
        } else {
            const folderPath = normalized.substring(0, lastSlash);
            const groupLabel = folderPath.replace(/\//g, " - ");
            if (!groups.has(groupLabel)) groups.set(groupLabel, []);
            groups.get(groupLabel).push({ value: m, display: cleanModelName(m) });
        }
    }
    return { groups, ungrouped };
}

function populateGroupedSelect(sel, models, keepFirst) {
    const firstOpt = keepFirst ? sel.options[0] : null;
    sel.innerHTML = "";
    if (firstOpt) sel.appendChild(firstOpt);
    const { groups, ungrouped } = groupModelOptions(models);
    for (const item of ungrouped) {
        const o = document.createElement("option");
        o.value = item.value; o.textContent = item.display;
        o.style.color = C.text;
        sel.appendChild(o);
    }
    for (const [groupLabel, items] of groups) {
        const og = document.createElement("optgroup");
        og.label = groupLabel;
        for (const item of items) {
            const o = document.createElement("option");
            o.value = item.value; o.textContent = item.display;
            o.style.color = C.text;
            og.appendChild(o);
        }
        sel.appendChild(og);
    }
}

// ── Generic select-row builder (same API as workflow_builder.js) ────
function makeSelectRow(label, initialValue, lazyFetch, onChange, grouped) {
    const row = makeEl("div", { ...ROW_STYLE });
    row.appendChild(makeEl("span", { color: C.textMuted, width: LABEL_W, flexShrink: "0" }, label));
    let _origVal = initialValue || "";
    let _recolor;
    const resetBtn = makeResetBtn(() => {
        if (![...sel.options].some(o => o.value === _origVal)) {
            const o = document.createElement("option"); o.value = _origVal;
            o.textContent = grouped ? cleanModelName(_origVal) : _origVal;
            sel.insertBefore(o, sel.firstChild);
        }
        sel.value = _origVal;
        resetBtn.style.visibility = "hidden";
        _recolor();
        if (row._onReset) row._onReset(_origVal);
        if (onChange) onChange(_origVal);
    });
    row.appendChild(resetBtn);
    const sel = document.createElement("select");
    Object.assign(sel.style, { ...INPUT_STYLE });
    {
        const defOpt = document.createElement("option");
        defOpt.value = "";
        defOpt.textContent = "(Default)";
        sel.appendChild(defOpt);
        if (initialValue && initialValue !== "\u2014" && !initialValue.startsWith("(")) {
            const o = document.createElement("option"); o.value = initialValue;
            o.textContent = grouped ? cleanModelName(initialValue) : initialValue;
            sel.appendChild(o);
            sel.value = initialValue;
        } else {
            sel.value = "";
        }
    }
    let _loaded = false;
    _recolor = () => {
        const cur = sel.value;
        const opt = [...sel.options].find(o => o.value === cur);
        sel.style.color = (opt && opt.dataset.missing) ? C.error : C.text;
    };
    sel.onfocus = async () => {
        if (_loaded) return;
        _loaded = true;
        const options = await lazyFetch();
        const currentVal = sel.value;
        sel.innerHTML = "";
        const defOpt = document.createElement("option");
        defOpt.value = ""; defOpt.textContent = "(Default)";
        sel.appendChild(defOpt);
        if (grouped) {
            populateGroupedSelect(sel, options, true);
        } else {
            for (const opt of options) {
                const o = document.createElement("option"); o.value = opt; o.textContent = opt;
                o.style.color = C.text;
                sel.appendChild(o);
            }
        }
        if (_origVal && ![...sel.options].some(o => o.value === _origVal)) {
            resetBtn.style.visibility = "visible";
        }
        sel.value = currentVal || sel.options[0]?.value || "";
        _recolor();
    };
    sel.onchange = () => {
        resetBtn.style.visibility = sel.value !== _origVal ? "visible" : "hidden";
        _recolor();
        if (onChange) onChange(sel.value);
    };
    row.appendChild(sel);
    row._sel = sel;
    row._label = row.querySelector("span");
    row._resetBtn = resetBtn;
    row._setOriginal = (v, found) => {
        _origVal = v || "";
        _loaded = false;
        sel.innerHTML = "";
        const defOpt = document.createElement("option");
        defOpt.value = "";
        defOpt.textContent = "(Default)";
        sel.appendChild(defOpt);
        if (v && v !== "\u2014" && !v.startsWith("(")) {
            const o = document.createElement("option"); o.value = v;
            o.textContent = grouped ? cleanModelName(v) : v;
            if (found === false) { o.style.color = C.error; o.dataset.missing = "1"; }
            else { o.style.color = C.text; }
            sel.appendChild(o);
            sel.value = v;
        } else {
            sel.value = "";
        }
        sel.style.color = (found === false && v && !v.startsWith("(")) ? C.error : C.text;
        resetBtn.style.visibility = "hidden";
    };
    row._getValue = () => sel.value;
    row._resetLoaded = () => { _loaded = false; };
    return row;
}

async function reloadGroupedSelect(row, fetchFn, grouped) {
    const options = await fetchFn();
    const sel = row._sel;
    if (grouped) {
        sel.innerHTML = "";
        const noneOpt = document.createElement("option");
        noneOpt.value = ""; noneOpt.textContent = "(Default)";
        noneOpt.style.color = C.textMuted;
        sel.appendChild(noneOpt);
        populateGroupedSelect(sel, options, true);
        sel.value = "";
    } else {
        sel.innerHTML = "";
        const defOpt = document.createElement("option");
        defOpt.value = ""; defOpt.textContent = "(Default)";
        defOpt.style.color = C.textMuted;
        sel.appendChild(defOpt);
        for (const opt of options) {
            const o = document.createElement("option"); o.value = opt; o.textContent = opt;
            o.style.color = C.text;
            sel.appendChild(o);
        }
        sel.value = "";
    }
    if (row._resetLoaded) row._resetLoaded();
    sel.style.color = C.text;
}

// ── Forward wheel to canvas for zoom ────────────────────────────────
function forwardWheelToCanvas(element) {
    element.addEventListener("wheel", (e) => {
        const canvas = app.canvas?.canvas || document.querySelector("canvas.lgraphcanvas");
        if (canvas) {
            const ev = new WheelEvent("wheel", {
                bubbles: true, cancelable: true,
                clientX: e.clientX, clientY: e.clientY,
                deltaX: e.deltaX, deltaY: e.deltaY, deltaZ: e.deltaZ,
                deltaMode: e.deltaMode,
                ctrlKey: e.ctrlKey, shiftKey: e.shiftKey, altKey: e.altKey, metaKey: e.metaKey,
            });
            canvas.dispatchEvent(ev);
        }
    }, { passive: true });
}

// ── Sync overrides into hidden widget + node.properties ─────────────
function syncOverrides(node) {
    const ov = node._wcOverrides || {};
    const blob = JSON.stringify(ov);
    // Write to hidden override_data widget
    const w = node.widgets?.find(w => w.name === "override_data");
    if (w) w.value = blob;
    // Persist in node.properties for tab-switch survival
    if (!node.properties) node.properties = {};
    node.properties.wc_override_data = blob;
}

// ── Apply saved overrides to dropdowns after DOM is built ───────────
function applyOverrides(node, overrides) {
    if (!overrides || typeof overrides !== "object") return;

    if (overrides._family && node._wcFamilySel) {
        // Ensure family option exists
        if (![...node._wcFamilySel.options].some(o => o.value === overrides._family)) {
            const o = document.createElement("option");
            o.value = overrides._family; o.textContent = overrides._family;
            node._wcFamilySel.appendChild(o);
        }
        node._wcFamilySel.value = overrides._family;
        node._wcFamily = overrides._family;
    }
    if (overrides.model_a && node._wcModelRow) {
        node._wcModelRow._setOriginal(overrides.model_a);
    }
    if (overrides.model_b && node._wcModelBRow) {
        node._wcModelBRow._setOriginal(overrides.model_b);
    }
    if (overrides.vae && node._wcVaeRow) {
        node._wcVaeRow._setOriginal(overrides.vae);
    }
    const clipVal = Array.isArray(overrides.clip_names) ? overrides.clip_names[0] : overrides.clip_names;
    if (clipVal && node._wcClipRow) {
        node._wcClipRow._setOriginal(clipVal);
    }
}

// =====================================================================
// Registration
// =====================================================================
const NODE_TYPE = "WorkflowContext";

// ── Read workflow_data from upstream connected node ──────────────────
function readUpstreamWorkflowData(node) {
    const wfInput = node.inputs?.find(i => i.name === "workflow_data");
    if (!wfInput || wfInput.link == null) return null;
    const linkInfo = node.graph?.links?.[wfInput.link];
    if (!linkInfo) return null;
    const srcNode = node.graph?.getNodeById(linkInfo.origin_id);
    if (!srcNode) return null;
    // PromptExtractor / WorkflowBuilder cache their last output
    const cache = srcNode.properties?.we_last_workflow_data
               || srcNode.properties?.wc_last_workflow_data;
    if (!cache) return null;
    try { return JSON.parse(cache); } catch { return null; }
}

// ── Populate dropdowns from workflow_data dict ──────────────────────
async function populateFromData(node, wf) {
    if (!wf) return;
    const family = wf.family || "";
    if (family && node._wcFamilySel) {
        if (![...node._wcFamilySel.options].some(o => o.value === family)) {
            const o = document.createElement("option");
            o.value = family; o.textContent = family;
            node._wcFamilySel.appendChild(o);
        }
        node._wcFamilySel.value = family;
        node._wcFamily = family;
    }
    // Reload dropdowns for family, then set values
    if (family && node._onFamilyChanged) {
        await node._onFamilyChanged(family, { fromPopulate: true });
    }
    if (wf.model_a && node._wcModelRow) node._wcModelRow._setOriginal(wf.model_a);
    if (wf.model_b && node._wcModelBRow) node._wcModelBRow._setOriginal(wf.model_b);
    if (wf.vae && node._wcVaeRow) {
        const vaeName = typeof wf.vae === "object" ? wf.vae.name : wf.vae;
        if (vaeName) node._wcVaeRow._setOriginal(vaeName);
    }
    if (wf.clip && node._wcClipRow) {
        const clipNames = Array.isArray(wf.clip) ? wf.clip : [wf.clip];
        if (clipNames[0]) node._wcClipRow._setOriginal(clipNames[0]);
    }
    syncOverrides(node);
}

app.registerExtension({
    name: "prompt-manager.workflow-context",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);
            const node = this;

            node._wcOverrides = {};
            node._wcFamily = "";
            const _sync = () => syncOverrides(node);

            // ── Build root container (matches Builder layout) ───────
            const root = makeEl("div", {
                display: "flex", flexDirection: "column",
                padding: "4px 4px 4px 4px", flexShrink: "0",
                backgroundColor: C.bgCard, borderRadius: "6px",
                width: "100%", boxSizing: "border-box",
            });
            root._wcNode = node;
            forwardWheelToCanvas(root);

            // ── Family row ──────────────────────────────────────────
            const familyRow = makeEl("div", { ...ROW_STYLE });
            familyRow.appendChild(makeEl("span", {
                color: C.accent, width: LABEL_W, flexShrink: "0", fontWeight: "bold",
            }, "Type"));
            familyRow.appendChild(makeEl("span", { width: "14px", flexShrink: "0" }));
            const familySel = document.createElement("select");
            Object.assign(familySel.style, { ...INPUT_STYLE, color: C.accent, fontWeight: "bold" });
            let _familiesLoaded = false;
            familySel.onfocus = async () => {
                if (_familiesLoaded) return;
                _familiesLoaded = true;
                try {
                    const r = await fetch("/workflow-extractor/list-families");
                    const d = await r.json();
                    const families = d.families || {};
                    const curVal = familySel.value;
                    familySel.innerHTML = "";
                    const defOpt = document.createElement("option");
                    defOpt.value = ""; defOpt.textContent = "(Default)";
                    familySel.appendChild(defOpt);
                    const keys = Object.keys(families).sort((a, b) => {
                        if (a === "sdxl") return -1;
                        if (b === "sdxl") return 1;
                        return families[a].localeCompare(families[b]);
                    });
                    for (const key of keys) {
                        const o = document.createElement("option");
                        o.value = key; o.textContent = families[key];
                        familySel.appendChild(o);
                    }
                    familySel.value = curVal;
                } catch (e) { /* ignore */ }
            };
            {
                const o = document.createElement("option");
                o.value = ""; o.textContent = "(Default)";
                familySel.appendChild(o);
                familySel.value = "";
            }
            familySel.onchange = () => onFamilyChanged(familySel.value);
            familyRow.appendChild(familySel);
            root.appendChild(familyRow);
            node._wcFamilySel = familySel;

            // ── Model A row ─────────────────────────────────────────
            const fetchModels = async () => {
                try {
                    const fam = node._wcFamily;
                    const url = fam
                        ? `/workflow-extractor/list-models?family=${encodeURIComponent(fam)}`
                        : `/workflow-extractor/list-models`;
                    const r = await fetch(url); const d = await r.json();
                    return d.models || [];
                } catch { return []; }
            };
            const modelRow = makeSelectRow("Model A", "", fetchModels,
                (v) => { node._wcOverrides.model_a = v; _sync(); }, true);
            root.appendChild(modelRow);
            node._wcModelRow = modelRow;

            // ── Model B row ─────────────────────────────────────────
            const modelBRow = makeSelectRow("Model B", "", fetchModels,
                (v) => { node._wcOverrides.model_b = v; _sync(); }, true);
            root.appendChild(modelBRow);
            node._wcModelBRow = modelBRow;

            // ── Separator ───────────────────────────────────────────
            root.appendChild(makeEl("div", {
                height: "2px", background: "rgba(255,255,255,0.4)",
                margin: "8px 0", width: "100%",
            }));

            // ── VAE row ─────────────────────────────────────────────
            const vaeRow = makeSelectRow("VAE", "",
                async () => {
                    try {
                        const fam = encodeURIComponent(node._wcFamily || "");
                        const r = await fetch(`/workflow-extractor/list-vaes?family=${fam}`);
                        const d = await r.json(); return d.vaes || [];
                    } catch { return []; }
                },
                (v) => { node._wcOverrides.vae = v; _sync(); }, false);
            root.appendChild(vaeRow);
            node._wcVaeRow = vaeRow;

            // ── CLIP row ────────────────────────────────────────────
            const clipRow = makeSelectRow("CLIP", "",
                async () => {
                    try {
                        const fam = encodeURIComponent(node._wcFamily || "");
                        const r = await fetch(`/workflow-extractor/list-clips?family=${fam}`);
                        const d = await r.json(); return d.clips || [];
                    } catch { return []; }
                },
                (v) => { node._wcOverrides.clip_names = v ? [v] : []; _sync(); }, false);
            root.appendChild(clipRow);
            node._wcClipRow = clipRow;

            // ── Family change handler ───────────────────────────────
            const onFamilyChanged = async (familyKey, { fromPopulate = false } = {}) => {
                node._wcFamily = familyKey;
                node._wcOverrides._family = familyKey;
                if (!fromPopulate) {
                    delete node._wcOverrides.model_a;
                    delete node._wcOverrides.model_b;
                    delete node._wcOverrides.vae;
                    delete node._wcOverrides.clip_names;
                }
                const fam = encodeURIComponent(familyKey || "");
                const fetchModelsForFamily = async () => {
                    try {
                        const r = await fetch(`/workflow-extractor/list-models?family=${fam}`);
                        const d = await r.json(); return d.models || [];
                    } catch { return []; }
                };
                const reloadVae = async () => {
                    try {
                        const r = await fetch(`/workflow-extractor/list-vaes?family=${fam}`);
                        const d = await r.json();
                        await reloadGroupedSelect(node._wcVaeRow, async () => d.vaes || [], false);
                    } catch { await reloadGroupedSelect(node._wcVaeRow, async () => [], false); }
                };
                const reloadClip = async () => {
                    try {
                        const r = await fetch(`/workflow-extractor/list-clips?family=${fam}`);
                        const d = await r.json();
                        await reloadGroupedSelect(node._wcClipRow, async () => d.clips || [], false);
                    } catch { await reloadGroupedSelect(node._wcClipRow, async () => [], false); }
                };
                await Promise.all([
                    reloadGroupedSelect(node._wcModelRow, fetchModelsForFamily, true),
                    reloadGroupedSelect(node._wcModelBRow, fetchModelsForFamily, true),
                    reloadVae(),
                    reloadClip(),
                ]);
                if (!fromPopulate) _sync();
            };
            node._onFamilyChanged = onFamilyChanged;

            // ── Register DOM widget ─────────────────────────────────
            const domW = node.addDOMWidget("wc_ui", "div", root, {
                hideOnZoom: false,
                serialize: false,
            });
            domW.computeSize = function (width) {
                // Fixed height like PMA: 5 rows (~24px each) + separator (18px) + padding (20px)
                return [width, 174];
            };

            // ── Hide override_data widget ────────────────────────
            const HIDE_WIDGETS = new Set(["override_data"]);
            for (const w of (node.widgets || [])) {
                if (w === domW) continue;
                if (HIDE_WIDGETS.has(w.name)) {
                    w.computeSize = () => [0, -4];
                    w.hidden = true;
                    w.draw = function () {};
                    if (w.element) w.element.style.display = "none";
                }
            }

            // ── Initial sync ────────────────────────────────────────
            _sync();

            // ── Populate from upstream workflow_data if available ────
            if (!node._configuredFromWorkflow) {
                requestAnimationFrame(async () => {
                    const wf = readUpstreamWorkflowData(node);
                    if (wf) await populateFromData(node, wf);
                });
            } else {
                // Restoring from workflow — apply saved overrides
                try {
                    const saved = JSON.parse(node.properties?.wc_override_data || "{}");
                    applyOverrides(node, saved);
                } catch { /* ignore */ }
            }
        };

        // ── Tab-switch / workflow-load restore ───────────────────────
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            if (onConfigure) onConfigure.apply(this, arguments);
            const node = this;
            node._configuredFromWorkflow = true;

            // Restore overrides from properties (synchronous — no API calls)
            try {
                const saved = JSON.parse(node.properties?.wc_override_data || "{}");
                node._wcOverrides = saved;
                node._wcFamily = saved._family || "";
                // DOM may not exist yet if onConfigure fires before onNodeCreated.
                // In that case, onNodeCreated's tail block handles it.
                if (node._wcFamilySel) {
                    applyOverrides(node, saved);
                }
            } catch { /* ignore */ }
        };
    },
});
