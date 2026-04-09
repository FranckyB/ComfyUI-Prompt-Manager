/**
 * Workflow Extractor – Full DOM-based UI
 *
 * Every native widget is hidden; the entire node surface is one DOM widget.
 * Uses native ComfyUI image widget for < > navigation, Browse button for modal.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createFileBrowserModal } from "./file_browser.js";

// ─── LoRA Manager preview tooltip (lazy-loaded) ─────────────────────────────
let loraManagerPreviewTooltip = null;
let loraManagerCheckDone = false;
let loraManagerAvailable = false;

async function getLoraManagerPreviewTooltip() {
    if (loraManagerCheckDone) return loraManagerPreviewTooltip;
    loraManagerCheckDone = true;
    try {
        const mod = await import("/extensions/ComfyUI-Lora-Manager/preview_tooltip.js");
        if (mod?.PreviewTooltip) {
            loraManagerPreviewTooltip = new mod.PreviewTooltip({ modelType: "loras" });
            loraManagerAvailable = true;
            console.log("[WorkflowExtractor] LoRA Manager preview integration enabled");
        }
    } catch (e) {
        console.log("[WorkflowExtractor] LoRA Manager preview not available:", e.message);
        loraManagerAvailable = false;
    }
    return loraManagerPreviewTooltip;
}
getLoraManagerPreviewTooltip();

// ─── Forward wheel to canvas for zoom ────────────────────────────────────────
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

// ─── Zoom-aware font scaling for select dropdowns ────────────────────────────
// ─── Larger dropdown text via CSS injection ──────────────────────────────────
// Inject a global style to make all select option text larger for readability.
(function injectSelectStyle() {
    if (document.getElementById("we-select-style")) return;
    const style = document.createElement("style");
    style.id = "we-select-style";
    style.textContent = `
        .we-select option, .we-select optgroup { font-size: 16px; }
        .we-select optgroup { font-weight: bold; color: #e06060; }
    `;
    document.head.appendChild(style);
})();

function applyZoomScaling(rootEl) {
    // Just tag all selects with the class for larger dropdown text
    const tag = (sel) => sel.classList.add("we-select");
    rootEl.querySelectorAll("select").forEach(tag);
    const obs = new MutationObserver((muts) => {
        for (const m of muts) for (const n of m.addedNodes) {
            if (n.nodeName === "SELECT") tag(n);
            else if (n.querySelectorAll) n.querySelectorAll("select").forEach(tag);
        }
    });
    obs.observe(rootEl, { childList: true, subtree: true });
}

// ─── Colour palette ─────────────────────────────────────────────────────────
const C = {
    bgDark:    "rgba(40, 44, 52, 0.6)",
    bgCard:    "rgba(40, 44, 52, 0.6)",
    bgInput:   "#1a1a1a",
    accent:    "rgba(66, 153, 225, 0.9)",
    accentDim: "rgba(66, 153, 225, 0.7)",
    text:      "#ccc",
    textMuted: "#aaa",
    border:    "rgba(226, 232, 240, 0.15)",
    success:   "#6c6",
    warning:   "#da3",
    error:     "rgba(220, 53, 69, 0.9)",
};

// ─── Tiny helpers ────────────────────────────────────────────────────────────
function makeEl(tag, style, text) {
    const el = document.createElement(tag);
    if (style) Object.assign(el.style, style);
    if (text !== undefined) el.textContent = text;
    return el;
}

function makeBtn(label, onclick, extraStyle) {
    const b = makeEl("button", {
        padding: "5px 10px", border: `1px solid ${C.border}`, borderRadius: "6px",
        background: C.bgInput, color: "#ccc", cursor: "pointer",
        fontSize: "12px", fontWeight: "600", width: "100%",
        ...extraStyle,
    }, label);
    const baseBg = extraStyle?.background || C.bgInput;
    b.onmouseenter = () => b.style.background = "#2a2a2a";
    b.onmouseleave = () => b.style.background = baseBg;
    b.onclick = onclick;
    return b;
}

// ─── Section builder ─────────────────────────────────────────────────────────
function makeSection(title, collapsed, bodyHeight, onToggle) {
    const wrap = makeEl("div", {
        borderRadius: "6px", overflow: "hidden", marginTop: "2px",
        backgroundColor: C.bgCard,
    });
    const header = makeEl("div", {
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "4px 8px", cursor: "pointer",
        fontSize: "11px", fontWeight: "600", color: "#aaa",
        userSelect: "none",
        borderBottom: "1px solid rgba(255,255,255,0.1)",
    });
    const arrow = makeEl("span", { transition: "transform .2s" }, collapsed ? "▶" : "▼");
    const label = makeEl("span", {}, title);
    header.append(arrow, label);
    const body = makeEl("div", {
        padding: "4px 8px", display: collapsed ? "none" : "block",
    });
    wrap._collapsed = !!collapsed;
    wrap._bodyH = bodyHeight || 0;
    header.onclick = () => {
        const open = body.style.display !== "none";
        body.style.display = open ? "none" : "block";
        arrow.textContent = open ? "▶" : "▼";
        wrap._collapsed = open;
        if (onToggle) onToggle();
    };
    wrap.append(header, body);
    wrap._body = body;
    return wrap;
}

// ─── Shared layout constants ─────────────────────────────────────────────────
const LABEL_W = "35%";
const INPUT_W = "58%";
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

function makeResetBtn(onReset) {
    const btn = makeEl("span", { ...RESET_STYLE }, "↺");
    btn.title = "Reset to extracted value";
    btn.onclick = (e) => { e.stopPropagation(); onReset(); };
    return btn;
}

function makeRow(label, value) {
    const row = makeEl("div", { ...ROW_STYLE });
    row.appendChild(makeEl("span", { color: C.textMuted, width: LABEL_W, flexShrink: "0" }, label));
    const resetBtn = makeEl("span", { ...RESET_STYLE });
    row.appendChild(resetBtn);
    const val = makeEl("span", {
        color: C.text, textAlign: "right", width: INPUT_W,
        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
    }, value);
    row.appendChild(val);
    row._val = val;
    row._resetBtn = resetBtn;
    return row;
}

// ─── Clean model name for display ────────────────────────────────────────────
function cleanModelName(fullPath) {
    // Remove extension
    let name = fullPath;
    const extIdx = name.lastIndexOf(".");
    if (extIdx > 0) name = name.substring(0, extIdx);
    // Get just the filename (no path)
    const slashIdx = Math.max(name.lastIndexOf("/"), name.lastIndexOf("\\"));
    if (slashIdx >= 0) name = name.substring(slashIdx + 1);
    return name;
}

// ─── Group models by folder path for dropdown ────────────────────────────────
function groupModelOptions(models) {
    // Group by folder prefix: "Illustrious\2D\file.ext" → group="Illustrious - 2D", name="file"
    const groups = new Map(); // groupLabel → [{value, display}]
    const ungrouped = [];

    for (const m of models) {
        const normalized = m.replace(/\\/g, "/");
        const lastSlash = normalized.lastIndexOf("/");
        if (lastSlash < 0) {
            // No folder — ungrouped
            ungrouped.push({ value: m, display: cleanModelName(m) });
        } else {
            const folderPath = normalized.substring(0, lastSlash);
            const groupLabel = folderPath.replace(/\//g, " - ");
            const display = cleanModelName(m);
            if (!groups.has(groupLabel)) groups.set(groupLabel, []);
            groups.get(groupLabel).push({ value: m, display });
        }
    }
    return { groups, ungrouped };
}

// Build a <select> with <optgroup> for grouped models
function populateGroupedSelect(sel, models, keepFirst) {
    const firstOpt = keepFirst ? sel.options[0] : null;
    sel.innerHTML = "";
    if (firstOpt) sel.appendChild(firstOpt);

    const { groups, ungrouped } = groupModelOptions(models);

    // Add ungrouped first
    for (const item of ungrouped) {
        const o = document.createElement("option");
        o.value = item.value; o.textContent = item.display;
        o.style.color = C.text;
        sel.appendChild(o);
    }

    // Add grouped with optgroups
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

// Editable select row with grouped model display
function makeSelectRow(label, initialValue, lazyFetch, onChange, grouped) {
    const row = makeEl("div", { ...ROW_STYLE });
    row.appendChild(makeEl("span", { color: C.textMuted, width: LABEL_W, flexShrink: "0" }, label));
    let _origVal = initialValue || "";
    let _recolor;
    const resetBtn = makeResetBtn(() => {
        // Re-add original to dropdown if missing
        if (![...sel.options].some(o => o.value === _origVal)) {
            const o = document.createElement("option"); o.value = _origVal;
            o.textContent = grouped ? cleanModelName(_origVal) : _origVal;
            sel.insertBefore(o, sel.firstChild);
        }
        sel.value = _origVal;
        resetBtn.style.visibility = "hidden";
        _recolor();
        // Notify parent (e.g. restore family + VAE/CLIP)
        if (row._onReset) row._onReset(_origVal);
        if (onChange) onChange(_origVal);
    });
    row.appendChild(resetBtn);
    const sel = document.createElement("select");
    Object.assign(sel.style, { ...INPUT_STYLE });
    if (initialValue) {
        const o = document.createElement("option"); o.value = initialValue;
        o.textContent = grouped ? cleanModelName(initialValue) : initialValue;
        sel.appendChild(o);
        sel.value = initialValue;
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
        if (grouped) {
            populateGroupedSelect(sel, options, false);
        } else {
            sel.innerHTML = "";
            for (const opt of options) {
                const o = document.createElement("option"); o.value = opt; o.textContent = opt;
                o.style.color = C.text;
                sel.appendChild(o);
            }
        }
        // Re-add original if not in list (mark as missing)
        if (_origVal && ![...sel.options].some(o => o.value === _origVal)) {
            // Don't show missing original in dropdown — use reset button to restore
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
    row._resetBtn = resetBtn;
    row._setOriginal = (v, found) => {
        _origVal = v || "";
        _loaded = false;
        sel.innerHTML = "";
        if (v) {
            const o = document.createElement("option"); o.value = v;
            o.textContent = grouped ? cleanModelName(v) : v;
            if (found === false) { o.style.color = C.error; o.dataset.missing = "1"; }
            else { o.style.color = C.text; }
            sel.appendChild(o);
            sel.value = v;
        }
        sel.style.color = (found === false) ? C.error : C.text;
        resetBtn.style.visibility = "hidden";
    };
    row._getValue = () => sel.value;
    row._resetLoaded = () => { _loaded = false; };
    return row;
}

// Helper to reload a grouped select from API
async function reloadGroupedSelect(row, fetchFn, grouped) {
    const options = await fetchFn();
    const sel = row._sel;
    if (grouped) {
        populateGroupedSelect(sel, options, false);
    } else {
        sel.innerHTML = "";
        for (const opt of options) {
            const o = document.createElement("option"); o.value = opt; o.textContent = opt;
            o.style.color = C.text;
            sel.appendChild(o);
        }
    }
    // Reset lazy-load flag so next focus re-fetches with current family
    if (row._resetLoaded) row._resetLoaded();
    if (sel.options.length) sel.value = sel.options[0].value;
    sel.style.color = C.text;
}

function makeInput(label, type, value, attrs, onChange) {
    const row = makeEl("div", { ...ROW_STYLE });
    row.appendChild(makeEl("span", { color: C.textMuted, width: LABEL_W, flexShrink: "0" }, label));
    let _origVal = value;
    const resetBtn = makeResetBtn(() => {
        inp.value = _origVal;
        resetBtn.style.visibility = "hidden";
        if (onChange) onChange();
    });
    row.appendChild(resetBtn);
    let inp;
    if (type === "select") {
        inp = document.createElement("select");
        for (const opt of (attrs?.options || [])) {
            const o = document.createElement("option");
            o.value = opt; o.textContent = opt;
            inp.appendChild(o);
        }
        inp.value = value;
    } else {
        inp = document.createElement("input");
        inp.type = type;
        inp.value = value;
        if (attrs?.step) inp.step = attrs.step;
        if (attrs?.min !== undefined) inp.min = attrs.min;
        if (attrs?.max !== undefined) inp.max = attrs.max;
    }
    Object.assign(inp.style, { ...INPUT_STYLE });
    inp.onchange = () => {
        resetBtn.style.visibility = String(inp.value) !== String(_origVal) ? "visible" : "hidden";
        if (onChange) onChange();
    };
    row.appendChild(inp);
    row._inp = inp;
    row._resetBtn = resetBtn;
    row._setOriginal = (v) => {
        _origVal = v;
        inp.value = v;
        resetBtn.style.visibility = "hidden";
    };
    return row;
}

// ─── LoRA tag builder (matches PM Advanced) ──────────────────────────────────
function makeLoraTag(lora, avail, onToggle, onStrength) {
    const name = lora.name || "?";
    const str = lora.model_strength ?? 1.0;
    const active = lora._active !== false;

    let bgColor, textColor, borderColor;
    if (!avail) {
        bgColor = active ? "rgba(220, 53, 69, 0.9)" : "rgba(220, 53, 69, 0.4)";
        textColor = active ? "white" : "rgba(255, 200, 200, 0.8)";
        borderColor = "rgba(220, 53, 69, 0.9)";
    } else if (active) {
        bgColor = "rgba(66, 153, 225, 0.9)";
        textColor = "white";
        borderColor = "rgba(122, 188, 243, 0.9)";
    } else {
        bgColor = "rgba(45, 55, 72, 0.7)";
        textColor = "rgba(226, 232, 240, 0.6)";
        borderColor = "rgba(226, 232, 240, 0.2)";
    }

    const tag = makeEl("div", {
        display: "inline-flex", alignItems: "center", gap: "6px",
        padding: "4px 10px", borderRadius: "6px", fontSize: "12px",
        cursor: "pointer", userSelect: "none", flexShrink: "0",
        transition: "background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease",
        backgroundColor: bgColor, color: textColor,
        border: `1px solid ${borderColor}`,
        width: "200px", height: "24px", boxSizing: "border-box",
    });

    if (!avail) {
        const warn = makeEl("span", { fontSize: "11px", marginRight: "-2px" }, "⚠️");
        tag.appendChild(warn);
    }

    const nameSpan = makeEl("span", {
        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        flexGrow: "1",
        textDecoration: !avail ? "line-through" : "none",
        opacity: !avail ? "0.8" : "1",
    }, name);
    tag.appendChild(nameSpan);

    const sinp = document.createElement("input");
    sinp.type = "text";
    sinp.value = str.toFixed(2);
    Object.assign(sinp.style, {
        fontSize: "10px", fontWeight: "600", padding: "1px 4px",
        borderRadius: "999px", backgroundColor: "rgba(255,255,255,0.15)",
        color: "rgba(255,255,255,0.9)", width: "38px", textAlign: "center",
        border: "1px solid transparent", outline: "none", cursor: "text",
    });
    sinp.addEventListener("focus", (e) => {
        e.stopPropagation(); sinp.select();
        sinp.style.backgroundColor = "rgba(255,255,255,0.25)";
        sinp.style.border = "1px solid rgba(66, 153, 225, 0.6)";
    });
    sinp.addEventListener("blur", () => {
        sinp.style.backgroundColor = "rgba(255,255,255,0.15)";
        sinp.style.border = "1px solid transparent";
        const v = parseFloat(sinp.value);
        if (!isNaN(v)) { onStrength?.(v); } else { sinp.value = str.toFixed(2); }
    });
    sinp.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); sinp.blur(); }
        else if (e.key === "Escape") { e.preventDefault(); sinp.value = str.toFixed(2); sinp.blur(); }
        e.stopPropagation();
    });
    sinp.addEventListener("click", (e) => e.stopPropagation());
    tag.appendChild(sinp);

    tag.addEventListener("click", (e) => {
        if (e.target !== sinp && onToggle) onToggle();
    });

    let hoverTimeout = null;
    tag.addEventListener("mouseenter", () => {
        tag.style.transform = "translateY(-1px)";
        tag.style.boxShadow = "0 2px 4px rgba(0,0,0,0.2)";
        if (loraManagerAvailable && avail) {
            const rect = tag.getBoundingClientRect();
            hoverTimeout = setTimeout(async () => {
                const tooltip = await getLoraManagerPreviewTooltip();
                if (tooltip) tooltip.show(name, rect.right, rect.top);
            }, 300);
        }
    });
    tag.addEventListener("mouseleave", () => {
        tag.style.transform = "translateY(0)";
        tag.style.boxShadow = "none";
        if (hoverTimeout) { clearTimeout(hoverTimeout); hoverTimeout = null; }
        if (loraManagerPreviewTooltip) loraManagerPreviewTooltip.hide();
    });

    if (!loraManagerAvailable || !avail) {
        tag.title = !avail
            ? `${name}\n⚠️ NOT FOUND - This LoRA is missing from your system`
            : `${name}\nStrength: ${str.toFixed(2)}\nClick to toggle on/off`;
    }

    return tag;
}

// ─── Extract call ────────────────────────────────────────────────────────────
async function doExtract(node) {
    const wGet = (name) => node.widgets?.find(w => w.name === name)?.value;
    const statusEl = node._weStatusEl;
    const filename = wGet("image");
    if (!filename || filename === "(none)") {
        if (statusEl) { statusEl.textContent = "Select an image to extract"; statusEl.style.color = C.textMuted; }
        return;
    }
    if (statusEl) { statusEl.textContent = "⏳ Extracting…"; statusEl.style.color = C.warning; }
    try {
        const resp = await fetch("/workflow-extractor/extract", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename, source: node._weSourceFolder || "input" }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        node._weExtracted = data;
        node._weLoraState = {};
        node._weOverrides = {};
        updateUI(node);
        if (statusEl) { statusEl.textContent = ""; }
    } catch (err) {
        console.error("[WE] extract error", err);
        if (statusEl) { statusEl.textContent = `❌ ${err.message}`; statusEl.style.color = C.error; }
    }
}

// ─── Update UI from extracted data ──────────────────────────────────────────
function updateUI(node) {
    const d = node._weExtracted;
    if (!d) return;

    // Model family
    const family = d.model_family || null;
    node._weFamily = family;
    if (node._weFamilySel) {
        const sel = node._weFamilySel;
        const label = d.model_family_label || "Unknown";
        if (family && ![...sel.options].some(o => o.value === family)) {
            const o = document.createElement("option");
            o.value = family; o.textContent = label;
            sel.appendChild(o);
        }
        sel.value = family || "";
    }

    // Model
    if (node._weModelRow?._setOriginal) {
        const name = d.model_a || "—";
        node._weModelRow._setOriginal(name, d.model_a_found !== false);
    }

    // LoRAs
    updateLoras(node);

    // VAE
    if (node._weVaeRow?._setOriginal) {
        const vn = d.vae?.name || "—";
        node._weVaeRow._setOriginal(vn, d.vae_found !== false);
    }

    // CLIP
    if (node._weClipRow?._setOriginal) {
        const cn = (d.clip?.names || []).map(n => n || "—").join(", ") || "—";
        node._weClipRow._setOriginal(cn);
    }

    // Sampler
    const s = d.sampler || {};
    if (node._weSamplerRows) {
        const rows = node._weSamplerRows;
        if (rows.steps?._setOriginal) rows.steps._setOriginal(s.steps ?? 20);
        if (rows.cfg?._setOriginal) rows.cfg._setOriginal(s.cfg ?? 7.0);
        if (rows.seed?._setOriginal) rows.seed._setOriginal(s.seed ?? 0);
        if (rows.sampler?._setOriginal) rows.sampler._setOriginal(s.sampler_name ?? "euler");
        if (rows.scheduler?._setOriginal) rows.scheduler._setOriginal(s.scheduler ?? "normal");
        if (rows.denoise?._setOriginal) rows.denoise._setOriginal(s.denoise ?? 1.0);
    }

    // Resolution
    const r = d.resolution || {};
    if (node._weResRows) {
        const rr = node._weResRows;
        if (rr.width?._setOriginal) rr.width._setOriginal(r.width ?? 512);
        if (rr.height?._setOriginal) rr.height._setOriginal(r.height ?? 512);
        if (rr.batch?._setOriginal) rr.batch._setOriginal(r.batch_size ?? 1);
        if (rr.length) {
            if (r.length != null) {
                rr.length.style.display = "flex";
                if (rr.length._setOriginal) rr.length._setOriginal(r.length);
            } else {
                rr.length.style.display = "none";
            }
        }
    }

    // Prompts
    if (node._wePosBox) node._wePosBox.value = d.positive_prompt || "";
    if (node._weNegBox) node._weNegBox.value = d.negative_prompt || "";

    syncHidden(node);

    // Recalc height after all sections are populated
    if (node._weRecalc) setTimeout(() => node._weRecalc(), 50);
}

function updateLoras(node) {
    const container = node._weLoraContainer;
    if (!container) return;
    container.innerHTML = "";
    const d = node._weExtracted;
    if (!d) return;
    const allLoras = [...(d.loras_a || []), ...(d.loras_b || [])];
    if (!allLoras.length) {
        container.appendChild(makeEl("div", {
            color: "rgba(200, 200, 200, 0.5)", fontStyle: "italic",
            fontSize: "11px", padding: "8px", width: "100%", textAlign: "center",
        }, "No LoRAs connected"));
        return;
    }
    for (const lora of allLoras) {
        const name = lora.name || "";
        const avail = d.lora_availability?.[name] !== false;
        if (node._weLoraState[name] === undefined) {
            node._weLoraState[name] = { active: true, model_strength: lora.model_strength ?? 1.0, clip_strength: lora.clip_strength ?? 1.0 };
        }
        const st = node._weLoraState[name];
        lora._active = st.active;
        lora.model_strength = st.model_strength;
        const tag = makeLoraTag(lora, avail,
            () => { st.active = !st.active; updateLoras(node); syncHidden(node); },
            (v) => { st.model_strength = v; st.clip_strength = v; syncHidden(node); },
        );
        container.appendChild(tag);
    }
    // Recalc node height after lora tags change
    if (node._weRecalc) setTimeout(() => node._weRecalc(), 10);
}

function syncHidden(node) {
    const wSet = (name, val) => {
        const w = node.widgets?.find(x => x.name === name);
        if (w) w.value = val;
    };
    const ov = { ...node._weOverrides };
    if (node._weModelRow?._getValue) {
        const v = node._weModelRow._getValue();
        if (v && v !== "—") ov.model_a = v;
    }
    if (node._weVaeRow?._getValue) {
        const v = node._weVaeRow._getValue();
        if (v && v !== "—") ov.vae = v;
    }
    if (node._weClipRow?._getValue) {
        const v = node._weClipRow._getValue();
        if (v && v !== "—") ov.clip_names = [v];
    }
    if (node._weSamplerRows) {
        const r = node._weSamplerRows;
        if (r.steps?._inp) ov.steps = parseInt(r.steps._inp.value) || 20;
        if (r.cfg?._inp) ov.cfg = parseFloat(r.cfg._inp.value) || 7.0;
        if (r.seed?._inp) ov.seed = parseInt(r.seed._inp.value) || 0;
        if (r.sampler?._inp) ov.sampler_name = r.sampler._inp.value;
        if (r.scheduler?._inp) ov.scheduler = r.scheduler._inp.value;
        if (r.denoise?._inp) ov.denoise = parseFloat(r.denoise._inp.value) || 1.0;
    }
    if (node._weResRows) {
        const r = node._weResRows;
        if (r.width?._inp) ov.width = parseInt(r.width._inp.value) || 512;
        if (r.height?._inp) ov.height = parseInt(r.height._inp.value) || 512;
        if (r.batch?._inp) ov.batch_size = parseInt(r.batch._inp.value) || 1;
        if (r.length?._inp && r.length.style.display !== "none") {
            ov.length = parseInt(r.length._inp.value) || 81;
        }
    }
    if (node._wePosBox) ov.positive_prompt = node._wePosBox.value;
    if (node._weNegBox) ov.negative_prompt = node._weNegBox.value;

    wSet("override_data", JSON.stringify(ov));
    wSet("lora_state", JSON.stringify(node._weLoraState || {}));
}

// ─── Apply saved overrides back to UI after reload ───────────────────────────
function applyOverrides(node, ovJson, lsJson) {
    let ov, ls;
    try { ov = JSON.parse(ovJson || "{}"); } catch { ov = {}; }
    try { ls = JSON.parse(lsJson || "{}"); } catch { ls = {}; }
    const d = node._weExtracted;
    if (!d) return;
    const isEmpty = (o) => !o || Object.keys(o).length === 0;
    if (isEmpty(ov) && isEmpty(ls)) return;

    // Helper: apply override to a select row if value differs from extracted
    const applySelect = (row, ovVal, grouped) => {
        if (!row || ovVal == null || ovVal === "—") return;
        const sel = row._sel;
        if (!sel) return;
        // Add option if not in dropdown
        if (![...sel.options].some(o => o.value === ovVal)) {
            const o = document.createElement("option");
            o.value = ovVal;
            o.textContent = grouped ? cleanModelName(ovVal) : ovVal;
            sel.appendChild(o);
        }
        sel.value = ovVal;
        if (row._resetBtn) {
            row._resetBtn.style.visibility = sel.value !== sel.options[0]?.value ? "visible" : "hidden";
        }
    };

    // Helper: apply override to an input row if value differs from extracted
    const applyInput = (row, ovVal) => {
        if (!row || ovVal == null) return;
        const inp = row._inp;
        if (!inp) return;
        inp.value = ovVal;
        if (row._resetBtn) {
            row._resetBtn.style.visibility = "visible";
        }
    };

    // Model
    if (ov.model_a && ov.model_a !== (d.model_a || "—")) {
        applySelect(node._weModelRow, ov.model_a, true);
    }

    // VAE
    if (ov.vae && ov.vae !== (d.vae?.name || "—")) {
        applySelect(node._weVaeRow, ov.vae, true);
    }

    // CLIP
    if (ov.clip_names?.length && ov.clip_names[0] !== ((d.clip?.names || [])[0] || "—")) {
        applySelect(node._weClipRow, ov.clip_names[0], true);
    }

    // Sampler fields
    const s = d.sampler || {};
    if (node._weSamplerRows) {
        const rows = node._weSamplerRows;
        if (ov.steps != null && ov.steps !== (s.steps ?? 20)) applyInput(rows.steps, ov.steps);
        if (ov.cfg != null && ov.cfg !== (s.cfg ?? 7.0)) applyInput(rows.cfg, ov.cfg);
        if (ov.seed != null && ov.seed !== (s.seed ?? 0)) applyInput(rows.seed, ov.seed);
        if (ov.sampler_name && ov.sampler_name !== (s.sampler_name ?? "euler")) applyInput(rows.sampler, ov.sampler_name);
        if (ov.scheduler && ov.scheduler !== (s.scheduler ?? "normal")) applyInput(rows.scheduler, ov.scheduler);
        if (ov.denoise != null && ov.denoise !== (s.denoise ?? 1.0)) applyInput(rows.denoise, ov.denoise);
    }

    // Resolution fields
    const r = d.resolution || {};
    if (node._weResRows) {
        const rr = node._weResRows;
        if (ov.width != null && ov.width !== (r.width ?? 512)) applyInput(rr.width, ov.width);
        if (ov.height != null && ov.height !== (r.height ?? 512)) applyInput(rr.height, ov.height);
        if (ov.batch_size != null && ov.batch_size !== (r.batch_size ?? 1)) applyInput(rr.batch, ov.batch_size);
        if (ov.length != null && ov.length !== r.length) applyInput(rr.length, ov.length);
    }

    // Prompts
    if (ov.positive_prompt != null && ov.positive_prompt !== (d.positive_prompt || "")) {
        if (node._wePosBox) node._wePosBox.value = ov.positive_prompt;
    }
    if (ov.negative_prompt != null && ov.negative_prompt !== (d.negative_prompt || "")) {
        if (node._weNegBox) node._weNegBox.value = ov.negative_prompt;
    }

    // LoRA state
    if (!isEmpty(ls)) {
        node._weLoraState = ls;
        updateLoras(node);
    }

    // Re-sync hidden widgets with the restored values
    syncHidden(node);
}

// ─── Samplers / Schedulers constants ─────────────────────────────────────────
const SAMPLERS = [
    "euler","euler_cfg_pp","euler_ancestral","euler_ancestral_cfg_pp",
    "heun","heunpp2","dpm_2","dpm_2_ancestral","lms","dpm_fast",
    "dpm_adaptive","dpmpp_2s_ancestral","dpmpp_sde","dpmpp_sde_gpu",
    "dpmpp_2m","dpmpp_2m_sde","dpmpp_2m_sde_gpu","dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu","ddpm","lcm","ipndm","ipndm_v","deis","ddim",
    "uni_pc","uni_pc_bh2",
];
const SCHEDULERS = [
    "normal","karras","exponential","sgm_uniform","simple","ddim_uniform",
    "beta",
];

// ─── Thumbnail helper ────────────────────────────────────────────────────────
function loadThumbnail(node, filename) {
    const thumbEl = node._weThumbEl;
    if (!thumbEl) return;
    if (!filename || filename === "(none)") {
        thumbEl.innerHTML = "";
        node._weThumbH = 0;
        if (node._weRecalc) node._weRecalc();
        return;
    }
    const src = node._weSourceFolder || "input";
    // Handle subfolder paths (e.g. "subfolder/image.png")
    let actualFilename = filename;
    let subfolder = "";
    if (filename.includes("/")) {
        const lastSlash = filename.lastIndexOf("/");
        subfolder = filename.substring(0, lastSlash);
        actualFilename = filename.substring(lastSlash + 1);
    }
    let url = `/view?filename=${encodeURIComponent(actualFilename)}&type=${encodeURIComponent(src)}`;
    if (subfolder) url += `&subfolder=${encodeURIComponent(subfolder)}`;
    url += `&t=${Date.now()}`;
    const img = document.createElement("img");
    img.src = url;
    Object.assign(img.style, {
        width: "100%", maxHeight: "200px", objectFit: "contain",
        borderRadius: "4px", display: "block", margin: "0 auto",
    });
    img.onload = () => {
        node._weThumbH = Math.min(img.naturalHeight, 200);
        if (node._weRecalc) node._weRecalc();
    };
    img.onerror = () => {
        thumbEl.innerHTML = "";
        node._weThumbH = 0;
        if (node._weRecalc) node._weRecalc();
    };
    thumbEl.innerHTML = "";
    thumbEl.appendChild(img);
}

// ═════════════════════════════════════════════════════════════════════════════
// ─── Main extension ─────────────────────────────────────────────────────────
// ═════════════════════════════════════════════════════════════════════════════
app.registerExtension({
    name: "FBnodes.WorkflowExtractor",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WorkflowExtractor") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated?.apply(this, arguments);
            const node = this;

            // ── State ────────────────────────────────────────────────
            node._weExtracted = null;
            node._weLoraState = {};
            node._weOverrides = {};
            node._weLastImage = null;
            node._weSourceFolder = "input";

            // ── Widget helpers ─────────────────────────────────────
            const wGet = (name) => node.widgets?.find(w => w.name === name)?.value;
            const wSet = (name, val) => {
                const w = node.widgets?.find(x => x.name === name);
                if (w) w.value = val;
            };

            // ── Native widgets (source_folder + image with < > arrows) ─
            const sourceW = node.widgets?.find(w => w.name === "source_folder");
            const imageW = node.widgets?.find(w => w.name === "image");

            // ── Hook source_folder to refresh file list (like Load Image+) ─
            if (sourceW) {
                node._weSourceFolder = sourceW.value || "input";
                const origSfCb = sourceW.callback;
                sourceW.callback = async function (value) {
                    if (origSfCb) origSfCb.apply(this, arguments);
                    node._weSourceFolder = value || "input";
                    try {
                        const resp = await api.fetchApi(`/fbnodes/list-files?source=${encodeURIComponent(value)}`);
                        if (resp.ok) {
                            const data = await resp.json();
                            if (imageW) {
                                imageW.options.values = ["(none)", ...(data.files || [])];
                                imageW.value = "(none)";
                                if (imageW.callback) imageW.callback("(none)");
                            }
                        }
                    } catch (err) {
                        console.warn("[WE] Could not fetch file list:", err);
                    }
                    node.setDirtyCanvas(true);
                };
            }

            // ── Hook image widget callback for auto-extract + thumbnail ─
            if (imageW) {
                const origImgCb = imageW.callback;
                imageW.callback = function (value) {
                    if (origImgCb) origImgCb.apply(this, arguments);
                    onImageChange(value);
                };

                // Splice in Browse Files button right after image widget (like Load Image+)
                const imgIdx = node.widgets.indexOf(imageW);
                const browseBtn = {
                    type: "button",
                    name: "📁 Browse Files",
                    value: null,
                    callback: () => {
                        const cur = imageW.value === "(none)" ? null : imageW.value;
                        createFileBrowserModal(cur, (sel) => {
                            if (!imageW.options.values.includes(sel)) {
                                imageW.options.values.push(sel);
                            }
                            imageW.value = sel;
                            if (imageW.callback) imageW.callback(sel);
                            node.setDirtyCanvas(true);
                        }, node._weSourceFolder);
                    },
                    serialize: false,
                };
                node.widgets.splice(imgIdx + 1, 0, browseBtn);
                Object.defineProperty(browseBtn, "node", { value: node });
            }

            // ══════════════════════════════════════════════════════════
            // ── Build the DOM UI (thumbnail + sections) ──────────────
            // ══════════════════════════════════════════════════════════
            const root = makeEl("div", {
                display: "flex",
                flexDirection: "column",
                gap: "4px",
                padding: "6px",
                width: "100%",
                boxSizing: "border-box",
                fontFamily: "Inter, system-ui, -apple-system, sans-serif",
                overflow: "hidden",
            });
            forwardWheelToCanvas(root);

            // ── Thumbnail preview ────────────────────────────────────
            const thumbEl = makeEl("div", {
                width: "100%", minHeight: "4px",
                flexShrink: "0", textAlign: "center",
                overflow: "hidden",
            });
            root.appendChild(thumbEl);
            node._weThumbEl = thumbEl;

            // ── Extract button ───────────────────────────────────────
            root.appendChild(makeBtn("🔍 Extract Parameters", () => doExtract(node), {
                fontSize: "11px", padding: "4px 8px",
            }));

            // ── PROMPTS (positive open, negative closed) ─────────────
            const posSec = makeSection("POSITIVE PROMPT", false, 110, recalcHeight);
            const posBox = document.createElement("textarea");
            posBox.placeholder = "Positive prompt";
            posBox.rows = 5;
            Object.assign(posBox.style, {
                width: "100%", boxSizing: "border-box",
                background: C.bgInput, color: C.text,
                border: `1px solid ${C.border}`, borderRadius: "4px",
                fontSize: "12px", padding: "6px", resize: "none",
                fontFamily: "inherit", lineHeight: "1.4",
                maxHeight: "90px", overflow: "auto",
            });
            posBox.onchange = () => syncHidden(node);
            posSec._body.appendChild(posBox);
            root.appendChild(posSec);
            node._wePosBox = posBox;

            const negSec = makeSection("NEGATIVE PROMPT", true, 110, recalcHeight);
            const negBox = document.createElement("textarea");
            negBox.placeholder = "Negative prompt";
            negBox.rows = 5;
            Object.assign(negBox.style, {
                width: "100%", boxSizing: "border-box",
                background: C.bgInput, color: C.text,
                border: `1px solid ${C.border}`, borderRadius: "4px",
                fontSize: "12px", padding: "6px", resize: "none",
                fontFamily: "inherit", lineHeight: "1.4",
                maxHeight: "90px", overflow: "auto",
            });
            negBox.onchange = () => syncHidden(node);
            negSec._body.appendChild(negBox);
            root.appendChild(negSec);
            node._weNegBox = negBox;

            // ── Height recalculation ─────────────────────────────
            // Native widgets above DOM: title(26) + source_folder(26) + image(26) + browse(26) = 104
            const NATIVE_H = 104;
            const HEADER_H = 26;
            const EXTRACT_BTN_H = 30;  // extract button + gap
            const STATUS_H = 18;       // status bar at bottom
            const PADDING = 20;        // root padding + gaps
            const allSections = [];
            let _domH = 400;
            node._weThumbH = 0;

            function recalcHeight() {
                let h = EXTRACT_BTN_H + STATUS_H + PADDING;
                // Thumbnail (actual rendered height, max 200)
                h += node._weThumbH || 0;
                if (node._weThumbH > 0) h += 4; // gap
                // Sections: header always visible, body measured from DOM if open
                for (const sec of allSections) {
                    h += HEADER_H;
                    if (!sec._collapsed) {
                        // Measure actual body content height
                        const measured = sec._body.scrollHeight;
                        h += measured > 0 ? measured + 12 : sec._bodyH; // +12 for padding
                    }
                }
                _domH = h;
                if (node.size) {
                    node.setSize([node.size[0], _domH + NATIVE_H]);
                    node.setDirtyCanvas(true, true);
                }
            }
            node._weRecalc = recalcHeight;

            // ── Family state ─────────────────────────────────────────
            node._weFamily = null;

            const onFamilyChanged = async (familyKey) => {
                node._weFamily = familyKey;
                const fam = encodeURIComponent(familyKey || "");
                await Promise.all([
                    reloadGroupedSelect(node._weModelRow, async () => {
                        try {
                            const r = await fetch(`/workflow-extractor/list-models?family=${fam}`);
                            const d = await r.json(); return d.models || [];
                        } catch { return []; }
                    }, true),
                    reloadGroupedSelect(node._weVaeRow, async () => {
                        try {
                            const r = await fetch(`/workflow-extractor/list-vaes?family=${fam}`);
                            const d = await r.json();
                            return ["(from checkpoint)", ...(d.vaes || [])];
                        } catch { return []; }
                    }, true),
                    reloadGroupedSelect(node._weClipRow, async () => {
                        try {
                            const r = await fetch(`/workflow-extractor/list-clips?family=${fam}`);
                            const d = await r.json();
                            return ["(from checkpoint)", ...(d.clips || [])];
                        } catch { return []; }
                    }, true),
                ]);
                syncHidden(node);
            };

            // ── Model section ────────────────────────────────────────
            const modelSec = makeSection("MODEL", false, 56, recalcHeight);

            // Family type row
            const familyRow = makeEl("div", { ...ROW_STYLE });
            familyRow.appendChild(makeEl("span", {
                color: C.accent, width: LABEL_W, flexShrink: "0", fontWeight: "bold",
            }, "Type"));
            familyRow.appendChild(makeEl("span", { width: "14px", flexShrink: "0" }));
            const familySel = document.createElement("select");
            Object.assign(familySel.style, {
                ...INPUT_STYLE, color: C.accent, fontWeight: "bold",
            });
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
                    for (const [key, label] of Object.entries(families)) {
                        const o = document.createElement("option");
                        o.value = key; o.textContent = label;
                        familySel.appendChild(o);
                    }
                    familySel.value = curVal;
                } catch {}
            };
            familySel.onchange = () => onFamilyChanged(familySel.value);
            familyRow.appendChild(familySel);
            modelSec._body.appendChild(familyRow);
            node._weFamilySel = familySel;

            // Checkpoint row (grouped display)
            const modelRow = makeSelectRow("Checkpoint", "—",
                async () => {
                    try {
                        const fam = node._weFamily;
                        const url = fam
                            ? `/workflow-extractor/list-models?family=${encodeURIComponent(fam)}`
                            : `/workflow-extractor/list-models?ref=${encodeURIComponent(node._weModelRow?._getValue() || "")}`;
                        const r = await fetch(url);
                        const d = await r.json();
                        return d.models || [];
                    } catch { return []; }
                },
                (v) => { node._weOverrides.model_a = v; syncHidden(node); },
                true, // grouped
            );
            modelSec._body.appendChild(modelRow);
            root.appendChild(modelSec);
            node._weModelRow = modelRow;

            // When checkpoint reset is clicked, also restore original family
            modelRow._onReset = () => {
                const origFamily = node._weExtracted?.model_family || null;
                if (origFamily && origFamily !== node._weFamily) {
                    if (node._weFamilySel) node._weFamilySel.value = origFamily;
                    onFamilyChanged(origFamily);
                }
            };

            // ── LoRAs section ────────────────────────────────────────
            const loraSec = makeSection("LORAS", true, 50, recalcHeight);
            const loraContainer = makeEl("div", {
                display: "flex", flexWrap: "wrap", gap: "4px", padding: "4px 0",
            });
            loraSec._body.appendChild(loraContainer);
            root.appendChild(loraSec);
            node._weLoraContainer = loraContainer;

            // ── VAE / CLIP section ───────────────────────────────────
            const vcSec = makeSection("VAE / CLIP", true, 56, recalcHeight);
            const vaeRow = makeSelectRow("VAE", "—",
                async () => {
                    try {
                        const fam = encodeURIComponent(node._weFamily || "");
                        const r = await fetch(`/workflow-extractor/list-vaes?family=${fam}`);
                        const d = await r.json();
                        return ["(from checkpoint)", ...(d.vaes || [])];
                    } catch { return []; }
                },
                (v) => { node._weOverrides.vae = v; syncHidden(node); },
                true,
            );
            const clipRow = makeSelectRow("CLIP", "—",
                async () => {
                    try {
                        const fam = encodeURIComponent(node._weFamily || "");
                        const r = await fetch(`/workflow-extractor/list-clips?family=${fam}`);
                        const d = await r.json();
                        return ["(from checkpoint)", ...(d.clips || [])];
                    } catch { return []; }
                },
                (v) => { node._weOverrides.clip_names = [v]; syncHidden(node); },
                true,
            );
            vcSec._body.append(vaeRow, clipRow);
            root.appendChild(vcSec);
            node._weVaeRow = vaeRow;
            node._weClipRow = clipRow;

            // ── Sampler section ──────────────────────────────────────
            const sampSec = makeSection("SAMPLER", true, 148, recalcHeight);
            const _syncS = () => syncHidden(node);
            const sampRows = {
                steps:     makeInput("Steps",     "number", 20,      { min: 1, max: 200, step: 1 }, _syncS),
                cfg:       makeInput("CFG",       "number", 7.0,     { min: 0, max: 100, step: 0.5 }, _syncS),
                seed:      makeInput("Seed",      "number", 0,       { min: 0, step: 1 }, _syncS),
                sampler:   makeInput("Sampler",   "select", "euler", { options: SAMPLERS }, _syncS),
                scheduler: makeInput("Scheduler", "select", "normal",{ options: SCHEDULERS }, _syncS),
                denoise:   makeInput("Denoise",   "number", 1.0,     { min: 0, max: 1, step: 0.05 }, _syncS),
            };
            for (const row of Object.values(sampRows)) {
                sampSec._body.appendChild(row);
            }
            root.appendChild(sampSec);
            node._weSamplerRows = sampRows;

            // ── Resolution section ───────────────────────────────────
            const resSec = makeSection("RESOLUTION", true, 100, recalcHeight);
            const _syncR = () => syncHidden(node);
            const resRows = {
                width:  makeInput("Width",  "number", 512, { min: 64, max: 8192, step: 8 }, _syncR),
                height: makeInput("Height", "number", 512, { min: 64, max: 8192, step: 8 }, _syncR),
                batch:  makeInput("Batch",  "number", 1,   { min: 1, max: 128, step: 1 }, _syncR),
                length: makeInput("Length", "number", 81,  { min: 1, max: 1000, step: 1 }, _syncR),
            };
            resRows.length.style.display = "none";
            for (const row of Object.values(resRows)) {
                resSec._body.appendChild(row);
            }
            root.appendChild(resSec);
            node._weResRows = resRows;

            // ── Status (at bottom) ───────────────────────────────────
            const statusEl = makeEl("div", {
                fontSize: "11px", color: C.textMuted, textAlign: "center",
                padding: "1px 0", minHeight: "14px",
            }, "");
            root.appendChild(statusEl);
            node._weStatusEl = statusEl;

            // Register all sections for height tracking
            allSections.push(posSec, negSec, modelSec, loraSec, vcSec, sampSec, resSec);

            // Calculate initial height
            recalcHeight();

            // ══════════════════════════════════════════════════════════
            // ── Register the DOM widget ──────────────────────────────
            // ══════════════════════════════════════════════════════════
            const domW = node.addDOMWidget("we_ui", "div", root, {
                hideOnZoom: false,
                serialize: false,
            });

            domW.computeSize = function (nodeWidth) {
                return [nodeWidth, _domH];
            };

            const _origComputeSize = node.computeSize;
            node.computeSize = function () {
                const s = _origComputeSize?.apply(this, arguments) || [440, 500];
                s[1] = Math.max(s[1], _domH + NATIVE_H);
                return s;
            };

            // ── Only hide data widgets, keep source_folder + image + browse visible ─
            const KEEP_VISIBLE = new Set(["source_folder", "image", "📁 Browse Files"]);
            for (const w of (node.widgets || [])) {
                if (w === domW || KEEP_VISIBLE.has(w.name)) continue;
                w.computeSize = () => [0, -4];
                w.type = "converted-widget";
                w.hidden = true;
                w.draw = function () {};
                if (w.element) w.element.style.display = "none";
            }

            // Set size to match PM Advanced width (440)
            node.setSize([440, _domH + NATIVE_H]);

            // ── Zoom-aware font scaling for dropdown text ────────────
            applyZoomScaling(root);

            // ── Image-change handler ─────────────────────────────────
            function onImageChange(filename) {
                node._weLastImage = filename;
                loadThumbnail(node, filename);
                if (filename && filename !== "(none)") {
                    doExtract(node);
                }
            }

            // ── Initial load if image is already set ─────────────────
            const initImg = wGet("image");
            if (initImg && initImg !== "(none)" && initImg !== "") {
                setTimeout(() => onImageChange(initImg), 300);
            }

            return r;
        };

        // ── On configure (graph load / paste) ────────────────────────
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            origConfigure?.apply(this, arguments);
            const node = this;
            setTimeout(() => {
                // Re-hide only data widgets on configure
                const domW = node.widgets?.find(w => w.name === "we_ui");
                const KEEP_VISIBLE = new Set(["source_folder", "image", "📁 Browse Files"]);
                for (const w of (node.widgets || [])) {
                    if (w === domW || KEEP_VISIBLE.has(w.name)) continue;
                    w.computeSize = () => [0, -4];
                    w.type = "converted-widget";
                    w.hidden = true;
                    w.draw = function () {};
                    if (w.element) w.element.style.display = "none";
                }

                const wGet = (name) => node.widgets?.find(w => w.name === name)?.value;
                const src = wGet("source_folder");
                if (src) node._weSourceFolder = src;

                // Restore output folder file list if needed
                if (node._weSourceFolder === "output") {
                    const imageW = node.widgets?.find(w => w.name === "image");
                    api.fetchApi("/fbnodes/list-files?source=output").then(resp => {
                        if (resp.ok) return resp.json();
                    }).then(data => {
                        if (data?.files && imageW) {
                            const saved = imageW.value;
                            imageW.options.values = ["(none)", ...data.files];
                            if (saved && data.files.includes(saved)) imageW.value = saved;
                        }
                    }).catch(() => {});
                }

                const img = wGet("image");
                if (img && img !== "(none)" && img !== "") {
                    loadThumbnail(node, img);
                    // Save overrides before re-extracting (they persist in widget values)
                    const savedOv = wGet("override_data");
                    const savedLs = wGet("lora_state");
                    doExtract(node).then(() => {
                        applyOverrides(node, savedOv, savedLs);
                    });
                }
                node.setDirtyCanvas(true, true);
            }, 200);
        };
    },
});
