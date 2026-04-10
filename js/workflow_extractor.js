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
            console.log("[WorkflowGenerator] LoRA Manager preview integration enabled");
        }
    } catch (e) {
        console.log("[WorkflowGenerator] LoRA Manager preview not available:", e.message);
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
    {
        // Initial default option
        const defOpt = document.createElement("option");
        defOpt.value = "";
        defOpt.textContent = grouped ? "(none)" : "(Default)";
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
        // Always prepend default option first
        const defOpt = document.createElement("option");
        defOpt.value = ""; defOpt.textContent = grouped ? "(none)" : "(Default)";
        sel.appendChild(defOpt);
        if (grouped) {
            populateGroupedSelect(sel, options, true); // keepFirst=true preserves (none)/(Default)
        } else {
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
        // Always add a default first option
        const defOpt = document.createElement("option");
        defOpt.value = "";
        // For model rows (grouped): "(none)"; for VAE/CLIP rows (non-grouped): "(Default)"
        defOpt.textContent = grouped ? "(none)" : "(Default)";
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

// Helper to reload a grouped select from API
// For VAE/CLIP (non-grouped): prepends (Default) option with value="",
// and if recommendedValue is provided, auto-selects that file.
async function reloadGroupedSelect(row, fetchFn, grouped, recommendedValue) {
    const options = await fetchFn();
    const sel = row._sel;
    if (grouped) {
        sel.innerHTML = "";
        const noneOpt = document.createElement("option");
        noneOpt.value = ""; noneOpt.textContent = "(none)";
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
        // Auto-select recommended file if provided and present in list
        if (recommendedValue && options.includes(recommendedValue)) {
            sel.value = recommendedValue;
        } else {
            sel.value = "";
        }
    }
    // Reset lazy-load flag so next focus re-fetches with current family
    if (row._resetLoaded) row._resetLoaded();
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
    inp.oninput = () => {
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
    sinp.className = "strength-input";
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
        node._weHasWorkflow = true;
        node._weLoraState = {};
        node._weOverrides = {};
        updateUI(node);
        // Persist state to properties immediately so tab switches preserve it
        syncHidden(node);
        if (statusEl) { statusEl.textContent = ""; }
        node.setDirtyCanvas(true);
    } catch (err) {
        console.error("[WE] extract error", err);
        node._weHasWorkflow = false;
        if (statusEl) { statusEl.textContent = `❌ ${err.message}`; statusEl.style.color = C.error; }
        node.setDirtyCanvas(true);
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

    // Model A
    if (node._weModelRow?._setOriginal) {
        const name = d.model_a || "—";
        node._weModelRow._setOriginal(name, d.model_a_found !== false);
    }

    // Model B (always visible)
    if (node._weModelBRow) {
        node._weModelBRow.style.display = "flex";  // always visible
        if (d.model_b) {
            node._weModelBRow._setOriginal(d.model_b, d.model_b_found !== false);
        } else {
            node._weModelBRow._setOriginal("", false);  // shows "(none)"
        }
    }

    // LoRAs
    updateLoras(node);

    // VAE
    if (node._weVaeRow?._setOriginal) {
        const vn = d.vae?.name || "";
        const isDefault = !vn || vn.startsWith("(") || vn === "\u2014";
        node._weVaeRow._setOriginal(isDefault ? "" : vn, d.vae_found !== false);
    }

    // CLIP
    if (node._weClipRow?._setOriginal) {
        const firstName = (d.clip?.names || [])[0] || "";
        const isDefault = !firstName || firstName.startsWith("(") || firstName === "\u2014";
        node._weClipRow._setOriginal(isDefault ? "" : firstName);
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

// ─── LoRA stack card — identical structure/CSS to Prompt Manager Advanced ─────
function createLoraStackContainer(title, onResetStrength, onToggleAll) {
    // Outer card (matches PMA createLoraDisplayContainer)
    const container = document.createElement("div");
    Object.assign(container.style, {
        display: "flex",
        flexDirection: "column",
        gap: "4px",
        padding: "8px",
        backgroundColor: "rgba(40, 44, 52, 0.6)",
        borderRadius: "6px",
        width: "100%",
        boxSizing: "border-box",
        marginTop: "4px",
    });

    // Title bar
    const titleBar = document.createElement("div");
    Object.assign(titleBar.style, {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "4px",
        paddingBottom: "4px",
        borderBottom: "1px solid rgba(255,255,255,0.1)",
    });

    const titleLabel = document.createElement("span");
    titleLabel.textContent = title;
    Object.assign(titleLabel.style, {
        fontSize: "12px",
        fontWeight: "bold",
        color: "#aaa",
    });
    titleBar.appendChild(titleLabel);

    // Buttons
    const btnContainer = document.createElement("div");
    Object.assign(btnContainer.style, { display: "flex", gap: "4px" });

    const mkBtn = (text, fn) => {
        const b = document.createElement("button");
        b.textContent = text;
        Object.assign(b.style, {
            fontSize: "10px", padding: "2px 8px",
            backgroundColor: "#333", color: "#ccc",
            border: "1px solid #555", borderRadius: "6px", cursor: "pointer",
        });
        b.onmouseenter = () => b.style.backgroundColor = "#444";
        b.onmouseleave = () => b.style.backgroundColor = "#333";
        b.onclick = (e) => { e.stopPropagation(); fn(); };
        return b;
    };
    btnContainer.appendChild(mkBtn("Reset Strength", onResetStrength));
    btnContainer.appendChild(mkBtn("Toggle All", onToggleAll));
    titleBar.appendChild(btnContainer);
    container.appendChild(titleBar);

    // Tags container (matches PMA tagsContainer)
    const tagsContainer = document.createElement("div");
    tagsContainer.className = "lora-tags-container";
    Object.assign(tagsContainer.style, {
        display: "flex",
        flexWrap: "wrap",
        gap: "4px",
        minHeight: "30px",
    });
    container.appendChild(tagsContainer);

    // Expose the inner tags container for populating
    container._tagsContainer = tagsContainer;
    return container;
}

function updateLoras(node) {
    const containerA = node._weLoraAContainer;
    const containerB = node._weLoraBContainer;
    const labelA = node._weLoraALabel;
    const labelB = node._weLoraBLabel;
    if (!containerA) return;
    containerA.innerHTML = "";
    if (containerB) containerB.innerHTML = "";

    const d = node._weExtracted;
    if (!d) return;

    const lorasA = d.loras_a || [];
    const lorasB = d.loras_b || [];
    const hasBoth = lorasA.length > 0 && lorasB.length > 0;

    // Both stack cards always visible (matches PMA behaviour)
    if (labelA) labelA.style.display = "flex";
    if (labelB) labelB.style.display = "flex";
    if (containerB) containerB.style.display = "flex";

    const noLorasMsg = () => makeEl("div", {
        color: "rgba(200, 200, 200, 0.5)", fontStyle: "italic",
        fontSize: "11px", padding: "8px", width: "100%", textAlign: "center",
    }, "No LoRAs connected");

    if (!lorasA.length) containerA.appendChild(noLorasMsg());
    if (!lorasB.length && containerB) containerB.appendChild(noLorasMsg());
    if (!lorasA.length && !lorasB.length) return;

    // Populate a container with lora tags
    const populateStack = (container, loras, stackKey) => {
        for (const lora of loras) {
            const name = lora.name || "";
            const stateKey = stackKey ? `${stackKey}:${name}` : name;
            const avail = d.lora_availability?.[name] !== false;
            if (node._weLoraState[stateKey] === undefined) {
                node._weLoraState[stateKey] = { active: true, model_strength: lora.model_strength ?? 1.0, clip_strength: lora.clip_strength ?? 1.0 };
            }
            const st = node._weLoraState[stateKey];
            lora._active = st.active;
            lora.model_strength = st.model_strength;
            const tag = makeLoraTag(lora, avail,
                () => { st.active = !st.active; updateLoras(node); syncHidden(node); },
                (v) => { st.model_strength = v; st.clip_strength = v; syncHidden(node); },
            );
            container.appendChild(tag);
        }
    };

    populateStack(containerA, lorasA, hasBoth ? "a" : "");
    if (containerB && lorasB.length) {
        populateStack(containerB, lorasB, "b");
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
    if (node._weModelBRow?._getValue) {
        const v = node._weModelBRow._getValue();
        if (v && v !== "—") ov.model_b = v;
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

    // Save family selection
    if (node._weFamily) ov._family = node._weFamily;

    // Save section collapse states
    if (node._weSections) {
        const ss = {};
        for (const [key, sec] of Object.entries(node._weSections)) {
            ss[key] = !!sec._collapsed;
        }
        ov._section_states = ss;
    }

    wSet("override_data", JSON.stringify(ov));
    wSet("lora_state", JSON.stringify(node._weLoraState || {}));
    // Cache extracted data for tab-switch persistence (avoids API call on restore)
    if (node._weExtracted) {
        wSet("extracted_cache", JSON.stringify(node._weExtracted));
    }

    // Also persist to node.properties (survives tab switch reliably)
    node.properties ||= {};
    node.properties.we_override_data = JSON.stringify(ov);
    node.properties.we_lora_state = JSON.stringify(node._weLoraState || {});
    if (node._weExtracted) {
        node.properties.we_extracted_cache = JSON.stringify(node._weExtracted);
    }
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
        // User-selected override is always valid — clear any "not found" red
        sel.style.color = C.text;
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

    // Restore family selection
    if (ov._family && node._weFamilySel) {
        const sel = node._weFamilySel;
        if (![...sel.options].some(o => o.value === ov._family)) {
            const o = document.createElement("option");
            o.value = ov._family;
            o.textContent = ov._family;
            sel.appendChild(o);
        }
        sel.value = ov._family;
        node._weFamily = ov._family;
    }

    // Model A
    if (ov.model_a && ov.model_a !== (d.model_a || "—")) {
        applySelect(node._weModelRow, ov.model_a, true);
    }

    // Model B
    if (ov.model_b && ov.model_b !== (d.model_b || "—")) {
        applySelect(node._weModelBRow, ov.model_b, true);
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

    // Restore section collapse states
    if (ov._section_states && node._weSections) {
        for (const [key, collapsed] of Object.entries(ov._section_states)) {
            const sec = node._weSections[key];
            if (!sec) continue;
            const isCurrentlyCollapsed = sec._collapsed;
            if (collapsed !== isCurrentlyCollapsed) {
                // Toggle it
                sec._body.style.display = collapsed ? "none" : "block";
                const arrow = sec.querySelector("span");
                if (arrow) arrow.textContent = collapsed ? "▶" : "▼";
                sec._collapsed = collapsed;
            }
        }
        if (node._weRecalc) node._weRecalc();
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

// ─── Preview modal helpers ───────────────────────────────────────────────────
function _showImageModal(filename, url) {
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", inset: "0", background: "rgba(0,0,0,0.85)",
        zIndex: "10000", display: "flex", alignItems: "center", justifyContent: "center",
    });
    const img = document.createElement("img");
    img.src = url;
    Object.assign(img.style, { maxWidth: "90%", maxHeight: "90vh", borderRadius: "8px", boxShadow: "0 10px 40px rgba(0,0,0,0.5)" });
    overlay.appendChild(img);
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
    document.addEventListener("keydown", function esc(e) { if (e.key === "Escape") { overlay.remove(); document.removeEventListener("keydown", esc); } });
    document.body.appendChild(overlay);
}

function _showVideoModal(filename, url) {
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", inset: "0", background: "rgba(0,0,0,0.85)",
        zIndex: "10000", display: "flex", alignItems: "center", justifyContent: "center",
    });
    const video = document.createElement("video");
    video.src = url;
    video.controls = true;
    video.autoplay = true;
    video.loop = true;
    Object.assign(video.style, { maxWidth: "90%", maxHeight: "90vh", borderRadius: "8px" });
    video.onerror = () => { video.remove(); overlay.appendChild(document.createTextNode("Cannot play this video format in browser.")); };
    overlay.appendChild(video);
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
    document.addEventListener("keydown", function esc(e) { if (e.key === "Escape") { overlay.remove(); document.removeEventListener("keydown", esc); } });
    document.body.appendChild(overlay);
}

// ─── Thumbnail helper ────────────────────────────────────────────────────────
const VIDEO_EXTENSIONS_THUMB = ['mp4', 'webm', 'mov', 'avi', 'mkv', 'm4v'];

function loadThumbnail(node, filename) {
    const thumbEl = node._weThumbEl;
    if (!thumbEl) return;
    if (!filename || filename === "(none)") {
        thumbEl.innerHTML = "";
        node._weThumbH = 0;
        node._weThumbFilename = null;
        thumbEl.style.display = "none";
        if (node._weRecalc) node._weRecalc();
        return;
    }
    node._weThumbFilename = filename;
    const src = node._weSourceFolder || "input";
    let actualFilename = filename;
    let subfolder = "";
    if (filename.includes("/")) {
        const lastSlash = filename.lastIndexOf("/");
        subfolder = filename.substring(0, lastSlash);
        actualFilename = filename.substring(lastSlash + 1);
    }
    const ext = actualFilename.split('.').pop().toLowerCase();
    const isVideo = VIDEO_EXTENSIONS_THUMB.includes(ext);

    thumbEl.innerHTML = "";

    if (isVideo) {
        // ── Video: try browser canvas extraction first, fall back to server endpoint ──
        let url = `/view?filename=${encodeURIComponent(actualFilename)}&type=${encodeURIComponent(src)}`;
        if (subfolder) url += `&subfolder=${encodeURIComponent(subfolder)}`;
        url += `&t=${Date.now()}`;

        const video = document.createElement("video");
        video.crossOrigin = "anonymous";
        video.preload = "auto";
        video.muted = true;
        video.playsInline = true;
        video.style.cssText = "position:fixed;top:-9999px;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none;";

        let cleaned = false;
        const cleanupVideo = () => {
            if (cleaned) return;
            cleaned = true;
            video.onloadedmetadata = null;
            video.onseeked = null;
            video.onerror = null;
            try { video.src = ""; video.load(); } catch(e) {}
            if (video.parentNode) video.parentNode.removeChild(video);
        };

        const showServerFrame = () => {
            cleanupVideo();
            // Fall back to Python /workflow-extractor/video-frame endpoint
            const frameUrl = `/workflow-extractor/video-frame?filename=${encodeURIComponent(filename)}&source=${src}&position=0`;
            const img = document.createElement("img");
            Object.assign(img.style, {
                width: "100%", maxHeight: "200px", objectFit: "contain",
                borderRadius: "4px", display: "block", margin: "0 auto",
            });
            img.onload = () => {
                thumbEl.innerHTML = "";
                thumbEl.appendChild(img);
                thumbEl.style.display = "block";
                node._weThumbH = Math.min(img.naturalHeight, 200);
                if (node._weRecalc) node._weRecalc();
            };
            img.onerror = () => {
                thumbEl.innerHTML = "";
                thumbEl.style.display = "none";
                node._weThumbH = 0;
                if (node._weRecalc) node._weRecalc();
            };
            img.src = frameUrl;
        };

        video.onloadedmetadata = () => {
            video.currentTime = 0.01; // seek to first frame
        };

        video.onseeked = () => {
            try {
                const canvas = document.createElement("canvas");
                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 360;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                cleanupVideo();
                const img = document.createElement("img");
                Object.assign(img.style, {
                    width: "100%", maxHeight: "200px", objectFit: "contain",
                    borderRadius: "4px", display: "block", margin: "0 auto",
                });
                img.onload = () => {
                    thumbEl.innerHTML = "";
                    thumbEl.appendChild(img);
                    thumbEl.style.display = "block";
                    node._weThumbH = Math.min(img.naturalHeight, 200);
                    if (node._weRecalc) node._weRecalc();
                };
                img.onerror = () => { thumbEl.innerHTML = ""; thumbEl.style.display = "none"; node._weThumbH = 0; if (node._weRecalc) node._weRecalc(); };
                img.src = canvas.toDataURL("image/png");
            } catch (e) {
                showServerFrame();
            }
        };

        video.onerror = () => { showServerFrame(); };

        // Timeout fallback (5s) in case onseeked never fires
        setTimeout(() => { if (!cleaned) showServerFrame(); }, 5000);

        document.body.appendChild(video);
        video.src = url;
    } else {
        // ── Image: original path ──
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
            thumbEl.innerHTML = "";
            thumbEl.appendChild(img);
            thumbEl.style.display = "block";
            node._weThumbH = Math.min(img.naturalHeight, 200);
            if (node._weRecalc) node._weRecalc();
        };
        img.onerror = () => {
            thumbEl.innerHTML = "";
            thumbEl.style.display = "none";
            node._weThumbH = 0;
            if (node._weRecalc) node._weRecalc();
        };
        thumbEl.innerHTML = "";
        thumbEl.appendChild(img);
        thumbEl.style.display = "block";
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// ─── Main extension ─────────────────────────────────────────────────────────
// ═════════════════════════════════════════════════════════════════════════════
app.registerExtension({
    name: "WorkflowGenerator",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WorkflowGenerator") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated?.apply(this, arguments);
            const node = this;

            // ── State ────────────────────────────────────────────────
            node._weExtracted = null;
            node._weHasWorkflow = false;
            node._weLoraState = {};
            node._weOverrides = {};
            node._weLastImage = null;
            node._weSourceFolder = "input";
            node._weSections = {};  // section refs for collapse state
            this.serialize_widgets = true;

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
                        const resp = await api.fetchApi(`/workflow-extractor/list-files?source=${encodeURIComponent(value)}`);
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
                marginTop: "-10px",   // cancel the ~10px ComfyUI injects above every DOM widget
                width: "100%",
                boxSizing: "border-box",
                fontFamily: "Inter, system-ui, -apple-system, sans-serif",
                overflow: "hidden",
            });
            forwardWheelToCanvas(root);
            node._weRoot = root;

            // ── Thumbnail preview ────────────────────────────────────
            const thumbEl = makeEl("div", {
                width: "100%",
                flexShrink: "0", textAlign: "center",
                overflow: "hidden",
                display: "none",
            });
            root.appendChild(thumbEl);
            node._weThumbEl = thumbEl;
            node._weThumbFilename = null;

            // ── Extract button ───────────────────────────────────────
            root.appendChild(makeBtn("🔍 Extract Parameters", () => doExtract(node), {
                fontSize: "11px", padding: "4px 8px",
            }));

            // ── PROMPTS (positive open, negative closed) ─────────────
            const posSec = makeSection("POSITIVE PROMPT", false, 110, () => { recalcHeight(); syncHidden(node); });
            node._weSections.positive = posSec;
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
            posBox.oninput = () => syncHidden(node);
            posSec._body.appendChild(posBox);
            root.appendChild(posSec);
            node._wePosBox = posBox;

            const negSec = makeSection("NEGATIVE PROMPT", true, 110, () => { recalcHeight(); syncHidden(node); });
            node._weSections.negative = negSec;
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
            negBox.oninput = () => syncHidden(node);
            negSec._body.appendChild(negBox);
            root.appendChild(negSec);
            node._weNegBox = negBox;

            // ── Height recalculation ─────────────────────────────
            // Native widgets above DOM: title(26) + source_folder(26) + image(26) + browse(26) = 104
            const NATIVE_H = 108;  // title(30) + source_folder(26) + image(26) + browse(26)
            const HEADER_H = 26;
            const EXTRACT_BTN_H = 30;  // extract button + gap
            const STATUS_H = 18;       // status bar at bottom
            const PADDING = 16;        // root padding top+bottom=12 + small buffer
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
                        h += Math.min(measured > 4 ? measured + 6 : sec._bodyH, 600);
                    }
                }
                _domH = h;
                if (node.size) {
                    node.setSize([node.size[0], _domH + NATIVE_H]);
                    node.setDirtyCanvas(true, true);
                }
            }
            node._weRecalc = recalcHeight;
            requestAnimationFrame(() => requestAnimationFrame(() => recalcHeight()));

            // ── Family state ─────────────────────────────────────────
            node._weFamily = null;

            const onFamilyChanged = async (familyKey) => {
                node._weFamily = familyKey;
                const fam = encodeURIComponent(familyKey || "");
                const fetchModelsForFamily = async () => {
                    try {
                        const r = await fetch(`/workflow-extractor/list-models?family=${fam}`);
                        const d = await r.json(); return d.models || [];
                    } catch { return []; }
                };
                // VAE: fetch with recommended, then reload select and auto-select best match
                const reloadVae = async () => {
                    try {
                        const r = await fetch(`/workflow-extractor/list-vaes?family=${fam}`);
                        const d = await r.json();
                        await reloadGroupedSelect(node._weVaeRow, async () => d.vaes || [], false, d.recommended || null);
                    } catch { await reloadGroupedSelect(node._weVaeRow, async () => [], false, null); }
                };
                // CLIP: fetch with recommended, then reload select and auto-select best match
                const reloadClip = async () => {
                    try {
                        const r = await fetch(`/workflow-extractor/list-clips?family=${fam}`);
                        const d = await r.json();
                        await reloadGroupedSelect(node._weClipRow, async () => d.clips || [], false, d.recommended || null);
                    } catch { await reloadGroupedSelect(node._weClipRow, async () => [], false, null); }
                };
                const reloads = [
                    reloadGroupedSelect(node._weModelRow, fetchModelsForFamily, true),
                    reloadVae(),
                    reloadClip(),
                ];
                // Always reload model B (now always visible)
                reloads.push(reloadGroupedSelect(node._weModelBRow, fetchModelsForFamily, true));
                await Promise.all(reloads);
                syncHidden(node);
            };

            // ── Model section ────────────────────────────────────────
            const modelSec = makeSection("MODEL", false, 56, () => { recalcHeight(); syncHidden(node); });
            node._weSections.model = modelSec;

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

            // Fetch models helper (shared by A and B rows)
            const fetchModels = async () => {
                try {
                    const fam = node._weFamily;
                    const url = fam
                        ? `/workflow-extractor/list-models?family=${encodeURIComponent(fam)}`
                        : `/workflow-extractor/list-models?ref=${encodeURIComponent(node._weModelRow?._getValue() || "")}`;
                    const r = await fetch(url);
                    const d = await r.json();
                    return d.models || [];
                } catch { return []; }
            };

            // Checkpoint A row (grouped display)
            const modelRow = makeSelectRow("Model A", "",
                fetchModels,
                (v) => { node._weOverrides.model_a = v; syncHidden(node); },
                true, // grouped
            );
            modelSec._body.appendChild(modelRow);
            node._weModelRow = modelRow;

            // Checkpoint B row (grouped display, hidden when no model_b)
            const modelBRow = makeSelectRow("Model B", "",
                fetchModels,
                (v) => { node._weOverrides.model_b = v; syncHidden(node); },
                true, // grouped
            );
            modelSec._body.appendChild(modelBRow);
            node._weModelBRow = modelBRow;

            root.appendChild(modelSec);

            // When checkpoint A reset is clicked, also restore original family
            modelRow._onReset = () => {
                const origFamily = node._weExtracted?.model_family || null;
                if (origFamily && origFamily !== node._weFamily) {
                    if (node._weFamilySel) node._weFamilySel.value = origFamily;
                    onFamilyChanged(origFamily);
                }
            };

            // ── LoRAs section ────────────────────────────────────────
            const loraSec = makeSection("LORAS", true, 140, () => { recalcHeight(); syncHidden(node); });
            node._weSections.loras = loraSec;

            // ── Stack A card (matches PMA createLoraDisplayContainer) ──
            const loraACard = createLoraStackContainer(
                "LoRA Stack A",
                () => {
                    // Reset Strength A
                    const d = node._weExtracted;
                    const lorasA = d?.loras_a || [];
                    const hasBoth = lorasA.length > 0 && (d?.loras_b || []).length > 0;
                    for (const lora of lorasA) {
                        const k = hasBoth ? `a:${lora.name}` : lora.name;
                        if (node._weLoraState[k]) {
                            node._weLoraState[k].model_strength = lora.model_strength ?? 1.0;
                            node._weLoraState[k].clip_strength  = lora.clip_strength  ?? 1.0;
                        }
                    }
                    updateLoras(node); syncHidden(node);
                },
                () => {
                    // Toggle All A
                    const d = node._weExtracted;
                    const lorasA = d?.loras_a || [];
                    const hasBoth = lorasA.length > 0 && (d?.loras_b || []).length > 0;
                    const anyActive = lorasA.some(l => {
                        const k = hasBoth ? `a:${l.name}` : l.name;
                        return node._weLoraState[k]?.active !== false;
                    });
                    for (const lora of lorasA) {
                        const k = hasBoth ? `a:${lora.name}` : lora.name;
                        if (!node._weLoraState[k]) node._weLoraState[k] = { active: true, model_strength: 1.0, clip_strength: 1.0 };
                        node._weLoraState[k].active = !anyActive;
                    }
                    updateLoras(node); syncHidden(node);
                },
            );
            loraSec._body.appendChild(loraACard);
            node._weLoraAContainer = loraACard._tagsContainer;
            node._weLoraALabel = loraACard;  // compat ref

            // ── Stack B card ──────────────────────────────────────────
            const loraBCard = createLoraStackContainer(
                "LoRA Stack B",
                () => {
                    // Reset Strength B
                    const d = node._weExtracted;
                    const lorasB = d?.loras_b || [];
                    for (const lora of lorasB) {
                        const k = `b:${lora.name}`;
                        if (node._weLoraState[k]) {
                            node._weLoraState[k].model_strength = lora.model_strength ?? 1.0;
                            node._weLoraState[k].clip_strength  = lora.clip_strength  ?? 1.0;
                        }
                    }
                    updateLoras(node); syncHidden(node);
                },
                () => {
                    // Toggle All B
                    const d = node._weExtracted;
                    const lorasB = d?.loras_b || [];
                    const anyActive = lorasB.some(l => node._weLoraState[`b:${l.name}`]?.active !== false);
                    for (const lora of lorasB) {
                        const k = `b:${lora.name}`;
                        if (!node._weLoraState[k]) node._weLoraState[k] = { active: true, model_strength: 1.0, clip_strength: 1.0 };
                        node._weLoraState[k].active = !anyActive;
                    }
                    updateLoras(node); syncHidden(node);
                },
            );
            loraSec._body.appendChild(loraBCard);
            node._weLoraBContainer = loraBCard._tagsContainer;
            node._weLoraBLabel = loraBCard;  // compat ref

            root.appendChild(loraSec);
            node._weLoraContainer = loraACard._tagsContainer; // backward compat

            // ── VAE / CLIP section ───────────────────────────────────
            const vcSec = makeSection("VAE / CLIP", true, 56, () => { recalcHeight(); syncHidden(node); });
            node._weSections.vae_clip = vcSec;
            const vaeRow = makeSelectRow("VAE", "",
                async () => {
                    try {
                        const fam = encodeURIComponent(node._weFamily || "");
                        const r = await fetch(`/workflow-extractor/list-vaes?family=${fam}`);
                        const d = await r.json();
                        return d.vaes || [];
                    } catch { return []; }
                },
                (v) => { node._weOverrides.vae = v; syncHidden(node); },
                false,
            );
            const clipRow = makeSelectRow("CLIP", "",
                async () => {
                    try {
                        const fam = encodeURIComponent(node._weFamily || "");
                        const r = await fetch(`/workflow-extractor/list-clips?family=${fam}`);
                        const d = await r.json();
                        return d.clips || [];
                    } catch { return []; }
                },
                (v) => { node._weOverrides.clip_names = [v]; syncHidden(node); },
                false,
            );
            vcSec._body.append(vaeRow, clipRow);
            root.appendChild(vcSec);
            node._weVaeRow = vaeRow;
            node._weClipRow = clipRow;

            // ── Sampler section ──────────────────────────────────────
            const sampSec = makeSection("SAMPLER", true, 148, () => { recalcHeight(); syncHidden(node); });
            node._weSections.sampler = sampSec;
            const _syncS = () => syncHidden(node);
            const sampRows = {
                steps:     makeInput("Steps",     "number", 20,      { min: 1, max: 200, step: 1 }, _syncS),
                cfg:       makeInput("CFG",       "number", 7.0,     { min: 0, max: 100, step: 0.5 }, _syncS),
                seed:      makeInput("Seed",      "number", 0,       { min: 0, step: 1 }, _syncS),
                sampler:   makeInput("Sampler",   "select", "euler", { options: SAMPLERS }, _syncS),
                scheduler: makeInput("Scheduler", "select", "normal",{ options: SCHEDULERS }, _syncS),
            };
            for (const row of Object.values(sampRows)) {
                sampSec._body.appendChild(row);
            }
            root.appendChild(sampSec);
            node._weSamplerRows = sampRows;

            // ── Resolution section ───────────────────────────────────
            const resSec = makeSection("RESOLUTION", true, 100, () => { recalcHeight(); syncHidden(node); });
            node._weSections.resolution = resSec;
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

            // ── Preview arrow + workflow status dot in title bar ─────────────────
            const _origDrawFg = node.onDrawForeground;
            node.onDrawForeground = function(ctx) {
                _origDrawFg?.apply(this, arguments);
                const titleH = LiteGraph.NODE_TITLE_HEIGHT || 30;
                const centerY = -(titleH / 2);
                if (node.flags && node.flags.collapsed) return;

                // ── Green/red dot: always visible once an image is selected ──
                const imageWidget = node.widgets?.find(w => w.name === "image");
                const currentFile = imageWidget?.value;
                if (currentFile && currentFile !== "(none)") {
                    const dotRadius = 7;
                    const dotX = node.size[0] - dotRadius - 8;
                    ctx.beginPath();
                    ctx.arc(dotX, centerY, dotRadius, 0, Math.PI * 2);
                    ctx.fillStyle = node._weHasWorkflow ? "#00ff00" : "#ff3333";
                    ctx.fill();
                    ctx.strokeStyle = node._weHasWorkflow ? "#054405" : "#550505";
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }

                // ── Preview triangle: only when thumbnail is loaded ──
                if (!node._weThumbFilename) {
                    node._wePreviewBounds = null;
                    return;
                }
                const triSize = 8;
                const dotSlot = (currentFile && currentFile !== "(none)") ? 24 : 0; // shift left if dot is shown
                const playX = node.size[0] - triSize - 12 - dotSlot;
                ctx.beginPath();
                ctx.moveTo(playX - triSize, centerY - triSize);
                ctx.lineTo(playX - triSize, centerY + triSize);
                ctx.lineTo(playX + triSize, centerY);
                ctx.closePath();
                ctx.fillStyle = node._weHoverPreview ? "#ffffff" : "rgba(255,255,255,0.7)";
                ctx.fill();
                node._wePreviewBounds = { x: playX - triSize - 3, y: centerY - triSize - 3, w: triSize * 2 + 6, h: triSize * 2 + 6 };
            };

            const _origMouseMove = node.onMouseMove;
            node.onMouseMove = function(e, localPos, canvas) {
                const res = _origMouseMove?.apply(this, arguments);
                if (node._wePreviewBounds) {
                    const b = node._wePreviewBounds;
                    const hit = localPos[0] >= b.x && localPos[0] <= b.x + b.w &&
                                localPos[1] >= b.y && localPos[1] <= b.y + b.h;
                    if (hit !== node._weHoverPreview) {
                        node._weHoverPreview = hit;
                        if (hit) canvas.canvas.style.cursor = "pointer";
                        else canvas.canvas.style.cursor = "";
                        node.setDirtyCanvas(true);
                    }
                }
                return res;
            };

            const _origMouseDown = node.onMouseDown;
            node.onMouseDown = function(e, localPos, canvas) {
                if (node._wePreviewBounds && node._weThumbFilename) {
                    const b = node._wePreviewBounds;
                    if (localPos[0] >= b.x && localPos[0] <= b.x + b.w &&
                        localPos[1] >= b.y && localPos[1] <= b.y + b.h) {
                        // Open preview — video or image
                        const src = node._weSourceFolder || "input";
                        const fname = node._weThumbFilename;
                        const ext = fname.split('.').pop().toLowerCase();
                        const isVid = VIDEO_EXTENSIONS_THUMB.includes(ext);
                        // Build preview URL
                        let af = fname, sf = "";
                        if (fname.includes("/")) { const sl = fname.lastIndexOf("/"); sf = fname.substring(0, sl); af = fname.substring(sl + 1); }
                        let previewUrl = `/view?filename=${encodeURIComponent(af)}&type=${encodeURIComponent(src)}`;
                        if (sf) previewUrl += `&subfolder=${encodeURIComponent(sf)}`;

                        if (isVid) {
                            _showVideoModal(fname, previewUrl);
                        } else {
                            _showImageModal(fname, previewUrl);
                        }
                        return true;
                    }
                }
                return _origMouseDown?.apply(this, arguments);
            };

            const _origComputeSize = node.computeSize;
            node.computeSize = function () {
                const s = _origComputeSize?.apply(this, arguments) || [450, 500];
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
            node.setSize([450, _domH + NATIVE_H]);

            // ── Zoom-aware font scaling for dropdown text ────────────
            applyZoomScaling(root);

            // ── Image-change handler (user-initiated only) ───────────
            function onImageChange(filename) {
                node._weLastImage = filename;
                loadThumbnail(node, filename);
                if (filename && filename !== "(none)") {
                    doExtract(node);
                }
            }

            // ── Initial load if image is already set ─────────────────
            // Only fires on FRESH node creation. On tab return / workflow
            // load, _configuredFromWorkflow is set synchronously by
            // onConfigure before this setTimeout fires.
            const initImg = wGet("image");
            if (initImg && initImg !== "(none)" && initImg !== "") {
                setTimeout(() => {
                    if (!node._configuredFromWorkflow) onImageChange(initImg);
                }, 300);
            }

            return r;
        };

        // ── On configure (graph load / paste / tab return) ───────────
        // Runs SYNCHRONOUSLY right after onNodeCreated. All DOM elements
        // already exist. Restore state from node.properties (reliable,
        // directly serialized in workflow JSON). No API calls, no
        // setTimeout, no thumbnail reload.
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            origConfigure?.apply(this, arguments);
            const node = this;

            // Permanent flag — blocks initImg setTimeout from re-extracting
            node._configuredFromWorkflow = true;

            // Re-hide data widgets (DOM was rebuilt by onNodeCreated)
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

            // Read saved state from node.properties (set by syncHidden)
            const props = node.properties || {};
            const savedOv = props.we_override_data;
            const savedLs = props.we_lora_state;
            const savedCache = props.we_extracted_cache;

            const wGet = (name) => node.widgets?.find(w => w.name === name)?.value;
            const src = wGet("source_folder");
            if (src) node._weSourceFolder = src;

            // Restore extracted data from cache (no network call)
            let cached = null;
            try { cached = JSON.parse(savedCache || "{}"); } catch { cached = null; }
            const hasCache = cached && Object.keys(cached).length > 0;

            if (hasCache) {
                node._weExtracted = cached;
                node._weHasWorkflow = true;  // restored from cache = workflow was found
                node._weLoraState = {};
                node._weOverrides = {};
                updateUI(node);
                applyOverrides(node, savedOv, savedLs);
            }

            // Restore thumbnail — reload from disk on page refresh, skip if already showing
            const img = wGet("image");
            if (img && img !== "(none)") {
                const alreadyLoaded = node._weThumbFilename === img && node._weThumbH > 0;
                node._weThumbFilename = img;
                if (!alreadyLoaded) {
                    setTimeout(() => loadThumbnail(node, img), 150);
                }
            }

            // Restore output folder file list (async, doesn't touch state)
            if (node._weSourceFolder === "output") {
                const imageW = node.widgets?.find(w => w.name === "image");
                api.fetchApi("/workflow-extractor/list-files?source=output").then(resp => {
                    if (resp.ok) return resp.json();
                }).then(data => {
                    if (data?.files && imageW) {
                        const saved = imageW.value;
                        imageW.options.values = ["(none)", ...data.files];
                        if (saved && data.files.includes(saved)) imageW.value = saved;
                    }
                }).catch(() => {});
            }

            // Recalculate DOM height and restore node size
            if (node._weRecalc) node._weRecalc();
            node.setDirtyCanvas(true, true);
        };
    },
});
