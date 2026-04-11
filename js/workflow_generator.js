/**
 * Workflow Generator - Full DOM-based UI
 *
 * Layout order: Resolution -> Model/VAE/CLIP -> Prompts -> Sampler -> LoRAs
 * Accepts workflow_data input (from PromptExtractor) + optional lora_stack inputs.
 * "use_workflow_data" toggle: ON -> populate & freeze fields; OFF -> editable (keeps last values).
 * "use_lora_input" toggle: ON -> display connected lora stacks; OFF -> manual/extracted only.
 * Non-WAN families show single "Model" / "LoRA Stack" labels; WAN shows A/B.
 * WAN family shows Frames input in Resolution section.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// --- LoRA Manager preview tooltip (lazy-loaded) ---
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
        }
    } catch (e) {
        loraManagerAvailable = false;
    }
    return loraManagerPreviewTooltip;
}
getLoraManagerPreviewTooltip();

// --- Forward wheel to canvas for zoom ---
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

// --- Larger dropdown text via CSS injection ---
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

// --- Colour palette ---
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

// --- Tiny helpers ---
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

// --- Section builder (always expanded) ---
function makeSection(title, startOpen = true) {
    const wrap = makeEl("div", {
        borderRadius: "6px", overflow: "hidden", marginTop: "2px",
        backgroundColor: C.bgCard, flexShrink: "0",
    });
    const header = makeEl("div", {
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "4px 8px",
        fontSize: "11px", fontWeight: "600", color: "#aaa",
        userSelect: "none", cursor: "pointer",
        borderBottom: "1px solid rgba(255,255,255,0.1)",
    });
    const label = makeEl("span", {}, title);
    const chevron = makeEl("span", {
        fontSize: "10px", color: C.textMuted, transition: "transform 0.15s",
        marginLeft: "4px", flexShrink: "0",
    }, startOpen ? "\u25BC" : "\u25B6");
    header.append(label, chevron);
    const body = makeEl("div", {
        padding: "4px 8px",
        display: startOpen ? "" : "none",
    });
    // Toggle body on header click — dirty canvas so computeSize is re-queried.
    header.addEventListener("click", () => {
        const opening = body.style.display === "none";
        body.style.display = opening ? "" : "none";
        chevron.textContent = opening ? "\u25BC" : "\u25B6";
        app.graph.setDirtyCanvas(true, true);
    });
    wrap._titleLabel = label;
    wrap._chevron = chevron;
    wrap.append(header, body);
    wrap._body = body;
    wrap._expand = () => {
        body.style.display = "";
        chevron.textContent = "\u25BC";
        app.graph.setDirtyCanvas(true, true);
    };
    wrap._collapse = () => {
        body.style.display = "none";
        chevron.textContent = "\u25B6";
        app.graph.setDirtyCanvas(true, true);
    };
    return wrap;
}

// --- Shared layout constants ---
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
    const btn = makeEl("span", { ...RESET_STYLE }, "\u21BA");
    btn.title = "Reset to extracted value";
    btn.onclick = (e) => { e.stopPropagation(); onReset(); };
    return btn;
}

// --- Clean model name for display ---
function cleanModelName(fullPath) {
    let name = fullPath;
    const extIdx = name.lastIndexOf(".");
    if (extIdx > 0) name = name.substring(0, extIdx);
    const slashIdx = Math.max(name.lastIndexOf("/"), name.lastIndexOf("\\"));
    if (slashIdx >= 0) name = name.substring(slashIdx + 1);
    return name;
}

// --- Group models by folder path for dropdown ---
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
            const display = cleanModelName(m);
            if (!groups.has(groupLabel)) groups.set(groupLabel, []);
            groups.get(groupLabel).push({ value: m, display });
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
        const defOpt = document.createElement("option");
        defOpt.value = ""; defOpt.textContent = grouped ? "(none)" : "(Default)";
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
        if (recommendedValue && options.includes(recommendedValue)) {
            sel.value = recommendedValue;
        } else {
            sel.value = "";
        }
    }
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
    row._label = row.querySelector("span");
    row._resetBtn = resetBtn;
    row._setOriginal = (v) => {
        _origVal = v;
        inp.value = v;
        resetBtn.style.visibility = "hidden";
    };
    return row;
}

// --- LoRA tag builder ---
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
        const warn = makeEl("span", { fontSize: "11px", marginRight: "-2px" }, "\u26A0\uFE0F");
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
        if (!isNaN(v)) { if (onStrength) onStrength(v); } else { sinp.value = str.toFixed(2); }
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
            ? `${name}\n\u26A0\uFE0F NOT FOUND - This LoRA is missing from your system`
            : `${name}\nStrength: ${str.toFixed(2)}\nClick to toggle on/off`;
    }

    return tag;
}

// --- LoRA stack card ---
function createLoraStackContainer(title, onResetStrength, onToggleAll) {
    const container = document.createElement("div");
    Object.assign(container.style, {
        display: "flex", flexDirection: "column", gap: "4px",
        padding: "8px", backgroundColor: "rgba(40, 44, 52, 0.6)",
        borderRadius: "6px", width: "100%", boxSizing: "border-box", marginTop: "4px",
    });
    const titleBar = document.createElement("div");
    Object.assign(titleBar.style, {
        display: "flex", justifyContent: "space-between", alignItems: "center",
        marginBottom: "4px", paddingBottom: "4px",
        borderBottom: "1px solid rgba(255,255,255,0.1)",
    });
    const titleLabel = document.createElement("span");
    titleLabel.textContent = title;
    Object.assign(titleLabel.style, { fontSize: "12px", fontWeight: "bold", color: "#aaa" });
    titleBar.appendChild(titleLabel);
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
    const tagsContainer = document.createElement("div");
    tagsContainer.className = "lora-tags-container";
    Object.assign(tagsContainer.style, {
        display: "flex", flexWrap: "wrap", gap: "4px", minHeight: "30px",
    });
    container.appendChild(tagsContainer);
    container._tagsContainer = tagsContainer;
    container._titleLabel = titleLabel;
    return container;
}

// --- Samplers / Schedulers constants ---
const SAMPLERS = [
    "euler","euler_cfg_pp","euler_ancestral","euler_ancestral_cfg_pp",
    "heun","heunpp2","dpm_2","dpm_2_ancestral","lms","dpm_fast",
    "dpm_adaptive","dpmpp_2s_ancestral","dpmpp_sde","dpmpp_sde_gpu",
    "dpmpp_2m","dpmpp_2m_sde","dpmpp_2m_sde_gpu","dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu","ddpm","lcm","ipndm","ipndm_v","deis","ddim",
    "uni_pc","uni_pc_bh2",
];
const SCHEDULERS = [
    "normal","karras","exponential","sgm_uniform","simple","ddim_uniform","beta",
];
const CONTROL_MODES = ["fixed", "increment", "decrement", "randomize"];

// --- Separator helper ---
function makeSeparator() {
    return makeEl("div", {
        height: "1px", background: "rgba(255,255,255,0.08)",
        margin: "6px 0", width: "100%",
    });
}



// ============================================================
// --- Update UI from workflow_data / extracted data ---
// ============================================================
async function updateUI(node) {
    const d = node._weExtracted;
    if (!d) return;

    // Family — must be set FIRST and dropdowns reloaded before setting
    // model/VAE/CLIP values, otherwise the wrong family's options are shown.
    const newFamily = d.model_family || d.family || null;
    console.log("[updateUI] incoming family:", newFamily, "current:", node._weFamily);
    const familyChanged = newFamily && newFamily !== node._weFamily;
    if (node._weFamilySel) {
        const sel = node._weFamilySel;
        // Pre-load the full families list if not yet done — this normally
        // happens lazily on focus, but we need it populated before we can
        // set sel.value to a non-SDXL family key.
        if (!sel._familiesLoaded) {
            try {
                const r = await fetch("/workflow-extractor/list-families");
                const fd = await r.json();
                const families = fd.families || {};
                const curVal = sel.value;
                sel.innerHTML = "";
                const keys = Object.keys(families).sort((a, b) => {
                    if (a === "sdxl") return -1;
                    if (b === "sdxl") return 1;
                    return families[a].localeCompare(families[b]);
                });
                for (const key of keys) {
                    const o = document.createElement("option");
                    o.value = key; o.textContent = families[key];
                    sel.appendChild(o);
                }
                sel.value = curVal;
                sel._familiesLoaded = true;
            } catch (e) {
                console.warn("[updateUI] Could not pre-load families list:", e);
            }
        }
        // If the target family still isn't in the list (server gap), add it.
        if (newFamily && ![...sel.options].some(o => o.value === newFamily)) {
            const o = document.createElement("option");
            o.value = newFamily; o.textContent = d.model_family_label || newFamily;
            sel.appendChild(o);
        }
        sel.value = newFamily || "sdxl";
        node._weFamily = newFamily;
        console.log("[updateUI] familySel set to:", sel.value, "options:", [...sel.options].map(o => o.value));
    }

    // If family changed, reload model/VAE/CLIP dropdown lists for the new
    // family before setting values — otherwise _setOriginal can't find the
    // option in the list and the selection falls back to (Default).
    if (familyChanged && node._onFamilyChanged) {
        await node._onFamilyChanged(newFamily, { fromUpdateUI: true });
    }

    // Model A
    if (node._weModelRow?._setOriginal) {
        node._weModelRow._setOriginal(d.model_a || "", d.model_a_found !== false);
    }
    // Model B
    if (node._weModelBRow?._setOriginal) {
        node._weModelBRow._setOriginal(d.model_b || "", d.model_b_found !== false);
    }

    // VAE
    if (node._weVaeRow?._setOriginal) {
        const vn = d.vae?.name || (typeof d.vae === "string" ? d.vae : "") || "";
        const isDefault = !vn || vn.startsWith("(") || vn === "\u2014";
        node._weVaeRow._setOriginal(isDefault ? "" : vn, d.vae_found !== false);
    }

    // CLIP
    if (node._weClipRow?._setOriginal) {
        const clipNames = d.clip?.names || (Array.isArray(d.clip) ? d.clip : []);
        const firstName = clipNames[0] || "";
        const isDefault = !firstName || firstName.startsWith("(") || firstName === "\u2014";
        node._weClipRow._setOriginal(isDefault ? "" : firstName);
    }

    // Sampler
    const s = d.sampler || {};
    if (node._weSamplerRows) {
        const rows = node._weSamplerRows;
        if (rows.steps?._setOriginal) rows.steps._setOriginal(s.steps ?? 20);
        if (rows.cfg?._setOriginal) rows.cfg._setOriginal(s.cfg ?? 5.0);
        if (rows.sampler?._setOriginal) rows.sampler._setOriginal(s.sampler_name ?? "euler");
        if (rows.scheduler?._setOriginal) rows.scheduler._setOriginal(s.scheduler ?? "simple");
        if (rows.seed?._setOriginal) rows.seed._setOriginal(s.seed ?? 0);
        if (rows.seed_b?._setOriginal) rows.seed_b._setOriginal(s.seed_b ?? s.seed ?? 0);
        // WAN Video dual steps
        if (rows.steps_high?._setOriginal) rows.steps_high._setOriginal(s.steps_high ?? 4);
        if (rows.steps_low?._setOriginal)  rows.steps_low._setOriginal(s.steps_low  ?? 20);
    }

    // Resolution
    const r = d.resolution || {};
    if (node._weResRows) {
        const rr = node._weResRows;
        if (rr.width?._setOriginal) rr.width._setOriginal(r.width ?? 768);
        if (rr.height?._setOriginal) rr.height._setOriginal(r.height ?? 1280);
        if (rr.batch?._setOriginal) rr.batch._setOriginal(r.batch_size ?? 1);
        if (rr.frames) {
            if (r.length != null) {
                rr.frames.style.display = "flex";
                if (rr.frames._setOriginal) rr.frames._setOriginal(r.length);
            }
        }
    }

    // Prompts
    if (node._wePosBox) node._wePosBox.value = d.positive_prompt || "";
    if (node._weNegBox) node._weNegBox.value = d.negative_prompt || "";

    // LoRAs
    updateLoras(node);

    // Update WAN-specific visibility
    updateWanVisibility(node);

    syncHidden(node);

    // Async: check LoRA availability and update display
    checkLoraAvailability(node);
}

// --- Async LoRA availability check ---
async function checkLoraAvailability(node) {
    const d = node._weExtracted;
    if (!d) return;
    const allLoras = [...(d.loras_a || []), ...(d.loras_b || [])];
    const names = [...new Set(allLoras.map(l => l.name).filter(Boolean))];
    if (names.length === 0) return;
    // Skip if we already have availability data from Python
    if (d.lora_availability && Object.keys(d.lora_availability).length >= names.length) return;
    try {
        const resp = await fetch("/prompt-manager-advanced/check-loras", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lora_names: names }),
        });
        const data = await resp.json();
        if (data.success && data.results) {
            d.lora_availability = {};
            for (const [name, found] of Object.entries(data.results)) {
                d.lora_availability[name] = found;
            }
            updateLoras(node);
        }
    } catch (e) {
        console.warn("[WorkflowGenerator] Error checking LoRA availability:", e);
    }
}

// --- WAN-specific visibility ---
function updateWanVisibility(node) {
    const family = node._weFamily;
    const isWanVideo = (family === "wan_video_i2v" || family === "wan_video_t2v");
    const isWan      = isWanVideo || (family === "wan_image");

    // Model A label: "Model A" for any WAN, "Model" otherwise
    if (node._weModelRow?._label) {
        node._weModelRow._label.textContent = isWan ? "Model A" : "Model";
    }
    // Model B row: only for WAN Video
    if (node._weModelBRow) {
        node._weModelBRow.style.display = isWanVideo ? "flex" : "none";
    }

    // LoRA stack titles
    if (node._weLoraACard?._titleLabel) {
        node._weLoraACard._titleLabel.textContent = isWanVideo ? "LoRA Stack (High)" : "LoRA Stack";
    }
    // LoRA Stack B: only for WAN Video
    if (node._weLoraB) {
        node._weLoraB.style.display = isWanVideo ? "flex" : "none";
    }

    // Frames row: show for any WAN (image or video)
    if (node._weResRows?.frames) {
        node._weResRows.frames.style.display = isWan ? "flex" : "none";
    }

    // Steps High / Low rows: only for WAN Video
    if (node._weSamplerRows?.steps_high) {
        node._weSamplerRows.steps_high.style.display = isWanVideo ? "flex" : "none";
    }
    if (node._weSamplerRows?.steps_low) {
        node._weSamplerRows.steps_low.style.display  = isWanVideo ? "flex" : "none";
    }
    // Standard Steps row: hide when WAN Video (replaced by high/low)
    if (node._weSamplerRows?.steps) {
        node._weSamplerRows.steps.style.display = isWanVideo ? "none" : "flex";
    }
    // Seed B: only for WAN Video; rename Seed label to Seed A when visible
    if (node._weSamplerRows?.seed_b) {
        node._weSamplerRows.seed_b.style.display = isWanVideo ? "flex" : "none";
    }
    if (node._weSamplerRows?.seed) {
        const seedLabel = node._weSamplerRows.seed.querySelector("label, .we-label, span");
        if (seedLabel) seedLabel.textContent = isWanVideo ? "Seed A" : "Seed";
    }

}

// --- LoRA display ---
function updateLoras(node) {
    const containerA = node._weLoraAContainer;
    const containerB = node._weLoraBContainer;
    if (!containerA) return;
    containerA.innerHTML = "";
    if (containerB) containerB.innerHTML = "";

    const d = node._weExtracted;
    if (!d) return;

    const lorasA = d.loras_a || [];
    const lorasB = d.loras_b || [];
    const hasBoth = lorasA.length > 0 && lorasB.length > 0;

    const noLorasMsg = () => makeEl("div", {
        color: "rgba(200, 200, 200, 0.5)", fontStyle: "italic",
        fontSize: "11px", padding: "8px", width: "100%", textAlign: "center",
    }, "No LoRAs");

    if (!lorasA.length) containerA.appendChild(noLorasMsg());
    if (!lorasB.length && containerB) containerB.appendChild(noLorasMsg());
    if (!lorasA.length && !lorasB.length) return;

    const populateStack = (container, loras, stackKey) => {
        for (const lora of loras) {
            const name = lora.name || "";
            const stateKey = stackKey ? `${stackKey}:${name}` : name;
            const avail = d.lora_availability?.[name] !== false;
            if (node._weLoraState[stateKey] === undefined) {
                node._weLoraState[stateKey] = {
                    active: true,
                    model_strength: lora.model_strength ?? 1.0,
                    clip_strength: lora.clip_strength ?? 1.0,
                };
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

}

// --- Freeze / thaw all inputs ---
function setFieldsFrozen(node, frozen) {
    const opacity = frozen ? "0.6" : "1.0";
    const pointer = frozen ? "none" : "auto";

    // Freeze all section bodies
    for (const sec of Object.values(node._weSections || {})) {
        if (sec._body) {
            sec._body.style.opacity = opacity;
            sec._body.style.pointerEvents = pointer;
        }
    }
}

// --- Error banner on node ---
function _showError(node, errorMsg) {
    const root = node._weRoot;
    if (!root) return;

    // Remove existing banner
    const existing = root.querySelector(".we-error-banner");
    if (existing) existing.remove();

    if (!errorMsg) return;

    const banner = makeEl("div", {
        padding: "8px 12px",
        backgroundColor: "rgba(220, 53, 69, 0.15)",
        border: "1px solid rgba(220, 53, 69, 0.6)",
        borderRadius: "6px",
        color: "#f88",
        fontSize: "12px",
        lineHeight: "1.4",
        marginBottom: "4px",
        wordBreak: "break-word",
    });
    banner.className = "we-error-banner";

    const icon = makeEl("span", { marginRight: "6px", fontSize: "13px" }, "\u26A0\uFE0F");
    const text = makeEl("span", {}, errorMsg);
    banner.append(icon, text);

    // Insert at the top of the root
    root.insertBefore(banner, root.firstChild);
}

// --- Sync hidden widgets ---
function syncHidden(node) {
    const wSet = (name, val) => {
        const w = node.widgets?.find(x => x.name === name);
        if (w) w.value = val;
    };
    const wfOn = node.widgets?.find(w => w.name === "use_workflow_data")?.value;
    const ov = { ...node._weOverrides };
    if (node._weModelRow?._getValue) {
        const v = node._weModelRow._getValue();
        if (v) ov.model_a = v;
    }
    if (node._weModelBRow?._getValue) {
        const v = node._weModelBRow._getValue();
        if (v) ov.model_b = v;
    }
    if (node._weVaeRow?._getValue) {
        const v = node._weVaeRow._getValue();
        if (v) ov.vae = v;
    }
    if (node._weClipRow?._getValue) {
        const v = node._weClipRow._getValue();
        if (v) ov.clip_names = [v];
    }
    // When workflow_data is driving the node, sampler/resolution/prompts come
    // from the source — do NOT capture them as overrides or they will freeze
    // at stale values and prevent the source from updating the UI.
    if (!wfOn) {
        if (node._weSamplerRows) {
            const r = node._weSamplerRows;
            if (r.steps?._inp) ov.steps = parseInt(r.steps._inp.value) || 20;
            if (r.cfg?._inp) ov.cfg = parseFloat(r.cfg._inp.value) || 5.0;
            if (r.seed?._inp) ov.seed = parseInt(r.seed._inp.value) || 0;
            if (r.seed_b?._inp && r.seed_b.style.display !== "none") {
                ov.seed_b = parseInt(r.seed_b._inp.value) || 0;
            }
            if (r.sampler?._inp) ov.sampler_name = r.sampler._inp.value;
            if (r.scheduler?._inp) ov.scheduler = r.scheduler._inp.value;
            // WAN Video dual steps
            if (r.steps_high?._inp && r.steps_high.style.display !== "none") {
                ov.steps_high = parseInt(r.steps_high._inp.value) || 4;
            }
            if (r.steps_low?._inp && r.steps_low.style.display !== "none") {
                ov.steps_low = parseInt(r.steps_low._inp.value) || 20;
            }
        }
        if (node._weResRows) {
            const r = node._weResRows;
            if (r.width?._inp) ov.width = parseInt(r.width._inp.value) || 768;
            if (r.height?._inp) ov.height = parseInt(r.height._inp.value) || 1280;
            if (r.batch?._inp) ov.batch_size = parseInt(r.batch._inp.value) || 1;
            if (r.frames?._inp && r.frames.style.display !== "none") {
                ov.length = parseInt(r.frames._inp.value) || 81;
            }
        }
        if (node._wePosBox) ov.positive_prompt = node._wePosBox.value;
        if (node._weNegBox) ov.negative_prompt = node._weNegBox.value;
    } else {
        // In workflow_data mode, remove any stale sampler/resolution/prompt overrides
        // so Python always uses the values from the source workflow.
        delete ov.steps; delete ov.cfg; delete ov.seed; delete ov.seed_b;
        delete ov.sampler_name; delete ov.scheduler;
        delete ov.steps_high; delete ov.steps_low;
        delete ov.width; delete ov.height; delete ov.batch_size; delete ov.length;
        delete ov.positive_prompt; delete ov.negative_prompt;
    }
    if (node._weFamily) ov._family = node._weFamily;

    // Control after generate
    if (node._weControlMode) {
        ov._control_after_generate = node._weControlMode._inp?.value || "randomize";
    }

    wSet("override_data", JSON.stringify(ov));
    wSet("lora_state", JSON.stringify(node._weLoraState || {}));

    // Persist to node.properties for tab-switch survival
    node.properties = node.properties || {};
    node.properties.we_override_data = JSON.stringify(ov);
    node.properties.we_lora_state = JSON.stringify(node._weLoraState || {});
    if (node._weExtracted) {
        node.properties.we_extracted_cache = JSON.stringify(node._weExtracted);
    }
}

// --- Apply saved overrides back to UI after reload ---
function applyOverrides(node, ovJson, lsJson) {
    let ov, ls;
    try { ov = JSON.parse(ovJson || "{}"); } catch { ov = {}; }
    try { ls = JSON.parse(lsJson || "{}"); } catch { ls = {}; }
    const d = node._weExtracted;
    if (!d) return;
    const isEmpty = (o) => !o || Object.keys(o).length === 0;
    if (isEmpty(ov) && isEmpty(ls)) return;

    const applySelect = (row, ovVal, grouped) => {
        if (!row || ovVal == null || ovVal === "\u2014") return;
        const sel = row._sel;
        if (!sel) return;
        if (![...sel.options].some(o => o.value === ovVal)) {
            const o = document.createElement("option");
            o.value = ovVal;
            o.textContent = grouped ? cleanModelName(ovVal) : ovVal;
            sel.appendChild(o);
        }
        sel.value = ovVal;
        sel.style.color = C.text;
        if (row._resetBtn) {
            row._resetBtn.style.visibility = sel.value !== sel.options[0]?.value ? "visible" : "hidden";
        }
    };

    const applyInput = (row, ovVal) => {
        if (!row || ovVal == null) return;
        const inp = row._inp;
        if (!inp) return;
        inp.value = ovVal;
        if (row._resetBtn) row._resetBtn.style.visibility = "visible";
    };

    // Restore family
    if (ov._family && node._weFamilySel) {
        const sel = node._weFamilySel;
        if (![...sel.options].some(o => o.value === ov._family)) {
            const o = document.createElement("option");
            o.value = ov._family; o.textContent = ov._family;
            sel.appendChild(o);
        }
        sel.value = ov._family;
        node._weFamily = ov._family;
    }

    if (ov.model_a) applySelect(node._weModelRow, ov.model_a, true);
    if (ov.model_b) applySelect(node._weModelBRow, ov.model_b, true);
    if (ov.vae) applySelect(node._weVaeRow, ov.vae, true);
    if (ov.clip_names?.length) applySelect(node._weClipRow, ov.clip_names[0], true);

    if (node._weSamplerRows) {
        const rows = node._weSamplerRows;
        if (ov.steps != null) applyInput(rows.steps, ov.steps);
        if (ov.cfg != null) applyInput(rows.cfg, ov.cfg);
        if (ov.seed   != null) applyInput(rows.seed,   ov.seed);
        if (ov.seed_b != null) applyInput(rows.seed_b, ov.seed_b);
        if (ov.sampler_name) applyInput(rows.sampler, ov.sampler_name);
        if (ov.scheduler) applyInput(rows.scheduler, ov.scheduler);
        if (ov.steps_high != null) applyInput(rows.steps_high, ov.steps_high);
        if (ov.steps_low  != null) applyInput(rows.steps_low,  ov.steps_low);
    }

    if (node._weResRows) {
        const rr = node._weResRows;
        if (ov.width != null) applyInput(rr.width, ov.width);
        if (ov.height != null) applyInput(rr.height, ov.height);
        if (ov.batch_size != null) applyInput(rr.batch, ov.batch_size);
        if (ov.length != null) applyInput(rr.frames, ov.length);
    }

    if (ov.positive_prompt != null && node._wePosBox) node._wePosBox.value = ov.positive_prompt;
    if (ov.negative_prompt != null && node._weNegBox) node._weNegBox.value = ov.negative_prompt;

    // Control after generate
    if (ov._control_after_generate && node._weControlMode?._inp) {
        node._weControlMode._inp.value = ov._control_after_generate;
    }

    if (!isEmpty(ls)) {
        node._weLoraState = ls;
        updateLoras(node);
    }

    updateWanVisibility(node);
    syncHidden(node);
}

// --- Parse workflow_data string into extracted format ---
function parseWorkflowData(jsonStr) {
    if (!jsonStr) return null;
    try {
        const d = JSON.parse(jsonStr);
        // Map from build_simplified_workflow_data schema to extracted schema
        return {
            positive_prompt: d.positive_prompt || "",
            negative_prompt: d.negative_prompt || "",
            loras_a: d.loras_a || [],
            loras_b: d.loras_b || [],
            model_a: d.model_a || "",
            model_b: d.model_b || "",
            model_a_found: d.model_a_found,
            model_b_found: d.model_b_found,
            vae: { name: d.vae || "", source: "workflow_data" },
            vae_found: d.vae_found,
            clip: {
                names: Array.isArray(d.clip) ? d.clip : (d.clip ? [d.clip] : []),
                type: "", source: "workflow_data",
            },
            sampler: d.sampler || {
                steps: 20, cfg: 5.0, seed: 0,
                sampler_name: "euler", scheduler: "simple",
            },
            resolution: d.resolution || { width: 768, height: 1280, batch_size: 1, length: null },
            model_family: d.family || "",
            model_family_label: d.family_strategy || "",
        };
    } catch (e) {
        console.warn("[WorkflowGenerator] Could not parse workflow_data:", e);
        return null;
    }
}


// ============================================================
// --- Main extension ---
// ============================================================
app.registerExtension({
    name: "WorkflowGenerator",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WorkflowGenerator") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated?.apply(this, arguments);
            const node = this;

            // -- State --
            node._weExtracted = null;
            node._weLoraState = {};
            node._weOverrides = {};
            node._weFamily = "sdxl";
            node._weSections = {};
            this.serialize_widgets = true;

            const _syncS = () => syncHidden(node);

            // ============================================================
            // -- Build the DOM UI --
            // ============================================================
            const root = makeEl("div", {
                display: "flex", flexDirection: "column", gap: "4px",
                // paddingBottom: 38px reserves space for the ComfyUI selection
                // toolbox that appears at the bottom of the node when selected,
                // preventing it from overlapping the last section.
                padding: "6px 6px 38px 6px", marginTop: "-10px",
                width: "100%", boxSizing: "border-box",
                fontFamily: "Inter, system-ui, -apple-system, sans-serif",
                overflow: "hidden",
            });
            forwardWheelToCanvas(root);
            node._weRoot = root;

            // -- 1. RESOLUTION section (open) --
            const resSec = makeSection("RESOLUTION");
            node._weSections.resolution = resSec;
            const resRows = {
                width:  makeInput("Width",  "number", 768, { min: 64, max: 8192, step: 8 }, _syncS),
                height: makeInput("Height", "number", 1280, { min: 64, max: 8192, step: 8 }, _syncS),
                batch:  makeInput("Batch",  "number", 1,   { min: 1, max: 128, step: 1 }, _syncS),
                frames: makeInput("Frames", "number", 81,  { min: 1, max: 1000, step: 1 }, _syncS),
            };
            resRows.frames.style.display = "none"; // hidden until WAN
            for (const row of Object.values(resRows)) resSec._body.appendChild(row);
            root.appendChild(resSec);
            node._weResRows = resRows;

            // -- 2. MODEL section (open, with VAE/CLIP) --
            const modelSec = makeSection("MODEL");
            node._weSections.model = modelSec;

            // Family type row
            const familyRow = makeEl("div", { ...ROW_STYLE });
            familyRow.appendChild(makeEl("span", {
                color: C.accent, width: LABEL_W, flexShrink: "0", fontWeight: "bold",
            }, "Type"));
            familyRow.appendChild(makeEl("span", { width: "14px", flexShrink: "0" }));
            const familySel = document.createElement("select");
            Object.assign(familySel.style, { ...INPUT_STYLE, color: C.accent, fontWeight: "bold" });
            let _familiesLoaded = false;
            familySel.onfocus = async () => {
                if (_familiesLoaded || familySel._familiesLoaded) return;
                _familiesLoaded = true;
                familySel._familiesLoaded = true;
                try {
                    const r = await fetch("/workflow-extractor/list-families");
                    const d = await r.json();
                    const families = d.families || {};
                    const curVal = familySel.value;
                    familySel.innerHTML = "";
                    // Sort alphabetically, SDXL first
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
            // Add SDXL as initial option
            {
                const o = document.createElement("option");
                o.value = "sdxl"; o.textContent = "SDXL";
                familySel.appendChild(o);
                familySel.value = "sdxl";
            }
            familySel.onchange = () => onFamilyChanged(familySel.value);
            familyRow.appendChild(familySel);
            modelSec._body.appendChild(familyRow);
            node._weFamilySel = familySel;

            // Fetch models helper
            const fetchModels = async () => {
                try {
                    const fam = node._weFamily;
                    const url = fam
                        ? `/workflow-extractor/list-models?family=${encodeURIComponent(fam)}`
                        : `/workflow-extractor/list-models`;
                    const r = await fetch(url); const d = await r.json();
                    return d.models || [];
                } catch { return []; }
            };

            // Model A row
            const modelRow = makeSelectRow("Model", "", fetchModels,
                (v) => { node._weOverrides.model_a = v; _syncS(); }, true);
            modelSec._body.appendChild(modelRow);
            node._weModelRow = modelRow;

            // Model B row (hidden unless WAN)
            const modelBRow = makeSelectRow("Model B", "", fetchModels,
                (v) => { node._weOverrides.model_b = v; _syncS(); }, true);
            modelBRow.style.display = "none";
            modelSec._body.appendChild(modelBRow);
            node._weModelBRow = modelBRow;

            // -- Separator before VAE/CLIP --
            modelSec._body.appendChild(makeSeparator());

            // VAE row
            const vaeRow = makeSelectRow("VAE", "",
                async () => {
                    try {
                        const fam = encodeURIComponent(node._weFamily || "");
                        const r = await fetch(`/workflow-extractor/list-vaes?family=${fam}`);
                        const d = await r.json(); return d.vaes || [];
                    } catch { return []; }
                },
                (v) => { node._weOverrides.vae = v; _syncS(); }, false);
            modelSec._body.appendChild(vaeRow);
            node._weVaeRow = vaeRow;

            // CLIP row
            const clipRow = makeSelectRow("CLIP", "",
                async () => {
                    try {
                        const fam = encodeURIComponent(node._weFamily || "");
                        const r = await fetch(`/workflow-extractor/list-clips?family=${fam}`);
                        const d = await r.json(); return d.clips || [];
                    } catch { return []; }
                },
                (v) => { node._weOverrides.clip_names = [v]; _syncS(); }, false);
            modelSec._body.appendChild(clipRow);
            node._weClipRow = clipRow;

            root.appendChild(modelSec);

            // When model A reset is clicked, also restore original family
            modelRow._onReset = () => {
                const origFamily = node._weExtracted?.model_family || null;
                if (origFamily && origFamily !== node._weFamily) {
                    if (node._weFamilySel) node._weFamilySel.value = origFamily;
                    onFamilyChanged(origFamily);
                }
            };

            // Family change handler — also stored on node so updateUI can call it.
            // fromUpdateUI=true: skip auto-select & _syncS — updateUI will call
            // _setOriginal and _syncS itself right after, with the correct values.
            const onFamilyChanged = async (familyKey, { fromUpdateUI = false } = {}) => {
                node._weFamily = familyKey;
                // Reset VAE/CLIP overrides — new family means old selections are invalid
                delete node._weOverrides.vae;
                delete node._weOverrides.clip_names;
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
                        await reloadGroupedSelect(node._weVaeRow, async () => d.vaes || [], false, null);
                    } catch { await reloadGroupedSelect(node._weVaeRow, async () => [], false, null); }
                };
                const reloadClip = async () => {
                    try {
                        const r = await fetch(`/workflow-extractor/list-clips?family=${fam}`);
                        const d = await r.json();
                        await reloadGroupedSelect(node._weClipRow, async () => d.clips || [], false, null);
                    } catch { await reloadGroupedSelect(node._weClipRow, async () => [], false, null); }
                };
                await Promise.all([
                    reloadGroupedSelect(node._weModelRow, fetchModelsForFamily, true),
                    reloadGroupedSelect(node._weModelBRow, fetchModelsForFamily, true),
                    reloadVae(),
                    reloadClip(),
                ]);
                if (!fromUpdateUI) {
                    // Manual family change — auto-select first model and sync.
                    // updateUI calls _setOriginal itself so we skip this.
                    const firstModel = node._weModelRow._sel.options[1]?.value || "";
                    if (firstModel) {
                        node._weModelRow._sel.value = firstModel;
                        node._weOverrides.model_a = firstModel;
                    }
                    updateWanVisibility(node);
                    _syncS();
                }
            };
            // Expose so updateUI can trigger family reload from workflow_data
            node._onFamilyChanged = onFamilyChanged;

            // -- 3. SAMPLER section --
            const sampSec = makeSection("SAMPLER");
            node._weSections.sampler = sampSec;

            // Standard steps (hidden when WAN Video)
            const stepsRow     = makeInput("Steps",      "number", 20,  { min: 1, max: 200, step: 1 }, _syncS);
            // WAN Video dual steps (hidden by default)
            const stepsHighRow = makeInput("Steps (High)", "number", 4,  { min: 1, max: 200, step: 1 }, _syncS);
            const stepsLowRow  = makeInput("Steps (Low)",  "number", 20, { min: 1, max: 200, step: 1 }, _syncS);
            stepsHighRow.style.display = "none";
            stepsLowRow.style.display  = "none";

            // WAN Video seed B (hidden by default; Seed renamed to Seed A when visible)
            const seedRow  = makeInput("Seed",   "number", 0, { min: 0, step: 1 }, _syncS);
            const seedBRow = makeInput("Seed B", "number", 0, { min: 0, step: 1 }, _syncS);
            seedBRow.style.display = "none";

            const sampRows = {
                steps:      stepsRow,
                steps_high: stepsHighRow,
                steps_low:  stepsLowRow,
                cfg:        makeInput("CFG",       "number", 5.0,      { min: 0, max: 100, step: 0.5 }, _syncS),
                sampler:    makeInput("Sampler",   "select", "euler",  { options: SAMPLERS }, _syncS),
                scheduler:  makeInput("Scheduler", "select", "simple", { options: SCHEDULERS }, _syncS),
                seed:       seedRow,
                seed_b:     seedBRow,
            };
            // Append in display order: steps, steps_high, steps_low, cfg, sampler, scheduler, seed, seed_b
            sampSec._body.appendChild(stepsRow);
            sampSec._body.appendChild(stepsHighRow);
            sampSec._body.appendChild(stepsLowRow);
            sampSec._body.appendChild(sampRows.cfg);
            sampSec._body.appendChild(sampRows.sampler);
            sampSec._body.appendChild(sampRows.scheduler);
            sampSec._body.appendChild(seedRow);
            sampSec._body.appendChild(seedBRow);

            // Control after generate
            const controlRow = makeInput("Control after generate", "select", "randomize",
                { options: CONTROL_MODES }, _syncS);
            sampSec._body.appendChild(controlRow);
            node._weControlMode = controlRow;

            root.appendChild(sampSec);
            node._weSamplerRows = sampRows;

            // -- 4. PROMPTS (positive open, negative closed) --
            const posSec = makeSection("POSITIVE PROMPT");
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
            posBox.onchange = _syncS;
            posBox.oninput = _syncS;
            posSec._body.appendChild(posBox);
            root.appendChild(posSec);
            node._wePosBox = posBox;

            const negSec = makeSection("NEGATIVE PROMPT");
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
            negBox.onchange = _syncS;
            negBox.oninput = _syncS;
            negSec._body.appendChild(negBox);
            root.appendChild(negSec);
            node._weNegBox = negBox;

            // -- 5. LORAS section --
            const loraSec = makeSection("LORAS");
            node._weSections.loras = loraSec;

            // Stack A card
            const loraACard = createLoraStackContainer("LoRA Stack",
                () => {
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
                    updateLoras(node); _syncS();
                },
                () => {
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
                    updateLoras(node); _syncS();
                },
            );
            loraSec._body.appendChild(loraACard);
            node._weLoraAContainer = loraACard._tagsContainer;
            node._weLoraACard = loraACard;

            // Stack B card
            const loraBCard = createLoraStackContainer("LoRA Stack B",
                () => {
                    const d = node._weExtracted;
                    for (const lora of (d?.loras_b || [])) {
                        const k = `b:${lora.name}`;
                        if (node._weLoraState[k]) {
                            node._weLoraState[k].model_strength = lora.model_strength ?? 1.0;
                            node._weLoraState[k].clip_strength  = lora.clip_strength  ?? 1.0;
                        }
                    }
                    updateLoras(node); _syncS();
                },
                () => {
                    const d = node._weExtracted;
                    const lorasB = d?.loras_b || [];
                    const anyActive = lorasB.some(l => node._weLoraState[`b:${l.name}`]?.active !== false);
                    for (const lora of lorasB) {
                        const k = `b:${lora.name}`;
                        if (!node._weLoraState[k]) node._weLoraState[k] = { active: true, model_strength: 1.0, clip_strength: 1.0 };
                        node._weLoraState[k].active = !anyActive;
                    }
                    updateLoras(node); _syncS();
                },
            );
            loraBCard.style.display = "none"; // hidden until WAN
            loraSec._body.appendChild(loraBCard);
            node._weLoraBContainer = loraBCard._tagsContainer;
            node._weLoraB = loraBCard;

            root.appendChild(loraSec);

            const MIN_W = 470;
            const MIN_H = 300;
            // Height of everything above the DOM widget: title bar ~30px +
            // two native toggle widgets ~28px each + small margin = 90px.
            const NATIVE_H = 90;

            // -- Register the DOM widget.
            // Use computeSize directly on the widget — the same pattern used by
            // prompt_manager_advanced.js (which has no grey-gap or resize issues).
            // LiteGraph re-queries widget.computeSize on every canvas redraw, so
            // calling app.graph.setDirtyCanvas(true, true) is all that is needed
            // to make the node grow/shrink when sections open or close.
            // No setSize(), no getHeight/getMinHeight, no _weRecalc loop needed.
            const domW = node.addDOMWidget("we_ui", "div", root, {
                hideOnZoom: false,
                serialize: false,
            });
            domW.computeSize = function (width) {
                const h = root.scrollHeight || MIN_H;
                return [width, h];
            };

            // -- Only hide data widgets, keep toggle switches visible --
            // IMPORTANT: override_data and lora_state must keep their original
            // STRING type so ComfyUI's graphToPrompt includes them in execution.
            // "converted-widget" type causes ComfyUI to skip the widget value.
            const KEEP_VISIBLE = new Set(["use_workflow_data", "use_lora_input"]);
            const KEEP_TYPE = new Set(["override_data", "lora_state"]);
            for (const w of (node.widgets || [])) {
                if (w === domW || KEEP_VISIBLE.has(w.name)) continue;
                w.computeSize = () => [0, -4];
                if (!KEEP_TYPE.has(w.name)) {
                    w.type = "converted-widget";
                }
                w.hidden = true;
                w.draw = function () {};
                if (w.element) w.element.style.display = "none";
            }

            // Enforce minimum size when user drags to resize.
            // Prevents dragging the node smaller than its content.
            const origOnResize = node.onResize;
            node.onResize = function (size) {
                const contentH = (root.scrollHeight || MIN_H) + NATIVE_H;
                size[0] = Math.max(MIN_W, size[0]);
                size[1] = Math.max(contentH, MIN_H, size[1]);
                if (origOnResize) return origOnResize.apply(this, arguments);
            };

            // Set initial size — deferred so the DOM has been painted and
            // scrollHeight reflects real content height.
            requestAnimationFrame(() => requestAnimationFrame(() => {
                const contentH = (root.scrollHeight || MIN_H) + NATIVE_H;
                node.setSize([MIN_W, Math.max(contentH, MIN_H)]);
                app.graph.setDirtyCanvas(true, true);
            }));

            applyZoomScaling(root);

            // -- Handle use_workflow_data toggle --
            const wfToggle = node.widgets?.find(w => w.name === "use_workflow_data");
            if (wfToggle) {
                const origWfCb = wfToggle.callback;
                wfToggle.callback = async function (value) {
                    if (origWfCb) origWfCb.apply(this, arguments);
                    if (value) {
                        // Try reading workflow_data from the connected source node's
                        // properties cache (forceInput — no local widget exists).
                        let parsed = null;
                        const wfInput = node.inputs?.find(i => i.name === "workflow_data_input");
                        if (wfInput?.link != null) {
                            const linkInfo = node.graph?.links?.[wfInput.link];
                            if (linkInfo) {
                                const srcNode = node.graph?.getNodeById(linkInfo.origin_id);
                                // PromptExtractor caches its last workflow_data in properties
                                const srcCache = srcNode?.properties?.we_last_workflow_data;
                                if (srcCache) parsed = parseWorkflowData(srcCache);
                            }
                        }
                        // Fallback: use our own extracted cache
                        if (!parsed && node._weExtracted) {
                            parsed = node._weExtracted;
                        }
                        if (parsed) {
                            node._weExtracted = parsed;
                            node._wePopulated = true;
                            node._weLoraState = {};
                            node._weOverrides = {};
                            // Await updateUI so family/_weFamily is set BEFORE
                            // _syncS() writes override_data — otherwise family
                            // snaps back to SDXL in the serialised overrides.
                            await updateUI(node);
                        }
                        setFieldsFrozen(node, true);
                    } else {
                        // LoRAs come from workflow_data — clear them when toggling off.
                        // Users who want LoRAs in manual mode connect lora_stack inputs.
                        if (node._weExtracted) {
                            node._weExtracted.loras_a = [];
                            node._weExtracted.loras_b = [];
                            node._weExtracted.lora_availability = {};
                        }
                        node._weLoraState = {};
                        updateLoras(node);
                        setFieldsFrozen(node, false);
                    }
                    _syncS();
                };
            }

            // -- Handle use_lora_input toggle --
            const loraToggle = node.widgets?.find(w => w.name === "use_lora_input");
            if (loraToggle) {
                const origLoraCb = loraToggle.callback;
                loraToggle.callback = function (value) {
                    if (origLoraCb) origLoraCb.apply(this, arguments);
                    if (!value && node._weExtracted) {
                        // Toggled OFF — clear LoRAs that came from connected inputs
                        // (unless use_workflow_data is ON, which is the other LoRA source)
                        const wfOn = node.widgets?.find(w => w.name === "use_workflow_data")?.value;
                        if (!wfOn) {
                            node._weExtracted.loras_a = [];
                            node._weExtracted.loras_b = [];
                            node._weExtracted.lora_availability = {};
                            node._weLoraState = {};
                            updateLoras(node);
                        }
                    }
                    _syncS();
                };
            }

            // -- Seed control after generate --
            node._onExecutedSeed = function () {
                const mode = node._weControlMode?._inp?.value || "fixed";
                const applyMode = (row) => {
                    if (!row?._inp || row.style.display === "none") return;
                    const cur = parseInt(row._inp.value) || 0;
                    if (mode === "randomize") {
                        row._inp.value = Math.floor(Math.random() * 2147483647);
                    } else if (mode === "increment") {
                        row._inp.value = cur + 1;
                    } else if (mode === "decrement") {
                        row._inp.value = Math.max(0, cur - 1);
                    }
                };
                applyMode(node._weSamplerRows?.seed);
                applyMode(node._weSamplerRows?.seed_b);
                // "fixed" -- do nothing
                _syncS();
            };

            // -- Pre-generation UI update (via send_sync from Python) --
            // Fires before model loading / sampling so UI feels instant.
            api.addEventListener("workflow-generator-pre-update", (event) => {
                if (String(event.detail?.node_id) !== String(node.id)) return;
                const info = event.detail?.info?.extracted;
                if (!info) return;

                // Clear any previous error banner
                _showError(node, null);

                const wfToggle = node.widgets?.find(w => w.name === "use_workflow_data");

                if (!wfToggle?.value) {
                    // ── Manual mode (use_workflow_data OFF) ──────────────────
                    // Python echoes effective values back.  We only update
                    // availability/found flags and LoRA list — never reset
                    // UI fields the user has already filled in.
                    if (!node._weExtracted) {
                        // Very first execution on a brand-new node: seed
                        // _weExtracted with the effective info so subsequent
                        // runs have something to compare against.
                        node._weExtracted = info;
                        node._wePopulated = true;
                        node.properties = node.properties || {};
                        node.properties.we_extracted_cache = JSON.stringify(info);
                        // Do NOT call updateUI — user's typed values live in
                        // _weOverrides / the DOM already; just sync family/visibility.
                        if (info.model_family && node._weFamilySel) {
                            const fam = info.model_family;
                            if (![...node._weFamilySel.options].some(o => o.value === fam)) {
                                const o = document.createElement("option");
                                o.value = fam; o.textContent = info.model_family_label || fam;
                                node._weFamilySel.appendChild(o);
                            }
                            node._weFamilySel.value = fam;
                            node._weFamily = fam;
                        }
                        updateWanVisibility(node);
                    } else {
                        // Subsequent runs — keep UI as-is, only update metadata
                        const loraToggle = node.widgets?.find(w => w.name === "use_lora_input");
                        if (loraToggle?.value) {
                            const oldNames = [...(node._weExtracted.loras_a || []), ...(node._weExtracted.loras_b || [])]
                                .map(l => l.name).sort().join(",");
                            node._weExtracted.loras_a = info.loras_a || [];
                            node._weExtracted.loras_b = info.loras_b || [];
                            const newNames = [...node._weExtracted.loras_a, ...node._weExtracted.loras_b]
                                .map(l => l.name).sort().join(",");
                            if (oldNames !== newNames) node._weLoraState = {};
                        }
                        if (info.lora_availability) node._weExtracted.lora_availability = info.lora_availability;
                        if (info.model_a_found !== undefined) node._weExtracted.model_a_found = info.model_a_found;
                        if (info.model_b_found !== undefined) node._weExtracted.model_b_found = info.model_b_found;
                        if (info.vae_found !== undefined) node._weExtracted.vae_found = info.vae_found;
                        node.properties = node.properties || {};
                        node.properties.we_extracted_cache = JSON.stringify(node._weExtracted);
                        updateLoras(node);
                    }
                } else {
                    // ── use_workflow_data ON — always reflect the source ──────
                    // syncHidden no longer captures sampler/resolution/prompts
                    // in this mode, so overrides never freeze them. Always call
                    // updateUI so the UI stays in sync with the source workflow.
                    node._weExtracted = info;
                    node.properties = node.properties || {};
                    node.properties.we_extracted_cache = JSON.stringify(info);

                    node._wePopulated = true;
                    node._weLoraState = {};
                    node._weOverrides = {};
                    if (node.properties) {
                        delete node.properties.we_override_data;
                        delete node.properties.we_lora_state;
                    }
                    updateUI(node);
                }
                node._preUpdateApplied = true;
            });

            // -- If not restoring from workflow, try initial load --
            if (!node._configuredFromWorkflow) {
                // No extraction needed on fresh node -- just defaults
            }

            return r;
        };

        // -- onExecuted --
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origOnExecuted?.apply(this, arguments);

            // Handle generation errors — show/clear error banner
            const wfInfo = message?.workflow_info?.[0];
            const genError = wfInfo?.error;
            _showError(this, genError || null);

            // UI population is handled by pre-update listener (fires before
            // generation).  Only run here as a fallback if send_sync failed.
            const info = wfInfo?.extracted;
            if (info && !this._preUpdateApplied) {
                const wfToggle = this.widgets?.find(w => w.name === "use_workflow_data");

                if (!wfToggle?.value) {
                    // Manual mode — mirror the pre-update handler: never reset user fields
                    if (!this._weExtracted) {
                        this._weExtracted = info;
                        this._wePopulated = true;
                        this.properties = this.properties || {};
                        this.properties.we_extracted_cache = JSON.stringify(info);
                        if (info.model_family && this._weFamilySel) {
                            const fam = info.model_family;
                            if (![...this._weFamilySel.options].some(o => o.value === fam)) {
                                const o = document.createElement("option");
                                o.value = fam; o.textContent = info.model_family_label || fam;
                                this._weFamilySel.appendChild(o);
                            }
                            this._weFamilySel.value = fam;
                            this._weFamily = fam;
                        }
                        updateWanVisibility(this);
                    } else {
                        const loraToggle = this.widgets?.find(w => w.name === "use_lora_input");
                        if (loraToggle?.value) {
                            const oldNames = [...(this._weExtracted.loras_a || []), ...(this._weExtracted.loras_b || [])]
                                .map(l => l.name).sort().join(",");
                            this._weExtracted.loras_a = info.loras_a || [];
                            this._weExtracted.loras_b = info.loras_b || [];
                            const newNames = [...this._weExtracted.loras_a, ...this._weExtracted.loras_b]
                                .map(l => l.name).sort().join(",");
                            if (oldNames !== newNames) this._weLoraState = {};
                        }
                        if (info.lora_availability) this._weExtracted.lora_availability = info.lora_availability;
                        if (info.model_a_found !== undefined) this._weExtracted.model_a_found = info.model_a_found;
                        if (info.model_b_found !== undefined) this._weExtracted.model_b_found = info.model_b_found;
                        if (info.vae_found !== undefined) this._weExtracted.vae_found = info.vae_found;
                        this.properties = this.properties || {};
                        this.properties.we_extracted_cache = JSON.stringify(this._weExtracted);
                        updateLoras(this);
                    }
                } else {
                    // use_workflow_data ON — always reflect the source
                    this._weExtracted = info;
                    this.properties = this.properties || {};
                    this.properties.we_extracted_cache = JSON.stringify(info);

                    this._wePopulated = true;
                    this._weLoraState = {};
                    this._weOverrides = {};
                    if (this.properties) {
                        delete this.properties.we_override_data;
                        delete this.properties.we_lora_state;
                    }
                    updateUI(this);
                }
            }
            // Reset flag for next execution
            this._preUpdateApplied = false;

            // Advance seed per control mode
            if (this._onExecutedSeed) this._onExecutedSeed();
        };

        // -- onConfigure (graph load / paste / tab return) --
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            origConfigure?.apply(this, arguments);
            const node = this;
            node._configuredFromWorkflow = true;

            // Re-hide data widgets (same logic as onNodeCreated)
            const domW = node.widgets?.find(w => w.name === "we_ui");
            const KEEP_VISIBLE = new Set(["use_workflow_data", "use_lora_input"]);
            const KEEP_TYPE = new Set(["override_data", "lora_state"]);
            for (const w of (node.widgets || [])) {
                if (w === domW || KEEP_VISIBLE.has(w.name)) continue;
                w.computeSize = () => [0, -4];
                if (!KEEP_TYPE.has(w.name)) {
                    w.type = "converted-widget";
                }
                w.hidden = true;
                w.draw = function () {};
                if (w.element) w.element.style.display = "none";
            }

            // Read saved state from node.properties
            const props = node.properties || {};
            const savedOv = props.we_override_data;
            const savedLs = props.we_lora_state;
            const savedCache = props.we_extracted_cache;

            let cached = null;
            try { cached = JSON.parse(savedCache || "{}"); } catch { cached = null; }
            const hasCache = cached && Object.keys(cached).length > 0;

            // Restore freeze state based on use_workflow_data toggle
            const wfToggle = node.widgets?.find(w => w.name === "use_workflow_data");

            if (hasCache) {
                node._weExtracted = cached;
                node._wePopulated = true;
                node._weLoraState = {};
                node._weOverrides = {};
                updateUI(node);
                // Only restore overrides in manual mode — in workflow_data mode
                // prompts/sampler/resolution come from the source on next run.
                if (!wfToggle?.value) {
                    applyOverrides(node, savedOv, savedLs);
                }
            }
            if (wfToggle?.value) {
                setFieldsFrozen(node, true);
            }

            node.setDirtyCanvas(true, true);
        };
    },
});
