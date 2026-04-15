/**
 * Workflow Builder - Full DOM-based UI
 *
 * Layout order: Resolution -> Model/VAE/CLIP -> Prompts -> Sampler -> LoRAs
 * Accepts workflow_data input (from PromptExtractor) + optional lora_stack / prompt / image inputs.
 * "Update Workflow" button pulls data from connected PromptExtractor.
 * Connected prompt inputs automatically ghost textareas and override prompts.
 * Connected LoRA stacks are always merged on execution.
 * Non-WAN families show "Model A" / "Model B" / "LoRA Stack A" / "LoRA Stack B" labels.
 * WAN family renames them to (High)/(Low) and shows Frames input in Resolution section.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Inject CSS to hide scrollbars on lora tag containers (webkit)
if (!document.getElementById("we-lora-scroll-css")) {
    const style = document.createElement("style");
    style.id = "we-lora-scroll-css";
    style.textContent = `.lora-tags-container::-webkit-scrollbar { display: none; }`;
    document.head.appendChild(style);
}
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
        .we-select option, .we-select optgroup { font-size: 14px; }
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

// --- Snap node height to fit content after section toggle ---
// Walks up from a section element to find the root (which stores _weNode),
// then forces the node to exactly fit its DOM content.
const _NATIVE_H = 90;   // title bar + toggle widgets above the DOM widget
const _MIN_W = 478;
const _MIN_H = 300;


// --- Section builder (always expanded, not collapsible) ---
function makeSection(title) {
    const wrap = makeEl("div", {
        borderRadius: "6px", overflow: "hidden", marginTop: "2px",
        backgroundColor: C.bgCard, flexShrink: "0",
    });
    const header = makeEl("div", {
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "4px 8px",
        fontSize: "11px", fontWeight: "600", color: "#aaa",
        userSelect: "none",
        borderBottom: "1px solid rgba(255,255,255,0.1)",
    });
    const label = makeEl("span", {}, title);
    header.append(label);
    const body = makeEl("div", {
        padding: "4px 8px",
    });
    wrap._titleLabel = label;
    wrap.append(header, body);
    wrap._body = body;
    return wrap;
}

// --- Shared layout constants ---
const LABEL_W = "35%";
const INPUT_W = "58%";
const ROW_STYLE = {
    display: "flex", alignItems: "center", padding: "2px 0", fontSize: "12px", gap: "2px",
};
const INPUT_STYLE = {
    background: C.bgInput, color: C.text, border: `1px solid ${C.border}`,
    borderRadius: "6px", padding: "3px 6px", fontSize: "12px",
    width: INPUT_W, boxSizing: "border-box", textAlign: "right",
};
const RESET_STYLE = {
    background: "none", border: "none", color: C.textMuted, cursor: "pointer",
    fontSize: "12px", padding: "0 2px", lineHeight: "1", flexShrink: "0",
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
        // Select all on click so user can immediately type a new value
        if (type === "number") {
            inp.addEventListener("focus", () => inp.select());
        }
        inp.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === "Escape") { e.preventDefault(); inp.blur(); }
        });
    }
    Object.assign(inp.style, { ...INPUT_STYLE });
    // Stop wheel propagation so scrolling over inputs doesn't zoom canvas
    inp.addEventListener("wheel", (e) => { e.stopPropagation(); });
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
        // For selects, ensure the value exists as an option before setting it
        if (type === "select" && v && ![...inp.options].some(o => o.value === v)) {
            const o = document.createElement("option");
            o.value = v; o.textContent = v;
            inp.appendChild(o);
        }
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

    // Right-click context menu
    tag.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        e.stopPropagation();
        document.querySelectorAll(".wb-lora-context-menu").forEach(m => m.remove());
        const menu = makeEl("div", {
            position: "fixed", left: e.clientX + "px", top: e.clientY + "px",
            background: "#2a2a2a", border: "1px solid #555", borderRadius: "6px",
            zIndex: "999999", boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
            minWidth: "120px", padding: "4px 0",
        });
        menu.className = "wb-lora-context-menu";
        const title = makeEl("div", {
            padding: "8px 12px", fontSize: "12px", color: "#aaa", fontWeight: "bold",
            borderBottom: "1px solid #444", marginBottom: "4px", whiteSpace: "nowrap",
        }, name);
        menu.appendChild(title);
        if (!avail) {
            const searchItem = makeEl("div", {
                padding: "8px 12px", cursor: "pointer", fontSize: "12px",
                color: "#4da6ff", whiteSpace: "nowrap",
            }, "\uD83D\uDD0D Search on CivitAI");
            searchItem.addEventListener("mouseenter", () => { searchItem.style.backgroundColor = "#3a3a3a"; });
            searchItem.addEventListener("mouseleave", () => { searchItem.style.backgroundColor = "transparent"; });
            searchItem.addEventListener("click", (ev) => {
                ev.stopPropagation(); menu.remove();
                window.open(`https://civitai.com/search/models?sortBy=models_v9&query=${encodeURIComponent(name)}&modelType=LORA`, "_blank");
            });
            menu.appendChild(searchItem);
        }
        document.body.appendChild(menu);
        const close = (ev) => { if (!menu.contains(ev.target)) { menu.remove(); document.removeEventListener("mousedown", close); } };
        document.addEventListener("mousedown", close);
    });

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
        display: "flex", flexWrap: "wrap", gap: "4px",
        alignContent: "flex-start",
        height: "100px", overflowY: "auto", scrollbarWidth: "none",
        border: "1px solid rgba(255,255,255,0.1)", borderRadius: "4px",
        padding: "4px",
    });
    // Stop wheel propagation so scrolling lora list doesn't zoom canvas
    tagsContainer.addEventListener("wheel", (e) => { e.stopPropagation(); });
    container.appendChild(tagsContainer);
    container._tagsContainer = tagsContainer;
    container._titleLabel = titleLabel;
    return container;
}

// --- Samplers / Schedulers ---
// Defaults used until the live list arrives from ComfyUI.
const _DEFAULT_SAMPLERS = [
    "euler","euler_cfg_pp","euler_ancestral","euler_ancestral_cfg_pp",
    "heun","heunpp2","dpm_2","dpm_2_ancestral","lms","dpm_fast",
    "dpm_adaptive","dpmpp_2s_ancestral","dpmpp_sde","dpmpp_sde_gpu",
    "dpmpp_2m","dpmpp_2m_sde","dpmpp_2m_sde_gpu","dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu","ddpm","lcm","ipndm","ipndm_v","deis","ddim",
    "uni_pc","uni_pc_bh2",
];
const _DEFAULT_SCHEDULERS = [
    "normal","karras","exponential","sgm_uniform","simple","ddim_uniform","beta",
];
let SAMPLERS = [..._DEFAULT_SAMPLERS];
let SCHEDULERS = [..._DEFAULT_SCHEDULERS];

// Fetch the authoritative sampler/scheduler lists from the running ComfyUI
// instance.  This picks up RES4LYF, ComfyUI-Extra-Samplers, etc.
let _samplersFetched = false;
async function _fetchSamplerSchedulerLists() {
    if (_samplersFetched) return;
    _samplersFetched = true;
    try {
        const resp = await api.fetchApi("/object_info/KSampler");
        if (!resp.ok) return;
        const data = await resp.json();
        const ks = data?.KSampler?.input?.required;
        if (ks?.sampler_name?.[0]?.length) SAMPLERS = ks.sampler_name[0];
        if (ks?.scheduler?.[0]?.length) SCHEDULERS = ks.scheduler[0];
    } catch { /* keep defaults */ }
}

// Update a <select> element's options list while preserving the current value.
function _refreshSelectOptions(selectEl, options) {
    if (!selectEl) return;
    const cur = selectEl.value;
    selectEl.innerHTML = "";
    for (const opt of options) {
        const o = document.createElement("option");
        o.value = opt; o.textContent = opt;
        selectEl.appendChild(o);
    }
    // Restore previous value if it still exists, otherwise keep first
    if (options.includes(cur)) selectEl.value = cur;
}
const CONTROL_MODES = ["fixed", "increment", "decrement", "randomize"];

// --- Per-family sampler defaults (applied on manual family switch) ---
// Edit values here to change what each family starts with.
const FAMILY_DEFAULTS = {
    sdxl:          { steps_a: 20, cfg: 6.0,  sampler: "euler",         scheduler: "normal" },
    sd15:          { steps_a: 20, cfg: 6.0,  sampler: "euler",         scheduler: "normal" },
    flux1:         { steps_a: 20, cfg: 1.0,  sampler: "euler",         scheduler: "simple" },
    flux2:         { steps_a: 20, cfg: 1.0,  sampler: "euler",         scheduler: "simple" },
    zimage:        { steps_a: 20, cfg: 1.0,  sampler: "euler",         scheduler: "simple" },
    ltxv:          { steps_a: 20, cfg: 1.0,  sampler: "euler",         scheduler: "simple" },
    wan_image:     { steps_a: 30, cfg: 1.0,  sampler: "lcm",           scheduler: "simple" },
    wan_video_t2v: { steps_a: 3,  cfg: 1.0,  sampler: "lcm",           scheduler: "simple",
                     steps_b: 3 },
    wan_video_i2v: { steps_a: 3,  cfg: 1.0,  sampler: "lcm",           scheduler: "simple",
                     steps_b: 3 },
    qwen_image:    { steps_a: 20, cfg: 5.0,  sampler: "euler",         scheduler: "simple" },
};

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
        if (rows.steps_a?._setOriginal) rows.steps_a._setOriginal(s.steps_a ?? s.steps ?? 20);
        if (rows.cfg?._setOriginal) rows.cfg._setOriginal(s.cfg ?? 5.0);
        if (rows.sampler?._setOriginal) rows.sampler._setOriginal(s.sampler_name ?? "euler");
        if (rows.scheduler?._setOriginal) rows.scheduler._setOriginal(s.scheduler ?? "simple");
        if (rows.seed_a?._setOriginal) rows.seed_a._setOriginal(s.seed_a ?? s.seed ?? 0);
        if (rows.seed_b?._setOriginal) rows.seed_b._setOriginal(s.seed_b ?? s.seed_a ?? s.seed ?? 0);
        // WAN Video dual steps
        if (rows.steps_b?._setOriginal) rows.steps_b._setOriginal(s.steps_b ?? s.steps_a ?? s.steps ?? 3);
    }

    // Resolution — skip width/height/batch if locked
    const r = d.resolution || {};
    if (node._weResRows) {
        const rr = node._weResRows;
        if (!node._weResLocked) {
            // Reset ratio to Freeform so the incoming values aren't constrained
            if (node._weRatioSel) {
                node._weRatioSel.value = "0";
                node._weRatio = "Freeform";
                if (node._weSetRatioIdx) node._weSetRatioIdx(0);
            }
            if (rr.width?._setOriginal) rr.width._setOriginal(r.width ?? 768);
            if (rr.height?._setOriginal) rr.height._setOriginal(r.height ?? 1280);
            if (rr.batch?._setOriginal) rr.batch._setOriginal(r.batch_size ?? 1);
        }
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
        console.warn("[WorkflowBuilder] Error checking LoRA availability:", e);
    }
}

// --- WAN-specific visibility ---
function updateWanVisibility(node) {
    const family = node._weFamily;
    const isWanVideo = (family === "wan_video_i2v" || family === "wan_video_t2v");
    const isWan      = isWanVideo || (family === "wan_image");

    // LoRA stack titles
    if (node._weLoraACard?._titleLabel) {
        node._weLoraACard._titleLabel.textContent = isWanVideo ? "LoRA Stack (High)" : "LoRA Stack";
    }
    if (node._weLoraB?._titleLabel) {
        node._weLoraB._titleLabel.textContent = isWanVideo ? "LoRA Stack (Low)" : "LoRA Stack B";
    }
    // LoRA Stack B section: only for WAN Video
    if (node._weLoraB) {
        node._weLoraB.style.display = isWanVideo ? "" : "none";
    }
    // LoRA Stack A container: taller when alone, normal when both visible
    if (node._weLoraAContainer) {
        node._weLoraAContainer.style.height = isWanVideo ? "100px" : "160px";
    }

    // Model A label: "Model A" for WAN Video, "Model" otherwise
    if (node._weModelRow?._label) {
        node._weModelRow._label.textContent = isWanVideo ? "Model A" : "Model";
    }

    // Frames row: only for WAN Video (not WAN Image)
    if (node._weResRows?.frames) {
        node._weResRows.frames.style.display = isWanVideo ? "flex" : "none";
    }

    // Model B: only for WAN Video (dual sampler)
    if (node._weModelBRow) {
        node._weModelBRow.style.display = isWanVideo ? "flex" : "none";
    }

    // Steps B row: only for WAN Video
    if (node._weSamplerRows?.steps_b) {
        node._weSamplerRows.steps_b.style.display = isWanVideo ? "flex" : "none";
    }
    // Steps A label: "Steps A" for WAN Video, "Steps" otherwise
    if (node._weSamplerRows?.steps_a?._label) {
        node._weSamplerRows.steps_a._label.textContent = isWanVideo ? "Steps A" : "Steps";
    }
    // Seed B: only for WAN Video
    if (node._weSamplerRows?.seed_b) {
        node._weSamplerRows.seed_b.style.display = isWanVideo ? "flex" : "none";
    }
    // Seed A label: "Seed A" for WAN Video, "Seed" otherwise
    if (node._weSamplerRows?.seed_a?._label) {
        node._weSamplerRows.seed_a._label.textContent = isWanVideo ? "Seed A" : "Seed";
    }

}

/** Merge two LoRA lists, dedup by name (second list wins on conflict). */
function _mergeLoraLists(listA, listB) {
    const byName = new Map();
    for (const l of listA) byName.set(l.name || "", l);
    for (const l of listB) byName.set(l.name || "", l);
    return [...byName.values()];
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

    // Freeze all section bodies except loras (always interactive)
    for (const [name, sec] of Object.entries(node._weSections || {})) {
        if (sec._body) {
            // LoRA section stays interactive so users can toggle/adjust strengths
            if (name === "loras") continue;
            sec._body.style.opacity = opacity;
            sec._body.style.pointerEvents = pointer;
            // Skip prompt textareas — they handle their own ghosting
            // to avoid double-opacity (0.6 * 0.5 = 0.3)
            if (node._wePosBox && sec._body.contains(node._wePosBox)) {
                node._wePosBox.style.opacity = "1";
                node._wePosBox.style.pointerEvents = "auto";
            }
            if (node._weNegBox && sec._body.contains(node._weNegBox)) {
                node._weNegBox.style.opacity = "1";
                node._weNegBox.style.pointerEvents = "auto";
            }
        }
    }
    // Re-apply prompt ghosting so it uses its own single opacity
    if (node._updatePromptGhosting) node._updatePromptGhosting();
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
    const ov = { ...node._weOverrides };
    // Always capture DOM values as overrides
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
        if (v) ov.vae = v; else delete ov.vae;
    }
    if (node._weClipRow?._getValue) {
        const v = node._weClipRow._getValue();
        if (v) ov.clip_names = [v]; else delete ov.clip_names;
    }
    if (node._weSamplerRows) {
        const r = node._weSamplerRows;
        if (r.steps_a?._inp) ov.steps_a = parseInt(r.steps_a._inp.value) || 20;
        if (r.cfg?._inp) ov.cfg = parseFloat(r.cfg._inp.value) || 5.0;
        if (r.seed_a?._inp) ov.seed_a = parseInt(r.seed_a._inp.value) || 0;
        if (r.seed_b?._inp && r.seed_b.style.display !== "none") {
            ov.seed_b = parseInt(r.seed_b._inp.value) || 0;
        }
        if (r.sampler?._inp) ov.sampler_name = r.sampler._inp.value;
        if (r.scheduler?._inp) ov.scheduler = r.scheduler._inp.value;
        // WAN Video dual steps
        if (r.steps_b?._inp && r.steps_b.style.display !== "none") {
            ov.steps_b = parseInt(r.steps_b._inp.value) || 3;
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
    // Persist resolution UI state
    if (node._weRatio) ov._ratio = node._weRatio;
    if (node._weRatioLandscape) ov._ratio_landscape = true;
    if (node._weResLocked) ov._res_locked = true;
    if (node._wePosBox) ov.positive_prompt = node._wePosBox.value;
    if (node._weNegBox) ov.negative_prompt = node._weNegBox.value;
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
    if (node._weWorkflowLoras) node.properties.we_workflow_loras = JSON.stringify(node._weWorkflowLoras);
    if (node._weInputLoras) node.properties.we_input_loras = JSON.stringify(node._weInputLoras);
    if (node._weWorkflowPrompts) node.properties.we_workflow_prompts = JSON.stringify(node._weWorkflowPrompts);

    // If no extracted cache yet (user edited manually before first execute),
    // build a minimal one from overrides so onConfigure can restore the UI.
    if (!node._weExtracted && Object.keys(ov).length > 0) {
        const isVideo = ["wan_video_t2v", "wan_video_i2v"].includes(ov._family);
        node._weExtracted = {
            positive_prompt: ov.positive_prompt || "",
            negative_prompt: ov.negative_prompt || "",
            model_a: ov.model_a || "",
            model_b: ov.model_b || "",
            model_family: ov._family || "sdxl",
            model_family_label: "",
            vae: { name: ov.vae || "", source: "manual" },
            clip: { names: ov.clip_names || [], type: "", source: "manual" },
            sampler: {
                steps_a: ov.steps_a || 20, cfg: ov.cfg || 5.0,
                seed_a: ov.seed_a || 0, seed_b: ov.seed_b,
                sampler_name: ov.sampler_name || "euler",
                scheduler: ov.scheduler || "simple",
                steps_b: ov.steps_b,
            },
            resolution: {
                width: ov.width || 768, height: ov.height || 1280,
                batch_size: ov.batch_size || 1, length: isVideo ? (ov.length || 81) : undefined,
            },
            loras_a: [], loras_b: [],
            is_video: isVideo,
        };
    }
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
        // For selects, ensure the value exists as an option
        if (inp.tagName === "SELECT" && ![...inp.options].some(o => o.value === String(ovVal))) {
            const o = document.createElement("option");
            o.value = ovVal; o.textContent = ovVal;
            inp.appendChild(o);
        }
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
        if (ov.steps_a != null) applyInput(rows.steps_a, ov.steps_a);
        if (ov.cfg != null) applyInput(rows.cfg, ov.cfg);
        if (ov.seed_a != null) applyInput(rows.seed_a, ov.seed_a);
        if (ov.seed_b != null) applyInput(rows.seed_b, ov.seed_b);
        if (ov.sampler_name) applyInput(rows.sampler, ov.sampler_name);
        if (ov.scheduler) applyInput(rows.scheduler, ov.scheduler);
        if (ov.steps_b != null) applyInput(rows.steps_b, ov.steps_b);
    }

    if (node._weResRows) {
        const rr = node._weResRows;
        if (ov.width != null) applyInput(rr.width, ov.width);
        if (ov.height != null) applyInput(rr.height, ov.height);
        if (ov.batch_size != null) applyInput(rr.batch, ov.batch_size);
        if (ov.length != null) applyInput(rr.frames, ov.length);
    }
    // Restore resolution UI state (ratio, landscape, lock)
    if (node._weRatioSel) {
        if (ov._ratio_landscape) {
            node._weRatioLandscape = true;
            if (node._weSetLandscape) node._weSetLandscape(true);
        }
        if (ov._ratio) {
            // Find matching ratio index by label
            const opts = node._weRatioSel.options;
            for (let i = 0; i < opts.length; i++) {
                if (opts[i].textContent === ov._ratio) {
                    node._weRatioSel.value = String(i);
                    node._weRatio = ov._ratio;
                    if (node._weSetRatioIdx) node._weSetRatioIdx(i);
                    break;
                }
            }
        }
    }
    if (ov._res_locked && node._weResLockBtn) {
        node._weResLocked = true;
        if (node._weSetResDisabled) node._weSetResDisabled(true);
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
                steps_a: 20, cfg: 5.0, seed_a: 0,
                sampler_name: "euler", scheduler: "simple",
            },
            resolution: d.resolution || { width: 768, height: 1280, batch_size: 1, length: null },
            model_family: d.family || "",
        };
    } catch (e) {
        console.warn("[WorkflowBuilder] Could not parse workflow_data:", e);
        return null;
    }
}


// ============================================================
// --- Main extension ---
// ============================================================
app.registerExtension({
    name: "WorkflowBuilder",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WorkflowBuilder") return;

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
                // paddingBottom: 12px small buffer at the bottom of the node.
                padding: "6px 6px 12px 6px", marginTop: "-10px",
                width: "100%", boxSizing: "border-box",
                fontFamily: "Inter, system-ui, -apple-system, sans-serif",
                overflow: "hidden",
            });
            // Forward wheel to canvas for zoom — interactive elements
            // (inputs, textareas, lora containers) stopPropagation so
            // their events never reach this handler.
            forwardWheelToCanvas(root);
            node._weRoot = root;
            node._weRoot = root;

            // -- "Update Workflow" button — pull data from PromptExtractor --
            const updateBtn = makeEl("button", {
                width: "100%", padding: "6px 0",
                background: C.accent, color: "#fff", border: "none",
                borderRadius: "4px", cursor: "pointer",
                fontWeight: "bold", fontSize: "13px",
                fontFamily: "inherit", marginBottom: "2px",
            }, "\u{1F504} Update Workflow");
            // Hide button initially if workflow_data not connected
            const wfConn = node.inputs?.find(i => i.name === "workflow_data");
            if (!wfConn || wfConn.link == null) updateBtn.style.display = "none";
            node._weUpdateBtn = updateBtn;
            updateBtn.onmouseenter = () => { updateBtn.style.background = C.accentDim; };
            updateBtn.onmouseleave = () => { updateBtn.style.background = C.accent; };
            updateBtn.onclick = async () => {
                // 1. Find a PromptExtractor node — prefer one connected via workflow_data input
                let peNode = null;
                const wfInput = node.inputs?.find(i => i.name === "workflow_data");
                if (wfInput?.link != null) {
                    const linkInfo = node.graph?.links?.[wfInput.link];
                    if (linkInfo) {
                        const srcNode = node.graph?.getNodeById(linkInfo.origin_id);
                        if (srcNode?.comfyClass === "PromptExtractor" || srcNode?.comfyClass === "WorkflowExtractor") {
                            peNode = srcNode;
                        }
                    }
                }
                // Fallback: find any PromptExtractor or WorkflowExtractor on the graph
                if (!peNode && app.graph?._nodes) {
                    for (const n of app.graph._nodes) {
                        if (n.comfyClass === "PromptExtractor" || n.comfyClass === "WorkflowExtractor" ||
                            n.type === "PromptExtractor" || n.type === "WorkflowExtractor") {
                            peNode = n;
                            break;
                        }
                    }
                }
                if (!peNode) {
                    _showError(node, "No PromptExtractor or WorkflowExtractor node found on the graph.");
                    return;
                }

                updateBtn.disabled = true;
                updateBtn.textContent = "\u23F3 Fetching\u2026";
                try {
                    let extracted = null;

                    // Determine PE's current file for staleness check
                    const peImageW = peNode.widgets?.find(w => w.name === "image");
                    const peSourceW = peNode.widgets?.find(w => w.name === "source_folder");
                    const peFilename = peImageW?.value || "";
                    const peSource = peSourceW?.value || "input";

                    // 2a. Try execution cache (includes merged LoRA inputs)
                    //     Only use if the cached file matches the current selection.
                    try {
                        const cacheResp = await fetch(
                            `/prompt-extractor/get-extracted-data?node_id=${encodeURIComponent(peNode.id)}`
                        );
                        const cacheData = await cacheResp.json();
                        if (cacheData.extracted) {
                            const cachedFile = cacheData.extracted._source_file || "";
                            const cachedFolder = cacheData.extracted._source_folder || "input";
                            if (cachedFile === peFilename && cachedFolder === peSource) {
                                extracted = cacheData.extracted;
                            } else {
                                console.log("[WorkflowBuilder] Execution cache stale — file changed");
                            }
                        }
                    } catch { /* fall through */ }

                    // 2b. Try client-side cache (set by PE JS on file selection)
                    if (!extracted) {
                        const cachedStr = peNode.properties?.pe_extracted_data;
                        if (cachedStr) {
                            try { extracted = JSON.parse(cachedStr); } catch { extracted = null; }
                        }
                    }

                    // 2c. Fallback: call extract-preview API
                    if (!extracted) {
                        if (!peFilename || peFilename === "(none)") {
                            _showError(node, "PromptExtractor has no file selected.");
                            return;
                        }
                        const resp = await fetch(
                            `/prompt-extractor/extract-preview?filename=${encodeURIComponent(peFilename)}&source=${encodeURIComponent(peSource)}`
                        );
                        const data = await resp.json();
                        extracted = data.extracted;
                        if (!extracted) {
                            _showError(node, data.error || "No metadata found in selected file.");
                            return;
                        }
                    }

                    // 3. Process through WB Python (matches execute() behavior)
                    const processResp = await fetch("/workflow-builder/process-extracted", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            extracted,
                            family_override: node._weFamily || null,
                        }),
                    });
                    const processData = await processResp.json();
                    if (processData.error) {
                        _showError(node, processData.error);
                        return;
                    }
                    const processed = processData.extracted || extracted;

                    // 4. Populate the UI
                    _showError(node, null);

                    // If prompts are connected, preserve current prompt values
                    const posConn = node.inputs?.find(i => i.name === "positive_prompt");
                    const negConn = node.inputs?.find(i => i.name === "negative_prompt");
                    // Save workflow prompts before overwriting
                    node._weWorkflowPrompts = {
                        positive: processed.positive_prompt || "",
                        negative: processed.negative_prompt || "",
                    };
                    if (posConn?.link != null) {
                        processed.positive_prompt = node._wePosBox?.value || "";
                    }
                    if (negConn?.link != null) {
                        processed.negative_prompt = node._weNegBox?.value || "";
                    }

                    node._weExtracted = processed;
                    node._wePopulated = true;
                    // Track workflow LoRAs separately; merge with existing input LoRAs
                    // (only if those inputs are still connected)
                    node._weWorkflowLoras = {
                        a: [...(processed.loras_a || [])],
                        b: [...(processed.loras_b || [])],
                    };
                    const loraAConn = node.inputs?.find(i => i.name === "lora_stack_a");
                    const loraBConn = node.inputs?.find(i => i.name === "lora_stack_b");
                    if (!node._weInputLoras) node._weInputLoras = { a: [], b: [] };
                    if (loraAConn?.link == null) node._weInputLoras.a = [];
                    if (loraBConn?.link == null) node._weInputLoras.b = [];
                    node._weExtracted.loras_a = _mergeLoraLists(processed.loras_a || [], node._weInputLoras.a);
                    node._weExtracted.loras_b = _mergeLoraLists(processed.loras_b || [], node._weInputLoras.b);
                    node._weLoraState = {};
                    node._weOverrides = {};
                    node.properties = node.properties || {};
                    node.properties.we_extracted_cache = JSON.stringify(processed);
                    delete node.properties.we_override_data;
                    delete node.properties.we_lora_state;
                    await updateUI(node);
                    syncHidden(node);
                    node.setDirtyCanvas(true, true);
                    app.graph.setDirtyCanvas(true, true);
                } catch (e) {
                    console.error("[WorkflowBuilder] Update Workflow error:", e);
                    _showError(node, "Failed to fetch data from PromptExtractor.");
                } finally {
                    updateBtn.disabled = false;
                    updateBtn.textContent = "\u{1F504} Update Workflow";
                }
            };
            root.appendChild(updateBtn);

            // -- 1. RESOLUTION section (open) --
            const resSec = makeSection("RESOLUTION");
            node._weSections.resolution = resSec;

            // SVG lock icons (white, clean outline)
            const _lockSvgOpen = `<svg width="10" height="12" viewBox="0 0 20 24" fill="none" stroke="rgba(255,255,255,0.4)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="14" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/></svg>`;
            const _lockSvgClosed = `<svg width="10" height="12" viewBox="0 0 20 24" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="14" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>`;

            const RES_GUTTER = "18px";

            // Aspect ratio definitions — stored as portrait (w < h)
            const RATIOS = [
                { w: 0,  h: 0  },   // Freeform
                { w: 1,  h: 1  },
                { w: 4,  h: 5  },
                { w: 3,  h: 4  },
                { w: 2,  h: 3  },
                { w: 9,  h: 16 },
            ];
            function _ratioLabel(r, land) {
                if (r.w === 0) return "Freeform";
                return land ? `${r.h}:${r.w}` : `${r.w}:${r.h}`;
            }

            let _landscape = false;
            let _resLocked = false;
            let _currentRatioIdx = 0;
            node._weResLocked = false;
            node._weRatioLandscape = false;
            node._weRatio = "Freeform";

            // Lock icon (created early so _setResDisabled can reference it)
            const lockIcon = makeEl("span", {
                cursor: "pointer", display: "inline-flex", alignItems: "center",
                justifyContent: "center", width: RES_GUTTER, flexShrink: "0",
            });
            lockIcon.innerHTML = _lockSvgOpen;
            lockIcon.title = "Lock resolution";
            lockIcon.onclick = () => {
                _resLocked = !_resLocked;
                node._weResLocked = _resLocked;
                _setResDisabled(_resLocked);
                _syncS();
            };

            // Helper: ghost / un-ghost resolution inputs + update lock icon
            function _setResDisabled(disabled) {
                if (!resRows) return;
                for (const key of ["width", "height", "batch", "frames"]) {
                    const inp = resRows[key]?._inp;
                    if (inp) { inp.disabled = disabled; inp.style.opacity = disabled ? "0.35" : "1"; }
                    const lbl = resRows[key]?._label;
                    if (lbl) lbl.style.opacity = disabled ? "0.35" : "1";
                }
                lockIcon.innerHTML = disabled ? _lockSvgClosed : _lockSvgOpen;
            }

            // Reduced label width for res rows so gutter + label = LABEL_W
            const RES_LABEL_W = `calc(${LABEL_W} - ${RES_GUTTER})`;

            // --- Ratio row: [label] [orient icon] [reset spacer] [dropdown] ---
            const ratioRow = makeEl("div", { ...ROW_STYLE });
            ratioRow.appendChild(makeEl("span", {
                color: C.textMuted, width: RES_LABEL_W, flexShrink: "0",
            }, "Ratio"));
            // Orient icon in the gutter column (after label, before reset spacer)
            const orientIcon = makeEl("span", {
                cursor: "pointer", display: "inline-flex", alignItems: "center",
                justifyContent: "center", width: RES_GUTTER, flexShrink: "0",
            });
            function _drawOrient() {
                if (_landscape) {
                    orientIcon.innerHTML = `<svg width="14" height="11" viewBox="0 0 14 11" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x=".5" y=".5" width="13" height="10" rx="1" stroke="rgba(255,255,255,0.55)" stroke-width="1"/><circle cx="7" cy="3.8" r="1.4" fill="rgba(255,255,255,0.45)"/><path d="M4.5 9.5v-.8c0-1.1.9-2 2-2h1c1.1 0 2 .9 2 2v.8" stroke="rgba(255,255,255,0.45)" stroke-width=".9" stroke-linecap="round" fill="none"/></svg>`;
                } else {
                    orientIcon.innerHTML = `<svg width="10" height="14" viewBox="0 0 10 14" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x=".5" y=".5" width="9" height="13" rx="1" stroke="rgba(255,255,255,0.55)" stroke-width="1"/><circle cx="5" cy="4.5" r="1.5" fill="rgba(255,255,255,0.45)"/><path d="M2.5 12.5v-1c0-1.2 1-2.2 2.2-2.2h.6c1.2 0 2.2 1 2.2 2.2v1" stroke="rgba(255,255,255,0.45)" stroke-width=".9" stroke-linecap="round" fill="none"/></svg>`;
                }
                orientIcon.title = _landscape ? "Landscape — click for portrait" : "Portrait — click for landscape";
            }
            _drawOrient();
            orientIcon.onclick = () => {
                _landscape = !_landscape;
                node._weRatioLandscape = _landscape;
                _drawOrient();
                // Swap width ↔ height so 1920×1080 becomes 1080×1920
                const oldW = resRows.width._inp.value;
                const oldH = resRows.height._inp.value;
                resRows.width._inp.value  = oldH;
                resRows.height._inp.value = oldW;
                _populateRatioSel();
                _syncS();
            };
            ratioRow.appendChild(orientIcon);
            ratioRow.appendChild(makeEl("span", { width: "14px", flexShrink: "0" }));

            // Ratio dropdown (same INPUT_W as other inputs)
            const ratioSel = document.createElement("select");
            function _populateRatioSel() {
                ratioSel.innerHTML = "";
                for (let i = 0; i < RATIOS.length; i++) {
                    const o = document.createElement("option");
                    o.value = String(i); o.textContent = _ratioLabel(RATIOS[i], _landscape);
                    ratioSel.appendChild(o);
                }
                ratioSel.value = String(_currentRatioIdx);
            }
            _populateRatioSel();
            Object.assign(ratioSel.style, { ...INPUT_STYLE });
            ratioRow.appendChild(ratioSel);

            // --- Ratio logic ---
            function _getRatio() { return RATIOS[_currentRatioIdx] || RATIOS[0]; }
            function _applyRatio() {
                const r = _getRatio();
                if (r.w === 0) return;
                const rw = _landscape ? r.h : r.w, rh = _landscape ? r.w : r.h;
                const curW = parseInt(resRows.width._inp.value) || 768;
                const curH = parseInt(resRows.height._inp.value) || 1280;
                let newW, newH;
                if (_landscape) { newW = curW; newH = Math.round(curW * rh / rw / 8) * 8; }
                else            { newH = curH; newW = Math.round(curH * rw / rh / 8) * 8; }
                resRows.width._inp.value  = Math.max(64, Math.min(8192, newW));
                resRows.height._inp.value = Math.max(64, Math.min(8192, newH));
                _syncS();
            }
            function _applyRatioFromWidth() {
                const r = _getRatio(); if (r.w === 0) return;
                const rw = _landscape ? r.h : r.w, rh = _landscape ? r.w : r.h;
                const curW = parseInt(resRows.width._inp.value) || 768;
                resRows.height._inp.value = Math.max(64, Math.min(8192, Math.round(curW * rh / rw / 8) * 8));
                _syncS();
            }
            function _applyRatioFromHeight() {
                const r = _getRatio(); if (r.w === 0) return;
                const rw = _landscape ? r.h : r.w, rh = _landscape ? r.w : r.h;
                const curH = parseInt(resRows.height._inp.value) || 1280;
                resRows.width._inp.value = Math.max(64, Math.min(8192, Math.round(curH * rw / rh / 8) * 8));
                _syncS();
            }
            ratioSel.onchange = () => {
                _currentRatioIdx = parseInt(ratioSel.value) || 0;
                node._weRatio = _ratioLabel(RATIOS[_currentRatioIdx], _landscape);
                _applyRatio(); _syncS();
            };

            // --- Input rows ---
            const resRows = {
                width:  makeInput("Width",  "number", 768,  { min: 64, max: 8192, step: 8 }, () => { _applyRatioFromWidth();  _syncS(); }),
                height: makeInput("Height", "number", 1280, { min: 64, max: 8192, step: 8 }, () => { _applyRatioFromHeight(); _syncS(); }),
                batch:  makeInput("Batch",  "number", 1,    { min: 1, max: 128, step: 1 }, _syncS),
                frames: makeInput("Frames", "number", 81,   { min: 1, max: 1000, step: 1 }, _syncS),
            };
            resRows.frames.style.display = "none";

            // Insert gutter elements AFTER label, BEFORE resetBtn in each row
            // makeInput creates: [label] [resetBtn] [input]
            // We insert between label and resetBtn: line segment or lock icon
            const _lineSvg = `<svg width="2" height="100%" viewBox="0 0 2 20" preserveAspectRatio="none"><line x1="1" y1="0" x2="1" y2="20" stroke="${C.border}" stroke-width="1"/></svg>`;
            function _makeLineGutter() {
                const g = makeEl("span", {
                    display: "inline-flex", alignItems: "stretch", justifyContent: "center",
                    width: RES_GUTTER, flexShrink: "0", alignSelf: "stretch",
                });
                const line = makeEl("span", {
                    width: "0", borderLeft: `1px solid ${C.border}`, alignSelf: "stretch",
                });
                g.appendChild(line);
                return g;
            }

            // Width: line, Height: lock, Batch: line, Frames: line
            const widthGutter = _makeLineGutter();
            const batchGutter = _makeLineGutter();
            const framesGutter = _makeLineGutter();

            // Shrink labels on all res rows so gutter + label = LABEL_W
            for (const key of ["width", "height", "batch", "frames"]) {
                if (resRows[key]?._label) resRows[key]._label.style.width = RES_LABEL_W;
            }

            // Insert gutters after label (label is firstChild) in each row
            resRows.width.insertBefore(widthGutter, resRows.width._resetBtn);
            resRows.height.insertBefore(lockIcon, resRows.height._resetBtn);
            resRows.batch.insertBefore(batchGutter, resRows.batch._resetBtn);
            resRows.frames.insertBefore(framesGutter, resRows.frames._resetBtn);

            // Assemble section
            resSec._body.appendChild(ratioRow);
            for (const key of ["width", "height", "batch", "frames"]) {
                resSec._body.appendChild(resRows[key]);
            }
            root.appendChild(resSec);

            node._weResRows = resRows;
            node._weRatio = "Freeform";
            node._weRatioSel = ratioSel;
            node._weRatioLandBtn = orientIcon;
            node._weResLockBtn = lockIcon;
            node._weSetLandscape = (v) => {
                _landscape = !!v;
                node._weRatioLandscape = _landscape;
                _drawOrient();
                _populateRatioSel();
            };
            node._weSetRatioIdx = (i) => { _currentRatioIdx = i; };
            node._weSetResDisabled = _setResDisabled;

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

            // Model A row (label updated by updateWanVisibility)
            const modelRow = makeSelectRow("Model", "", fetchModels,
                (v) => { node._weOverrides.model_a = v; _syncS(); }, true);
            modelSec._body.appendChild(modelRow);
            node._weModelRow = modelRow;

            // Model B row (hidden by default, shown by updateWanVisibility for WAN Video)
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
                (v) => { if (v) node._weOverrides.vae = v; else delete node._weOverrides.vae; _syncS(); }, false);
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
                (v) => { if (v) node._weOverrides.clip_names = [v]; else delete node._weOverrides.clip_names; _syncS(); }, false);
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
                    // Apply sensible sampler defaults for the new family
                    const defs = FAMILY_DEFAULTS[familyKey];
                    if (defs && node._weSamplerRows) {
                        const rows = node._weSamplerRows;
                        if (defs.steps_a != null    && rows.steps_a?._inp)    rows.steps_a._inp.value = defs.steps_a;
                        if (defs.steps_b != null    && rows.steps_b?._inp)    rows.steps_b._inp.value = defs.steps_b;
                        if (defs.cfg != null         && rows.cfg?._inp)        rows.cfg._inp.value = defs.cfg;
                        if (defs.sampler             && rows.sampler?._inp)    rows.sampler._inp.value = defs.sampler;
                        if (defs.scheduler           && rows.scheduler?._inp)  rows.scheduler._inp.value = defs.scheduler;
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

            // Steps A (always visible, labeled "Steps" for non-WAN, "Steps A" for WAN Video)
            const stepsARow    = makeInput("Steps",        "number", 20, { min: 1, max: 200, step: 1 }, _syncS);
            // Steps B — WAN Video low-pass (hidden by default)
            const stepsBRow    = makeInput("Steps B",      "number", 3,  { min: 1, max: 200, step: 1 }, _syncS);
            stepsBRow.style.display = "none";

            // Seed rows (labels updated by updateWanVisibility)
            const seedRow  = makeInput("Seed", "number", 0, { min: 0, step: 1 }, _syncS);
            const seedBRow = makeInput("Seed B", "number", 0, { min: 0, step: 1 }, _syncS);
            seedBRow.style.display = "none";

            const sampRows = {
                steps_a:    stepsARow,
                steps_b:    stepsBRow,
                cfg:        makeInput("CFG",       "number", 5.0,      { min: 0, max: 100, step: 0.5 }, _syncS),
                sampler:    makeInput("Sampler",   "select", "euler",  { options: SAMPLERS }, _syncS),
                scheduler:  makeInput("Scheduler", "select", "simple", { options: SCHEDULERS }, _syncS),
                seed_a:     seedRow,
                seed_b:     seedBRow,
            };
            // Append in display order: steps_a, steps_b, cfg, sampler, scheduler, seed, seed_b
            sampSec._body.appendChild(stepsARow);
            sampSec._body.appendChild(stepsBRow);
            sampSec._body.appendChild(sampRows.cfg);
            sampSec._body.appendChild(sampRows.sampler);
            sampSec._body.appendChild(sampRows.scheduler);
            sampSec._body.appendChild(seedRow);
            sampSec._body.appendChild(seedBRow);

            // Control after generate
            const controlRow = makeInput("Control after generate", "select", "fixed",
                { options: CONTROL_MODES }, _syncS);
            sampSec._body.appendChild(controlRow);
            node._weControlMode = controlRow;

            root.appendChild(sampSec);
            node._weSamplerRows = sampRows;

            // Fetch live sampler/scheduler lists and update dropdowns
            _fetchSamplerSchedulerLists().then(() => {
                _refreshSelectOptions(sampRows.sampler?._inp, SAMPLERS);
                _refreshSelectOptions(sampRows.scheduler?._inp, SCHEDULERS);
            });

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
            posBox.addEventListener("wheel", (e) => { e.stopPropagation(); });
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
            negBox.addEventListener("wheel", (e) => { e.stopPropagation(); });
            negSec._body.appendChild(negBox);
            root.appendChild(negSec);
            node._weNegBox = negBox;

            // -- 5. LORAS section --
            const loraSec = makeSection("LORAS");
            node._weSections.loras = loraSec;

            // Stack A card
            const loraACard = createLoraStackContainer("LoRA Stack A",
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
            loraBCard.style.display = "flex"; // always visible
            loraSec._body.appendChild(loraBCard);
            node._weLoraBContainer = loraBCard._tagsContainer;
            node._weLoraB = loraBCard;

            root.appendChild(loraSec);

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
                const h = root.scrollHeight || _MIN_H;
                return [width, h];
            };

            // -- Only hide data widgets, keep toggle switches visible --
            // IMPORTANT: override_data and lora_state must keep their original
            // STRING type so ComfyUI's graphToPrompt includes them in execution.
            // "converted-widget" type causes ComfyUI to skip the widget value.
            const KEEP_VISIBLE = new Set([]);
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
                const contentH = (root.scrollHeight || _MIN_H) + _NATIVE_H;
                size[0] = Math.max(_MIN_W, size[0]);
                size[1] = Math.max(contentH, _MIN_H, size[1]);
                if (origOnResize) return origOnResize.apply(this, arguments);
            };

            // Set initial size — deferred so the DOM has been painted and
            // scrollHeight reflects real content height.
            requestAnimationFrame(() => requestAnimationFrame(() => {
                const contentH = (root.scrollHeight || _MIN_H) + _NATIVE_H;
                node.setSize([_MIN_W, Math.max(contentH, _MIN_H)]);
                app.graph.setDirtyCanvas(true, true);
            }));

            applyZoomScaling(root);

            // -- Auto-ghost prompt textareas when inputs are connected --
            function _updatePromptGhosting() {
                const posConn = node.inputs?.find(i => i.name === "positive_prompt");
                const negConn = node.inputs?.find(i => i.name === "negative_prompt");
                const posLinked = posConn && posConn.link != null;
                const negLinked = negConn && negConn.link != null;

                if (node._wePosBox) {
                    node._wePosBox.readOnly = posLinked;
                    node._wePosBox.style.opacity = posLinked ? "0.5" : "1";
                    node._wePosBox.style.pointerEvents = "auto";
                }
                if (node._weNegBox) {
                    node._weNegBox.readOnly = negLinked;
                    node._weNegBox.style.opacity = negLinked ? "0.5" : "1";
                    node._weNegBox.style.pointerEvents = "auto";
                }
            }
            node._updatePromptGhosting = _updatePromptGhosting;

            // Update ghosting and LoRA inputs when connections change
            const origConnInput = node.onConnectionsChange;
            node.onConnectionsChange = function () {
                if (origConnInput) origConnInput.apply(this, arguments);

                // Restore workflow prompts when prompt inputs are disconnected
                // Check BEFORE _updatePromptGhosting clears readOnly
                const posConn = node.inputs?.find(i => i.name === "positive_prompt");
                const negConn = node.inputs?.find(i => i.name === "negative_prompt");
                const wp = node._weWorkflowPrompts;
                if (wp) {
                    if (posConn?.link == null && node._wePosBox?.readOnly) {
                        node._wePosBox.value = wp.positive;
                    }
                    if (negConn?.link == null && node._weNegBox?.readOnly) {
                        node._weNegBox.value = wp.negative;
                    }
                }

                _updatePromptGhosting();
                if (wp) syncHidden(node);

                // Show/hide Update Workflow button based on workflow_data connection
                const wfDataConn = node.inputs?.find(i => i.name === "workflow_data");
                if (node._weUpdateBtn) {
                    node._weUpdateBtn.style.display = (wfDataConn?.link != null) ? "" : "none";
                }

                // Clear input LoRAs for disconnected stacks and re-merge
                const loraAConn = node.inputs?.find(i => i.name === "lora_stack_a");
                const loraBConn = node.inputs?.find(i => i.name === "lora_stack_b");
                let changed = false;
                if (!node._weInputLoras) node._weInputLoras = { a: [], b: [] };
                if (loraAConn?.link == null && node._weInputLoras.a.length > 0) {
                    node._weInputLoras.a = [];
                    changed = true;
                }
                if (loraBConn?.link == null && node._weInputLoras.b.length > 0) {
                    node._weInputLoras.b = [];
                    changed = true;
                }
                if (changed && node._weExtracted) {
                    const wl = node._weWorkflowLoras || { a: [], b: [] };
                    node._weExtracted.loras_a = _mergeLoraLists(wl.a, node._weInputLoras.a);
                    node._weExtracted.loras_b = _mergeLoraLists(wl.b, node._weInputLoras.b);
                    node._weLoraState = {};
                    updateLoras(node);
                    syncHidden(node);
                    node.setDirtyCanvas(true, true);
                }
            };
            // Apply initial ghosting state
            _updatePromptGhosting();

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
                applyMode(node._weSamplerRows?.seed_a);
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

                // Update availability/found flags and LoRA list from execution.
                // Never reset UI fields the user has already set.
                if (!node._weExtracted) {
                    // Very first execution on a brand-new node: seed
                    // _weExtracted with the effective info so subsequent
                    // runs have something to compare against.
                    node._weExtracted = info;
                    node._wePopulated = true;
                    node._weWorkflowLoras = {
                        a: [...(info.workflow_loras_a || info.loras_a || [])],
                        b: [...(info.workflow_loras_b || info.loras_b || [])],
                    };
                    node._weInputLoras = {
                        a: [...(info.input_loras_a || [])],
                        b: [...(info.input_loras_b || [])],
                    };
                    node.properties = node.properties || {};
                    node.properties.we_extracted_cache = JSON.stringify(info);
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
                    // Subsequent runs — keep UI as-is, only update metadata + LoRAs
                    // Track separate LoRA sources for change detection
                    const oldWl = node._weWorkflowLoras || { a: [], b: [] };
                    const oldIl = node._weInputLoras || { a: [], b: [] };
                    const oldSig = [oldWl.a, oldWl.b, oldIl.a, oldIl.b]
                        .map(l => (l || []).map(x => x.name).sort().join(",")).join("|");
                    node._weWorkflowLoras = {
                        a: [...(info.workflow_loras_a || oldWl.a)],
                        b: [...(info.workflow_loras_b || oldWl.b)],
                    };
                    node._weInputLoras = {
                        a: [...(info.input_loras_a || [])],
                        b: [...(info.input_loras_b || [])],
                    };
                    const newSig = [node._weWorkflowLoras.a, node._weWorkflowLoras.b,
                        node._weInputLoras.a, node._weInputLoras.b]
                        .map(l => (l || []).map(x => x.name).sort().join(",")).join("|");
                    // Merged lists (Python already merged them)
                    node._weExtracted.loras_a = info.loras_a || [];
                    node._weExtracted.loras_b = info.loras_b || [];
                    if (oldSig !== newSig) node._weLoraState = {};

                    if (info.lora_availability) node._weExtracted.lora_availability = info.lora_availability;
                    if (info.model_a_found !== undefined) node._weExtracted.model_a_found = info.model_a_found;
                    if (info.model_b_found !== undefined) node._weExtracted.model_b_found = info.model_b_found;
                    if (info.vae_found !== undefined) node._weExtracted.vae_found = info.vae_found;
                    node.properties = node.properties || {};
                    node.properties.we_extracted_cache = JSON.stringify(node._weExtracted);
                    updateLoras(node);
                }

                // Show connected prompt values in ghosted textareas
                const posConn = node.inputs?.find(i => i.name === "positive_prompt");
                const negConn = node.inputs?.find(i => i.name === "negative_prompt");
                if (posConn?.link != null) {
                    if (info.positive_prompt != null && node._wePosBox) {
                        node._wePosBox.value = info.positive_prompt;
                    }
                }
                if (negConn?.link != null) {
                    if (info.negative_prompt != null && node._weNegBox) {
                        node._weNegBox.value = info.negative_prompt;
                    }
                }

                node._preUpdateApplied = true;
            });

            // -- If not restoring from workflow, try initial load --
            if (!node._configuredFromWorkflow) {
                // Apply initial visibility for default family (sdxl)
                updateWanVisibility(node);
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
                if (!this._weExtracted) {
                    this._weExtracted = info;
                    this._wePopulated = true;
                    this._weWorkflowLoras = {
                        a: [...(info.workflow_loras_a || info.loras_a || [])],
                        b: [...(info.workflow_loras_b || info.loras_b || [])],
                    };
                    this._weInputLoras = {
                        a: [...(info.input_loras_a || [])],
                        b: [...(info.input_loras_b || [])],
                    };
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
                    // Subsequent runs — always update LoRAs from execution
                    const oldWl = this._weWorkflowLoras || { a: [], b: [] };
                    const oldIl = this._weInputLoras || { a: [], b: [] };
                    const oldSig = [oldWl.a, oldWl.b, oldIl.a, oldIl.b]
                        .map(l => (l || []).map(x => x.name).sort().join(",")).join("|");
                    this._weWorkflowLoras = {
                        a: [...(info.workflow_loras_a || oldWl.a)],
                        b: [...(info.workflow_loras_b || oldWl.b)],
                    };
                    this._weInputLoras = {
                        a: [...(info.input_loras_a || [])],
                        b: [...(info.input_loras_b || [])],
                    };
                    const newSig = [this._weWorkflowLoras.a, this._weWorkflowLoras.b,
                        this._weInputLoras.a, this._weInputLoras.b]
                        .map(l => (l || []).map(x => x.name).sort().join(",")).join("|");
                    this._weExtracted.loras_a = info.loras_a || [];
                    this._weExtracted.loras_b = info.loras_b || [];
                    if (oldSig !== newSig) this._weLoraState = {};

                    if (info.lora_availability) this._weExtracted.lora_availability = info.lora_availability;
                    if (info.model_a_found !== undefined) this._weExtracted.model_a_found = info.model_a_found;
                    if (info.model_b_found !== undefined) this._weExtracted.model_b_found = info.model_b_found;
                    if (info.vae_found !== undefined) this._weExtracted.vae_found = info.vae_found;
                    this.properties = this.properties || {};
                    this.properties.we_extracted_cache = JSON.stringify(this._weExtracted);
                    updateLoras(this);
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

            // ── Migration: remove stale inputs/outputs from old workflows ──
            // LiteGraph restores slots from saved JSON; if INPUT_TYPES or
            // RETURN_TYPES changed between versions, phantom slots persist.
            const VALID_INPUTS = new Set([
                "workflow_data",
                "positive_prompt", "negative_prompt",
                "lora_stack_a", "lora_stack_b",
            ]);
            if (node.inputs) {
                for (let i = node.inputs.length - 1; i >= 0; i--) {
                    if (!VALID_INPUTS.has(node.inputs[i].name)) {
                        node.removeInput(i);
                    }
                }
            }
            const VALID_OUTPUTS = [{ name: "workflow_data", type: "WORKFLOW_DATA" }];
            if (node.outputs) {
                const namesMatch = node.outputs.length === VALID_OUTPUTS.length &&
                    VALID_OUTPUTS.every((v, i) => node.outputs[i]?.name === v.name && node.outputs[i]?.type === v.type);
                if (!namesMatch) {
                    const savedLinks = node.outputs.map(o => o.links ? [...o.links] : null);
                    node.outputs.length = 0;
                    for (let i = 0; i < VALID_OUTPUTS.length; i++) {
                        node.addOutput(VALID_OUTPUTS[i].name, VALID_OUTPUTS[i].type);
                        if (savedLinks[i]) node.outputs[i].links = savedLinks[i];
                    }
                }
            }

            // Re-hide data widgets (same logic as onNodeCreated)
            const domW = node.widgets?.find(w => w.name === "we_ui");
            const KEEP_VISIBLE = new Set([]);
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
            const savedWl = props.we_workflow_loras;
            const savedIl = props.we_input_loras;

            let cached = null;
            try { cached = JSON.parse(savedCache || "{}"); } catch { cached = null; }
            const hasCache = cached && Object.keys(cached).length > 0;

            if (hasCache) {
                node._weExtracted = cached;
                node._wePopulated = true;
                node._weLoraState = {};
                node._weOverrides = {};
                try { node._weWorkflowLoras = JSON.parse(savedWl || '{"a":[],"b":[]}'); } catch { node._weWorkflowLoras = { a: [], b: [] }; }
                try { node._weInputLoras = JSON.parse(savedIl || '{"a":[],"b":[]}'); } catch { node._weInputLoras = { a: [], b: [] }; }
                try { node._weWorkflowPrompts = JSON.parse(props.we_workflow_prompts || 'null'); } catch { node._weWorkflowPrompts = null; }
                const uiReady = updateUI(node);
                if (uiReady && typeof uiReady.then === "function") {
                    uiReady.then(() => {
                        applyOverrides(node, savedOv, savedLs);
                        if (node._updatePromptGhosting) node._updatePromptGhosting();
                        node.setDirtyCanvas(true, true);
                    });
                } else {
                    applyOverrides(node, savedOv, savedLs);
                    if (node._updatePromptGhosting) node._updatePromptGhosting();
                    node.setDirtyCanvas(true, true);
                }
            } else {
                // No extracted cache — but user may have entered values manually
                // before executing. Try to restore from override data alone.
                const hasOverrides = savedOv && savedOv !== "{}";
                if (hasOverrides) {
                    try {
                        const ov = JSON.parse(savedOv);
                        const isVideo = ["wan_video_t2v", "wan_video_i2v"].includes(ov._family);
                        node._weExtracted = {
                            positive_prompt: ov.positive_prompt || "",
                            negative_prompt: ov.negative_prompt || "",
                            model_a: ov.model_a || "",
                            model_b: ov.model_b || "",
                            model_family: ov._family || "sdxl",
                            model_family_label: "",
                            vae: { name: ov.vae || "", source: "manual" },
                            clip: { names: ov.clip_names || [], type: "", source: "manual" },
                            sampler: {
                                steps_a: ov.steps_a || 20, cfg: ov.cfg || 5.0,
                                seed_a: ov.seed_a || 0, seed_b: ov.seed_b,
                                sampler_name: ov.sampler_name || "euler",
                                scheduler: ov.scheduler || "simple",
                                steps_b: ov.steps_b,
                            },
                            resolution: {
                                width: ov.width || 768, height: ov.height || 1280,
                                batch_size: ov.batch_size || 1,
                                length: isVideo ? (ov.length || 81) : undefined,
                            },
                            loras_a: [], loras_b: [],
                            is_video: isVideo,
                        };
                        node._wePopulated = true;
                        const uiReady = updateUI(node);
                        if (uiReady && typeof uiReady.then === "function") {
                            uiReady.then(() => {
                                applyOverrides(node, savedOv, savedLs);
                                if (node._updatePromptGhosting) node._updatePromptGhosting();
                                node.setDirtyCanvas(true, true);
                            });
                        } else {
                            applyOverrides(node, savedOv, savedLs);
                            if (node._updatePromptGhosting) node._updatePromptGhosting();
                            node.setDirtyCanvas(true, true);
                        }
                    } catch {
                        // Fall through to basic restore
                    }
                }
                if (node._updatePromptGhosting) node._updatePromptGhosting();
                node.setDirtyCanvas(true, true);
            }

            // Show/hide Update Workflow button based on workflow_data connection
            const wfSlot = node.inputs?.find(i => i.name === "workflow_data");
            if (node._weUpdateBtn) {
                node._weUpdateBtn.style.display = (wfSlot?.link != null) ? "" : "none";
            }
        };
    },
});
