/**
 * Workflow Builder - Full DOM-based UI
 *
 * Layout order: Resolution -> Model/VAE/CLIP -> Prompts -> Sampler -> LoRAs
 * Accepts recipe_data input (from PromptExtractor) + optional lora_stack / prompt / image inputs.
 * recipe_data is applied during normal workflow execution.
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

function withModelFilteringPreference(url, showAllModels) {
    if (!showAllModels) return url;
    return `${url}${url.includes("?") ? "&" : "?"}show_all=1`;
}

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

const UI_ICON_STROKE = 1.6;
const LOCK_ICON_STROKE = 2.7;
const UI_ICON_COLOR = "rgba(255,255,255,0.85)";
const UI_ICON_COLOR_DIM = "rgba(255,255,255,0.72)";
const SECTION_LOCK_SVG_OPEN = `<svg width="14" height="16" viewBox="0 0 20 24" fill="none" stroke="${UI_ICON_COLOR}" stroke-width="${LOCK_ICON_STROKE}" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="14" height="10" rx="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/></svg>`;
const SECTION_LOCK_SVG_CLOSED = `<svg width="14" height="16" viewBox="0 0 20 24" fill="none" stroke="${UI_ICON_COLOR}" stroke-width="${LOCK_ICON_STROKE}" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="14" height="10" rx="2" fill="${UI_ICON_COLOR}" stroke="${UI_ICON_COLOR}"/><path d="M6 11V7a4 4 0 0 1 8 0v4"/></svg>`;

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
const _NATIVE_H = 60;   // title bar + toggle widgets above the DOM widget
const _MIN_W = 478;
const _MIN_H = 200;

function reflowNode(node) {
    if (!node?._weRoot) return;
    const root = node._weRoot;
    let visibleBottom = 0;
    for (const child of Array.from(root.children)) {
        if (!child || child.style?.display === "none") continue;
        visibleBottom = Math.max(visibleBottom, child.offsetTop + child.offsetHeight);
    }
    const contentH = Math.ceil(Math.max(root.scrollHeight, visibleBottom) + _NATIVE_H);
    const width = Math.max(_MIN_W, node.size?.[0] || _MIN_W);
    const height = Math.max(contentH, _MIN_H);
    node.setSize([width, height]);
    node.setDirtyCanvas(true, true);
    app.graph?.setDirtyCanvas(true, true);
    requestAnimationFrame(() => {
        let visibleBottom2 = 0;
        for (const child of Array.from(root.children)) {
            if (!child || child.style?.display === "none") continue;
            visibleBottom2 = Math.max(visibleBottom2, child.offsetTop + child.offsetHeight);
        }
        const contentH2 = Math.ceil(Math.max(root.scrollHeight, visibleBottom2) + _NATIVE_H);
        const height2 = Math.max(contentH2, _MIN_H);
        if (Math.abs(height2 - (node.size?.[1] || 0)) > 1) {
            node.setSize([width, height2]);
            node.setDirtyCanvas(true, true);
            app.graph?.setDirtyCanvas(true, true);
        }
    });
}

async function normalizeSectionLayout(node, closingKey = null) {
    if (!node?._weSections) return;
    const root = node._weRoot;
    const sections = node._weSections;
    const reopenKeys = Object.entries(sections)
        .filter(([key, sec]) => key !== closingKey && sec && !sec._collapsed)
        .map(([key]) => key);

    if (root) root.style.visibility = "hidden";

    for (const sec of Object.values(sections)) {
        if (!sec?.setCollapsed || sec._collapsed) continue;
        sec.setCollapsed(true, { sync: false });
        await new Promise(requestAnimationFrame);
        reflowNode(node);
    }
    await new Promise(requestAnimationFrame);
    reflowNode(node);

    for (const key of reopenKeys) {
        const sec = sections[key];
        if (sec?.setCollapsed) sec.setCollapsed(false, { sync: false });
        await new Promise(requestAnimationFrame);
        reflowNode(node);
    }

    if (root) root.style.visibility = "";
    syncHidden(node);
    requestAnimationFrame(() => reflowNode(node));
}

// --- Section builder (collapsible + lockable) ---
function makeSection(title, node = null, key = null, opts = {}) {
    const collapsed = !!opts.collapsed;
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
        cursor: "pointer",
    });
    const left = makeEl("div", {
        display: "flex", alignItems: "center", gap: "6px", minWidth: "0", flex: "1",
    });
    const chevron = makeEl("span", {
        width: "12px", display: "inline-flex", alignItems: "center", justifyContent: "center",
        flexShrink: "0", color: C.textMuted,
    });
    const label = makeEl("span", {
        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
    }, title);
    const lockBtn = makeEl("span", {
        width: "20px", flexShrink: "0", display: "inline-flex",
        alignItems: "center", justifyContent: "center", cursor: "pointer",
    });
    left.append(chevron, label);
    header.append(left, lockBtn);
    const body = makeEl("div", {
        padding: "4px 8px",
    });
    wrap._locked = false;
    wrap._collapsed = collapsed;
    wrap._setLocked = (locked) => {
        wrap._locked = !!locked;
        lockBtn.innerHTML = wrap._locked ? SECTION_LOCK_SVG_CLOSED : SECTION_LOCK_SVG_OPEN;
        lockBtn.title = wrap._locked ? "Section locked" : "Section unlocked";
        body.style.opacity = wrap._locked ? "0.45" : "1";
        body.style.pointerEvents = wrap._locked ? "none" : "auto";
    };
    wrap._setCollapsed = (next, { sync = true } = {}) => {
        const target = !!next;
        const wasCollapsed = !!wrap._collapsed;
        wrap._collapsed = target;
        chevron.textContent = wrap._collapsed ? "▶" : "▼";
        body.style.display = wrap._collapsed ? "none" : "";
        header.style.borderBottom = wrap._collapsed ? "none" : "1px solid rgba(255,255,255,0.1)";
        if (node?._weCollapsedSections && key) node._weCollapsedSections[key] = wrap._collapsed;
        if (sync && node) {
            if (!target && wasCollapsed) {
                syncHidden(node);
                requestAnimationFrame(() => reflowNode(node));
            } else if (target && !wasCollapsed && node._weSections && key) {
                normalizeSectionLayout(node, key);
            } else {
                syncHidden(node);
                requestAnimationFrame(() => reflowNode(node));
            }
        }
    };
    wrap._setLocked(false);
    wrap._setCollapsed(collapsed, { sync: false });
    header.onclick = () => {
        wrap._setCollapsed(!wrap._collapsed);
    };
    lockBtn.onclick = (e) => {
        e.stopPropagation();
        const next = !wrap._locked;
        wrap._setLocked(next);
        if (wrap._onLockChanged) wrap._onLockChanged(next);
    };
    wrap._lockBtn = lockBtn;
    wrap._titleLabel = label;
    wrap._chevron = chevron;
    wrap._body = body;
    wrap.setLocked = wrap._setLocked;
    wrap.setCollapsed = wrap._setCollapsed;
    wrap.append(header, body);
    return wrap;
}

// --- Shared layout constants ---
const LABEL_W = "35%";
const INPUT_W = "55%";
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
    const sorted = Array.isArray(models)
        ? [...models].sort((a, b) =>
            normalizeModelPath(a).localeCompare(
                normalizeModelPath(b),
                undefined,
                { numeric: true, sensitivity: "base" }
            )
        )
        : [];
    const groups = new Map();
    const ungrouped = [];
    for (const m of sorted) {
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

function normalizeModelPath(modelPath) {
    return String(modelPath || "").replace(/\\/g, "/").toLowerCase();
}

function _leafName(v) {
    const p = normalizeModelPath(v);
    const idx = p.lastIndexOf("/");
    return idx >= 0 ? p.substring(idx + 1) : p;
}

function _leafStem(v) {
    const leaf = _leafName(v);
    const dot = leaf.lastIndexOf(".");
    return dot > 0 ? leaf.substring(0, dot) : leaf;
}

function _canonicalInputName(name) {
    const n = String(name || "").trim().toLowerCase();
    if (n === "positive_prompt") return "pos_prompt";
    if (n === "negative_prompt") return "neg_prompt";
    if (n === "workflow_data") return "recipe_data";
    return n;
}

function _findInput(node, ...names) {
    const wanted = new Set(names.map((n) => String(n || "").trim().toLowerCase()));
    return node?.inputs?.find((i) => wanted.has(String(i?.name || "").trim().toLowerCase())) || null;
}

const MODEL_SLOT_KEYS = ["model_a", "model_b", "model_c", "model_d"];

function _normalizeModelSlotKey(v) {
    const key = String(v || "model_a").trim().toLowerCase();
    return MODEL_SLOT_KEYS.includes(key) ? key : "model_a";
}

function _normalizeModelSlotKeyStrict(v) {
    const key = String(v || "").trim().toLowerCase();
    return MODEL_SLOT_KEYS.includes(key) ? key : null;
}

function _nextModelSlotKey(v) {
    const key = _normalizeModelSlotKey(v);
    const idx = MODEL_SLOT_KEYS.indexOf(key);
    if (idx < 0 || idx >= MODEL_SLOT_KEYS.length - 1) return null;
    return MODEL_SLOT_KEYS[idx + 1];
}

function _getCurrentlySelectedModelSlot(node) {
    const uiSlot = node?._weSendModelSlotRow?._inp?.value;
    if (uiSlot) return _normalizeModelSlotKeyStrict(uiSlot);
    return _normalizeModelSlotKeyStrict(node?._weActiveModelSlot || node?._weSendModelSlot || "");
}

function _hydrateExtractedFromSelectedSlotProfile(node) {
    if (!node?._weUseSlotProfiles) return;
    const activeSlot = _getCurrentlySelectedModelSlot(node);
    if (!activeSlot) return;
    node._weActiveModelSlot = activeSlot;
    const activeProfile = node?._weSlotProfiles?.[activeSlot];
    if (!activeProfile?.ov || typeof activeProfile.ov !== "object") return;

    const ov = activeProfile.ov;
    const sampler = {
        steps_a: ov.steps_a ?? 20,
        cfg: ov.cfg ?? 5.0,
        denoise: ov.denoise ?? 1.0,
        seed_a: ov.seed_a ?? 0,
        sampler_name: ov.sampler_name || "euler",
        scheduler: ov.scheduler || "simple",
    };

    node._weExtracted = {
        ...(node._weExtracted || {}),
        positive_prompt: ov.positive_prompt ?? "",
        negative_prompt: ov.negative_prompt ?? "",
        model_a: ov.model_a ?? "",
        model_family: ov._family || "",
        vae: { name: ov.vae ?? "", source: "manual" },
        clip: {
            names: Array.isArray(ov.clip_names) ? ov.clip_names : [],
            type: "",
            source: "manual",
        },
        loras_a: Array.isArray(ov.loras_a) ? ov.loras_a : (node._weExtracted?.loras_a || []),
        sampler,
    };
}

function _makeEmptySlotProfile() {
    return {
        ov: {
            positive_prompt: "",
            negative_prompt: "",
            model_a: "",
            model_b: "",
            vae: "",
            clip_names: [],
            _family: "",
            steps_a: 20,
            cfg: 5.0,
            denoise: 1.0,
            seed_a: 0,
            sampler_name: "euler",
            scheduler: "simple",
            loras_a: [],
            loras_b: [],
            _input_ghosts: {
                positive: false,
                negative: false,
                seed: false,
                loras: false,
            },
            _section_locks: {
                model: false,
                sampler: false,
                positive: false,
                negative: false,
                loras: false,
            },
        },
        ls: {},
    };
}

function _emptySlotProfiles() {
    return {
        model_a: _makeEmptySlotProfile(),
        model_b: _makeEmptySlotProfile(),
        model_c: _makeEmptySlotProfile(),
        model_d: _makeEmptySlotProfile(),
    };
}

function _normalizeSlotProfiles(raw) {
    const out = _emptySlotProfiles();
    if (!raw || typeof raw !== "object") return out;
    for (const slot of MODEL_SLOT_KEYS) {
        const row = raw[slot];
        if (!row || typeof row !== "object") continue;
        const ov = (row.ov && typeof row.ov === "object")
            ? { ...out[slot].ov, ...row.ov }
            : { ...out[slot].ov };
        const ls = (row.ls && typeof row.ls === "object") ? { ...row.ls } : {};
        out[slot] = { ov, ls };
    }
    return out;
}

function _mergeSlotProfiles(existingRaw, incomingRaw) {
    const merged = _normalizeSlotProfiles(existingRaw);
    if (!incomingRaw || typeof incomingRaw !== "object") return merged;

    for (const slot of MODEL_SLOT_KEYS) {
        const incomingRow = incomingRaw[slot];
        if (!incomingRow || typeof incomingRow !== "object") continue;

        const current = merged[slot] || _makeEmptySlotProfile();
        const nextOv = (incomingRow.ov && typeof incomingRow.ov === "object")
            ? { ...current.ov, ...incomingRow.ov }
            : { ...current.ov };
        const nextLs = (incomingRow.ls && typeof incomingRow.ls === "object")
            ? { ...incomingRow.ls }
            : { ...current.ls };

        merged[slot] = { ov: nextOv, ls: nextLs };
    }

    return merged;
}

function _getSelectedSlotInputGhosts(node) {
    if (!node?._weUseSlotProfiles) {
        return { positive: false, negative: false, seed: false };
    }
    const activeSlot = _getCurrentlySelectedModelSlot(node);
    if (!activeSlot) {
        return { positive: false, negative: false, seed: false };
    }
    const ov = node?._weSlotProfiles?.[activeSlot]?.ov;
    const raw = (ov && typeof ov._input_ghosts === "object") ? ov._input_ghosts : {};
    return {
        positive: !!raw.positive,
        negative: !!raw.negative,
        seed: !!raw.seed,
    };
}

function _isLoraAvailableForSort(lora, availabilityMap = null) {
    if (!lora) return true;
    if (lora.available === false || lora.found === false) return false;
    const name = String(lora.name || "");
    if (availabilityMap && name && availabilityMap[name] === false) return false;
    return true;
}

function _findEquivalentOptionValue(sel, value) {
    const v = String(value ?? "");
    if (!v) return "";
    const opts = [...(sel?.options || [])];
    if (!opts.length) return "";

    // 1) exact value
    const exact = opts.find(o => o.value === v);
    if (exact) return exact.value;

    // 2) normalized full path match
    const nv = normalizeModelPath(v);
    const byNorm = opts.find(o => normalizeModelPath(o.value) === nv);
    if (byNorm) return byNorm.value;

    // 3) filename (with extension) match ignoring folders
    const leaf = _leafName(v);
    const byLeaf = opts.find(o => _leafName(o.value) === leaf);
    if (byLeaf) return byLeaf.value;

    // 4) filename stem match ignoring folders + extension
    const stem = _leafStem(v);
    const byStem = opts.find(o => _leafStem(o.value) === stem);
    if (byStem) return byStem.value;

    return "";
}

function applyWanVideoModelSplit(models, familyKey, { preferHigh = false, preferLow = false } = {}) {
    let filtered = Array.isArray(models) ? [...models] : [];
    const isWanVideo = (familyKey === "wan_video_i2v" || familyKey === "wan_video_t2v");
    if (!isWanVideo) return filtered;

    // Strict family filtering: i2v only sees i2v, t2v only sees t2v.
    if (familyKey === "wan_video_i2v") {
        filtered = filtered.filter((m) => {
            const p = normalizeModelPath(m);
            return p.includes("i2v");
        });
    } else if (familyKey === "wan_video_t2v") {
        filtered = filtered.filter((m) => {
            const p = normalizeModelPath(m);
            return p.includes("t2v");
        });
    }

    if (preferHigh) {
        filtered = filtered.filter((m) => normalizeModelPath(m).includes("high"));
    } else if (preferLow) {
        filtered = filtered.filter((m) => normalizeModelPath(m).includes("low"));
    }

    return filtered;
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

function _ensureOptionValue(sel, value, grouped) {
    const v = String(value ?? "");
    if (!v) return;
    if (_findEquivalentOptionValue(sel, v)) return;
    const o = document.createElement("option");
    o.value = v;
    o.textContent = grouped ? cleanModelName(v) : v;
    o.style.color = C.text;
    sel.appendChild(o);
}

function makeSelectRow(label, initialValue, lazyFetch, onChange, grouped, includeEmptyOption = true, allowMissingInjection = true) {
    const row = makeEl("div", { ...ROW_STYLE });
    row.appendChild(makeEl("span", { color: C.textMuted, width: LABEL_W, flexShrink: "0" }, label));
    let _origVal = initialValue || "";
    let _recolor;
    const resetBtn = makeResetBtn(() => {
        if (![...sel.options].some(o => o.value === _origVal)) {
            if (allowMissingInjection && _origVal) {
                const o = document.createElement("option"); o.value = _origVal;
                o.textContent = grouped ? cleanModelName(_origVal) : _origVal;
                sel.insertBefore(o, sel.firstChild);
            } else {
                sel.value = includeEmptyOption ? "" : (sel.options[0]?.value || "");
                resetBtn.style.visibility = "hidden";
                _recolor();
                if (row._onReset) row._onReset(sel.value);
                if (onChange) onChange(sel.value);
                return;
            }
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
        if (includeEmptyOption) {
            const defOpt = document.createElement("option");
            defOpt.value = "";
            defOpt.textContent = grouped ? "(none)" : "(Default)";
            sel.appendChild(defOpt);
        }
        if (initialValue && initialValue !== "\u2014" && !initialValue.startsWith("(")) {
            const o = document.createElement("option"); o.value = initialValue;
            o.textContent = grouped ? cleanModelName(initialValue) : initialValue;
            sel.appendChild(o);
            sel.value = initialValue;
        } else {
            sel.value = includeEmptyOption ? "" : (sel.options[0]?.value || "");
        }
    }
    let _loaded = false;
    let _loadingPromise = null;
    let _origFound = true;
    _recolor = () => {
        const cur = sel.value;
        const opt = [...sel.options].find(o => o.value === cur);
        sel.style.color = (opt && opt.dataset.missing) ? C.error : C.text;
    };
    const _ensureLoaded = async () => {
        if (_loaded) return;
        if (_loadingPromise) return _loadingPromise;

        _loadingPromise = (async () => {
            const options = await lazyFetch();
            const currentVal = sel.value;
            sel.innerHTML = "";
            if (includeEmptyOption) {
                const defOpt = document.createElement("option");
                defOpt.value = ""; defOpt.textContent = grouped ? "(none)" : "(Default)";
                sel.appendChild(defOpt);
            }
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
            // Keep existing selection stable even if fetched options changed.
            const equivalentBeforeInject = _findEquivalentOptionValue(sel, currentVal);
            const canInjectMissing = allowMissingInjection && !grouped;
            const shouldMarkMissing = (
                canInjectMissing &&
                _origFound === false &&
                !!currentVal &&
                !String(currentVal).startsWith("(") &&
                !equivalentBeforeInject
            );
            if (canInjectMissing) {
                _ensureOptionValue(sel, currentVal, grouped);
                if (shouldMarkMissing) {
                    const injected = [...sel.options].find(o => o.value === String(currentVal));
                    if (injected) {
                        injected.style.color = C.error;
                        injected.dataset.missing = "1";
                    }
                }
            }
            const resolvedCurrent = _findEquivalentOptionValue(sel, currentVal);
            if (resolvedCurrent) {
                sel.value = resolvedCurrent;
            } else {
                sel.value = sel.options[0]?.value || "";
            }
            _recolor();
            _loaded = true;
        })();

        try {
            await _loadingPromise;
        } finally {
            _loadingPromise = null;
        }
    };

    sel.onfocus = _ensureLoaded;
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
        _origFound = found !== false;
        _loaded = false;
        _loadingPromise = null;
        sel.innerHTML = "";
        if (includeEmptyOption) {
            const defOpt = document.createElement("option");
            defOpt.value = "";
            defOpt.textContent = grouped ? "(none)" : "(Default)";
            sel.appendChild(defOpt);
        }
        if (v && v !== "\u2014" && !v.startsWith("(")) {
            const o = document.createElement("option"); o.value = v;
            o.textContent = grouped ? cleanModelName(v) : v;
            const equivalentNow = _findEquivalentOptionValue(sel, v);
            const missing = (found === false) && !equivalentNow;
            if (missing) { o.style.color = C.error; o.dataset.missing = "1"; }
            else { o.style.color = C.text; }
            sel.appendChild(o);
            sel.value = equivalentNow || v;
        } else {
            sel.value = includeEmptyOption ? "" : (sel.options[0]?.value || "");
        }
        const resolvedAfterSet = _findEquivalentOptionValue(sel, v);
        const missingAfterSet = (found === false && v && !v.startsWith("(") && !resolvedAfterSet);
        sel.style.color = missingAfterSet ? C.error : C.text;
        resetBtn.style.visibility = "hidden";
    };
    row._getValue = () => sel.value;
    row._allowMissingInjection = allowMissingInjection;
    row._resetLoaded = () => { _loaded = false; };
    row._markLoaded = () => {
        _loaded = true;
        _loadingPromise = null;
    };
    row._ensureLoaded = _ensureLoaded;
    return row;
}

async function reloadGroupedSelect(row, fetchFn, grouped, recommendedValue, includeEmptyOption = true, preservePrevious = true) {
    const options = await fetchFn();
    const sel = row._sel;
    const previousVal = sel.value;
    const allowMissingInjection = row?._allowMissingInjection !== false;
    if (grouped) {
        sel.innerHTML = "";
        if (includeEmptyOption) {
            const noneOpt = document.createElement("option");
            noneOpt.value = ""; noneOpt.textContent = "(none)";
            noneOpt.style.color = C.textMuted;
            sel.appendChild(noneOpt);
        }
        populateGroupedSelect(sel, options, true);
        if (preservePrevious) {
            const resolvedPrev = _findEquivalentOptionValue(sel, previousVal);
            if (resolvedPrev) {
                sel.value = resolvedPrev;
            } else {
                sel.value = includeEmptyOption ? "" : (sel.options[0]?.value || "");
            }
        } else {
            sel.value = includeEmptyOption ? "" : (sel.options[0]?.value || "");
        }
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
            if (preservePrevious) {
                const resolvedPrev = _findEquivalentOptionValue(sel, previousVal);
                if (resolvedPrev) {
                    sel.value = resolvedPrev;
                } else if (previousVal && allowMissingInjection) {
                    _ensureOptionValue(sel, previousVal, grouped);
                    sel.value = previousVal;
                } else {
                    sel.value = "";
                }
            } else {
                sel.value = "";
            }
        }
    }
    if (row._markLoaded) row._markLoaded();
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
            if (opt && typeof opt === "object") {
                const val = String(opt.value ?? opt.label ?? "");
                const text = String(opt.label ?? opt.value ?? "");
                o.value = val;
                o.textContent = text;
            } else {
                o.value = opt;
                o.textContent = opt;
            }
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
function makeLoraTag(lora, avail, onToggle, onStrength, onMoveUp, onMoveDown, onDelete) {
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

        const addMenuItem = (label, color, onClick) => {
            const item = makeEl("div", {
                padding: "8px 12px", cursor: "pointer", fontSize: "12px",
                color, whiteSpace: "nowrap",
            }, label);
            item.addEventListener("mouseenter", () => { item.style.backgroundColor = "#3a3a3a"; });
            item.addEventListener("mouseleave", () => { item.style.backgroundColor = "transparent"; });
            item.addEventListener("click", (ev) => {
                ev.stopPropagation();
                menu.remove();
                if (onClick) onClick();
            });
            menu.appendChild(item);
        };

        addMenuItem("Move Up", "#ddd", onMoveUp);
        addMenuItem("Move Down", "#ddd", onMoveDown);

        if (!avail) {
            addMenuItem("\uD83D\uDD0D Search on CivitAI", "#4da6ff", () => {
                window.open(`https://civitai.red/search/models?sortBy=models_v9&query=${encodeURIComponent(name)}&modelType=LORA`, "_blank");
            });
        }
        addMenuItem("Delete", "#ff6b6b", onDelete);
        document.body.appendChild(menu);
        const close = (ev) => { if (!menu.contains(ev.target)) { menu.remove(); document.removeEventListener("mousedown", close); } };
        document.addEventListener("mousedown", close);
    });

    return tag;
}

// --- LoRA stack card ---
function updateLoraContainerHeight(container, node = null) {
    const tagsContainer = container?._tagsContainer || container?.tagsContainer;
    if (!tagsContainer || !node) return;

    const children = Array.from(tagsContainer.children || []).filter(el => el.style?.display !== "none");
    const actualWidth = Math.max(220, Number(node.size?.[0] || 0) - 32);
    const tagWidth = 204;
    const tagsPerRow = Math.max(1, Math.floor(actualWidth / tagWidth));
    const tagCount = Math.max(1, children.length);
    const rows = Math.max(1, Math.ceil(tagCount / tagsPerRow));
    const rowHeight = 28;
    const topChrome = 8;
    const target = topChrome + rows * rowHeight;

    tagsContainer.style.minHeight = `${target}px`;
    tagsContainer.style.height = `${target}px`;
}

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
        minHeight: "30px", overflowY: "auto", scrollbarWidth: "none",
        border: "1px solid rgba(255,255,255,0.1)", borderRadius: "4px",
        padding: "4px",
    });
    // Stop wheel propagation so scrolling lora list doesn't zoom canvas
    tagsContainer.addEventListener("wheel", (e) => { e.stopPropagation(); });
    container.appendChild(tagsContainer);
    container._tagsContainer = tagsContainer;
    container._titleLabel = titleLabel;
    container.updateHeight = (node = null) => updateLoraContainerHeight(container, node);
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
// --- Per-family sampler defaults (applied on manual family switch) ---
// Edit values here to change what each family starts with.
const FAMILY_DEFAULTS = {
    ernie:         { steps_a: 4,  cfg: 1.0,  sampler: "euler_ancestral", scheduler: "beta" },
    sdxl:          { steps_a: 20, cfg: 5.0,  sampler: "dpmpp_2m_sde",    scheduler: "karras" },
    flux1:         { steps_a: 20, cfg: 1.0,  sampler: "euler",           scheduler: "simple" },
    flux2:         { steps_a: 4,  cfg: 1.0,  sampler: "euler",           scheduler: "simple" },
    zimage:        { steps_a: 9,  cfg: 1.0,  sampler: "euler",           scheduler: "simple" },
    // ltxv:       { steps_a: 8,  cfg: 1.0,  sampler: "euler",           scheduler: "simple" },
    wan_image:     { steps_a: 10, cfg: 1.0,  sampler: "lcm",             scheduler: "simple" },
    wan_video_t2v: { steps_a: 3,  cfg: 1.0,  sampler: "lcm",             scheduler: "simple",
                     steps_b: 3 },
    wan_video_i2v: { steps_a: 3,  cfg: 1.0,  sampler: "lcm",             scheduler: "simple",
                     steps_b: 3 },
    qwen_image:    { steps_a: 10, cfg: 1.0,  sampler: "euler",           scheduler: "simple" },
};

function _isFiniteScalarNumber(value) {
    if (typeof value === "number") return Number.isFinite(value);
    if (typeof value === "string") {
        const trimmed = value.trim();
        if (!trimmed) return false;
        const n = Number(trimmed);
        return Number.isFinite(n);
    }
    return false;
}

function _coerceNumericField(value, fallback, { integer = false, min = null } = {}) {
    let out = _isFiniteScalarNumber(value) ? Number(value) : Number(fallback);
    if (!Number.isFinite(out)) out = Number(fallback);
    if (integer) out = Math.trunc(out);
    if (min != null && out < min) out = min;
    return out;
}

function _coerceChoiceField(value, fallback, allowed = null) {
    const raw = (typeof value === "string") ? value.trim() : "";
    let out = raw || String(fallback || "");
    if (Array.isArray(allowed) && allowed.length > 0 && !allowed.includes(out)) {
        out = String(fallback || allowed[0] || "");
    }
    return out;
}

function normalizeSelectableFamily(family, fallback = "sdxl") {
    const f = String(family || "").trim();
    if (!f) return fallback;
    return f;
}

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

    if (node._wePullModelSlotRow?._inp) {
        const modelSlotRaw = d.model_slot || node._weActiveModelSlot || node._wePullModelSlot || "model_a";
        let pullSlot = _normalizeModelSlotKey(modelSlotRaw);
        if (node._wePullModelSlotRow._setOriginal) node._wePullModelSlotRow._setOriginal(pullSlot);
        node._wePullModelSlotRow._inp.value = pullSlot;
        node._wePullModelSlot = pullSlot;
    }
    if (node._weSendModelSlotRow?._inp) {
        const modelSlotRaw = d.model_slot || node._weActiveModelSlot || node._weSendModelSlot || node._wePullModelSlot || "model_a";
        const sendSlot = _normalizeModelSlotKey(modelSlotRaw);
        if (node._weSendModelSlotRow._setOriginal) node._weSendModelSlotRow._setOriginal(sendSlot);
        node._weSendModelSlotRow._inp.value = sendSlot;
        node._weSendModelSlot = sendSlot;
        if (node._weUseSlotProfiles) node._weActiveModelSlot = sendSlot;
    }

    const sectionLocks = node._weSectionLocks || {};
    const modelLocked = !!sectionLocks.model;
    const samplerLocked = !!sectionLocks.sampler;
    const resolutionLocked = !!sectionLocks.resolution;
    const positiveLocked = !!sectionLocks.positive;
    const negativeLocked = !!sectionLocks.negative;
    const lorasLocked = !!sectionLocks.loras;

    console.log("[WB UpdateTrace] updateUI start", {
        incomingResolution: d?.resolution || null,
        resolutionLocked,
        sectionLocks: { ...sectionLocks },
        weResLocked: !!node._weResLocked,
    });

    // Family — must be set FIRST and dropdowns reloaded before setting
    // model/VAE/CLIP values, otherwise the wrong family's options are shown.
    const newFamily = normalizeSelectableFamily(d.model_family || d.family || null, "sdxl");
    console.log("[updateUI] incoming family:", newFamily, "current:", node._weFamily);
    const familyChanged = !modelLocked && newFamily && newFamily !== node._weFamily;
    if (!modelLocked && node._weFamilySel) {
        const sel = node._weFamilySel;
        // Pre-load the full families list if not yet done — this normally
        // happens lazily on focus, but we need it populated before we can
        // set sel.value to a non-SDXL family key.
        if (!sel._familiesLoaded) {
            try {
                const r = await fetch("/workflow-extractor/list-families");
                const fd = await r.json();
                const families = Object.fromEntries(Object.entries(fd.families || {}));
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
        const resolvedFamily = [...sel.options].some(o => o.value === newFamily) ? newFamily : "sdxl";
        sel.value = resolvedFamily;
        node._weFamily = resolvedFamily;
        console.log("[updateUI] familySel set to:", sel.value, "options:", [...sel.options].map(o => o.value));
    }

    // If family changed, reload model/VAE/CLIP dropdown lists for the new
    // family before setting values — otherwise _setOriginal can't find the
    // option in the list and the selection falls back to (Default).
    if (familyChanged && node._onFamilyChanged) {
        await node._onFamilyChanged(newFamily, { fromUpdateUI: true });
    }

    if (!modelLocked) {
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

        // Prewarm options so the first dropdown open after Update Recipe
        // already has the selected item highlighted.
        await Promise.allSettled([
            node._weModelRow?._ensureLoaded?.(),
            node._weModelBRow?._ensureLoaded?.(),
            node._weVaeRow?._ensureLoaded?.(),
            node._weClipRow?._ensureLoaded?.(),
        ]);
    }

    // Keep pulled-empty slots truly empty on Update Recipe; otherwise preserve
    // existing safety behavior that ensures a visible model selection.
    const pulledModelAEmpty = String(d.model_a || "").trim() === "";
    if (node._weEnsureModelSelection && !pulledModelAEmpty) {
        await node._weEnsureModelSelection({ reloadIfEmpty: true, sync: false });
    }

    // Sampler
    const s = d.sampler || {};
    if (!samplerLocked && node._weSamplerRows) {
        const rows = node._weSamplerRows;
        const seedAInputConn = _findInput(node, "seed", "seed_a");
        const seedBInputConn = node.inputs?.find(i => i.name === "seed_b");
        const seedALinked = seedAInputConn?.link != null;
        const seedBLinked = seedBInputConn?.link != null;
        const familyDefaults = FAMILY_DEFAULTS[newFamily] || FAMILY_DEFAULTS.sdxl || {};
        const defaultStepsA = _coerceNumericField(familyDefaults.steps_a, 20, { integer: true, min: 1 });
        const defaultCfg = _coerceNumericField(familyDefaults.cfg, 5.0, { min: 0 });
        const defaultSampler = _coerceChoiceField(familyDefaults.sampler, "euler", SAMPLERS);
        const defaultScheduler = _coerceChoiceField(familyDefaults.scheduler, "simple", SCHEDULERS);
        const defaultStepsB = _coerceNumericField(
            familyDefaults.steps_b,
            (newFamily === "wan_video_i2v" || newFamily === "wan_video_t2v") ? 3 : defaultStepsA,
            { integer: true, min: 1 }
        );

        const stepsA = _coerceNumericField(s.steps_a ?? s.steps, defaultStepsA, { integer: true, min: 1 });
        const cfg = _coerceNumericField(s.cfg, defaultCfg, { min: 0 });
        const samplerName = _coerceChoiceField(s.sampler_name, defaultSampler, SAMPLERS);
        const schedulerName = _coerceChoiceField(s.scheduler, defaultScheduler, SCHEDULERS);
        const denoise = _coerceNumericField(s.denoise, 1.0, { min: 0 });
        const seedA = _coerceNumericField(s.seed_a ?? s.seed, 0, { integer: true, min: 0 });
        const seedB = _coerceNumericField(s.seed_b ?? s.seed_a ?? s.seed, seedA, { integer: true, min: 0 });
        const stepsB = _coerceNumericField(s.steps_b ?? s.steps_a ?? s.steps, defaultStepsB, { integer: true, min: 1 });

        if (rows.steps_a?._setOriginal) rows.steps_a._setOriginal(stepsA);
        if (rows.cfg?._setOriginal) rows.cfg._setOriginal(cfg);
        if (rows.sampler?._setOriginal) rows.sampler._setOriginal(samplerName);
        if (rows.scheduler?._setOriginal) rows.scheduler._setOriginal(schedulerName);
        if (rows.denoise?._setOriginal) rows.denoise._setOriginal(denoise);
        if (!seedALinked && rows.seed_a?._setOriginal) rows.seed_a._setOriginal(seedA);
        if (!seedBLinked && rows.seed_b?._setOriginal) rows.seed_b._setOriginal(seedB);
        // WAN Video dual steps
        if (rows.steps_b?._setOriginal) rows.steps_b._setOriginal(stepsB);
    }

    // Resolution — skip width/height/batch if locked
    const r = d.resolution || {};
    if (node._weResRows) {
        const rr = node._weResRows;
        if (!resolutionLocked) {
            const inW = _coerceNumericField(r.width, 768, { integer: true, min: 1 });
            const inH = _coerceNumericField(r.height, 1280, { integer: true, min: 1 });
            const inBatch = _coerceNumericField(r.batch_size, 1, { integer: true, min: 1 });

            console.log("[WB UpdateTrace] updateUI applying resolution", {
                inW,
                inH,
                batch: r.batch_size ?? 1,
                length: r.length ?? null,
            });

            // Match orientation to incoming resolution.
            if (node._weSetLandscape) {
                node._weSetLandscape(inW > inH);
            }

            if (rr.width?._setOriginal) rr.width._setOriginal(inW);
            if (rr.height?._setOriginal) rr.height._setOriginal(inH);
            if (rr.batch?._setOriginal) rr.batch._setOriginal(inBatch);

            // If incoming resolution exactly matches a known ratio preset,
            // select that ratio; otherwise keep None.
            if (node._weRatioSel && node._weSetRatioIdx) {
                const ratios = Array.isArray(node._weRatioDefs) ? node._weRatioDefs : [];
                const landscape = inW > inH;
                let matchedIdx = 0;

                for (let i = 1; i < ratios.length; i++) {
                    const ratio = ratios[i];
                    if (!ratio || !ratio.w || !ratio.h) continue;
                    const isMatch = landscape
                        ? (inW * ratio.w === inH * ratio.h)
                        : (inW * ratio.h === inH * ratio.w);
                    if (isMatch) {
                        matchedIdx = i;
                        break;
                    }
                }

                node._weSetRatioIdx(matchedIdx);
                node._weRatioSel.value = String(matchedIdx);
                if (matchedIdx === 0) {
                    node._weRatio = "None";
                } else {
                    const ratio = ratios[matchedIdx];
                    node._weRatio = landscape
                        ? `${ratio.h}:${ratio.w}`
                        : `${ratio.w}:${ratio.h}`;
                }

                console.log("[WB UpdateTrace] updateUI ratio/orientation", {
                    matchedIdx,
                    ratioLabel: node._weRatio,
                    landscape,
                });
            }
        } else {
            console.warn("[WB UpdateTrace] resolution apply skipped because section is locked", {
                sectionLocks: { ...sectionLocks },
                weResLocked: !!node._weResLocked,
                incomingResolution: r,
            });
        }
        if (rr.frames) {
            if (r.length != null) {
                const frames = _coerceNumericField(r.length, 81, { integer: true, min: 1 });
                rr.frames.style.display = "flex";
                if (rr.frames._setOriginal) rr.frames._setOriginal(frames);
            }
        }

        console.log("[WB UpdateTrace] updateUI final resolution UI", {
            width: rr.width?._inp?.value,
            height: rr.height?._inp?.value,
            batch: rr.batch?._inp?.value,
            frames: rr.frames?._inp?.value,
            ratio: node._weRatio,
        });
    }

    // Prompts
    if (!positiveLocked && node._wePosBox) node._wePosBox.value = d.positive_prompt || "";
    if (!negativeLocked && node._weNegBox) node._weNegBox.value = d.negative_prompt || "";

    // LoRAs
    if (!lorasLocked) updateLoras(node);

    // Update WAN-specific visibility
    updateWanVisibility(node);
    if (node._updatePromptGhosting) node._updatePromptGhosting();
    if (node._updateSeedGhosting) node._updateSeedGhosting();

    syncHidden(node);

    // Async: check LoRA availability and update display
    if (!lorasLocked) checkLoraAvailability(node);
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
        console.warn("[RecipeBuilder] Error checking LoRA availability:", e);
    }
}

// --- Family-specific visibility ---
function updateWanVisibility(node) {
    const family = node._weFamily;
    const isWanVideo = (family === "wan_video_i2v" || family === "wan_video_t2v");

    // Frames row: only for WAN Video (not WAN Image)
    if (node._weResRows?.frames) {
        node._weResRows.frames.style.display = isWanVideo ? "flex" : "none";
    }
}

/** Merge two LoRA lists, dedup by name (second list wins on conflict). */
function _sortLorasByName(loras) {
    const src = Array.isArray(loras) ? [...loras] : [];
    return src.sort((a, b) => String(a?.name || "").localeCompare(String(b?.name || "")));
}

function _sortLorasMissingLast(loras, availabilityMap = null) {
    const src = Array.isArray(loras) ? [...loras] : [];
    return src.sort((a, b) => {
        const aAvailable = _isLoraAvailableForSort(a, availabilityMap);
        const bAvailable = _isLoraAvailableForSort(b, availabilityMap);
        if (aAvailable !== bAvailable) return aAvailable ? -1 : 1;
        return String(a?.name || "").localeCompare(String(b?.name || ""));
    });
}

function _applyPreferredLoraOrder(loras, preferredOrder) {
    const src = Array.isArray(loras) ? loras : [];
    const pref = Array.isArray(preferredOrder) ? preferredOrder : [];
    if (!pref.length || !src.length) return src;

    const preferredNames = pref
        .map((item) => String(item?.name || item || "").trim())
        .filter(Boolean);
    if (!preferredNames.length) return src;

    const out = [];
    const used = new Set();

    for (const pname of preferredNames) {
        const idx = src.findIndex((item, i) => {
            if (used.has(i)) return false;
            return String(item?.name || "").trim() === pname;
        });
        if (idx >= 0) {
            out.push(src[idx]);
            used.add(idx);
        }
    }

    for (let i = 0; i < src.length; i++) {
        if (!used.has(i)) out.push(src[i]);
    }

    return out;
}

/** Merge two LoRA lists, dedup by name (second list wins on conflict), preserving preferred UI order when provided. */
function _mergeLoraLists(listA, listB, preferredOrder = null) {
    const byName = new Map();
    const orderedNames = [];
    const apply = (list) => {
        for (const l of (Array.isArray(list) ? list : [])) {
            const name = String(l?.name || "").trim();
            if (!name) continue;
            if (!byName.has(name)) orderedNames.push(name);
            byName.set(name, l);
        }
    };
    apply(listA);
    apply(listB);
    const merged = orderedNames.map((name) => byName.get(name)).filter(Boolean);
    return _applyPreferredLoraOrder(merged, preferredOrder);
}

function _loraListSignature(list) {
    const src = Array.isArray(list) ? list : [];
    return src
        .map((item) => {
            const l = item || {};
            const name = String(l.name || "").trim();
            const path = String(l.path || "").trim();
            const ms = Number(l.model_strength ?? l.strength ?? 1);
            const cs = Number(l.clip_strength ?? l.model_strength ?? l.strength ?? 1);
            const active = (l.active !== false) ? "1" : "0";
            return `${name}|${path}|${ms}|${cs}|${active}`;
        })
        .join(",");
}

function _loraStacksSignature(workflowLoras, inputLoras) {
    const wl = workflowLoras || { a: [], b: [] };
    const il = inputLoras || { a: [], b: [] };
    return [
        _loraListSignature(wl.a),
        _loraListSignature(wl.b),
        _loraListSignature(il.a),
        _loraListSignature(il.b),
    ].join("||");
}

function _loraEntrySignature(lora) {
    const l = lora || {};
    const name = String(l.name || "").trim();
    const path = String(l.path || "").trim();
    const ms = Number(l.model_strength ?? l.strength ?? 1);
    const cs = Number(l.clip_strength ?? l.model_strength ?? l.strength ?? 1);
    const active = (l.active !== false) ? "1" : "0";
    return `${name}|${path}|${ms}|${cs}|${active}`;
}

function _buildCanonicalLoraMap(lorasA, lorasB) {
    const aList = Array.isArray(lorasA) ? lorasA : [];
    const bList = Array.isArray(lorasB) ? lorasB : [];
    const hasBoth = aList.length > 0 && bList.length > 0;
    const out = {};

    for (const lora of aList) {
        const name = String(lora?.name || "").trim();
        if (!name) continue;
        const canonicalKey = `a:${name}`;
        out[canonicalKey] = {
            signature: _loraEntrySignature(lora),
            stateKey: hasBoth ? canonicalKey : name,
            stack: "a",
        };
    }
    for (const lora of bList) {
        const name = String(lora?.name || "").trim();
        if (!name) continue;
        const canonicalKey = `b:${name}`;
        out[canonicalKey] = {
            signature: _loraEntrySignature(lora),
            stateKey: canonicalKey,
            stack: "b",
        };
    }

    return out;
}

function _resetChangedLoraState(node, oldLorasA, oldLorasB, newLorasA, newLorasB) {
    if (!node?._weLoraState) return;

    const oldMap = _buildCanonicalLoraMap(oldLorasA, oldLorasB);
    const newMap = _buildCanonicalLoraMap(newLorasA, newLorasB);
    const keys = new Set([...Object.keys(oldMap), ...Object.keys(newMap)]);
    const stateKeysToClear = new Set();

    for (const key of keys) {
        const oldEntry = oldMap[key];
        const newEntry = newMap[key];
        const oldSig = oldEntry?.signature || "";
        const newSig = newEntry?.signature || "";
        if (oldSig === newSig) continue;

        if (oldEntry?.stateKey) stateKeysToClear.add(oldEntry.stateKey);
        if (newEntry?.stateKey) stateKeysToClear.add(newEntry.stateKey);
        if (key.startsWith("a:")) stateKeysToClear.add(key.substring(2));
    }

    for (const stateKey of stateKeysToClear) {
        if (stateKey in node._weLoraState) delete node._weLoraState[stateKey];
    }
}

function _captureLoraOriginalStrengths(node, lorasA, lorasB) {
    if (!node) return;
    const aList = Array.isArray(lorasA) ? lorasA : [];
    const bList = Array.isArray(lorasB) ? lorasB : [];
    const hasBoth = aList.length > 0 && bList.length > 0;
    const baseline = {};

    for (const lora of aList) {
        const name = String(lora?.name || "").trim();
        if (!name) continue;
        const modelStrength = Number(lora?.model_strength ?? lora?.strength ?? 1);
        const clipStrength = Number(lora?.clip_strength ?? lora?.model_strength ?? lora?.strength ?? modelStrength);
        const st = {
            model_strength: Number.isFinite(modelStrength) ? modelStrength : 1,
            clip_strength: Number.isFinite(clipStrength) ? clipStrength : 1,
        };
        const key = hasBoth ? `a:${name}` : name;
        baseline[key] = st;
        if (!(name in baseline)) baseline[name] = st;
    }

    for (const lora of bList) {
        const name = String(lora?.name || "").trim();
        if (!name) continue;
        const modelStrength = Number(lora?.model_strength ?? lora?.strength ?? 1);
        const clipStrength = Number(lora?.clip_strength ?? lora?.model_strength ?? lora?.strength ?? modelStrength);
        const st = {
            model_strength: Number.isFinite(modelStrength) ? modelStrength : 1,
            clip_strength: Number.isFinite(clipStrength) ? clipStrength : 1,
        };
        baseline[`b:${name}`] = st;
        if (!(name in baseline)) baseline[name] = st;
    }

    node._weLoraOriginalStrengths = baseline;
}

function _getLoraOriginalStrength(node, stateKey, lora) {
    const baseline = node?._weLoraOriginalStrengths || {};
    const byKey = baseline[stateKey];
    if (byKey && typeof byKey === "object") return byKey;

    const name = String(lora?.name || "").trim();
    const key = String(stateKey || "").trim();
    if (name) {
        if (key === name) {
            const byAKey = baseline[`a:${name}`];
            if (byAKey && typeof byAKey === "object") return byAKey;
        } else if (key === `a:${name}`) {
            const byNameKey = baseline[name];
            if (byNameKey && typeof byNameKey === "object") return byNameKey;
        }
    }
    const byName = baseline[name];
    if (byName && typeof byName === "object") return byName;

    const modelStrength = Number(lora?.model_strength ?? lora?.strength ?? 1);
    const clipStrength = Number(lora?.clip_strength ?? lora?.model_strength ?? lora?.strength ?? modelStrength);
    return {
        model_strength: Number.isFinite(modelStrength) ? modelStrength : 1,
        clip_strength: Number.isFinite(clipStrength) ? clipStrength : 1,
    };
}

// --- LoRA display ---
    requestAnimationFrame(() => reflowNode(node));

function updateLoras(node) {
    const containerA = node._weLoraAContainer;
    const containerB = node._weLoraBContainer;
    if (!containerA) return;
    containerA.innerHTML = "";
    if (containerB) containerB.innerHTML = "";

    const d = node._weExtracted;
    if (!d) return;

    const lorasA = Array.isArray(d.loras_a) ? d.loras_a : [];
    const lorasB = Array.isArray(d.loras_b) ? d.loras_b : [];
    const hasBoth = lorasA.length > 0 && lorasB.length > 0;

    const moveInStack = (stackKey, index, delta) => {
        const effectiveStack = stackKey === "b" ? "b" : "a";
        const src = effectiveStack === "b"
            ? (Array.isArray(d.loras_b) ? d.loras_b : [])
            : (Array.isArray(d.loras_a) ? d.loras_a : []);
        const from = Number(index);
        const to = from + Number(delta || 0);
        if (!Number.isInteger(from) || !Number.isInteger(to) || from < 0 || to < 0 || from >= src.length || to >= src.length) {
            return;
        }
        const moved = src[from];
        src.splice(from, 1);
        src.splice(to, 0, moved);
        updateLoras(node);
        syncHidden(node);
    };

    const deleteFromStack = (stackKey, index, name) => {
        const effectiveStack = stackKey === "b" ? "b" : "a";
        const src = effectiveStack === "b"
            ? (Array.isArray(d.loras_b) ? d.loras_b : [])
            : (Array.isArray(d.loras_a) ? d.loras_a : []);
        const idx = Number(index);
        if (!Number.isInteger(idx) || idx < 0 || idx >= src.length) return;

        src.splice(idx, 1);

        const nm = String(name || "").trim();
        if (nm && node._weLoraState) {
            const keys = [nm, `a:${nm}`, `b:${nm}`];
            for (const key of keys) {
                if (key in node._weLoraState) delete node._weLoraState[key];
            }
        }

        updateLoras(node);
        syncHidden(node);
    };

    const noLorasMsg = () => makeEl("div", {
        color: "rgba(200, 200, 200, 0.5)", fontStyle: "italic",
        fontSize: "11px", padding: "8px", width: "100%", textAlign: "center",
    }, "No LoRAs");

    if (!lorasA.length) containerA.appendChild(noLorasMsg());
    if (!lorasB.length && containerB) containerB.appendChild(noLorasMsg());
    if (!lorasA.length && !lorasB.length) return;

    const populateStack = (container, loras, stackKey) => {
        for (let index = 0; index < loras.length; index++) {
            const lora = loras[index];
            const name = lora.name || "";
            const stateKey = stackKey ? `${stackKey}:${name}` : name;
            const avail = d.lora_availability?.[name] !== false;
            if (node._weLoraState[stateKey] === undefined) {
                node._weLoraState[stateKey] = {
                    active: lora.active !== false,
                    model_strength: lora.model_strength ?? 1.0,
                    clip_strength: lora.clip_strength ?? 1.0,
                };
            }
            const st = node._weLoraState[stateKey];
            lora._active = st.active;
            lora.model_strength = st.model_strength;
            const stackForMove = stackKey || "a";
            const tag = makeLoraTag(lora, avail,
                () => { st.active = !st.active; updateLoras(node); syncHidden(node); },
                (v) => { st.model_strength = v; st.clip_strength = v; syncHidden(node); },
                () => moveInStack(stackForMove, index, -1),
                () => moveInStack(stackForMove, index, 1),
                () => deleteFromStack(stackForMove, index, name),
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
    if (node._updateSeedGhosting) node._updateSeedGhosting();
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

function _showInfo(node, infoMsg) {
    const root = node._weRoot;
    if (!root) return;

    const existing = root.querySelector(".we-info-banner");
    if (existing) existing.remove();

    if (!infoMsg) return;

    const banner = makeEl("div", {
        padding: "8px 12px",
        backgroundColor: "rgba(66, 153, 225, 0.16)",
        border: "1px solid rgba(66, 153, 225, 0.65)",
        borderRadius: "6px",
        color: "#b9dcff",
        fontSize: "12px",
        lineHeight: "1.4",
        marginBottom: "4px",
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
    });
    banner.className = "we-info-banner";

    const text = makeEl("span", {}, infoMsg);
    banner.append(text);
    root.insertBefore(banner, root.firstChild);
}
    requestAnimationFrame(() => reflowNode(node));

// --- Sync hidden widgets ---
function syncHidden(node) {
    const wSet = (name, val) => {
        const w = node.widgets?.find(x => x.name === name);
        if (w) w.value = val;
    };
    const _resolvePersistedLoraStrengths = (st, lora) => {
        const baseModelRaw = Number(lora?.model_strength ?? lora?.strength ?? 1.0);
        const baseModel = Number.isFinite(baseModelRaw) ? baseModelRaw : 1.0;
        const stateModelRaw = Number(st?.model_strength);
        const hasStateModel = Number.isFinite(stateModelRaw);
        const modelStrength = hasStateModel ? stateModelRaw : baseModel;
        const clipStrength = modelStrength;

        return {
            model_strength: Number.isFinite(modelStrength) ? modelStrength : 1.0,
            clip_strength: Number.isFinite(clipStrength) ? clipStrength : 1.0,
        };
    };
    const _normalizeLoraStateForPersistence = () => {
        const extracted = node._weExtracted || {};
        const lorasA = Array.isArray(extracted.loras_a) ? extracted.loras_a : [];
        const lorasB = Array.isArray(extracted.loras_b) ? extracted.loras_b : [];
        const hasBothStacks = lorasA.length > 0 && lorasB.length > 0;
        const srcState = node._weLoraState || {};
        const normalized = {};

        for (const lora of lorasA) {
            const name = String(lora?.name || "").trim();
            if (!name) continue;
            const stateKey = hasBothStacks ? `a:${name}` : name;
            const aliasKey = hasBothStacks ? name : `a:${name}`;
            const st = srcState[stateKey] || srcState[aliasKey] || {};
            const strengths = _resolvePersistedLoraStrengths(st, lora);
            const val = {
                active: st.active !== undefined ? (st.active !== false) : (lora._active !== false && lora.active !== false),
                model_strength: strengths.model_strength,
                clip_strength: strengths.clip_strength,
            };
            normalized[stateKey] = val;
            // Keep both aliases in sync when A is single-stack so Python
            // pref-key resolution cannot pick a stale value.
            if (!hasBothStacks) normalized[`a:${name}`] = val;
        }

        for (const lora of lorasB) {
            const name = String(lora?.name || "").trim();
            if (!name) continue;
            const stateKey = `b:${name}`;
            const st = srcState[stateKey] || {};
            const strengths = _resolvePersistedLoraStrengths(st, lora);
            normalized[stateKey] = {
                active: st.active !== undefined ? (st.active !== false) : (lora._active !== false && lora.active !== false),
                model_strength: strengths.model_strength,
                clip_strength: strengths.clip_strength,
            };
        }

        return normalized;
    };
    const serializeLoraList = (list, stackKey = "", stateMap = null) => {
        const src = Array.isArray(list) ? list : [];
        const state = stateMap || node._weLoraState || {};
        return src
            .map((lora) => {
                const name = String(lora?.name || "").trim();
                if (!name) return null;
                const prefKey = stackKey ? `${stackKey}:${name}` : name;
                const st = state[prefKey] || state[name] || {};
                const strengths = _resolvePersistedLoraStrengths(st, lora);
                return {
                    name,
                    path: lora.path || name,
                    model_strength: strengths.model_strength,
                    clip_strength: strengths.clip_strength,
                    active: st.active !== undefined ? (st.active !== false) : (lora._active !== false && lora.active !== false),
                    available: node._weExtracted?.lora_availability?.[name] !== false,
                    source_input: lora?.source_input === true,
                };
            })
            .filter(Boolean);
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
        if (r.denoise?._inp && r.denoise.style.display !== "none") ov.denoise = parseFloat(r.denoise._inp.value) || 1.0;
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
    const sectionLocks = node._weSectionLocks || {};
    const persistedLocks = {};
    const sectionCollapsed = node._weCollapsedSections || {};
    for (const [key, locked] of Object.entries(sectionLocks)) {
        if (node._weUseSlotProfiles) {
            // Multi-slot mode: persist explicit lock booleans per slot profile
            // except resolution, which is shared across all slots.
            if (key !== "resolution") persistedLocks[key] = !!locked;
        } else if (locked) {
            persistedLocks[key] = true;
        }
    }
    if (Object.keys(persistedLocks).length > 0) ov._section_locks = persistedLocks;
    else delete ov._section_locks;

    const persistedCollapsed = {};
    for (const [key, collapsed] of Object.entries(sectionCollapsed)) {
        if (collapsed) persistedCollapsed[key] = true;
    }
    if (Object.keys(persistedCollapsed).length > 0) ov._section_collapsed = persistedCollapsed;
    else delete ov._section_collapsed;

    // Persist resolution UI state
    if (node._weRatio) ov._ratio = node._weRatio;
    if (node._weRatioLandscape) ov._ratio_landscape = true;
    if (sectionLocks.resolution || node._weResLocked) ov._res_locked = true;
    else delete ov._res_locked;
    // Persist prompt boxes as authoritative UI state. Empty string means the
    // user intentionally cleared the prompt and must not fall back to stale
    // remembered workflow prompt text.
    if (node._wePosBox) {
        ov.positive_prompt = String(node._wePosBox.value || "");
    }
    if (node._weNegBox) {
        ov.negative_prompt = String(node._weNegBox.value || "");
    }
    if (node._weFamily) ov._family = node._weFamily;
    if (node._weUseSlotProfiles) {
        const activeSlot = _normalizeModelSlotKey(node._weActiveModelSlot || node._weSendModelSlot || "model_a");
        const activeProfileOv = node?._weSlotProfiles?.[activeSlot]?.ov;
        node._weActiveModelSlot = activeSlot;
        ov._active_model_slot = activeSlot;
        ov._model_slot = activeSlot;
        if (activeProfileOv && typeof activeProfileOv._input_ghosts === "object") {
            ov._input_ghosts = { ...activeProfileOv._input_ghosts };
        } else {
            delete ov._input_ghosts;
        }
    } else {
        delete ov._active_model_slot;
        if (node._weSendModelSlot) {
            ov._model_slot = _normalizeModelSlotKey(node._weSendModelSlot);
        } else {
            delete ov._model_slot;
        }
    }
    // Remove legacy selector fields.
    delete ov._send_model_slot;
    delete ov._pull_model_slot;
    if (node._weShowAllModels) ov._show_all_models = true;
    else delete ov._show_all_models;

    const persistedLoraState = _normalizeLoraStateForPersistence();

    // Persist full LoRA stacks as part of authoritative UI state so tab
    // switch rehydrate restores the same lists (not just toggle/strength state).
    if (node._weExtracted) {
        const lorasA = node._weExtracted.loras_a || [];
        const lorasB = node._weExtracted.loras_b || [];
        const hasBothStacks = lorasA.length > 0 && lorasB.length > 0;
        ov.loras_a = serializeLoraList(lorasA, hasBothStacks ? "a" : "", persistedLoraState);
        ov.loras_b = serializeLoraList(lorasB, "b", persistedLoraState);
        if (node._weExtracted.lora_availability && typeof node._weExtracted.lora_availability === "object") {
            ov._lora_availability = { ...node._weExtracted.lora_availability };
        }
    }

    if (node._weUseSlotProfiles) {
        const activeSlot = _normalizeModelSlotKey(node._weActiveModelSlot || "model_a");
        const slotProfiles = _normalizeSlotProfiles(node._weSlotProfiles);

        const slotOv = { ...ov };
        delete slotOv._slot_profiles;
        delete slotOv._active_model_slot;

        // Resolution settings are global for multi-slot builder.
        delete slotOv.width;
        delete slotOv.height;
        delete slotOv.batch_size;
        delete slotOv.length;
        delete slotOv._ratio;
        delete slotOv._ratio_landscape;
        delete slotOv._res_locked;
        if (slotOv._section_locks && typeof slotOv._section_locks === "object") {
            const nextLocks = { ...slotOv._section_locks };
            delete nextLocks.resolution;
            slotOv._section_locks = nextLocks;
        }
        if (slotOv._section_collapsed && typeof slotOv._section_collapsed === "object") {
            const nextCollapsed = { ...slotOv._section_collapsed };
            delete nextCollapsed.resolution;
            slotOv._section_collapsed = nextCollapsed;
        }

        slotProfiles[activeSlot] = {
            ov: slotOv,
            ls: { ...persistedLoraState },
        };

        node._weSlotProfiles = slotProfiles;
        ov._slot_profiles = slotProfiles;
        ov._active_model_slot = activeSlot;
        ov._model_slot = activeSlot;
    } else {
        delete ov._slot_profiles;
    }

    const ovJson = JSON.stringify(ov);
    const lsJson = JSON.stringify(persistedLoraState);
    wSet("override_data", ovJson);
    wSet("lora_state", lsJson);

    // Persist to node.properties for tab-switch survival
    node.properties = node.properties || {};
    // Canonical saved UI state for workflow reload restore.
    node.properties.we_ui_state = ovJson;
    node.properties.we_lora_state = lsJson;
    if (node._weWorkflowLoras) node.properties.we_workflow_loras = JSON.stringify(node._weWorkflowLoras);
    if (node._weInputLoras) node.properties.we_input_loras = JSON.stringify(node._weInputLoras);
    if (node._weWorkflowPrompts) node.properties.we_workflow_prompts = JSON.stringify(node._weWorkflowPrompts);
    // Avoid persisting heavyweight extracted snapshot into workflow drafts.
    delete node.properties.we_extracted_cache;

    // Keep a lightweight extracted snapshot for runtime features (e.g. Builder->Builder
    // Update Recipe cache). Restore uses UI state (we_ui_state/override_data), not this.
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
                denoise: ov.denoise ?? 1.0,
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
}

// --- Apply saved overrides back to UI after reload ---
function applyOverrides(node, ovJson, lsJson) {
    let ov, ls;
    try { ov = JSON.parse(ovJson || "{}"); } catch { ov = {}; }
    try { ls = JSON.parse(lsJson || "{}"); } catch { ls = {}; }

    if (node?._weUseSlotProfiles) {
        const slotProfiles = _normalizeSlotProfiles(ov?._slot_profiles);
        node._weSlotProfiles = slotProfiles;

        const activeSlot = _normalizeModelSlotKey(ov?._active_model_slot || ov?._model_slot || "model_a");
        node._weActiveModelSlot = activeSlot;

        const activeProfile = slotProfiles?.[activeSlot];
        if (activeProfile && typeof activeProfile === "object") {
            if (activeProfile.ov && typeof activeProfile.ov === "object") {
                ov = { ...ov, ...activeProfile.ov };
            }
            if (activeProfile.ls && typeof activeProfile.ls === "object" && Object.keys(activeProfile.ls).length > 0) {
                ls = { ...activeProfile.ls };
            }
        }

        ov._model_slot = activeSlot;
    }

    const d = node._weExtracted;
    if (!d) return;
    const isEmpty = (o) => !o || Object.keys(o).length === 0;
    if (isEmpty(ov) && isEmpty(ls)) return;

    const applySelect = (row, ovVal, grouped) => {
        if (!row || ovVal == null || ovVal === "\u2014") return;
        const sel = row._sel;
        if (!sel) return;
        if (ovVal === "") {
            sel.selectedIndex = 0;
            sel.style.color = C.textMuted;
            if (row._resetBtn) row._resetBtn.style.visibility = "hidden";
            return;
        }
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
        const targetFamily = normalizeSelectableFamily(ov._family, "sdxl");
        const resolvedFamily = [...sel.options].some(o => o.value === targetFamily) ? targetFamily : "sdxl";
        sel.value = resolvedFamily;
        node._weFamily = resolvedFamily;
    }

    if (Object.prototype.hasOwnProperty.call(ov, "_show_all_models")) {
        if (node._weSetModelFilterDisabled) {
            node._weSetModelFilterDisabled(!!ov._show_all_models);
        } else {
            node._weShowAllModels = !!ov._show_all_models;
        }
        if (node._weApplyModelFilterTooltip) {
            node._weApplyModelFilterTooltip(node._weModelRow?._sel);
            node._weApplyModelFilterTooltip(node._weModelBRow?._sel);
        }
    }

    if (node._wePullModelSlotRow?._inp) {
        const slot = _normalizeModelSlotKey(ov._model_slot || "model_a");
        node._wePullModelSlotRow._inp.value = slot;
        node._wePullModelSlot = slot;
        if (node._wePullModelSlotRow._setOriginal) node._wePullModelSlotRow._setOriginal(slot);
    }

    if (node._weSendModelSlotRow?._inp) {
        const slot = _normalizeModelSlotKey(ov._model_slot || node._wePullModelSlot || "model_a");
        node._weSendModelSlotRow._inp.value = slot;
        node._weSendModelSlot = slot;
        if (node._weSendModelSlotRow._setOriginal) node._weSendModelSlotRow._setOriginal(slot);
    }

    if (node._weUseSlotProfiles) {
        const lockSrc = (ov._section_locks && typeof ov._section_locks === "object") ? ov._section_locks : {};
        // Multi-slot mode: restore per-slot lock state for all per-model
        // sections so lock toggles do not bleed between slots.
        for (const key of ["model", "sampler", "positive", "negative", "loras"]) {
            if (node._weSetSectionLock) node._weSetSectionLock(key, !!lockSrc[key], { sync: false });
        }
    } else if (ov._section_locks && typeof ov._section_locks === "object") {
        for (const key of ["resolution", "model", "sampler", "positive", "negative", "loras"]) {
            if (Object.prototype.hasOwnProperty.call(ov._section_locks, key) && node._weSetSectionLock) {
                node._weSetSectionLock(key, !!ov._section_locks[key], { sync: false });
            }
        }
    }
    if (ov._res_locked && !(ov._section_locks && Object.prototype.hasOwnProperty.call(ov._section_locks, "resolution"))) {
        if (node._weSetSectionLock) node._weSetSectionLock("resolution", true, { sync: false });
        else if (node._weSetResDisabled) node._weSetResDisabled(true);
    }

    if (Object.prototype.hasOwnProperty.call(ov, "model_a")) {
        applySelect(node._weModelRow, ov.model_a ?? "", true);
    }
    if (Object.prototype.hasOwnProperty.call(ov, "model_b")) {
        applySelect(node._weModelBRow, ov.model_b ?? "", true);
    }
    if (Object.prototype.hasOwnProperty.call(ov, "vae")) {
        applySelect(node._weVaeRow, ov.vae ?? "", true);
    }
    if (Object.prototype.hasOwnProperty.call(ov, "clip_names")) {
        const clipName = Array.isArray(ov.clip_names) && ov.clip_names.length ? ov.clip_names[0] : "";
        applySelect(node._weClipRow, clipName, true);
    }

    if (node._weSamplerRows) {
        const rows = node._weSamplerRows;
        if (ov.steps_a != null) applyInput(rows.steps_a, ov.steps_a);
        if (ov.cfg != null) applyInput(rows.cfg, ov.cfg);
        if (ov.denoise != null) applyInput(rows.denoise, ov.denoise);
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
    if (ov.positive_prompt != null && node._wePosBox) node._wePosBox.value = ov.positive_prompt;
    if (ov.negative_prompt != null && node._weNegBox) node._weNegBox.value = ov.negative_prompt;

    if (Array.isArray(ov.loras_a)) d.loras_a = [...ov.loras_a];
    if (Array.isArray(ov.loras_b)) d.loras_b = [...ov.loras_b];
    if (ov._lora_availability && typeof ov._lora_availability === "object") {
        d.lora_availability = { ...ov._lora_availability };
    }

    if (!isEmpty(ls)) {
        node._weLoraState = ls;
    }

    updateLoras(node);

    updateWanVisibility(node);
    if (node._updatePromptGhosting) node._updatePromptGhosting();
    if (node._updateSeedGhosting) node._updateSeedGhosting();
    syncHidden(node);
}

// --- Parse workflow_data string into extracted format ---
function parseWorkflowData(jsonStr, modelSlot = "model_a") {
    if (!jsonStr) return null;
    try {
        const d = JSON.parse(jsonStr);
        const strictSlot = _normalizeModelSlotKeyStrict(modelSlot);
        if (!strictSlot) {
            return {
                positive_prompt: "",
                negative_prompt: "",
                loras_a: [],
                loras_b: [],
                model_a: "",
                model_b: "",
                model_slot: "",
                vae: { name: "", source: "workflow_data" },
                clip: { names: [], type: "", source: "workflow_data" },
                sampler: {
                    steps_a: 20,
                    cfg: 5.0,
                    denoise: 1.0,
                    seed_a: 0,
                    sampler_name: "euler",
                    scheduler: "simple",
                },
                resolution: { width: 768, height: 1280, batch_size: 1, length: null },
                model_family: "",
            };
        }
        const selectedSlot = strictSlot;

        if (d?.models && typeof d.models === "object") {
            const primary = d.models[selectedSlot] && typeof d.models[selectedSlot] === "object"
                ? d.models[selectedSlot]
                : {};

            const rootSampler = (d.sampler && typeof d.sampler === "object") ? d.sampler : {};
            const primarySampler = (primary.sampler && typeof primary.sampler === "object") ? primary.sampler : {};
            const rootRes = (d.resolution && typeof d.resolution === "object") ? d.resolution : {};
            const primaryRes = (primary.resolution && typeof primary.resolution === "object") ? primary.resolution : {};
            const mergedSampler = { ...rootSampler, ...primarySampler };
            const mergedResolution = { ...rootRes, ...primaryRes };
            const primaryClipNames = Array.isArray(primary.clip)
                ? primary.clip
                : (primary.clip ? [primary.clip] : []);

            return {
                positive_prompt: primary.positive_prompt || "",
                negative_prompt: primary.negative_prompt || "",
                loras_a: Array.isArray(primary.loras) ? primary.loras : [],
                loras_b: [],
                model_a: primary.model || "",
                model_b: "",
                model_slot: selectedSlot,
                vae: { name: primary.vae || "", source: "workflow_data" },
                clip: {
                    names: primaryClipNames,
                    type: primary.clip_type || d.clip_type || "",
                    source: "workflow_data",
                },
                sampler: {
                    steps_a: mergedSampler.steps ?? mergedSampler.steps_a ?? 20,
                    steps_b: undefined,
                    cfg: mergedSampler.cfg ?? 5.0,
                    denoise: mergedSampler.denoise ?? 1.0,
                    seed_a: mergedSampler.seed ?? mergedSampler.seed_a ?? 0,
                    seed_b: undefined,
                    sampler_name: mergedSampler.sampler_name ?? "euler",
                    scheduler: mergedSampler.scheduler ?? "simple",
                },
                resolution: {
                    width: mergedResolution.width ?? 768,
                    height: mergedResolution.height ?? 1280,
                    batch_size: mergedResolution.batch_size ?? 1,
                    length: mergedResolution.length ?? null,
                },
                model_family: primary.family || d.family || "",
            };
        }

        // Legacy flat schema fallback.
        const selectedModel = String(d?.[selectedSlot] ?? "").trim();
        const isExtendedSlot = selectedSlot === "model_c" || selectedSlot === "model_d";

        const slotPrompt = (slot, kind) => {
            const slotVal = String(d?.[`${slot}_${kind}`] ?? "").trim();
            if (slotVal) return slotVal;
            const altVal = String(d?.[`${kind}_${slot}`] ?? "").trim();
            if (altVal) return altVal;
            return "";
        };

        const legacyPositive = isExtendedSlot ? slotPrompt(selectedSlot, "positive_prompt") : String(d.positive_prompt || "");
        const legacyNegative = isExtendedSlot ? slotPrompt(selectedSlot, "negative_prompt") : String(d.negative_prompt || "");

        const legacyPrimaryModel = selectedModel || (
            selectedSlot === "model_a" ? String(d.model_a || "") :
                selectedSlot === "model_b" ? String(d.model_b || "") : ""
        );

        const lorasBySlot = (slot) => {
            if (slot === "model_a") return Array.isArray(d.loras_a) ? d.loras_a : [];
            if (slot === "model_b") return Array.isArray(d.loras_b) ? d.loras_b : [];
            if (slot === "model_c") return Array.isArray(d.loras_c) ? d.loras_c : [];
            if (slot === "model_d") return Array.isArray(d.loras_d) ? d.loras_d : [];
            return [];
        };

        // Map from build_simplified_workflow_data schema to extracted schema
        return {
            positive_prompt: legacyPositive,
            negative_prompt: legacyNegative,
            loras_a: lorasBySlot(selectedSlot),
            loras_b: [],
            model_a: legacyPrimaryModel,
            model_b: "",
            model_slot: selectedSlot,
            model_a_found: d.model_a_found,
            model_b_found: d.model_b_found,
            vae: { name: d.vae || "", source: "workflow_data" },
            vae_found: d.vae_found,
            clip: {
                names: Array.isArray(d.clip) ? d.clip : (d.clip ? [d.clip] : []),
                type: "", source: "workflow_data",
            },
            sampler: d.sampler || {
                steps_a: 20, cfg: 5.0, denoise: 1.0, seed_a: 0,
                sampler_name: "euler", scheduler: "simple",
            },
            resolution: d.resolution || { width: 768, height: 1280, batch_size: 1, length: null },
            model_family: d.family || "",
        };
    } catch (e) {
        console.warn("[RecipeBuilder] Could not parse workflow_data:", e);
        return null;
    }
}

function buildSlotProfilesFromWorkflowData(rawWorkflowData) {
    let d = null;
    if (rawWorkflowData && typeof rawWorkflowData === "object") {
        d = rawWorkflowData;
    } else if (typeof rawWorkflowData === "string" && rawWorkflowData.trim()) {
        try {
            d = JSON.parse(rawWorkflowData);
        } catch {
            d = null;
        }
    }
    if (!d || typeof d !== "object") return null;
    if (!(d.models && typeof d.models === "object")) return null;

    const profiles = _emptySlotProfiles();

    const toProfileFromExtracted = (extracted) => {
        const src = extracted && typeof extracted === "object" ? extracted : {};
        const sampler = src.sampler && typeof src.sampler === "object" ? src.sampler : {};
        const loras = Array.isArray(src.loras_a) ? src.loras_a : [];
        const ls = {};

        for (const lora of loras) {
            const name = String(lora?.name || "").trim();
            if (!name) continue;
            const modelStrength = Number(lora?.model_strength ?? lora?.strength ?? 1.0);
            const clipStrength = Number(lora?.clip_strength ?? lora?.strength ?? modelStrength);
            const entry = {
                active: lora?.active !== false,
                model_strength: Number.isFinite(modelStrength) ? modelStrength : 1.0,
                clip_strength: Number.isFinite(clipStrength) ? clipStrength : (Number.isFinite(modelStrength) ? modelStrength : 1.0),
            };
            ls[name] = entry;
            ls[`a:${name}`] = entry;
        }

        return {
            ov: {
                positive_prompt: String(src.positive_prompt || ""),
                negative_prompt: String(src.negative_prompt || ""),
                model_a: String(src.model_a || ""),
                model_b: "",
                vae: String(src?.vae?.name || ""),
                clip_names: Array.isArray(src?.clip?.names) ? src.clip.names : [],
                clip_type: String(src?.clip?.type || ""),
                loader_type: "",
                _family: String(src.model_family || "sdxl"),
                steps_a: Number(sampler.steps_a ?? 20),
                cfg: Number(sampler.cfg ?? 5.0),
                denoise: Number(sampler.denoise ?? 1.0),
                seed_a: Number(sampler.seed_a ?? 0),
                sampler_name: String(sampler.sampler_name || "euler"),
                scheduler: String(sampler.scheduler || "simple"),
                loras_a: loras,
                loras_b: [],
                _section_locks: {
                    model: false,
                    sampler: false,
                    positive: false,
                    negative: false,
                    loras: false,
                },
            },
            ls,
        };
    };

    for (const slot of MODEL_SLOT_KEYS) {
        const block = d.models?.[slot];
        if (!block || typeof block !== "object") {
            profiles[slot] = _makeEmptySlotProfile();
            continue;
        }

        const extracted = parseWorkflowData(JSON.stringify(d), slot, "simple");
        profiles[slot] = toProfileFromExtracted(extracted);
    }

    return profiles;
}

function mergeSlotProfileWithLocks(currentProfile, incomingProfile) {
    const cur = currentProfile && typeof currentProfile === "object" ? currentProfile : { ov: {}, ls: {} };
    const inc = incomingProfile && typeof incomingProfile === "object" ? incomingProfile : { ov: {}, ls: {} };
    const curOv = cur.ov && typeof cur.ov === "object" ? cur.ov : {};
    const incOv = inc.ov && typeof inc.ov === "object" ? inc.ov : {};
    const curLs = cur.ls && typeof cur.ls === "object" ? cur.ls : {};
    const incLs = inc.ls && typeof inc.ls === "object" ? inc.ls : {};

    const locks = (curOv._section_locks && typeof curOv._section_locks === "object")
        ? curOv._section_locks
        : {};

    const outOv = { ...curOv, ...incOv };
    outOv._section_locks = {
        model: !!locks.model,
        sampler: !!locks.sampler,
        positive: !!locks.positive,
        negative: !!locks.negative,
        loras: !!locks.loras,
    };

    if (locks.model) {
        for (const k of ["model_a", "model_b", "vae", "clip_names", "clip_type", "loader_type", "_family"]) {
            if (Object.prototype.hasOwnProperty.call(curOv, k)) outOv[k] = curOv[k];
        }
    }
    if (locks.sampler) {
        for (const k of ["steps_a", "steps_b", "cfg", "denoise", "seed_a", "seed_b", "sampler_name", "scheduler"]) {
            if (Object.prototype.hasOwnProperty.call(curOv, k)) outOv[k] = curOv[k];
        }
    }
    if (locks.positive && Object.prototype.hasOwnProperty.call(curOv, "positive_prompt")) {
        outOv.positive_prompt = curOv.positive_prompt;
    }
    if (locks.negative && Object.prototype.hasOwnProperty.call(curOv, "negative_prompt")) {
        outOv.negative_prompt = curOv.negative_prompt;
    }

    let outLs = { ...incLs };
    if (locks.loras) {
        if (Object.prototype.hasOwnProperty.call(curOv, "loras_a")) outOv.loras_a = curOv.loras_a;
        if (Object.prototype.hasOwnProperty.call(curOv, "loras_b")) outOv.loras_b = curOv.loras_b;
        outLs = { ...curLs };
    }

    return {
        ov: outOv,
        ls: outLs,
    };
}


// ============================================================
// --- Main extension ---
// ============================================================
app.registerExtension({
    name: "RecipeBuilder",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RecipeBuilder") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated?.apply(this, arguments);
            const node = this;
            const isMultiBuilder = nodeData.name === "RecipeBuilder";
            node._weUseSlotProfiles = isMultiBuilder === true;

            const _normalizeInputNameForVariant = (name) => {
                let key = _canonicalInputName(name);
                if (key === "seed_a") key = "seed";
                if (key === "lora_stack_a") key = "lora_stack";
                return key;
            };

            const _normalizeRecipePortsNow = () => {
                const VALID_INPUTS = new Set([
                    "recipe_data",
                    "builder_data",
                ]);
                if (node.inputs) {
                    for (const inp of node.inputs) {
                        if (!inp || !inp.name) continue;
                        inp.name = _normalizeInputNameForVariant(inp.name);
                    }
                    const keepIndexByName = new Map();
                    for (let i = 0; i < node.inputs.length; i++) {
                        const inp = node.inputs[i];
                        if (!inp || !inp.name) continue;
                        const key = _normalizeInputNameForVariant(inp.name);
                        if (!VALID_INPUTS.has(key)) continue;
                        if (!keepIndexByName.has(key)) {
                            keepIndexByName.set(key, i);
                            continue;
                        }
                        const keepIdx = keepIndexByName.get(key);
                        const keepInp = node.inputs[keepIdx];
                        const curLinked = inp?.link != null;
                        const keepLinked = keepInp?.link != null;
                        if (curLinked && !keepLinked) keepIndexByName.set(key, i);
                    }
                    for (let i = node.inputs.length - 1; i >= 0; i--) {
                        const inp = node.inputs[i];
                        if (!inp || !inp.name) continue;
                        const key = _normalizeInputNameForVariant(inp.name);
                        if (!VALID_INPUTS.has(key) || keepIndexByName.get(key) !== i) {
                            node.removeInput(i);
                        }
                    }

                    const hasInput = (name) => node.inputs.some((inp) => _normalizeInputNameForVariant(inp?.name || "") === name);
                    if (!hasInput("recipe_data")) node.addInput("recipe_data", "RECIPE_DATA");
                    if (!hasInput("builder_data")) node.addInput("builder_data", "BUILDER_DATA");
                }

                const VALID_OUTPUTS = [
                    { name: "recipe_data", type: "RECIPE_DATA" },
                ];
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
            };

            _normalizeRecipePortsNow();

            // -- State --
            node._weExtracted = null;
            node._weLoraState = {};
            node._weOverrides = {};
            node._weFamily = "sdxl";
            node._weShowAllModels = false;
            node._weSections = {};
            node._weCollapsedSections = { resolution: true, model: true, sampler: true, positive: false, negative: false, loras: false };
            node._weSectionLocks = {
                resolution: false,
                model: false,
                sampler: false,
                positive: false,
                negative: false,
                loras: false,
            };
            // During graph/tab rehydrate, callbacks like onConnectionsChange can
            // fire before restore completes. Guard sync writes until hydration ends.
            node._weHydrating = true;
            this.serialize_widgets = true;

            const _syncS = ({ force = false } = {}) => {
                if (node._weHydrating && !force) return;
                syncHidden(node);
            };
            node._weSetSectionLock = (sectionName, locked, { sync = true } = {}) => {
                const key = String(sectionName || "");
                if (!key) return;
                const val = !!locked;
                node._weSectionLocks[key] = val;
                const sec = node._weSections?.[key];
                if (sec?._setLocked) sec._setLocked(val);
                if (key === "resolution") {
                    node._weResLocked = val;
                    if (node._weSetResDisabled) node._weSetResDisabled(val);
                }
                if (sync) _syncS();
            };

            // Ensure latest UI state is captured in hidden widgets/properties
            // when the graph is serialized (save/export), even if some controls
            // haven't emitted their final change event yet.
            const origOnSerialize = node.onSerialize;
            node.onSerialize = function (o) {
                try { syncHidden(node); } catch { /* non-fatal */ }
                if (origOnSerialize) return origOnSerialize.apply(this, arguments);
                return o;
            };

            // ============================================================
            // -- Build the DOM UI --
            // ============================================================
            const root = makeEl("div", {
                position: "relative",
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

            const FILTER_ICON_GLYPH = "👁\uFE0E";

            // -- Top-right help icon on node frame (canvas-drawn) --
            node._weHelpText = [
                "Connect recipe_data and execute the workflow to refresh Builder data.",
                "Builder can also be used standalone or chained in a workflow.",
                "- If chaining, first Builder drives downstream Builders.",
                "- Chained Builders update their UI at execution.",
                "- Builders can be executed by pressing their Comfy ▶️ button.",
                `Model family filtering can be toggled with ${FILTER_ICON_GLYPH}.`,
                "Lock any section to prevent changes when executing the workflow.",
            ].join("\n");
            node._weHelpVisible = false;
            node._weHelpIconRect = null;

            const origDrawForeground = node.onDrawForeground;
            node.onDrawForeground = function (ctx) {
                if (origDrawForeground) origDrawForeground.apply(this, arguments);

                const w = this.size?.[0] || 0;
                const x = Math.max(4, w - 18);
                const y = -12;
                this._weHelpIconRect = { x: x - 4, y: y - 7, w: 12, h: 14 };

                ctx.save();
                ctx.fillStyle = "rgba(255,255,255,0.9)";
                ctx.font = "bold 19px sans-serif";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillText("?", x, y);
                ctx.restore();
            };

            const origMouseDown = node.onMouseDown;
            node.onMouseDown = function (e, pos) {
                const r = this._weHelpIconRect;
                const overHelp = !!(r && pos && pos[0] >= r.x && pos[0] <= (r.x + r.w) && pos[1] >= r.y && pos[1] <= (r.y + r.h));
                if (overHelp) {
                    this._weHelpVisible = !this._weHelpVisible;
                    _showInfo(this, this._weHelpVisible ? this._weHelpText : null);
                    this.setDirtyCanvas(true, true);
                    return true;
                }
                if (origMouseDown) return origMouseDown.apply(this, arguments);
                return false;
            };

            const origMouseMove = node.onMouseMove;
            node.onMouseMove = function (e, pos, graphcanvas) {
                const r = this._weHelpIconRect;
                const overHelp = !!(r && pos && pos[0] >= r.x && pos[0] <= (r.x + r.w) && pos[1] >= r.y && pos[1] <= (r.y + r.h));
                if (graphcanvas?.canvas) {
                    graphcanvas.canvas.style.cursor = overHelp ? "help" : "";
                }
                if (origMouseMove) return origMouseMove.apply(this, arguments);
                return false;
            };

            // Execute-only flow: no manual Update Recipe button.
            node._weUpdateBtn = null;

            const pullSlotOptions = [...MODEL_SLOT_KEYS];
            const sendSlotOptions = [...MODEL_SLOT_KEYS];

            const pullModelSlotRow = makeInput("Pull From", "select", "model_a", {
                options: pullSlotOptions,
            }, () => {
                const picked = _normalizeModelSlotKey(pullModelSlotRow._inp?.value);
                pullModelSlotRow._inp.value = picked;
                node._wePullModelSlot = picked;
                _syncS();
            });
            pullModelSlotRow._inp.title = "Select which model slot to pull from when reading connected recipe_data during execute.";
            pullModelSlotRow._inp.style.color = C.accent;
            pullModelSlotRow._inp.style.fontWeight = "bold";
            pullModelSlotRow.style.marginBottom = "0";
            node._wePullModelSlotRow = pullModelSlotRow;
            node._wePullModelSlot = "model_a";

            const sendModelSlotRow = makeInput("Send To", "select", "model_a", {
                options: sendSlotOptions,
            }, () => {
                const picked = _normalizeModelSlotKey(sendModelSlotRow._inp?.value || "model_a");
                sendModelSlotRow._inp.value = picked;
                node._weSendModelSlot = picked;
                _syncS();
            });
            sendModelSlotRow._inp.title = "Select which model slot this Builder writes into on output.";
            sendModelSlotRow.style.marginBottom = "0";
            node._weSendModelSlotRow = sendModelSlotRow;
            node._weSendModelSlot = "model_a";
            node._weActiveModelSlot = "model_a";
            node._weSlotProfiles = _emptySlotProfiles();

            if (node._weUseSlotProfiles) {
                sendModelSlotRow._label.textContent = "Model Slot";
                sendModelSlotRow._inp.title = "Switch between model slots A/B/C/D. Each slot keeps its own Recipe Builder settings.";
                pullModelSlotRow.style.display = "none";

                const _captureCurrentSlotProfile = () => {
                    const slot = _normalizeModelSlotKey(node._weActiveModelSlot || node._weSendModelSlot || "model_a");
                    const slotProfiles = _normalizeSlotProfiles(node._weSlotProfiles);
                    const ovRaw = node.properties?.we_ui_state || node.properties?.we_override_data || "{}";
                    const lsRaw = node.properties?.we_lora_state || "{}";
                    let ovObj = {};
                    let lsObj = {};
                    try { ovObj = JSON.parse(ovRaw || "{}"); } catch { ovObj = {}; }
                    try { lsObj = JSON.parse(lsRaw || "{}"); } catch { lsObj = {}; }
                    delete ovObj._slot_profiles;
                    delete ovObj._active_model_slot;
                    // Resolution is shared for all model slots.
                    delete ovObj.width;
                    delete ovObj.height;
                    delete ovObj.batch_size;
                    delete ovObj.length;
                    delete ovObj._ratio;
                    delete ovObj._ratio_landscape;
                    delete ovObj._res_locked;
                    if (ovObj._section_locks && typeof ovObj._section_locks === "object") {
                        const nextLocks = { ...ovObj._section_locks };
                        delete nextLocks.resolution;
                        ovObj._section_locks = nextLocks;
                    }
                    if (ovObj._section_collapsed && typeof ovObj._section_collapsed === "object") {
                        const nextCollapsed = { ...ovObj._section_collapsed };
                        delete nextCollapsed.resolution;
                        ovObj._section_collapsed = nextCollapsed;
                    }
                    ovObj._model_slot = slot;
                    slotProfiles[slot] = { ov: ovObj, ls: lsObj };
                    node._weSlotProfiles = slotProfiles;
                };

                const _loadSlotProfile = (slot) => {
                    const targetSlot = _normalizeModelSlotKey(slot);
                    const slotProfiles = _normalizeSlotProfiles(node._weSlotProfiles);
                    let profile = slotProfiles[targetSlot];
                    if (!profile || typeof profile !== "object") {
                        profile = {
                            ov: {
                                positive_prompt: "",
                                negative_prompt: "",
                                model_a: "",
                                model_b: "",
                                vae: "",
                                clip_names: [],
                                _family: "",
                                steps_a: 20,
                                cfg: 5.0,
                                denoise: 1.0,
                                seed_a: 0,
                                sampler_name: "euler",
                                scheduler: "simple",
                                loras_a: [],
                                loras_b: [],
                                _section_locks: {
                                    model: false,
                                    sampler: false,
                                    positive: false,
                                    negative: false,
                                    loras: false,
                                },
                            },
                            ls: {},
                        };
                        slotProfiles[targetSlot] = profile;
                    }
                    const profileOv = (profile.ov && typeof profile.ov === "object") ? { ...profile.ov } : {};
                    profileOv._model_slot = targetSlot;
                    profileOv._active_model_slot = targetSlot;
                    profileOv._slot_profiles = slotProfiles;

                    node._weActiveModelSlot = targetSlot;
                    node._weSendModelSlot = targetSlot;
                    node._wePullModelSlot = targetSlot;

                    applyOverrides(node, JSON.stringify(profileOv), JSON.stringify(profile.ls || {}));
                };

                const _onSlotChanged = () => {
                    if (node._weHydrating) {
                        const pickedHydrating = _normalizeModelSlotKey(sendModelSlotRow._inp?.value || "model_a");
                        node._weActiveModelSlot = pickedHydrating;
                        node._weSendModelSlot = pickedHydrating;
                        node._wePullModelSlot = pickedHydrating;
                        _syncS();
                        return;
                    }

                    syncHidden(node);

                    const picked = _normalizeModelSlotKey(sendModelSlotRow._inp?.value || "model_a");
                    _loadSlotProfile(picked);
                    _syncS();
                };

                sendModelSlotRow._inp.onchange = _onSlotChanged;
                sendModelSlotRow._inp.oninput = _onSlotChanged;
            }

            const _findWorkflowDataInput = (targetNode = node) => {
                const ins = targetNode?.inputs || [];
                let fallback = null;
                for (const inp of ins) {
                    if (!inp) continue;
                    if (_canonicalInputName(inp.name) !== "recipe_data") continue;
                    if (fallback == null) fallback = inp;
                    if (inp.link != null) return inp;
                }
                return fallback;
            };
            const _normalizeNodeKind = (v) => String(v || "").toLowerCase().replace(/[^a-z0-9]/g, "");

            const _isUpdateSourceHint = (n) => {
                if (!n) return false;
                const ccNorm = _normalizeNodeKind(n.comfyClass);
                const tyNorm = _normalizeNodeKind(n.type);
                return ccNorm === "promptextractor" || ccNorm === "recipeextractor" ||
                    tyNorm === "promptextractor" || tyNorm === "recipeextractor" ||
                    ccNorm === "promptmanageradvanced" || tyNorm === "promptmanageradvanced" ||
                    ccNorm === "recipemanager" || tyNorm === "recipemanager" ||
                    ccNorm === "reciperelay" || tyNorm === "reciperelay";
            };
            const _isRerouteNodeHint = (n) => {
                if (!n) return false;
                const cc = (n.comfyClass || "").toLowerCase();
                const ty = (n.type || "").toLowerCase();
                return cc.includes("reroute") || ty.includes("reroute");
            };
            const _resolveUpstreamNodeHint = (graph, startLinkId) => {
                if (!graph || startLinkId == null) return null;
                let linkId = startLinkId;
                const seen = new Set();
                for (let hop = 0; hop < 24; hop++) {
                    if (linkId == null || seen.has(linkId)) break;
                    seen.add(linkId);
                    const linkInfo = graph.links?.[linkId];
                    if (!linkInfo) break;
                    const srcNode = graph.getNodeById?.(linkInfo.origin_id);
                    if (!srcNode) break;
                    if (!_isRerouteNodeHint(srcNode)) return srcNode;
                    const in0 = srcNode.inputs?.[0];
                    linkId = in0?.link ?? null;
                }
                return null;
            };
            const _syncPullFromOptionMode = (recipeDataConnected) => {
                const row = node._wePullModelSlotRow;
                const sel = row?._inp;
                if (!sel) return;

                if (node._weUseSlotProfiles) {
                    const active = _normalizeModelSlotKey(node._weActiveModelSlot || node._weSendModelSlot || "model_a");
                    node._weActiveModelSlot = active;
                    node._wePullModelSlot = active;
                    node._weSendModelSlot = active;
                    sel.value = active;
                    return;
                }

                const wasConnected = node._weRecipeDataConnectedState;
                const hasPrevState = typeof wasConnected === "boolean";
                const connectionChanged = !hasPrevState || wasConnected !== recipeDataConnected;
                node._weRecipeDataConnectedState = recipeDataConnected;

                const current = String(sel.value || node._wePullModelSlot || "");
                const preferredConnected = _normalizeModelSlotKey(current || "model_a");

                // Auto-select defaults only when connection mode actually changes and
                // never during workflow hydration (restore should win).
                let nextVal = sel.value || current;
                if (!node._weHydrating && connectionChanged) {
                    nextVal = preferredConnected;
                }
                if (![...sel.options].some((o) => o.value === nextVal)) {
                    nextVal = preferredConnected;
                }

                const prevVal = node._wePullModelSlot;
                sel.value = nextVal;
                node._wePullModelSlot = nextVal;
                if (row._setOriginal) row._setOriginal(nextVal);
                if (!node._weHydrating && prevVal !== nextVal) syncHidden(node);
            };
            const _isConnectedWorkflowDataExtractor = () => {
                const wfInput = _findWorkflowDataInput(node);
                if (wfInput?.link == null) return false;
                const upstream = _resolveUpstreamNodeHint(node.graph, wfInput.link);
                return _isUpdateSourceHint(upstream);
            };
            node._weRefreshUpdateWorkflowTooltip = null;
            node._weRefreshUpdateWorkflowButtonVisibility = null;
            node._weDeferUpdateWorkflowButtonRefresh = null;
            node._weIsConnectedWorkflowDataExtractor = _isConnectedWorkflowDataExtractor;
            // Legacy update-button interactions removed.
            const modelSlotGroup = makeEl("div", {
                display: "flex",
                flexDirection: "column",
                gap: "2px",
            });
            modelSlotGroup.appendChild(pullModelSlotRow);
            modelSlotGroup.appendChild(sendModelSlotRow);

            const modelSlotBox = makeEl("div", {
                borderRadius: "6px",
                overflow: "hidden",
                marginTop: "2px",
                marginBottom: "2px",
                backgroundColor: C.bgCard,
                flexShrink: "0",
                padding: "4px 8px",
            });
            modelSlotBox.appendChild(modelSlotGroup);
            // Intentionally do not render manual "Update Recipe" button.
            // Builder refreshes from recipe_data during normal execute flow.

            // -- 1. RESOLUTION section (open) --
            const resSec = makeSection("RESOLUTION", node, "resolution", { collapsed: !!node._weCollapsedSections?.resolution });
            node._weSections.resolution = resSec;

            // Aspect ratio definitions — stored as portrait (w < h)
            const RATIOS = [
                { w: 0,  h: 0  },   // None
                { w: 1,  h: 1  },
                { w: 4,  h: 5  },
                { w: 3,  h: 4  },
                { w: 2,  h: 3  },
                { w: 9,  h: 16 },
                { w: 1,  h: 2  },
            ];
            node._weRatioDefs = RATIOS;
            function _ratioLabel(r, land) {
                if (r.w === 0) return "None";
                return land ? `${r.h}:${r.w}` : `${r.w}:${r.h}`;
            }

            let _landscape = false;
            let _resLocked = false;
            let _currentRatioIdx = 0;
            node._weResLocked = false;
            node._weRatioLandscape = false;
            node._weRatio = "None";

            // Helper: ghost / un-ghost resolution inputs + update lock icon
            function _setResDisabled(disabled) {
                if (!resRows) return;
                for (const key of ["width", "height", "batch", "frames"]) {
                    const inp = resRows[key]?._inp;
                    if (inp) { inp.disabled = disabled; inp.style.opacity = disabled ? "0.35" : "1"; }
                    const lbl = resRows[key]?._label;
                    if (lbl) lbl.style.opacity = disabled ? "0.35" : "1";
                }
            }
            resSec._onLockChanged = (locked) => {
                _resLocked = !!locked;
                node._weResLocked = _resLocked;
                _setResDisabled(_resLocked);
                node._weSectionLocks.resolution = _resLocked;
                _syncS();
            };

            // --- Ratio row: [label] [reset spacer] [dropdown] [orient icon] ---
            const ratioRow = makeEl("div", { ...ROW_STYLE });
            ratioRow.appendChild(makeEl("span", {
                color: C.textMuted, width: LABEL_W, flexShrink: "0",
            }, "Ratio"));
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

            // Orient icon in the right icon column
            const orientIcon = makeEl("span", {
                cursor: "pointer", display: "inline-flex", alignItems: "center",
                justifyContent: "center", width: "18px", flexShrink: "0", marginLeft: "auto",
            });
            function _drawOrient() {
                if (_landscape) {
                    orientIcon.innerHTML = `<svg width="17" height="13" viewBox="0 0 14 11" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x=".5" y=".5" width="13" height="10" rx="1" stroke="${UI_ICON_COLOR}" stroke-width="${UI_ICON_STROKE}"/><circle cx="7" cy="3.8" r="1.35" fill="${UI_ICON_COLOR_DIM}"/><path d="M4.5 9.5v-.8c0-1.1.9-2 2-2h1c1.1 0 2 .9 2 2v.8" stroke="${UI_ICON_COLOR_DIM}" stroke-width="${UI_ICON_STROKE}" stroke-linecap="round" fill="none"/></svg>`;
                } else {
                    orientIcon.innerHTML = `<svg width="12" height="17" viewBox="0 0 10 14" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x=".5" y=".5" width="9" height="13" rx="1" stroke="${UI_ICON_COLOR}" stroke-width="${UI_ICON_STROKE}"/><circle cx="5" cy="4.5" r="1.35" fill="${UI_ICON_COLOR_DIM}"/><path d="M2.5 12.5v-1c0-1.2 1-2.2 2.2-2.2h.6c1.2 0 2.2 1 2.2 2.2v1" stroke="${UI_ICON_COLOR_DIM}" stroke-width="${UI_ICON_STROKE}" stroke-linecap="round" fill="none"/></svg>`;
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

            // Assemble section
            resSec._body.appendChild(ratioRow);
            for (const key of ["width", "height", "batch", "frames"]) {
                resSec._body.appendChild(resRows[key]);
            }
            root.appendChild(resSec);
            if (node._weUseSlotProfiles) {
                root.appendChild(modelSlotBox);
            } else {
                root.appendChild(modelSlotBox);
            }

            node._weResRows = resRows;
            node._weRatio = "None";
            node._weRatioSel = ratioSel;
            node._weRatioLandBtn = orientIcon;
            node._weSetLandscape = (v) => {
                _landscape = !!v;
                node._weRatioLandscape = _landscape;
                _drawOrient();
                _populateRatioSel();
            };
            node._weSetRatioIdx = (i) => { _currentRatioIdx = i; };
            node._weSetResDisabled = _setResDisabled;
            if (resSec._setLocked) resSec._setLocked(false);

            // -- 2. MODEL section (open, with VAE/CLIP) --
            const modelSec = makeSection("MODEL", node, "model", { collapsed: !!node._weCollapsedSections?.model });
            node._weSections.model = modelSec;
            modelSec._onLockChanged = (locked) => {
                node._weSectionLocks.model = !!locked;
                _syncS();
            };
            if (modelSec._setLocked) modelSec._setLocked(false);

            // Family type row
            const familyRow = makeEl("div", { ...ROW_STYLE });
            familyRow.appendChild(makeEl("span", {
                color: C.text, width: LABEL_W, flexShrink: "0", fontWeight: "bold",
            }, "Type"));
            familyRow.appendChild(makeEl("span", { width: "14px", flexShrink: "0" }));
            const familySel = document.createElement("select");
            Object.assign(familySel.style, { ...INPUT_STYLE, color: C.accent, fontWeight: "bold" });
            const modelFilterIcon = makeEl("span", {
                width: "20px", flexShrink: "0", display: "inline-flex",
                alignItems: "center", justifyContent: "center", cursor: "pointer", marginLeft: "auto",
                lineHeight: "1",
            });
            const _eyeIconSvg = (color) => (`<svg width="18" height="14" viewBox="0 0 18 14" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M1 7C2.9 3.8 5.7 2 9 2C12.3 2 15.1 3.8 17 7C15.1 10.2 12.3 12 9 12C5.7 12 2.9 10.2 1 7Z" stroke="${color}" stroke-width="${UI_ICON_STROKE}" stroke-linecap="round" stroke-linejoin="round"/><circle cx="9" cy="7" r="1.5" fill="${color}"/></svg>`);
            let _familiesLoaded = false;
            let _familiesLoadingPromise = null;
            const _ensureFamiliesLoaded = async () => {
                if (_familiesLoaded || familySel._familiesLoaded) return;
                if (_familiesLoadingPromise) return _familiesLoadingPromise;

                _familiesLoadingPromise = (async () => {
                    try {
                        const r = await fetch("/workflow-extractor/list-families");
                        const d = await r.json();
                        let families = d.families || {};
                        const curVal = familySel.value || node._weFamily || "sdxl";
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
                        if (!familySel.options.length) {
                            const o = document.createElement("option");
                            o.value = "sdxl";
                            o.textContent = "SDXL";
                            familySel.appendChild(o);
                        }
                        familySel.value = [...familySel.options].some((o) => o.value === curVal)
                            ? curVal
                            : (familySel.options[0]?.value || "sdxl");
                        _familiesLoaded = true;
                        familySel._familiesLoaded = true;
                    } catch (e) { /* ignore */ }
                })();

                try {
                    await _familiesLoadingPromise;
                } finally {
                    _familiesLoadingPromise = null;
                }
            };

            familySel.onfocus = _ensureFamiliesLoaded;
            // Add SDXL as initial option
            {
                const o = document.createElement("option");
                o.value = "sdxl";
                o.textContent = "SDXL";
                familySel.appendChild(o);
                familySel.value = "sdxl";
            }
            requestAnimationFrame(() => {
                _ensureFamiliesLoaded().catch(() => { /* ignore */ });
            });
            familySel.onchange = () => {
                // Persist family immediately so fast tab-switches do not lose it.
                const picked = familySel.value || "sdxl";
                node._weFamily = picked;
                node._weOverrides = node._weOverrides || {};
                node._weOverrides._family = node._weFamily;
                _syncS();
                onFamilyChanged(node._weFamily);
            };
            familyRow.appendChild(familySel);
            familyRow.appendChild(modelFilterIcon);
            modelSec._body.appendChild(familyRow);
            node._weFamilySel = familySel;

            const updateModelFilterIcon = () => {
                const showAll = !!node._weShowAllModels;
                const iconColor = showAll ? "rgba(255,120,120,1)" : "rgba(255,255,255,0.9)";
                modelFilterIcon.innerHTML = _eyeIconSvg(iconColor);
                modelFilterIcon.title = showAll
                    ? "Model filtering OFF (showing all models). Click to filter by family."
                    : "Model filtering ON (by family). Click to show all models.";
            };
            updateModelFilterIcon();
            node._weModelFilterIcon = modelFilterIcon;
            node._weSetModelFilterDisabled = (disabled) => {
                node._weShowAllModels = !!disabled;
                updateModelFilterIcon();
            };

            const applyModelFilterTooltip = (sel) => {
                if (!sel) return;
                const filterOff = !!node._weShowAllModels;
                const famText = (node._weFamilySel?.selectedOptions?.[0]?.textContent || node._weFamily || "selected family");
                sel.title = filterOff
                    ? `Showing all models on disk. Family (${famText}) still controls renderer pipeline.`
                    : `Showing models filtered for family: ${famText}.`;
            };
            node._weApplyModelFilterTooltip = applyModelFilterTooltip;
            familySel.title = "Select family (render pipeline type).";

            // Fetch models helper
            const fetchBaseModels = async () => {
                try {
                    applyModelFilterTooltip(node._weModelRow?._sel);
                    applyModelFilterTooltip(node._weModelBRow?._sel);
                    const fam = node._weFamily;
                    const baseUrl = fam
                        ? `/workflow-extractor/list-models?family=${encodeURIComponent(fam)}`
                        : `/workflow-extractor/list-models`;
                    const url = withModelFilteringPreference(baseUrl, node._weShowAllModels);
                    const r = await fetch(url); const d = await r.json();
                    return d.models || [];
                } catch { return []; }
            };

            const getWanSlotSplitPreference = () => {
                const activeSlot = _normalizeModelSlotKey(node._weActiveModelSlot || node._weSendModelSlot || "model_a");
                if (activeSlot === "model_b" || activeSlot === "model_d") {
                    return { preferHigh: false, preferLow: true };
                }
                return { preferHigh: true, preferLow: false };
            };

            const fetchModelsA = async () => {
                const models = await fetchBaseModels();
                if (node._weShowAllModels) return models;
                return applyWanVideoModelSplit(models, node._weFamily, getWanSlotSplitPreference());
            };
            const fetchModelsB = async () => {
                const models = await fetchBaseModels();
                if (node._weShowAllModels) return models;
                return applyWanVideoModelSplit(models, node._weFamily, { preferLow: true });
            };

            const reloadModelRowsForFamily = async (familyKey) => {
                const fam = encodeURIComponent(familyKey || "");
                const fetchModelsForFamilyBase = async () => {
                    try {
                        const baseUrl = `/workflow-extractor/list-models?family=${fam}`;
                        const url = withModelFilteringPreference(baseUrl, node._weShowAllModels);
                        const r = await fetch(url);
                        const d = await r.json(); return d.models || [];
                    } catch { return []; }
                };
                const fetchModelsForFamilyA = async () => {
                    const models = await fetchModelsForFamilyBase();
                    if (node._weShowAllModels) return models;
                    return applyWanVideoModelSplit(models, familyKey, getWanSlotSplitPreference());
                };
                const fetchModelsForFamilyB = async () => {
                    const models = await fetchModelsForFamilyBase();
                    if (node._weShowAllModels) return models;
                    return applyWanVideoModelSplit(models, familyKey, { preferLow: true });
                };
                await Promise.all([
                    reloadGroupedSelect(node._weModelRow, fetchModelsForFamilyA, true, null, false, false),
                    reloadGroupedSelect(node._weModelBRow, fetchModelsForFamilyB, true, null, false, false),
                ]);
            };

            node._weEnsureModelSelection = async ({ reloadIfEmpty = false, sync = true } = {}) => {
                const sel = node._weModelRow?._sel;
                if (!sel) return false;

                const pickFirst = () => {
                    const first = [...sel.options].find(o => o.value);
                    if (!first) return false;
                    sel.value = first.value;
                    node._weOverrides = node._weOverrides || {};
                    node._weOverrides.model_a = first.value;
                    if (node._weExtracted) node._weExtracted.model_a = first.value;
                    return true;
                };

                if (sel.value) return true;
                if (pickFirst()) {
                    if (sync) _syncS();
                    return true;
                }

                if (reloadIfEmpty) {
                    const fam = node._weFamily || node._weFamilySel?.value || "sdxl";
                    await reloadModelRowsForFamily(fam);
                    if (pickFirst()) {
                        if (sync) _syncS();
                        return true;
                    }
                }

                return false;
            };

            modelFilterIcon.onclick = async () => {
                const prevModelA = node._weModelRow?._sel?.value || "";
                const prevModelB = node._weModelBRow?._sel?.value || "";

                node._weShowAllModels = !node._weShowAllModels;
                updateModelFilterIcon();
                applyModelFilterTooltip(node._weModelRow?._sel);
                applyModelFilterTooltip(node._weModelBRow?._sel);
                await reloadModelRowsForFamily(node._weFamily);

                const selA = node._weModelRow?._sel;
                if (selA) {
                    const hasPrevA = prevModelA && [...selA.options].some(o => o.value === prevModelA);
                    if (hasPrevA) {
                        selA.value = prevModelA;
                    } else {
                        const firstA = selA.options?.[0]?.value || "";
                        if (firstA) selA.value = firstA;
                    }
                }

                const selB = node._weModelBRow?._sel;
                if (selB) {
                    const hasPrevB = prevModelB && [...selB.options].some(o => o.value === prevModelB);
                    if (hasPrevB) {
                        selB.value = prevModelB;
                    } else {
                        const firstB = selB.options?.[0]?.value || "";
                        if (firstB) selB.value = firstB;
                    }
                }

                _syncS();
            };

            // Model A row (label updated by updateWanVisibility)
            const modelRow = makeSelectRow("Model", "", fetchModelsA,
                (v) => { node._weOverrides.model_a = v; _syncS(); }, true, false);
            modelSec._body.appendChild(modelRow);
            node._weModelRow = modelRow;
            applyModelFilterTooltip(modelRow._sel);

            // Preload model options once after mount so the first dropdown open
            // does not race with async option rebuilding.
            requestAnimationFrame(() => {
                node._weModelRow?._ensureLoaded?.().catch(() => { /* ignore */ });
            });

            // Model B row (hidden by default, shown by updateWanVisibility for WAN Video)
            const modelBRow = makeSelectRow("Model B", "", fetchModelsB,
                (v) => { node._weOverrides.model_b = v; _syncS(); }, true, false);
            modelBRow.style.display = "none";
            modelSec._body.appendChild(modelBRow);
            node._weModelBRow = modelBRow;
            applyModelFilterTooltip(modelBRow._sel);

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
                (v) => { if (v) node._weOverrides.vae = v; else delete node._weOverrides.vae; _syncS(); }, false, true, false);
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
                (v) => { if (v) node._weOverrides.clip_names = [v]; else delete node._weOverrides.clip_names; _syncS(); }, false, true, false);
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
                // Persist early before async reloads complete.
                node._weOverrides = node._weOverrides || {};
                node._weOverrides._family = familyKey;
                _syncS();
                applyModelFilterTooltip(node._weModelRow?._sel);
                applyModelFilterTooltip(node._weModelBRow?._sel);
                // Reset VAE/CLIP overrides — new family means old selections are invalid
                delete node._weOverrides.vae;
                delete node._weOverrides.clip_names;
                const fam = encodeURIComponent(familyKey || "");
                const reloadVae = async () => {
                    try {
                        const r = await fetch(`/workflow-extractor/list-vaes?family=${fam}`);
                        const d = await r.json();
                        await reloadGroupedSelect(node._weVaeRow, async () => d.vaes || [], false, null, true, false);
                    } catch { await reloadGroupedSelect(node._weVaeRow, async () => [], false, null, true, false); }
                };
                const reloadClip = async () => {
                    try {
                        const r = await fetch(`/workflow-extractor/list-clips?family=${fam}`);
                        const d = await r.json();
                        await reloadGroupedSelect(node._weClipRow, async () => d.clips || [], false, null, true, false);
                    } catch { await reloadGroupedSelect(node._weClipRow, async () => [], false, null, true, false); }
                };
                await Promise.all([
                    reloadModelRowsForFamily(familyKey),
                    reloadVae(),
                    reloadClip(),
                ]);
                await node._weEnsureModelSelection({ reloadIfEmpty: false, sync: false });
                if (!fromUpdateUI) {
                    // Manual family change — auto-select first model and sync.
                    // updateUI calls _setOriginal itself so we skip this.
                    const firstModel = node._weModelRow._sel.options[0]?.value || "";
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
                        if (rows.denoise?._inp)    rows.denoise._inp.value = (defs.denoise != null ? defs.denoise : 1.0);
                        if (defs.sampler             && rows.sampler?._inp)    rows.sampler._inp.value = defs.sampler;
                        if (defs.scheduler           && rows.scheduler?._inp)  rows.scheduler._inp.value = defs.scheduler;
                    }
                    updateWanVisibility(node);
                    _syncS();
                }
            };
            // Expose so updateUI can trigger family reload from recipe_data
            node._onFamilyChanged = onFamilyChanged;

            // -- 3. SAMPLER section --
            const sampSec = makeSection("SAMPLER", node, "sampler", { collapsed: !!node._weCollapsedSections?.sampler });
            node._weSections.sampler = sampSec;
            sampSec._onLockChanged = (locked) => {
                node._weSectionLocks.sampler = !!locked;
                _syncS();
            };
            if (sampSec._setLocked) sampSec._setLocked(false);

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
                denoise:    makeInput("Denoise",   "number", 1.0,      { min: 0, max: 1, step: 0.01 }, _syncS),
                seed_a:     seedRow,
                seed_b:     seedBRow,
            };
            // Append in display order: steps_a, steps_b, cfg, sampler, scheduler, denoise, seed, seed_b
            sampSec._body.appendChild(stepsARow);
            sampSec._body.appendChild(stepsBRow);
            sampSec._body.appendChild(sampRows.cfg);
            sampSec._body.appendChild(sampRows.sampler);
            sampSec._body.appendChild(sampRows.scheduler);
            sampSec._body.appendChild(sampRows.denoise);
            sampSec._body.appendChild(seedRow);
            sampSec._body.appendChild(seedBRow);

            root.appendChild(sampSec);
            node._weSamplerRows = sampRows;

            // Fetch live sampler/scheduler lists and update dropdowns
            _fetchSamplerSchedulerLists().then(() => {
                _refreshSelectOptions(sampRows.sampler?._inp, SAMPLERS);
                _refreshSelectOptions(sampRows.scheduler?._inp, SCHEDULERS);
            });

            // -- 4. PROMPTS (positive open, negative closed) --
            const posSec = makeSection("POSITIVE PROMPT", node, "positive", { collapsed: !!node._weCollapsedSections?.positive });
            node._weSections.positive = posSec;
            posSec._onLockChanged = (locked) => {
                node._weSectionLocks.positive = !!locked;
                _syncS();
            };
            if (posSec._setLocked) posSec._setLocked(false);
            const posBox = document.createElement("textarea");
            posBox.placeholder = "Positive prompt";
            posBox.rows = 8;
            Object.assign(posBox.style, {
                width: "100%", boxSizing: "border-box",
                background: C.bgInput, color: C.text,
                border: `1px solid ${C.border}`, borderRadius: "4px",
                fontSize: "12px", padding: "6px", resize: "none",
                fontFamily: "inherit", lineHeight: "1.4",
                minHeight: "128px", height: "128px", maxHeight: "none", overflow: "auto",
            });
            posBox.onchange = _syncS;
            posBox.oninput = () => { _syncS(); requestAnimationFrame(() => reflowNode(node)); };
            posBox.addEventListener("wheel", (e) => { e.stopPropagation(); });
            posSec._body.appendChild(posBox);
            root.appendChild(posSec);
            node._wePosBox = posBox;

            const negSec = makeSection("NEGATIVE PROMPT", node, "negative", { collapsed: !!node._weCollapsedSections?.negative });
            node._weSections.negative = negSec;
            negSec._onLockChanged = (locked) => {
                node._weSectionLocks.negative = !!locked;
                _syncS();
            };
            if (negSec._setLocked) negSec._setLocked(false);
            const negBox = document.createElement("textarea");
            negBox.placeholder = "Negative prompt";
            negBox.rows = 8;
            Object.assign(negBox.style, {
                width: "100%", boxSizing: "border-box",
                background: C.bgInput, color: C.text,
                border: `1px solid ${C.border}`, borderRadius: "4px",
                fontSize: "12px", padding: "6px", resize: "none",
                fontFamily: "inherit", lineHeight: "1.4",
                minHeight: "128px", height: "128px", maxHeight: "none", overflow: "auto",
            });
            negBox.onchange = _syncS;
            negBox.oninput = () => { _syncS(); requestAnimationFrame(() => reflowNode(node)); };
            negBox.addEventListener("wheel", (e) => { e.stopPropagation(); });
            negSec._body.appendChild(negBox);
            root.appendChild(negSec);
            node._weNegBox = negBox;
            node._weNegSection = negSec;
            negSec.style.display = "";

            // -- 5. LORAS section --
            const loraSec = makeSection("LORAS", node, "loras", { collapsed: !!node._weCollapsedSections?.loras });
            node._weSections.loras = loraSec;
            loraSec._onLockChanged = (locked) => {
                node._weSectionLocks.loras = !!locked;
                _syncS();
            };
            if (loraSec._setLocked) loraSec._setLocked(false);

            // Stack A card
            const loraACard = createLoraStackContainer("LoRA Stack A",
                () => {
                    const d = node._weExtracted;
                    const lorasA = d?.loras_a || [];
                    const hasBoth = lorasA.length > 0 && (d?.loras_b || []).length > 0;
                    for (const lora of lorasA) {
                        const name = String(lora?.name || "");
                        if (!name) continue;
                        const stateKey = hasBoth ? `a:${name}` : name;
                        const altAKey = `a:${name}`;
                        const altNameKey = name;

                        if (!node._weLoraState[stateKey]) {
                            node._weLoraState[stateKey] = { active: lora.active !== false, model_strength: 1.0, clip_strength: 1.0 };
                        }
                        const orig = _getLoraOriginalStrength(node, stateKey, lora);
                        node._weLoraState[stateKey].model_strength = orig.model_strength;
                        node._weLoraState[stateKey].clip_strength = orig.clip_strength;

                        // Keep both A-key variants in sync so reset works across
                        // transitions between single-stack and dual-stack modes.
                        for (const key of [altAKey, altNameKey]) {
                            if (!key || key === stateKey) continue;
                            if (!node._weLoraState[key]) continue;
                            node._weLoraState[key].model_strength = orig.model_strength;
                            node._weLoraState[key].clip_strength = orig.clip_strength;
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
            loraACard.updateHeight?.(node);

            // Stack B card
            const loraBCard = createLoraStackContainer("LoRA Stack B",
                () => {
                    const d = node._weExtracted;
                    for (const lora of (d?.loras_b || [])) {
                        const k = `b:${lora.name}`;
                        if (!node._weLoraState[k]) {
                            node._weLoraState[k] = { active: lora.active !== false, model_strength: 1.0, clip_strength: 1.0 };
                        }
                        const orig = _getLoraOriginalStrength(node, k, lora);
                        node._weLoraState[k].model_strength = orig.model_strength;
                        node._weLoraState[k].clip_strength = orig.clip_strength;
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
            loraBCard.style.display = "none";
            loraSec._body.appendChild(loraBCard);
            node._weLoraBContainer = loraBCard._tagsContainer;
            node._weLoraB = loraBCard;
            loraBCard.updateHeight?.(node);

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
                const calcLoraHeight = (card) => {
                    const tags = card?._tagsContainer;
                    if (!tags) return;
                    const children = Array.from(tags.children || []).filter(el => el.style?.display !== "none");
                    const actualWidth = Math.max(220, Number(width || node.size?.[0] || 0) - 32);
                    const tagWidth = 204;
                    const tagsPerRow = Math.max(1, Math.floor(actualWidth / tagWidth));
                    const rows = Math.max(1, Math.ceil(Math.max(1, children.length) / tagsPerRow));
                    const rowHeight = 28;
                    const topChrome = 8;
                    const target = topChrome + rows * rowHeight;
                    tags.style.minHeight = `${target}px`;
                    tags.style.height = `${target}px`;
                };

                calcLoraHeight(node._weLoraACard);
                calcLoraHeight(node._weLoraB);
                const h = Math.max(Math.ceil(root.scrollHeight || 0), _MIN_H);
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
                const ghosts = _getSelectedSlotInputGhosts(node);

                if (node._wePosBox) {
                    node._wePosBox.readOnly = ghosts.positive;
                    node._wePosBox.style.opacity = ghosts.positive ? "0.5" : "1";
                    node._wePosBox.style.pointerEvents = "auto";
                }
                if (node._weNegBox) {
                    node._weNegBox.readOnly = ghosts.negative;
                    node._weNegBox.style.opacity = ghosts.negative ? "0.5" : "1";
                    node._weNegBox.style.pointerEvents = "auto";
                }
            }
            node._updatePromptGhosting = _updatePromptGhosting;

            function _readConnectedInputValue(inputName) {
                const conn = node.inputs?.find(i => i.name === inputName);
                if (!conn || conn.link == null) return { linked: false, value: null };

                const linkInfo = node.graph?.links?.[conn.link];
                if (!linkInfo) return { linked: true, value: null };

                const srcNode = node.graph?.getNodeById?.(linkInfo.origin_id);
                if (!srcNode) return { linked: true, value: null };

                let value = srcNode.getOutputData?.(linkInfo.origin_slot);
                if (value === undefined && Array.isArray(srcNode.widgets)) {
                    const widget = srcNode.widgets.find(w => {
                        const name = String(w?.name || "").toLowerCase();
                        return name === "value" || name === "seed" || name === "int" || name === "number";
                    });
                    if (widget) value = widget.value;
                }

                return { linked: true, value };
            }

            function _updateSeedGhosting() {
                const rows = node._weSamplerRows;
                if (!rows) return;
                const ghosts = _getSelectedSlotInputGhosts(node);

                const applySeedState = (inputName, row, ghosted = false) => {
                    if (!row?._inp) return;
                    const { linked, value } = _readConnectedInputValue(inputName);
                    const isGhosted = !!ghosted || !!linked;

                    row._inp.readOnly = isGhosted;
                    row._inp.style.opacity = isGhosted ? "0.5" : "1";
                    row._inp.style.pointerEvents = "auto";

                    if (linked && value !== undefined && value !== null) {
                        const n = Number(value);
                        if (Number.isFinite(n)) row._inp.value = String(Math.trunc(n));
                    }
                };

                applySeedState("seed", rows.seed_a, ghosts.seed);
                applySeedState("seed_b", rows.seed_b);
            }
            node._updateSeedGhosting = _updateSeedGhosting;

            // Update ghosting and LoRA inputs when connections change
            const origConnInput = node.onConnectionsChange;
            node.onConnectionsChange = function () {
                if (origConnInput) origConnInput.apply(this, arguments);

                _normalizeRecipePortsNow();

                // Ignore persistence writes during hydration to avoid clobbering
                // restored values with defaults (e.g. SDXL).
                if (node._weHydrating) {
                    _updatePromptGhosting();
                    _updateSeedGhosting();
                    return;
                }

                _updatePromptGhosting();
                _updateSeedGhosting();

                // Clear input LoRAs for disconnected stacks and re-merge
                const loraAConn = _findInput(node, "lora_stack", "lora_stack_a");
                const loraBConn = node.inputs?.find(i => i.name === "lora_stack_b");
                let changed = false;
                const oldMergedLorasA = node._weExtracted?.loras_a || [];
                const oldMergedLorasB = node._weExtracted?.loras_b || [];
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
                    _captureLoraOriginalStrengths(node, node._weExtracted.loras_a, node._weExtracted.loras_b);
                    _resetChangedLoraState(node, oldMergedLorasA, oldMergedLorasB, node._weExtracted.loras_a, node._weExtracted.loras_b);
                    updateLoras(node);
                    node.setDirtyCanvas(true, true);
                }

                // Persist after connection-driven LoRA cleanup so execute()
                // reads current stacks on the same run.
                syncHidden(node);
            };
            // Apply initial ghosting state
            _updatePromptGhosting();
            _updateSeedGhosting();
            // Persist an initial baseline snapshot so tab switching can
            // restore even before the first execute/save.
            _syncS();

            // -- Pre-generation UI update (via send_sync from Python) --
            // Fires before model loading / sampling so UI feels instant.
            api.addEventListener("workflow-generator-pre-update", (event) => {
                if (String(event.detail?.node_id) !== String(node.id)) return;
                const info = event.detail?.info?.extracted;
                if (!info) return;

                if (node._weUseSlotProfiles && info._slot_profiles && typeof info._slot_profiles === "object") {
                    node._weSlotProfiles = _mergeSlotProfiles(node._weSlotProfiles, info._slot_profiles);
                    _hydrateExtractedFromSelectedSlotProfile(node);
                }

                // Always mirror connected prompt inputs into textareas,
                // even when extractor-connected mode skips full UI refresh.
                const posInputConn = node.inputs?.find(i => i.name === "pos_prompt");
                const negInputConn = node.inputs?.find(i => i.name === "neg_prompt");
                if (posInputConn?.link != null && info.positive_prompt != null && node._wePosBox) {
                    node._wePosBox.value = info.positive_prompt;
                }
                if (negInputConn?.link != null && info.negative_prompt != null && node._weNegBox) {
                    node._weNegBox.value = info.negative_prompt;
                }
                const seedAInputConn = _findInput(node, "seed", "seed_a");
                const seedBInputConn = node.inputs?.find(i => i.name === "seed_b");
                const samplerInfo = info.sampler || {};
                if (seedAInputConn?.link != null && samplerInfo.seed_a != null && node._weSamplerRows?.seed_a?._inp) {
                    node._weSamplerRows.seed_a._inp.value = String(Math.trunc(Number(samplerInfo.seed_a) || 0));
                }
                if (seedBInputConn?.link != null && samplerInfo.seed_b != null && node._weSamplerRows?.seed_b?._inp) {
                    node._weSamplerRows.seed_b._inp.value = String(Math.trunc(Number(samplerInfo.seed_b) || 0));
                }
                if ((posInputConn?.link != null || negInputConn?.link != null) && node._updatePromptGhosting) {
                    node._updatePromptGhosting();
                    syncHidden(node);
                }
                if ((seedAInputConn?.link != null || seedBInputConn?.link != null) && node._updateSeedGhosting) {
                    node._updateSeedGhosting();
                    syncHidden(node);
                }

                const oldWl = node._weWorkflowLoras || { a: [], b: [] };
                const oldIl = node._weInputLoras || { a: [], b: [] };
                const isExtractorConnected = (!node._weUseSlotProfiles) && (node._weIsConnectedWorkflowDataExtractor?.() === true);
                const oldSig = isExtractorConnected
                    ? `${_loraListSignature(node._weExtracted?.loras_a || [])}||${_loraListSignature(node._weExtracted?.loras_b || [])}`
                    : _loraStacksSignature(oldWl, oldIl);

                node._weWorkflowLoras = isExtractorConnected
                    ? {
                        // Keep extractor-connected baseline aligned with the
                        // effective stack Python executed (UI truth), so queue
                        // runs never reintroduce stale order/state.
                        a: [...(info.workflow_loras_a || info.loras_a || node._weExtracted?.loras_a || oldWl.a || [])],
                        b: [...(info.workflow_loras_b || info.loras_b || node._weExtracted?.loras_b || oldWl.b || [])],
                    }
                    : {
                        a: [...(info.workflow_loras_a || info.loras_a || oldWl.a || [])],
                        b: [...(info.workflow_loras_b || info.loras_b || oldWl.b || [])],
                    };
                node._weInputLoras = {
                    a: [...(info.input_loras_a || oldIl.a || [])],
                    b: [...(info.input_loras_b || oldIl.b || [])],
                };

                const mergedLorasA = _mergeLoraLists(node._weWorkflowLoras.a, node._weInputLoras.a);
                const mergedLorasB = _mergeLoraLists(node._weWorkflowLoras.b, node._weInputLoras.b);

                const newSig = isExtractorConnected
                    ? `${_loraListSignature(info.loras_a || mergedLorasA)}||${_loraListSignature(info.loras_b || mergedLorasB)}`
                    : _loraStacksSignature(node._weWorkflowLoras, node._weInputLoras);
                const lorasLocked = !!(node._weSectionLocks?.loras);
                if (!lorasLocked && oldSig !== newSig) {
                    _captureLoraOriginalStrengths(node, mergedLorasA, mergedLorasB);
                    _resetChangedLoraState(
                        node,
                        node._weExtracted?.loras_a || [],
                        node._weExtracted?.loras_b || [],
                        mergedLorasA,
                        mergedLorasB
                    );
                }
                if (isExtractorConnected) {
                    // In extractor-connected mode, keep manual section behavior,
                    // but always reflect latest merged LoRA stacks from execution.
                    node._weExtracted = {
                        ...(node._weExtracted || {}),
                        loras_a: [...(info.loras_a || mergedLorasA)],
                        loras_b: [...(info.loras_b || mergedLorasB)],
                        lora_availability: info.lora_availability || node._weExtracted?.lora_availability || {},
                    };
                    node._wePopulated = true;
                    updateLoras(node);
                    syncHidden(node);
                    node.setDirtyCanvas(true, true);
                    app.graph.setDirtyCanvas(true, true);
                    node._preUpdateApplied = true;
                    return;
                }

                // Clear any previous error banner
                _showError(node, null);

                // Always refresh extracted state from latest execution so
                // recipe_data inputs drive UI every queue run.
                node._weExtracted = {
                    ...(node._weExtracted || {}),
                    ...info,
                    loras_a: mergedLorasA,
                    loras_b: mergedLorasB,
                };
                _hydrateExtractedFromSelectedSlotProfile(node);
                node._wePopulated = true;

                if (info.model_family && node._weFamilySel) {
                    const fam = info.model_family;
                    if (![...node._weFamilySel.options].some(o => o.value === fam)) {
                        const o = document.createElement("option");
                        o.value = fam; o.textContent = info.model_family_label || fam;
                        node._weFamilySel.appendChild(o);
                    }
                }

                const uiReady = updateUI(node);
                if (uiReady && typeof uiReady.then === "function") {
                    uiReady.finally(() => {
                        node.properties = node.properties || {};
                        delete node.properties.we_extracted_cache;
                    });
                } else {
                    node.properties = node.properties || {};
                    delete node.properties.we_extracted_cache;
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

            // Always mirror connected prompt inputs into textareas so users
            // can see generated prompts immediately.
            const info = wfInfo?.extracted;
            const posConn = this.inputs?.find(i => i.name === "pos_prompt");
            const negConn = this.inputs?.find(i => i.name === "neg_prompt");
            if (info) {
                if (this._weUseSlotProfiles && info._slot_profiles && typeof info._slot_profiles === "object") {
                    this._weSlotProfiles = _mergeSlotProfiles(this._weSlotProfiles, info._slot_profiles);
                    _hydrateExtractedFromSelectedSlotProfile(this);
                }
                if (posConn?.link != null && info.positive_prompt != null && this._wePosBox) {
                    this._wePosBox.value = info.positive_prompt;
                }
                if (negConn?.link != null && info.negative_prompt != null && this._weNegBox) {
                    this._weNegBox.value = info.negative_prompt;
                }
                const seedAConn = _findInput(this, "seed", "seed_a");
                const seedBConn = this.inputs?.find(i => i.name === "seed_b");
                const samplerInfo = info.sampler || {};
                if (seedAConn?.link != null && samplerInfo.seed_a != null && this._weSamplerRows?.seed_a?._inp) {
                    this._weSamplerRows.seed_a._inp.value = String(Math.trunc(Number(samplerInfo.seed_a) || 0));
                }
                if (seedBConn?.link != null && samplerInfo.seed_b != null && this._weSamplerRows?.seed_b?._inp) {
                    this._weSamplerRows.seed_b._inp.value = String(Math.trunc(Number(samplerInfo.seed_b) || 0));
                }
                if ((posConn?.link != null || negConn?.link != null) && this._updatePromptGhosting) {
                    this._updatePromptGhosting();
                    syncHidden(this);
                }
                if ((seedAConn?.link != null || seedBConn?.link != null) && this._updateSeedGhosting) {
                    this._updateSeedGhosting();
                    syncHidden(this);
                }
            }

            // UI population is handled by pre-update listener (fires before
            // generation).  Only run here as a fallback if send_sync failed.
            const isExtractorConnected = (!this._weUseSlotProfiles) && (this._weIsConnectedWorkflowDataExtractor?.() === true);
            if (info && !this._preUpdateApplied) {
                const oldWl = this._weWorkflowLoras || { a: [], b: [] };
                const oldIl = this._weInputLoras || { a: [], b: [] };
                const oldSig = isExtractorConnected
                    ? `${_loraListSignature(this._weExtracted?.loras_a || [])}||${_loraListSignature(this._weExtracted?.loras_b || [])}`
                    : _loraStacksSignature(oldWl, oldIl);

                this._weWorkflowLoras = isExtractorConnected
                    ? {
                        a: [...(info.workflow_loras_a || info.loras_a || this._weExtracted?.loras_a || oldWl.a || [])],
                        b: [...(info.workflow_loras_b || info.loras_b || this._weExtracted?.loras_b || oldWl.b || [])],
                    }
                    : {
                        a: [...(info.workflow_loras_a || info.loras_a || oldWl.a || [])],
                        b: [...(info.workflow_loras_b || info.loras_b || oldWl.b || [])],
                    };
                this._weInputLoras = {
                    a: [...(info.input_loras_a || oldIl.a || [])],
                    b: [...(info.input_loras_b || oldIl.b || [])],
                };

                const mergedLorasA = _mergeLoraLists(this._weWorkflowLoras.a, this._weInputLoras.a);
                const mergedLorasB = _mergeLoraLists(this._weWorkflowLoras.b, this._weInputLoras.b);

                const newSig = isExtractorConnected
                    ? `${_loraListSignature(info.loras_a || mergedLorasA)}||${_loraListSignature(info.loras_b || mergedLorasB)}`
                    : _loraStacksSignature(this._weWorkflowLoras, this._weInputLoras);
                const lorasLocked = !!(this._weSectionLocks?.loras);
                if (!lorasLocked && oldSig !== newSig) {
                    _captureLoraOriginalStrengths(this, mergedLorasA, mergedLorasB);
                    _resetChangedLoraState(
                        this,
                        this._weExtracted?.loras_a || [],
                        this._weExtracted?.loras_b || [],
                        mergedLorasA,
                        mergedLorasB
                    );
                }

                if (isExtractorConnected) {
                    this._weExtracted = {
                        ...(this._weExtracted || {}),
                        loras_a: [...(info.loras_a || mergedLorasA)],
                        loras_b: [...(info.loras_b || mergedLorasB)],
                        lora_availability: info.lora_availability || this._weExtracted?.lora_availability || {},
                    };
                    this._wePopulated = true;
                    updateLoras(this);
                    syncHidden(this);
                    this.setDirtyCanvas(true, true);
                    app.graph.setDirtyCanvas(true, true);
                    this._preUpdateApplied = false;
                    return;
                }

                this._weExtracted = {
                    ...(this._weExtracted || {}),
                    ...info,
                    loras_a: mergedLorasA,
                    loras_b: mergedLorasB,
                };
                _hydrateExtractedFromSelectedSlotProfile(this);
                this._wePopulated = true;

                if (info.model_family && this._weFamilySel) {
                    const fam = info.model_family;
                    if (![...this._weFamilySel.options].some(o => o.value === fam)) {
                        const o = document.createElement("option");
                        o.value = fam; o.textContent = info.model_family_label || fam;
                        this._weFamilySel.appendChild(o);
                    }
                }

                const uiReady = updateUI(this);
                if (uiReady && typeof uiReady.then === "function") {
                    uiReady.finally(() => {
                        this.properties = this.properties || {};
                        delete this.properties.we_extracted_cache;
                    });
                } else {
                    this.properties = this.properties || {};
                    delete this.properties.we_extracted_cache;
                }
            }
            // Reset flag for next execution
            this._preUpdateApplied = false;
        };

        // -- onConfigure (graph load / paste / tab return) --
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            origConfigure?.apply(this, arguments);
            const node = this;
            node._configuredFromWorkflow = true;
            node._weHydrating = true;
            const _finishHydration = () => {
                node._weHydrating = false;
                if (node._updatePromptGhosting) node._updatePromptGhosting();
                if (node._updateSeedGhosting) node._updateSeedGhosting();
                syncHidden(node);
            };

            const getInfoWidgetVal = (name, fallback = undefined) => {
                try {
                    if (!Array.isArray(info?.widgets_values) || !Array.isArray(node.widgets)) {
                        return fallback;
                    }
                    const idx = node.widgets.findIndex(w => w?.name === name);
                    if (idx < 0) return fallback;
                    const v = info.widgets_values[idx];
                    return (v !== undefined && v !== null) ? v : fallback;
                } catch {
                    return fallback;
                }
            };

            // ── Migration: remove stale inputs/outputs from old workflows ──
            // LiteGraph restores slots from saved JSON; if INPUT_TYPES or
            // RETURN_TYPES changed between versions, phantom slots persist.
            if (node._weUseSlotProfiles == null) {
                const cls = String(node.comfyClass || node.type || "");
                node._weUseSlotProfiles = (cls === "RecipeBuilder") || (cls === "WorkflowBuilderMulti");
            }

            const VALID_INPUTS = new Set([
                "recipe_data",
                "builder_data",
            ]);
            const _normalizeInputNameForVariant = (name) => {
                let key = _canonicalInputName(name);
                if (key === "seed_a") key = "seed";
                if (key === "lora_stack_a") key = "lora_stack";
                return key;
            };
            // Normalize legacy prompt input slot names on load.
            if (node.inputs) {
                for (const inp of node.inputs) {
                    if (!inp || !inp.name) continue;
                    inp.name = _normalizeInputNameForVariant(inp.name);
                }

                // Deduplicate legacy duplicate inputs by name. Prefer keeping
                // the linked slot when duplicates exist.
                const keepIndexByName = new Map();
                for (let i = 0; i < node.inputs.length; i++) {
                    const inp = node.inputs[i];
                    if (!inp || !inp.name) continue;
                    const key = _normalizeInputNameForVariant(inp.name);
                    if (!VALID_INPUTS.has(key)) continue;
                    if (!keepIndexByName.has(key)) {
                        keepIndexByName.set(key, i);
                        continue;
                    }
                    const keepIdx = keepIndexByName.get(key);
                    const keepInp = node.inputs[keepIdx];
                    const curLinked = inp?.link != null;
                    const keepLinked = keepInp?.link != null;
                    if (curLinked && !keepLinked) {
                        keepIndexByName.set(key, i);
                    }
                }
                for (let i = node.inputs.length - 1; i >= 0; i--) {
                    const inp = node.inputs[i];
                    if (!inp || !inp.name) continue;
                    const key = _normalizeInputNameForVariant(inp.name);
                    if (VALID_INPUTS.has(key) && keepIndexByName.get(key) !== i) {
                        node.removeInput(i);
                    }
                }
            }
            if (node.inputs) {
                for (let i = node.inputs.length - 1; i >= 0; i--) {
                    const key = _normalizeInputNameForVariant(node.inputs[i].name);
                    if (!VALID_INPUTS.has(key)) {
                        node.removeInput(i);
                    }
                }

                // Ensure newly introduced inputs exist on restored nodes.
                // Older saved workflows can miss optional inputs added later.
                const hasInput = (name) => node.inputs.some((inp) => _normalizeInputNameForVariant(inp?.name || "") === name);
                if (!hasInput("recipe_data")) {
                    node.addInput("recipe_data", "RECIPE_DATA");
                }
                if (node._weUseSlotProfiles && !hasInput("builder_data")) {
                    node.addInput("builder_data", "BUILDER_DATA");
                }
            }
            const VALID_OUTPUTS = [
                { name: "recipe_data", type: "RECIPE_DATA" },
            ];
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
            const getWidgetVal = (name, fallback = undefined) => {
                const w = node.widgets?.find(x => x.name === name);
                return w?.value ?? fallback;
            };
            const savedUiState = props.we_ui_state ?? "";
            const savedOvFromInfo = getInfoWidgetVal("override_data", "{}");
            const savedLsFromInfo = getInfoWidgetVal("lora_state", "{}");
            const savedOv = (savedUiState && savedUiState !== "{}")
                ? savedUiState
                : (props.we_override_data ?? getWidgetVal("override_data", savedOvFromInfo));
            const savedLs = props.we_lora_state ?? getWidgetVal("lora_state", savedLsFromInfo);
            const savedWl = props.we_workflow_loras;
            const savedIl = props.we_input_loras;

            let savedWorkflowLoras = { a: [], b: [] };
            let savedInputLoras = { a: [], b: [] };
            try { savedWorkflowLoras = JSON.parse(savedWl || '{"a":[],"b":[]}'); } catch { savedWorkflowLoras = { a: [], b: [] }; }
            try { savedInputLoras = JSON.parse(savedIl || '{"a":[],"b":[]}'); } catch { savedInputLoras = { a: [], b: [] }; }
            const savedLsHasData = !!(savedLs && savedLs !== "{}" && savedLs !== "null");
            const savedLoraListsCount =
                (Array.isArray(savedWorkflowLoras.a) ? savedWorkflowLoras.a.length : 0) +
                (Array.isArray(savedWorkflowLoras.b) ? savedWorkflowLoras.b.length : 0) +
                (Array.isArray(savedInputLoras.a) ? savedInputLoras.a.length : 0) +
                (Array.isArray(savedInputLoras.b) ? savedInputLoras.b.length : 0);

            let ovObj = {};
            try { ovObj = JSON.parse(savedOv || "{}"); } catch { ovObj = {}; }
            // Prompt fallback: old saves may have prompts only in we_workflow_prompts.
            let savedWp = null;
            try { savedWp = JSON.parse(props.we_workflow_prompts || "null"); } catch { savedWp = null; }
            if (savedWp && typeof savedWp === "object") {
                if ((!ovObj.positive_prompt || ovObj.positive_prompt === "") && savedWp.positive) {
                    ovObj.positive_prompt = savedWp.positive;
                }
                if ((!ovObj.negative_prompt || ovObj.negative_prompt === "") && savedWp.negative) {
                    ovObj.negative_prompt = savedWp.negative;
                }
            }
            const hasAuthoritativeOverrides = !!(ovObj && Object.keys(ovObj).length > 0);
            const hasSavedLoraContext = savedLsHasData || savedLoraListsCount > 0;

            const buildExtractedFromOverrides = (ov, fallback = {}) => {
                const fam = ov?._family || fallback.model_family || "sdxl";
                const isVideo = ["wan_video_t2v", "wan_video_i2v"].includes(fam);
                const fbSampler = fallback.sampler || {};
                const fbRes = fallback.resolution || {};
                const lorasA = Array.isArray(ov?.loras_a)
                    ? ov.loras_a
                    : (Array.isArray(fallback.loras_a) ? fallback.loras_a : []);
                const lorasB = Array.isArray(ov?.loras_b)
                    ? ov.loras_b
                    : (Array.isArray(fallback.loras_b) ? fallback.loras_b : []);
                const mergedAvailability = {};
                for (const l of [...lorasA, ...lorasB]) {
                    const n = String(l?.name || "").trim();
                    if (!n) continue;
                    if (l?.available === false) mergedAvailability[n] = false;
                    else if (!(n in mergedAvailability)) mergedAvailability[n] = true;
                }
                const loraAvailability =
                    (ov?._lora_availability && typeof ov._lora_availability === "object")
                        ? ov._lora_availability
                        : ((fallback.lora_availability && typeof fallback.lora_availability === "object")
                            ? fallback.lora_availability
                            : mergedAvailability);
                return {
                    positive_prompt: (ov?.positive_prompt != null) ? ov.positive_prompt : (fallback.positive_prompt || ""),
                    negative_prompt: (ov?.negative_prompt != null) ? ov.negative_prompt : (fallback.negative_prompt || ""),
                    model_a: (ov?.model_a != null) ? ov.model_a : (fallback.model_a || ""),
                    model_b: (ov?.model_b != null) ? ov.model_b : (fallback.model_b || ""),
                    model_family: fam,
                    model_family_label: fallback.model_family_label || "",
                    vae: {
                        name: (ov?.vae != null) ? ov.vae : (fallback.vae?.name || ""),
                        source: "manual",
                    },
                    clip: {
                        names: (Array.isArray(ov?.clip_names) ? ov.clip_names : (fallback.clip?.names || [])),
                        type: fallback.clip?.type || "",
                        source: "manual",
                    },
                    sampler: {
                        steps_a: (ov?.steps_a != null) ? ov.steps_a : (fbSampler.steps_a ?? 20),
                        steps_b: (ov?.steps_b != null) ? ov.steps_b : (fbSampler.steps_b ?? null),
                        cfg: (ov?.cfg != null) ? ov.cfg : (fbSampler.cfg ?? 5.0),
                        denoise: (ov?.denoise != null) ? ov.denoise : (fbSampler.denoise ?? 1.0),
                        seed_a: (ov?.seed_a != null) ? ov.seed_a : (fbSampler.seed_a ?? 0),
                        seed_b: (ov?.seed_b != null) ? ov.seed_b : (fbSampler.seed_b ?? null),
                        sampler_name: (ov?.sampler_name != null) ? ov.sampler_name : (fbSampler.sampler_name || "euler"),
                        scheduler: (ov?.scheduler != null) ? ov.scheduler : (fbSampler.scheduler || "simple"),
                    },
                    resolution: {
                        width: (ov?.width != null) ? ov.width : (fbRes.width ?? 768),
                        height: (ov?.height != null) ? ov.height : (fbRes.height ?? 1280),
                        batch_size: (ov?.batch_size != null) ? ov.batch_size : (fbRes.batch_size ?? 1),
                        length: (ov?.length != null) ? ov.length : (isVideo ? (fbRes.length ?? 81) : undefined),
                    },
                    loras_a: lorasA,
                    loras_b: lorasB,
                    lora_availability: loraAvailability,
                    is_video: isVideo,
                };
            };
            if (hasAuthoritativeOverrides || hasSavedLoraContext) {
                // UI state is the sole source of truth on restore.
                // Intentionally ignore we_extracted_cache because it may be stale.
                const mergedLorasA = _mergeLoraLists(savedWorkflowLoras?.a || [], savedInputLoras?.a || [], ovObj?.loras_a || []);
                const mergedLorasB = _mergeLoraLists(savedWorkflowLoras?.b || [], savedInputLoras?.b || [], ovObj?.loras_b || []);
                node._weExtracted = buildExtractedFromOverrides(ovObj, {
                    loras_a: mergedLorasA,
                    loras_b: mergedLorasB,
                });
                _captureLoraOriginalStrengths(node, mergedLorasA, mergedLorasB);
                node._wePopulated = true;
                node._weLoraState = {};
                node._weOverrides = {};
                node._weWorkflowLoras = savedWorkflowLoras;
                node._weInputLoras = savedInputLoras;
                try { node._weWorkflowPrompts = JSON.parse(props.we_workflow_prompts || 'null'); } catch { node._weWorkflowPrompts = null; }
                const uiReady = updateUI(node);
                if (uiReady && typeof uiReady.then === "function") {
                    uiReady.then(() => {
                        applyOverrides(node, savedOv, savedLs);
                        syncHidden(node);
                        if (node._updatePromptGhosting) node._updatePromptGhosting();
                        if (node._updateSeedGhosting) node._updateSeedGhosting();
                        node.setDirtyCanvas(true, true);
                    }).finally(() => _finishHydration());
                } else {
                    applyOverrides(node, savedOv, savedLs);
                    syncHidden(node);
                    if (node._updatePromptGhosting) node._updatePromptGhosting();
                    if (node._updateSeedGhosting) node._updateSeedGhosting();
                    node.setDirtyCanvas(true, true);
                    _finishHydration();
                }
            } else {
                // No usable UI-state overrides available: keep default UI.
                if (node._updatePromptGhosting) node._updatePromptGhosting();
                if (node._updateSeedGhosting) node._updateSeedGhosting();
                node.setDirtyCanvas(true, true);
                _finishHydration();
            }

            // Execute-only flow: no manual Update Recipe button refresh.
        };
    },
});
