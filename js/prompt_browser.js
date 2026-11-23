import { createPromptBrowserEditPanel, getPromptTypeChoices } from "./prompt_browser_edit.js";
import { PM_UI_PALETTE } from "./ui_palette.js";
import { formDialog } from "./pm_dialogs.js";
import { api } from "../../scripts/api.js";
import { button, ICONS } from "./fbui/controls.js";
import { el } from "./fbui/dom.js";
import { confirm, openMenu, openModal, prompt } from "./fbui/modal.js";
import { ensureKitCss, FONT, RADIUS, RADIUS_CARD, ROW_HEIGHT, T, Z_MENU, Z_MODAL } from "./fbui/theme.js";
import {
    buildSavePromptRequestBodyForEndpoint,
    getCategoryPromptEntriesForEndpoint,
    getCategoryPromptEntryForEndpoint,
    isHiddenPromptEntryKey,
    SYSTEM_PROMPTS_ENDPOINT_PREFIX,
} from "./prompt_store_adapters.js";

let app = null;
let UI = PM_UI_PALETTE;

let DEFAULT_THUMBNAIL = "";

// The browser's own stylesheet on the kit tokens (injected once).
const BROWSER_CSS_ID = "pm-browser-css-v1";
function ensurePromptBrowserCss() {
    ensureKitCss();
    if (document.getElementById(BROWSER_CSS_ID)) return;
    const style = document.createElement("style");
    style.id = BROWSER_CSS_ID;
    style.textContent = `
@keyframes pm-spin{to{transform:rotate(360deg)}}
.pm-spin{flex:none;width:16px;height:16px;box-sizing:border-box;border:2.5px solid ${T.brand};border-top-color:transparent;border-radius:50%;animation:pm-spin 1s linear infinite}
.pm-progress{position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);z-index:${Z_MENU};display:flex;align-items:center;gap:10px;min-width:320px;max-width:min(560px,90vw);padding:12px 12px 12px 16px;box-sizing:border-box;border:1px solid ${T.plateEdge};border-radius:${RADIUS_CARD}px;background:${T.panel};box-shadow:0 12px 40px rgba(0,0,0,.6);color:${T.inkSoft};font:13px ${FONT}}
.pm-progress .text{flex:1;min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-variant-numeric:tabular-nums}
.pm-browser{position:fixed;top:50%;transform:translateY(-50%);z-index:${Z_MODAL};max-height:95vh;display:flex;flex-direction:column;padding:16px;border:1px solid ${T.plateEdge};border-radius:${RADIUS_CARD}px;background:${T.panel};box-shadow:0 12px 40px rgba(0,0,0,.6);color:${T.inkSoft};font:12px ${FONT};user-select:none;-webkit-user-select:none}
.pm-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid ${T.rule}}
.pm-head .title{color:${T.ink};font:600 15px ${FONT}}
.pm-controls{display:flex;align-items:center;gap:6px;margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid ${T.rule}}
.pm-controls .fbui-btn{white-space:nowrap}
.pm-search{flex:1;min-width:0;display:flex;align-items:center;gap:4px;height:${ROW_HEIGHT}px;padding:0 3px 0 8px;border-radius:${RADIUS}px;background:${T.fill}}
.pm-search .lead{display:grid;color:${T.faint}}
.pm-search .fbui-text{flex:1;min-width:0;height:100%;padding:0;border:none;background:transparent;user-select:text;-webkit-user-select:text}
.pm-viewgroup{display:flex;gap:2px}
.pm-categories{display:flex;gap:8px;min-height:0;margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid ${T.rule}}
.pm-category-controls{display:flex;gap:6px;flex:none;align-items:center}
.pm-tabs{display:flex;gap:4px;flex:1;min-width:0;align-items:center;overflow-x:auto;overflow-y:hidden;scrollbar-width:thin;scrollbar-color:${T.fill} transparent}
.pm-tab{flex:none;height:${ROW_HEIGHT}px;padding:0 12px;border:1px solid transparent;border-radius:${RADIUS}px;background:${T.fill};color:${T.label};font:500 12px ${FONT};white-space:nowrap;cursor:pointer}
.pm-tab:hover{background:${T.fillHover};color:${T.ink}}
.pm-tab.on{background:${T.brandSoftHover};border-color:${T.brand};color:${T.ink}}
.pm-tab.nsfw:not(.on){border-color:${T.danger}}
.pm-divider{flex:none;width:1px;align-self:stretch;margin:0 2px;background:${T.rule}}
.pm-grid{display:grid;align-content:start;padding:4px 0}
.pm-empty{grid-column:1/-1;padding:40px;color:${T.faint};font-style:italic;text-align:center}
.pm-card{position:relative;display:flex;flex-direction:column;align-items:center;padding:6px;border:2px solid transparent;border-radius:${RADIUS_CARD - 2}px;background:${T.plate};cursor:pointer;transition:background .12s,border-color .12s}
.pm-card:hover{background:${T.fill};border-color:${T.plateEdge}}
.pm-card.on{background:${T.brandSoftHover};border-color:${T.brand}}
.pm-card.blank{justify-content:center;border-style:dashed;border-color:${T.plateEdge}}
.pm-thumb{flex:none;border-radius:${RADIUS}px;background:${T.card} center/cover no-repeat}
.pm-thumb.blank{display:flex;align-items:center;justify-content:center;box-sizing:border-box;border:1px dashed ${T.plateEdge};color:${T.faint};line-height:1}
.pm-name{width:100%;margin-top:6px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;text-align:center;color:${T.inkSoft};font-size:12px}
.pm-badges{position:absolute;top:4px;right:4px;z-index:1;display:flex;flex-direction:column;align-items:flex-end;gap:2px}
.pm-badge-nsfw{padding:1px 4px;border-radius:3px;background:${T.danger};color:${T.ink};font:700 8px/1.2 ${FONT}}
.pm-kind{display:grid;place-items:center;width:14px;height:14px;border-radius:50%;color:${T.ink};font:700 9px/1 ${FONT}}
.pm-badges.corner{top:auto;right:-2px;bottom:-2px}
.pm-badges.corner .pm-kind{width:12px;height:12px;font-size:8px}
.pm-list{display:flex;flex-direction:column;position:relative}
.pm-row{display:grid;align-items:center;padding:0 8px;border-radius:${RADIUS}px;cursor:pointer;transition:background .1s}
.pm-row:hover{background:${T.fill}}
.pm-row.on{background:${T.brandSoftHover};box-shadow:inset 0 0 0 1px ${T.brand}}
.pm-row-name{overflow:hidden;white-space:nowrap;text-overflow:ellipsis;color:${T.ink};font-size:13px}
.thumbnail-grid-container::-webkit-scrollbar{display:none}
.category-scroll-bar::-webkit-scrollbar{height:6px}
.category-scroll-bar::-webkit-scrollbar-track,.pm-type-rail-list::-webkit-scrollbar-track,.category-scroll-bar::-webkit-scrollbar-corner{background:transparent}
.category-scroll-bar::-webkit-scrollbar-thumb,.pm-type-rail-list::-webkit-scrollbar-thumb{background:${T.fill};border-radius:3px}
.category-scroll-bar::-webkit-scrollbar-thumb:hover,.pm-type-rail-list::-webkit-scrollbar-thumb:hover{background:${T.fillHover}}
.pm-type-rail-list::-webkit-scrollbar{width:6px}
`;
    document.head.append(style);
}
let loadPrompts = async () => ({});
let ensureThumbnailRenderSelection = async () => null;
let resolveThumbnailFallbackBase = async () => null;
let generateThumbnailForPrompt = async () => null;
let generateThumbnailWorkflowFromWorkflowData = async () => null;
let buildRendererFallbackWorkflowDataWithBase = async () => null;
let showThumbnailRenderPicker = async () => null;
let saveThumbnailRenderSelection = () => {};
let _thumbnailLeafName = (path) => String(path || "");
let getThumbnailRenderState = null;
const sessionBrowserEditMode = new Map();
const COMPOSER_TYPE_PLACEHOLDER_URL = new URL("./placeholder.png", import.meta.url).href;
const COMPOSER_TYPE_ALL_URL = new URL("./all.png", import.meta.url).href;
const COMPOSER_CATEGORY_KEY_SEPARATOR = "::";

function buildComposerCategoryKey(typeFile, categoryName) {
    const normalizedTypeFile = String(typeFile || "").trim();
    const normalizedCategory = String(categoryName || "").trim();
    if (!normalizedTypeFile) return normalizedCategory;
    return `${normalizedTypeFile}${COMPOSER_CATEGORY_KEY_SEPARATOR}${normalizedCategory}`;
}

function parseComposerCategoryKey(category) {
    const raw = String(category || "").trim();
    const separatorIndex = raw.indexOf(COMPOSER_CATEGORY_KEY_SEPARATOR);
    if (separatorIndex <= 0) {
        return { typeFile: "", categoryName: raw };
    }
    return {
        typeFile: raw.slice(0, separatorIndex).trim(),
        categoryName: raw.slice(separatorIndex + COMPOSER_CATEGORY_KEY_SEPARATOR.length).trim(),
    };
}

function applyPromptPayloadToNode(node, payload) {
    if (!node || !payload || typeof payload !== "object") return;
    if (payload.library && typeof payload.library === "object") {
        node.composerPromptLibrary = payload.library;
    }
    if (payload.prompts && typeof payload.prompts === "object") {
        node.prompts = payload.prompts;
    }
}

let _thumbnailRenderFamily = null;
let _thumbnailRenderModel = null;
let _thumbnailRenderLora1 = null;
let _thumbnailRenderLora2 = null;
let _thumbnailRenderSteps = null;

function _normalizeThumbnailPromptStrength(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return 1.0;
    return Math.max(0.0, Math.min(5.0, numeric));
}

function _formatThumbnailPromptText(text, promptData) {
    const trimmed = String(text || "").trim();
    if (!trimmed) return "";
    const generationMode = String(promptData?.__pm_generation_mode || "image").trim().toLowerCase();
    if (generationMode === "video") return trimmed;
    const strength = _normalizeThumbnailPromptStrength(promptData?.__pm_part_strength);
    if (strength === 1.0) return trimmed;
    return `(${trimmed}:${strength.toFixed(15).replace(/\.?0+$/, "")})`;
}

// System prompts are instructions, not image descriptions, so their thumbnail is a title card.
function _buildTitleCardThumbnailPrompt(promptName) {
    const title = String(promptName || "").replace(/["“”]/g, "'").replace(/\s+/g, " ").trim() || "Untitled";
    return [
        `A clean minimalist typographic title card. The only text on the card is "${title}",`,
        "spelled exactly, written in large bold modern sans-serif lettering, perfectly centered and crisp.",
        "Flat dark charcoal background with a soft subtle color gradient and a thin glowing accent border with rounded corners,",
        "sleek app-icon poster design, high contrast, sharp edges, balanced negative space, vector graphic style.",
        "No other words, no letters besides the title, no people, no photos, no logos, no watermark.",
    ].join(" ");
}

let _thumbQueueTotal = 0;
let _thumbQueueDone = 0;
let _thumbQueueFailed = 0;
let _thumbQueueCancelled = false;
let _thumbQueueProgress = null;
let _thumbQueuePromiseChain = Promise.resolve();

function _syncThumbnailRenderState() {
    // Guard against TDZ: `getThumbnailRenderState` is a `let` that may be in its
    // temporal dead zone if this is invoked while the module is mid-evaluation
    // (e.g. via a circular import). `typeof` on a TDZ binding throws, so we read it
    // through a function that swallows the ReferenceError.
    let getter = null;
    try {
        getter = getThumbnailRenderState;
    } catch (e) {
        return; // TDZ / not yet initialized - nothing to sync.
    }
    if (typeof getter !== "function") return;
    const state = getter() || {};
    _thumbnailRenderFamily = state.family || null;
    _thumbnailRenderModel = state.model || null;
    _thumbnailRenderLora1 = state.lora1 || null;
    _thumbnailRenderLora2 = state.lora2 || null;
    _thumbnailRenderSteps = state.steps ?? null;
}

function getThumbnailRenderLabelParts() {
    _syncThumbnailRenderState();
    const selectedLoras = [_thumbnailRenderLora1, _thumbnailRenderLora2].filter(Boolean);
    if (_thumbnailRenderFamily && _thumbnailRenderModel) {
        const leafName = _thumbnailLeafName(_thumbnailRenderModel);
        const stepsPart = _thumbnailRenderSteps ? ` · ${_thumbnailRenderSteps} steps` : "";
        return {
            shortLabel: typeSafeTruncate(`🔧 ${leafName}`, 18),
            iconLabel: "🔧",
            fullLabel: `Thumbnail: ${_thumbnailRenderFamily} : ${leafName}${selectedLoras.length ? ` + ${selectedLoras.length} LoRA${selectedLoras.length > 1 ? "s" : ""}` : ""}${stepsPart}`,
        };
    }
    return {
        shortLabel: "🔧 Model",
        iconLabel: "🔧",
        fullLabel: "Select thumbnail family and model",
    };
}

function typeSafeTruncate(value, maxLength) {
    const text = String(value || "");
    const limit = Number(maxLength) || 0;
    if (!limit || text.length <= limit) return text;
    return `${text.slice(0, Math.max(0, limit - 1))}…`;
}

function showTextInputDialog(title, message, defaultValue = "") {
    return prompt({ title, message, value: String(defaultValue || "") });
}

// The browser's message, confirm and category dialogs, on the kit modal.
function showInfo(title, message) {
    return new Promise((resolve) => {
        const text = el("div", "", message);
        text.style.cssText = "max-width:420px;line-height:1.45;white-space:pre-wrap";
        const dialog = openModal({
            title,
            body: text,
            footer: button("OK", { primary: true, onClick: () => dialog.close() }),
            onClose: () => resolve(),
        });
    });
}

function showConfirm(title, message, confirmText = "Delete", { danger = true } = {}) {
    return confirm({ title, message, confirmText, danger });
}

function showNewCategoryDialog() {
    return formDialog({
        title: "New Category",
        confirmText: "Create",
        fields: [
            { key: "name", label: "Category name" },
            { key: "nsfw", label: "Mark as NSFW", type: "check" },
        ],
    });
}

// Rename a category; for the composer, also move it to another prompt group.
async function showRenameCategoryDialog(title, message, categories, defaultCategory, options = {}) {
    const groupOptions = Array.isArray(options.groupOptions) && options.groupOptions.length ? options.groupOptions : null;
    const defaultGroup = String(options.defaultGroupValue || groupOptions?.[0]?.value || "");
    const fields = [{ key: "newCategory", label: message, value: defaultCategory }];
    if (groupOptions) fields.push({ key: "typeFile", label: "Prompt Group", type: "select", options: groupOptions, value: defaultGroup });
    const result = await formDialog({ title, fields });
    if (!result) return null;
    return { oldCategory: String(defaultCategory || ""), newCategory: result.newCategory, typeFile: result.typeFile || defaultGroup };
}

function showNewPromptGroupDialog() {
    return formDialog({
        title: "New Prompt Group",
        confirmText: "Create",
        fields: [
            { key: "groupName", label: "Prompt group name" },
            { key: "categoryName", label: "Starting category" },
        ],
    });
}

function _getComposerImageThumbnailLoras(promptData) {
    const loraName = String(promptData?.lora_image || promptData?.lora || "").trim();
    if (!loraName) return [];
    const strength = Number(promptData?.lora_image_strength ?? promptData?.lora_strength ?? 1.0);
    const baseStrength = Number.isFinite(strength) ? strength : 1.0;
    const safeStrength = baseStrength * _normalizeThumbnailPromptStrength(promptData?.__pm_part_strength);
    return [{
        name: loraName,
        path: loraName,
        model_strength: safeStrength,
        clip_strength: safeStrength,
        active: true,
        available: true,
    }];
}

function _mergeThumbnailWorkflowLoras(baseLoras, extraLoras) {
    const base = Array.isArray(baseLoras) ? baseLoras : [];
    const extras = Array.isArray(extraLoras) ? extraLoras : [];
    if (!extras.length) return [...base];
    const extraKeys = new Set(extras.map((lora) => String(lora?.name || lora?.path || "").trim().toLowerCase()).filter(Boolean));
    const merged = base.filter((lora) => !extraKeys.has(String(lora?.name || lora?.path || "").trim().toLowerCase()));
    merged.push(...extras);
    return merged;
}

function _applyPromptImageLoraToThumbnailWorkflow(workflowData, promptData, slot = "model_a") {
    const extraLoras = _getComposerImageThumbnailLoras(promptData);
    if (!extraLoras.length || !workflowData || typeof workflowData !== "object") {
        return workflowData;
    }

    const wf = workflowData;
    const targetSlot = String(slot || "model_a").trim().toLowerCase();
    if (Number(wf.version || 0) >= 2 && wf.models && typeof wf.models === "object") {
        const block = (wf.models[targetSlot] && typeof wf.models[targetSlot] === "object")
            ? wf.models[targetSlot]
            : (wf.models[targetSlot] = {});
        block.loras = _mergeThumbnailWorkflowLoras(Array.isArray(block.loras) ? block.loras : [], extraLoras);
        return wf;
    }

    const legacyKey = (targetSlot === "model_b" || targetSlot === "model_d") ? "loras_b" : "loras_a";
    wf[legacyKey] = _mergeThumbnailWorkflowLoras(Array.isArray(wf[legacyKey]) ? wf[legacyKey] : [], extraLoras);
    return wf;
}

function _resolveThumbnailWorkflowSlot(workflowData) {
    const modelKeys = ["model_a", "model_b", "model_c", "model_d"];
    const explicitSlot = String(workflowData?.model_slot || "").trim().toLowerCase();
    if (modelKeys.includes(explicitSlot)) {
        return explicitSlot;
    }

    const models = workflowData?.models;
    if (models && typeof models === "object") {
        for (const slot of modelKeys) {
            const block = models[slot];
            if (!block || typeof block !== "object") continue;
            const modelName = String(block.model || "").trim();
            if (modelName) {
                return slot;
            }
        }
    }

    return "model_a";
}

export function configurePromptBrowserDeps(deps = {}) {
    if (deps.app) app = deps.app;
    if (deps.UI) UI = deps.UI;
    if (typeof deps.DEFAULT_THUMBNAIL === "string") DEFAULT_THUMBNAIL = deps.DEFAULT_THUMBNAIL;

    if (typeof deps.loadPrompts === "function") loadPrompts = deps.loadPrompts;
    if (typeof deps.ensureThumbnailRenderSelection === "function") ensureThumbnailRenderSelection = deps.ensureThumbnailRenderSelection;
    if (typeof deps.resolveThumbnailFallbackBase === "function") resolveThumbnailFallbackBase = deps.resolveThumbnailFallbackBase;
    if (typeof deps.generateThumbnailForPrompt === "function") generateThumbnailForPrompt = deps.generateThumbnailForPrompt;
    if (typeof deps.generateThumbnailWorkflowFromWorkflowData === "function") generateThumbnailWorkflowFromWorkflowData = deps.generateThumbnailWorkflowFromWorkflowData;
    if (typeof deps.buildRendererFallbackWorkflowDataWithBase === "function") buildRendererFallbackWorkflowDataWithBase = deps.buildRendererFallbackWorkflowDataWithBase;
    if (typeof deps.showThumbnailRenderPicker === "function") showThumbnailRenderPicker = deps.showThumbnailRenderPicker;
    if (typeof deps.saveThumbnailRenderSelection === "function") {
        const wrappedSave = deps.saveThumbnailRenderSelection;
        saveThumbnailRenderSelection = (selection) => {
            wrappedSave(selection);
            _syncThumbnailRenderState();
        };
    }
    if (typeof deps.thumbnailLeafName === "function") _thumbnailLeafName = deps.thumbnailLeafName;
    if (typeof deps.getThumbnailRenderState === "function") getThumbnailRenderState = deps.getThumbnailRenderState;

    _syncThumbnailRenderState();
}

// The thumbnail queue's toast: a spinner, "Generating i / n: name" and a
// button that cancels what is left.
function _ensureThumbQueueProgress() {
    if (_thumbQueueProgress) return;

    _thumbQueueCancelled = false;
    ensurePromptBrowserCss();
    const progress = el("div", "pm-progress");
    const spinner = el("span", "pm-spin");
    const text = el("span", "text");
    const cancelBtn = button("", { icon: ICONS.close, small: true, title: "Cancel the remaining thumbnails", onClick: () => {
        _thumbQueueCancelled = true;
        cancelBtn.disabled = true;
        cancelBtn.title = "Cancelling…";
    } });
    progress.append(spinner, text, cancelBtn);
    document.body.append(progress);

    _thumbQueueProgress = { root: progress, text, cancelBtn };
}

// One thumbnail into the shared queue (the toast counts it). Resolves to
// { value } with what make() returned, or { error }; null value when
// cancelled. Never rejects, so the queue keeps going.
function queueThumbnail(name, make) {
    _thumbQueueTotal++;
    _ensureThumbQueueProgress();
    _updateThumbQueueProgress(name);
    return new Promise((resolve) => {
        _thumbQueuePromiseChain = _thumbQueuePromiseChain.then(async () => {
            try {
                if (_thumbQueueCancelled) {
                    _thumbQueueDone++;
                    resolve({ value: null });
                    return;
                }
                _updateThumbQueueProgress(name);
                const value = await make();
                _thumbQueueDone++;
                resolve({ value });
            } catch (e) {
                const error = e instanceof Error ? e : new Error(e == null ? "Thumbnail generation failed." : String(e));
                console.error(`[ThumbnailGen] Failed for "${name}":`, error);
                _thumbQueueFailed++;
                resolve({ error });
            } finally {
                if (_thumbQueueDone + _thumbQueueFailed >= _thumbQueueTotal) _finishThumbQueueProgress();
            }
        });
    });
}

function _updateThumbQueueProgress(currentName) {
    if (!_thumbQueueProgress) return;
    const done = _thumbQueueDone + _thumbQueueFailed;
    const remaining = Math.max(0, _thumbQueueTotal - done);
    _thumbQueueProgress.text.textContent = `Generating ${Math.min(done + 1, _thumbQueueTotal)} / ${_thumbQueueTotal}: ${currentName} (${remaining} remaining)`;
}

function _finishThumbQueueProgress() {
    if (!_thumbQueueProgress) return;
    if (_thumbQueueProgress.root.parentNode) {
        _thumbQueueProgress.root.parentNode.removeChild(_thumbQueueProgress.root);
    }
    _thumbQueueProgress = null;
    _thumbQueueTotal = 0;
    _thumbQueueDone = 0;
    _thumbQueueFailed = 0;
    _thumbQueueCancelled = false;
    _thumbQueuePromiseChain = Promise.resolve();
}

async function logThumbnailToServer(category, name, seed, prompt, mode) {
    try {
        await api.fetchApi("/prompt-manager/log-thumbnail", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                category: category,
                name: name,
                seed: seed,
                prompt: prompt,
                mode: mode,
            })
        });
    } catch (e) {
        console.error("[PromptBrowser] Failed to log thumbnail to server:", e);
    }
}

async function _generateThumbnailForBrowserCategory(node, category, promptName, onUpdate, options = {}) {
    const {
        renderSelection: providedRenderSelection = null,
        fallbackBase = null,
        endpointPrefix = "/prompt-manager-advanced",
        draftPromptData = null,
        persistThumbnail = true,
        promptStrength = 1.0,
        generationMode = "image",
        staticSeed = 42,
    } = options || {};

    // If browser has not been wired with low-level generation helpers yet,
    // fall back to legacy injected implementation.
    if (
        typeof generateThumbnailWorkflowFromWorkflowData !== "function" ||
        typeof buildRendererFallbackWorkflowDataWithBase !== "function" ||
        typeof saveThumbnailEntry !== "function"
    ) {
        return await generateThumbnailForPrompt(node, category, promptName, onUpdate, {
            queue: true,
            renderSelection: providedRenderSelection,
            fallbackBase,
            endpointPrefix,
        });
    }

    const savedPromptData = getCategoryPromptEntry(node?.prompts?.[category], promptName, endpointPrefix);
    const promptData = draftPromptData && typeof draftPromptData === "object"
        ? { ...(savedPromptData && typeof savedPromptData === "object" ? savedPromptData : {}), ...draftPromptData }
        : (savedPromptData && typeof savedPromptData === "object" ? { ...savedPromptData } : savedPromptData);
    if (!promptData) return;

    if (promptData && typeof promptData === "object") {
        promptData.__pm_part_strength = _normalizeThumbnailPromptStrength(promptStrength);
        promptData.__pm_generation_mode = String(generationMode || "image").trim().toLowerCase() || "image";
    }

    let promptText;
    let categoryBasePrompt = "";
    if (endpointPrefix === SYSTEM_PROMPTS_ENDPOINT_PREFIX) {
        promptText = _buildTitleCardThumbnailPrompt(promptName);
    } else {
        categoryBasePrompt = getCategoryBasePrompt(node, category);
        const categoryPromptPrefix = getCategoryPromptPrefix(node, category);
        const prefixedPromptText = prependCategoryPromptPrefix(
            categoryPromptPrefix,
            _formatThumbnailPromptText(promptData.prompt || promptName || "", promptData)
        );
        promptText = [String(categoryBasePrompt || "").trim(), prefixedPromptText]
            .filter(Boolean)
            .join(" ");
    }

    const isComposerManager = endpointPrefix === "/prompt-manager/compose";
    const numericSeed = Number(staticSeed);
    const staticSeedForRun = Number.isFinite(numericSeed) ? Math.trunc(numericSeed) : 42;

    console.log(`[ThumbnailGen] Preparing thumbnail for "${category}/${promptName}" | seed=${staticSeedForRun ?? "random"}`);
    console.log(`[ThumbnailGen] Effective prompt text: ${promptText}`);
    logThumbnailToServer(category, promptName, staticSeedForRun, promptText, isComposerManager ? "composer" : "pma");

    const activeRenderSelection = providedRenderSelection || (
        (_thumbnailRenderFamily && _thumbnailRenderModel)
            ? {
                family: _thumbnailRenderFamily,
                model: _thumbnailRenderModel,
                loras: [_thumbnailRenderLora1, _thumbnailRenderLora2].filter(Boolean),
            }
            : null
    );

    let thumbnail = null;

    const rawWorkflowData = promptData.workflow_data;
    let parsedWorkflowData = null;
    if (rawWorkflowData && typeof rawWorkflowData === "object") {
        parsedWorkflowData = rawWorkflowData;
    } else if (typeof rawWorkflowData === "string" && rawWorkflowData.trim()) {
        try {
            const maybeObj = JSON.parse(rawWorkflowData);
            if (maybeObj && typeof maybeObj === "object") {
                parsedWorkflowData = maybeObj;
            }
        } catch {
            parsedWorkflowData = null;
        }
    }

    if (parsedWorkflowData) {
        const effectivePrompt = String(promptText || "").trim();
        const thumbnailSlot = _resolveThumbnailWorkflowSlot(parsedWorkflowData);
        if (Number(parsedWorkflowData.version || 0) >= 2 && parsedWorkflowData.models && typeof parsedWorkflowData.models === "object") {
            const targetModel = parsedWorkflowData.models[thumbnailSlot] || parsedWorkflowData.models.model_a;
            if (targetModel && typeof targetModel === "object") {
                targetModel.positive_prompt = effectivePrompt;
            }
        } else {
            parsedWorkflowData.positive_prompt = effectivePrompt;
        }
        _applyPromptImageLoraToThumbnailWorkflow(parsedWorkflowData, promptData, thumbnailSlot);
        thumbnail = await generateThumbnailWorkflowFromWorkflowData(parsedWorkflowData, activeRenderSelection, {
            staticSeed: staticSeedForRun,
            promptData,
        });
    }

    if (!thumbnail) {
        const renderSelection = providedRenderSelection || await ensureThumbnailRenderSelection();
        if (!renderSelection) return;

        const fallbackWorkflowData = await buildRendererFallbackWorkflowDataWithBase(
            promptText,
            { ...promptData, base_prompt: categoryBasePrompt },
            renderSelection,
            fallbackBase
        );
        thumbnail = await generateThumbnailWorkflowFromWorkflowData(fallbackWorkflowData, renderSelection, {
            staticSeed: staticSeedForRun,
            promptData,
        });
    }

    if (thumbnail && persistThumbnail) {
        await saveThumbnailEntry(node, category, promptName, thumbnail, endpointPrefix);
        onUpdate?.();
    }

    return thumbnail;
}

async function saveThumbnailEntry(node, category, promptName, thumbnail, endpointPrefix = "/prompt-manager-advanced") {
    try {
        const url = `${endpointPrefix}/save-thumbnail`;
        const composerCategory = endpointPrefix === "/prompt-manager/compose"
            ? getComposerCategoryRequestIdentity(node, category)
            : null;
        const response = await api.fetchApi(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                category: composerCategory?.categoryName || category,
                name: promptName,
                thumbnail: thumbnail,
                ...(composerCategory?.typeFile ? { type_file: composerCategory.typeFile } : {}),
            })
        });

        const rawText = await response.text();
        let data;
        try {
            data = JSON.parse(rawText);
        } catch (parseErr) {
            console.error("[PromptBrowser] Non-JSON save-thumbnail response:", response.status, url, rawText);
            throw parseErr;
        }
        if (data.success) {
            const entry = getCategoryPromptEntry(node?.prompts?.[category], promptName, endpointPrefix);
            if (entry) {
                if (thumbnail) {
                    entry.thumbnail = thumbnail;
                } else {
                    delete entry.thumbnail;
                }
            }
        } else {
            await showInfo("Error", data.error || "Failed to save thumbnail");
        }
    } catch (error) {
        console.error("[PromptBrowser] Error saving thumbnail:", error);
        await showInfo("Error", "Failed to save thumbnail");
    }
}

async function deletePromptEntry(node, category, promptName, endpointPrefix = "/prompt-manager-advanced") {
    try {
        const composerCategory = endpointPrefix === "/prompt-manager/compose"
            ? getComposerCategoryRequestIdentity(node, category)
            : null;
        const response = await api.fetchApi(`${endpointPrefix}/delete-prompt`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                category: composerCategory?.categoryName || category,
                name: promptName,
                ...(composerCategory?.typeFile ? { type_file: composerCategory.typeFile } : {}),
            })
        });
        const data = await response.json();
        if (data.success) {
            applyPromptPayloadToNode(node, data);
        } else {
            await showInfo("Error", data.error || "Failed to delete prompt");
        }
    } catch (error) {
        console.error("[PromptBrowser] Error deleting prompt:", error);
        await showInfo("Error", "Failed to delete prompt");
    }
}

function resizeImageToThumbnail(file, minSize = 200) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                let width = img.width;
                let height = img.height;
                const minDim = Math.min(width, height);

                if (minDim > minSize) {
                    const scale = minSize / minDim;
                    width = Math.round(width * scale);
                    height = Math.round(height * scale);
                }

                const canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                resolve(canvas.toDataURL('image/jpeg', 0.85));
            };
            img.onerror = reject;
            img.src = e.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function resizeImageToCoverDataUrl(file, outputSize = 64, outputType = "image/png", quality = 0.92) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement("canvas");
                canvas.width = outputSize;
                canvas.height = outputSize;
                const ctx = canvas.getContext("2d");
                if (!ctx) {
                    reject(new Error("Canvas context unavailable"));
                    return;
                }

                const sourceWidth = Math.max(1, Number(img.width) || 1);
                const sourceHeight = Math.max(1, Number(img.height) || 1);
                const scale = Math.max(outputSize / sourceWidth, outputSize / sourceHeight);
                const drawWidth = sourceWidth * scale;
                const drawHeight = sourceHeight * scale;
                const offsetX = (outputSize - drawWidth) / 2;
                const offsetY = (outputSize - drawHeight) / 2;

                ctx.clearRect(0, 0, outputSize, outputSize);
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = "high";
                ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);

                resolve(canvas.toDataURL(outputType, quality));
            };
            img.onerror = reject;
            img.src = e.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

async function readClipboardImageFile() {
    const items = await navigator.clipboard.read();
    for (const item of items) {
        for (const type of item.types) {
            if (!type.startsWith("image/")) continue;
            const blob = await item.getType(type);
            return new File([blob], "clipboard-image", { type: blob.type || type });
        }
    }
    return null;
}

function _attachThumbnailDropHandlers(element, node, category, promptName, onUpdate, endpointPrefix = "/prompt-manager-advanced") {
    element.addEventListener("dragover", (e) => {
        const hasFiles = Array.from(e.dataTransfer?.types || []).includes("Files");
        if (!hasFiles) return;
        e.preventDefault();
        e.stopPropagation();
        element.style.outline = `2px dashed ${T.brand}`;
        element.style.outlineOffset = "2px";
    });
    element.addEventListener("dragleave", (e) => {
        e.preventDefault();
        e.stopPropagation();
        element.style.outline = "";
        element.style.outlineOffset = "";
    });
    element.addEventListener("drop", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        element.style.outline = "";
        element.style.outlineOffset = "";

        const files = Array.from(e.dataTransfer?.files || []);
        const imageFile = files.find((f) => f.type?.startsWith("image/"));
        if (!imageFile) return;

        const existing = getCategoryPromptEntry(node?.prompts?.[category], promptName, endpointPrefix)?.thumbnail;
        if (existing) {
            const confirmed = await showConfirm(
                "Replace Thumbnail",
                `"${promptName}" already has a thumbnail. Do you want to replace it with the dropped image?`,
                "Replace",
            );
            if (!confirmed) return;
        }

        try {
            const thumbnail = await resizeImageToThumbnail(imageFile, 200);
            await saveThumbnailEntry(node, category, promptName, thumbnail, endpointPrefix);
            onUpdate();
        } catch (error) {
            console.error("[PromptBrowser] Error setting thumbnail from drop:", error);
            await showInfo("Error", "Failed to set thumbnail from dropped image.");
        }
    });
}

// The Prompt Group select (composer) and the Category select that follows it.
function categoryFields(categories, defaultCategory, options) {
    const groupOptions = Array.isArray(options.groupOptions) && options.groupOptions.length ? options.groupOptions : null;
    const categoriesByGroup = options.categoriesByGroup && typeof options.categoriesByGroup === "object" ? options.categoriesByGroup : null;
    const defaultGroup = String(options.defaultGroupValue || groupOptions?.[0]?.value || "");
    const fields = [];
    if (groupOptions) fields.push({ key: "typeFile", label: "Prompt Group", type: "select", options: groupOptions, value: defaultGroup });
    fields.push({
        key: "category",
        label: options.categoryLabel || "Category",
        type: "select",
        value: String(defaultCategory || ""),
        options: (values) => (categoriesByGroup ? categoriesByGroup[values.typeFile ?? defaultGroup] || [] : categories || []),
    });
    return { fields, defaultGroup };
}

async function showPromptWithCategoryDialog(title, defaultName, categories, defaultCategory, options = {}) {
    const { fields, defaultGroup } = categoryFields(categories, defaultCategory, options);
    const result = await formDialog({ title, fields: [{ key: "name", label: "Prompt name", value: defaultName }, ...fields] });
    if (!result) return null;
    return { name: result.name, category: result.category || String(defaultCategory || ""), typeFile: result.typeFile || defaultGroup };
}

async function showCategoryPickerDialog(title, categories, defaultCategory, options = {}) {
    const { fields, defaultGroup } = categoryFields(categories, defaultCategory, { ...options, categoryLabel: "Target category" });
    const result = await formDialog({ title, fields, confirmText: "Move" });
    if (!result) return null;
    return { category: result.category || String(defaultCategory || ""), typeFile: result.typeFile || defaultGroup };
}

function showThumbnailContextMenu(event, node, category, promptName, onUpdate, endpointPrefix = "/prompt-manager-advanced", options = {}) {
    const onDelete = typeof options?.onDelete === "function" ? options.onDelete : null;
    const isComposerManager = endpointPrefix === "/prompt-manager/compose";
    const composerCategory = isComposerManager ? getComposerCategoryRequestIdentity(node, category) : null;
    const categoryLabel = composerCategory?.categoryName || category;
    const promptData = getCategoryPromptEntry(node?.prompts?.[category], promptName, endpointPrefix);
    const isNSFW = promptData?.nsfw === true;

    const pasteThumbnail = async () => {
        try {
            const file = await readClipboardImageFile();
            if (file) {
                const thumbnail = await resizeImageToThumbnail(file, 200);
                await saveThumbnailEntry(node, category, promptName, thumbnail, endpointPrefix);
                onUpdate();
                return;
            }
            await showInfo("No Image", "No image found in clipboard");
        } catch (error) {
            console.error("[PromptBrowser] Error reading clipboard:", error);
            await showInfo("Error", "Failed to read clipboard. Make sure you have an image copied.");
        }
    };

    const toggleNSFW = async () => {
        try {
            const resp = await api.fetchApi(`${endpointPrefix}/toggle-nsfw`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    type: "prompt",
                    category: categoryLabel,
                    name: promptName,
                    ...(composerCategory?.typeFile ? { type_file: composerCategory.typeFile } : {}),
                }),
            });
            const result = await resp.json();
            if (result.success) {
                applyPromptPayloadToNode(node, result);
                onUpdate();
            }
        } catch (err) {
            console.error("[PromptBrowser] Error toggling prompt NSFW:", err);
        }
    };

    const renameOrMove = async () => {
        const allCategories = Object.keys(node.prompts || {}).filter(c => c !== "__meta__").sort((a, b) => a.localeCompare(b));
        const currentTypeFile = composerCategory?.typeFile || "";
        const composerMoveTargets = isComposerManager ? getComposerMoveTargetOptions(node) : null;
        const result = await showPromptWithCategoryDialog(
            "Rename / Move Prompt",
            promptName,
            allCategories,
            categoryLabel,
            isComposerManager ? {
                groupOptions: composerMoveTargets?.groupOptions || [],
                categoriesByGroup: composerMoveTargets?.categoriesByGroup || {},
                defaultGroupValue: currentTypeFile,
            } : {}
        );
        if (!result?.name?.trim()) return;
        const newName = result.name.trim();
        const newCat = result.category;
        const newTypeFile = isComposerManager ? String(result.typeFile || currentTypeFile || "") : "";
        if (newName === promptName && newCat === categoryLabel && (!isComposerManager || newTypeFile === currentTypeFile)) return;
        try {
            const resp = await api.fetchApi(`${endpointPrefix}/rename-prompt`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    category: categoryLabel,
                    old_name: promptName,
                    new_name: newName,
                    new_category: newCat,
                    ...(isComposerManager ? {
                        type_file: currentTypeFile,
                        new_type_file: newTypeFile || currentTypeFile,
                    } : {}),
                }),
            });
            const data = await resp.json();
            if (data.success) {
                applyPromptPayloadToNode(node, data);
                onUpdate();
            } else {
                await showInfo("Error", data.error || "Failed to rename prompt");
            }
        } catch (err) {
            console.error("[PromptBrowser] Error renaming prompt:", err);
        }
    };

    openMenu(event.clientX, event.clientY, [
        { label: "Paste thumbnail", action: pasteThumbnail },
        null,
        { label: "Generate thumbnail", action: () => queueThumbnail(promptName, () => _generateThumbnailForBrowserCategory(node, category, promptName, onUpdate, { endpointPrefix })) },
        promptData?.thumbnail ? { label: "Remove thumbnail", action: async () => {
            await saveThumbnailEntry(node, category, promptName, null, endpointPrefix);
            onUpdate();
        } } : undefined,
        null,
        { label: isNSFW ? "✓ NSFW" : "Mark as NSFW", danger: isNSFW, action: toggleNSFW },
        null,
        { label: "Rename / Move…", action: renameOrMove },
        null,
        { label: "Delete prompt", danger: true, action: async () => {
            if (await showConfirm("Delete Prompt", `Are you sure you want to delete prompt "${promptName}"?`)) {
                await deletePromptEntry(node, category, promptName, endpointPrefix);
                await onDelete?.(category, promptName);
                onUpdate();
            }
        } },
    ].filter((item) => item !== undefined));
}

// Browser session state persists during a UI session and resets on restart.
let _sessionHideNSFW = null;
let _sessionViewModeByScope = {
    manager: null,
    composer: null,
};
let _sessionEnableThumbnailPreview = null;
let _sessionBrowserContentFilter = null;
let _sessionPromptColumnWidthByScope = {
    manager: null,
    composer: null,
};
let _sessionTypeRailExpandedByScope = {
    manager: null,
    composer: null,
};

function normalizeBrowserPrefScope(scope) {
    return scope === "composer" ? "composer" : "manager";
}

function isHiddenCategoryEntryKey(key) {
    return isHiddenPromptEntryKey(key);
}

function getCategoryPromptEntries(categoryPrompts, endpointPrefix = "/prompt-manager-advanced") {
    return getCategoryPromptEntriesForEndpoint(categoryPrompts, endpointPrefix);
}

function getCategoryBasePrompt(node, category) {
    const raw = node?.prompts?.[category]?._base_prompt_;
    return typeof raw === "string" ? raw : "";
}

function getCategoryPromptPrefix(node, category) {
    const raw = node?.prompts?.[category]?._prompt_prefix_;
    return typeof raw === "string" ? raw : "";
}

function prependCategoryPromptPrefix(prefix, promptText) {
    const normalizedPromptText = String(promptText || "").trim();
    if (!normalizedPromptText) return "";
    const normalizedPrefix = String(prefix || "").trim();
    if (!normalizedPrefix) return normalizedPromptText;
    return `${normalizedPrefix} ${normalizedPromptText}`;
}

function getCategoryPromptType(node, category) {
    const raw = node?.prompts?.[category]?._prompt_type_;
    return typeof raw === "string" ? raw : "";
}

function getComposerTypeLibrary(node) {
    const types = node?.composerPromptLibrary?._types_;
    return types && typeof types === "object" && !Array.isArray(types) ? types : null;
}

function getOrderedComposerTypeEntries(node) {
    const types = getComposerTypeLibrary(node);
    const entries = Object.entries(types || {});
    const hasExplicitOrder = entries.some(([, typeData]) => Number.isInteger(Number(typeData?.order)));
    if (!hasExplicitOrder) {
        return entries.sort((a, b) => String(a[0] || "").localeCompare(String(b[0] || ""), undefined, { sensitivity: "base" }));
    }
    return entries.sort((a, b) => {
        const aOrder = Number.isInteger(Number(a[1]?.order)) ? Number(a[1].order) : Number.POSITIVE_INFINITY;
        const bOrder = Number.isInteger(Number(b[1]?.order)) ? Number(b[1].order) : Number.POSITIVE_INFINITY;
        if (aOrder !== bOrder) return aOrder - bOrder;
        return String(a[0] || "").localeCompare(String(b[0] || ""), undefined, { sensitivity: "base" });
    });
}

function getComposerMoveTargetOptions(node) {
    const groupOptions = [];
    const categoriesByGroup = {};
    for (const [typeFile, typeData] of getOrderedComposerTypeEntries(node)) {
        const resolvedTypeFile = String(typeFile || "").trim();
        if (!resolvedTypeFile) continue;
        groupOptions.push({
            value: resolvedTypeFile,
            label: String(typeData?.name || resolvedTypeFile.replace(/\.json$/i, "") || "").trim() || resolvedTypeFile,
        });
        const categories = typeData?.categories;
        categoriesByGroup[resolvedTypeFile] = categories && typeof categories === "object" && !Array.isArray(categories)
            ? Object.keys(categories).sort((a, b) => String(a || "").localeCompare(String(b || ""), undefined, { sensitivity: "base" }))
            : [];
    }
    return { groupOptions, categoriesByGroup };
}

function resolveComposerCategoryKey(node, category, explicitTypeFile = "") {
    const raw = String(category || "").trim();
    if (!raw) return "";
    if (node?.prompts?.[raw] && typeof node.prompts[raw] === "object") {
        return raw;
    }

    const parsed = parseComposerCategoryKey(raw);
    const normalizedCategoryName = String(parsed.categoryName || raw).trim().toLowerCase();
    const normalizedTypeFile = String(explicitTypeFile || parsed.typeFile || "").trim().toLowerCase();
    let fallback = "";

    for (const [categoryKey, categoryData] of Object.entries(node?.prompts || {})) {
        if (categoryKey === "__meta__" || !categoryData || typeof categoryData !== "object") continue;
        const candidateCategoryName = String(categoryData._category_name_ || categoryKey).trim().toLowerCase();
        if (candidateCategoryName !== normalizedCategoryName) continue;
        const candidateTypeFile = String(categoryData._type_file_ || "").trim().toLowerCase();
        if (normalizedTypeFile && candidateTypeFile === normalizedTypeFile) {
            return categoryKey;
        }
        if (!fallback) {
            fallback = categoryKey;
        }
    }

    if (fallback) return fallback;
    if (parsed.typeFile && parsed.categoryName) {
        return buildComposerCategoryKey(parsed.typeFile, parsed.categoryName);
    }
    if (explicitTypeFile) {
        return buildComposerCategoryKey(explicitTypeFile, parsed.categoryName || raw);
    }
    return raw;
}

function getComposerCategoryDisplayName(node, category) {
    const resolvedKey = resolveComposerCategoryKey(node, category);
    const metadataName = String(node?.prompts?.[resolvedKey]?._category_name_ || "").trim();
    if (metadataName) return metadataName;
    return parseComposerCategoryKey(resolvedKey || category).categoryName || String(category || "").trim();
}

function getComposerCategoryTypeFile(node, category) {
    const resolvedKey = resolveComposerCategoryKey(node, category);
    const flattenedTypeFile = String(node?.prompts?.[resolvedKey]?._type_file_ || "").trim();
    if (flattenedTypeFile) return flattenedTypeFile;

    const parsed = parseComposerCategoryKey(category);
    if (parsed.typeFile) return parsed.typeFile;

    const normalizedCategory = String(getComposerCategoryDisplayName(node, category) || "").trim().toLowerCase();
    if (!normalizedCategory) return "";
    for (const [typeFile, typeData] of getOrderedComposerTypeEntries(node)) {
        const categories = typeData?.categories;
        if (!categories || typeof categories !== "object" || Array.isArray(categories)) continue;
        const match = Object.keys(categories).find((name) => String(name || "").trim().toLowerCase() === normalizedCategory);
        if (match) return String(typeFile || "").trim();
    }
    return "";
}

function getComposerCategoryRequestIdentity(node, category, explicitTypeFile = "") {
    const categoryKey = resolveComposerCategoryKey(node, category, explicitTypeFile);
    const categoryName = getComposerCategoryDisplayName(node, categoryKey || category);
    const typeFile = String(explicitTypeFile || getComposerCategoryTypeFile(node, categoryKey || category) || "").trim();
    return { categoryKey: categoryKey || String(category || "").trim(), categoryName, typeFile };
}

function getComposerTypeFile(node, typeValue) {
    const normalized = String(typeValue || "").trim().toLowerCase();
    if (!normalized) return "";
    const types = getComposerTypeLibrary(node);
    if (types) {
        for (const typeFile of Object.keys(types)) {
            if (String(typeFile || "").replace(/\.json$/i, "").trim().toLowerCase() === normalized) {
                return String(typeFile || "");
            }
        }
    }
    const categoryMatch = Object.values(node?.prompts || {}).find((categoryData) => (
        categoryData
        && typeof categoryData === "object"
        && String(categoryData._prompt_type_ || "").trim().toLowerCase() === normalized
        && String(categoryData._type_file_ || "").trim()
    ));
    return String(categoryMatch?._type_file_ || `${normalized}.json`);
}

function getComposerTypeDisplayName(node, typeValue) {
    const typeFile = getComposerTypeFile(node, typeValue);
    const types = getComposerTypeLibrary(node);
    const typeData = typeFile ? types?.[typeFile] : null;
    if (typeData && String(typeData.name || "").trim()) {
        return String(typeData.name || "").trim();
    }
    const categoryMatch = Object.values(node?.prompts || {}).find((categoryData) => (
        categoryData
        && typeof categoryData === "object"
        && String(categoryData._prompt_type_ || "").trim().toLowerCase() === String(typeValue || "").trim().toLowerCase()
        && String(categoryData._type_name_ || "").trim()
    ));
    return String(categoryMatch?._type_name_ || typeValue || "").trim();
}

function getComposerTypeIconValue(node, typeValue) {
    const normalizedTypeValue = String(typeValue || "").trim().toLowerCase();
    if (normalizedTypeValue === "__all__") return COMPOSER_TYPE_ALL_URL;

    const typeFile = getComposerTypeFile(node, normalizedTypeValue);
    const types = getComposerTypeLibrary(node);
    const typeData = typeFile ? types?.[typeFile] : null;
    const embeddedIcon = String(typeData?.icon || typeData?.icon_url || "").trim();
    if (embeddedIcon) return embeddedIcon;

    const categoryMatch = Object.values(node?.prompts || {}).find((categoryData) => (
        categoryData
        && typeof categoryData === "object"
        && String(categoryData._prompt_type_ || "").trim().toLowerCase() === normalizedTypeValue
        && String(categoryData._type_icon_ || categoryData._type_icon_url_ || "").trim()
    ));
    const categoryIcon = String(categoryMatch?._type_icon_ || categoryMatch?._type_icon_url_ || "").trim();
    if (categoryIcon) return categoryIcon;
    return COMPOSER_TYPE_PLACEHOLDER_URL;
}

function isComposerTypeNSFW(node, typeValue) {
    const typeFile = getComposerTypeFile(node, typeValue);
    const types = getComposerTypeLibrary(node);
    const typeData = typeFile ? types?.[typeFile] : null;
    if (typeData && typeData.nsfw === true) {
        return true;
    }
    return Object.values(node?.prompts || {}).some((categoryData) => (
        categoryData
        && typeof categoryData === "object"
        && String(categoryData._prompt_type_ || "").trim().toLowerCase() === String(typeValue || "").trim().toLowerCase()
        && categoryData._type_nsfw_ === true
    ));
}

function getCategoriesForComposerType(node, typeValue) {
    const normalized = String(typeValue || "").trim().toLowerCase();
    if (!normalized) return [];
    return Object.keys(node?.prompts || {})
        .filter((category) => category !== "__meta__")
        .filter((category) => String(getCategoryPromptType(node, category) || "").trim().toLowerCase() === normalized);
}

function getOrderedComposerCategories(node, categories = null) {
    const sourceCategories = Array.isArray(categories)
        ? categories.filter((category) => String(category || "").trim() && String(category) !== "__meta__")
        : Object.keys(node?.prompts || {}).filter((category) => String(category || "").trim() && String(category) !== "__meta__");

    const canonicalEntries = getOrderedComposerTypeEntries(node);
    if (!canonicalEntries.length) {
        return [...sourceCategories].sort((a, b) => a.localeCompare(b));
    }

    const categoryOrder = new Map();
    let nextIndex = 0;
    for (const [typeFile, typeData] of canonicalEntries) {
        const categoryNames = Object.keys(typeData?.categories || {}).sort((a, b) => a.localeCompare(b));
        for (const categoryName of categoryNames) {
            const compositeKey = buildComposerCategoryKey(typeFile, categoryName);
            if (!categoryOrder.has(compositeKey)) {
                categoryOrder.set(compositeKey, nextIndex++);
            }
            if (!categoryOrder.has(categoryName)) {
                categoryOrder.set(categoryName, categoryOrder.get(compositeKey));
            }
        }
    }

    return [...sourceCategories].sort((a, b) => {
        const aIndex = categoryOrder.has(a) ? categoryOrder.get(a) : Number.POSITIVE_INFINITY;
        const bIndex = categoryOrder.has(b) ? categoryOrder.get(b) : Number.POSITIVE_INFINITY;
        if (aIndex !== bIndex) return aIndex - bIndex;
        return a.localeCompare(b);
    });
}

function getViewModeStorageKey(scope) {
    return normalizeBrowserPrefScope(scope) === "composer"
        ? "PromptManager.BrowserViewMode.Composer"
        : "PromptManager.BrowserViewMode";
}

function getPromptColumnWidthStorageKey(scope) {
    return normalizeBrowserPrefScope(scope) === "composer"
        ? "PromptManager.ListPromptColumnWidth.Composer"
        : "PromptManager.ListPromptColumnWidth";
}

function getTypeRailExpandedStorageKey(scope) {
    return normalizeBrowserPrefScope(scope) === "composer"
        ? "PromptManager.TypeRailExpanded.Composer"
        : "PromptManager.TypeRailExpanded";
}

export function getHideNSFW(runtimeApp = app) {
    if (_sessionHideNSFW !== null) return _sessionHideNSFW;
    return runtimeApp?.ui?.settings?.getSettingValue("PromptManager.DefaultHideNSFW");
}

export function setHideNSFW(value) {
    _sessionHideNSFW = value;
}

export function getViewMode(compactBrowser = false, scope = "manager") {
    const prefScope = normalizeBrowserPrefScope(scope);
    const sessionMode = _sessionViewModeByScope[prefScope];
    if (sessionMode !== null) {
        if (compactBrowser && sessionMode === "icon") return "grid";
        return sessionMode;
    }
    const storageKey = getViewModeStorageKey(prefScope);
    const stored = localStorage.getItem(storageKey);
    const mode = stored === "list" || stored === "icon" ? stored : "grid";
    if (compactBrowser && mode === "icon") return "grid";
    return mode;
}

export function setViewMode(value, scope = "manager") {
    const prefScope = normalizeBrowserPrefScope(scope);
    _sessionViewModeByScope[prefScope] = value;
}

function clampPromptColumnWidth(value, promptOnly = false) {
    const min = promptOnly ? 220 : 200;
    const max = promptOnly ? 1200 : 960;
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return promptOnly ? 360 : 320;
    return Math.max(min, Math.min(max, Math.round(numeric)));
}

export function getPromptColumnWidth(scope = "manager", promptOnly = false, fallbackWidth = null) {
    const prefScope = normalizeBrowserPrefScope(scope);
    const sessionValue = _sessionPromptColumnWidthByScope[prefScope];
    if (sessionValue !== null && sessionValue !== undefined) {
        return clampPromptColumnWidth(sessionValue, promptOnly);
    }
    const storageKey = getPromptColumnWidthStorageKey(prefScope);
    const stored = localStorage.getItem(storageKey);
    if (stored === null) {
        const fallback = fallbackWidth !== null && fallbackWidth !== undefined
            ? fallbackWidth
            : (promptOnly ? 520 : 420);
        return clampPromptColumnWidth(fallback, promptOnly);
    }
    return clampPromptColumnWidth(stored, promptOnly);
}

export function setPromptColumnWidth(value, scope = "manager", promptOnly = false) {
    const prefScope = normalizeBrowserPrefScope(scope);
    _sessionPromptColumnWidthByScope[prefScope] = clampPromptColumnWidth(value, promptOnly);
}

export function getTypeRailExpanded(scope = "manager") {
    const prefScope = normalizeBrowserPrefScope(scope);
    const sessionValue = _sessionTypeRailExpandedByScope[prefScope];
    if (sessionValue !== null) {
        return sessionValue === true;
    }
    const stored = localStorage.getItem(getTypeRailExpandedStorageKey(prefScope));
    return stored === "true";
}

export function setTypeRailExpanded(value, scope = "manager") {
    const prefScope = normalizeBrowserPrefScope(scope);
    const normalized = value === true;
    _sessionTypeRailExpandedByScope[prefScope] = normalized;
}

export function getThumbnailPreviewEnabled(runtimeApp = app) {
    if (_sessionEnableThumbnailPreview !== null) return _sessionEnableThumbnailPreview;
    const settingValue = runtimeApp?.ui?.settings?.getSettingValue("PromptManager.EnableThumbnailPreview");
    if (settingValue !== undefined) return settingValue;
    const stored = localStorage.getItem("PromptManager.EnableThumbnailPreview");
    return stored !== null ? stored === "true" : true;
}

export function setThumbnailPreviewEnabled(value) {
    _sessionEnableThumbnailPreview = value;
}

export function getBrowserContentFilter(runtimeApp = app) {
    if (_sessionBrowserContentFilter !== null) return _sessionBrowserContentFilter;
    const saved = String(runtimeApp?.ui?.settings?.getSettingValue("PromptManager.BrowserContentFilter") || "all").toLowerCase();
    return ["prompt", "recipe", "compose", "layout", "all"].includes(saved) ? saved : "all";
}

export function setBrowserContentFilter(value) {
    _sessionBrowserContentFilter = value;
}

function parseWorkflowDataCandidate(rawWorkflowData) {
    if (rawWorkflowData && typeof rawWorkflowData === "object") {
        return rawWorkflowData;
    }
    if (typeof rawWorkflowData === "string" && rawWorkflowData.trim()) {
        try {
            const parsed = JSON.parse(rawWorkflowData);
            return parsed && typeof parsed === "object" ? parsed : null;
        } catch {
            return null;
        }
    }
    return null;
}

export function hasWorkflowDataPayload(rawWorkflowData) {
    return (
        (typeof rawWorkflowData === "string" && rawWorkflowData.trim().length > 0) ||
        (rawWorkflowData && typeof rawWorkflowData === "object" && Object.keys(rawWorkflowData).length > 0)
    );
}

// Library entries are classified by the manager_data block their source node wrote, not by who saved them.
function getWorkflowBlock(promptData, key) {
    const workflowData = parseWorkflowDataCandidate(promptData?.workflow_data);
    const block = workflowData && typeof workflowData === "object" ? workflowData[key] : null;
    return block && typeof block === "object" ? block : null;
}

// Prompt Composer or Prompt Layout Builder data.
export function hasComposeLikePayload(promptData) {
    return !!(getWorkflowBlock(promptData, "prompt_composer") || getWorkflowBlock(promptData, "prompt_layout"));
}

// Prompt Layout Builder state; compose-like, but Prompt Composer cannot use it.
export function hasLayoutPayload(promptData) {
    return !!getWorkflowBlock(promptData, "prompt_layout");
}

export function hasPromptManagerPayload(promptData) {
    return !!getWorkflowBlock(promptData, "prompt_manager");
}

const LAYOUT_BADGE_COLOR = "rgba(124, 92, 220, 0.95)";
const COMPOSE_BADGE_COLOR = "rgba(47, 146, 72, 0.95)";
const RECIPE_BADGE_COLOR = "rgba(235, 140, 35, 0.95)";

export function hasRecipeLikePayload(promptData) {
    if (!promptData || typeof promptData !== "object") return false;
    if (hasWorkflowDataPayload(promptData.workflow_data)) return true;
    return Object.prototype.hasOwnProperty.call(promptData, "workflow_data");
}

export function hasPromptPresetPayload(promptData) {
    if (!promptData || typeof promptData !== "object") return false;
    const hasExplicitPromptField = Object.prototype.hasOwnProperty.call(promptData, "prompt");
    const promptText = String(promptData.prompt || "").trim();
    const negativeText = String(promptData.negative_prompt || "").trim();
    const hasLorasA = Array.isArray(promptData.loras_a) && promptData.loras_a.some((lora) => String(lora?.name || "").trim().length > 0);
    const hasLorasB = Array.isArray(promptData.loras_b) && promptData.loras_b.some((lora) => String(lora?.name || "").trim().length > 0);
    const hasLorasC = Array.isArray(promptData.loras_c) && promptData.loras_c.some((lora) => String(lora?.name || "").trim().length > 0);
    const hasLorasD = Array.isArray(promptData.loras_d) && promptData.loras_d.some((lora) => String(lora?.name || "").trim().length > 0);
    return hasExplicitPromptField || promptText.length > 0 || negativeText.length > 0 || hasLorasA || hasLorasB || hasLorasC || hasLorasD;
}

function getCategoryPromptEntry(categoryPrompts, promptName, endpointPrefix = "/prompt-manager-advanced") {
    return getCategoryPromptEntryForEndpoint(categoryPrompts, promptName, endpointPrefix);
}

export function getPromptNamesForCategory(node, category, options = {}) {
    const hideNSFW = options.hideNSFW === true;
    const workflowOnly = options.workflowOnly === true;
    const contentFilter = String(options.contentFilter || "all").toLowerCase();
    const endpointPrefix = typeof options.endpointPrefix === "string" ? options.endpointPrefix : "/prompt-manager-advanced";
    const categoryPrompts = node?.prompts?.[category];
    if (!categoryPrompts || typeof categoryPrompts !== "object") return [];
    const promptEntries = getCategoryPromptEntries(categoryPrompts, endpointPrefix);

    let promptNames = Object.keys(promptEntries)
        .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

    if (hideNSFW) {
        promptNames = promptNames.filter((name) => promptEntries?.[name]?.nsfw !== true);
    }

    if (workflowOnly || contentFilter !== "all") {
        promptNames = promptNames.filter((name) => {
            const entry = promptEntries?.[name];
            const hasComposeData = hasComposeLikePayload(entry);
            const hasLayoutData = hasComposeData && hasLayoutPayload(entry);
            const hasRecipeData = !hasComposeData && !hasPromptManagerPayload(entry) && hasRecipeLikePayload(entry);
            const hasPromptData = !hasComposeData && hasPromptPresetPayload(entry);

            if (workflowOnly && !(hasComposeData || hasRecipeData || hasPromptData)) return false;
            if (contentFilter === "prompt") return hasPromptData;
            if (contentFilter === "compose") return hasComposeData && !hasLayoutData;
            if (contentFilter === "layout") return hasLayoutData;
            if (contentFilter === "recipe") return hasRecipeData;
            return true;
        });
    }

    return promptNames;
}

export function getVisibleCategories(node, options = {}) {
    const hideNSFW = options.hideNSFW === true;
    const workflowOnly = options.workflowOnly === true;
    const contentFilter = String(options.contentFilter || "all").toLowerCase();
    const filterEmptyCategories = options.filterEmptyCategories === true;
    const keepCategory = String(options.keepCategory || "");
    const endpointPrefix = typeof options.endpointPrefix === "string" ? options.endpointPrefix : "/prompt-manager-advanced";

    const categories = Object.keys(node?.prompts || {})
        .filter((c) => c !== "__meta__")
        .filter((category) => {
            if (!hideNSFW || endpointPrefix !== "/prompt-manager/compose") {
                return true;
            }
            const categoryData = node?.prompts?.[category];
            if (!categoryData || typeof categoryData !== "object") {
                return true;
            }
            return categoryData["__meta__"]?.nsfw !== true && categoryData._type_nsfw_ !== true;
        });

    const orderedCategories = endpointPrefix === "/prompt-manager/compose"
        ? getOrderedComposerCategories(node, categories)
        : [...categories].sort((a, b) => a.localeCompare(b));

    if (!filterEmptyCategories) return orderedCategories;

    return orderedCategories.filter((category) => {
        if (keepCategory && category === keepCategory) return true;
        const names = getPromptNamesForCategory(node, category, { hideNSFW, workflowOnly, contentFilter, endpointPrefix });
        return names.length > 0;
    });
}
async function standaloneShowThumbnailBrowser(node, currentCategory, currentPrompt, options = {}) {
    // Reload prompts to ensure we have the latest data. Callers can pass a custom
    // loader (e.g. Prompt Composer) so the browser uses the correct JSON store.
    const loadPromptsFn = typeof options?.loadPromptsFn === "function" ? options.loadPromptsFn : loadPrompts;
    await loadPromptsFn(node);

    const mode = options?.mode === "save" ? "save" : "select";
    const onSave = typeof options?.onSave === "function" ? options.onSave : null;
    const saveButtonText = options?.saveButtonText || "Save";
    const saveTitle = options?.title || "Save Workflow";
    const saveNamePlaceholder = options?.namePlaceholder || "Prompt name";
    const initialSaveName = typeof options?.initialName === "string" ? options.initialName : "";
    const workflowOnly = options?.workflowOnly === true || node?._isWorkflowManager === true;
    const allowedCategories = Array.isArray(options?.allowedCategories)
        ? options.allowedCategories.map((c) => String(c))
        : null;
    const allowedCategorySet = allowedCategories
        ? new Set(allowedCategories.map((c) => c.toLowerCase()))
        : null;
    const selectTitle = typeof options?.title === "string" ? options.title : null;
    const multiSelect = options?.multiSelect === true;
    const allowMultiSelect = options?.allowMultiSelect !== false;
    const supportsMultiSelect = mode !== "save" && allowMultiSelect;
    const startInMultiSelect = multiSelect && options?.startInMultiSelect !== false;
    const multiCategorySelect = multiSelect && options?.multiCategorySelect === true;
    const clearSelectionOnCategorySwitch = options?.clearSelectionOnCategorySwitch === true;
    const multiSelectActionMode = String(options?.multiSelectActionMode || "select").trim().toLowerCase();
    const endpointPrefix = typeof options?.endpointPrefix === "string" ? options.endpointPrefix : "/prompt-manager-advanced";
    const promptStrength = _normalizeThumbnailPromptStrength(options?.promptStrength);
    const thumbnailGenerationMode = String(options?.thumbnailGenerationMode || "image").trim().toLowerCase() || "image";
    const showCategoryTypeFilter = endpointPrefix === "/prompt-manager/compose";
    const requireDoubleClickToSelect = mode !== "save" && endpointPrefix === "/prompt-manager/compose";
    const initialCategoryTypeFilter = showCategoryTypeFilter
        ? (String(options?.initialCategoryTypeFilter || "__all__").trim() || "__all__")
        : "__all__";
    const promptOnly = options?.promptOnly === true;
    const browserPrefScope = normalizeBrowserPrefScope(
        options?.preferenceScope || (promptOnly ? "composer" : "manager")
    );
    const allowEditMode = options?.allowEditMode !== false;
    const saveUsesEditPanel = mode === "save" && allowEditMode && options?.saveWithEditPanel === true;
    const initialContentFilter = (() => {
        const raw = String(options?.contentFilter || "").trim().toLowerCase();
        return ["prompt", "recipe", "compose", "layout", "all"].includes(raw) ? raw : "";
    })();
    const filterEmptyCategories = options?.filterEmptyCategories === true;
    const showAllCategoriesToggle = options?.showAllCategoriesToggle === true;
    const hideContentFilterControl = options?.hideContentFilterControl === true;
    // Optional caller-owned Type filter cycle (e.g. Composition Manager); not saved to the global setting.
    const contentFilterCycle = Array.isArray(options?.contentFilterCycle) ? options.contentFilterCycle : null;
    const onContentFilterChange = typeof options?.onContentFilterChange === "function" ? options.onContentFilterChange : null;
    const useComposerMultiSelectActions = multiSelectActionMode === "composer-add";
    const rememberedEditMode = sessionBrowserEditMode.has(browserPrefScope)
        ? sessionBrowserEditMode.get(browserPrefScope) === true
        : null;
    let editMode = saveUsesEditPanel
        ? true
        : (allowEditMode && (rememberedEditMode ?? (options?.editMode === true)));
    let multiSelectMode = startInMultiSelect;
    let updateSelectButton = () => {};
    let updateFooterText = () => {};
    let updateSelectionToolbar = () => {};
    // Saving may target a category with no matching entries yet, so list empty categories by default.
    let showAllCategoriesState = mode === "save";

    const filterAllowedCategories = (categories) => {
        if (!allowedCategorySet || !Array.isArray(categories)) return categories;
        return categories.filter((cat) => {
            const normalizedKey = String(cat || "").toLowerCase();
            const normalizedCategoryName = getComposerCategoryDisplayName(node, cat).toLowerCase();
            return allowedCategorySet.has(normalizedKey) || allowedCategorySet.has(normalizedCategoryName);
        });
    };

    // Check if thumbnail preview is enabled from user preferences
    const previewEnabled = getThumbnailPreviewEnabled(app);
    
    return new Promise((resolve) => {
        // Clean up any stale preview elements from previous modal openings
        const stalePreviews = document.querySelectorAll('[data-pm-thumbnail-preview]');
        stalePreviews.forEach(preview => {
            if (preview.parentNode) {
                preview.parentNode.removeChild(preview);
            }
        });
        
        let selectedCategory = showCategoryTypeFilter
            ? resolveComposerCategoryKey(
                node,
                currentCategory,
                initialCategoryTypeFilter !== "__all__" ? getComposerTypeFile(node, initialCategoryTypeFilter) : "",
            )
            : currentCategory;
        let lastSelectedName = currentPrompt;
        let blankPromptExplicitSelection = false;
        const findMatchingPromptName = (category, name) => {
            const normalizedName = String(name || "").trim().toLowerCase();
            if (!category || !normalizedName) return "";
            const categoryPrompts = node?.prompts?.[category];
            if (!categoryPrompts || typeof categoryPrompts !== "object") return "";
            for (const promptName of Object.keys(getCategoryPromptEntries(categoryPrompts, endpointPrefix))) {
                if (String(promptName || "").trim().toLowerCase() === normalizedName) {
                    return promptName;
                }
            }
            return "";
        };
        let currentPromptCategory = String(selectedCategory || "").trim();
        const setCurrentPromptSelection = (name, category = selectedCategory) => {
            currentPrompt = name;
            currentPromptCategory = String(category || "").trim();
            blankPromptExplicitSelection = false;
        };
        const setBlankPromptSelection = () => {
            currentPrompt = "";
            currentPromptCategory = "";
            blankPromptExplicitSelection = true;
        };
        const clearPromptBrowserSelection = async (options = {}) => {
            const {
                clearMulti = true,
                categoryKey = selectedCategory,
            } = options || {};

            setBlankPromptSelection();

            if (clearMulti) {
                selectedNames?.clear?.();
                if (multiCategorySelect && categoryKey) {
                    delete selectedByCategory[categoryKey];
                }
                multiSelectAnchorName = "";
                updateSelectButton();
            }

            if (!saveUsesEditPanel && editMode && editPanel && typeof editPanel.clearPrompt === "function") {
                await editPanel.clearPrompt({ skipConfirm: true });
            }
        };
        const getResultCategoryName = (category = selectedCategory) => showCategoryTypeFilter
            ? getComposerCategoryDisplayName(node, category)
            : String(category || "").trim();
        const getSelectionCategory = () => String(selectedCategory || "").trim();
        let editPanel = null;
        let selectedByCategory = {};
        let selectedNames;
        if (multiCategorySelect) {
            const selectedPromptsByCategory = options?.selectedPromptsByCategory;
            if (selectedPromptsByCategory && typeof selectedPromptsByCategory === "object") {
                for (const [categoryName, promptNames] of Object.entries(selectedPromptsByCategory)) {
                    const normalizedCategory = String(categoryName || "").trim();
                    if (!normalizedCategory) continue;
                    const normalizedPrompts = Array.isArray(promptNames)
                        ? promptNames.map((name) => String(name || "").trim()).filter(Boolean)
                        : [];
                    if (normalizedPrompts.length > 0) {
                        selectedByCategory[normalizedCategory] = new Set(normalizedPrompts);
                    }
                }
            }
            const initialCat = currentPromptCategory || currentCategory || "";
            if (initialCat && !selectedByCategory[initialCat]) {
                const initialPrompts = Array.isArray(options?.selectedPrompts)
                    ? options.selectedPrompts
                    : (currentPrompt ? [currentPrompt] : []);
                if (initialPrompts.length > 0) {
                    selectedByCategory[initialCat] = new Set(initialPrompts);
                }
            }
            selectedNames = selectedByCategory[selectedCategory] || new Set();
        } else {
            selectedNames = new Set(multiSelect && Array.isArray(options?.selectedPrompts)
                ? options.selectedPrompts
                : (currentPrompt ? [currentPrompt] : []));
        }
        let multiSelectAnchorName = selectedNames.size > 0 ? Array.from(selectedNames)[0] : "";

        const clearMultiSelection = () => {
            if (!supportsMultiSelect || selectedNames.size === 0) return;
            selectedNames.clear();
            if (multiCategorySelect && selectedCategory) {
                delete selectedByCategory[selectedCategory];
            }
            multiSelectAnchorName = "";
            updateSelectButton();
        };

        const setSelectedCategory = (newCategory) => {
            const normalizedNewCategory = showCategoryTypeFilter ? resolveComposerCategoryKey(node, newCategory) : newCategory;
            const previousCategory = selectedCategory;
            if (previousCategory === normalizedNewCategory) return;
            if (multiCategorySelect && previousCategory) {
                if (selectedNames.size > 0) {
                    selectedByCategory[previousCategory] = selectedNames;
                } else {
                    delete selectedByCategory[previousCategory];
                }
            }
            selectedCategory = normalizedNewCategory;
            if (multiCategorySelect) {
                selectedNames = selectedByCategory[selectedCategory] || new Set();
                multiSelectAnchorName = selectedNames.size > 0 ? Array.from(selectedNames)[0] : "";
            } else if (clearSelectionOnCategorySwitch) {
                clearMultiSelection();
            }
            if (editMode && editPanel) {
                editPanel.loadCategorySettings(selectedCategory);
                if (saveUsesEditPanel && typeof editPanel.showPromptSettings === "function") {
                    editPanel.showPromptSettings();
                } else if (typeof editPanel.showCategorySettings === "function") {
                    editPanel.showCategorySettings();
                }
            }
        };

        const canChangeEditorContext = async () => {
            if (saveUsesEditPanel) {
                return true;
            }
            if (!editMode || !editPanel || typeof editPanel.confirmDiscardChanges !== "function") {
                return true;
            }
            return await editPanel.confirmDiscardChanges();
        };

        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: ${UI.overlay};
            z-index: 9999;
        `;

        const compactBrowserPref = Boolean(app.ui.settings.getSettingValue("PromptManager.CompactPromptBrowser"));
        // Auto-detect compact mode if the viewport is too small for the full layout.
        // The full dialog needs the grid height plus header, controls, category tabs,
        // footer toolbar, and padding/margins. window.innerHeight includes ComfyUI UI
        // chrome, so we compare against the estimated total dialog height.
        const FULL_DIALOG_HEIGHT = 1280;
        const FULL_DIALOG_WIDTH = 1480;
        const compactBrowser = compactBrowserPref || (
            window.innerHeight < FULL_DIALOG_HEIGHT || window.innerWidth < FULL_DIALOG_WIDTH
        );
        const EDIT_PANEL_WIDTH = compactBrowser ? 280 : 320;
        const TYPE_RAIL_COLLAPSED_WIDTH = showCategoryTypeFilter ? (compactBrowser ? 54 : 60) : 0;
        const TYPE_RAIL_EXPANDED_WIDTH = showCategoryTypeFilter ? (compactBrowser ? 164 : 176) : 0;
        const withTypeRailBaseWidth = (layout) => {
            if (!showCategoryTypeFilter) return layout;
            return {
                ...layout,
                width: layout.width + TYPE_RAIL_COLLAPSED_WIDTH + 10,
            };
        };
        const baseNormalBrowserLayout = compactBrowser
            ? { width: 654, height: 680, cols: 5, itemWidth: 120, gap: 4, thumbWidth: 100, thumbHeight: 132 }
            : {
                width: 1400,
                height: 985,
                iconHeight: 1185,
                cols: 6,
                itemWidth: 220,
                gap: 8,
                thumbWidth: 200,
                thumbHeight: 264,
                iconCols: 11,
                iconItemWidth: 120,
                iconGap: 4,
                iconThumbWidth: 100,
                iconThumbHeight: 132,
            };
        const baseEditBrowserLayout = compactBrowser
            ? {
                ...baseNormalBrowserLayout,
                cols: 3,
                itemWidth: 110,
                gap: 4,
                thumbWidth: 100,
                thumbHeight: 132,
            }
            : {
                ...baseNormalBrowserLayout,
                cols: 6,
                gap: 4,
                itemWidth: 170,
                thumbWidth: 150,
                thumbHeight: 200,
                iconCols: 11,
                iconItemWidth: 91,
                iconGap: 4,
                iconThumbWidth: 80,
                iconThumbHeight: 106,
            };
        const normalBrowserLayout = withTypeRailBaseWidth(baseNormalBrowserLayout);
        const normalBrowserLayoutExpandedRail = showCategoryTypeFilter
            ? withTypeRailBaseWidth(compactBrowser
                ? { ...baseNormalBrowserLayout, cols: 5, itemWidth: 108, thumbWidth: 96, thumbHeight: 126, iconCols: 5, iconItemWidth: 108, iconThumbWidth: 88, iconThumbHeight: 116 }
                : { ...baseNormalBrowserLayout, cols: 6, itemWidth: 200, thumbWidth: 194, thumbHeight: 256, iconCols: 11, iconItemWidth: 104, iconThumbWidth: 86, iconThumbHeight: 114 })
            : normalBrowserLayout;
        const editBrowserLayout = withTypeRailBaseWidth(baseEditBrowserLayout);
        const editBrowserLayoutExpandedRail = showCategoryTypeFilter
            ? withTypeRailBaseWidth(compactBrowser
                ? { ...baseEditBrowserLayout, cols: 3, itemWidth: 102, thumbWidth: 98, thumbHeight: 130 }
                : { ...baseEditBrowserLayout, cols: 6, itemWidth: 152, thumbWidth: 144, thumbHeight: 192, iconCols: 11, iconItemWidth: 82, iconThumbWidth: 72, iconThumbHeight: 96 })
            : editBrowserLayout;
        let typeRailExpanded = showCategoryTypeFilter ? getTypeRailExpanded(browserPrefScope) : false;
        const DIALOG_VIEWPORT_MARGIN = compactBrowser ? 40 : 80;
        const DIALOG_HORIZONTAL_PADDING = 32;
        const DIALOG_HORIZONTAL_BORDER = 2;
        const DIALOG_HORIZONTAL_CHROME = DIALOG_HORIZONTAL_PADDING + DIALOG_HORIZONTAL_BORDER;
        const getViewportDialogWidthBudget = () => Math.max(320, window.innerWidth - DIALOG_VIEWPORT_MARGIN);
        const getViewportDialogLeftMargin = () => Math.max(20, Math.floor(DIALOG_VIEWPORT_MARGIN / 2));
        const getNormalLayout = () => (
            showCategoryTypeFilter && typeRailExpanded ? normalBrowserLayoutExpandedRail : normalBrowserLayout
        );
        const getCompressedEditLayout = () => (
            showCategoryTypeFilter && typeRailExpanded ? editBrowserLayoutExpandedRail : editBrowserLayout
        );
        function getEditPanelWidth() {
            if (compactBrowser) return 280;
            return showCategoryTypeFilter && typeRailExpanded ? 314 : 320;
        }
        const shouldExpandDialogForEditPanel = () => {
            if (!editMode) return false;
            const expandedWidth = getNormalLayout().width + getEditPanelWidth() + DIALOG_HORIZONTAL_CHROME;
            return expandedWidth <= getViewportDialogWidthBudget();
        };
        const canExpandDialogForEditPanel = () => {
            return shouldExpandDialogForEditPanel();
        };
        const getActiveBrowserLayout = () => {
            if (!editMode) {
                return getNormalLayout();
            }
            return shouldExpandDialogForEditPanel() ? getNormalLayout() : getCompressedEditLayout();
        };
        const getDialogContentWidth = () => {
            const activeLayout = getActiveBrowserLayout();
            return activeLayout.width + (shouldExpandDialogForEditPanel() ? getEditPanelWidth() : 0);
        };
        const getDialogOuterWidth = () => getDialogContentWidth() + DIALOG_HORIZONTAL_CHROME;
        const getDialogLeftPosition = () => {
            const viewportWidth = Math.max(320, window.innerWidth || 0);
            const dialogOuterWidth = getDialogOuterWidth();
            const minLeft = getViewportDialogLeftMargin();
            const maxLeft = Math.max(minLeft, viewportWidth - dialogOuterWidth - minLeft);
            const centeredLeft = Math.round((viewportWidth - dialogOuterWidth) / 2);

            if (!editMode || !shouldExpandDialogForEditPanel()) {
                return Math.min(maxLeft, Math.max(minLeft, centeredLeft));
            }

            const normalDialogOuterWidth = getNormalLayout().width + DIALOG_HORIZONTAL_CHROME;
            const anchoredLeft = Math.round((viewportWidth - normalDialogOuterWidth) / 2);
            if (anchoredLeft >= minLeft && anchoredLeft + dialogOuterWidth <= viewportWidth - minLeft) {
                return anchoredLeft;
            }

            return Math.min(maxLeft, Math.max(minLeft, centeredLeft));
        };
        let browserLayout = getActiveBrowserLayout();
        const computeMinGridWidth = () => browserLayout.cols * browserLayout.itemWidth + Math.max(0, browserLayout.cols - 1) * browserLayout.gap;

        ensurePromptBrowserCss();
        const dialog = el("div", "pm-browser");
        dialog.style.left = `${getDialogLeftPosition()}px`;
        dialog.style.width = `${getDialogContentWidth()}px`;

        // Keep native text editing context menus for input fields while preserving
        // custom right-click handling elsewhere in the browser.
        dialog.addEventListener("contextmenu", (e) => {
            const target = e.target;
            if (
                target instanceof HTMLInputElement
                || target instanceof HTMLTextAreaElement
                || target instanceof HTMLSelectElement
                || target?.isContentEditable
            ) {
                e.stopPropagation();
                return;
            }
            e.preventDefault();
        });

        // Header: the title and a close button.
        const header = el("div", "pm-head");
        const title = el("div", "title", mode === "save"
            ? saveTitle
            : (selectTitle || (workflowOnly ? "Select Recipe" : "Select Prompt")));
        const closeBtn = button("", { icon: ICONS.close, small: true, title: "Close (Esc)" });
        header.append(title, closeBtn);

        // Controls bar: search, then the filters and view switches.
        const controlsBar = el("div", "pm-controls");
        const searchInput = el("input", "fbui-text");
        searchInput.type = "text";
        searchInput.spellcheck = false;
        searchInput.placeholder = workflowOnly ? "Search recipes and prompts…" : "Search prompts…";
        const clearBtn = button("", { icon: ICONS.close, small: true, title: "Clear search", onClick: () => {
            searchInput.value = "";
            clearBtn.hidden = true;
            searchInput.focus();
            rebuildCategoryList();
            renderContent("");
        } });
        clearBtn.hidden = true;
        searchInput.addEventListener("input", () => {
            clearBtn.hidden = !searchInput.value;
        });
        const searchWrapper = el("div", "pm-search");
        const searchLead = el("span", "lead");
        searchLead.innerHTML = ICONS.search;
        searchWrapper.append(searchLead, searchInput, clearBtn);

        // Prompt/Recipe/Compose/All filter button
        let contentFilterState = initialContentFilter || getBrowserContentFilter(app);
        const contentFilterBtn = button("");

        // For prompt-only stores (e.g. Prompt Composer) the content filter has no meaning.
        if (promptOnly) {
            contentFilterState = "prompt";
        }

        // NSFW toggle button
        let hideNSFWState = options?.showNSFWOnOpen === true ? false : getHideNSFW(app);
        const nsfwBtn = button("");
        // Toolbar toggles are kit buttons: "on" while the filter or mode is active.
        const updateNsfwBtn = () => {
            nsfwBtn.textContent = hideNSFWState ? "NSFW: Hidden" : "NSFW: Visible";
            nsfwBtn.classList.toggle("on", hideNSFWState);
            nsfwBtn.title = hideNSFWState ? "NSFW content is hidden — click to show" : "NSFW content is visible — click to hide";
        };
        const CONTENT_FILTER_LABELS = {
            prompt: ["Type: Prompt", "Showing prompt entries only"],
            compose: ["Type: Compose", "Showing compose entries only"],
            layout: ["Type: Layout", "Showing layout entries only"],
            recipe: ["Type: Recipe", "Showing recipe entries only"],
            all: ["Type: All", "Showing prompts, recipes, compose and layout entries"],
        };
        const updateContentFilterBtn = () => {
            const [text, tip] = CONTENT_FILTER_LABELS[contentFilterState] ?? CONTENT_FILTER_LABELS.all;
            contentFilterBtn.textContent = text;
            contentFilterBtn.title = tip;
            contentFilterBtn.classList.toggle("on", contentFilterState !== "all" && Boolean(CONTENT_FILTER_LABELS[contentFilterState]));
        };
        const allCategoriesBtn = button("Category");
        const updateAllCategoriesBtn = () => {
            allCategoriesBtn.classList.toggle("on", showAllCategoriesState);
            allCategoriesBtn.title = showAllCategoriesState
                ? "Showing all categories, including empty ones for the current type filter"
                : "Hide categories with no entries for the current type filter";
        };
        updateContentFilterBtn();
        updateAllCategoriesBtn();
        updateNsfwBtn();
        const thumbnailModelBtn = button("");
        const updateThumbnailModelBtn = () => {
            const label = getThumbnailRenderLabelParts();
            thumbnailModelBtn.textContent = label.iconLabel || "🔧";
            thumbnailModelBtn.title = label.fullLabel;
        };
        thumbnailModelBtn.onclick = async () => {
            const picked = await showThumbnailRenderPicker(
                _thumbnailRenderFamily,
                _thumbnailRenderModel,
                _thumbnailRenderLora1,
                _thumbnailRenderLora2,
                _thumbnailRenderSteps,
            );
            if (picked) {
                saveThumbnailRenderSelection(picked);
                updateThumbnailModelBtn();
                editPanel?.refreshThumbnailModelButton?.();
            }
        };
        updateThumbnailModelBtn();

        // View mode: Grid (large), Icon (small), List; compact windows have
        // no Icon view.
        let currentViewMode = getViewMode(compactBrowser, browserPrefScope);
        const VIEW_MODES = [
            ["grid", ICONS.grid, "Large thumbnails"],
            ["icon", ICONS.tiles, "Small thumbnails"],
            ["list", ICONS.list, "List"],
        ].filter(([mode]) => !compactBrowser || mode !== "icon");
        const viewModeBtn = el("div", "pm-viewgroup");
        const viewModeButtons = new Map(VIEW_MODES.map(([mode, icon, tip]) => {
            const item = button("", { icon, title: tip, onClick: () => setBrowserViewMode(mode) });
            viewModeBtn.append(item);
            return [mode, item];
        }));
        const updateViewModeBtn = () => {
            for (const [mode, item] of viewModeButtons) item.classList.toggle("on", mode === currentViewMode);
        };
        updateViewModeBtn();

        // Multi-select toggle
        let multiSelectBtn = null;
        let enableMultiSelect = () => {};
        let updateMultiSelectBtn = () => {};
        const resetAllMultiSelections = () => {
            selectedNames.clear();
            if (multiCategorySelect) {
                Object.keys(selectedByCategory).forEach((cat) => {
                    selectedByCategory[cat].clear();
                });
            }
            multiSelectAnchorName = "";
        };
        const disableMultiSelect = () => {
            if (!multiSelectMode) return;
            multiSelectMode = false;
            resetAllMultiSelections();
            updateMultiSelectBtn();
            updateSelectButton();
            updateFooterText();
            updateSelectionToolbar();
        };
        if (supportsMultiSelect) {
            multiSelectBtn = button("Multi", { icon: ICONS.check });
            updateMultiSelectBtn = () => {
                multiSelectBtn.classList.toggle("on", multiSelectMode);
                multiSelectBtn.title = multiSelectMode ? "Multi-select is on (Shift / Ctrl-click also picks several)" : "Pick several prompts (Shift / Ctrl-click)";
            };
            multiSelectBtn.onclick = () => {
                if (multiSelectMode) {
                    disableMultiSelect();
                } else {
                    enableMultiSelect({ preserveCurrentSelection: true });
                }
                renderContent(searchInput.value);
            };
            updateMultiSelectBtn();

            enableMultiSelect = (options = null) => {
                if (multiSelectMode) return;
                multiSelectMode = true;
                resetAllMultiSelections();
                const preserveCurrentSelection = options?.preserveCurrentSelection === true;
                if (preserveCurrentSelection && currentPrompt) {
                    selectedNames.add(currentPrompt);
                    const selectionCategory = currentPromptCategory || selectedCategory;
                    if (multiCategorySelect && selectionCategory) {
                        selectedByCategory[selectionCategory] = selectedNames;
                    }
                    multiSelectAnchorName = currentPrompt;
                }
                updateMultiSelectBtn();
                updateSelectButton();
                updateFooterText();
                updateSelectionToolbar();
                updateEditModeLayout();
            };
        }

        // Edit mode toggle
        let editModeBtn = null;
        let updateEditModeBtn = () => {};
        const syncEditPanelSelection = async () => {
            if (!editPanel) return;
            if (saveUsesEditPanel) {
                editPanel.loadCategorySettings(selectedCategory);
                return;
            }
            if (currentPrompt || blankPromptExplicitSelection) {
                await editPanel.loadPrompt(selectedCategory, currentPrompt);
                return;
            }
            editPanel.loadCategorySettings(selectedCategory);
        };
        const applyBrowserLayout = () => {
            browserLayout = getActiveBrowserLayout();
            dialog.style.width = `${getDialogContentWidth()}px`;
            dialog.style.left = `${getDialogLeftPosition()}px`;
            gridContainer.style.minWidth = `${computeMinGridWidth()}px`;
            gridContainer.style.height = `${browserLayout.height}px`;
            if (editPanel) {
                editPanel.element.style.width = `${getEditPanelWidth()}px`;
            }
        };
        const updateEditModeLayout = () => {
            if (!editPanel) return;
            applyBrowserLayout();
            if (editMode) {
                editPanel.element.style.display = "flex";
                void syncEditPanelSelection();
            } else {
                editPanel.element.style.display = "none";
            }
            app.graph.setDirtyCanvas(true, true);
        };
        if (allowEditMode && mode !== "save") {
            editModeBtn = button("Edit", { icon: ICONS.brush });
            updateEditModeBtn = () => {
                editModeBtn.classList.toggle("on", editMode);
                editModeBtn.title = editMode ? "Edit mode is on — click to turn off" : "Edit prompts, categories and thumbnails";
            };
            editModeBtn.onclick = () => {
                const nextEditMode = !editMode;
                if (nextEditMode && multiSelectMode) {
                    disableMultiSelect();
                }
                editMode = nextEditMode;
                sessionBrowserEditMode.set(browserPrefScope, editMode);
                updateMultiSelectBtn();
                updateEditModeBtn();
                updateEditModeLayout();
                requestAnimationFrame(() => {
                    renderContent(searchInput.value);
                });
            };
            updateEditModeBtn();
        }

        controlsBar.appendChild(searchWrapper);
        if (!allowedCategories && !promptOnly && !hideContentFilterControl) {
            controlsBar.appendChild(contentFilterBtn);
        }
        if (!allowedCategories && !promptOnly && showAllCategoriesToggle) {
            controlsBar.appendChild(allCategoriesBtn);
        }
        controlsBar.appendChild(thumbnailModelBtn);
        controlsBar.appendChild(nsfwBtn);
        controlsBar.appendChild(viewModeBtn);
        if (multiSelectBtn) {
            controlsBar.appendChild(multiSelectBtn);
        }
        if (editModeBtn) {
            controlsBar.appendChild(editModeBtn);
        }

        // Category bar: static filter/add controls + horizontally scrollable category tabs
        const categoryBar = el("div", "pm-categories");

        // Static controls (always visible, left side)
        const categoryControls = el("div", "pm-category-controls");

        const addCategoryBtn = button("+", { title: "New category" });
        const updateCategoryCreateControls = () => {
            const showCategoryCreate = !showCategoryTypeFilter || categoryTypeFilter !== "__all__";
            addCategoryBtn.style.display = showCategoryCreate ? "" : "none";
            categoryContainerLeftBorder.style.display = showCategoryCreate ? "block" : "none";
        };
        addCategoryBtn.onclick = async () => {
            if (showCategoryTypeFilter && categoryTypeFilter === "__all__") return;
            const result = await showNewCategoryDialog();
            if (result && result.name && result.name.trim()) {
                const categoryName = result.name.trim();
                const existingCategories = showCategoryTypeFilter ? [] : Object.keys(node.prompts || {});
                const existingCategoryName = existingCategories.find(cat => cat.toLowerCase() === categoryName.toLowerCase());
                if (existingCategoryName) {
                    await showInfo("Category Exists", `Category already exists as "${existingCategoryName}".`);
                    return;
                }
                try {
                    const promptType = showCategoryTypeFilter && categoryTypeFilter !== "__all__"
                        ? categoryTypeFilter
                        : "";
                    const resp = await api.fetchApi(`${endpointPrefix}/save-category`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ category_name: categoryName, nsfw: result.nsfw, prompt_type: promptType })
                    });
                    const data = await resp.json();
                    if (data.success) {
                        applyPromptPayloadToNode(node, data);
                        const canProceed = await canChangeEditorContext();
                        if (!canProceed) {
                            rebuildCategoryList();
                            renderContent(searchInput.value);
                            return;
                        }
                        setBlankPromptSelection();
                        if (!saveUsesEditPanel && editMode && editPanel && typeof editPanel.clearPrompt === "function") {
                            await editPanel.clearPrompt({ skipConfirm: true });
                        }
                        setSelectedCategory(resolveComposerCategoryKey(node, categoryName, data.type_file || ""));
                        rebuildCategoryList();
                        renderContent(searchInput.value);
                    } else {
                        await showInfo("Error", data.error);
                    }
                } catch (err) {
                    console.error("[PromptManagerAdvanced] Error creating category:", err);
                }
            }
        };
        categoryControls.appendChild(addCategoryBtn);

        // Horizontally scrollable category tabs
        const categoryContainer = el("div", "category-scroll-bar pm-tabs");
        categoryContainer.addEventListener("wheel", (e) => {
            if (e.deltaY !== 0) {
                e.preventDefault();
                categoryContainer.scrollLeft += e.deltaY;
            }
        }, { passive: false });

        const categoryContainerLeftBorder = el("div", "pm-divider");

        categoryBar.appendChild(categoryControls);
        categoryBar.appendChild(categoryContainerLeftBorder);
        categoryBar.appendChild(categoryContainer);

        let allCategories = [];
        let categoryTypeFilter = initialCategoryTypeFilter;
        let categories = [];
        let selectedSaveName = initialSaveName;
        let categoryButtons = [];
        let editModeLastClickPrompt = "";
        let editModeLastClickAt = 0;
        let promptTypeFilters = [];

        const buildPromptTypeFilters = () => {
            let choices = [];
            if (showCategoryTypeFilter) {
                const canonicalEntries = getOrderedComposerTypeEntries(node);
                if (canonicalEntries.length) {
                    choices = canonicalEntries.map(([typeFile, typeData]) => {
                        const normalizedValue = String(typeFile || "").replace(/\.json$/i, "").trim().toLowerCase();
                        return {
                            value: normalizedValue,
                            label: String(typeData?.name || "").trim() || normalizedValue,
                            typeFile: String(typeFile || ""),
                            iconUrl: String(typeData?.icon || typeData?.icon_url || "").trim() || getComposerTypeIconValue(node, normalizedValue),
                        };
                    });
                } else {
                    choices = getPromptTypeChoices(node?.prompts)
                        .filter((choice) => String(choice?.value || "").trim())
                        .map((choice) => {
                            const normalizedValue = String(choice.value || "").trim().toLowerCase();
                            return {
                                value: normalizedValue,
                                label: getComposerTypeDisplayName(node, normalizedValue) || choice.label || choice.value,
                                typeFile: getComposerTypeFile(node, normalizedValue),
                                iconUrl: getComposerTypeIconValue(node, normalizedValue),
                            };
                        });
                }
            }

            return [
                { value: "__all__", label: "All Types", typeFile: "", iconUrl: getComposerTypeIconValue(node, "__all__") },
                ...choices,
            ];
        };

        const refreshPromptTypeFilters = () => {
            promptTypeFilters = buildPromptTypeFilters().filter((choice, index, array) => (
                index === 0 || array.findIndex((candidate) => candidate.value === choice.value) === index
            ));
            const validCategoryTypeValues = new Set(promptTypeFilters.map((choice) => choice.value));
            categoryTypeFilter = validCategoryTypeValues.has(String(categoryTypeFilter || "").trim().toLowerCase())
                ? (String(categoryTypeFilter || "__all__").trim().toLowerCase() || "__all__")
                : "__all__";
            if (hideNSFWState && categoryTypeFilter !== "__all__" && isComposerTypeNSFW(node, categoryTypeFilter)) {
                categoryTypeFilter = "__all__";
            }
        };

        const getComposerTypeFilterForCategory = (categoryName, explicitTypeFile = "") => {
            if (!showCategoryTypeFilter) return "";
            const normalizedExplicit = String(explicitTypeFile || "").replace(/\.json$/i, "").trim().toLowerCase();
            if (normalizedExplicit) return normalizedExplicit;
            const flattenedTypeFile = String(node?.prompts?.[categoryName]?._type_file_ || "").replace(/\.json$/i, "").trim().toLowerCase();
            if (flattenedTypeFile) return flattenedTypeFile;
            return String(getCategoryPromptType(node, categoryName) || "").trim().toLowerCase();
        };

        const syncComposerTypeFilterForCategory = (categoryName, explicitTypeFile = "") => {
            const nextType = getComposerTypeFilterForCategory(categoryName, explicitTypeFile);
            if (!showCategoryTypeFilter || !nextType || nextType === "__all__" || categoryTypeFilter === nextType) {
                return false;
            }
            categoryTypeFilter = nextType;
            updateTypeRailButtons();
            return true;
        };

        refreshPromptTypeFilters();
        updateCategoryCreateControls();

        const isCategoryNSFW = (cat) => {
            return node.prompts?.[cat]?.["__meta__"]?.nsfw === true;
        };

        const applyCategoryTypeFilter = () => {
            refreshPromptTypeFilters();
            if (categoryTypeFilter === "__all__") {
                categories = [...allCategories];
            } else {
                categories = allCategories.filter(cat => String(getCategoryPromptType(node, cat) || "").trim().toLowerCase() === categoryTypeFilter);
            }
        };

        const categoryHasSearchMatch = (category, filter = "") => {
            const normalizedFilter = String(filter || "").trim().toLowerCase();
            if (!normalizedFilter) return true;
            const promptNames = getPromptNamesForCategory(node, category, {
                hideNSFW: hideNSFWState,
                workflowOnly,
                contentFilter: contentFilterState,
                endpointPrefix,
            });
            return promptNames.some((name) => String(name || "").toLowerCase().includes(normalizedFilter));
        };

        const applyCategorySearchFilter = (filter = "") => {
            categories = categories.filter((cat) => categoryHasSearchMatch(cat, filter));
        };

        const ensureSelectedCategory = () => {
            const resolveVisibleCategories = () => filterAllowedCategories(getVisibleCategories(node, {
                hideNSFW: hideNSFWState,
                workflowOnly,
                contentFilter: contentFilterState,
                filterEmptyCategories: filterEmptyCategories && !showAllCategoriesState,
                endpointPrefix,
            }));

            allCategories = resolveVisibleCategories();
            if (
                allCategories.length === 0
                && showAllCategoriesToggle
                && filterEmptyCategories
                && !showAllCategoriesState
            ) {
                showAllCategoriesState = true;
                updateAllCategoriesBtn();
                allCategories = resolveVisibleCategories();
            }
            applyCategoryTypeFilter();
            applyCategorySearchFilter(searchInput?.value || "");

            if (!Array.isArray(categories) || categories.length === 0) {
                setSelectedCategory("");
                return "";
            }

            let newCategory = selectedCategory;
            if (!newCategory || !categories.includes(newCategory)) {
                newCategory = categories[0];
            }

            if (hideNSFWState && isCategoryNSFW(newCategory)) {
                newCategory = categories.find(c => !isCategoryNSFW(c)) || "";
            }

            setSelectedCategory(newCategory);
            return selectedCategory;
        };

        let typeRail = null;
        let typeRailToggle = null;
        let typeRailList = null;
        let typeRailAddButton = null;
        let typeRailButtons = [];
        let typeRailTooltip = null;
        let draggingTypeFile = "";
        let dragHoverTypeFile = "";
        let dragHoverPosition = "";
        let suppressTypeRailClickUntil = 0;

        const ensureTypeRailTooltip = () => {
            if (typeRailTooltip) return typeRailTooltip;
            typeRailTooltip = document.createElement("div");
            typeRailTooltip.style.cssText = `
                position: fixed;
                display: none;
                background: ${T.panel};
                border: 1px solid ${T.plateEdge};
                border-radius: ${RADIUS}px;
                color: ${T.inkSoft};
                font: 12px/1.2 ${FONT};
                padding: 7px 10px;
                box-shadow: 0 10px 28px rgba(0,0,0,0.55);
                z-index: ${Z_MENU};
                pointer-events: none;
                white-space: nowrap;
            `;
            document.body.appendChild(typeRailTooltip);
            return typeRailTooltip;
        };

        const moveTypeRailTooltip = (x, y) => {
            if (!typeRailTooltip || typeRailTooltip.style.display === "none") return;
            const margin = 12;
            const width = typeRailTooltip.offsetWidth || 120;
            const height = typeRailTooltip.offsetHeight || 34;
            let left = x + margin;
            let top = y + margin;
            if (left + width > window.innerWidth - 8) {
                left = Math.max(8, x - width - margin);
            }
            if (top + height > window.innerHeight - 8) {
                top = Math.max(8, y - height - margin);
            }
            typeRailTooltip.style.left = `${left}px`;
            typeRailTooltip.style.top = `${top}px`;
        };

        const showTypeRailTooltip = (text, x, y) => {
            const tip = ensureTypeRailTooltip();
            tip.textContent = text;
            tip.style.display = "block";
            moveTypeRailTooltip(x, y);
        };

        const hideTypeRailTooltip = () => {
            if (typeRailTooltip) {
                typeRailTooltip.style.display = "none";
            }
        };

        const clearTypeRailDropIndicators = () => {
            for (const btn of typeRailButtons) {
                btn.style.boxShadow = "none";
                btn.style.outline = "none";
                btn.style.outlineOffset = "0";
            }
            dragHoverTypeFile = "";
            dragHoverPosition = "";
        };

        const getVisibleTypeRailButtons = () => typeRailButtons.filter((btn) => btn.style.display !== "none");

        const updateTypeRailDropIndicators = () => {
            for (const btn of typeRailButtons) {
                btn.style.boxShadow = "none";
                btn.style.outline = "none";
                btn.style.outlineOffset = "0";
            }
            if (!dragHoverTypeFile) return;

            const visibleButtons = getVisibleTypeRailButtons();
            const targetIndex = visibleButtons.findIndex((btn) => btn.dataset.typeFile === dragHoverTypeFile);
            if (targetIndex < 0) return;

            const targetBtn = visibleButtons[targetIndex];
            const previousBtn = targetIndex > 0 ? visibleButtons[targetIndex - 1] : null;
            const nextBtn = targetIndex < visibleButtons.length - 1 ? visibleButtons[targetIndex + 1] : null;
            const highlight = T.brandInk;
            const glow = T.brandSoftHover;

            if (dragHoverPosition === "before") {
                targetBtn.style.boxShadow = `inset 0 4px 0 ${highlight}, 0 0 0 1px ${glow}`;
                targetBtn.style.outline = `2px solid ${glow}`;
                targetBtn.style.outlineOffset = "1px";
                if (previousBtn) {
                    previousBtn.style.boxShadow = `inset 0 -4px 0 ${highlight}`;
                }
                return;
            }

            targetBtn.style.boxShadow = `inset 0 -4px 0 ${highlight}, 0 0 0 1px ${glow}`;
            targetBtn.style.outline = `2px solid ${glow}`;
            targetBtn.style.outlineOffset = "1px";
            if (nextBtn) {
                nextBtn.style.boxShadow = `inset 0 4px 0 ${highlight}`;
            }
        };

        const getTypeRailDropPosition = (btn, clientY) => {
            const rect = btn.getBoundingClientRect();
            return clientY >= rect.top + (rect.height / 2) ? "after" : "before";
        };

        const commitTypeReorder = async (sourceTypeFile, targetTypeFile, position) => {
            const normalizedSource = String(sourceTypeFile || "").trim();
            const normalizedTarget = String(targetTypeFile || "").trim();
            if (!normalizedSource || !normalizedTarget || normalizedSource === normalizedTarget) return;
            try {
                const resp = await api.fetchApi(`${endpointPrefix}/reorder-types`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        source_type_file: normalizedSource,
                        target_type_file: normalizedTarget,
                        position: position === "after" ? "after" : "before",
                    }),
                });
                const data = await resp.json();
                if (data.success) {
                    applyPromptPayloadToNode(node, data);
                    rebuildCategoryList();
                    rebuildTypeRailButtons();
                    renderContent(searchInput.value);
                } else {
                    await showInfo("Error", data.error || "Failed to reorder prompt groups.");
                }
            } catch (err) {
                console.error("[PromptBrowser] Error reordering prompt groups:", err);
            }
        };

        const updateTypeRailButtons = () => {
            if (!typeRailButtons.length) return;
            const visibleButtons = getVisibleTypeRailButtons();
            const dropHighlightTypeFiles = new Set();
            if (dragHoverTypeFile) {
                const targetIndex = visibleButtons.findIndex((btn) => btn.dataset.typeFile === dragHoverTypeFile);
                if (targetIndex >= 0) {
                    const targetBtn = visibleButtons[targetIndex];
                    const adjacentBtn = dragHoverPosition === "before"
                        ? visibleButtons[targetIndex - 1] || null
                        : visibleButtons[targetIndex + 1] || null;
                    if (targetBtn?.dataset.typeFile) {
                        dropHighlightTypeFiles.add(targetBtn.dataset.typeFile);
                    }
                    if (adjacentBtn?.dataset.typeFile) {
                        dropHighlightTypeFiles.add(adjacentBtn.dataset.typeFile);
                    }
                }
            }
            for (const btn of typeRailButtons) {
                const isSelected = btn.dataset.typeValue === categoryTypeFilter;
                const isHidden = hideNSFWState && btn.dataset.typeValue !== "__all__" && isComposerTypeNSFW(node, btn.dataset.typeValue);
                const isNSFW = btn.dataset.typeValue !== "__all__" && isComposerTypeNSFW(node, btn.dataset.typeValue);
                const isDraggingSelf = !!draggingTypeFile && btn.dataset.typeFile === draggingTypeFile;
                const isDraggingOther = !!draggingTypeFile && !isDraggingSelf;
                const isDropHighlight = !!dragHoverTypeFile && dropHighlightTypeFiles.has(btn.dataset.typeFile || "");
                const labelEl = btn.querySelector(".pm-type-rail-label");
                const iconEl = btn.querySelector(".pm-type-rail-icon");
                btn.style.display = isHidden ? "none" : "flex";
                btn.style.background = isSelected ? T.brandSoftHover : "transparent";
                if (isNSFW && !isSelected) {
                    btn.style.borderColor = T.danger;
                } else {
                    btn.style.borderColor = isSelected ? T.brand : T.plateEdge;
                }
                btn.style.color = isSelected ? T.ink : T.label;
                btn.style.justifyContent = typeRailExpanded ? "flex-start" : "center";
                btn.style.padding = typeRailExpanded ? "3px 8px" : "2px";
                btn.title = "";
                if (labelEl) {
                    labelEl.style.display = typeRailExpanded ? "block" : "none";
                }
                if (iconEl) {
                    iconEl.style.width = typeRailExpanded ? "32px" : "38px";
                    iconEl.style.height = typeRailExpanded ? "32px" : "38px";
                    iconEl.style.minWidth = typeRailExpanded ? "32px" : "38px";
                }
                btn.style.opacity = isDraggingOther && !isDropHighlight ? "0.28" : "1";
                btn.style.filter = isDraggingOther && !isDropHighlight ? "brightness(0.6) saturate(0.7)" : "none";
                btn.style.transform = isDraggingSelf ? "scale(1.04)" : "none";
                btn.style.zIndex = isDraggingSelf ? "2" : "0";
                if (isDraggingSelf) {
                    btn.style.background = T.brandSoftHover;
                    btn.style.borderColor = T.brand;
                    btn.style.color = T.ink;
                } else if (isDropHighlight) {
                    btn.style.background = isSelected ? T.brandSoftHover : T.brandSoft;
                }
                btn.style.boxShadow = "none";
                btn.style.outline = "none";
                btn.style.outlineOffset = "0";
            }
            if (typeRail) {
                typeRail.style.width = `${typeRailExpanded ? TYPE_RAIL_EXPANDED_WIDTH : TYPE_RAIL_COLLAPSED_WIDTH}px`;
            }
            if (typeRailToggle) {
                typeRailToggle.textContent = typeRailExpanded ? "‹" : "›";
                typeRailToggle.title = typeRailExpanded ? "Collapse type filters" : "Expand type filters";
            }
            updateTypeRailDropIndicators();
        };

        const rebuildTypeRailButtons = () => {
            if (!typeRailList) return;
            refreshPromptTypeFilters();
            hideTypeRailTooltip();
            typeRailList.replaceChildren();
            typeRailButtons = promptTypeFilters.map((choice) => {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.dataset.typeValue = choice.value;
                btn.dataset.typeLabel = choice.label;
                btn.dataset.typeFile = choice.typeFile || "";
                btn.style.cssText = `
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    width: 100%;
                    min-height: 42px;
                    padding: 2px;
                    border-radius: ${RADIUS_CARD}px;
                    border: 1px solid ${T.plateEdge};
                    background: transparent;
                    color: ${T.label};
                    cursor: pointer;
                    font-size: 13px;
                    white-space: nowrap;
                    overflow: hidden;
                    box-sizing: border-box;
                    transition: background 0.15s ease, border-color 0.15s ease;
                `;

                const iconEl = document.createElement("img");
                iconEl.className = "pm-type-rail-icon";
                iconEl.src = choice.iconUrl;
                iconEl.alt = choice.label;
                iconEl.onerror = () => {
                    iconEl.onerror = null;
                    iconEl.src = getComposerTypeIconValue(node, choice.value);
                };
                iconEl.style.cssText = `
                    width: 38px;
                    height: 38px;
                    min-width: 38px;
                    object-fit: cover;
                    border-radius: 7px;
                    display: block;
                `;

                const labelEl = document.createElement("span");
                labelEl.className = "pm-type-rail-label";
                labelEl.textContent = choice.label;
                labelEl.style.cssText = `
                    display: none;
                    overflow: hidden;
                    text-overflow: ellipsis;
                `;

                btn.onmouseover = (evt) => {
                    if (btn.dataset.typeValue !== categoryTypeFilter) {
                        btn.style.background = "rgba(255,255,255,0.04)";
                    }
                    if (!typeRailExpanded) {
                        showTypeRailTooltip(choice.label, evt.clientX, evt.clientY);
                    }
                };
                btn.onmousemove = (evt) => {
                    if (!typeRailExpanded) {
                        showTypeRailTooltip(choice.label, evt.clientX, evt.clientY);
                    }
                };
                btn.onmouseout = () => {
                    hideTypeRailTooltip();
                    updateTypeRailButtons();
                };
                btn.onclick = async () => {
                    if (Date.now() < suppressTypeRailClickUntil) return;
                    const canProceed = await canChangeEditorContext();
                    if (!canProceed) return;
                    hideTypeRailTooltip();
                    setBlankPromptSelection();
                    if (editMode && editPanel && typeof editPanel.clearPrompt === "function") {
                        await editPanel.clearPrompt({ skipConfirm: true });
                    }
                    categoryTypeFilter = choice.value;
                    rebuildCategoryList();
                    if (editMode && editPanel) {
                        if (choice.value === "__all__") {
                            if (typeof editPanel.loadCategorySettings === "function") {
                                editPanel.loadCategorySettings(selectedCategory);
                            }
                            if (typeof editPanel.showCategorySettings === "function") {
                                editPanel.showCategorySettings();
                            }
                        } else {
                            if (typeof editPanel.loadTypeSettings === "function") {
                                editPanel.loadTypeSettings(choice.value);
                            }
                            if (typeof editPanel.showTypeSettings === "function") {
                                editPanel.showTypeSettings();
                            }
                        }
                    }
                    updateTypeRailButtons();
                    renderContent(searchInput.value);
                };
                if (choice.value !== "__all__") {
                    btn.draggable = true;
                    btn.style.cursor = "grab";
                    btn.addEventListener("dragstart", (evt) => {
                        draggingTypeFile = btn.dataset.typeFile || "";
                        dragHoverTypeFile = "";
                        dragHoverPosition = "";
                        if (evt.dataTransfer) {
                            evt.dataTransfer.effectAllowed = "move";
                            evt.dataTransfer.setData("text/plain", draggingTypeFile);
                        }
                        updateTypeRailButtons();
                    });
                    btn.addEventListener("dragend", () => {
                        draggingTypeFile = "";
                        clearTypeRailDropIndicators();
                        updateTypeRailButtons();
                    });
                    btn.addEventListener("dragover", (evt) => {
                        if (!draggingTypeFile || draggingTypeFile === btn.dataset.typeFile) return;
                        evt.preventDefault();
                        dragHoverTypeFile = btn.dataset.typeFile || "";
                        dragHoverPosition = getTypeRailDropPosition(btn, evt.clientY);
                        if (evt.dataTransfer) {
                            evt.dataTransfer.dropEffect = "move";
                        }
                        updateTypeRailButtons();
                    });
                    btn.addEventListener("drop", async (evt) => {
                        if (!draggingTypeFile || draggingTypeFile === btn.dataset.typeFile) return;
                        evt.preventDefault();
                        const targetTypeFile = btn.dataset.typeFile || "";
                        const dropPosition = getTypeRailDropPosition(btn, evt.clientY);
                        suppressTypeRailClickUntil = Date.now() + 250;
                        clearTypeRailDropIndicators();
                        await commitTypeReorder(draggingTypeFile, targetTypeFile, dropPosition);
                    });
                    btn.oncontextmenu = (evt) => {
                        evt.preventDefault();
                        evt.stopPropagation();
                        showTypeContextMenu(evt, choice.value);
                    };
                }

                btn.appendChild(iconEl);
                btn.appendChild(labelEl);
                typeRailList.appendChild(btn);
                return btn;
            });

            updateTypeRailButtons();
            typeRailList.getBoundingClientRect();
            typeRail?.getBoundingClientRect();
        };

        const renderTypeRail = () => {
            if (!showCategoryTypeFilter) return null;

            const rail = document.createElement("div");
            rail.style.cssText = `
                display: flex;
                flex-direction: column;
                flex: 0 0 auto;
                width: ${TYPE_RAIL_COLLAPSED_WIDTH}px;
                min-width: ${TYPE_RAIL_COLLAPSED_WIDTH}px;
                max-width: ${TYPE_RAIL_EXPANDED_WIDTH}px;
                margin-right: 10px;
                border-right: 1px solid ${UI.sectionBorder};
                padding-right: 8px;
                overflow: hidden;
                transition: width 0.16s ease;
            `;

            const toggleBtn = document.createElement("button");
            toggleBtn.type = "button";
            toggleBtn.style.cssText = `
                width: 100%;
                height: 34px;
                border-radius: ${RADIUS}px;
                border: none;
                background: ${T.fill};
                color: ${T.label};
                cursor: pointer;
                font-size: 14px;
                margin-bottom: 8px;
                flex-shrink: 0;
            `;

            typeRailList = document.createElement("div");
            typeRailList.className = "pm-type-rail-list";
            typeRailList.style.cssText = `
                display: flex;
                flex-direction: column;
                gap: 4px;
                overflow-y: auto;
                min-height: 0;
                padding-right: 2px;
                scrollbar-width: thin;
            `;

            const addTypeBtn = document.createElement("button");
            addTypeBtn.type = "button";
            addTypeBtn.style.cssText = `
                display: flex;
                align-items: center;
                justify-content: center;
                align-self: center;
                width: 34px;
                height: 34px;
                min-width: 34px;
                min-height: 34px;
                padding: 0;
                border-radius: ${RADIUS}px;
                border: none;
                background: ${T.fill};
                color: ${T.label};
                cursor: pointer;
                font-size: 13px;
                white-space: nowrap;
                overflow: hidden;
                box-sizing: border-box;
                transition: background 0.15s ease;
                margin-top: 6px;
                flex-shrink: 0;
            `;
            addTypeBtn.title = "New Prompt Group";

            const addTypeIcon = document.createElement("div");
            addTypeIcon.className = "pm-type-rail-icon";
            addTypeIcon.textContent = "+";
            addTypeIcon.style.cssText = `
                width: auto;
                min-width: 0;
                height: auto;
                border-radius: 0;
                display: block;
                background: transparent;
                color: inherit;
                font-size: 18px;
                line-height: 1;
            `;

            const addTypeLabel = document.createElement("span");
            addTypeLabel.className = "pm-type-rail-label";
            addTypeLabel.textContent = "New Prompt Group";
            addTypeLabel.style.cssText = `
                display: none;
                overflow: hidden;
                text-overflow: ellipsis;
            `;

            addTypeBtn.onmouseover = (evt) => {
                addTypeBtn.style.background = T.fillHover;
                addTypeBtn.style.color = T.ink;
                if (!typeRailExpanded) {
                    showTypeRailTooltip("New Prompt Group", evt.clientX, evt.clientY);
                }
            };
            addTypeBtn.onmousemove = (evt) => {
                if (!typeRailExpanded) {
                    showTypeRailTooltip("New Prompt Group", evt.clientX, evt.clientY);
                }
            };
            addTypeBtn.onmouseout = () => {
                hideTypeRailTooltip();
                addTypeBtn.style.background = T.fill;
                addTypeBtn.style.color = T.label;
            };
            addTypeBtn.onclick = async () => {
                const canProceed = await canChangeEditorContext();
                if (!canProceed) return;
                hideTypeRailTooltip();
                const result = await showNewPromptGroupDialog();
                if (result === null) return;
                const groupName = String(result?.groupName || "").trim();
                const categoryName = String(result?.categoryName || "").trim();
                if (!groupName) return;
                if (!categoryName) {
                    await showInfo("Missing Category", "Please enter a starting category for the new prompt group.");
                    return;
                }

                const existingChoice = promptTypeFilters.find((choice) => (
                    choice.value !== "__all__"
                    && (
                        String(choice.label || "").trim().toLowerCase() === groupName.toLowerCase()
                        || String(choice.value || "").trim().toLowerCase() === groupName.toLowerCase()
                    )
                ));
                if (existingChoice) {
                    await showInfo("Prompt Group Exists", `Prompt group already exists as "${existingChoice.label || groupName}".`);
                    return;
                }

                try {
                    const resp = await api.fetchApi(`${endpointPrefix}/save-type-settings`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            prompt_type: groupName,
                            type_name: groupName,
                            initial_category_name: categoryName,
                        }),
                    });
                    const data = await resp.json();
                    if (data.success) {
                        applyPromptPayloadToNode(node, data);
                        node.composerPrompts = data.prompts;
                        categoryTypeFilter = String(data.type_file || groupName).replace(/\.json$/i, "").trim().toLowerCase() || "__all__";
                        setBlankPromptSelection();
                        const nextCategoryKey = resolveComposerCategoryKey(node, categoryName, data.type_file || "");
                        setSelectedCategory(nextCategoryKey);
                        rebuildCategoryList();
                        rebuildTypeRailButtons();
                        if (editMode && editPanel) {
                            if (!saveUsesEditPanel && typeof editPanel.clearPrompt === "function") {
                                await editPanel.clearPrompt({ skipConfirm: true });
                            }
                            if (typeof editPanel.loadCategorySettings === "function") {
                                editPanel.loadCategorySettings(nextCategoryKey);
                            }
                            if (typeof editPanel.showCategorySettings === "function") {
                                editPanel.showCategorySettings();
                            }
                        }
                        renderContent(searchInput.value);
                    } else {
                        await showInfo("Error", data.error || "Failed to create prompt group.");
                    }
                } catch (err) {
                    console.error("[PromptBrowser] Error creating prompt group:", err);
                }
            };
            addTypeBtn.append(addTypeIcon, addTypeLabel);

            toggleBtn.onclick = () => {
                typeRailExpanded = !typeRailExpanded;
                setTypeRailExpanded(typeRailExpanded, browserPrefScope);
                localStorage.setItem(getTypeRailExpandedStorageKey(browserPrefScope), String(typeRailExpanded));
                applyBrowserLayout();
                updateTypeRailButtons();
                renderContent(searchInput.value);
            };

            rail.appendChild(toggleBtn);
            rail.appendChild(typeRailList);
            rail.appendChild(addTypeBtn);

            typeRail = rail;
            typeRailToggle = toggleBtn;
            typeRailAddButton = addTypeBtn;
            rebuildTypeRailButtons();
            return rail;
        };

        ensureSelectedCategory();
        const updateCategoryButtons = () => {
            ensureSelectedCategory();

            let selectedBtn = null;
            categoryButtons.forEach(btn => {
                const cat = btn.dataset.category;
                const isSelected = cat === selectedCategory;
                const isNSFW = isCategoryNSFW(cat);

                // Hide NSFW categories when filter is active
                if (hideNSFWState && isNSFW) {
                    btn.style.display = "none";
                    return;
                }
                btn.style.display = "";

                btn.classList.toggle("on", isSelected);
                btn.classList.toggle("nsfw", isNSFW);

                if (isSelected) selectedBtn = btn;
            });

            // Keep the active category visible in the scroll bar
            if (selectedBtn) {
                selectedBtn.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
            }

            // If selected category is now hidden/empty under current filters, switch.
            ensureSelectedCategory();
        };

        // Category context menu for NSFW toggle
        const showCategoryContextMenu = (event, cat) => {
            const isComposerManager = endpointPrefix === "/prompt-manager/compose";
            const isNSFW = isCategoryNSFW(cat);
            const categoryLabel = showCategoryTypeFilter ? getComposerCategoryDisplayName(node, cat) : cat;

            const runBatchGeneration = async (names, title, options = {}) => {
                const regenerateAll = options.regenerateAll === true;
                if (names.length === 0) {
                    await showInfo(
                        regenerateAll ? "No Prompts to Re-Generate" : "No Thumbnails to Generate",
                        regenerateAll
                            ? `No prompts were found in "${categoryLabel}" to re-generate thumbnails for.`
                            : `All prompts in "${categoryLabel}" already have thumbnails.`
                    );
                    return;
                }

                const confirmMessage = regenerateAll
                    ? `Re-generate thumbnails for ${names.length} prompt(s) in "${categoryLabel}"? Existing thumbnails will be replaced.`
                    : `Generate thumbnails for ${names.length} prompt(s) in "${categoryLabel}"?`;
                const confirmText = regenerateAll ? "Re-Generate" : "Generate";
                if (!await showConfirm(title, confirmMessage, confirmText, { danger: false })) {
                    return;
                }

                // Ensure renderer selection is ready for prompts without saved workflow_data.
                const renderSelection = await ensureThumbnailRenderSelection();
                if (!renderSelection) return;

                // Resolve family-compatible defaults once for the batch.
                const fallbackBase = await resolveThumbnailFallbackBase(renderSelection);

                for (const pName of names) {
                    queueThumbnail(pName, () => _generateThumbnailForBrowserCategory(node, cat, pName, () => {
                        renderContent(searchInput.value);
                    }, { renderSelection, fallbackBase, endpointPrefix }));
                }
            };

            openMenu(event.clientX, event.clientY, [
                { label: "Generate missing thumbnails", action: async () => {
                const catPrompts = node.prompts[cat];
                if (!catPrompts) return;

                // Collect prompts without thumbnails
                const missing = getPromptNamesForCategory(node, cat, { hideNSFW: false, workflowOnly: false, contentFilter: "all", endpointPrefix }).filter(name => {
                    const data = getCategoryPromptEntry(catPrompts, name, endpointPrefix);
                    return data && typeof data === "object" && !data.thumbnail;
                });

                await runBatchGeneration(missing, "Generate Missing Thumbnails");
                } },
                { label: "Re-generate all thumbnails", action: async () => {
                const catPrompts = node.prompts[cat];
                if (!catPrompts) return;

                const all = getPromptNamesForCategory(node, cat, { hideNSFW: false, workflowOnly: false, contentFilter: "all", endpointPrefix }).filter(name => {
                    const data = getCategoryPromptEntry(catPrompts, name, endpointPrefix);
                    return data && typeof data === "object";
                });

                await runBatchGeneration(all, "Re-Generate All Thumbnails", { regenerateAll: true });
                } },
                null,
                { label: isNSFW ? "✓ NSFW" : "Mark as NSFW", danger: isNSFW, action: async () => {
                try {
                    const resp = await api.fetchApi(`${endpointPrefix}/toggle-nsfw`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            type: "category",
                            category: categoryLabel,
                            ...(isComposerManager ? { type_file: getComposerCategoryTypeFile(node, cat) } : {}),
                        })
                    });
                    const result = await resp.json();
                    if (result.success) {
                        applyPromptPayloadToNode(node, result);
                        updateCategoryButtons();
                        renderContent(searchInput.value);
                    }
                } catch (err) {
                    console.error("[PromptManagerAdvanced] Error toggling category NSFW:", err);
                }
                } },
                null,
                { label: isComposerManager ? "Rename / Move…" : "Rename…", action: async () => {
                const currentTypeFile = isComposerManager ? getComposerCategoryTypeFile(node, cat) : "";
                const composerMoveTargets = isComposerManager ? getComposerMoveTargetOptions(node) : null;
                const result = await showRenameCategoryDialog(
                    isComposerManager ? "Rename / Move Category" : "Rename Category",
                    "Enter new category name:",
                    [categoryLabel],
                    categoryLabel,
                    isComposerManager ? {
                        groupOptions: composerMoveTargets?.groupOptions || [],
                        defaultGroupValue: currentTypeFile,
                    } : {}
                );
                if (result && result.newCategory && result.newCategory.trim()) {
                    const newCat = result.newCategory.trim();
                    const newTypeFile = isComposerManager ? String(result.typeFile || currentTypeFile || "") : "";
                    if (newCat === cat && (!isComposerManager || newTypeFile === currentTypeFile)) return;
                    try {
                        const resp = await api.fetchApi(`${endpointPrefix}/rename-category`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                old_category: categoryLabel,
                                new_category: newCat,
                                ...(isComposerManager ? {
                                    type_file: currentTypeFile,
                                    new_type_file: newTypeFile || currentTypeFile,
                                } : {}),
                            })
                        });
                        const data = await resp.json();
                        if (data.success) {
                            applyPromptPayloadToNode(node, data);
                            if (selectedCategory === cat) {
                                const nextCategoryKey = resolveComposerCategoryKey(node, data.new_category, data.type_file || newTypeFile || currentTypeFile);
                                syncComposerTypeFilterForCategory(nextCategoryKey, data.type_file || newTypeFile || currentTypeFile);
                                if (multiCategorySelect && selectedByCategory[cat]) {
                                    selectedByCategory[nextCategoryKey] = selectedByCategory[cat];
                                    delete selectedByCategory[cat];
                                }
                                setSelectedCategory(nextCategoryKey);
                            }
                            rebuildCategoryList();
                            renderContent(searchInput.value);
                        } else {
                            await showInfo("Error", data.error);
                        }
                    } catch (err) {
                        console.error("[PromptManagerAdvanced] Error renaming category:", err);
                    }
                }
                } },
                null,
                { label: "Delete category", danger: true, action: async () => {
                if (await showConfirm("Delete Category", `Are you sure you want to delete category "${categoryLabel}" and all its prompts?`)) {
                    try {
                        const resp = await api.fetchApi(`${endpointPrefix}/delete-category`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                category: categoryLabel,
                                ...(isComposerManager ? { type_file: getComposerCategoryTypeFile(node, cat) } : {}),
                            })
                        });
                        const data = await resp.json();
                        if (data.success) {
                            applyPromptPayloadToNode(node, data);
                            await clearPromptBrowserSelection({ categoryKey: cat });
                            if (selectedCategory === cat) {
                                const cats = Object.keys(node.prompts).filter(c => c !== "__meta__");
                                setSelectedCategory(cats[0] || "");
                            }
                            rebuildCategoryList();
                            renderContent(searchInput.value);
                        } else {
                            await showInfo("Error", data.error);
                        }
                    } catch (err) {
                        console.error("[PromptManagerAdvanced] Error deleting category:", err);
                    }
                }
                } },
            ]);
        };

        const showTypeContextMenu = (event, typeValue) => {
            if (!showCategoryTypeFilter || !typeValue || typeValue === "__all__") return;

            const typeFile = getComposerTypeFile(node, typeValue);
            const typeLabel = getComposerTypeDisplayName(node, typeValue) || typeValue;
            const items = [];

            const createMenuItem = (label, onClick, danger = false) => ({ label, action: onClick, danger });

            const typeRefs = () => getCategoriesForComposerType(node, typeValue)
                .flatMap((category) => getPromptNamesForCategory(node, category, {
                    hideNSFW: false,
                    workflowOnly: false,
                    contentFilter: "all",
                    endpointPrefix,
                }).map((name) => ({ category, name })));

            const runTypeBatchGeneration = async (title, options = {}) => {
                const refs = typeRefs().filter((ref) => {
                    const data = getCategoryPromptEntry(node?.prompts?.[ref.category], ref.name, endpointPrefix);
                    if (!data || typeof data !== "object") return false;
                    return options.regenerateAll === true ? true : !data.thumbnail;
                });
                if (!refs.length) {
                    await showInfo(
                        options.regenerateAll ? "No Prompts to Re-Generate" : "No Thumbnails to Generate",
                        options.regenerateAll
                            ? `No prompts were found in "${typeLabel}" to re-generate thumbnails for.`
                            : `All prompts in "${typeLabel}" already have thumbnails.`
                    );
                    return;
                }

                const confirmed = await showConfirm(
                    title,
                    options.regenerateAll
                        ? `Re-generate thumbnails for ${refs.length} prompt(s) in "${typeLabel}"? Existing thumbnails will be replaced.`
                        : `Generate thumbnails for ${refs.length} prompt(s) in "${typeLabel}"?`,
                    options.regenerateAll ? "Re-Generate" : "Generate",
                    { danger: false }
                );
                if (!confirmed) return;

                const renderSelection = await ensureThumbnailRenderSelection();
                if (!renderSelection) return;
                const fallbackBase = await resolveThumbnailFallbackBase(renderSelection);

                for (const ref of refs) {
                    queueThumbnail(ref.name, () => _generateThumbnailForBrowserCategory(node, ref.category, ref.name, () => {
                        renderContent(searchInput.value);
                    }, { renderSelection, fallbackBase, endpointPrefix }));
                }
            };

            const isNSFW = isComposerTypeNSFW(node, typeValue);
            items.push(createMenuItem("Paste Icon", async () => {
                try {
                    const file = await readClipboardImageFile();
                    if (!file) {
                        await showInfo("No Image", "No image found in clipboard");
                        return;
                    }

                    const icon = await resizeImageToCoverDataUrl(file, 64, "image/png");
                    const resp = await api.fetchApi(`${endpointPrefix}/save-type-settings`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            type_file: typeFile,
                            prompt_type: typeValue,
                            type_name: typeLabel,
                            icon,
                        }),
                    });
                    const result = await resp.json();
                    if (result.success) {
                        applyPromptPayloadToNode(node, result);
                        rebuildCategoryList();
                        rebuildTypeRailButtons();
                        renderContent(searchInput.value);
                    } else {
                        await showInfo("Error", result.error || "Failed to save prompt group icon.");
                    }
                } catch (error) {
                    console.error("[PromptBrowser] Error pasting type icon:", error);
                    await showInfo("Error", "Failed to paste prompt group icon. Make sure you have an image copied.");
                }
            }));

            items.push(null);


            items.push(createMenuItem("Generate Missing Thumbnails", async () => {
                await runTypeBatchGeneration("Generate Missing Thumbnails");
            }));

            items.push(createMenuItem("Re-Generate All Thumbnails", async () => {
                await runTypeBatchGeneration("Re-Generate All Thumbnails", { regenerateAll: true });
            }));

            items.push(null);

            items.push(createMenuItem(isNSFW ? "✓ NSFW" : "Mark as NSFW", async () => {
                try {
                    const resp = await api.fetchApi(`${endpointPrefix}/toggle-nsfw`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ type: "type", type_file: typeFile })
                    });
                    const result = await resp.json();
                    if (result.success) {
                        applyPromptPayloadToNode(node, result);
                        rebuildCategoryList();
                        rebuildTypeRailButtons();
                        renderContent(searchInput.value);
                    }
                } catch (err) {
                    console.error("[PromptBrowser] Error toggling type NSFW:", err);
                }
            }, isNSFW));

            items.push(null);

            items.push(createMenuItem("Rename", async () => {
                const result = await showTextInputDialog(
                    "Rename Prompt Group",
                    "Enter new prompt group name:",
                    typeLabel,
                );
                if (result === null) return;
                const newName = String(result || "").trim();
                if (!newName) return;
                if (newName === typeLabel) return;
                try {
                    const resp = await api.fetchApi(`${endpointPrefix}/rename-type`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ type_file: typeFile, new_name: newName })
                    });
                    const data = await resp.json();
                    if (data.success) {
                        applyPromptPayloadToNode(node, data);
                        categoryTypeFilter = typeValue;
                        rebuildCategoryList();
                        rebuildTypeRailButtons();
                        renderContent(searchInput.value);
                    } else {
                        await showInfo("Error", data.error || "Failed to rename type.");
                    }
                } catch (err) {
                    console.error("[PromptBrowser] Error renaming type:", err);
                }
            }));

            items.push(null);

            items.push(createMenuItem("Delete Group", async () => {
                if (!await showConfirm("Delete Group", `Are you sure you want to delete group "${typeLabel}" and all its categories?`)) {
                    return;
                }
                try {
                    const resp = await api.fetchApi(`${endpointPrefix}/delete-type`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ type_file: typeFile })
                    });
                    const data = await resp.json();
                    if (data.success) {
                        applyPromptPayloadToNode(node, data);
                        refreshPromptTypeFilters();
                        const remainingTypeChoices = promptTypeFilters.filter((choice) => choice.value !== "__all__");
                        const hasCurrentType = promptTypeFilters.some((choice) => choice.value === categoryTypeFilter);
                        if (categoryTypeFilter === typeValue || !hasCurrentType) {
                            categoryTypeFilter = remainingTypeChoices[0]?.value || "__all__";
                        }
                        await clearPromptBrowserSelection({ categoryKey: selectedCategory });
                        ensureSelectedCategory();
                        rebuildCategoryList();
                        rebuildTypeRailButtons();
                        if (editMode && editPanel) {
                            if (categoryTypeFilter !== "__all__") {
                                if (typeof editPanel.loadTypeSettings === "function") {
                                    editPanel.loadTypeSettings(categoryTypeFilter);
                                }
                                if (typeof editPanel.showTypeSettings === "function") {
                                    editPanel.showTypeSettings();
                                }
                            } else {
                                if (typeof editPanel.loadCategorySettings === "function") {
                                    editPanel.loadCategorySettings(selectedCategory);
                                }
                                if (typeof editPanel.showCategorySettings === "function") {
                                    editPanel.showCategorySettings();
                                }
                            }
                        }
                        renderContent(searchInput.value);
                    } else {
                        await showInfo("Error", data.error || "Failed to delete group.");
                    }
                } catch (err) {
                    console.error("[PromptBrowser] Error deleting group:", err);
                }
            }, true));

            openMenu(event.clientX, event.clientY, items);
        };

        const rebuildCategoryList = () => {
            allCategories = filterAllowedCategories(getVisibleCategories(node, {
                hideNSFW: hideNSFWState,
                workflowOnly,
                contentFilter: contentFilterState,
                endpointPrefix,
            }));
            ensureSelectedCategory();
            updateCategoryCreateControls();
            categoryButtons = [];
            categoryContainer.innerHTML = "";
            categories.forEach(cat => {
                const btn = el("button", "pm-tab");
                btn.type = "button";
                btn.dataset.category = cat;

                btn.textContent = showCategoryTypeFilter ? getComposerCategoryDisplayName(node, cat) : cat;

                btn.onclick = async () => {
                    const canProceed = await canChangeEditorContext();
                    if (!canProceed) return;
                    setBlankPromptSelection();
                    if (!saveUsesEditPanel && editMode && editPanel && typeof editPanel.clearPrompt === "function") {
                        await editPanel.clearPrompt({ skipConfirm: true });
                    }
                    setSelectedCategory(cat);
                    updateCategoryButtons();
                    renderContent(searchInput.value);
                };
                btn.oncontextmenu = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    showCategoryContextMenu(e, cat);
                };
                categoryButtons.push(btn);
                categoryContainer.appendChild(btn);
            });

            updateCategoryButtons();
        };
        rebuildCategoryList();

        // When the caller locks the browser to a single category, hide the whole bar.
        const hideCategoryBar = allowedCategories && allowedCategories.length === 1;
        if (hideCategoryBar) {
            categoryBar.style.display = "none";
        }

        // Main content area: grid + optional edit panel
        const contentRow = document.createElement("div");
        contentRow.style.cssText = `
            display: flex;
            flex: 1;
            min-height: 0;
            overflow: hidden;
        `;

        if (showCategoryTypeFilter) {
            const typeRailElement = renderTypeRail();
            if (typeRailElement) {
                contentRow.appendChild(typeRailElement);
            }
        }

        // Content container - fixed size so thumbnails never get encroached by bottom toolbar
        const gridContainer = document.createElement("div");
        gridContainer.className = "thumbnail-grid-container";
        gridContainer.style.cssText = `
            overflow-y: auto;
            flex: 1;
            min-width: ${computeMinGridWidth()}px;
            height: ${browserLayout.height}px;
            scrollbar-width: none;
            -ms-overflow-style: none;
        `;

        if (typeRail) {
            updateTypeRailButtons();
        }

        contentRow.appendChild(gridContainer);

        // Edit panel
        if (allowEditMode && (mode !== "save" || saveUsesEditPanel)) {
            editPanel = createPromptBrowserEditPanel({
                node,
                endpointPrefix,
                showInfo,
                showConfirm,
                loadPrompts: loadPromptsFn,
                savePrompt: async (payload) => {
                    try {
                        const body = buildSavePromptRequestBodyForEndpoint(endpointPrefix, payload);
                        if (endpointPrefix === "/prompt-manager/compose") {
                            const composerCategory = getComposerCategoryRequestIdentity(node, payload.category, payload.type_file || "");
                            body.category = composerCategory.categoryName;
                            body.type_file = composerCategory.typeFile;
                            if (Object.prototype.hasOwnProperty.call(body, "old_category")) {
                                body.old_category = getComposerCategoryDisplayName(node, payload.old_category || payload.category);
                            }
                        }
                        const resp = await api.fetchApi(`${endpointPrefix}/save-prompt`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(body),
                        });
                        const result = await resp.json();
                        if (result?.success) {
                            if (result?.prompts && typeof result.prompts === "object") {
                                applyPromptPayloadToNode(node, result);
                            }
                            const savedCategory = endpointPrefix === "/prompt-manager/compose"
                                ? resolveComposerCategoryKey(node, body.category, body.type_file || "")
                                : body.category;
                            syncComposerTypeFilterForCategory(savedCategory, body.type_file || "");
                            setCurrentPromptSelection(body.name, savedCategory);
                        }
                        return result;
                    } catch (err) {
                        console.error("[PromptBrowser] Error saving prompt:", err);
                        return { success: false, error: String(err) };
                    }
                },
                selectPrompt: (category, name) => {
                    if (category) {
                        const nextCategory = showCategoryTypeFilter ? resolveComposerCategoryKey(node, category) : category;
                        syncComposerTypeFilterForCategory(nextCategory);
                        selectedCategory = nextCategory;
                        rebuildCategoryList();
                    }
                    setCurrentPromptSelection(name);
                    renderContent(searchInput.value);
                },
                syncPromptSelection: (category, name) => {
                    const resolvedCategory = showCategoryTypeFilter ? resolveComposerCategoryKey(node, category || selectedCategory) : (category || selectedCategory);
                    const matchedName = findMatchingPromptName(resolvedCategory, name);
                    if (matchedName) {
                        setCurrentPromptSelection(matchedName, resolvedCategory);
                    } else {
                        setBlankPromptSelection();
                    }
                    renderContent(searchInput.value);
                },
                generateThumbnail: async (category, promptName, draftPromptData = null) => {
                    const isDraftGeneration = draftPromptData && typeof draftPromptData === "object";
                    const requestedSeed = Number(draftPromptData?.__pm_thumbnail_seed);
                    const staticSeed = Number.isFinite(requestedSeed) ? Math.trunc(requestedSeed) : 42;
                    const job = await queueThumbnail(promptName, async () => {
                        const generatedThumbnail = await _generateThumbnailForBrowserCategory(node, category, promptName, () => {
                            renderContent(searchInput.value);
                        }, {
                            endpointPrefix,
                            draftPromptData,
                            persistThumbnail: !isDraftGeneration,
                            promptStrength,
                            generationMode: thumbnailGenerationMode,
                            staticSeed,
                        });
                        if (isDraftGeneration) return generatedThumbnail || null;
                        await loadPromptsFn(node);
                        return getCategoryPromptEntry(node?.prompts?.[category], promptName, endpointPrefix)?.thumbnail || null;
                    });
                    if (job.error) throw job.error;
                    return job.value;
                },
                onChange: () => {
                    renderContent(searchInput.value);
                },
                onCategorySettingsSaved: async ({ category, previousPromptType, promptType }) => {
                    const previousType = String(previousPromptType || "").trim().toLowerCase();
                    const nextType = String(promptType || "").trim().toLowerCase();
                    const typeChanged = previousType !== nextType;

                    if (showCategoryTypeFilter && typeChanged) {
                        categoryTypeFilter = nextType || "__all__";
                        updateTypeRailButtons();
                    }

                    selectedCategory = showCategoryTypeFilter
                        ? resolveComposerCategoryKey(node, category, nextType && nextType !== "__all__" ? getComposerTypeFile(node, nextType) : "")
                        : category;
                    setBlankPromptSelection();
                    if (!saveUsesEditPanel && typeof editPanel.clearPrompt === "function") {
                        await editPanel.clearPrompt({ skipConfirm: true });
                    }
                    rebuildCategoryList();
                    if (editPanel) {
                        editPanel.loadCategorySettings(selectedCategory);
                        if (saveUsesEditPanel && typeof editPanel.showPromptSettings === "function") {
                            editPanel.showPromptSettings();
                        } else if (typeof editPanel.showCategorySettings === "function") {
                            editPanel.showCategorySettings();
                        }
                    }
                    renderContent(searchInput.value);
                },
                pickThumbnailModel: async () => {
                    const picked = await showThumbnailRenderPicker(
                        _thumbnailRenderFamily,
                        _thumbnailRenderModel,
                        _thumbnailRenderLora1,
                        _thumbnailRenderLora2,
                        _thumbnailRenderSteps,
                    );
                    if (picked) {
                        saveThumbnailRenderSelection(picked);
                        updateThumbnailModelBtn();
                        return true;
                    }
                    return false;
                },
                getThumbnailModelLabel: () => getThumbnailRenderLabelParts(),
                saveMode: mode === "save",
                compact: compactBrowser,
                width: getEditPanelWidth(),
            });
            if (!editMode) {
                editPanel.element.style.display = "none";
            }
            contentRow.appendChild(editPanel.element);
            if (editMode) {
                updateEditModeLayout();
            }
        }

        const grid = document.createElement("div");

        // Helper: get filtered prompt names for the selected category
        const getFilteredPrompts = (filter = "") => {
            let promptNames = getPromptNamesForCategory(node, selectedCategory, {
                hideNSFW: hideNSFWState,
                workflowOnly,
                contentFilter: contentFilterState,
                endpointPrefix,
            });

            // Filter by search
            if (filter) {
                promptNames = promptNames.filter(name => name.toLowerCase().includes(filter.toLowerCase()));
            }
            return promptNames;
        };

        const applyMultiSelectInteraction = (promptName, filteredPrompts, event) => {
            if (!isMultiSelectActive()) return "none";

            const promptList = Array.isArray(filteredPrompts) ? filteredPrompts : [];
            const isShiftRange = event?.shiftKey && multiSelectAnchorName;

            if (isShiftRange) {
                const anchorIndex = promptList.indexOf(multiSelectAnchorName);
                const targetIndex = promptList.indexOf(promptName);
                if (anchorIndex >= 0 && targetIndex >= 0) {
                    const start = Math.min(anchorIndex, targetIndex);
                    const end = Math.max(anchorIndex, targetIndex);
                    for (let i = start; i <= end; i++) {
                        selectedNames.add(promptList[i]);
                    }
                    if (multiCategorySelect && selectedCategory) {
                        selectedByCategory[selectedCategory] = selectedNames;
                    }
                    updateSelectButton();
                    return "rerender";
                }
            }

            if (selectedNames.has(promptName)) {
                selectedNames.delete(promptName);
            } else {
                selectedNames.add(promptName);
            }
            if (multiCategorySelect && selectedCategory) {
                selectedByCategory[selectedCategory] = selectedNames;
            }
            multiSelectAnchorName = promptName;
            updateSelectButton();
            return "single";
        };

        const isMultiSelectActive = () => supportsMultiSelect && multiSelectMode;

        const showMultiPromptContextMenu = (e, promptNames) => {
            const items = [];

            const createMenuItem = (label, onClick, danger = false) => ({ label, action: onClick, danger });

            const selectedList = Array.from(promptNames || []);

            items.push(createMenuItem(`Generate Thumbnails (${selectedList.length})`, async () => {
                const category = selectedCategory;
                for (const name of selectedList) {
                    queueThumbnail(name, () => _generateThumbnailForBrowserCategory(node, category, name, () => {
                        renderContent(searchInput.value);
                    }, { endpointPrefix }));
                }
            }));

            items.push(null);

            items.push(createMenuItem(`Move (${selectedList.length})`, async () => {
                const isComposerManager = endpointPrefix === "/prompt-manager/compose";
                const allCategories = Object.keys(node.prompts || {}).filter(c => c !== "__meta__").sort((a, b) => a.localeCompare(b));
                const currentComposerCategory = isComposerManager ? getComposerCategoryRequestIdentity(node, selectedCategory) : null;
                const currentTypeFile = currentComposerCategory?.typeFile || "";
                const composerMoveTargets = isComposerManager ? getComposerMoveTargetOptions(node) : null;
                const picked = await showCategoryPickerDialog(
                    "Move Selected Prompts",
                    allCategories,
                    isComposerManager ? currentComposerCategory?.categoryName || selectedCategory : selectedCategory,
                    isComposerManager ? {
                        groupOptions: composerMoveTargets?.groupOptions || [],
                        categoriesByGroup: composerMoveTargets?.categoriesByGroup || {},
                        defaultGroupValue: currentTypeFile,
                    } : {}
                );
                const targetCategory = picked?.category;
                const targetTypeFile = isComposerManager ? String(picked?.typeFile || currentTypeFile || "") : "";
                if (!targetCategory || (targetCategory === (currentComposerCategory?.categoryName || selectedCategory) && (!isComposerManager || targetTypeFile === currentTypeFile))) return;

                let movedCount = 0;
                const errors = [];
                for (const promptName of selectedList) {
                    try {
                        const resp = await api.fetchApi(`${endpointPrefix}/rename-prompt`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                category: currentComposerCategory?.categoryName || selectedCategory,
                                old_name: promptName,
                                new_name: promptName,
                                new_category: targetCategory,
                                ...(isComposerManager ? {
                                    type_file: currentComposerCategory?.typeFile || currentTypeFile,
                                    new_type_file: targetTypeFile || currentTypeFile,
                                } : {}),
                            })
                        });
                        const data = await resp.json();
                        if (data.success) {
                            applyPromptPayloadToNode(node, data);
                            movedCount++;
                        } else {
                            errors.push(`${promptName}: ${data.error || "move failed"}`);
                        }
                    } catch (err) {
                        errors.push(`${promptName}: ${err?.message || "request failed"}`);
                    }
                }

                selectedNames.clear();
                multiSelectAnchorName = "";
                updateSelectButton();
                updateEditModeLayout();
                renderContent(searchInput.value);

                if (errors.length > 0) {
                    await showInfo(
                        "Move Completed With Errors",
                        `Moved ${movedCount}/${selectedList.length}.\n\n${errors.slice(0, 8).join("\n")}${errors.length > 8 ? "\n..." : ""}`
                    );
                }
            }));

            items.push(null);

            items.push(createMenuItem(`Delete Prompts (${selectedList.length})`, async () => {
                const confirmed = await showConfirm(
                    "Delete Prompts",
                    `Are you sure you want to delete ${selectedList.length} selected prompt(s)?`,
                    "Delete",
                );
                if (!confirmed) return;

                for (const promptName of selectedList) {
                    await deletePromptEntry(node, selectedCategory, promptName, endpointPrefix);
                }
                await clearPromptBrowserSelection();
                updateEditModeLayout();
                renderContent(searchInput.value);
            }, true));

            openMenu(e.clientX, e.clientY, items);
        };

        // Shared right-click handler for prompt items (works in both grid and list view)
        const promptContextMenu = (e, promptName) => {
            e.preventDefault();
            if (isMultiSelectActive() && selectedNames.size > 1 && selectedNames.has(promptName)) {
                showMultiPromptContextMenu(e, selectedNames);
                return;
            }
            showThumbnailContextMenu(
                e,
                node,
                selectedCategory,
                promptName,
                () => {
                    renderContent(searchInput.value);
                },
                endpointPrefix,
                {
                    onDelete: async (deletedCategory, deletedPromptName) => {
                        await clearPromptBrowserSelection({ categoryKey: deletedCategory || selectedCategory });
                    },
                }
            );
        };

        // ---- Grid (Large Thumbnail) View ----
        // The badges in a card's corner: NSFW, then one mark for the entry's
        // kind (C compose, L layout, R recipe, P prompt manager). Null when none.
        const promptBadges = (promptData, { isNSFW, hasWorkflowData, hasComposeData, hasPromptPayload }) => {
            const showManagerBadge = !hasComposeData && hasPromptManagerPayload(promptData);
            const showWorkflowBadge = !hasComposeData && !showManagerBadge && !workflowOnly && hasWorkflowData;
            const showPromptBadge = showManagerBadge || (!hasComposeData && workflowOnly && !hasWorkflowData && hasPromptPayload);
            const marks = [];
            if (hasComposeData) {
                const layout = hasLayoutPayload(promptData);
                marks.push(layout ? ["L", "Has Layout Data", LAYOUT_BADGE_COLOR] : ["C", "Has Compose Data", COMPOSE_BADGE_COLOR]);
            } else if (showWorkflowBadge) {
                marks.push(["R", "Has Recipe Data", RECIPE_BADGE_COLOR]);
            } else if (showPromptBadge) {
                marks.push(["P", showManagerBadge ? "Has Prompt Manager Data" : "Prompt Data (Converted to Recipe)", T.brand]);
            }
            if (!isNSFW && !marks.length) return null;
            const stack = el("div", "pm-badges");
            if (isNSFW) stack.append(el("div", "pm-badge-nsfw", "NSFW"));
            for (const [letter, tip, color] of marks) {
                const mark = el("div", "pm-kind", letter);
                mark.title = tip;
                mark.style.background = color;
                stack.append(mark);
            }
            return stack;
        };

        // Thumbnail cards; `small` is the Icon view's sizes.
        const renderGridView = (filter = "", small = false) => {
            const cols = small ? browserLayout.iconCols : browserLayout.cols;
            const itemWidth = small ? browserLayout.iconItemWidth : browserLayout.itemWidth;
            const thumbWidth = small ? browserLayout.iconThumbWidth : browserLayout.thumbWidth;
            const thumbHeight = small ? browserLayout.iconThumbHeight : browserLayout.thumbHeight;
            grid.className = "pm-grid";
            grid.style.cssText = `grid-template-columns: repeat(${cols}, ${itemWidth}px); gap: ${small ? browserLayout.iconGap : browserLayout.gap}px;`;
            grid.innerHTML = "";

            const categoryPrompts = node.prompts[selectedCategory] || {};
            const filteredPrompts = getFilteredPrompts(filter);
            const showEditBlank = editMode && !!editPanel;

            if (filteredPrompts.length === 0 && !showEditBlank) {
                const emptyMsg = el("div", "pm-empty", filter ? "No matching prompts found" : "No prompts in this category");
                grid.appendChild(emptyMsg);
                return;
            }

            const updateCardSelection = (card, promptName) => {
                const isSel = selectedNames.has(promptName);
                card.dataset.selectedPrompt = isSel ? "true" : "false";
                card.classList.toggle("on", isSel);
                updateSelectButton();
            };

            const updateGridSelections = () => {
                grid.querySelectorAll("[data-prompt-name]").forEach((item) => {
                    updateCardSelection(item, item.dataset.promptName || "");
                });
                updateEditModeLayout();
            };

            filteredPrompts.forEach(promptName => {
                const promptData = getCategoryPromptEntry(categoryPrompts, promptName, endpointPrefix);
                const thumbnail = promptData?.thumbnail || DEFAULT_THUMBNAIL;
                const isSelected = isMultiSelectActive() ? selectedNames.has(promptName) : (promptName === currentPrompt && selectedCategory === currentPromptCategory);
                const isNSFW = promptData?.nsfw === true || isCategoryNSFW(selectedCategory);
                const rawWorkflowData = promptData?.workflow_data;
                const hasWorkflowData = !promptOnly && (
                    (typeof rawWorkflowData === "string" && rawWorkflowData.trim().length > 0) ||
                    (rawWorkflowData && typeof rawWorkflowData === "object" && Object.keys(rawWorkflowData).length > 0)
                );
                const hasComposeData = !promptOnly && hasComposeLikePayload(promptData);
                const hasPromptPayload = !promptOnly && hasPromptPresetPayload(promptData);

                const card = el("div", "pm-card");
                card.dataset.promptName = promptName;
                if (isSelected) {
                    card.dataset.selectedPrompt = "true";
                    card.classList.add("on");
                }

                // Top-right badge stack: NSFW first, the entry's kind under it.
                const badges = promptBadges(promptData, { isNSFW, hasWorkflowData, hasComposeData, hasPromptPayload });
                if (badges) card.appendChild(badges);

                // Thumbnail (using div with background-image to avoid browser extension interference)
                const thumbDiv = el("div", "pm-thumb");
                thumbDiv.style.cssText = `width: ${thumbWidth}px; height: ${thumbHeight}px; background-image: url(${thumbnail});`;
                _attachThumbnailDropHandlers(thumbDiv, node, selectedCategory, promptName, () => renderContent(searchInput.value), endpointPrefix);

                // A hover preview helps only where the thumbnails are small.
                if ((small || compactBrowser) && previewEnabled && thumbnail !== DEFAULT_THUMBNAIL) {
                    thumbDiv.addEventListener("mouseenter", (e) => {
                        e.stopPropagation();
                        showPreviewWithDelay(thumbnail, thumbDiv);
                    });
                    thumbDiv.addEventListener("mouseleave", (e) => {
                        e.stopPropagation();
                        scheduleHidePreview();
                    });
                }

                // Prompt name
                const nameLabel = el("div", "pm-name", promptName);
                nameLabel.title = promptName;

                card.appendChild(thumbDiv);
                card.appendChild(nameLabel);

                card.onclick = async (e) => {
                    if (supportsMultiSelect && (e.shiftKey || e.ctrlKey || e.metaKey)) {
                        enableMultiSelect({ preserveCurrentSelection: true });
                        applyMultiSelectInteraction(promptName, filteredPrompts, e);
                        if (editMode && editPanel && !saveUsesEditPanel) {
                            editPanel.loadPrompt(selectedCategory, promptName);
                        }
                        renderContent(searchInput.value);
                        return;
                    }

                    if (isMultiSelectActive()) {
                        const action = applyMultiSelectInteraction(promptName, filteredPrompts, e);
                        if (action === "rerender") {
                            if (editMode && editPanel && !saveUsesEditPanel) {
                                editPanel.loadPrompt(selectedCategory, promptName);
                            }
                            renderContent(searchInput.value);
                            return;
                        }
                        if (editMode && editPanel && !saveUsesEditPanel) {
                            editPanel.loadPrompt(selectedCategory, promptName);
                        }
                        updateCardSelection(card, promptName);
                        updateEditModeLayout();
                        return;
                    }

                    // In edit mode, ask before discarding unsaved changes.
                    if (editMode && editPanel && !saveUsesEditPanel) {
                        const now = Date.now();
                        if (editModeLastClickPrompt === promptName && (now - editModeLastClickAt) <= 500) {
                            resolve({ category: getSelectionCategory(), prompt: promptName, prompts: [promptName] });
                            cleanup();
                            return;
                        }
                        const loaded = await editPanel.loadPrompt(selectedCategory, promptName);
                        if (!loaded) return; // user cancelled, keep previous selection
                        editModeLastClickPrompt = promptName;
                        editModeLastClickAt = now;
                        selectedNames.clear();
                        if (multiCategorySelect) {
                            Object.keys(selectedByCategory).forEach((cat) => {
                                selectedByCategory[cat].clear();
                            });
                        }
                        setCurrentPromptSelection(promptName);
                        renderContent(searchInput.value);
                        return;
                    }

                    if (mode === "save") {
                        selectedSaveName = promptName;
                        if (saveUsesEditPanel && editPanel?.setPromptName) {
                            editPanel.setPromptName(promptName);
                        }
                        if (saveNameInput) {
                            saveNameInput.value = promptName;
                            saveNameInput.focus();
                            saveNameInput.select();
                        }
                        setCurrentPromptSelection(promptName);
                        updateGridSelections();
                        return;
                    }

                    // Normal click: forget any initial pre-selection and select only this prompt.
                    selectedNames.clear();
                    if (multiCategorySelect) {
                        Object.keys(selectedByCategory).forEach((cat) => {
                            selectedByCategory[cat].clear();
                        });
                    }
                    setCurrentPromptSelection(promptName);

                    if (requireDoubleClickToSelect) {
                        selectedNames.add(promptName);
                        updateGridSelections();
                        return;
                    }

                    resolve({ category: getSelectionCategory(), prompt: promptName, prompts: [promptName] });
                    cleanup();
                };

                if (mode === "save") {
                    card.ondblclick = async () => {
                        selectedSaveName = promptName;
                        if (saveUsesEditPanel && editPanel?.setPromptName) {
                            editPanel.setPromptName(promptName);
                        }
                        if (saveNameInput) {
                            saveNameInput.value = promptName;
                        }
                        setCurrentPromptSelection(promptName);
                        await handleSaveAction();
                    };
                } else if (requireDoubleClickToSelect || isMultiSelectActive()) {
                    card.ondblclick = () => {
                        resolve({ category: getSelectionCategory(), prompt: promptName, prompts: [promptName] });
                        cleanup();
                    };
                }

                card.oncontextmenu = (e) => promptContextMenu(e, promptName);

                grid.appendChild(card);
            });

            if (showEditBlank) {
                const blankCard = el("div", "pm-card blank");
                const blankThumb = el("div", "pm-thumb blank", "+");
                blankThumb.style.cssText = `width: ${thumbWidth}px; height: ${thumbHeight}px; font-size: ${small ? 24 : 36}px;`;
                const blankLabel = el("div", "pm-name", "Blank Prompt");

                blankCard.appendChild(blankThumb);
                blankCard.appendChild(blankLabel);
                blankCard.onclick = () => {
                    setBlankPromptSelection();
                    if (typeof editPanel.clearPrompt === "function") {
                        editPanel.clearPrompt();
                    }
                    if (typeof editPanel.showPromptSettings === "function") {
                        editPanel.showPromptSettings();
                    }
                    renderContent(searchInput.value);
                };

                grid.appendChild(blankCard);
            }
        };

        // ---- Icon (Small Thumbnail) Grid View: the grid at icon sizes ----
        const renderCompactGridView = (filter = "") => renderGridView(filter, true);

        // ---- List View ----
        const renderListView = (filter = "") => {
            grid.className = "pm-list";
            grid.style.cssText = "";
            grid.innerHTML = "";

            const categoryPrompts = node.prompts[selectedCategory] || {};
            const filteredPrompts = getFilteredPrompts(filter);
            const showEditBlank = editMode && !!editPanel;

            if (filteredPrompts.length === 0 && !showEditBlank) {
                const emptyMsg = el("div", "pm-empty", filter ? "No matching prompts found" : "No prompts in this category");
                grid.appendChild(emptyMsg);
                return;
            }

            const updateRowSelection = (row, promptName) => {
                const isSel = selectedNames.has(promptName);
                row.dataset.selectedPrompt = isSel ? "true" : "false";
                row.classList.toggle("on", isSel);
                updateSelectButton();
            };

            const updateListSelections = () => {
                grid.querySelectorAll("[data-prompt-name]").forEach((item) => {
                    updateRowSelection(item, item.dataset.promptName || "");
                });
                updateEditModeLayout();
            };

            const listViewportWidth = Math.max(
                520,
                browserLayout.width - 18
            );
            const defaultPromptColumnWidth = promptOnly
                ? Math.round(listViewportWidth * (browserPrefScope === "composer" ? 0.68 : 0.5))
                : Math.round(listViewportWidth * 0.43);
            let promptColumnWidth = getPromptColumnWidth(
                browserPrefScope,
                promptOnly,
                defaultPromptColumnWidth
            );
            const buildListColumns = () => {
                const promptCol = clampPromptColumnWidth(promptColumnWidth, promptOnly);
                return promptOnly
                    ? `44px minmax(120px, 1fr) ${promptCol}px`
                    : `44px minmax(150px, 1fr) ${promptCol}px 70px 70px 70px`;
            };

            // Column headers
            const headerRow = document.createElement("div");
            const listColumns = buildListColumns();
            headerRow.style.cssText = `
                display: grid;
                grid-template-columns: ${listColumns};
                gap: 0;
                padding: 0 8px 6px 8px;
                border-bottom: 1px solid ${T.rule};
                margin-bottom: 4px;
                align-items: center;
                position: relative;
            `;
            const promptHeaderPad = document.createElement("span");
            promptHeaderPad.style.cssText = "display: inline-block; width: 0.6em;";
            const headers = promptOnly
                ? ["", "Name", "Prompt"]
                : ["", "Name", "Prompt", "LoRAs A", "LoRAs B", "Triggers"];
            headers.forEach((h, i) => {
                const hDiv = document.createElement("div");
                if (h === "Prompt") {
                    hDiv.appendChild(promptHeaderPad.cloneNode(true));
                    hDiv.appendChild(document.createTextNode(h));
                } else {
                    hDiv.textContent = h;
                }
                hDiv.style.cssText = `
                    font-size: 10px;
                    color: ${T.faint};
                    font-weight: bold;
                    text-transform: uppercase;
                    text-align: ${i === 0 ? 'center' : (!promptOnly && i >= 3 ? 'center' : 'left')};
                `;

                if (i > 0) {
                    hDiv.style.paddingLeft = "16px";
                    hDiv.style.boxSizing = "border-box";
                }

                if (i === 2) {
                    hDiv.style.position = "relative";
                    const resizeHandle = document.createElement("div");
                    resizeHandle.title = "Drag to resize Prompt column";
                    resizeHandle.style.cssText = `
                        position: absolute;
                        left: -10px;
                        top: -6px;
                        bottom: -6px;
                        width: 20px;
                        cursor: col-resize;
                        z-index: 2;
                        background: transparent;
                    `;

                    resizeHandle.addEventListener("mousedown", (evt) => {
                        evt.preventDefault();
                        evt.stopPropagation();
                        const startX = evt.clientX;
                        const startWidth = promptColumnWidth;
                        const previousUserSelect = document.body.style.userSelect;
                        const previousCursor = document.body.style.cursor;
                        document.body.style.userSelect = "none";
                        document.body.style.cursor = "col-resize";

                        const applyTemplate = (width) => {
                            const clampedWidth = clampPromptColumnWidth(width, promptOnly);
                            const template = promptOnly
                                ? `44px minmax(120px, 1fr) ${clampedWidth}px`
                                : `44px minmax(150px, 1fr) ${clampedWidth}px 70px 70px 70px`;
                            headerRow.style.gridTemplateColumns = template;
                            const rows = grid.querySelectorAll('[data-pm-list-row="true"]');
                            rows.forEach((rowEl) => {
                                rowEl.style.gridTemplateColumns = template;
                            });
                            promptColumnWidth = clampedWidth;
                            refreshContinuousDividers();
                        };

                        const onMouseMove = (moveEvt) => {
                            const delta = moveEvt.clientX - startX;
                            applyTemplate(startWidth - delta);
                        };

                        const onMouseUp = () => {
                            document.removeEventListener("mousemove", onMouseMove);
                            document.removeEventListener("mouseup", onMouseUp);
                            document.body.style.userSelect = previousUserSelect;
                            document.body.style.cursor = previousCursor;
                            setPromptColumnWidth(promptColumnWidth, browserPrefScope, promptOnly);
                            localStorage.setItem(
                                getPromptColumnWidthStorageKey(browserPrefScope),
                                String(promptColumnWidth)
                            );
                        };

                        document.addEventListener("mousemove", onMouseMove);
                        document.addEventListener("mouseup", onMouseUp);
                    });

                    hDiv.appendChild(resizeHandle);
                }
                headerRow.appendChild(hDiv);
            });
            grid.appendChild(headerRow);

            let dividerLayer = null;
            const refreshContinuousDividers = () => {
                if (dividerLayer && dividerLayer.parentNode) {
                    dividerLayer.parentNode.removeChild(dividerLayer);
                }

                dividerLayer = document.createElement("div");
                dividerLayer.style.cssText = `
                    position: absolute;
                    left: 8px;
                    right: 8px;
                    top: 0;
                    bottom: 0;
                    pointer-events: none;
                    z-index: 0;
                `;

                const headerCells = Array.from(headerRow.children);
                for (let i = 1; i < headerCells.length; i += 1) {
                    const cell = headerCells[i];
                    const line = document.createElement("div");
                    line.style.cssText = `
                        position: absolute;
                        left: ${cell.offsetLeft}px;
                        top: 0;
                        bottom: 0;
                        width: 1px;
                        background: ${UI.sectionBorder};
                    `;
                    dividerLayer.appendChild(line);
                }

                grid.appendChild(dividerLayer);
            };

            filteredPrompts.forEach(promptName => {
                const promptData = getCategoryPromptEntry(categoryPrompts, promptName, endpointPrefix);
                const thumbnail = promptData?.thumbnail || DEFAULT_THUMBNAIL;
                const isSelected = isMultiSelectActive() ? selectedNames.has(promptName) : (promptName === currentPrompt && selectedCategory === currentPromptCategory);
                const isNSFW = promptData?.nsfw === true || isCategoryNSFW(selectedCategory);
                const rawWorkflowData = promptData?.workflow_data;
                const hasWorkflowData = !promptOnly && (
                    (typeof rawWorkflowData === "string" && rawWorkflowData.trim().length > 0) ||
                    (rawWorkflowData && typeof rawWorkflowData === "object" && Object.keys(rawWorkflowData).length > 0)
                );
                const hasPromptPayload = !promptOnly && hasPromptPresetPayload(promptData);
                const lorasACount = promptOnly ? 0 : (promptData?.loras_a || []).length;
                const lorasBCount = promptOnly ? 0 : (promptData?.loras_b || []).length;
                const triggerCount = promptOnly ? 0 : (promptData?.trigger_words || []).length;

                const row = document.createElement("div");
                row.dataset.pmListRow = "true";
                row.dataset.promptName = promptName;
                if (isSelected) {
                    row.dataset.selectedPrompt = "true";
                }
                row.className = isSelected ? "pm-row on" : "pm-row";
                row.style.gridTemplateColumns = listColumns;

                // Thumbnail icon (using div with background-image to avoid browser extension interference)
                const thumbWrap = document.createElement("div");
                thumbWrap.style.cssText = `
                    width: 36px;
                    height: 36px;
                    position: relative;
                    flex-shrink: 0;
                    margin: 0 auto;
                `;

                const thumbDiv = el("div", "pm-thumb");
                thumbDiv.style.cssText = `width: 36px; height: 36px; background-image: url(${thumbnail});`;
                _attachThumbnailDropHandlers(thumbDiv, node, selectedCategory, promptName, () => renderContent(searchInput.value), endpointPrefix);

                // Add hover preview with proper event handling (if enabled and not placeholder)
                if (previewEnabled && thumbnail !== DEFAULT_THUMBNAIL) {
                    thumbDiv.addEventListener("mouseenter", (e) => {
                        e.stopPropagation();
                        showPreviewWithDelay(thumbnail, thumbDiv);
                    });
                    thumbDiv.addEventListener("mouseleave", (e) => {
                        e.stopPropagation();
                        scheduleHidePreview();
                    });
                }

                thumbWrap.appendChild(thumbDiv);

                const hasComposeData = !promptOnly && hasComposeLikePayload(promptData);
                const badges = promptBadges(promptData, { isNSFW: false, hasWorkflowData, hasComposeData, hasPromptPayload });
                if (badges) {
                    badges.classList.add("corner");
                    thumbWrap.appendChild(badges);
                }

                // Name + optional NSFW badge
                const nameDiv = document.createElement("div");
                nameDiv.style.cssText = `
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    padding: 4px 16px;
                    overflow: hidden;
                    box-sizing: border-box;
                `;
                const nameSpan = document.createElement("span");
                nameSpan.textContent = promptName;
                nameSpan.title = promptName;
                nameSpan.className = "pm-row-name";
                nameDiv.appendChild(nameSpan);
                if (isNSFW) {
                    const badge = el("span", "pm-badge-nsfw", "NSFW");
                    nameDiv.appendChild(badge);
                }

                const promptTextRaw = String(promptData?.prompt || "").trim();
                const promptTextDisplay = promptTextRaw || "—";
                const promptDiv = document.createElement("div");
                promptDiv.style.cssText = `
                    font-size: 12px;
                    color: ${T.label};
                    padding: 4px 16px;
                    box-sizing: border-box;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                `;
                promptDiv.textContent = promptTextDisplay;

                if (promptTextRaw) {
                    promptDiv.addEventListener("mouseenter", (evt) => {
                        showPromptTextTooltip(promptTextRaw, evt.clientX, evt.clientY);
                    });
                    promptDiv.addEventListener("mousemove", (evt) => {
                        movePromptTextTooltip(evt.clientX, evt.clientY);
                    });
                    promptDiv.addEventListener("mouseleave", () => {
                        hidePromptTextTooltip();
                    });
                }

                // Counts
                const makeCount = (n) => {
                    const d = document.createElement("div");
                    d.textContent = n > 0 ? n : "—";
                    d.style.cssText = `
                        font-size: 12px;
                        color: ${n > 0 ? T.label : T.faint};
                        text-align: center;
                        padding: 4px 16px;
                        box-sizing: border-box;
                    `;
                    return d;
                };

                row.appendChild(thumbWrap);
                row.appendChild(nameDiv);
                row.appendChild(promptDiv);
                if (!promptOnly) {
                    row.appendChild(makeCount(lorasACount));
                    row.appendChild(makeCount(lorasBCount));
                    row.appendChild(makeCount(triggerCount));
                }

                row.onclick = async (e) => {
                    if (supportsMultiSelect && (e.shiftKey || e.ctrlKey || e.metaKey)) {
                        enableMultiSelect({ preserveCurrentSelection: true });
                        applyMultiSelectInteraction(promptName, filteredPrompts, e);
                        if (editMode && editPanel && !saveUsesEditPanel) {
                            editPanel.loadPrompt(selectedCategory, promptName);
                        }
                        renderContent(searchInput.value);
                        return;
                    }

                    if (isMultiSelectActive()) {
                        const action = applyMultiSelectInteraction(promptName, filteredPrompts, e);
                        if (action === "rerender") {
                            if (editMode && editPanel && !saveUsesEditPanel) {
                                editPanel.loadPrompt(selectedCategory, promptName);
                            }
                            renderContent(searchInput.value);
                            return;
                        }
                        if (editMode && editPanel && !saveUsesEditPanel) {
                            editPanel.loadPrompt(selectedCategory, promptName);
                        }
                        updateRowSelection(row, promptName);
                        updateEditModeLayout();
                        return;
                    }

                    if (editMode && editPanel && !saveUsesEditPanel) {
                        const now = Date.now();
                        if (editModeLastClickPrompt === promptName && (now - editModeLastClickAt) <= 1000) {
                            resolve({ category: getSelectionCategory(), prompt: promptName, prompts: [promptName] });
                            cleanup();
                            return;
                        }
                        const loaded = await editPanel.loadPrompt(selectedCategory, promptName);
                        if (!loaded) return; // user cancelled, keep previous selection
                        editModeLastClickPrompt = promptName;
                        editModeLastClickAt = now;
                        selectedNames.clear();
                        if (multiCategorySelect) {
                            Object.keys(selectedByCategory).forEach((cat) => {
                                selectedByCategory[cat].clear();
                            });
                        }
                        setCurrentPromptSelection(promptName);
                        renderContent(searchInput.value);
                        return;
                    }

                    if (mode === "save") {
                        selectedSaveName = promptName;
                        if (saveUsesEditPanel && editPanel?.setPromptName) {
                            editPanel.setPromptName(promptName);
                        }
                        if (saveNameInput) {
                            saveNameInput.value = promptName;
                            saveNameInput.focus();
                            saveNameInput.select();
                        }
                        setCurrentPromptSelection(promptName);
                        updateListSelections();
                        return;
                    }

                    // Normal click: forget any initial pre-selection and select only this prompt.
                    selectedNames.clear();
                    if (multiCategorySelect) {
                        Object.keys(selectedByCategory).forEach((cat) => {
                            selectedByCategory[cat].clear();
                        });
                    }
                    setCurrentPromptSelection(promptName);

                    if (requireDoubleClickToSelect) {
                        selectedNames.add(promptName);
                        updateListSelections();
                        return;
                    }

                    resolve({ category: getSelectionCategory(), prompt: promptName, prompts: [promptName] });
                    cleanup();
                };

                if (mode === "save") {
                    row.ondblclick = async () => {
                        selectedSaveName = promptName;
                        if (saveUsesEditPanel && editPanel?.setPromptName) {
                            editPanel.setPromptName(promptName);
                        }
                        if (saveNameInput) {
                            saveNameInput.value = promptName;
                        }
                        setCurrentPromptSelection(promptName);
                        await handleSaveAction();
                    };
                } else if (requireDoubleClickToSelect || isMultiSelectActive()) {
                    row.ondblclick = () => {
                        resolve({ category: getSelectionCategory(), prompt: promptName, prompts: [promptName] });
                        cleanup();
                    };
                }

                row.oncontextmenu = (e) => promptContextMenu(e, promptName);

                grid.appendChild(row);
            });

            if (showEditBlank) {
                const row = el("div", "pm-row");
                row.dataset.pmListRow = "true";
                row.style.gridTemplateColumns = listColumns;

                const blankThumbWrap = el("div", "pm-thumb blank", "+");
                blankThumbWrap.style.cssText = "width: 36px; height: 36px; margin: 0 auto; font-size: 20px;";

                const blankName = el("div", "pm-row-name", "Blank Prompt");
                blankName.style.padding = "4px 16px";

                const blankPrompt = document.createElement("div");
                blankPrompt.textContent = "New empty prompt";
                blankPrompt.style.cssText = `
                    font-size: 12px;
                    color: ${T.label};
                    padding: 4px 16px;
                    box-sizing: border-box;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                `;

                row.appendChild(blankThumbWrap);
                row.appendChild(blankName);
                row.appendChild(blankPrompt);
                if (!promptOnly) {
                    const dash = () => {
                        const d = document.createElement("div");
                        d.textContent = "—";
                        d.style.cssText = `font-size: 12px; color: ${T.faint}; text-align: center; padding: 4px 16px; box-sizing: border-box;`;
                        return d;
                    };
                    row.appendChild(dash());
                    row.appendChild(dash());
                    row.appendChild(dash());
                }

                row.onclick = () => {
                    setBlankPromptSelection();
                    if (typeof editPanel.clearPrompt === "function") {
                        editPanel.clearPrompt();
                    }
                    if (typeof editPanel.showPromptSettings === "function") {
                        editPanel.showPromptSettings();
                    }
                    renderContent(searchInput.value);
                };

                grid.appendChild(row);
            }

            requestAnimationFrame(() => {
                refreshContinuousDividers();
            });
        };

        // Thumbnail hover preview system
        let hoverPreview = null;
        let hoverTimer = null;
        let hideTimer = null;
        let resetTimer = null;
        let previewActivated = false;  // Tracks if preview system is "warmed up"
        let currentMouseX = 0;
        let currentMouseY = 0;
        let currentThumbnail = "";
        let previewWidth = 0;
        let previewHeight = 0;
        let promptTextTooltip = null;

        const ensurePromptTextTooltip = () => {
            if (promptTextTooltip) return promptTextTooltip;
            promptTextTooltip = document.createElement("div");
            promptTextTooltip.setAttribute("data-pm-prompt-text-tooltip", "true");
            promptTextTooltip.style.cssText = `
                position: fixed;
                display: none;
                max-width: min(560px, 70vw);
                max-height: min(320px, 50vh);
                overflow: auto;
                white-space: pre-wrap;
                word-break: break-word;
                background: ${T.panel};
                border: 1px solid ${T.plateEdge};
                border-radius: ${RADIUS_CARD}px;
                color: ${T.inkSoft};
                font: 14px/1.55 ${FONT};
                padding: 12px 14px;
                box-shadow: 0 10px 28px rgba(0,0,0,0.55);
                z-index: ${Z_MENU};
                pointer-events: none;
            `;
            document.body.appendChild(promptTextTooltip);
            return promptTextTooltip;
        };

        const movePromptTextTooltip = (x, y) => {
            if (!promptTextTooltip || promptTextTooltip.style.display === "none") return;
            const margin = 14;
            const w = promptTextTooltip.offsetWidth || 360;
            const h = promptTextTooltip.offsetHeight || 160;
            let left = x + margin;
            let top = y + margin;
            if (left + w > window.innerWidth - 8) {
                left = Math.max(8, x - w - margin);
            }
            if (top + h > window.innerHeight - 8) {
                top = Math.max(8, y - h - margin);
            }
            promptTextTooltip.style.left = `${left}px`;
            promptTextTooltip.style.top = `${top}px`;
        };

        const showPromptTextTooltip = (text, x, y) => {
            const tip = ensurePromptTextTooltip();
            tip.textContent = text;
            tip.style.display = "block";
            movePromptTextTooltip(x, y);
        };

        const hidePromptTextTooltip = () => {
            if (promptTextTooltip) {
                promptTextTooltip.style.display = "none";
            }
        };

        const createHoverPreview = () => {
            if (!hoverPreview) {
                hoverPreview = document.createElement("div");
                hoverPreview.setAttribute('data-pm-thumbnail-preview', 'true');
                hoverPreview.style.cssText = `
                    position: fixed;
                    pointer-events: none;
                    z-index: ${Z_MENU};
                    display: none;
                    border: 1px solid ${T.plateEdge};
                    border-radius: ${RADIUS_CARD}px;
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.8);
                    background-color: ${T.card};
                    background-size: contain;
                    background-position: center;
                    background-repeat: no-repeat;
                `;
                
                document.body.appendChild(hoverPreview);
            }
            return hoverPreview;
        };

        const updatePreviewPosition = () => {
            if (!hoverPreview || !previewWidth || !previewHeight) return;
            
            // Center on thumbnail position, keeping on screen
            let left = currentMouseX - previewWidth / 2;
            let top = currentMouseY - previewHeight / 2;
            
            // Keep on screen with padding
            left = Math.max(5, Math.min(left, window.innerWidth - previewWidth - 9));
            top = Math.max(5, Math.min(top, window.innerHeight - previewHeight - 9));
            
            hoverPreview.style.left = left + "px";
            hoverPreview.style.top = top + "px";
        };

        const showPreviewWithDelay = (thumbnailSrc, thumbnailElement) => {
            // Cancel any pending hide operation and reset timer
            clearTimeout(hideTimer);
            clearTimeout(resetTimer);
            hideTimer = null;
            resetTimer = null;
            
            const rect = thumbnailElement.getBoundingClientRect();
            currentMouseX = rect.left + rect.width / 2;
            currentMouseY = rect.top + rect.height / 2;
            currentThumbnail = thumbnailSrc;
            
            // Intelligent delay: 2000ms initial, 10ms once activated
            const delay = previewActivated ? 10 : 2000;
            
            clearTimeout(hoverTimer);
            hoverTimer = setTimeout(() => {
                previewActivated = true;  // Activate fast mode after first preview
                const preview = createHoverPreview();
                
                // Load image in memory (not in DOM) to get dimensions
                const tempImg = new Image();
                tempImg.onload = function() {
                    const naturalWidth = this.naturalWidth;
                    const naturalHeight = this.naturalHeight;
                    
                    // Scale logic:
                    // - Thumbnails <= 200px in both dimensions: 2x scale (200px -> 400px)
                    // - Thumbnails > 200px in either dimension: 1x scale (keep original size)
                    const scale = (naturalWidth > 200 || naturalHeight > 200) ? 1 : 1;
                    
                    previewWidth = naturalWidth * scale;
                    previewHeight = naturalHeight * scale;
                    
                    // Update preview div with background image and dimensions
                    preview.style.width = previewWidth + 'px';
                    preview.style.height = previewHeight + 'px';
                    preview.style.backgroundImage = `url(${thumbnailSrc})`;
                    
                    updatePreviewPosition();
                    preview.style.display = "block";
                };
                
                // Load the image
                tempImg.src = thumbnailSrc;
            }, delay);
        };

        const hidePreview = () => {
            clearTimeout(hoverTimer);
            clearTimeout(hideTimer);
            hideTimer = null;
            
            if (hoverPreview) {
                hoverPreview.style.display = "none";
                previewWidth = 0;
                previewHeight = 0;
            }
        };

        const scheduleHidePreview = () => {
            clearTimeout(hideTimer);
            hideTimer = setTimeout(() => {
                hidePreview();
                
                // Start reset timer: after 0.5 seconds of no hovering, reset to slow mode
                clearTimeout(resetTimer);
                resetTimer = setTimeout(() => {
                    previewActivated = false;
                }, 500);
            }, 100);
        };

        // Unified render function: picks grid vs list based on currentViewMode
        const renderContent = (filter = "") => {
            if (currentViewMode === "list") {
                renderListView(filter);
            } else if (currentViewMode === "icon") {
                renderCompactGridView(filter);
            } else {
                renderGridView(filter);
            }
        };

        // Event handlers for controls
        contentFilterBtn.onclick = () => {
            if (contentFilterCycle) {
                const index = contentFilterCycle.indexOf(contentFilterState);
                contentFilterState = contentFilterCycle[(index + 1) % contentFilterCycle.length];
                onContentFilterChange?.(contentFilterState);
                updateContentFilterBtn();
                rebuildCategoryList();
                renderContent(searchInput.value);
                return;
            }
            if (contentFilterState === "all") {
                contentFilterState = "prompt";
            } else if (contentFilterState === "prompt") {
                contentFilterState = "recipe";
            } else if (contentFilterState === "recipe") {
                contentFilterState = "compose";
            } else if (contentFilterState === "compose") {
                contentFilterState = "layout";
            } else {
                contentFilterState = "all";
            }

            setBrowserContentFilter(contentFilterState);
            app.ui.settings.setSettingValue("PromptManager.BrowserContentFilter", contentFilterState);
            updateContentFilterBtn();
            rebuildCategoryList();
            renderContent(searchInput.value);
        };

        allCategoriesBtn.onclick = () => {
            showAllCategoriesState = !showAllCategoriesState;
            updateAllCategoriesBtn();
            rebuildCategoryList();
            renderContent(searchInput.value);
        };

        nsfwBtn.onclick = () => {
            hideNSFWState = !hideNSFWState;
            setHideNSFW(hideNSFWState);
            updateNsfwBtn();
            rebuildCategoryList();
            rebuildTypeRailButtons();
            renderContent(searchInput.value);
        };

        function setBrowserViewMode(mode) {
            if (mode === currentViewMode) return;
            currentViewMode = mode;
            setViewMode(currentViewMode, browserPrefScope);
            localStorage.setItem(getViewModeStorageKey(browserPrefScope), currentViewMode);
            updateViewModeBtn();
            renderContent(searchInput.value);
        }

        let saveNameInput = null;
        let saveActionButton = null;
        let cancelSaveButton = null;

        const handleSaveAction = async () => {
            if (mode !== "save" || !onSave) {
                return;
            }

            const category = ensureSelectedCategory();
            if (!category) {
                await showInfo("Missing Category", "Please create or select a category before saving.");
                return;
            }

            const draftPayload = saveUsesEditPanel && editPanel?.getCurrentPromptDraft
                ? editPanel.getCurrentPromptDraft()
                : null;
            const name = String(draftPayload?.name ?? saveNameInput?.value ?? "").trim();
            if (!name) {
                await showInfo("Missing Name", "Please enter a name before saving.");
                if (saveNameInput) saveNameInput.focus();
                return;
            }

            const existing = getCategoryPromptEntry(node?.prompts?.[category], name, endpointPrefix);
            const categoryLabel = getResultCategoryName(category);
            let overwrite = false;
            if (existing) {
                overwrite = await showConfirm(
                    "Overwrite Prompt",
                    `Prompt "${name}" already exists in category "${categoryLabel}". Do you want to replace it?`,
                    "Replace",
                );
                if (!overwrite) {
                    return;
                }
            }

            const saveResult = await onSave({
                ...(draftPayload && typeof draftPayload === "object" ? draftPayload : {}),
                category: categoryLabel,
                name,
                overwrite,
                text: typeof draftPayload?.text === "string" ? draftPayload.text : undefined,
                thumbnail: draftPayload?.thumbnail,
            });
            if (saveResult?.success) {
                resolve(saveResult);
                cleanup();
            } else {
                await showInfo("Save Failed", saveResult?.error || "Failed to save workflow.");
            }
        };

        // Initial render
        renderContent();

        // Scroll to selected prompt after rendering
        setTimeout(() => {
            const selectedElement = gridContainer.querySelector('[data-selected-prompt="true"]');
            if (selectedElement) {
                selectedElement.scrollIntoView({ behavior: 'instant', block: 'center' });
            }
        }, 50);

        // Search filtering
        searchInput.oninput = () => {
            rebuildCategoryList();
            renderContent(searchInput.value);
        };

        gridContainer.appendChild(grid);

        // Footer with hint
        const footer = document.createElement("div");
        footer.style.cssText = `
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-top: 6px;
            margin-bottom: 0;
            font-size: 12px;
            line-height: 1.35;
            color: ${T.faint};
            text-align: center;
        `;

        const saveBar = document.createElement("div");
        if (mode === "save") {
            saveBar.style.cssText = `
                display: ${saveUsesEditPanel ? "none" : "flex"};
                gap: 8px;
                align-items: center;
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid ${T.rule};
            `;

            if (!saveUsesEditPanel) {
                saveNameInput = el("input", "fbui-text");
                saveNameInput.type = "text";
                saveNameInput.spellcheck = false;
                saveNameInput.value = selectedSaveName;
                saveNameInput.placeholder = saveNamePlaceholder;
                saveNameInput.style.cssText = "flex: 1; min-width: 0; user-select: text;";
                saveNameInput.onkeydown = async (e) => {
                    if (e.key === "Enter") {
                        e.preventDefault();
                        e.stopPropagation();
                        await handleSaveAction();
                    }
                };
            }

            saveActionButton = button(saveButtonText, { primary: true });
            saveActionButton.onclick = async () => {
                await handleSaveAction();
            };

            cancelSaveButton = button("Cancel");
            cancelSaveButton.onclick = () => {
                resolve(null);
                cleanup();
            };

            if (saveNameInput) {
                saveBar.appendChild(saveNameInput);
            }
            saveBar.appendChild(cancelSaveButton);
            saveBar.appendChild(saveActionButton);
        } else if (supportsMultiSelect) {
            saveBar.style.cssText = `
                display: flex;
                gap: 8px;
                align-items: center;
                justify-content: space-between;
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid ${T.rule};
            `;

            const selectionTools = document.createElement("div");
            selectionTools.style.cssText = `
                display: flex;
                align-items: center;
                gap: 8px;
            `;

            const actionButtons = document.createElement("div");
            actionButtons.style.cssText = `
                display: flex;
                align-items: center;
                gap: 8px;
            `;

            const clearSelectionBtn = button("Clear");
            clearSelectionBtn.onclick = () => {
                selectedNames.clear();
                if (multiCategorySelect) {
                    Object.keys(selectedByCategory).forEach((cat) => {
                        selectedByCategory[cat].clear();
                    });
                }
                multiSelectAnchorName = "";
                updateSelectButton();
                updateEditModeLayout();
                renderContent(searchInput.value);
            };

            const selectAllBtn = button("Select All");
            selectAllBtn.onclick = () => {
                const visiblePrompts = getFilteredPrompts(searchInput.value);
                visiblePrompts.forEach((promptName) => selectedNames.add(promptName));
                if (multiCategorySelect && selectedCategory) {
                    selectedByCategory[selectedCategory] = selectedNames;
                }
                updateSelectButton();
                updateEditModeLayout();
                renderContent(searchInput.value);
            };

            const cancelBtn = button("Cancel");
            cancelBtn.onclick = () => {
                resolve(null);
                cleanup();
            };

            const buildMultiSelectResult = (selectionMode = "select") => {
                if (multiCategorySelect && selectedCategory && selectedNames.size > 0) {
                    selectedByCategory[selectedCategory] = selectedNames;
                }
                const result = {
                    category: selectedCategory,
                    prompt: "",
                    prompts: [],
                    selectionMode,
                };
                if (multiCategorySelect) {
                    result.selectionsByCategory = {};
                    for (const [cat, names] of Object.entries(selectedByCategory)) {
                        if (names.size > 0) {
                            result.selectionsByCategory[cat] = Array.from(names);
                        }
                    }
                    const allPrompts = Object.values(result.selectionsByCategory).flat();
                    result.prompts = allPrompts;
                    result.prompt = allPrompts[0] || "";
                } else {
                    result.prompts = Array.from(selectedNames);
                    result.prompt = selectedNames.size > 0 ? Array.from(selectedNames)[0] : "";
                }
                return result;
            };

            const selectBtn = button(useComposerMultiSelectActions
                ? `Add as 1 Prompt (${selectedNames.size})`
                : `Select (${selectedNames.size})`, { primary: true });
            updateSelectButton = () => {
                const count = multiCategorySelect
                    ? Object.values(selectedByCategory).reduce((sum, set) => sum + set.size, 0)
                    : selectedNames.size;
                selectBtn.textContent = useComposerMultiSelectActions
                    ? `Add as 1 Prompt (${count})`
                    : `Select (${count})`;
                selectBtn.disabled = count === 0;
                if (addAllPromptsBtn) {
                    addAllPromptsBtn.textContent = `Add All Prompts (${count})`;
                    addAllPromptsBtn.disabled = count === 0;
                }
            };
            selectBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                if (selectBtn.disabled) {
                    return;
                }
                resolve(buildMultiSelectResult(useComposerMultiSelectActions ? "combine" : "select"));
                cleanup();
            });

            let addAllPromptsBtn = null;
            if (useComposerMultiSelectActions) {
                addAllPromptsBtn = button("", { primary: true });
                addAllPromptsBtn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    if (addAllPromptsBtn.disabled) {
                        return;
                    }
                    resolve(buildMultiSelectResult("split"));
                    cleanup();
                });
            }

            selectionTools.appendChild(clearSelectionBtn);
            selectionTools.appendChild(selectAllBtn);
            actionButtons.appendChild(cancelBtn);
            actionButtons.appendChild(selectBtn);
            if (addAllPromptsBtn) {
                actionButtons.appendChild(addAllPromptsBtn);
            }

            saveBar.appendChild(selectionTools);
            saveBar.appendChild(actionButtons);

            updateSelectionToolbar = () => {
                saveBar.style.display = (supportsMultiSelect && multiSelectMode) ? "flex" : "none";
            };
            updateSelectButton();
            updateSelectionToolbar();
        }

        if (mode === "save" || supportsMultiSelect) {
            footer.appendChild(saveBar);
        }

        const footerHint = document.createElement("div");
        footerHint.style.cssText = `
            border-top: 1px solid ${UI.sectionBorder};
            padding-top: 8px;
            text-align: center;
        `;
        footer.appendChild(footerHint);

        updateFooterText = () => {
            footerHint.textContent = mode === "save"
                ? "Right-click a prompt or category for more options (thumbnails, NSFW, delete). Single-click fills name; double-click replaces."
                : ((supportsMultiSelect && multiSelectMode)
                    ? (multiCategorySelect
                        ? "Multi-select is ON. Select prompts across categories. Shift+click for range selection. Right-click selected prompts for batch actions."
                        : "Multi-select is ON. Click prompts to select/deselect, Shift+click for range selection, and right-click selected prompts for batch actions.")
                    : (requireDoubleClickToSelect
                        ? "Click a prompt to select it. Double-click it to send it. Right-click a prompt or category for more options."
                        : "Right-click a prompt or category for more options (thumbnails, NSFW, delete). Turn Multi on for batch actions."));
        };
        updateFooterText();

        dialog.appendChild(header);
        dialog.appendChild(controlsBar);
        dialog.appendChild(categoryBar);
        dialog.appendChild(contentRow);
        dialog.appendChild(footer);

        const cleanup = () => {
            clearTimeout(hoverTimer);
            clearTimeout(hideTimer);
            clearTimeout(resetTimer);
            window.removeEventListener("resize", handleWindowResize);
            hidePromptTextTooltip();
            hidePreview();
            if (hoverPreview && hoverPreview.parentNode) {
                document.body.removeChild(hoverPreview);
            }
            hoverPreview = null;
            if (promptTextTooltip && promptTextTooltip.parentNode) {
                document.body.removeChild(promptTextTooltip);
            }
            promptTextTooltip = null;
            if (typeRailTooltip && typeRailTooltip.parentNode) {
                document.body.removeChild(typeRailTooltip);
            }
            typeRailTooltip = null;
            // Clean up any stale preview elements
            const allPreviews = document.querySelectorAll('[data-pm-thumbnail-preview]');
            allPreviews.forEach(p => {
                if (p.parentNode) p.parentNode.removeChild(p);
            });
            document.body.removeChild(overlay);
            document.body.removeChild(dialog);
        };

        closeBtn.onclick = () => {
            resolve(null);
            cleanup();
        };

        overlay.onclick = () => {
            resolve(null);
            cleanup();
        };

        // Prevent dialog click from closing
        dialog.onclick = (e) => e.stopPropagation();

        // Keyboard shortcuts
        dialog.onkeydown = async (e) => {
            if (e.key === "Escape") {
                resolve(null);
                cleanup();
                return;
            }
            if (mode === "save" && e.key === "Enter") {
                const target = e.target;
                if (target !== searchInput) {
                    e.preventDefault();
                    e.stopPropagation();
                    await handleSaveAction();
                }
            }
        };

        const handleWindowResize = () => {
            applyBrowserLayout();
            renderContent(searchInput.value);
        };

        window.addEventListener("resize", handleWindowResize);

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        updateEditModeLayout();
        if (saveUsesEditPanel && editPanel?.loadDraft) {
            void editPanel.loadDraft(options?.saveDraft || {}, {
                category: selectedCategory,
                name: selectedSaveName,
            });
        }
        searchInput.focus();
    });
}

function standaloneOpenPromptBrowserForSave(options = {}) {
    const browserNode = options.node || {};
    const currentCategory = options.currentCategory || "Default";
    const currentPrompt = options.currentPrompt || "";
    return standaloneShowThumbnailBrowser(browserNode, currentCategory, currentPrompt, {
        mode: "save",
        onSave: options.onSave,
        title: options.title || "Save Workflow",
        saveButtonText: options.saveButtonText || "Save",
        namePlaceholder: options.namePlaceholder || "Prompt name",
        initialName: options.initialName || "",
        workflowOnly: options.workflowOnly === true || browserNode?._isWorkflowManager === true,
        promptOnly: options.promptOnly === true,
        contentFilter: options.contentFilter,
        filterEmptyCategories: options.filterEmptyCategories === true,
        showAllCategoriesToggle: options.showAllCategoriesToggle === true,
        hideContentFilterControl: options.hideContentFilterControl === true,
        contentFilterCycle: options.contentFilterCycle,
        onContentFilterChange: options.onContentFilterChange,
        saveWithEditPanel: options.saveWithEditPanel === true,
        showNSFWOnOpen: options.showNSFWOnOpen === true,
        saveDraft: options.saveDraft,
        endpointPrefix: options.endpointPrefix,
        loadPromptsFn: options.loadPromptsFn,
    });
}



export async function showThumbnailBrowser(node, currentCategory, currentPrompt, options = {}) {
    _syncThumbnailRenderState();
    return standaloneShowThumbnailBrowser(node, currentCategory, currentPrompt, options);
}

export async function openPromptBrowserForSave(options = {}) {
    return standaloneOpenPromptBrowserForSave(options);
}
