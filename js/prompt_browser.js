import { createPromptBrowserEditPanel, getPromptTypeChoices } from "./prompt_browser_edit.js";
import {
    buildSavePromptRequestBodyForEndpoint,
    getCategoryPromptEntriesForEndpoint,
    getCategoryPromptEntryForEndpoint,
    isHiddenPromptEntryKey,
} from "./prompt_store_adapters.js";

let app = null;
let UI = {
    overlay: "hsl(0 0% 0% / 0.8)",
    panel: "hsl(216 11% 15%)",
    panelBorder: "hsl(216 20% 65% / 0.24)",
    sectionBorder: "hsl(216 20% 65% / 0.20)",
    inputBg: "hsl(220 15% 10%)",
    inputBorder: "hsl(218 10% 41%)",
    buttonBg: "hsl(219 16% 18%)",
    cardBg: "hsl(219 16% 18%)",
    accentSoft: "hsl(208 73% 57% / 0.16)",
    accentBorder: "hsl(208 73% 57% / 0.65)",
};

let DEFAULT_THUMBNAIL = "";
let loadPrompts = async () => ({});
let showInfo = async () => {};
let showConfirm = async () => false;
let showRenameCategoryDialog = async () => null;
let showNewCategoryDialog = async () => null;
let ensureThumbnailRenderSelection = async () => null;
let resolveThumbnailFallbackBase = async () => null;
let generateThumbnailForPrompt = async () => null;
let generateThumbnailWorkflowFromWorkflowData = async () => null;
let buildRendererFallbackWorkflowDataWithBase = async () => null;
let showThumbnailRenderPicker = async () => null;
let saveThumbnailRenderSelection = () => {};
let _thumbnailLeafName = (path) => String(path || "");
let getThumbnailRenderState = null;
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
}

function getThumbnailRenderLabelParts() {
    _syncThumbnailRenderState();
    const selectedLoras = [_thumbnailRenderLora1, _thumbnailRenderLora2].filter(Boolean);
    if (_thumbnailRenderFamily && _thumbnailRenderModel) {
        const leafName = _thumbnailLeafName(_thumbnailRenderModel);
        return {
            shortLabel: typeSafeTruncate(`🔧 ${leafName}`, 18),
            iconLabel: "🔧",
            fullLabel: `Thumbnail: ${_thumbnailRenderFamily} : ${leafName}${selectedLoras.length ? ` + ${selectedLoras.length} LoRA${selectedLoras.length > 1 ? "s" : ""}` : ""}`,
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
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.7);
            z-index: 9999;
        `;

        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #222;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 20px;
            z-index: 10000;
            min-width: 320px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        `;

        dialog.innerHTML = `
            <div style="margin-bottom: 15px; font-size: 16px; font-weight: bold; color: #fff;">${title}</div>
            <div style="margin-bottom: 6px; color: #aaa; font-size: 12px;">${message}</div>
            <input type="text" value="${String(defaultValue || "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;")}" style="width: 100%; padding: 8px; margin-bottom: 15px; background: #333; border: 1px solid #555; color: #fff; border-radius: 4px; font-size: 14px; box-sizing: border-box;" />
            <div style="display: flex; gap: 10px; justify-content: flex-end;">
                <button class="cancel-btn" style="padding: 8px 16px; background: #555; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
                <button class="ok-btn" style="padding: 8px 16px; background: #0a0; color: #fff; border: none; border-radius: 4px; cursor: pointer;">OK</button>
            </div>
        `;

        const input = dialog.querySelector("input");
        const okBtn = dialog.querySelector(".ok-btn");
        const cancelBtn = dialog.querySelector(".cancel-btn");

        const cleanup = () => {
            if (overlay.parentNode) document.body.removeChild(overlay);
            if (dialog.parentNode) document.body.removeChild(dialog);
        };

        const handleOk = () => {
            resolve(String(input?.value || ""));
            cleanup();
        };

        const handleCancel = () => {
            resolve(null);
            cleanup();
        };

        okBtn.onclick = handleOk;
        cancelBtn.onclick = handleCancel;
        overlay.onclick = handleCancel;
        input.onkeydown = (event) => {
            if (event.key === "Enter") handleOk();
            if (event.key === "Escape") handleCancel();
        };

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        input.focus();
        input.select();
    });
}

function showNewPromptGroupDialog() {
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.7);
            z-index: 9999;
        `;

        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #222;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 20px;
            z-index: 10000;
            width: min(380px, calc(100vw - 36px));
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            color: #fff;
        `;

        dialog.innerHTML = `
            <div style="margin-bottom: 15px; font-size: 16px; font-weight: bold; color: #fff;">New Prompt Group</div>
            <div style="margin-bottom: 6px; color: #aaa; font-size: 12px;">Prompt group name</div>
            <input class="pm-group-name" type="text" value="" style="width: 100%; padding: 8px; margin-bottom: 12px; background: #333; border: 1px solid #555; color: #fff; border-radius: 4px; font-size: 14px; box-sizing: border-box;" />
            <div style="margin-bottom: 6px; color: #aaa; font-size: 12px;">Starting category</div>
            <input class="pm-category-name" type="text" value="" style="width: 100%; padding: 8px; margin-bottom: 15px; background: #333; border: 1px solid #555; color: #fff; border-radius: 4px; font-size: 14px; box-sizing: border-box;" />
            <div style="display: flex; gap: 10px; justify-content: flex-end;">
                <button class="cancel-btn" style="padding: 8px 16px; background: #555; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
                <button class="ok-btn" style="padding: 8px 16px; background: #0a0; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Create</button>
            </div>
        `;

        const groupInput = dialog.querySelector(".pm-group-name");
        const categoryInput = dialog.querySelector(".pm-category-name");
        const okBtn = dialog.querySelector(".ok-btn");
        const cancelBtn = dialog.querySelector(".cancel-btn");

        const cleanup = () => {
            if (overlay.parentNode) document.body.removeChild(overlay);
            if (dialog.parentNode) document.body.removeChild(dialog);
        };

        const handleOk = () => {
            resolve({
                groupName: String(groupInput?.value || ""),
                categoryName: String(categoryInput?.value || ""),
            });
            cleanup();
        };

        const handleCancel = () => {
            resolve(null);
            cleanup();
        };

        okBtn.onclick = handleOk;
        cancelBtn.onclick = handleCancel;
        overlay.onclick = handleCancel;
        groupInput.onkeydown = (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                categoryInput.focus();
                categoryInput.select();
            } else if (event.key === "Escape") {
                event.preventDefault();
                handleCancel();
            }
        };
        categoryInput.onkeydown = (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                handleOk();
            } else if (event.key === "Escape") {
                event.preventDefault();
                handleCancel();
            }
        };

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        groupInput.focus();
        groupInput.select();
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
    if (typeof deps.showInfo === "function") showInfo = deps.showInfo;
    if (typeof deps.showConfirm === "function") showConfirm = deps.showConfirm;
    if (typeof deps.showRenameCategoryDialog === "function") showRenameCategoryDialog = deps.showRenameCategoryDialog;
    if (typeof deps.showNewCategoryDialog === "function") showNewCategoryDialog = deps.showNewCategoryDialog;
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

function _ensureThumbQueueProgress() {
    if (_thumbQueueProgress) return;

    _thumbQueueCancelled = false;
    const progress = document.createElement("div");
    progress.style.cssText = `
        position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
        background: #222; border: 2px solid #4CAF50; border-radius: 8px;
        padding: 14px 16px; z-index: 10000; color: #fff; font-size: 14px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5); min-width: 320px;
    `;

    const progressRow = document.createElement("div");
    progressRow.style.cssText = `
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
    `;

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "✕";
    cancelBtn.title = "Cancel remaining thumbnails";
    cancelBtn.style.cssText = `
        width: 24px; height: 24px; line-height: 22px;
        border-radius: 6px; border: 1px solid #666;
        background: #2f2f2f; color: #eee;
        cursor: pointer; font-size: 14px; padding: 0;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    `;
    cancelBtn.onclick = () => {
        _thumbQueueCancelled = true;
        cancelBtn.disabled = true;
        cancelBtn.style.opacity = "0.6";
        cancelBtn.style.cursor = "default";
        cancelBtn.title = "Cancelling...";
    };

    const progressText = document.createElement("div");
    progressText.style.cssText = `display: flex; align-items: center; gap: 10px; min-width: 0; flex: 1;`;
    progressText.innerHTML = `
        <div style="width: 18px; height: 18px; border: 3px solid #4CAF50; border-top-color: transparent; border-radius: 50%; animation: thumb-spin 1s linear infinite; flex-shrink: 0;"></div>
        <span style="line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"></span>
    `;
    progressRow.appendChild(progressText);
    progressRow.appendChild(cancelBtn);
    progress.appendChild(progressRow);
    const styleEl = document.createElement("style");
    styleEl.textContent = `@keyframes thumb-spin { to { transform: rotate(360deg); } }`;
    progress.appendChild(styleEl);
    document.body.appendChild(progress);

    _thumbQueueProgress = { root: progress, text: progressText.querySelector("span"), cancelBtn };
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
        await fetch("/prompt-manager/log-thumbnail", {
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

    const categoryBasePrompt = getCategoryBasePrompt(node, category);
    const categoryPromptPrefix = getCategoryPromptPrefix(node, category);
    const prefixedPromptText = prependCategoryPromptPrefix(
        categoryPromptPrefix,
        _formatThumbnailPromptText(promptData.prompt || promptName || "", promptData)
    );
    const promptText = [String(categoryBasePrompt || "").trim(), prefixedPromptText]
        .filter(Boolean)
        .join(" ");

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
        const response = await fetch(url, {
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
        const response = await fetch(`${endpointPrefix}/delete-prompt`, {
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
        element.style.outline = "2px dashed #4CAF50";
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
                "#c00"
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

function buildDialogSelectOptionsHtml(items, selectedValue) {
    return (items || []).map((item) => {
        const isObject = item && typeof item === "object";
        const value = String(isObject ? item.value : item || "");
        const label = String(isObject ? item.label : item || "");
        const selected = value === String(selectedValue || "") ? " selected" : "";
        return `<option value="${value}"${selected}>${label}</option>`;
    }).join("");
}

function showPromptWithCategoryDialog(title, defaultName, categories, defaultCategory, options = {}) {
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 10000;
        `;

        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #222;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 16px;
            z-index: 10001;
            width: min(520px, calc(100vw - 36px));
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            color: #fff;
        `;

        const groupOptions = Array.isArray(options.groupOptions) ? options.groupOptions : null;
        const categoriesByGroup = options.categoriesByGroup && typeof options.categoriesByGroup === "object"
            ? options.categoriesByGroup
            : null;
        const fallbackGroupValue = groupOptions && groupOptions.length > 0
            ? String(groupOptions[0]?.value || "")
            : "";
        const defaultGroupValue = String(options.defaultGroupValue || fallbackGroupValue || "");
        const resolvedDefaultCategory = String(defaultCategory || "");
        const initialCategories = categoriesByGroup
            ? (categoriesByGroup[defaultGroupValue] || [])
            : (categories || []);
        const initialCategoryValue = initialCategories.includes(resolvedDefaultCategory)
            ? resolvedDefaultCategory
            : (initialCategories[0] || resolvedDefaultCategory);
        const groupFieldHtml = groupOptions && groupOptions.length > 0
            ? `
            <div style="margin-bottom: 6px; color: #aaa; font-size: 12px;">Prompt Group</div>
            <select class="group-select" style="width: 100%; padding: 8px; margin-bottom: 10px; background: #303030; border: 1px solid #555; color: #fff; border-radius: 4px; box-sizing: border-box;">${buildDialogSelectOptionsHtml(groupOptions, defaultGroupValue)}</select>
            `
            : "";

        dialog.innerHTML = `
            <div style="font-size: 16px; font-weight: bold; margin-bottom: 10px;">${title}</div>
            <div style="margin-bottom: 6px; color: #aaa; font-size: 12px;">Prompt name</div>
            <input class="name-input" type="text" value="${String(defaultName || "")}" style="width: 100%; padding: 8px; margin-bottom: 10px; background: #303030; border: 1px solid #555; color: #fff; border-radius: 4px; box-sizing: border-box;" />
            ${groupFieldHtml}
            <div style="margin-bottom: 6px; color: #aaa; font-size: 12px;">Category</div>
            <select class="cat-select" style="width: 100%; padding: 8px; margin-bottom: 12px; background: #303030; border: 1px solid #555; color: #fff; border-radius: 4px; box-sizing: border-box;">${buildDialogSelectOptionsHtml(initialCategories, initialCategoryValue)}</select>
            <div style="display: flex; justify-content: flex-end; gap: 10px;">
                <button class="cancel-btn" style="padding: 8px 14px; background: #555; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
                <button class="ok-btn" style="padding: 8px 14px; background: #2b7cff; color: #fff; border: none; border-radius: 4px; cursor: pointer;">OK</button>
            </div>
        `;

        const nameInput = dialog.querySelector(".name-input");
        const groupSelect = dialog.querySelector(".group-select");
        const catSelect = dialog.querySelector(".cat-select");
        const okBtn = dialog.querySelector(".ok-btn");
        const cancelBtn = dialog.querySelector(".cancel-btn");

        const syncCategoryOptions = () => {
            if (!categoriesByGroup || !groupSelect || !catSelect) return;
            const selectedGroupValue = String(groupSelect.value || defaultGroupValue || "");
            const groupCategories = Array.isArray(categoriesByGroup[selectedGroupValue])
                ? categoriesByGroup[selectedGroupValue]
                : [];
            const preferredCategory = selectedGroupValue === defaultGroupValue && groupCategories.includes(resolvedDefaultCategory)
                ? resolvedDefaultCategory
                : (groupCategories[0] || "");
            catSelect.innerHTML = buildDialogSelectOptionsHtml(groupCategories, preferredCategory);
            if (!catSelect.value && preferredCategory) {
                catSelect.value = preferredCategory;
            }
        };

        const cleanup = () => {
            if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
            if (dialog.parentNode) dialog.parentNode.removeChild(dialog);
        };

        const handleOk = () => {
            resolve({
                name: String(nameInput.value || ""),
                category: String(catSelect.value || resolvedDefaultCategory || ""),
                typeFile: String(groupSelect?.value || defaultGroupValue || ""),
            });
            cleanup();
        };

        const handleCancel = () => {
            resolve(null);
            cleanup();
        };

        okBtn.onclick = handleOk;
        cancelBtn.onclick = handleCancel;
        overlay.onclick = handleCancel;
        if (groupSelect) {
            groupSelect.onchange = () => syncCategoryOptions();
        }
        nameInput.onkeydown = (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                handleOk();
            } else if (e.key === "Escape") {
                e.preventDefault();
                handleCancel();
            }
        };

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        syncCategoryOptions();
        nameInput.focus();
        nameInput.select();
    });
}

function showCategoryPickerDialog(title, categories, defaultCategory, options = {}) {
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 10000;
        `;

        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #222;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 16px;
            z-index: 10001;
            width: min(460px, calc(100vw - 36px));
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            color: #fff;
        `;

        const groupOptions = Array.isArray(options.groupOptions) ? options.groupOptions : null;
        const categoriesByGroup = options.categoriesByGroup && typeof options.categoriesByGroup === "object"
            ? options.categoriesByGroup
            : null;
        const fallbackGroupValue = groupOptions && groupOptions.length > 0
            ? String(groupOptions[0]?.value || "")
            : "";
        const defaultGroupValue = String(options.defaultGroupValue || fallbackGroupValue || "");
        const resolvedDefaultCategory = String(defaultCategory || "");
        const initialCategories = categoriesByGroup
            ? (categoriesByGroup[defaultGroupValue] || [])
            : (categories || []);
        const initialCategoryValue = initialCategories.includes(resolvedDefaultCategory)
            ? resolvedDefaultCategory
            : (initialCategories[0] || resolvedDefaultCategory);
        const groupFieldHtml = groupOptions && groupOptions.length > 0
            ? `
            <div style="margin-bottom: 6px; color: #aaa; font-size: 12px;">Prompt Group</div>
            <select class="group-select" style="width: 100%; padding: 8px; margin-bottom: 10px; background: #303030; border: 1px solid #555; color: #fff; border-radius: 4px; box-sizing: border-box;">${buildDialogSelectOptionsHtml(groupOptions, defaultGroupValue)}</select>
            `
            : "";

        dialog.innerHTML = `
            <div style="font-size: 16px; font-weight: bold; margin-bottom: 10px;">${title}</div>
            ${groupFieldHtml}
            <div style="margin-bottom: 6px; color: #aaa; font-size: 12px;">Target category</div>
            <select class="cat-select" style="width: 100%; padding: 8px; margin-bottom: 12px; background: #303030; border: 1px solid #555; color: #fff; border-radius: 4px; box-sizing: border-box;">${buildDialogSelectOptionsHtml(initialCategories, initialCategoryValue)}</select>
            <div style="display: flex; justify-content: flex-end; gap: 10px;">
                <button class="cancel-btn" style="padding: 8px 14px; background: #555; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
                <button class="ok-btn" style="padding: 8px 14px; background: #2b7cff; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Move</button>
            </div>
        `;

        const groupSelect = dialog.querySelector(".group-select");
        const catSelect = dialog.querySelector(".cat-select");
        const okBtn = dialog.querySelector(".ok-btn");
        const cancelBtn = dialog.querySelector(".cancel-btn");

        const syncCategoryOptions = () => {
            if (!categoriesByGroup || !groupSelect || !catSelect) return;
            const selectedGroupValue = String(groupSelect.value || defaultGroupValue || "");
            const groupCategories = Array.isArray(categoriesByGroup[selectedGroupValue])
                ? categoriesByGroup[selectedGroupValue]
                : [];
            const preferredCategory = selectedGroupValue === defaultGroupValue && groupCategories.includes(resolvedDefaultCategory)
                ? resolvedDefaultCategory
                : (groupCategories[0] || "");
            catSelect.innerHTML = buildDialogSelectOptionsHtml(groupCategories, preferredCategory);
            if (!catSelect.value && preferredCategory) {
                catSelect.value = preferredCategory;
            }
        };

        const cleanup = () => {
            if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
            if (dialog.parentNode) dialog.parentNode.removeChild(dialog);
        };

        const handleOk = () => {
            resolve({
                category: String(catSelect.value || resolvedDefaultCategory || ""),
                typeFile: String(groupSelect?.value || defaultGroupValue || ""),
            });
            cleanup();
        };

        const handleCancel = () => {
            resolve(null);
            cleanup();
        };

        okBtn.onclick = handleOk;
        cancelBtn.onclick = handleCancel;
        overlay.onclick = handleCancel;
        if (groupSelect) {
            groupSelect.onchange = () => syncCategoryOptions();
        }
        catSelect.onkeydown = (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                handleOk();
            } else if (e.key === "Escape") {
                e.preventDefault();
                handleCancel();
            }
        };

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        syncCategoryOptions();
        catSelect.focus();
    });
}

function showThumbnailContextMenu(event, node, category, promptName, onUpdate, endpointPrefix = "/prompt-manager-advanced", options = {}) {
    const onDelete = typeof options?.onDelete === "function" ? options.onDelete : null;
    const existing = document.querySelector('.thumbnail-context-menu');
    if (existing) existing.remove();
    const isComposerManager = endpointPrefix === "/prompt-manager/compose";
    const composerCategory = isComposerManager ? getComposerCategoryRequestIdentity(node, category) : null;
    const categoryLabel = composerCategory?.categoryName || category;

    const menu = document.createElement("div");
    menu.className = "thumbnail-context-menu";
    menu.style.cssText = `
        position: fixed;
        left: ${event.clientX}px;
        top: ${event.clientY}px;
        background: #2a2a2a;
        border: 1px solid #444;
        border-radius: 6px;
        padding: 4px 0;
        z-index: 10001;
        min-width: 150px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    `;

    const createMenuItem = (label, onClick) => {
        const item = document.createElement("div");
        item.textContent = label;
        item.style.cssText = `
            padding: 8px 16px;
            color: #ccc;
            cursor: pointer;
            font-size: 13px;
        `;
        item.onmouseover = () => item.style.background = '#3a3a3a';
        item.onmouseout = () => item.style.background = 'transparent';
        item.onclick = () => {
            menu.remove();
            onClick();
        };
        return item;
    };

    menu.appendChild(createMenuItem("📋 Paste Thumbnail", async () => {
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
    }));

    const genDivider = document.createElement("div");
    genDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
    menu.appendChild(genDivider);

    menu.appendChild(createMenuItem("🎨 Generate Thumbnail", async () => {
        const queuedName = promptName;
        _thumbQueueTotal++;
        _ensureThumbQueueProgress();
        _updateThumbQueueProgress(queuedName);

        _thumbQueuePromiseChain = _thumbQueuePromiseChain.then(async () => {
            if (_thumbQueueCancelled) {
                _thumbQueueDone++;
                if (_thumbQueueDone + _thumbQueueFailed >= _thumbQueueTotal) {
                    _finishThumbQueueProgress();
                }
                return;
            }
            _updateThumbQueueProgress(queuedName);
            try {
                await _generateThumbnailForBrowserCategory(node, category, queuedName, onUpdate, { endpointPrefix });
                _thumbQueueDone++;
            } catch (e) {
                console.error(`[ThumbnailGen] Failed for "${queuedName}":`, e);
                _thumbQueueFailed++;
            } finally {
                if (_thumbQueueProgress && _thumbQueueDone + _thumbQueueFailed >= _thumbQueueTotal) {
                    _finishThumbQueueProgress();
                }
            }
        });
    }));

    const promptData = getCategoryPromptEntry(node?.prompts?.[category], promptName, endpointPrefix);
    if (promptData?.thumbnail) {
        const divider = document.createElement("div");
        divider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
        menu.appendChild(divider);

        menu.appendChild(createMenuItem("🗑️ Remove Thumbnail", async () => {
            await saveThumbnailEntry(node, category, promptName, null, endpointPrefix);
            onUpdate();
        }));
    }

    const nsfwDivider = document.createElement("div");
    nsfwDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
    menu.appendChild(nsfwDivider);

    const isNSFW = promptData?.nsfw === true;
    const nsfwItem = createMenuItem(isNSFW ? "✓ NSFW" : "Mark as NSFW", async () => {
        try {
            const resp = await fetch(`${endpointPrefix}/toggle-nsfw`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    type: "prompt",
                    category: categoryLabel,
                    name: promptName,
                    ...(composerCategory?.typeFile ? { type_file: composerCategory.typeFile } : {}),
                })
            });
            const result = await resp.json();
            if (result.success) {
                applyPromptPayloadToNode(node, result);
                onUpdate();
            }
        } catch (err) {
            console.error("[PromptBrowser] Error toggling prompt NSFW:", err);
        }
    });
    nsfwItem.style.color = isNSFW ? '#f66' : '#ccc';
    menu.appendChild(nsfwItem);

    const renameDivider = document.createElement("div");
    renameDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
    menu.appendChild(renameDivider);

    menu.appendChild(createMenuItem("✏️ Rename / Move", async () => {
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
        if (result && result.name && result.name.trim()) {
            const newName = result.name.trim();
            const newCat = result.category;
            const newTypeFile = isComposerManager ? String(result.typeFile || currentTypeFile || "") : "";
            if (newName === promptName && newCat === categoryLabel && (!isComposerManager || newTypeFile === currentTypeFile)) return;
            try {
                const resp = await fetch(`${endpointPrefix}/rename-prompt`, {
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
                    })
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
        }
    }));

    const deleteDivider = document.createElement("div");
    deleteDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
    menu.appendChild(deleteDivider);

    menu.appendChild(createMenuItem("🗑️ Delete Prompt", async () => {
        if (await showConfirm("Delete Prompt", `Are you sure you want to delete prompt "${promptName}"?`)) {
            await deletePromptEntry(node, category, promptName, endpointPrefix);
            await onDelete?.(category, promptName);
            onUpdate();
        }
    }));
    menu.lastChild.style.color = '#f66';

    const closeMenu = (e) => {
        if (!menu.contains(e.target)) {
            menu.remove();
            document.removeEventListener('mousedown', closeMenu, true);
        }
    };
    setTimeout(() => {
        document.addEventListener('mousedown', closeMenu, true);
    }, 10);

    document.body.appendChild(menu);
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

function showEditBasePromptDialog(categoryName, currentValue) {
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 10000;
        `;

        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #222;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 16px;
            z-index: 10001;
            width: min(680px, calc(100vw - 36px));
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            color: #fff;
        `;

        dialog.innerHTML = `
            <div style="font-size: 16px; font-weight: bold; margin-bottom: 8px;">Edit Base Prompt</div>
            <div style="font-size: 12px; color: #aaa; margin-bottom: 10px;">Category: ${categoryName}</div>
            <textarea style="width: 100%; min-height: 140px; resize: vertical; background: #303030; border: 1px solid #555; color: #fff; border-radius: 4px; padding: 8px; box-sizing: border-box; font-size: 13px;"></textarea>
            <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 12px;">
                <button class="cancel-btn" style="padding: 8px 14px; background: #555; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
                <button class="save-btn" style="padding: 8px 14px; background: #2b7cff; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Save</button>
            </div>
        `;

        const textarea = dialog.querySelector("textarea");
        const saveBtn = dialog.querySelector(".save-btn");
        const cancelBtn = dialog.querySelector(".cancel-btn");
        textarea.value = String(currentValue || "");

        const cleanup = () => {
            if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
            if (dialog.parentNode) dialog.parentNode.removeChild(dialog);
        };

        const handleSave = () => {
            resolve(String(textarea.value || ""));
            cleanup();
        };

        const handleCancel = () => {
            resolve(null);
            cleanup();
        };

        saveBtn.onclick = handleSave;
        cancelBtn.onclick = handleCancel;
        overlay.onclick = handleCancel;

        textarea.onkeydown = (e) => {
            if (e.key === "Escape") {
                e.preventDefault();
                e.stopPropagation();
                handleCancel();
            }
        };

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        textarea.focus();
        textarea.setSelectionRange(textarea.value.length, textarea.value.length);
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
    return (saved === "prompt" || saved === "recipe" || saved === "compose" || saved === "all") ? saved : "all";
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

function hasComposeLikePayload(promptData) {
    if (!promptData || typeof promptData !== "object") return false;
    const savedFrom = String(promptData.saved_from || "").trim();
    if (savedFrom === "PromptComposerManager" || savedFrom === "ComposerManager") return true;
    const workflowData = parseWorkflowDataCandidate(promptData.workflow_data);
    if (!workflowData || typeof workflowData !== "object") return false;
    const workflowSource = String(workflowData._source || "").trim();
    if (workflowSource === "PromptComposerManager" || workflowSource === "ComposerManager") return true;
    return workflowData.prompt_composer && typeof workflowData.prompt_composer === "object";
}

function hasRecipeLikePayload(promptData) {
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
        .sort((a, b) => a.localeCompare(b));

    if (hideNSFW) {
        promptNames = promptNames.filter((name) => promptEntries?.[name]?.nsfw !== true);
    }

    if (workflowOnly || contentFilter !== "all") {
        promptNames = promptNames.filter((name) => {
            const entry = promptEntries?.[name];
            const hasComposeData = hasComposeLikePayload(entry);
            const hasRecipeData = !hasComposeData && hasRecipeLikePayload(entry);
            const hasPromptData = !hasComposeData && hasPromptPresetPayload(entry);

            if (workflowOnly && !(hasComposeData || hasRecipeData || hasPromptData)) return false;
            if (contentFilter === "prompt") return hasPromptData;
            if (contentFilter === "compose") return hasComposeData;
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
    const initialContentFilter = (() => {
        const raw = String(options?.contentFilter || "").trim().toLowerCase();
        return raw === "prompt" || raw === "recipe" || raw === "compose" || raw === "all" ? raw : "";
    })();
    const filterEmptyCategories = options?.filterEmptyCategories === true;
    const useComposerMultiSelectActions = multiSelectActionMode === "composer-add";
    let editMode = allowEditMode && options?.editMode === true;
    let multiSelectMode = startInMultiSelect;
    let updateSelectButton = () => {};
    let updateFooterText = () => {};
    let updateSelectionToolbar = () => {};

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

            if (editMode && editPanel && typeof editPanel.clearPrompt === "function") {
                await editPanel.clearPrompt({ skipConfirm: true });
            }
        };
        const getResultCategoryName = (category = selectedCategory) => showCategoryTypeFilter
            ? getComposerCategoryDisplayName(node, category)
            : String(category || "").trim();
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
                if (typeof editPanel.showCategorySettings === "function") {
                    editPanel.showCategorySettings();
                }
            }
        };

        const canChangeEditorContext = async () => {
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
        console.log(
            `[PromptBrowser] viewport=${window.innerWidth}x${window.innerHeight}, ` +
            `pref=${compactBrowserPref}, compact=${compactBrowser}`
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
        const getEditPanelWidth = () => {
            if (compactBrowser) return 280;
            return showCategoryTypeFilter && typeRailExpanded ? 314 : 320;
        };

        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: ${getDialogLeftPosition()}px;
            transform: translateY(-50%);
            background: ${UI.panel};
            border: 1px solid ${UI.panelBorder};
            border-radius: 12px;
            padding: 16px;
            z-index: 10000;
            width: ${getDialogContentWidth()}px;
            max-height: 95vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
            user-select: none;
            -webkit-user-select: none;
        `;

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

        // Header with close button
        const header = document.createElement("div");
        header.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid ${UI.sectionBorder};
        `;

        const title = document.createElement("div");
        title.textContent = mode === "save"
            ? saveTitle
            : (selectTitle || (workflowOnly ? "Select Recipe" : "Select Prompt"));
        title.style.cssText = `
            font-size: 18px;
            font-weight: bold;
            color: #fff;
        `;

        const closeBtn = document.createElement("button");
        closeBtn.textContent = "✕";
        closeBtn.style.cssText = `
            background: transparent;
            border: none;
            color: #888;
            font-size: 20px;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
        `;
        closeBtn.onmouseover = () => closeBtn.style.color = "#fff";
        closeBtn.onmouseout = () => closeBtn.style.color = "#888";

        header.appendChild(title);
        header.appendChild(closeBtn);

        // Controls bar: Search + NSFW button + View Mode button
        const controlsBar = document.createElement("div");
        controlsBar.style.cssText = `
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid ${UI.sectionBorder};
        `;

        // Search input (fills remaining space)
        const searchInput = document.createElement("input");
        searchInput.type = "text";
        searchInput.placeholder = workflowOnly ? "Search recipes and prompts..." : "Search prompts...";
        searchInput.style.cssText = `
            flex: 1;
            min-width: 0;
            padding: 6px 10px;
            background: ${UI.inputBg};
            border: 1px solid ${UI.inputBorder};
            border-radius: 4px;
            color: #fff;
            font-size: 13px;
            box-sizing: border-box;
            outline: none;
            user-select: text;
            -webkit-user-select: text;
        `;
        searchInput.onfocus = () => searchInput.style.borderColor = UI.accent;
        searchInput.onblur = () => searchInput.style.borderColor = UI.inputBorder;

        // Wrap search input in a container with clear button
        const searchWrapper = document.createElement("div");
        searchWrapper.style.cssText = `
            flex: 1;
            min-width: 0;
            position: relative;
            display: flex;
            align-items: center;
        `;
        searchInput.style.flex = "1";
        searchInput.style.paddingRight = "28px";
        const clearBtn = document.createElement("span");
        clearBtn.textContent = "\u00d7";
        clearBtn.title = "Clear search";
        clearBtn.style.cssText = `
            position: absolute;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
            color: #666;
            font-size: 16px;
            cursor: pointer;
            line-height: 1;
            display: none;
            user-select: none;
        `;
        clearBtn.onmouseover = () => clearBtn.style.color = "#fff";
        clearBtn.onmouseout = () => clearBtn.style.color = "#666";
        clearBtn.onclick = () => {
            searchInput.value = "";
            clearBtn.style.display = "none";
            searchInput.focus();
            rebuildCategoryList();
            renderContent("");
        };
        const origOninput = searchInput.oninput;
        searchInput.addEventListener("input", () => {
            clearBtn.style.display = searchInput.value ? "" : "none";
        });
        searchWrapper.appendChild(searchInput);
        searchWrapper.appendChild(clearBtn);

        // Prompt/Recipe/Compose/All filter button
        let contentFilterState = initialContentFilter || getBrowserContentFilter(app);
        const contentFilterBtn = document.createElement("button");

        // For prompt-only stores (e.g. Prompt Composer) the content filter has no meaning.
        if (promptOnly) {
            contentFilterState = "prompt";
        }

        // NSFW toggle button
        let hideNSFWState = getHideNSFW(app);
        const nsfwBtn = document.createElement("button");
        const btnStyle = `
            background: ${UI.buttonBg};
            border: 1px solid ${UI.inputBorder};
            color: #aaa;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            white-space: nowrap;
        `;
        const updateNsfwBtn = () => {
            if (hideNSFWState) {
                nsfwBtn.textContent = "NSFW: Hidden";
                nsfwBtn.style.cssText = btnStyle + `background: #453339; border-color: #9b727b; color: #d2a8b0;`;
            } else {
                nsfwBtn.textContent = "NSFW: Visible";
                nsfwBtn.style.cssText = btnStyle;
            }
            nsfwBtn.title = hideNSFWState ? "NSFW content is hidden — click to show" : "NSFW content is visible — click to hide";
        };
        const updateContentFilterBtn = () => {
            if (contentFilterState === "prompt") {
                contentFilterBtn.textContent = "Type: Prompt";
                contentFilterBtn.style.cssText = btnStyle + `background: rgba(56, 130, 246, 0.18); border-color: rgba(56, 130, 246, 0.75); color: #dbeafe;`;
                contentFilterBtn.title = "Showing prompt entries only";
            } else if (contentFilterState === "compose") {
                contentFilterBtn.textContent = "Type: Compose";
                contentFilterBtn.style.cssText = btnStyle + `background: rgba(47, 111, 146, 0.18); border-color: rgba(47, 111, 146, 0.82); color: #d7edf8;`;
                contentFilterBtn.title = "Showing compose entries only";
            } else if (contentFilterState === "recipe") {
                contentFilterBtn.textContent = "Type: Recipe";
                contentFilterBtn.style.cssText = btnStyle + `background: rgba(235, 140, 35, 0.18); border-color: rgba(235, 140, 35, 0.8); color: #ffe7c2;`;
                contentFilterBtn.title = "Showing recipe entries only";
            } else {
                contentFilterBtn.textContent = "Type: All";
                contentFilterBtn.style.cssText = btnStyle;
                contentFilterBtn.title = "Showing prompts, recipes, and compose entries";
            }
        };
        updateContentFilterBtn();
        updateNsfwBtn();
        const thumbnailModelBtn = document.createElement("button");
        const updateThumbnailModelBtn = () => {
            const label = getThumbnailRenderLabelParts();
            thumbnailModelBtn.textContent = label.iconLabel || "🔧";
            thumbnailModelBtn.title = label.fullLabel;
            thumbnailModelBtn.style.cssText = btnStyle;
        };
        thumbnailModelBtn.onmouseover = () => { thumbnailModelBtn.style.background = '#38414c'; thumbnailModelBtn.style.color = '#fff'; };
        thumbnailModelBtn.onmouseout = () => { thumbnailModelBtn.style.background = '#313843'; thumbnailModelBtn.style.color = '#aaa'; };
        thumbnailModelBtn.onclick = async () => {
            const picked = await showThumbnailRenderPicker(
                _thumbnailRenderFamily,
                _thumbnailRenderModel,
                _thumbnailRenderLora1,
                _thumbnailRenderLora2,
            );
            if (picked) {
                saveThumbnailRenderSelection(picked);
                updateThumbnailModelBtn();
                editPanel?.refreshThumbnailModelButton?.();
            }
        };
        updateThumbnailModelBtn();
        contentFilterBtn.onmouseover = () => {
            if (contentFilterState === "all") {
                contentFilterBtn.style.background = "#38414c";
                contentFilterBtn.style.color = "#fff";
            }
        };
        contentFilterBtn.onmouseout = () => {
            if (contentFilterState === "all") {
                contentFilterBtn.style.background = "#313843";
                contentFilterBtn.style.color = "#aaa";
            }
        };
        nsfwBtn.onmouseover = () => { if (!hideNSFWState) { nsfwBtn.style.background = '#38414c'; nsfwBtn.style.color = '#fff'; } };
        nsfwBtn.onmouseout = () => { if (!hideNSFWState) { nsfwBtn.style.background = '#313843'; nsfwBtn.style.color = '#aaa'; } };

        // View mode toggle button
        let currentViewMode = getViewMode(compactBrowser, browserPrefScope);
        const viewModeBtn = document.createElement("button");
        const updateViewModeBtn = () => {
            const labels = {
                grid: "⊞ Grid",
                icon: "⊞ Icon",
                list: "☰ List",
            };
            viewModeBtn.textContent = labels[currentViewMode] || labels.grid;
            const titles = {
                grid: "Large thumbnail grid",
                icon: "Small icon grid",
                list: "Compact list view",
            };
            viewModeBtn.title = titles[currentViewMode] || titles.grid;
        };
        viewModeBtn.style.cssText = btnStyle;
        viewModeBtn.onmouseover = () => { viewModeBtn.style.background = '#38414c'; viewModeBtn.style.color = '#fff'; };
        viewModeBtn.onmouseout = () => { viewModeBtn.style.background = '#313843'; viewModeBtn.style.color = '#aaa'; };
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
            multiSelectBtn = document.createElement("button");
            updateMultiSelectBtn = () => {
                if (multiSelectMode) {
                    multiSelectBtn.textContent = "☑ Multi: On";
                    multiSelectBtn.style.cssText = btnStyle + `background: rgba(56, 130, 246, 0.22); border-color: rgba(56, 130, 246, 0.85); color: #dbeafe;`;
                    multiSelectBtn.title = "Multi-select is on";
                } else {
                    multiSelectBtn.textContent = "☐ Multi: Off";
                    multiSelectBtn.style.cssText = btnStyle;
                    multiSelectBtn.title = "Turn on multi-select";
                }
            };
            multiSelectBtn.onmouseover = () => {
                if (!multiSelectMode) { multiSelectBtn.style.background = '#38414c'; multiSelectBtn.style.color = '#fff'; }
            };
            multiSelectBtn.onmouseout = () => {
                updateMultiSelectBtn();
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
            editModeBtn = document.createElement("button");
            updateEditModeBtn = () => {
                if (editMode) {
                    editModeBtn.textContent = "✎ Edit: On";
                    editModeBtn.style.cssText = btnStyle + `background: rgba(56, 130, 246, 0.22); border-color: rgba(56, 130, 246, 0.85); color: #dbeafe;`;
                    editModeBtn.title = "Edit mode is on — click to turn off";
                } else {
                    editModeBtn.textContent = "✎ Edit: Off";
                    editModeBtn.style.cssText = btnStyle;
                    editModeBtn.title = "Toggle edit mode";
                }
            };
            editModeBtn.onmouseover = () => {
                if (!editMode) { editModeBtn.style.background = '#38414c'; editModeBtn.style.color = '#fff'; }
            };
            editModeBtn.onmouseout = () => {
                updateEditModeBtn();
            };
            editModeBtn.onclick = () => {
                const nextEditMode = !editMode;
                if (nextEditMode && multiSelectMode) {
                    disableMultiSelect();
                }
                editMode = nextEditMode;
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
        if (!allowedCategories && !promptOnly) {
            controlsBar.appendChild(contentFilterBtn);
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
        const categoryBar = document.createElement("div");
        categoryBar.style.cssText = `
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
            padding: 8px 0;
            border-top: 1px solid rgba(255, 255, 255, 0.15);
            border-bottom: 1px solid rgba(255, 255, 255, 0.15);
            min-height: 0;
        `;

        // Static controls (always visible, left side)
        const categoryControls = document.createElement("div");
        categoryControls.style.cssText = `
            display: flex;
            gap: 6px;
            flex-shrink: 0;
            align-items: flex-start;
        `;

        const addCategoryBtn = document.createElement("button");
        addCategoryBtn.textContent = "+";
        addCategoryBtn.title = "New Category";
        addCategoryBtn.style.cssText = `
            padding: 6px 14px;
            border-radius: 6px;
            border: 1px solid #5f6773;
            background: #313843;
            color: #aaa;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.15s ease;
            flex-shrink: 0;
            margin-top: 1px;
        `;
        const updateCategoryCreateControls = () => {
            const showCategoryCreate = !showCategoryTypeFilter || categoryTypeFilter !== "__all__";
            addCategoryBtn.style.display = showCategoryCreate ? "" : "none";
            categoryContainerLeftBorder.style.display = showCategoryCreate ? "block" : "none";
        };
        addCategoryBtn.onmouseover = () => { addCategoryBtn.style.background = '#38414c'; addCategoryBtn.style.color = '#fff'; };
        addCategoryBtn.onmouseout = () => { addCategoryBtn.style.background = '#313843'; addCategoryBtn.style.color = '#aaa'; };
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
                    const resp = await fetch(`${endpointPrefix}/save-category`, {
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
                        if (editMode && editPanel && typeof editPanel.clearPrompt === "function") {
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
        const categoryContainer = document.createElement("div");
        categoryContainer.className = "category-scroll-bar";
        categoryContainer.style.cssText = `
            display: flex;
            gap: 6px;
            flex-wrap: nowrap;
            align-items: center;
            overflow-x: auto;
            overflow-y: hidden;
            scrollbar-width: thin;
            scrollbar-color: rgba(74, 138, 212, 0.65) transparent;
            flex: 1;
            min-width: 0;
            padding-right: 1px;
        `;
        categoryContainer.addEventListener("wheel", (e) => {
            if (e.deltaY !== 0) {
                e.preventDefault();
                categoryContainer.scrollLeft += e.deltaY;
            }
        }, { passive: false });

        const categoryContainerLeftBorder = document.createElement("div");
        categoryContainerLeftBorder.style.cssText = `
            width: 1px;
            align-self: stretch;
            background: rgba(255, 255, 255, 0.22);
            margin: 0 2px;
            flex-shrink: 0;
            display: block;
        `;

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
            allCategories = filterAllowedCategories(getVisibleCategories(node, {
                hideNSFW: hideNSFWState,
                workflowOnly,
                contentFilter: contentFilterState,
                filterEmptyCategories,
                endpointPrefix,
            }));
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
                const firstVisible = categories.find(c => !isCategoryNSFW(c));
                if (firstVisible) {
                    newCategory = firstVisible;
                }
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
                background: ${UI.panel};
                border: 1px solid ${UI.accentBorder};
                border-radius: 8px;
                color: ${UI.textPrimary || "#ddd"};
                font-size: 13px;
                line-height: 1.2;
                padding: 8px 10px;
                box-shadow: 0 10px 28px rgba(0,0,0,0.55);
                z-index: 10003;
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
            const highlight = "rgba(147, 197, 253, 1)";
            const glow = "rgba(56, 130, 246, 0.5)";

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
                const resp = await fetch(`${endpointPrefix}/reorder-types`, {
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
                btn.style.background = isSelected ? "rgba(56, 130, 246, 0.22)" : "transparent";
                if (isNSFW && !isSelected) {
                    btn.style.borderColor = "#944";
                } else {
                    btn.style.borderColor = isSelected ? "rgba(56, 130, 246, 0.85)" : "rgba(95, 103, 115, 0.75)";
                }
                btn.style.color = isSelected ? "#dbeafe" : "#c7d0db";
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
                    btn.style.background = "rgba(56, 130, 246, 0.3)";
                    btn.style.borderColor = "rgba(56, 130, 246, 1)";
                    btn.style.color = "#eef6ff";
                } else if (isDropHighlight) {
                    btn.style.background = isSelected ? "rgba(56, 130, 246, 0.28)" : "rgba(147, 197, 253, 0.1)";
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
                    border-radius: 10px;
                    border: 1px solid rgba(95, 103, 115, 0.75);
                    background: transparent;
                    color: #c7d0db;
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
                border-radius: 8px;
                border: 1px solid ${UI.inputBorder};
                background: ${UI.buttonBg};
                color: #c7d0db;
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
                border-radius: 6px;
                border: 1px solid #5f6773;
                background: #313843;
                color: #aaa;
                cursor: pointer;
                font-size: 13px;
                white-space: nowrap;
                overflow: hidden;
                box-sizing: border-box;
                transition: all 0.15s ease;
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
                addTypeBtn.style.background = "#38414c";
                addTypeBtn.style.color = "#fff";
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
                addTypeBtn.style.background = "#313843";
                addTypeBtn.style.color = "#aaa";
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
                    const resp = await fetch(`${endpointPrefix}/save-type-settings`, {
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
                            if (typeof editPanel.clearPrompt === "function") {
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

                btn.style.background = isSelected ? '#4a8ad4' : '#2a2a2a';
                btn.style.background = isSelected ? UI.accentSoft : UI.buttonBg;
                btn.style.color = isSelected ? '#fff' : '#aaa';

                // NSFW categories get a red border, otherwise normal
                if (isNSFW && !isSelected) {
                    btn.style.borderColor = '#944';
                } else {
                    btn.style.borderColor = isSelected ? '#5a9ae4' : '#444';
                }

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
            const existing = document.querySelector('.category-context-menu');
            if (existing) existing.remove();
            const isComposerManager = endpointPrefix === "/prompt-manager/compose";

            const menu = document.createElement("div");
            menu.className = "category-context-menu";
            menu.style.cssText = `
                position: fixed;
                left: ${event.clientX}px;
                top: ${event.clientY}px;
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 4px 0;
                z-index: 10001;
                min-width: 150px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            `;

            const isNSFW = isCategoryNSFW(cat);
            const categoryLabel = showCategoryTypeFilter ? getComposerCategoryDisplayName(node, cat) : cat;
            const item = document.createElement("div");
            item.textContent = isNSFW ? "✓ NSFW" : "Mark as NSFW";
            item.style.cssText = `
                padding: 8px 16px;
                color: ${isNSFW ? '#f66' : '#ccc'};
                cursor: pointer;
                font-size: 13px;
            `;
            item.onmouseover = () => item.style.background = '#3a3a3a';
            item.onmouseout = () => item.style.background = 'transparent';
            item.onclick = async () => {
                menu.remove();
                try {
                    const resp = await fetch(`${endpointPrefix}/toggle-nsfw`, {
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
            };
            menu.appendChild(item);

            // Rename Category
            const renameDivider = document.createElement("div");
            renameDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
            menu.appendChild(renameDivider);

            const renameItem = document.createElement("div");
            renameItem.textContent = isComposerManager ? "✏️ Rename / Move" : "✏️ Rename";
            renameItem.style.cssText = `
                padding: 8px 16px;
                color: #ccc;
                cursor: pointer;
                font-size: 13px;
            `;
            renameItem.onmouseover = () => renameItem.style.background = '#3a3a3a';
            renameItem.onmouseout = () => renameItem.style.background = 'transparent';
            renameItem.onclick = async () => {
                menu.remove();
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
                        const resp = await fetch(`${endpointPrefix}/rename-category`, {
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
            };
            menu.appendChild(renameItem);

            // Generate Missing Thumbnails
            const thumbDivider = document.createElement("div");
            thumbDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
            menu.appendChild(thumbDivider);

            const thumbItem = document.createElement("div");
            const buildProgressUI = () => {
                const progress = document.createElement("div");
                progress.style.cssText = `
                    position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                    background: #222; border: 2px solid #4CAF50; border-radius: 8px;
                    padding: 14px 16px; z-index: 10000; color: #fff; font-size: 14px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.5); min-width: 320px;
                `;

                const progressRow = document.createElement("div");
                progressRow.style.cssText = `
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 14px;
                `;

                const cancelBtn = document.createElement("button");
                cancelBtn.textContent = "✕";
                cancelBtn.title = "Cancel remaining thumbnails";
                cancelBtn.style.cssText = `
                    width: 24px; height: 24px; line-height: 22px;
                    border-radius: 6px; border: 1px solid #666;
                    background: #2f2f2f; color: #eee;
                    cursor: pointer; font-size: 14px; padding: 0;
                    display: flex; align-items: center; justify-content: center;
                    flex-shrink: 0;
                `;

                const progressText = document.createElement("div");
                progressText.style.cssText = `display: flex; align-items: center; gap: 10px; min-width: 0; flex: 1;`;
                progressText.innerHTML = `
                    <div style="width: 18px; height: 18px; border: 3px solid #4CAF50; border-top-color: transparent; border-radius: 50%; animation: thumb-spin 1s linear infinite; flex-shrink: 0;"></div>
                    <span style="line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"></span>
                `;
                progressRow.appendChild(progressText);
                progressRow.appendChild(cancelBtn);
                progress.appendChild(progressRow);
                const styleEl = document.createElement("style");
                styleEl.textContent = `@keyframes thumb-spin { to { transform: rotate(360deg); } }`;
                progress.appendChild(styleEl);
                document.body.appendChild(progress);
                return { progress, progressText, cancelBtn };
            };

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
                if (!await showConfirm(title, confirmMessage, confirmText, "#4CAF50")) {
                    return;
                }

                // Ensure renderer selection is ready for prompts without saved workflow_data.
                const renderSelection = await ensureThumbnailRenderSelection();
                if (!renderSelection) return;

                // Resolve family-compatible defaults once for the batch.
                const fallbackBase = await resolveThumbnailFallbackBase(renderSelection);

                for (let i = 0; i < names.length; i++) {
                    const pName = names[i];
                    _thumbQueueTotal++;
                    _ensureThumbQueueProgress();
                    _updateThumbQueueProgress(pName);

                    _thumbQueuePromiseChain = _thumbQueuePromiseChain.then(async () => {
                        if (_thumbQueueCancelled) {
                            _thumbQueueDone++;
                            if (_thumbQueueDone + _thumbQueueFailed >= _thumbQueueTotal) {
                                _finishThumbQueueProgress();
                            }
                            return;
                        }
                        _updateThumbQueueProgress(pName);
                        try {
                            await _generateThumbnailForBrowserCategory(node, cat, pName, () => {
                                renderContent(searchInput.value);
                            }, {
                                renderSelection,
                                fallbackBase,
                                endpointPrefix,
                            });
                            _thumbQueueDone++;
                        } catch (e) {
                            console.error(`[ThumbnailGen] Failed for "${pName}":`, e);
                            _thumbQueueFailed++;
                        } finally {
                            if (_thumbQueueProgress && _thumbQueueDone + _thumbQueueFailed >= _thumbQueueTotal) {
                                _finishThumbQueueProgress();
                            }
                        }
                    });
                }
            };

            // Generate Missing Thumbnails
            const missingItem = document.createElement("div");
            missingItem.textContent = "🎨 Generate Missing Thumbnails";
            missingItem.style.cssText = `
                padding: 8px 16px;
                color: #ccc;
                cursor: pointer;
                font-size: 13px;
            `;
            missingItem.onmouseover = () => missingItem.style.background = '#3a3a3a';
            missingItem.onmouseout = () => missingItem.style.background = 'transparent';
            missingItem.onclick = async () => {
                menu.remove();
                const catPrompts = node.prompts[cat];
                if (!catPrompts) return;

                // Collect prompts without thumbnails
                const missing = getPromptNamesForCategory(node, cat, { hideNSFW: false, workflowOnly: false, contentFilter: "all", endpointPrefix }).filter(name => {
                    const data = getCategoryPromptEntry(catPrompts, name, endpointPrefix);
                    return data && typeof data === "object" && !data.thumbnail;
                });

                await runBatchGeneration(missing, "Generate Missing Thumbnails");
            };
            menu.appendChild(missingItem);

            // Regenerate All Thumbnails
            const regenerateItem = document.createElement("div");
            regenerateItem.textContent = "🔄 Re-Generate All Thumbnails";
            regenerateItem.style.cssText = `
                padding: 8px 16px;
                color: #ccc;
                cursor: pointer;
                font-size: 13px;
            `;
            regenerateItem.onmouseover = () => regenerateItem.style.background = '#3a3a3a';
            regenerateItem.onmouseout = () => regenerateItem.style.background = 'transparent';
            regenerateItem.onclick = async () => {
                menu.remove();
                const catPrompts = node.prompts[cat];
                if (!catPrompts) return;

                const all = getPromptNamesForCategory(node, cat, { hideNSFW: false, workflowOnly: false, contentFilter: "all", endpointPrefix }).filter(name => {
                    const data = getCategoryPromptEntry(catPrompts, name, endpointPrefix);
                    return data && typeof data === "object";
                });

                await runBatchGeneration(all, "Re-Generate All Thumbnails", { regenerateAll: true });
            };
            menu.appendChild(regenerateItem);

            // Delete Category (always last, separated by a divider)
            const deleteDivider = document.createElement("div");
            deleteDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
            menu.appendChild(deleteDivider);

            const deleteItem = document.createElement("div");
            deleteItem.textContent = "🗑️ Delete Category";
            deleteItem.style.cssText = `
                padding: 8px 16px;
                color: #f66;
                cursor: pointer;
                font-size: 13px;
            `;
            deleteItem.onmouseover = () => deleteItem.style.background = '#3a3a3a';
            deleteItem.onmouseout = () => deleteItem.style.background = 'transparent';
            deleteItem.onclick = async () => {
                menu.remove();
                if (await showConfirm("Delete Category", `Are you sure you want to delete category "${categoryLabel}" and all its prompts?`)) {
                    try {
                        const resp = await fetch(`${endpointPrefix}/delete-category`, {
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
            };
            menu.appendChild(deleteItem);

            document.body.appendChild(menu);
            const closeMenu = (e) => {
                if (!menu.contains(e.target)) {
                    menu.remove();
                    document.removeEventListener("mousedown", closeMenu, true);
                    document.removeEventListener("contextmenu", closeMenu, true);
                }
            };
            setTimeout(() => {
                document.addEventListener("mousedown", closeMenu, true);
                document.addEventListener("contextmenu", closeMenu, true);
            }, 0);
        };

        const showTypeContextMenu = (event, typeValue) => {
            if (!showCategoryTypeFilter || !typeValue || typeValue === "__all__") return;
            const existing = document.querySelector('.category-context-menu');
            if (existing) existing.remove();

            const typeFile = getComposerTypeFile(node, typeValue);
            const typeLabel = getComposerTypeDisplayName(node, typeValue) || typeValue;
            const menu = document.createElement("div");
            menu.className = "category-context-menu";
            menu.style.cssText = `
                position: fixed;
                left: ${event.clientX}px;
                top: ${event.clientY}px;
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 4px 0;
                z-index: 10001;
                min-width: 170px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            `;

            const createMenuItem = (label, onClick, danger = false) => {
                const item = document.createElement("div");
                item.textContent = label;
                item.style.cssText = `
                    padding: 8px 16px;
                    color: ${danger ? '#f66' : '#ccc'};
                    cursor: pointer;
                    font-size: 13px;
                `;
                item.onmouseover = () => item.style.background = '#3a3a3a';
                item.onmouseout = () => item.style.background = 'transparent';
                item.onclick = async () => {
                    menu.remove();
                    await onClick();
                };
                return item;
            };

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
                    "#4CAF50"
                );
                if (!confirmed) return;

                const renderSelection = await ensureThumbnailRenderSelection();
                if (!renderSelection) return;
                const fallbackBase = await resolveThumbnailFallbackBase(renderSelection);

                for (const ref of refs) {
                    _thumbQueueTotal++;
                    _ensureThumbQueueProgress();
                    _updateThumbQueueProgress(ref.name);

                    _thumbQueuePromiseChain = _thumbQueuePromiseChain.then(async () => {
                        if (_thumbQueueCancelled) {
                            _thumbQueueDone++;
                            if (_thumbQueueDone + _thumbQueueFailed >= _thumbQueueTotal) {
                                _finishThumbQueueProgress();
                            }
                            return;
                        }
                        _updateThumbQueueProgress(ref.name);
                        try {
                            await _generateThumbnailForBrowserCategory(node, ref.category, ref.name, () => {
                                renderContent(searchInput.value);
                            }, {
                                renderSelection,
                                fallbackBase,
                                endpointPrefix,
                            });
                            _thumbQueueDone++;
                        } catch (err) {
                            console.error(`[ThumbnailGen] Failed for "${ref.category}/${ref.name}":`, err);
                            _thumbQueueFailed++;
                        } finally {
                            if (_thumbQueueProgress && _thumbQueueDone + _thumbQueueFailed >= _thumbQueueTotal) {
                                _finishThumbQueueProgress();
                            }
                        }
                    });
                }
            };

            const isNSFW = isComposerTypeNSFW(node, typeValue);
            menu.appendChild(createMenuItem("📋 Paste Icon", async () => {
                try {
                    const file = await readClipboardImageFile();
                    if (!file) {
                        await showInfo("No Image", "No image found in clipboard");
                        return;
                    }

                    const icon = await resizeImageToCoverDataUrl(file, 64, "image/png");
                    const resp = await fetch(`${endpointPrefix}/save-type-settings`, {
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

            const pasteDivider = document.createElement("div");
            pasteDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
            menu.appendChild(pasteDivider);

            menu.appendChild(createMenuItem(isNSFW ? "✓ NSFW" : "Mark as NSFW", async () => {
                try {
                    const resp = await fetch(`${endpointPrefix}/toggle-nsfw`, {
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

            const renameDivider = document.createElement("div");
            renameDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
            menu.appendChild(renameDivider);

            menu.appendChild(createMenuItem("✏️ Rename", async () => {
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
                    const resp = await fetch(`${endpointPrefix}/rename-type`, {
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

            const thumbDivider = document.createElement("div");
            thumbDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
            menu.appendChild(thumbDivider);

            menu.appendChild(createMenuItem("🎨 Generate Missing Thumbnails", async () => {
                await runTypeBatchGeneration("Generate Missing Thumbnails");
            }));

            menu.appendChild(createMenuItem("🔄 Re-Generate All Thumbnails", async () => {
                await runTypeBatchGeneration("Re-Generate All Thumbnails", { regenerateAll: true });
            }));

            const deleteDivider = document.createElement("div");
            deleteDivider.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
            menu.appendChild(deleteDivider);

            menu.appendChild(createMenuItem("🗑️ Delete Type", async () => {
                if (!await showConfirm("Delete Type", `Are you sure you want to delete type "${typeLabel}" and all its categories?`)) {
                    return;
                }
                try {
                    const resp = await fetch(`${endpointPrefix}/delete-type`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ type_file: typeFile })
                    });
                    const data = await resp.json();
                    if (data.success) {
                        applyPromptPayloadToNode(node, data);
                        if (categoryTypeFilter === typeValue) {
                            categoryTypeFilter = "__all__";
                        }
                        rebuildCategoryList();
                        rebuildTypeRailButtons();
                        renderContent(searchInput.value);
                    } else {
                        await showInfo("Error", data.error || "Failed to delete type.");
                    }
                } catch (err) {
                    console.error("[PromptBrowser] Error deleting type:", err);
                }
            }, true));

            document.body.appendChild(menu);
            const closeMenu = (e) => {
                if (!menu.contains(e.target)) {
                    menu.remove();
                    document.removeEventListener("mousedown", closeMenu, true);
                    document.removeEventListener("contextmenu", closeMenu, true);
                }
            };
            setTimeout(() => {
                document.addEventListener("mousedown", closeMenu, true);
                document.addEventListener("contextmenu", closeMenu, true);
            }, 0);
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
                const btn = document.createElement("button");
                btn.dataset.category = cat;
                btn.style.cssText = `
                    padding: 6px 14px;
                    border-radius: 6px;
                    border: 1px solid ${UI.inputBorder};
                    background: ${UI.buttonBg};
                    color: #aaa;
                    cursor: pointer;
                    font-size: 13px;
                    transition: all 0.15s ease;
                    position: relative;
                    flex-shrink: 0;
                    white-space: nowrap;
                    margin-top: 1px;
                `;

                btn.textContent = showCategoryTypeFilter ? getComposerCategoryDisplayName(node, cat) : cat;

                btn.onclick = async () => {
                    const canProceed = await canChangeEditorContext();
                    if (!canProceed) return;
                    setBlankPromptSelection();
                    if (editMode && editPanel && typeof editPanel.clearPrompt === "function") {
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

            // Add "+" button to create a new category
            const addBtn = document.createElement("button");
            addBtn.textContent = "+";
            addBtn.title = "New Category";
            addBtn.style.cssText = `
                padding: 6px 14px;
                border-radius: 6px;
                border: 1px solid #5f6773;
                background: #313843;
                color: #aaa;
                cursor: pointer;
                font-size: 13px;
                transition: all 0.15s ease;
                flex-shrink: 0;
                margin-bottom: 2px;
            `;
            const canCreateCategoryInCurrentFilter = !showCategoryTypeFilter || categoryTypeFilter !== "__all__";
            addBtn.style.display = canCreateCategoryInCurrentFilter ? "" : "none";
            addBtn.onmouseover = () => { addBtn.style.background = '#38414c'; addBtn.style.color = '#fff'; };
            addBtn.onmouseout = () => { addBtn.style.background = '#313843'; addBtn.style.color = '#aaa'; };
            addBtn.onclick = async () => {
                if (!canCreateCategoryInCurrentFilter) return;
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
                        const resp = await fetch(`${endpointPrefix}/save-category`, {
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
                            if (editMode && editPanel && typeof editPanel.clearPrompt === "function") {
                                await editPanel.clearPrompt({ skipConfirm: true });
                            }
                            const nextCategoryKey = resolveComposerCategoryKey(node, categoryName, data.type_file || "");
                            syncComposerTypeFilterForCategory(nextCategoryKey, data.type_file || "");
                            setSelectedCategory(nextCategoryKey);
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
        // Hide grid scrollbar; style category scrollbar for webkit browsers
        const style = document.createElement("style");
        style.textContent = `
            .thumbnail-grid-container::-webkit-scrollbar { display: none; }
            .category-scroll-bar::-webkit-scrollbar { height: 6px; }
            .category-scroll-bar::-webkit-scrollbar:horizontal { display: block; }
            .category-scroll-bar::-webkit-scrollbar-track { background: transparent; }
            .category-scroll-bar::-webkit-scrollbar-thumb { background: rgba(74, 138, 212, 0.55); border-radius: 3px; }
            .category-scroll-bar::-webkit-scrollbar-thumb:hover { background: rgba(74, 138, 212, 0.85); }
            .category-scroll-bar::-webkit-scrollbar-corner { background: transparent; }
            .pm-type-rail-list::-webkit-scrollbar { width: 6px; }
            .pm-type-rail-list::-webkit-scrollbar-track { background: transparent; }
            .pm-type-rail-list::-webkit-scrollbar-thumb { background: rgba(74, 138, 212, 0.45); border-radius: 3px; }
            .pm-type-rail-list::-webkit-scrollbar-thumb:hover { background: rgba(74, 138, 212, 0.75); }
        `;
        document.head.appendChild(style);

        if (typeRail) {
            updateTypeRailButtons();
        }

        contentRow.appendChild(gridContainer);

        // Edit panel
        if (allowEditMode && mode !== "save") {
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
                        const resp = await fetch(`${endpointPrefix}/save-prompt`, {
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
                        renderCategoryTabs();
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
                    return new Promise((resolve, reject) => {
                        const queuedName = promptName;
                        const isDraftGeneration = draftPromptData && typeof draftPromptData === "object";
                        const requestedSeed = Number(draftPromptData?.__pm_thumbnail_seed);
                        const staticSeed = Number.isFinite(requestedSeed) ? Math.trunc(requestedSeed) : 42;
                        _thumbQueueTotal++;
                        _ensureThumbQueueProgress();
                        _updateThumbQueueProgress(queuedName);

                        _thumbQueuePromiseChain = _thumbQueuePromiseChain.then(async () => {
                            if (_thumbQueueCancelled) {
                                _thumbQueueDone++;
                                if (_thumbQueueDone + _thumbQueueFailed >= _thumbQueueTotal) {
                                    _finishThumbQueueProgress();
                                }
                                resolve(null);
                                return;
                            }
                            _updateThumbQueueProgress(queuedName);
                            try {
                                const generatedThumbnail = await _generateThumbnailForBrowserCategory(node, category, queuedName, () => {
                                    renderContent(searchInput.value);
                                }, {
                                    endpointPrefix,
                                    draftPromptData,
                                    persistThumbnail: !isDraftGeneration,
                                    promptStrength,
                                    generationMode: thumbnailGenerationMode,
                                    staticSeed,
                                });
                                _thumbQueueDone++;
                                if (isDraftGeneration) {
                                    resolve(generatedThumbnail || null);
                                    return;
                                }
                                await loadPromptsFn(node);
                                resolve(getCategoryPromptEntry(node?.prompts?.[category], queuedName, endpointPrefix)?.thumbnail || null);
                            } catch (e) {
                                const error = e instanceof Error
                                    ? e
                                    : new Error(e == null ? "Thumbnail generation failed." : String(e));
                                console.error(`[ThumbnailGen] Failed for "${queuedName}":`, error);
                                _thumbQueueFailed++;
                                reject(error);
                                return;
                            } finally {
                                if (_thumbQueueProgress && _thumbQueueDone + _thumbQueueFailed >= _thumbQueueTotal) {
                                    _finishThumbQueueProgress();
                                }
                            }
                        });
                    });
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
                    if (typeof editPanel.clearPrompt === "function") {
                        await editPanel.clearPrompt({ skipConfirm: true });
                    }
                    rebuildCategoryList();
                    if (editPanel) {
                        editPanel.loadCategorySettings(selectedCategory);
                        if (typeof editPanel.showCategorySettings === "function") {
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
                    );
                    if (picked) {
                        saveThumbnailRenderSelection(picked);
                        updateThumbnailModelBtn();
                        return true;
                    }
                    return false;
                },
                getThumbnailModelLabel: () => getThumbnailRenderLabelParts(),
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
            const existing = document.querySelector('.thumbnail-context-menu');
            if (existing) existing.remove();

            const menu = document.createElement("div");
            menu.className = "thumbnail-context-menu";
            menu.style.cssText = `
                position: fixed;
                left: ${e.clientX}px;
                top: ${e.clientY}px;
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 4px 0;
                z-index: 10001;
                min-width: 170px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            `;

            const createMenuItem = (label, onClick, danger = false) => {
                const item = document.createElement("div");
                item.textContent = label;
                item.style.cssText = `
                    padding: 8px 16px;
                    color: ${danger ? '#f66' : '#ccc'};
                    cursor: pointer;
                    font-size: 13px;
                `;
                item.onmouseover = () => item.style.background = '#3a3a3a';
                item.onmouseout = () => item.style.background = 'transparent';
                item.onclick = () => {
                    menu.remove();
                    onClick();
                };
                return item;
            };

            const selectedList = Array.from(promptNames || []);

            menu.appendChild(createMenuItem(`🎨 Generate Thumbnails (${selectedList.length})`, async () => {
                const names = selectedList.slice();
                _thumbQueueTotal += names.length;
                _ensureThumbQueueProgress();
                _updateThumbQueueProgress(names[0] || "");

                _thumbQueuePromiseChain = _thumbQueuePromiseChain.then(async () => {
                    for (const name of names) {
                        if (_thumbQueueCancelled) {
                            _thumbQueueDone++;
                            continue;
                        }
                        _updateThumbQueueProgress(name);
                        try {
                            await _generateThumbnailForBrowserCategory(node, selectedCategory, name, () => {
                                renderContent(searchInput.value);
                            }, { endpointPrefix });
                            _thumbQueueDone++;
                        } catch (err) {
                            console.error(`[ThumbnailGen] Failed for "${name}":`, err);
                            _thumbQueueFailed++;
                        }
                    }
                    if (_thumbQueueProgress && _thumbQueueDone + _thumbQueueFailed >= _thumbQueueTotal) {
                        _finishThumbQueueProgress();
                    }
                });
            }));

            const dividerA = document.createElement("div");
            dividerA.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
            menu.appendChild(dividerA);

            menu.appendChild(createMenuItem(`📁 Move (${selectedList.length})`, async () => {
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
                        const resp = await fetch(`${endpointPrefix}/rename-prompt`, {
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

            const dividerB = document.createElement("div");
            dividerB.style.cssText = `height: 1px; background: #444; margin: 4px 0;`;
            menu.appendChild(dividerB);

            menu.appendChild(createMenuItem(`🗑️ Delete Prompts (${selectedList.length})`, async () => {
                const confirmed = await showConfirm(
                    "Delete Prompts",
                    `Are you sure you want to delete ${selectedList.length} selected prompt(s)?`,
                    "Delete",
                    "#c44"
                );
                if (!confirmed) return;

                for (const promptName of selectedList) {
                    await deletePromptEntry(node, selectedCategory, promptName, endpointPrefix);
                }
                await clearPromptBrowserSelection();
                updateEditModeLayout();
                renderContent(searchInput.value);
            }, true));

            const closeMenu = (evt) => {
                if (!menu.contains(evt.target)) {
                    menu.remove();
                    document.removeEventListener('mousedown', closeMenu, true);
                }
            };
            setTimeout(() => {
                document.addEventListener('mousedown', closeMenu, true);
            }, 10);

            document.body.appendChild(menu);
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
        const renderGridView = (filter = "") => {
            grid.style.cssText = `
                display: grid;
                grid-template-columns: repeat(${browserLayout.cols}, ${browserLayout.itemWidth}px);
                gap: ${browserLayout.gap}px;
                padding: 4px 0;
            `;
            grid.innerHTML = "";

            const categoryPrompts = node.prompts[selectedCategory] || {};
            const filteredPrompts = getFilteredPrompts(filter);
            const showEditBlank = editMode && !!editPanel && mode !== "save";

            if (filteredPrompts.length === 0 && !showEditBlank) {
                const emptyMsg = document.createElement("div");
                emptyMsg.textContent = filter ? "No matching prompts found" : "No prompts in this category";
                emptyMsg.style.cssText = `
                    grid-column: 1 / -1;
                    text-align: center;
                    color: #666;
                    padding: 40px;
                    font-style: italic;
                `;
                grid.appendChild(emptyMsg);
                return;
            }

            const updateCardSelection = (card, promptName) => {
                const isSel = selectedNames.has(promptName);
                card.dataset.selectedPrompt = isSel ? "true" : "false";
                card.style.background = isSel ? UI.accentSoft : UI.cardBg;
                card.style.borderColor = isSel ? UI.accentBorder : UI.cardBorder;
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

                const card = document.createElement("div");
                card.dataset.promptName = promptName;
                if (isSelected) {
                    card.dataset.selectedPrompt = "true";
                }
                card.style.cssText = `
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 8px;
                    background: ${isSelected ? UI.accentSoft : UI.cardBg};
                    border: 2px solid ${isSelected ? UI.accentBorder : UI.cardBorder};
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.15s ease;
                    position: relative;
                `;

                card.onmouseenter = () => {
                    if (!selectedNames.has(promptName)) {
                        card.style.background = UI.accentSoft;
                        card.style.borderColor = UI.accentBorder;
                    }
                };
                card.onmouseleave = () => {
                    updateCardSelection(card, promptName);
                };

                // Top-right badge stack: NSFW first, workflow badge under it.
                const showComposeBadge = hasComposeData;
                const showWorkflowBadge = !showComposeBadge && !workflowOnly && hasWorkflowData;
                const showPromptBadge = !showComposeBadge && workflowOnly && !hasWorkflowData && hasPromptPayload;
                if (isNSFW || showComposeBadge || showWorkflowBadge || showPromptBadge) {
                    const badgeStack = document.createElement("div");
                    badgeStack.style.cssText = `
                        position: absolute;
                        top: 4px;
                        right: 4px;
                        display: flex;
                        flex-direction: column;
                        align-items: flex-end;
                        gap: 2px;
                        z-index: 1;
                    `;

                    if (isNSFW) {
                        const badge = document.createElement("div");
                        badge.textContent = "NSFW";
                        badge.style.cssText = `
                            background: rgba(204, 0, 0, 0.85);
                            color: #fff;
                            font-size: 8px;
                            font-weight: bold;
                            padding: 1px 4px;
                            border-radius: 3px;
                            line-height: 1.2;
                        `;
                        badgeStack.appendChild(badge);
                    }

                    if (showComposeBadge) {
                        const composeBadge = document.createElement("div");
                        composeBadge.textContent = "C";
                        composeBadge.title = "Has Compose Data";
                        composeBadge.style.cssText = `
                            width: 14px;
                            height: 14px;
                            border-radius: 50%;
                            background: rgba(47, 146, 72, 0.95);
                            color: #fff;
                            font-size: 9px;
                            font-weight: bold;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            line-height: 1;
                        `;
                        badgeStack.appendChild(composeBadge);
                    }

                    if (showWorkflowBadge) {
                        const workflowBadge = document.createElement("div");
                        workflowBadge.textContent = "R";
                        workflowBadge.title = "Has Recipe Data";
                        workflowBadge.style.cssText = `
                            width: 14px;
                            height: 14px;
                            border-radius: 50%;
                            background: rgba(235, 140, 35, 0.95);
                            color: #fff;
                            font-size: 9px;
                            font-weight: bold;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            line-height: 1;
                        `;
                        badgeStack.appendChild(workflowBadge);
                    }

                    if (showPromptBadge) {
                        const promptBadge = document.createElement("div");
                        promptBadge.textContent = "P";
                        promptBadge.title = "Prompt Data (Converted to Recipe)";
                        promptBadge.style.cssText = `
                            width: 14px;
                            height: 14px;
                            border-radius: 50%;
                            background: rgba(56, 130, 246, 0.96);
                            color: #fff;
                            font-size: 9px;
                            font-weight: bold;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            line-height: 1;
                        `;
                        badgeStack.appendChild(promptBadge);
                    }

                    card.appendChild(badgeStack);
                }

                // Thumbnail (using div with background-image to avoid browser extension interference)
                const thumbWidth = browserLayout.thumbWidth;
                const thumbHeight = browserLayout.thumbHeight;
                const thumbDiv = document.createElement("div");
                thumbDiv.style.cssText = `
                    width: ${thumbWidth}px;
                    height: ${thumbHeight}px;
                    background-image: url(${thumbnail});
                    background-size: cover;
                    background-position: center;
                    border-radius: 6px;
                    background-color: #1a1a1a;
                    flex-shrink: 0;
                    cursor: pointer;
                `;
                _attachThumbnailDropHandlers(thumbDiv, node, selectedCategory, promptName, () => renderContent(searchInput.value), endpointPrefix);

                // Add hover preview with proper event handling (if enabled and not placeholder)
                // Hover preview is only useful in compact mode; disabled when thumbnails are already large.
                if (compactBrowser && previewEnabled && thumbnail !== DEFAULT_THUMBNAIL) {
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
                const nameLabel = document.createElement("div");
                nameLabel.textContent = promptName;
                nameLabel.title = promptName;
                nameLabel.style.cssText = `
                    margin-top: 8px;
                    font-size: 12px;
                    color: #ccc;
                    text-align: center;
                    width: 100%;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                `;

                card.appendChild(thumbDiv);
                card.appendChild(nameLabel);

                card.onclick = async (e) => {
                    if (supportsMultiSelect && (e.shiftKey || e.ctrlKey || e.metaKey)) {
                        enableMultiSelect({ preserveCurrentSelection: true });
                        applyMultiSelectInteraction(promptName, filteredPrompts, e);
                        if (editMode && editPanel) {
                            editPanel.loadPrompt(selectedCategory, promptName);
                        }
                        renderContent(searchInput.value);
                        return;
                    }

                    if (isMultiSelectActive()) {
                        const action = applyMultiSelectInteraction(promptName, filteredPrompts, e);
                        if (action === "rerender") {
                            if (editMode && editPanel) {
                                editPanel.loadPrompt(selectedCategory, promptName);
                            }
                            renderContent(searchInput.value);
                            return;
                        }
                        if (editMode && editPanel) {
                            editPanel.loadPrompt(selectedCategory, promptName);
                        }
                        updateCardSelection(card, promptName);
                        updateEditModeLayout();
                        return;
                    }

                    // In edit mode, ask before discarding unsaved changes.
                    if (editMode && editPanel) {
                        const now = Date.now();
                        if (editModeLastClickPrompt === promptName && (now - editModeLastClickAt) <= 500) {
                            resolve({ category: getResultCategoryName(), prompt: promptName, prompts: [promptName] });
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
                        if (saveNameInput) {
                            saveNameInput.value = promptName;
                            saveNameInput.focus();
                            saveNameInput.select();
                        }
                        setCurrentPromptSelection(promptName);
                        renderContent(searchInput.value);
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

                    resolve({ category: getResultCategoryName(), prompt: promptName, prompts: [promptName] });
                    cleanup();
                };

                if (mode === "save") {
                    card.ondblclick = async () => {
                        if (!onSave) return;
                        const overwriteOk = await showConfirm(
                            "Overwrite Prompt",
                            `Prompt "${promptName}" already exists in category "${selectedCategory}". Do you want to replace it?`,
                            "Replace",
                            "#c44"
                        );
                        if (!overwriteOk) return;

                        const saveResult = await onSave({
                            category: selectedCategory,
                            name: promptName,
                            overwrite: true,
                        });

                        if (saveResult?.success) {
                            resolve(saveResult);
                            cleanup();
                        } else {
                            await showInfo("Save Failed", saveResult?.error || "Failed to save workflow.");
                        }
                    };
                } else if (requireDoubleClickToSelect || isMultiSelectActive()) {
                    card.ondblclick = () => {
                        resolve({ category: getResultCategoryName(), prompt: promptName, prompts: [promptName] });
                        cleanup();
                    };
                }

                card.oncontextmenu = (e) => promptContextMenu(e, promptName);

                grid.appendChild(card);
            });

            if (showEditBlank) {
                const blankCard = document.createElement("div");
                blankCard.style.cssText = `
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 8px;
                    background: ${UI.cardBg};
                    border: 2px dashed ${UI.cardBorder};
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.15s ease;
                    min-height: ${browserLayout.thumbHeight + 34}px;
                `;

                const blankThumb = document.createElement("div");
                blankThumb.style.cssText = `
                    width: ${browserLayout.thumbWidth}px;
                    height: ${browserLayout.thumbHeight}px;
                    border-radius: 6px;
                    background: ${UI.inputBg};
                    border: 1px dashed ${UI.inputBorder};
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #98a3af;
                    font-size: 36px;
                    line-height: 1;
                `;
                blankThumb.textContent = "+";

                const blankLabel = document.createElement("div");
                blankLabel.textContent = "Blank Prompt";
                blankLabel.style.cssText = `
                    margin-top: 8px;
                    font-size: 12px;
                    color: #ccc;
                    text-align: center;
                    width: 100%;
                    white-space: nowrap;
                `;

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

        // ---- Icon (Small Thumbnail) Grid View ----
        const renderCompactGridView = (filter = "") => {
            grid.style.cssText = `
                display: grid;
                grid-template-columns: repeat(${browserLayout.iconCols}, ${browserLayout.iconItemWidth}px);
                gap: ${browserLayout.iconGap}px;
                padding: 4px 0;
            `;
            grid.innerHTML = "";

            const categoryPrompts = node.prompts[selectedCategory] || {};
            const filteredPrompts = getFilteredPrompts(filter);
            const showEditBlank = editMode && !!editPanel && mode !== "save";

            if (filteredPrompts.length === 0 && !showEditBlank) {
                const emptyMsg = document.createElement("div");
                emptyMsg.textContent = filter ? "No matching prompts found" : "No prompts in this category";
                emptyMsg.style.cssText = `
                    grid-column: 1 / -1;
                    text-align: center;
                    color: #666;
                    padding: 40px;
                    font-style: italic;
                `;
                grid.appendChild(emptyMsg);
                return;
            }

            const updateCardSelection = (card, promptName) => {
                const isSel = selectedNames.has(promptName);
                card.dataset.selectedPrompt = isSel ? "true" : "false";
                card.style.background = isSel ? UI.accentSoft : UI.cardBg;
                card.style.borderColor = isSel ? UI.accentBorder : UI.cardBorder;
                updateSelectButton();
            };

            const updateCompactGridSelections = () => {
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

                const card = document.createElement("div");
                card.dataset.promptName = promptName;
                if (isSelected) {
                    card.dataset.selectedPrompt = "true";
                }
                card.style.cssText = `
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 8px;
                    background: ${isSelected ? UI.accentSoft : UI.cardBg};
                    border: 2px solid ${isSelected ? UI.accentBorder : UI.cardBorder};
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.15s ease;
                    position: relative;
                `;

                card.onmouseenter = () => {
                    if (!selectedNames.has(promptName)) {
                        card.style.background = UI.accentSoft;
                        card.style.borderColor = UI.accentBorder;
                    }
                };
                card.onmouseleave = () => {
                    updateCardSelection(card, promptName);
                };

                // Top-right badge stack: NSFW first, workflow badge under it.
                const showComposeBadge = hasComposeData;
                const showWorkflowBadge = !showComposeBadge && !workflowOnly && hasWorkflowData;
                const showPromptBadge = !showComposeBadge && workflowOnly && !hasWorkflowData && hasPromptPayload;
                if (isNSFW || showComposeBadge || showWorkflowBadge || showPromptBadge) {
                    const badgeStack = document.createElement("div");
                    badgeStack.style.cssText = `
                        position: absolute;
                        top: 4px;
                        right: 4px;
                        display: flex;
                        flex-direction: column;
                        align-items: flex-end;
                        gap: 2px;
                        z-index: 1;
                    `;

                    if (isNSFW) {
                        const badge = document.createElement("div");
                        badge.textContent = "NSFW";
                        badge.style.cssText = `
                            background: rgba(204, 0, 0, 0.85);
                            color: #fff;
                            font-size: 8px;
                            font-weight: bold;
                            padding: 1px 4px;
                            border-radius: 3px;
                            line-height: 1.2;
                        `;
                        badgeStack.appendChild(badge);
                    }

                    if (showComposeBadge) {
                        const composeBadge = document.createElement("div");
                        composeBadge.textContent = "C";
                        composeBadge.title = "Has Compose Data";
                        composeBadge.style.cssText = `
                            width: 14px;
                            height: 14px;
                            border-radius: 50%;
                            background: rgba(47, 146, 72, 0.95);
                            color: #fff;
                            font-size: 9px;
                            font-weight: bold;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            line-height: 1;
                        `;
                        badgeStack.appendChild(composeBadge);
                    }

                    if (showWorkflowBadge) {
                        const workflowBadge = document.createElement("div");
                        workflowBadge.textContent = "R";
                        workflowBadge.title = "Has Recipe Data";
                        workflowBadge.style.cssText = `
                            width: 14px;
                            height: 14px;
                            border-radius: 50%;
                            background: rgba(235, 140, 35, 0.95);
                            color: #fff;
                            font-size: 9px;
                            font-weight: bold;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            line-height: 1;
                        `;
                        badgeStack.appendChild(workflowBadge);
                    }

                    if (showPromptBadge) {
                        const promptBadge = document.createElement("div");
                        promptBadge.textContent = "P";
                        promptBadge.title = "Prompt Data (Converted to Recipe)";
                        promptBadge.style.cssText = `
                            width: 14px;
                            height: 14px;
                            border-radius: 50%;
                            background: rgba(56, 130, 246, 0.96);
                            color: #fff;
                            font-size: 9px;
                            font-weight: bold;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            line-height: 1;
                        `;
                        badgeStack.appendChild(promptBadge);
                    }

                    card.appendChild(badgeStack);
                }

                // Thumbnail (using div with background-image to avoid browser extension interference)
                const thumbDiv = document.createElement("div");
                thumbDiv.style.cssText = `
                    width: ${browserLayout.iconThumbWidth}px;
                    height: ${browserLayout.iconThumbHeight}px;
                    background-image: url(${thumbnail});
                    background-size: cover;
                    background-position: center;
                    border-radius: 6px;
                    background-color: #1a1a1a;
                    flex-shrink: 0;
                    cursor: pointer;
                `;
                _attachThumbnailDropHandlers(thumbDiv, node, selectedCategory, promptName, () => renderContent(searchInput.value), endpointPrefix);

                // Hover preview enabled for compact grid thumbnails.
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

                // Prompt name
                const nameLabel = document.createElement("div");
                nameLabel.textContent = promptName;
                nameLabel.title = promptName;
                nameLabel.style.cssText = `
                    margin-top: 8px;
                    font-size: 12px;
                    color: #ccc;
                    text-align: center;
                    width: 100%;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                `;

                card.appendChild(thumbDiv);
                card.appendChild(nameLabel);

                card.onclick = async (e) => {
                    if (supportsMultiSelect && (e.shiftKey || e.ctrlKey || e.metaKey)) {
                        enableMultiSelect({ preserveCurrentSelection: true });
                        applyMultiSelectInteraction(promptName, filteredPrompts, e);
                        if (editMode && editPanel) {
                            editPanel.loadPrompt(selectedCategory, promptName);
                        }
                        renderContent(searchInput.value);
                        return;
                    }

                    if (isMultiSelectActive()) {
                        const action = applyMultiSelectInteraction(promptName, filteredPrompts, e);
                        if (action === "rerender") {
                            if (editMode && editPanel) {
                                editPanel.loadPrompt(selectedCategory, promptName);
                            }
                            renderContent(searchInput.value);
                            return;
                        }
                        if (editMode && editPanel) {
                            editPanel.loadPrompt(selectedCategory, promptName);
                        }
                        updateCardSelection(card, promptName);
                        updateEditModeLayout();
                        return;
                    }

                    if (editMode && editPanel) {
                        const now = Date.now();
                        if (editModeLastClickPrompt === promptName && (now - editModeLastClickAt) <= 500) {
                            resolve({ category: getResultCategoryName(), prompt: promptName, prompts: [promptName] });
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
                        if (saveNameInput) {
                            saveNameInput.value = promptName;
                            saveNameInput.focus();
                            saveNameInput.select();
                        }
                        setCurrentPromptSelection(promptName);
                        renderContent(searchInput.value);
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
                        updateCompactGridSelections();
                        return;
                    }

                    resolve({ category: getResultCategoryName(), prompt: promptName, prompts: [promptName] });
                    cleanup();
                };

                if (mode === "save") {
                    card.ondblclick = async () => {
                        if (!onSave) return;
                        const overwriteOk = await showConfirm(
                            "Overwrite Prompt",
                            `Prompt "${promptName}" already exists in category "${selectedCategory}". Do you want to replace it?`,
                            "Replace",
                            "#c44"
                        );
                        if (!overwriteOk) return;

                        const saveResult = await onSave({
                            category: selectedCategory,
                            name: promptName,
                            overwrite: true,
                        });

                        if (saveResult?.success) {
                            resolve(saveResult);
                            cleanup();
                        } else {
                            await showInfo("Save Failed", saveResult?.error || "Failed to save workflow.");
                        }
                    };
                } else if (requireDoubleClickToSelect || isMultiSelectActive()) {
                    card.ondblclick = () => {
                        resolve({ category: getResultCategoryName(), prompt: promptName, prompts: [promptName] });
                        cleanup();
                    };
                }

                card.oncontextmenu = (e) => promptContextMenu(e, promptName);

                grid.appendChild(card);
            });

            if (showEditBlank) {
                const blankCard = document.createElement("div");
                blankCard.style.cssText = `
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 8px;
                    background: ${UI.cardBg};
                    border: 2px dashed ${UI.cardBorder};
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.15s ease;
                `;

                const blankThumb = document.createElement("div");
                blankThumb.style.cssText = `
                    width: ${browserLayout.iconThumbWidth}px;
                    height: ${browserLayout.iconThumbHeight}px;
                    border-radius: 6px;
                    background: ${UI.inputBg};
                    border: 1px dashed ${UI.inputBorder};
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #98a3af;
                    font-size: 24px;
                    line-height: 1;
                `;
                blankThumb.textContent = "+";

                const blankLabel = document.createElement("div");
                blankLabel.textContent = "Blank Prompt";
                blankLabel.style.cssText = `
                    margin-top: 8px;
                    font-size: 12px;
                    color: #ccc;
                    text-align: center;
                    width: 100%;
                    white-space: nowrap;
                `;

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

        // ---- List View ----
        const renderListView = (filter = "") => {
            grid.style.cssText = `
                display: flex;
                flex-direction: column;
                gap: 0;
                padding: 0;
                position: relative;
            `;
            grid.innerHTML = "";

            const categoryPrompts = node.prompts[selectedCategory] || {};
            const filteredPrompts = getFilteredPrompts(filter);
            const showEditBlank = editMode && !!editPanel && mode !== "save";

            if (filteredPrompts.length === 0 && !showEditBlank) {
                const emptyMsg = document.createElement("div");
                emptyMsg.textContent = filter ? "No matching prompts found" : "No prompts in this category";
                emptyMsg.style.cssText = `
                    text-align: center;
                    color: #666;
                    padding: 40px;
                    font-style: italic;
                `;
                grid.appendChild(emptyMsg);
                return;
            }

            const updateRowSelection = (row, promptName) => {
                const isSel = selectedNames.has(promptName);
                row.dataset.selectedPrompt = isSel ? "true" : "false";
                row.style.background = isSel ? UI.accentSoft : 'transparent';
                row.style.outline = isSel ? `2px solid ${UI.accentBorder}` : "none";
                row.style.outlineOffset = isSel ? "-2px" : "0";
                row.style.boxShadow = isSel ? `0 0 8px ${UI.accentSoft}` : "none";
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
                border-bottom: 1px solid rgba(148,163,184,0.2);
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
                    color: #888;
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
                row.style.cssText = `
                    display: grid;
                    grid-template-columns: ${listColumns};
                    gap: 0;
                    padding: 0 8px;
                    background: ${isSelected ? UI.accentSoft : 'transparent'};
                    border-radius: 4px;
                    cursor: pointer;
                    align-items: center;
                    transition: background 0.1s ease;
                    outline: ${isSelected ? `2px solid ${UI.accentBorder}` : "none"};
                    outline-offset: ${isSelected ? "-2px" : "0"};
                    box-shadow: ${isSelected ? `0 0 8px ${UI.accentSoft}` : "none"};
                `;
                row.onmouseenter = () => {
                    if (!selectedNames.has(promptName)) {
                        row.style.background = '#2a2a2a';
                        row.style.outline = `1px solid ${UI.accentBorder}`;
                        row.style.outlineOffset = "-1px";
                    }
                };
                row.onmouseleave = () => { updateRowSelection(row, promptName); };

                // Thumbnail icon (using div with background-image to avoid browser extension interference)
                const thumbWrap = document.createElement("div");
                thumbWrap.style.cssText = `
                    width: 36px;
                    height: 36px;
                    position: relative;
                    flex-shrink: 0;
                    margin: 0 auto;
                `;

                const thumbDiv = document.createElement("div");
                thumbDiv.style.cssText = `
                    width: 36px;
                    height: 36px;
                    background-image: url(${thumbnail});
                    background-size: cover;
                    background-position: center;
                    border-radius: 4px;
                    background-color: #1a1a1a;
                    cursor: pointer;
                `;
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
                const showComposeBadge = hasComposeData;
                const showWorkflowBadge = !showComposeBadge && !workflowOnly && hasWorkflowData;
                const showPromptBadge = !showComposeBadge && workflowOnly && !hasWorkflowData && hasPromptPayload;
                if (showComposeBadge) {
                    const composeBadge = document.createElement("div");
                    composeBadge.textContent = "C";
                    composeBadge.title = "Has Compose Data";
                    composeBadge.style.cssText = `
                        position: absolute;
                        right: -2px;
                        bottom: -2px;
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        background: rgba(47, 146, 72, 0.95);
                        color: #fff;
                        font-size: 8px;
                        font-weight: bold;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        line-height: 1;
                        border: 1px solid rgba(0, 0, 0, 0.4);
                        z-index: 1;
                    `;
                    thumbWrap.appendChild(composeBadge);
                }
                if (showWorkflowBadge) {
                    const workflowBadge = document.createElement("div");
                    workflowBadge.textContent = "R";
                    workflowBadge.title = "Has Recipe Data";
                    workflowBadge.style.cssText = `
                        position: absolute;
                        right: -2px;
                        bottom: -2px;
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        background: rgba(235, 140, 35, 0.95);
                        color: #fff;
                        font-size: 8px;
                        font-weight: bold;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        line-height: 1;
                        border: 1px solid rgba(0, 0, 0, 0.4);
                        z-index: 1;
                    `;
                    thumbWrap.appendChild(workflowBadge);
                }

                if (showPromptBadge) {
                    const promptBadge = document.createElement("div");
                    promptBadge.textContent = "P";
                    promptBadge.title = "Prompt Data (Converted to Recipe)";
                    promptBadge.style.cssText = `
                        position: absolute;
                        right: -2px;
                        bottom: -2px;
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        background: rgba(56, 130, 246, 0.96);
                        color: #fff;
                        font-size: 8px;
                        font-weight: bold;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        line-height: 1;
                        border: 1px solid rgba(0, 0, 0, 0.4);
                        z-index: 1;
                    `;
                    thumbWrap.appendChild(promptBadge);
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
                nameSpan.style.cssText = `
                    font-size: 13px;
                    color: #ddd;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                `;
                nameDiv.appendChild(nameSpan);
                if (isNSFW) {
                    const badge = document.createElement("span");
                    badge.textContent = "NSFW";
                    badge.style.cssText = `
                        background: rgba(204, 0, 0, 0.85);
                        color: #fff;
                        font-size: 8px;
                        font-weight: bold;
                        padding: 1px 4px;
                        border-radius: 3px;
                        flex-shrink: 0;
                    `;
                    nameDiv.appendChild(badge);
                }

                const promptTextRaw = String(promptData?.prompt || "").trim();
                const promptTextDisplay = promptTextRaw || "—";
                const promptDiv = document.createElement("div");
                promptDiv.style.cssText = `
                    font-size: 12px;
                    color: #9fb0c3;
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
                        color: ${n > 0 ? '#aaa' : '#555'};
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
                        if (editMode && editPanel) {
                            editPanel.loadPrompt(selectedCategory, promptName);
                        }
                        renderContent(searchInput.value);
                        return;
                    }

                    if (isMultiSelectActive()) {
                        const action = applyMultiSelectInteraction(promptName, filteredPrompts, e);
                        if (action === "rerender") {
                            if (editMode && editPanel) {
                                editPanel.loadPrompt(selectedCategory, promptName);
                            }
                            renderContent(searchInput.value);
                            return;
                        }
                        if (editMode && editPanel) {
                            editPanel.loadPrompt(selectedCategory, promptName);
                        }
                        updateRowSelection(row, promptName);
                        updateEditModeLayout();
                        return;
                    }

                    if (editMode && editPanel) {
                        const now = Date.now();
                        if (editModeLastClickPrompt === promptName && (now - editModeLastClickAt) <= 1000) {
                            resolve({ category: getResultCategoryName(), prompt: promptName, prompts: [promptName] });
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
                        if (saveNameInput) {
                            saveNameInput.value = promptName;
                            saveNameInput.focus();
                            saveNameInput.select();
                        }
                        setCurrentPromptSelection(promptName);
                        renderContent(searchInput.value);
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

                    resolve({ category: getResultCategoryName(), prompt: promptName, prompts: [promptName] });
                    cleanup();
                };

                if (mode === "save") {
                    row.ondblclick = async () => {
                        if (!onSave) return;
                        const overwriteOk = await showConfirm(
                            "Overwrite Prompt",
                            `Prompt "${promptName}" already exists in category "${selectedCategory}". Do you want to replace it?`,
                            "Replace",
                            "#c44"
                        );
                        if (!overwriteOk) return;

                        const saveResult = await onSave({
                            category: selectedCategory,
                            name: promptName,
                            overwrite: true,
                        });

                        if (saveResult?.success) {
                            resolve(saveResult);
                            cleanup();
                        } else {
                            await showInfo("Save Failed", saveResult?.error || "Failed to save workflow.");
                        }
                    };
                } else if (requireDoubleClickToSelect || isMultiSelectActive()) {
                    row.ondblclick = () => {
                        resolve({ category: getResultCategoryName(), prompt: promptName, prompts: [promptName] });
                        cleanup();
                    };
                }

                row.oncontextmenu = (e) => promptContextMenu(e, promptName);

                grid.appendChild(row);
            });

            if (showEditBlank) {
                const row = document.createElement("div");
                row.dataset.pmListRow = "true";
                row.style.cssText = `
                    display: grid;
                    grid-template-columns: ${listColumns};
                    gap: 0;
                    padding: 0 8px;
                    background: transparent;
                    border-radius: 4px;
                    cursor: pointer;
                    align-items: center;
                    transition: background 0.1s ease;
                    outline: none;
                    outline-offset: 0;
                    box-shadow: none;
                `;

                const blankThumbWrap = document.createElement("div");
                blankThumbWrap.style.cssText = `
                    width: 36px;
                    height: 36px;
                    margin: 0 auto;
                    border-radius: 4px;
                    background: ${UI.inputBg};
                    border: 1px dashed ${UI.inputBorder};
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #98a3af;
                    font-size: 20px;
                    line-height: 1;
                `;
                blankThumbWrap.textContent = "+";

                const blankName = document.createElement("div");
                blankName.textContent = "Blank Prompt";
                blankName.style.cssText = `
                    font-size: 13px;
                    color: #ddd;
                    padding: 4px 16px;
                    box-sizing: border-box;
                `;

                const blankPrompt = document.createElement("div");
                blankPrompt.textContent = "New empty prompt";
                blankPrompt.style.cssText = `
                    font-size: 12px;
                    color: #9fb0c3;
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
                        d.style.cssText = `font-size: 12px; color: #555; text-align: center; padding: 4px 16px; box-sizing: border-box;`;
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
                background: ${UI.panel};
                border: 1px solid ${UI.accentBorder};
                border-radius: 8px;
                color: ${UI.textPrimary || "#ddd"};
                font-size: 15px;
                line-height: 1.55;
                padding: 12px 14px;
                box-shadow: 0 10px 28px rgba(0,0,0,0.55);
                z-index: 10003;
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
                    z-index: 10001;
                    display: none;
                    border: 2px solid ${UI.inputBorder};
                    border-radius: 8px;
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.8);
                    background-color: #1a1a1a;
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
            if (contentFilterState === "all") {
                contentFilterState = "prompt";
            } else if (contentFilterState === "prompt") {
                contentFilterState = "recipe";
            } else if (contentFilterState === "recipe") {
                contentFilterState = "compose";
            } else {
                contentFilterState = "all";
            }

            setBrowserContentFilter(contentFilterState);
            app.ui.settings.setSettingValue("PromptManager.BrowserContentFilter", contentFilterState);
            updateContentFilterBtn();
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

        viewModeBtn.onclick = () => {
            if (compactBrowser) {
                // Compact mode: only Grid and List make sense.
                currentViewMode = currentViewMode === "grid" ? "list" : "grid";
            } else {
                // Full mode: cycle Grid (large) -> Icon (medium) -> List (small) -> Grid.
                if (currentViewMode === "grid") {
                    currentViewMode = "icon";
                } else if (currentViewMode === "icon") {
                    currentViewMode = "list";
                } else {
                    currentViewMode = "grid";
                }
            }
            setViewMode(currentViewMode, browserPrefScope);
            localStorage.setItem(getViewModeStorageKey(browserPrefScope), currentViewMode);
            updateViewModeBtn();
            renderContent(searchInput.value);
        };

        let saveNameInput = null;
        let saveActionButton = null;
        let cancelSaveButton = null;

        const handleSaveAction = async () => {
            if (mode !== "save" || !onSave || !saveNameInput) {
                return;
            }

            const category = ensureSelectedCategory();
            if (!category) {
                await showInfo("Missing Category", "Please create or select a category before saving.");
                return;
            }

            const name = (saveNameInput.value || "").trim();
            if (!name) {
                await showInfo("Missing Name", "Please enter a name before saving.");
                saveNameInput.focus();
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
                    "#c44",
                    false
                );
                if (!overwrite) {
                    return;
                }
            }

            const saveResult = await onSave({ category: categoryLabel, name, overwrite });
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
            color: #8a95a6;
            text-align: center;
        `;

        const saveBar = document.createElement("div");
        if (mode === "save") {
            saveBar.style.cssText = `
                display: flex;
                gap: 8px;
                align-items: center;
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid ${UI.sectionBorder};
            `;

            saveNameInput = document.createElement("input");
            saveNameInput.type = "text";
            saveNameInput.value = selectedSaveName;
            saveNameInput.placeholder = saveNamePlaceholder;
            saveNameInput.style.cssText = `
                flex: 1;
                min-width: 0;
                padding: 7px 10px;
                background: ${UI.inputBg};
                border: 1px solid ${UI.inputBorder};
                border-radius: 4px;
                color: #fff;
                font-size: 13px;
                box-sizing: border-box;
                outline: none;
            `;
            saveNameInput.onfocus = () => saveNameInput.style.borderColor = UI.accent;
            saveNameInput.onblur = () => saveNameInput.style.borderColor = UI.inputBorder;
            saveNameInput.onkeydown = async (e) => {
                if (e.key === "Enter") {
                    e.preventDefault();
                    e.stopPropagation();
                    await handleSaveAction();
                }
            };

            saveActionButton = document.createElement("button");
            saveActionButton.textContent = saveButtonText;
            saveActionButton.style.cssText = `
                background: #2b6d3a;
                border: 1px solid #4a9158;
                color: #fff;
                padding: 7px 14px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                white-space: nowrap;
            `;
            saveActionButton.onclick = async () => {
                await handleSaveAction();
            };

            cancelSaveButton = document.createElement("button");
            cancelSaveButton.textContent = "Cancel";
            cancelSaveButton.style.cssText = `
                background: #313843;
                border: 1px solid #5f6773;
                color: #ccc;
                padding: 7px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                white-space: nowrap;
            `;
            cancelSaveButton.onclick = () => {
                resolve(null);
                cleanup();
            };

            saveBar.appendChild(saveNameInput);
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
                border-top: 1px solid ${UI.sectionBorder};
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

            const clearSelectionBtn = document.createElement("button");
            clearSelectionBtn.textContent = "Clear";
            clearSelectionBtn.style.cssText = `
                background: #313843;
                border: 1px solid #5f6773;
                color: #ccc;
                padding: 7px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                white-space: nowrap;
            `;
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

            const selectAllBtn = document.createElement("button");
            selectAllBtn.textContent = "Select All";
            selectAllBtn.style.cssText = `
                background: #313843;
                border: 1px solid #5f6773;
                color: #ccc;
                padding: 7px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                white-space: nowrap;
            `;
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

            const cancelBtn = document.createElement("button");
            cancelBtn.textContent = "Cancel";
            cancelBtn.style.cssText = `
                background: #313843;
                border: 1px solid #5f6773;
                color: #ccc;
                padding: 7px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                white-space: nowrap;
            `;
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

            const selectBtn = document.createElement("button");
            selectBtn.textContent = useComposerMultiSelectActions
                ? `Add as 1 Prompt (${selectedNames.size})`
                : `Select (${selectedNames.size})`;
            selectBtn.style.cssText = `
                background: #2b6d3a;
                border: 1px solid #4a9158;
                color: #fff;
                padding: 7px 14px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                white-space: nowrap;
            `;
            updateSelectButton = () => {
                const count = multiCategorySelect
                    ? Object.values(selectedByCategory).reduce((sum, set) => sum + set.size, 0)
                    : selectedNames.size;
                selectBtn.textContent = useComposerMultiSelectActions
                    ? `Add as 1 Prompt (${count})`
                    : `Select (${count})`;
                selectBtn.disabled = count === 0;
                selectBtn.style.opacity = count === 0 ? "0.55" : "1";
                selectBtn.style.cursor = count === 0 ? "default" : "pointer";
                if (addAllPromptsBtn) {
                    addAllPromptsBtn.textContent = `Add All Prompts (${count})`;
                    addAllPromptsBtn.disabled = count === 0;
                    addAllPromptsBtn.style.opacity = count === 0 ? "0.55" : "1";
                    addAllPromptsBtn.style.cursor = count === 0 ? "default" : "pointer";
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
                addAllPromptsBtn = document.createElement("button");
                addAllPromptsBtn.style.cssText = `
                    background: #2b6d3a;
                    border: 1px solid #4a9158;
                    color: #fff;
                    padding: 7px 14px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 13px;
                    white-space: nowrap;
                `;
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
