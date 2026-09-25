import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { PM_UI_PALETTE as UI } from "./ui_palette.js";
import {
    showInfo,
    showConfirm,
    DEFAULT_THUMBNAIL,
} from "./prompt_manager_advanced.js";
import { showThumbnailBrowser } from "./prompt_browser.js";
import { saveComposerCategorySettings } from "./prompt_composer_common.js";
import {
    buildSavePromptRequestBodyForSource,
    COMPOSER_ENDPOINT_PREFIX,
    getCategoryPromptEntriesForSource,
    getEndpointPrefixForSource,
    PROMPT_ENDPOINT_PREFIX,
    SOURCE_COMPOSE,
    SOURCE_PROMPT,
    SOURCE_SYSTEM_PROMPTS,
    SYSTEM_PROMPTS_ENDPOINT_PREFIX,
} from "./prompt_store_adapters.js";

const PMA_THEME = {
    panel: UI.panel || "hsl(216 11% 15%)",
    panelBorder: UI.panelBorder || "hsl(216 20% 65% / 0.24)",
    sectionBorder: UI.sectionBorder || "hsl(216 20% 65% / 0.20)",
    inputBg: UI.inputBg || "hsl(220 15% 10%)",
    inputBorder: UI.inputBorder || "hsl(218 10% 41%)",
    buttonBg: UI.buttonBg || "hsl(219 16% 18%)",
    cardBg: UI.cardBg || "hsl(219 16% 18%)",
    textPrimary: UI.textPrimary || "hsl(0 0% 87%)",
    textMuted: UI.textMuted || "hsl(0 0% 67%)",
    textHint: UI.textHint || "hsl(216 15% 65%)",
    accent: UI.accent || "hsl(208 73% 57% / 0.9)",
    accentSoft: UI.accentSoft || "hsl(208 73% 57% / 0.16)",
    accentBorder: UI.accentBorder || "hsl(208 73% 57% / 0.65)",
};

const NODE_CHROME_HEIGHT = 86;
const PROMPT_BROWSER_MIN_EXTRA_HEIGHT = 500;
const PROMPT_BROWSER_DEFAULT_PREVIEW_HEIGHT = 200;
const PROMPT_BROWSER_DEFAULT_NODE_WIDTH = 440;
const PROMPT_BROWSER_DEFAULT_NODE_HEIGHT = 700;
// Hard minimum node size - the resize handle can never shrink below this.
const PROMPT_BROWSER_MIN_NODE_HEIGHT = 500;
const PROMPT_BROWSER_SOURCE_PROP = "prompt_browser_source";
const PROMPT_BROWSER_BACKUP_DIR = "/mnt/Neuralnet/ComfyUI/user/default/prompt_backups";

function computePromptBrowserUiHeight(node) {
    const nodeHeight = Number(node?.size?.[1]) || PROMPT_BROWSER_DEFAULT_NODE_HEIGHT;
    return Math.max(220, nodeHeight - NODE_CHROME_HEIGHT);
}

function getComposerImportPromptEntries(categoryData) {
    return getCategoryPromptEntriesForSource(categoryData, SOURCE_COMPOSE);
}

function findExistingCategoryName(node, category) {
    const normalized = String(category || "").trim().toLowerCase();
    if (!normalized) return null;
    return getNodeCategories(node).find((existing) => String(existing || "").trim().toLowerCase() === normalized) || null;
}

function findExistingPromptName(node, category, promptName) {
    const canonicalCategory = findExistingCategoryName(node, category);
    if (!canonicalCategory) return null;
    const normalized = String(promptName || "").trim().toLowerCase();
    if (!normalized) return null;
    return getNodeNames(node, canonicalCategory).find((existing) => String(existing || "").trim().toLowerCase() === normalized) || null;
}

function analyzeComposerImportConflicts(node, importedData) {
    const conflicts = {
        duplicatePrompts: [],
        duplicateCategorySettings: [],
    };

    for (const [category, entries] of Object.entries(importedData || {})) {
        if (category === "__meta__" || !entries || typeof entries !== "object") continue;
        const existingCategory = findExistingCategoryName(node, category);
        const basePrompt = typeof entries?._base_prompt_ === "string" ? entries._base_prompt_.trim() : "";
        const promptType = typeof entries?._prompt_type_ === "string" ? entries._prompt_type_.trim() : "";
        const promptPrefix = typeof entries?._prompt_prefix_ === "string" ? entries._prompt_prefix_.trim() : "";

        if (existingCategory && (basePrompt || promptType || promptPrefix)) {
            conflicts.duplicateCategorySettings.push({
                category,
                existingCategory,
            });
        }

        for (const [name] of Object.entries(getComposerImportPromptEntries(entries))) {
            const existingPrompt = findExistingPromptName(node, category, name);
            if (existingPrompt) {
                conflicts.duplicatePrompts.push({
                    category,
                    existingCategory: existingCategory || category,
                    name,
                    existingPrompt,
                });
            }
        }
    }

    return conflicts;
}

function showComposerImportModeDialog({ duplicatePromptCount = 0, duplicateCategorySettingsCount = 0 }) {
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
            width: min(640px, calc(100vw - 48px));
            background: ${PMA_THEME.panel};
            border: 1px solid ${PMA_THEME.panelBorder};
            border-radius: 10px;
            padding: 18px 20px;
            box-shadow: 0 10px 32px rgba(0,0,0,0.45);
            color: ${PMA_THEME.textPrimary};
            z-index: 10000;
            box-sizing: border-box;
        `;

        const summaryLines = [];
        if (duplicatePromptCount > 0) {
            summaryLines.push(`${duplicatePromptCount} duplicate prompt${duplicatePromptCount === 1 ? "" : "s"}`);
        }
        if (duplicateCategorySettingsCount > 0) {
            summaryLines.push(`${duplicateCategorySettingsCount} existing categor${duplicateCategorySettingsCount === 1 ? "y" : "ies"} with settings`);
        }

        dialog.innerHTML = `
            <div style="font-size: 16px; font-weight: 700; margin-bottom: 10px;">Import Duplicates</div>
            <div style="color: ${PMA_THEME.textMuted}; line-height: 1.45; margin-bottom: 14px; white-space: normal; word-break: break-word;">
                The imported file contains entries that already exist.<br><br>
                ${summaryLines.join("<br>")}
            </div>
            <div style="color: ${PMA_THEME.textHint}; line-height: 1.45; margin-bottom: 18px;">
                Choose how duplicate entries should be handled.
            </div>
            <div style="display: flex; justify-content: flex-end; gap: 10px; flex-wrap: wrap;">
                <button class="cancel-btn" style="padding: 8px 14px; background: ${PMA_THEME.buttonBg}; color: ${PMA_THEME.textPrimary}; border: 1px solid ${PMA_THEME.inputBorder}; border-radius: 6px; cursor: pointer;">Cancel</button>
                <button class="skip-btn" style="padding: 8px 14px; background: ${PMA_THEME.buttonBg}; color: ${PMA_THEME.textPrimary}; border: 1px solid ${PMA_THEME.inputBorder}; border-radius: 6px; cursor: pointer;">Skip Existing</button>
                <button class="replace-btn" style="padding: 8px 14px; background: #c77dff; color: #fff; border: 1px solid transparent; border-radius: 6px; cursor: pointer;">Replace Existing</button>
            </div>
        `;

        const cleanup = () => {
            if (overlay.parentNode) document.body.removeChild(overlay);
            if (dialog.parentNode) document.body.removeChild(dialog);
            document.removeEventListener("keydown", onKeyDown, true);
        };

        const finish = (result) => {
            cleanup();
            resolve(result);
        };

        const onKeyDown = (event) => {
            if (event.key === "Escape") {
                finish("cancel");
            }
        };

        dialog.querySelector(".cancel-btn").onclick = () => finish("cancel");
        dialog.querySelector(".skip-btn").onclick = () => finish("skip_existing");
        dialog.querySelector(".replace-btn").onclick = () => finish("replace_existing");
        overlay.onclick = () => finish("cancel");

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        document.addEventListener("keydown", onKeyDown, true);
        dialog.querySelector(".skip-btn")?.focus();
    });
}

function getSourceWidget(node) {
    return node.widgets?.find((w) => w.name === "source") || null;
}

function getSourceValue(node) {
    const widget = getSourceWidget(node);
    return String(widget?.value || SOURCE_SYSTEM_PROMPTS);
}

function normalizePromptBrowserSource(value) {
    const raw = String(value || "");
    if (raw === SOURCE_PROMPT) return SOURCE_PROMPT;
    if (raw === SOURCE_SYSTEM_PROMPTS) return SOURCE_SYSTEM_PROMPTS;
    return SOURCE_COMPOSE;
}

function coercePromptBrowserSourceOrNull(value) {
    const raw = String(value || "");
    if (raw === SOURCE_PROMPT) return SOURCE_PROMPT;
    if (raw === SOURCE_COMPOSE) return SOURCE_COMPOSE;
    if (raw === SOURCE_SYSTEM_PROMPTS) return SOURCE_SYSTEM_PROMPTS;
    return null;
}

function persistPromptBrowserSource(node, sourceValue) {
    if (!node) return;
    node.properties = node.properties || {};
    node.properties[PROMPT_BROWSER_SOURCE_PROP] = normalizePromptBrowserSource(sourceValue);
}

function restorePromptBrowserSource(node, sourceWidget, info = null, options = {}) {
    if (!node) return SOURCE_SYSTEM_PROMPTS;
    const persist = options.persist !== false;
    const hasSavedProperty = Object.prototype.hasOwnProperty.call(node.properties || {}, PROMPT_BROWSER_SOURCE_PROP);
    const fromProperty = hasSavedProperty ? coercePromptBrowserSourceOrNull(node.properties?.[PROMPT_BROWSER_SOURCE_PROP]) : null;
    const fromWidget = coercePromptBrowserSourceOrNull(sourceWidget?.value);
    const fromInfo = coercePromptBrowserSourceOrNull(info?.properties?.[PROMPT_BROWSER_SOURCE_PROP]);
    // On workflow/tab restore, trust serialized workflow source first. New nodes
    // fall back to the widget default (System Prompts).
    const effective = info
        ? (fromInfo || fromWidget || fromProperty || SOURCE_SYSTEM_PROMPTS)
        : (fromProperty || fromWidget || SOURCE_SYSTEM_PROMPTS);
    if (sourceWidget) {
        sourceWidget.value = effective;
    }
    if (persist) {
        persistPromptBrowserSource(node, effective);
    }
    return effective;
}

function getPreferenceScopeForSource(source) {
    if (source === SOURCE_PROMPT) return "manager";
    if (source === SOURCE_SYSTEM_PROMPTS) return "system";
    return "composer";
}

function normalizePromptLibraryFilename(name, fallback = "prompt_library_data.json") {
    const trimmed = String(name || "").trim();
    if (!trimmed) return fallback;
    return trimmed.toLowerCase().endsWith(".json") ? trimmed : `${trimmed}.json`;
}

function joinPromptLibraryBrowserPath(base, leaf) {
    if (!base || !leaf) return leaf || base || "";
    const cleanBase = String(base).replace(/[\\/]+$/, "");
    const cleanLeaf = String(leaf).replace(/^[\\/]+/, "");
    return `${cleanBase}/${cleanLeaf}`;
}

function promptLibraryBrowserBasename(path) {
    const normalized = String(path || "").replace(/\\/g, "/");
    const idx = normalized.lastIndexOf("/");
    return idx >= 0 ? normalized.slice(idx + 1) : normalized;
}

async function fetchPromptLibraryBrowserListing(path = "") {
    const params = new URLSearchParams();
    if (path) params.set("path", path);
    params.set("kind", "json");
    const query = params.toString();
    const response = await fetch(`/prompt-extractor/path-browser/list${query ? `?${query}` : ""}`);
    if (!response.ok) {
        let message = `Request failed (${response.status})`;
        try {
            const err = await response.json();
            if (err?.error) message = err.error;
        } catch {
            // ignore
        }
        throw new Error(message);
    }
    return await response.json();
}

async function showPromptLibraryBrowser({
    mode = "open",
    title = "Browse Prompt Library JSON",
    confirmLabel = "Select",
    defaultFilename = "prompt_library_data.json",
} = {}) {
    return await new Promise((resolve) => {
        const preferredStartDir = PROMPT_BROWSER_BACKUP_DIR;
        const isSaveMode = mode === "save";
        let currentDir = "";
        let currentParent = null;
        let roots = [];
        let currentFiles = [];
        let currentDirs = [];
        let selectedFilePath = "";

        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.74);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;

        const dialog = document.createElement("div");
        dialog.style.cssText = `
            background: #17191d;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 4px;
            width: min(920px, calc(100vw - 48px));
            height: min(700px, calc(100vh - 48px));
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 18px 42px rgba(0,0,0,0.48);
        `;

        const header = document.createElement("div");
        header.style.cssText = `
            padding: 10px 14px 8px 14px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            display: flex;
            flex-direction: column;
            gap: 8px;
        `;

        const topRow = document.createElement("div");
        topRow.style.cssText = "display:flex; justify-content:space-between; align-items:center; gap:12px;";
        topRow.innerHTML = `
            <h3 style="margin:0; color:#e5e7eb; font-size:14px; font-weight:600; letter-spacing:0.01em;">${title}</h3>
            <button class="close-btn" style="background:none; border:none; color:#8b9098; font-size:22px; cursor:pointer; padding:0; width:28px; height:28px;">×</button>
        `;
        header.appendChild(topRow);

        const navRow = document.createElement("div");
        navRow.style.cssText = "display:flex; gap:6px; align-items:center; flex-wrap:nowrap;";
        const makeNavButton = (label) => {
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = label;
            button.style.cssText = "background:#23262b; border:1px solid rgba(255,255,255,0.08); border-radius:3px; color:#cfd5de; min-width:28px; height:28px; padding:0 8px; cursor:pointer; font-size:11px;";
            button.onmouseover = () => { button.style.background = "#2a2e34"; };
            button.onmouseout = () => { button.style.background = "#23262b"; };
            return button;
        };
        const upBtn = makeNavButton("◀");
        const refreshBtn = makeNavButton("↻");
        const inputBtn = makeNavButton("In");
        const outputBtn = makeNavButton("Out");
        const pathInput = document.createElement("input");
        pathInput.type = "text";
        pathInput.placeholder = "Paste folder path and press Enter";
        pathInput.style.cssText = "flex:1; min-width:240px; height:30px; font-size:12px; color:#e5e7eb; background:#111317; border:1px solid rgba(255,255,255,0.08); border-radius:3px; padding:0 10px; box-sizing:border-box;";
        navRow.appendChild(upBtn);
        navRow.appendChild(refreshBtn);
        navRow.appendChild(inputBtn);
        navRow.appendChild(outputBtn);
        navRow.appendChild(pathInput);
        header.appendChild(navRow);

        const body = document.createElement("div");
        body.style.cssText = "flex:1; min-height:0; overflow:hidden; padding:10px 14px 0 14px; display:flex; flex-direction:column; gap:0;";
        const listHeader = document.createElement("div");
        listHeader.textContent = "Name";
        listHeader.style.cssText = "height:30px; display:flex; align-items:center; padding:0 12px; background:#212121; color:#d7dbe1; font-size:12px; border:1px solid rgba(255,255,255,0.08); border-bottom:none; box-sizing:border-box;";
        const listing = document.createElement("div");
        listing.style.cssText = "flex:1; min-height:0; overflow:auto; display:flex; flex-direction:column; gap:0; background:#17191d; border:1px solid rgba(255,255,255,0.08);";
        body.appendChild(listHeader);
        body.appendChild(listing);

        const footer = document.createElement("div");
        footer.style.cssText = `
            padding: 10px 14px 14px 14px;
            border-top: 1px solid rgba(255,255,255,0.1);
            display: flex;
            flex-direction: column;
            gap: 10px;
        `;
        const filenameRow = document.createElement("div");
        filenameRow.style.cssText = "display:flex; align-items:center; gap:10px;";
        const filenameLabel = document.createElement("span");
        filenameLabel.textContent = "Name:";
        filenameLabel.style.cssText = "font-size:12px; color:#d7dbe1; min-width:48px;";
        const filenameInput = document.createElement("input");
        filenameInput.type = "text";
        filenameInput.value = isSaveMode ? normalizePromptLibraryFilename(defaultFilename, "prompt_library_data.json") : "";
        filenameInput.readOnly = !isSaveMode;
        filenameInput.style.cssText = "flex:1; height:32px; padding:0 10px; border-radius:3px; border:1px solid rgba(255,255,255,0.08); background:#111317; color:#e5e7eb; box-sizing:border-box;";
        filenameRow.appendChild(filenameLabel);
        filenameRow.appendChild(filenameInput);
        footer.appendChild(filenameRow);

        const footerButtons = document.createElement("div");
        footerButtons.style.cssText = "display:flex; justify-content:flex-end; gap:10px; flex-wrap:wrap;";
        const cancelBtn = document.createElement("button");
        cancelBtn.type = "button";
        cancelBtn.textContent = "Cancel";
        cancelBtn.style.cssText = "padding:7px 12px; background:#2a2f36; color:#e5e7eb; border:1px solid rgba(255,255,255,0.14); border-radius:7px; cursor:pointer;";
        const confirmBtn = document.createElement("button");
        confirmBtn.type = "button";
        confirmBtn.textContent = confirmLabel;
        confirmBtn.style.cssText = "padding:7px 12px; background:#23262b; color:#e5e7eb; border:1px solid rgba(255,255,255,0.14); border-radius:3px; cursor:pointer;";
        footerButtons.appendChild(cancelBtn);
        footerButtons.appendChild(confirmBtn);
        footer.appendChild(footerButtons);

        dialog.appendChild(header);
        dialog.appendChild(body);
        dialog.appendChild(footer);
        overlay.appendChild(dialog);

        const cleanup = () => {
            overlay.parentNode?.removeChild(overlay);
            document.removeEventListener("keydown", onKeyDown, true);
        };

        const finish = (value) => {
            cleanup();
            resolve(value);
        };

        const renderListing = () => {
            listing.innerHTML = "";
            pathInput.value = currentDir || "";
            upBtn.disabled = !currentParent;
            upBtn.style.opacity = currentParent ? "1" : "0.45";

            if (!currentDir && !roots.length) {
                const empty = document.createElement("div");
                empty.textContent = "No locations available.";
                empty.style.cssText = "text-align:center; padding:40px; color:#888;";
                listing.appendChild(empty);
                return;
            }

            if (!currentDir && roots.length) {
                roots.forEach((rootPath) => {
                    const item = document.createElement("button");
                    item.type = "button";
                    item.textContent = rootPath;
                    item.style.cssText = "text-align:left; min-height:38px; padding:0 12px; background:#17191d; color:#dce6f2; border:none; border-bottom:1px solid rgba(255,255,255,0.06); cursor:pointer;";
                    item.onmouseover = () => { item.style.background = "#1f2937"; };
                    item.onmouseout = () => { item.style.background = "#17191d"; };
                    item.onclick = async () => {
                        currentDir = rootPath;
                        await loadListing(rootPath);
                    };
                    listing.appendChild(item);
                });
                return;
            }

            if (!currentDirs.length && !currentFiles.length) {
                const empty = document.createElement("div");
                empty.textContent = "This folder is empty";
                empty.style.cssText = "padding:12px; color:#6b7280; font-size:12px; border-bottom:1px solid rgba(255,255,255,0.04);";
                listing.appendChild(empty);
                return;
            }

            currentDirs.forEach((dir) => {
                const item = document.createElement("button");
                item.type = "button";
                item.textContent = dir.name;
                item.style.cssText = "text-align:left; min-height:40px; padding:0 12px; background:#17191d; color:#dce6f2; border:none; border-bottom:1px solid rgba(255,255,255,0.06); cursor:pointer;";
                item.onmouseover = () => { item.style.background = "#1d344d"; };
                item.onmouseout = () => { item.style.background = "#17191d"; };
                item.onclick = async () => {
                    currentDir = dir.path;
                    selectedFilePath = "";
                    if (!isSaveMode) {
                        filenameInput.value = "";
                    }
                    await loadListing(dir.path);
                };
                listing.appendChild(item);
            });

            currentFiles.forEach((file) => {
                const item = document.createElement("button");
                item.type = "button";
                item.textContent = file.name;
                item.style.cssText = "text-align:left; min-height:40px; padding:0 12px; background:#17191d; color:#dbeafe; border:none; border-bottom:1px solid rgba(255,255,255,0.06); cursor:pointer;";
                item.onmouseover = () => { item.style.background = "#125d90"; };
                item.onmouseout = () => { item.style.background = "#17191d"; };
                item.onclick = () => {
                    selectedFilePath = file.path || joinPromptLibraryBrowserPath(currentDir, file.name);
                    filenameInput.value = file.name;
                    item.style.background = "#125d90";
                };
                item.ondblclick = () => {
                    selectedFilePath = file.path || joinPromptLibraryBrowserPath(currentDir, file.name);
                    filenameInput.value = file.name;
                    void confirmSelection();
                };
                listing.appendChild(item);
            });
        };

        const loadListing = async (targetPath = currentDir) => {
            listing.innerHTML = '<div style="text-align:center; padding:40px; color:#888;">Loading...</div>';
            try {
                let data = await fetchPromptLibraryBrowserListing(targetPath || "");
                if (data.mode === "roots") {
                    roots = Array.isArray(data.roots) ? data.roots : [];
                    if (!targetPath) {
                        currentDir = roots[0] || "";
                        if (currentDir) {
                            data = await fetchPromptLibraryBrowserListing(currentDir);
                        } else {
                            currentParent = null;
                            currentDirs = [];
                            currentFiles = [];
                            renderListing();
                            return true;
                        }
                    }
                }

                currentDir = data.current_path || currentDir || targetPath || "";
                currentParent = data.parent_path || null;
                roots = Array.isArray(data.roots) ? data.roots : roots;
                currentDirs = Array.isArray(data.dirs) ? data.dirs : [];
                currentFiles = Array.isArray(data.files) ? data.files : [];
                renderListing();
                return true;
            } catch (error) {
                console.error("[PromptBrowser] Error loading JSON browser listing:", error);
                listing.innerHTML = `<div style="text-align:center; padding:40px; color:rgba(220,53,69,0.9);">${String(error?.message || "Error loading folders")}</div>`;
                return false;
            }
        };

        const confirmSelection = async () => {
            if (isSaveMode) {
                const filename = normalizePromptLibraryFilename(filenameInput.value, defaultFilename);
                if (!currentDir) {
                    await showInfo("Save Failed", "Choose a target folder first.");
                    return;
                }
                if (!filename) {
                    await showInfo("Save Failed", "Enter a filename first.");
                    return;
                }
                finish(joinPromptLibraryBrowserPath(currentDir, filename));
                return;
            }

            if (!selectedFilePath) {
                await showInfo("Open Failed", "Choose a JSON file first.");
                return;
            }
            finish(selectedFilePath);
        };

        const onKeyDown = async (event) => {
            if (event.key === "Escape") {
                finish(null);
            } else if (event.key === "Enter" && document.activeElement === pathInput) {
                event.preventDefault();
                const next = pathInput.value.trim();
                if (next) {
                    currentDir = next;
                    await loadListing(next);
                }
            } else if (event.key === "Enter" && (document.activeElement === filenameInput || !isSaveMode)) {
                event.preventDefault();
                await confirmSelection();
            }
        };

        topRow.querySelector(".close-btn").onclick = () => finish(null);
        cancelBtn.onclick = () => finish(null);
        confirmBtn.onclick = async () => {
            await confirmSelection();
        };
        overlay.onclick = (event) => {
            if (event.target === overlay) finish(null);
        };
        upBtn.onclick = async () => {
            if (!currentParent) return;
            currentDir = currentParent;
            selectedFilePath = "";
            if (!isSaveMode) {
                filenameInput.value = "";
            }
            await loadListing(currentDir);
        };
        refreshBtn.onclick = async () => {
            await loadListing(currentDir);
        };
        inputBtn.onclick = async () => {
            const inputRoot = Array.isArray(roots) ? roots[0] : "";
            if (!inputRoot) return;
            currentDir = inputRoot;
            selectedFilePath = "";
            if (!isSaveMode) {
                filenameInput.value = "";
            }
            await loadListing(currentDir);
        };
        outputBtn.onclick = async () => {
            const outputRoot = Array.isArray(roots) ? roots[1] : "";
            if (!outputRoot) return;
            currentDir = outputRoot;
            selectedFilePath = "";
            if (!isSaveMode) {
                filenameInput.value = "";
            }
            await loadListing(currentDir);
        };

        document.body.appendChild(overlay);
        document.addEventListener("keydown", onKeyDown, true);
        void (async () => {
            const loadedPreferred = await loadListing(preferredStartDir);
            if (!loadedPreferred) {
                await loadListing("");
            }
        })();
        filenameInput.focus();
        filenameInput.select();
    });
}

async function loadPromptsFromEndpoint(endpointPrefix) {
    const suffix = endpointPrefix === PROMPT_ENDPOINT_PREFIX ? "/get-prompts" : "/get-prompts";
    const response = await fetch(`${endpointPrefix}${suffix}`);
    return await response.json();
}

async function loadActivePrompts(node) {
    const requestId = (node._promptBrowserLoadRequestId || 0) + 1;
    node._promptBrowserLoadRequestId = requestId;
    try {
        const source = getSourceValue(node);
        const endpointPrefix = getEndpointPrefixForSource(source);
        const prompts = await loadPromptsFromEndpoint(endpointPrefix);
        if (requestId !== node._promptBrowserLoadRequestId) {
            return prompts;
        }
        node.composerPrompts = prompts;
        node.prompts = prompts;
        node._composerEndpointPrefix = endpointPrefix;
        node._composerSource = source;
        return prompts;
    } catch (err) {
        console.error("[PromptBrowser] Error loading prompts:", err);
        node.composerPrompts = {};
        node.prompts = {};
        return {};
    }
}

function getNodePromptsData(node) {
    return node.prompts || node.composerPrompts || {};
}

function getNodeCategories(node) {
    const data = getNodePromptsData(node);
    return Object.keys(data)
        .filter((category) => category !== "__meta__")
        .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
}

function getNodePromptEntries(node, categoryData) {
    const source = getSourceValue(node);
    return getCategoryPromptEntriesForSource(categoryData, source);
}

function getNodeNames(node, category) {
    const data = getNodePromptsData(node);
    const categoryData = data?.[category];
    if (!categoryData || typeof categoryData !== "object") return [];
    return Object.keys(getNodePromptEntries(node, categoryData))
        .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
}

function getNodeEntry(node, category, name) {
    const data = getNodePromptsData(node);
    const categoryData = data?.[category];
    if (!categoryData || typeof categoryData !== "object") return null;
    if (name && getNodePromptEntries(node, categoryData)[name] === undefined) return null;
    return getNodePromptEntries(node, categoryData)[name] || null;
}

async function saveComposerCategory(category, promptType = "") {
    try {
        const resp = await fetch(`${COMPOSER_ENDPOINT_PREFIX}/save-category`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                category_name: category,
                prompt_type: promptType || "",
            }),
        });
        return await resp.json();
    } catch (err) {
        console.error("[PromptBrowser] Error saving composer category:", err);
        return { success: false, error: String(err) };
    }
}

function getSelectedPrompts(node) {
    const widget = node.widgets?.find((w) => w.name === "selected_prompts");
    if (!widget) return [];
    try {
        const parsed = JSON.parse(widget.value || "[]");
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
}

function setSelectedPrompts(node, names) {
    const widget = node.widgets?.find((w) => w.name === "selected_prompts");
    if (!widget) return;
    const normalized = Array.isArray(names) ? names.map((name) => String(name || "").trim()).filter(Boolean) : [];
    widget.value = JSON.stringify(normalized);
}

async function saveSourcePrompt(node, source, category, name, text, thumbnail = null, promptCategory = null) {
    try {
        const endpointPrefix = getEndpointPrefixForSource(source);
        const payload = buildSavePromptRequestBodyForSource(source, {
            category,
            name,
            text,
            thumbnail,
            prompt_category: promptCategory,
        });
        const resp = await fetch(`${endpointPrefix}/save-prompt`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (data.success) {
            node.composerPrompts = data.prompts;
            node.prompts = data.prompts;
        }
        return data;
    } catch (err) {
        console.error("[PromptBrowser] Error saving prompt:", err);
        return { success: false, error: String(err) };
    }
}

async function deleteSourcePrompt(node, source, category, name) {
    try {
        const endpointPrefix = getEndpointPrefixForSource(source);
        const resp = await fetch(`${endpointPrefix}/delete-prompt`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ category, name }),
        });
        const data = await resp.json();
        if (data.success) {
            node.composerPrompts = data.prompts;
            node.prompts = data.prompts;
        }
        return data;
    } catch (err) {
        console.error("[PromptBrowser] Error deleting prompt:", err);
        return { success: false, error: String(err) };
    }
}

function createComposerButton(text, callback) {
    const button = document.createElement("button");
    button.textContent = text;
    button.style.flex = "1";
    button.style.minWidth = "70px";
    button.style.padding = "6px 8px";
    button.style.cursor = "pointer";
    button.style.backgroundColor = "#222";
    button.style.color = "#fff";
    button.style.border = "1px solid #444";
    button.style.borderRadius = "6px";
    button.style.fontSize = "11px";
    button.style.whiteSpace = "nowrap";
    button.style.overflow = "hidden";
    button.style.textOverflow = "ellipsis";
    button.style.height = "28px";
    button.style.display = "flex";
    button.style.alignItems = "center";
    button.style.justifyContent = "center";
    button.onclick = callback;
    return button;
}

function createComposerDropdownButton(text, items) {
    const container = document.createElement("div");
    container.style.position = "relative";
    container.style.flex = "1";
    container.style.minWidth = "70px";

    const button = document.createElement("button");
    button.textContent = text;
    button.style.width = "100%";
    button.style.padding = "6px 8px";
    button.style.cursor = "pointer";
    button.style.backgroundColor = "#222";
    button.style.color = "#fff";
    button.style.border = "1px solid #444";
    button.style.borderRadius = "6px";
    button.style.fontSize = "11px";
    button.style.whiteSpace = "nowrap";
    button.style.height = "28px";
    button.style.display = "flex";
    button.style.alignItems = "center";
    button.style.justifyContent = "center";

    const dropdown = document.createElement("div");
    dropdown.style.cssText = `
        position: fixed;
        background: ${PMA_THEME.panel};
        border: 1px solid ${PMA_THEME.inputBorder};
        border-radius: 6px;
        z-index: 999999;
        display: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        min-width: 140px;
    `;
    document.body.appendChild(dropdown);

    function buildItems() {
        dropdown.innerHTML = "";
        const resolved = typeof items === "function" ? items() : items;
        resolved.forEach((item) => {
            if (item.divider) {
                const divider = document.createElement("div");
                divider.style.cssText = `height: 1px; background: ${PMA_THEME.sectionBorder}; margin: 4px 0;`;
                dropdown.appendChild(divider);
            } else {
                const menuItem = document.createElement("div");
                menuItem.textContent = item.label;
                menuItem.style.cssText = `
                    padding: 8px 12px;
                    cursor: pointer;
                    font-size: 11px;
                    color: ${PMA_THEME.textPrimary};
                    white-space: nowrap;
                `;
                menuItem.addEventListener("mouseenter", () => {
                    menuItem.style.backgroundColor = PMA_THEME.accentSoft;
                });
                menuItem.addEventListener("mouseleave", () => {
                    menuItem.style.backgroundColor = "transparent";
                });
                menuItem.addEventListener("click", (e) => {
                    e.stopPropagation();
                    dropdown.style.display = "none";
                    item.action();
                });
                dropdown.appendChild(menuItem);
            }
        });
    }

    button.addEventListener("click", (e) => {
        e.stopPropagation();
        const isVisible = dropdown.style.display === "block";
        if (isVisible) {
            dropdown.style.display = "none";
        } else {
            buildItems();
            const rect = button.getBoundingClientRect();
            dropdown.style.left = rect.left + "px";
            dropdown.style.top = (rect.bottom + 2) + "px";
            dropdown.style.display = "block";
        }
    });

    document.addEventListener("click", (e) => {
        if (!dropdown.contains(e.target) && e.target !== button) {
            dropdown.style.display = "none";
        }
    });

    container.appendChild(button);
    return container;
}

function setupPromptBrowserNode(nodeType, nodeData) {
    if (nodeData?.name !== "PromptBrowser") return;

    const getPromptBrowserMinHeight = (node) => {
        return PROMPT_BROWSER_MIN_NODE_HEIGHT;
    };

    const enforcePromptBrowserMinHeight = (node) => {
        if (!node || !Array.isArray(node.size)) return;
        const minHeight = getPromptBrowserMinHeight(node);
        if (node.size[1] < minHeight) {
            node.setSize([node.size[0], minHeight]);
        }
    };

    const clearPromptBrowserSelection = (node) => {
        if (!node) return;
        const categoryWidget = node.widgets?.find((w) => w.name === "category");
        const nameWidget = node.widgets?.find((w) => w.name === "name");
        const textWidget = node.widgets?.find((w) => w.name === "text");
        if (categoryWidget) categoryWidget.value = "";
        if (nameWidget) nameWidget.value = "";
        if (textWidget) textWidget.value = "";
        setSelectedPrompts(node, []);

        if (typeof node.updateComposerSelectorDisplay === "function") {
            node.updateComposerSelectorDisplay();
        }
        if (typeof node.updateComposerPromptEditor === "function") {
            node.updateComposerPromptEditor();
        }
        if (typeof node.updateComposerPreview === "function") {
            node.updateComposerPreview();
        }
    };

    const attachPromptBrowserInstanceResizeGuard = (node) => {
        if (!node || node._promptBrowserInstanceResizeWrapped) return;
        const onResize = node.onResize;
        node.onResize = function (size) {
            // Hard clamp so the node can never shrink below the image + UI.
            if (size) {
                if (size[0] < PROMPT_BROWSER_DEFAULT_NODE_WIDTH) size[0] = PROMPT_BROWSER_DEFAULT_NODE_WIDTH;
                if (size[1] < PROMPT_BROWSER_MIN_NODE_HEIGHT) size[1] = PROMPT_BROWSER_MIN_NODE_HEIGHT;
            }
            const result = onResize ? onResize.apply(this, arguments) : undefined;
            this.updateComposerRootLayout?.();
            return result;
        };
        node._promptBrowserInstanceResizeWrapped = true;
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated?.apply(this, arguments);
        const node = this;

        node.composerPrompts = {};
        node.prompts = {};
        node._configuredFromWorkflow = false;
        node._restoringFromWorkflow = false;
        node.isNewUnsavedPrompt = false;
        node.newPromptCategory = null;
        node.newPromptName = null;
        node._composerEndpointPrefix = getEndpointPrefixForSource(getSourceValue(node));
        if (!Number.isFinite(Number(node._composerPreviewHeight))) {
            node._composerPreviewHeight = PROMPT_BROWSER_DEFAULT_PREVIEW_HEIGHT;
        }

        // Match Prompt Manager Advanced footprint.
        node.setSize([PROMPT_BROWSER_DEFAULT_NODE_WIDTH, PROMPT_BROWSER_DEFAULT_NODE_HEIGHT]);
        enforcePromptBrowserMinHeight(node);
        attachPromptBrowserInstanceResizeGuard(node);

        // Hide native source/category/name/text widgets; custom DOM controls replace them.
        const sourceWidget = node.widgets.find((w) => w.name === "source");
        const categoryWidget = node.widgets.find((w) => w.name === "category");
        const nameWidget = node.widgets.find((w) => w.name === "name");
        const textWidget = node.widgets.find((w) => w.name === "text");
        const selectedPromptsWidget = node.widgets.find((w) => w.name === "selected_prompts");
        if (sourceWidget) {
            sourceWidget.type = "converted-widget";
            sourceWidget.computeSize = () => [0, 0];
            sourceWidget.hidden = true;
            // Do not persist during initial create; onConfigure may still restore
            // the real serialized workflow source in the same lifecycle.
            restorePromptBrowserSource(node, sourceWidget, null, { persist: false });
        }
        if (categoryWidget) {
            categoryWidget.type = "converted-widget";
            categoryWidget.computeSize = () => [0, 0];
            categoryWidget.hidden = true;
        }
        if (nameWidget) {
            nameWidget.type = "converted-widget";
            nameWidget.computeSize = () => [0, 0];
            nameWidget.hidden = true;
        }
        if (textWidget) {
            textWidget.type = "converted-widget";
            textWidget.computeSize = () => [0, 0];
            textWidget.hidden = true;
        }
        if (selectedPromptsWidget) {
            selectedPromptsWidget.type = "converted-widget";
            selectedPromptsWidget.computeSize = () => [0, 0];
            selectedPromptsWidget.hidden = true;
        }
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const inp = node.inputs[i];
            if (inp.name === "category" || inp.name === "name") {
                node.removeInput(i);
            }
        }

        if (selectedPromptsWidget && (selectedPromptsWidget.value == null || selectedPromptsWidget.value === "")) {
            selectedPromptsWidget.value = "[]";
        }

        // Build the root DOM widget synchronously so LiteGraph/ComfyUI can position
        // and size the widget container before the first paint. Prompt Composer uses
        // the same pattern. The child UI sections are populated after data loads.
        buildComposerRoot(node);

        loadActivePrompts(node).then(() => {
            buildComposerPreview(node);
            buildComposerSourceBar(node);
            buildComposerSelectorBar(node);
            buildComposerPromptEditor(node);
            buildComposerButtonBar(node);
            // Only clear defaults for brand-new nodes. During workflow/tab restore,
            // onConfigure runs before this async block completes and sets
            // _configuredFromWorkflow=true, which preserves serialized selection.
            if (!node._configuredFromWorkflow) {
                clearPromptBrowserSelection(node);
            }
            syncSelectorToData(node);
            updateComposerLastSavedState(node);
            refreshComposerPromptInputGhosting(node);
            if (typeof node.refreshComposerMultiUiState === "function") {
                node.refreshComposerMultiUiState();
            }
            enforcePromptBrowserMinHeight(node);
        });

        api.addEventListener("prompt-manager-update-text", (event) => {
            if (String(event.detail.node_id) !== String(node.id)) return;
            node._composerIncomingPrompt = event.detail.prompt || "";
            const usePromptInput = event.detail.use_prompt_input === true;
            if (textWidget && usePromptInput) {
                textWidget.value = node._composerIncomingPrompt;
            }
            refreshComposerPromptInputGhosting(node);
            app.graph.setDirtyCanvas(true, true);
        });

        const usePromptInputWidget = node.widgets.find((w) => w.name === "use_prompt_input");
        if (usePromptInputWidget && !usePromptInputWidget._composerWrapped) {
            const originalCallback = usePromptInputWidget.callback;
            usePromptInputWidget.callback = function () {
                if (typeof originalCallback === "function") {
                    originalCallback.apply(this, arguments);
                }
                refreshComposerPromptInputGhosting(node);
            };
            usePromptInputWidget._composerWrapped = true;
        }

        const sourceValueWidget = getSourceWidget(node);
        if (sourceValueWidget && !sourceValueWidget._composerWrapped) {
            const originalSourceCallback = sourceValueWidget.callback;
            sourceValueWidget.callback = async function () {
                if (typeof originalSourceCallback === "function") {
                    await originalSourceCallback.apply(this, arguments);
                }
                persistPromptBrowserSource(node, sourceValueWidget.value);
                await loadActivePrompts(node);
                if (!node._restoringFromWorkflow) {
                    clearPromptBrowserSelection(node);
                }
                syncSelectorToData(node);
                updateComposerLastSavedState(node);
                if (typeof node.refreshComposerMultiUiState === "function") {
                    node.refreshComposerMultiUiState();
                }
                app.graph.setDirtyCanvas(true, true);
            };
            sourceValueWidget._composerWrapped = true;
        }

        refreshComposerPromptInputGhosting(node);

        return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
        const result = onConfigure?.apply(this, arguments);
        const node = this;
        node._configuredFromWorkflow = true;
        node._restoringFromWorkflow = true;

        const sourceWidget = node.widgets.find((w) => w.name === "source");
        const categoryWidget = node.widgets.find((w) => w.name === "category");
        const nameWidget = node.widgets.find((w) => w.name === "name");
        const textWidget = node.widgets.find((w) => w.name === "text");
        const selectedPromptsWidget = node.widgets.find((w) => w.name === "selected_prompts");
        if (sourceWidget) {
            sourceWidget.type = "converted-widget";
            sourceWidget.computeSize = () => [0, 0];
            sourceWidget.hidden = true;
            restorePromptBrowserSource(node, sourceWidget, info);
        }
        if (categoryWidget) {
            categoryWidget.type = "converted-widget";
            categoryWidget.computeSize = () => [0, 0];
            categoryWidget.hidden = true;
        }
        if (nameWidget) {
            nameWidget.type = "converted-widget";
            nameWidget.computeSize = () => [0, 0];
            nameWidget.hidden = true;
        }
        if (textWidget) {
            textWidget.type = "converted-widget";
            textWidget.computeSize = () => [0, 0];
            textWidget.hidden = true;
        }
        if (selectedPromptsWidget) {
            selectedPromptsWidget.type = "converted-widget";
            selectedPromptsWidget.computeSize = () => [0, 0];
            selectedPromptsWidget.hidden = true;
        }
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const inp = node.inputs[i];
            if (inp.name === "category" || inp.name === "name") {
                node.removeInput(i);
            }
        }

        // Re-create/ensure the root DOM widget synchronously during workflow restore
        // so the widget container is positioned before async data loads.
        buildComposerRoot(node);

        loadActivePrompts(node).then(() => {
            buildComposerPreview(node);
            buildComposerSourceBar(node);
            buildComposerSelectorBar(node);
            buildComposerPromptEditor(node);
            buildComposerButtonBar(node);
            syncSelectorToData(node);
            updateComposerLastSavedState(node);
            refreshComposerPromptInputGhosting(node);
            if (typeof node.refreshComposerMultiUiState === "function") {
                node.refreshComposerMultiUiState();
            }
            attachPromptBrowserInstanceResizeGuard(node);
            enforcePromptBrowserMinHeight(node);
        }).finally(() => {
            node._restoringFromWorkflow = false;
        });

        refreshComposerPromptInputGhosting(node);
        attachPromptBrowserInstanceResizeGuard(node);
        enforcePromptBrowserMinHeight(node);

        return result;
    };

    if (!nodeType.prototype._promptBrowserDrawClampWrapped) {
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            // Final safety net: enforce min height on every draw.
            enforcePromptBrowserMinHeight(this);
            return onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;
        };
        nodeType.prototype._promptBrowserDrawClampWrapped = true;
    }
}

function syncSelectorToData(node) {
    if (typeof node.updateComposerSourceDisplay === "function") {
        node.updateComposerSourceDisplay();
    }
    if (typeof node.updateComposerSelectorDisplay === "function") {
        node.updateComposerSelectorDisplay();
    }
}

function buildComposerRoot(node) {
    if (node._composerRootBuilt) return;
    node._composerRootBuilt = true;

    const root = document.createElement("div");
    root.style.cssText = `
        display: flex;
        flex-direction: column;
        width: 100%;
        height: 100%;
        min-height: 0;
        box-sizing: border-box;
        margin-top: -6px;
        padding: 0;
        overflow: hidden;
        position: relative;
    `;

    const surface = document.createElement("div");
    surface.style.cssText = `
        display: flex;
        flex-direction: column;
        width: 100%;
        height: 100%;
        min-height: 0;
        gap: 8px;
        box-sizing: border-box;
        padding: 0;
        overflow: hidden;
    `;

    const mkSection = () => {
        const section = document.createElement("div");
        section.style.cssText = "width: 100%; min-width: 0; box-sizing: border-box;";
        return section;
    };

    const previewSection = mkSection();
    previewSection.style.flex = "0 0 auto";
    const sourceSection = mkSection();
    sourceSection.style.flex = "0 0 26px";
    const nameSection = mkSection();
    nameSection.style.flex = "0 0 26px";
    const promptSection = mkSection();
    promptSection.style.flex = "1 1 auto";
    promptSection.style.minHeight = "110px";
    const buttonsSection = mkSection();
    buttonsSection.style.flex = "0 0 34px";

    surface.appendChild(previewSection);
    surface.appendChild(sourceSection);
    surface.appendChild(nameSection);
    surface.appendChild(promptSection);
    surface.appendChild(buttonsSection);
    root.appendChild(surface);

    const widget = node.addDOMWidget("composer_root", "div", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => computePromptBrowserUiHeight(node),
        getHeight: () => "100%",
    });

    node._composerRoot = {
        root,
        surface,
        previewSection,
        sourceSection,
        nameSection,
        promptSection,
        buttonsSection,
        widget,
    };

    node.refreshComposerRootHeight = () => {
        const h = computePromptBrowserUiHeight(node);
        root.style.setProperty("--comfy-widget-min-height", `${h}px`);
        root.style.setProperty("--comfy-widget-height", `${h}px`);
    };

    node.updateComposerRootLayout = () => {
        if (!node._composerRoot?.root) return;
        const previewHeight = Math.max(120, Number(node._composerPreviewHeight) || 190);
        if (node._composerRoot.previewSection) {
            node._composerRoot.previewSection.style.height = `${previewHeight}px`;
            node._composerRoot.previewSection.style.flex = `0 0 ${previewHeight}px`;
        }
        node.refreshComposerRootHeight?.();
    };
    node.updateComposerRootLayout();
    node.refreshComposerRootHeight?.();
}

function shouldWarnComposerUnsavedChanges() {
    return app.ui.settings.getSettingValue("PromptManager.WarnUnsavedChanges") !== false;
}

function refreshComposerPromptInputGhosting(node) {
    const textWidget = node.widgets?.find((w) => w.name === "text");
    const usePromptInputWidget = node.widgets?.find((w) => w.name === "use_prompt_input");
    if (!textWidget || !usePromptInputWidget) return;

    const promptInputConnection = node.inputs?.find((inp) => inp.name === "prompt");
    const isPromptConnected = promptInputConnection && promptInputConnection.link != null;
    const usePromptInput = usePromptInputWidget.value === true;

    if (usePromptInput && isPromptConnected) {
        if (typeof node._composerIncomingPrompt === "string") {
            textWidget.value = node._composerIncomingPrompt;
        }
        textWidget.disabled = true;
        if (textWidget.inputEl) {
            textWidget.inputEl.style.pointerEvents = "auto";
            textWidget.inputEl.readOnly = true;
        }
    } else {
        textWidget.disabled = false;
        if (textWidget.inputEl) {
            textWidget.inputEl.readOnly = false;
        }
    }

    if (typeof node.updateComposerPromptEditor === "function") {
        node.updateComposerPromptEditor();
    }
    if (typeof node.updateComposerPreview === "function") {
        node.updateComposerPreview();
    }
}

function refreshComposerMultiUiState(node) {
    const multiCount = getSelectedPrompts(node).length;
    const isMulti = multiCount > 1;
    node._composerIsMultiSelection = isMulti;

    if (node._composerPromptEditor?.container) {
        node._composerPromptEditor.container.style.display = "";
    }

    const saveBtn = node._composerSaveButton;
    if (saveBtn) {
        saveBtn.disabled = isMulti;
        saveBtn.style.opacity = isMulti ? "0.45" : "1";
        saveBtn.style.cursor = isMulti ? "not-allowed" : "pointer";
        saveBtn.title = isMulti ? "Save is disabled while multiple prompts are selected" : "";
    }

    if (typeof node.updateComposerSelectorDisplay === "function") {
        node.updateComposerSelectorDisplay();
    }
    if (typeof node.updateComposerPromptEditor === "function") {
        node.updateComposerPromptEditor();
    }
    if (typeof node.updateComposerPreview === "function") {
        node.updateComposerPreview();
    }
    if (typeof node.updateComposerRootLayout === "function") {
        node.updateComposerRootLayout();
    }
    app.graph.setDirtyCanvas(true, true);
}

function buildComposerSourceBar(node) {
    if (node._composerSourceBarBuilt) return;
    node._composerSourceBarBuilt = true;

    const sourceWidget = getSourceWidget(node);
    if (!sourceWidget) return;

    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        align-items: center;
        gap: 0;
        background: #1a1a1a;
        overflow: visible;
        height: 26px;
        margin: 0;
        box-sizing: border-box;
    `;

    const arrowStyle = `
        background: #2a2a2a;
        border: none;
        color: #888;
        padding: 0 10px;
        height: 100%;
        cursor: pointer;
        font-size: 10px;
        transition: all 0.15s ease;
    `;

    const makeArrow = (label) => {
        const btn = document.createElement("button");
        btn.textContent = label;
        btn.style.cssText = arrowStyle;
        btn.onmouseover = () => {
            btn.style.background = "#3a3a3a";
            btn.style.color = "#fff";
        };
        btn.onmouseout = () => {
            btn.style.background = "#2a2a2a";
            btn.style.color = "#888";
        };
        return btn;
    };

    const valueDisplay = document.createElement("div");
    valueDisplay.style.cssText = `
        flex: 1;
        text-align: center;
        color: #ddd;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0 10px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        background: #1f1f1f;
        border: 1px solid #3a3a3a;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
    `;

    const leftArrow = makeArrow("◀");
    const rightArrow = makeArrow("▶");

    const getValues = () => {
        const raw = sourceWidget?.options?.values;
        if (Array.isArray(raw) && raw.length > 0) {
            return raw.map((v) => String(v));
        }
        return [SOURCE_COMPOSE, SOURCE_PROMPT, SOURCE_SYSTEM_PROMPTS];
    };

    const updateDisplay = () => {
        valueDisplay.textContent = String(sourceWidget.value || SOURCE_SYSTEM_PROMPTS).toUpperCase();
    };

    const cycleSource = async (direction) => {
        const values = getValues();
        const current = String(sourceWidget.value || values[0]);
        const idx = Math.max(0, values.indexOf(current));
        const nextIdx = (idx + direction + values.length) % values.length;
        const nextValue = values[nextIdx];

        sourceWidget.value = nextValue;
        persistPromptBrowserSource(node, nextValue);
        if (typeof sourceWidget.callback === "function") {
            await sourceWidget.callback(nextValue);
        }
        updateDisplay();
    };

    leftArrow.onclick = async (e) => {
        e.stopPropagation();
        await cycleSource(-1);
    };
    rightArrow.onclick = async (e) => {
        e.stopPropagation();
        await cycleSource(1);
    };

    container.appendChild(leftArrow);
    container.appendChild(valueDisplay);
    container.appendChild(rightArrow);

    if (node._composerRoot?.sourceSection) {
        node._composerRoot.sourceSection.innerHTML = "";
        node._composerRoot.sourceSection.appendChild(container);
    } else {
        const widget = node.addDOMWidget("composer_source", "div", container, { hideOnZoom: false });
        widget.computeSize = (width) => [width, 24];
    }

    node._composerSourceBar = { container, valueDisplay };
    node.updateComposerSourceDisplay = updateDisplay;
    updateDisplay();
}

function buildComposerPreview(node) {
    if (node._composerPreviewBuilt) return;
    node._composerPreviewBuilt = true;

    const defaultHeight = PROMPT_BROWSER_DEFAULT_PREVIEW_HEIGHT;
    const container = document.createElement("div");
    container.style.cssText = `display: flex; width: 100%; height: 100%; box-sizing: border-box;`;

    const previewBox = document.createElement("div");
    previewBox.style.cssText = `
        position: relative;
        width: 100%;
        height: ${defaultHeight}px;
        border-radius: 4px;
        background: ${PMA_THEME.inputBg};
        border: 1px solid ${PMA_THEME.inputBorder};
        overflow: hidden;
        min-height: 120px;
        box-sizing: border-box;
    `;

    const image = document.createElement("img");
    image.style.cssText = `
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
        object-position: center;
        display: none;
        user-select: none;
        pointer-events: none;
    `;

    const placeholderViewport = document.createElement("div");
    placeholderViewport.style.cssText = `
        position: absolute;
        inset: 0;
        display: none;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        pointer-events: none;
    `;

    const placeholderPoster = document.createElement("div");
    placeholderPoster.style.cssText = `
        height: 100%;
        aspect-ratio: 3 / 4;
        background-image: url(${DEFAULT_THUMBNAIL});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        border-radius: 2px;
    `;
    placeholderViewport.appendChild(placeholderPoster);

    const emptyLabel = document.createElement("div");
    emptyLabel.style.cssText = `
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-size: 12px;
        color: ${PMA_THEME.textMuted};
        padding: 8px;
        box-sizing: border-box;
    `;
    emptyLabel.textContent = "No prompt selected";

    const tiles = document.createElement("div");
    tiles.style.cssText = `
        position: absolute;
        inset: 0;
        display: none;
        grid-template-columns: repeat(auto-fit, minmax(64px, 1fr));
        grid-auto-rows: minmax(48px, 1fr);
        gap: 2px;
        padding: 2px;
        box-sizing: border-box;
    `;

    previewBox.appendChild(image);
    previewBox.appendChild(placeholderViewport);
    previewBox.appendChild(tiles);
    previewBox.appendChild(emptyLabel);
    container.appendChild(previewBox);

    if (node._composerRoot?.previewSection) {
        node._composerRoot.previewSection.innerHTML = "";
        node._composerRoot.previewSection.appendChild(container);
    } else {
        const widget = node.addDOMWidget("composer_preview", "div", container, { hideOnZoom: false });
        widget.computeSize = function (width) {
            const h = Math.max(120, Number(node._composerPreviewHeight) || defaultHeight);
            return [width, h + 4];
        };
    }

    node._composerPreview = { container, previewBox, image, placeholderViewport, emptyLabel, tiles };

    previewBox.style.cursor = "pointer";
    const resizeCornerSize = 18;
    const dragThresholdPx = 6;
    let pointerDown = false;
    let pointerStartX = 0;
    let pointerStartY = 0;
    let pointerMoved = false;
    let pointerStartedInResizeCorner = false;
    let suppressNextPreviewClick = false;

    const isInResizeCorner = (event) => {
        const rect = previewBox.getBoundingClientRect();
        const localX = Number(event?.clientX) - rect.left;
        const localY = Number(event?.clientY) - rect.top;
        if (!Number.isFinite(localX) || !Number.isFinite(localY)) return false;
        return localX >= (rect.width - resizeCornerSize) && localY >= (rect.height - resizeCornerSize);
    };

    previewBox.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        pointerDown = true;
        pointerStartX = Number(e.clientX) || 0;
        pointerStartY = Number(e.clientY) || 0;
        pointerMoved = false;
        pointerStartedInResizeCorner = isInResizeCorner(e);
    });

    previewBox.addEventListener("pointermove", (e) => {
        if (!pointerDown) return;
        const dx = Math.abs((Number(e.clientX) || 0) - pointerStartX);
        const dy = Math.abs((Number(e.clientY) || 0) - pointerStartY);
        if (dx >= dragThresholdPx || dy >= dragThresholdPx) {
            pointerMoved = true;
        }
    });

    previewBox.addEventListener("pointercancel", () => {
        pointerDown = false;
        pointerMoved = false;
        pointerStartedInResizeCorner = false;
    });

    previewBox.addEventListener("pointerup", () => {
        if (pointerDown) {
            suppressNextPreviewClick = pointerStartedInResizeCorner || pointerMoved;
        }
        pointerDown = false;
        pointerMoved = false;
        pointerStartedInResizeCorner = false;
    });

    previewBox.addEventListener("click", async (e) => {
        if (suppressNextPreviewClick) {
            suppressNextPreviewClick = false;
            return;
        }
        if (typeof node.openComposerPromptBrowser === "function") {
            await node.openComposerPromptBrowser(e);
        }
    });

    node.updateComposerPreview = () => {
        const ui = node._composerPreview;
        if (!ui) return;

        const categoryWidget = node.widgets.find((w) => w.name === "category");
        const nameWidget = node.widgets.find((w) => w.name === "name");
        const category = categoryWidget?.value || "";
        const name = nameWidget?.value || "";
        const selected = getSelectedPrompts(node);
        const isMulti = selected.length > 1;

        ui.tiles.innerHTML = "";

        if (isMulti) {
            ui.image.style.display = "none";
            ui.placeholderViewport.style.display = "none";
            ui.emptyLabel.style.display = "none";
            ui.tiles.style.display = "grid";

            selected.forEach((promptName) => {
                const entry = getNodeEntry(node, category, promptName);
                const tile = document.createElement("div");
                tile.style.cssText = `
                    border: 1px solid ${PMA_THEME.accentBorder};
                    background: #1a1a1a;
                    background-size: cover;
                    background-position: center;
                `;
                if (entry?.thumbnail) {
                    tile.style.backgroundImage = `url(${entry.thumbnail})`;
                } else {
                    tile.style.backgroundImage = `url(${DEFAULT_THUMBNAIL})`;
                }
                tile.title = promptName;
                ui.tiles.appendChild(tile);
            });
            return;
        }

        ui.tiles.style.display = "none";

        const entry = name ? getNodeEntry(node, category, name) : null;
        if (entry?.thumbnail) {
            ui.image.src = entry.thumbnail;
            ui.image.style.objectFit = "contain";
            ui.image.style.display = "block";
            ui.placeholderViewport.style.display = "none";
            ui.emptyLabel.style.display = "none";
        } else if (name) {
            ui.image.removeAttribute("src");
            ui.image.style.display = "none";
            ui.placeholderViewport.style.display = "flex";
            ui.emptyLabel.style.display = "none";
        } else {
            ui.image.removeAttribute("src");
            ui.image.style.display = "none";
            ui.placeholderViewport.style.display = "flex";
            ui.emptyLabel.style.display = "none";
        }
    };

    node.updateComposerPreview();
    if (typeof node.updateComposerRootLayout === "function") {
        node.updateComposerRootLayout();
    }
}

function buildComposerPromptEditor(node) {
    if (node._composerPromptEditorBuilt) return;
    node._composerPromptEditorBuilt = true;

    const textWidget = node.widgets.find((w) => w.name === "text");
    const usePromptInputWidget = node.widgets.find((w) => w.name === "use_prompt_input");
    if (!textWidget) return;

    const container = document.createElement("div");
    container.style.cssText = "display: flex; width: 100%; height: 100%; box-sizing: border-box;";

    const textArea = document.createElement("textarea");
    textArea.style.cssText = `
        width: 100%;
        height: 100%;
        min-height: 110px;
        resize: none;
        border-radius: 6px;
        border: 1px solid ${PMA_THEME.inputBorder};
        background: ${PMA_THEME.inputBg};
        color: ${PMA_THEME.textPrimary};
        padding: 10px;
        font-size: 13px;
        box-sizing: border-box;
        font-family: inherit;
    `;
    textArea.placeholder = "Prompt text";
    textArea.value = String(textWidget.value || "");

    textArea.addEventListener("input", () => {
        textWidget.value = textArea.value;
        app.graph.setDirtyCanvas(true, true);
    });

    container.appendChild(textArea);

    if (node._composerRoot?.promptSection) {
        node._composerRoot.promptSection.innerHTML = "";
        node._composerRoot.promptSection.appendChild(container);
    } else {
        const widget = node.addDOMWidget("composer_prompt_editor", "div", container, { hideOnZoom: false });
        widget.computeSize = (width) => [width, Math.max(116, Number(node._composerPromptEditorHeight) || 136)];
    }

    node._composerPromptEditor = { container, textArea, usePromptInputWidget };
    node.updateComposerPromptEditor = () => {
        const ui = node._composerPromptEditor;
        if (!ui) return;

        if (node._composerIsMultiSelection === true) {
            ui.textArea.value = "";
            ui.textArea.placeholder = "Multiple prompts selected";
            ui.textArea.readOnly = true;
            ui.textArea.style.opacity = "0.45";
            ui.textArea.style.pointerEvents = "none";
            return;
        }

        const incoming = String(textWidget.value || "");
        if (ui.textArea.value !== incoming) {
            ui.textArea.value = incoming;
        }
        ui.textArea.placeholder = "Prompt text";
        const isInputDriven = ui.usePromptInputWidget?.value === true;
        ui.textArea.readOnly = isInputDriven;
        ui.textArea.style.opacity = isInputDriven ? "0.7" : "1";
        ui.textArea.style.pointerEvents = "auto";
    };

    node.updateComposerPromptEditor();
}

function buildComposerSelectorBar(node) {
    if (node._composerSelectorBuilt) return;
    node._composerSelectorBuilt = true;

    const categoryWidget = node.widgets.find((w) => w.name === "category");
    const nameWidget = node.widgets.find((w) => w.name === "name");
    const textWidget = node.widgets.find((w) => w.name === "text");
    if (!categoryWidget || !nameWidget) return;

    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        align-items: center;
        gap: 0;
        background: #1a1a1a;
        border-radius: 4px;
        overflow: visible;
        height: 26px;
        margin: 0;
        position: relative;
    `;

    const leftArrow = document.createElement("button");
    leftArrow.textContent = "◀";
    leftArrow.style.cssText = `
        background: #2a2a2a;
        border: none;
        color: #888;
        padding: 0 10px;
        height: 100%;
        cursor: pointer;
        font-size: 10px;
        transition: all 0.15s ease;
    `;
    leftArrow.onmouseover = () => { leftArrow.style.background = "#3a3a3a"; leftArrow.style.color = "#fff"; };
    leftArrow.onmouseout = () => { leftArrow.style.background = "#2a2a2a"; leftArrow.style.color = "#888"; };

    const nameDisplay = document.createElement("div");
    nameDisplay.style.cssText = `
        flex: 1;
        text-align: center;
        color: #dbeafe;
        font-size: 13px;
        padding: 0 10px;
        cursor: pointer;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        background: rgba(56, 130, 246, 0.22);
        border: 1px solid rgba(56, 130, 246, 0.85);
        border-radius: 4px;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s ease;
        box-sizing: border-box;
    `;
    nameDisplay.onmouseover = () => { nameDisplay.style.background = "rgba(56, 130, 246, 0.32)"; };
    nameDisplay.onmouseout = () => { nameDisplay.style.background = "rgba(56, 130, 246, 0.22)"; };

    const rightArrow = document.createElement("button");
    rightArrow.textContent = "▶";
    rightArrow.style.cssText = `
        background: #2a2a2a;
        border: none;
        color: #888;
        padding: 0 10px;
        height: 100%;
        cursor: pointer;
        font-size: 10px;
        transition: all 0.15s ease;
    `;
    rightArrow.onmouseover = () => { rightArrow.style.background = "#3a3a3a"; rightArrow.style.color = "#fff"; };
    rightArrow.onmouseout = () => { rightArrow.style.background = "#2a2a2a"; rightArrow.style.color = "#888"; };

    container.appendChild(leftArrow);
    container.appendChild(nameDisplay);
    container.appendChild(rightArrow);

    const getAllFlat = () => {
        const list = [];
        for (const cat of getNodeCategories(node)) {
            for (const name of getNodeNames(node, cat)) {
                list.push({ category: cat, prompt: name });
            }
        }
        return list;
    };

    const getCurrentIndex = (list) => {
        return list.findIndex((p) => p.category === categoryWidget.value && p.prompt === nameWidget.value);
    };

    const navigateTo = async (item, skipCheck = false) => {
        const sameTarget = item.category === categoryWidget.value && item.prompt === nameWidget.value;
        if (sameTarget) {
            return true;
        }

        if (!skipCheck && shouldWarnComposerUnsavedChanges() && hasComposerUnsavedChanges(node)) {
            const confirmed = await showConfirm(
                "Unsaved Changes",
                "You have unsaved changes to the current prompt. Discard them and switch?",
                "Discard & Switch",
                "#f80"
            );
            if (!confirmed) return false;
        }

        node.isNewUnsavedPrompt = false;
        node.newPromptCategory = null;
        node.newPromptName = null;

        const categoryChanged = item.category !== categoryWidget.value;
        if (categoryChanged) {
            categoryWidget.value = item.category;
            if (typeof categoryWidget.callback === "function") {
                await categoryWidget.callback(item.category);
            }
        }

        nameWidget.value = item.prompt;
        if (typeof nameWidget.callback === "function") {
            await nameWidget.callback(item.prompt);
        }
        setSelectedPrompts(node, [item.prompt]);

        const usePromptInputWidget = node.widgets.find((w) => w.name === "use_prompt_input");
        if (textWidget && usePromptInputWidget?.value !== true) {
            const entry = getNodeEntry(node, item.category, item.prompt);
            textWidget.value = entry?.prompt || "";
        }

        updateComposerLastSavedState(node);
        refreshComposerPromptInputGhosting(node);
        refreshComposerMultiUiState(node);
        updateDisplay();
        app.graph.setDirtyCanvas(true, true);
        return true;
    };

    leftArrow.onclick = async (e) => {
        e.stopPropagation();
        const list = getAllFlat();
        if (list.length === 0) return;
        const idx = getCurrentIndex(list);
        const newIdx = idx <= 0 ? list.length - 1 : idx - 1;
        await navigateTo(list[newIdx]);
    };

    rightArrow.onclick = async (e) => {
        e.stopPropagation();
        const list = getAllFlat();
        if (list.length === 0) return;
        const idx = getCurrentIndex(list);
        const newIdx = idx >= list.length - 1 ? 0 : idx + 1;
        await navigateTo(list[newIdx]);
    };

    const openPromptBrowserPicker = async (e) => {
        if (e?.stopPropagation) {
            e.stopPropagation();
        }
        if (shouldWarnComposerUnsavedChanges() && hasComposerUnsavedChanges(node)) {
            const confirmed = await showConfirm(
                "Unsaved Changes",
                "You have unsaved changes to the current prompt. Discard them and edit?",
                "Discard & Edit",
                "#f80"
            );
            if (!confirmed) return;
        }

        const source = getSourceValue(node);
        const endpointPrefix = getEndpointPrefixForSource(source);
        const preferenceScope = getPreferenceScopeForSource(source);
        const selectedPrompts = getSelectedPrompts(node);

        const selection = await showThumbnailBrowser(node, categoryWidget.value, nameWidget.value, {
            title: "Select Prompt Browser Prompts",
            multiSelect: true,
            startInMultiSelect: false,
            clearSelectionOnCategorySwitch: true,
            selectedPrompts,
            promptOnly: true,
            editMode: true,
            endpointPrefix,
            loadPromptsFn: () => loadActivePrompts(node),
            preferenceScope,
        });

        if (selection?.prompt) {
            categoryWidget.value = selection.category;
            if (typeof categoryWidget.callback === "function") {
                await categoryWidget.callback(selection.category);
            }
            nameWidget.value = selection.prompt;
            if (typeof nameWidget.callback === "function") {
                await nameWidget.callback(selection.prompt);
            }
            const multiNames = Array.isArray(selection.prompts) && selection.prompts.length > 0
                ? selection.prompts
                : [selection.prompt];
            setSelectedPrompts(node, multiNames);

            const entry = getNodeEntry(node, selection.category, selection.prompt);
            if (textWidget && entry && multiNames.length <= 1) {
                textWidget.value = entry.prompt || "";
            }
            updateComposerLastSavedState(node);
            refreshComposerMultiUiState(node);
            updateDisplay();
            app.graph.setDirtyCanvas(true, true);
        }
    };

    nameDisplay.onclick = openPromptBrowserPicker;
    node.openComposerPromptBrowser = openPromptBrowserPicker;

    const updateDisplay = () => {
        const category = categoryWidget.value || "";
        const prompt = nameWidget.value || "new prompt";
        const selected = getSelectedPrompts(node);
        if (selected.length > 1) {
            nameDisplay.textContent = category ? `${category} : (Multi ${selected.length})` : `(Multi ${selected.length})`;
            nameDisplay.title = `Random pick from ${selected.length} selected prompts`;
            return;
        }
        nameDisplay.textContent = `${category} : ${prompt}`;
        nameDisplay.title = `${category} : ${prompt}`;
    };

    updateDisplay();

    if (node._composerRoot?.nameSection) {
        node._composerRoot.nameSection.innerHTML = "";
        node._composerRoot.nameSection.appendChild(container);
    } else {
        const widget = node.addDOMWidget("composer_selector", "div", container);
        widget.computeSize = function(width) {
            return [width, 24];
        };
    }
    node._composerSelectorContainer = container;
    node.updateComposerSelectorDisplay = updateDisplay;
}

function buildComposerButtonBar(node) {
    if (node._composerButtonBarBuilt) return;
    node._composerButtonBarBuilt = true;

    const categoryWidget = node.widgets.find((w) => w.name === "category");
    const nameWidget = node.widgets.find((w) => w.name === "name");
    const textWidget = node.widgets.find((w) => w.name === "text");
    if (!categoryWidget || !nameWidget || !textWidget) return;

    const buttonContainer = document.createElement("div");
    buttonContainer.style.cssText = `
        display: flex;
        gap: 8px;
        padding: 0;
        align-items: center;
        justify-content: space-between;
        width: 100%;
        height: 100%;
        box-sizing: border-box;
    `;

    const saveBtn = createComposerButton("Save Prompt", async () => {
        if (saveBtn.disabled) return;
        const currentCategory = String(categoryWidget.value || "").trim();
        const currentName = String(nameWidget.value || "").trim();
        const source = getSourceValue(node);
        const endpointPrefix = getEndpointPrefixForSource(source);
        const preferenceScope = getPreferenceScopeForSource(source);

        const selection = await showThumbnailBrowser(node, currentCategory, currentName, {
            title: "Save Prompt Composer Prompt",
            endpointPrefix,
            promptOnly: true,
            mode: "save",
            initialName: currentName,
            saveButtonText: "Save",
            onSave: async (payload) => {
                // prompt_browser.js calls onSave with an object: { category, name, overwrite }
                const category = String(payload?.category || "").trim();
                const name = String(payload?.name || "").trim();
                const text = String(textWidget.value || "").trim();
                const existing = getNodeEntry(node, category, name);
                const selectedEntry = getNodeEntry(node, currentCategory, currentName);
                const thumbnailToSave = existing?.thumbnail || selectedEntry?.thumbnail || null;
                const result = await saveSourcePrompt(node, source, category, name, text, thumbnailToSave);
                if (result?.success) {
                    categoryWidget.value = category;
                    if (typeof categoryWidget.callback === "function") {
                        await categoryWidget.callback(category);
                    }
                    nameWidget.value = name;
                    if (typeof nameWidget.callback === "function") {
                        await nameWidget.callback(name);
                    }
                    setSelectedPrompts(node, name ? [name] : []);
                    updateComposerLastSavedState(node);
                    if (typeof node.updateComposerSelectorDisplay === "function") {
                        node.updateComposerSelectorDisplay();
                    }
                    refreshComposerMultiUiState(node);
                    app.graph.setDirtyCanvas(true, true);
                }
                return result;
            },
            loadPromptsFn: () => loadActivePrompts(node),
            preferenceScope,
        });

        if (selection?.prompt) {
            categoryWidget.value = selection.category;
            if (typeof categoryWidget.callback === "function") {
                await categoryWidget.callback(selection.category);
            }
            nameWidget.value = selection.prompt;
            if (typeof nameWidget.callback === "function") {
                await nameWidget.callback(selection.prompt);
            }
            setSelectedPrompts(node, selection.prompt ? [selection.prompt] : []);
            const entry = getNodeEntry(node, selection.category, selection.prompt);
            if (textWidget && entry) {
                textWidget.value = entry.prompt || "";
            }
            updateComposerLastSavedState(node);
            if (typeof node.updateComposerSelectorDisplay === "function") {
                node.updateComposerSelectorDisplay();
            }
            refreshComposerMultiUiState(node);
            app.graph.setDirtyCanvas(true, true);
        }
    });

    const newBtn = createComposerButton("New Prompt", async () => {
        if (shouldWarnComposerUnsavedChanges() && hasComposerUnsavedChanges(node)) {
            const confirmed = await showConfirm(
                "Unsaved Changes",
                "You have unsaved changes to the current prompt. Discard them and start fresh?",
                "Discard & Continue",
                "#f80"
            );
            if (!confirmed) return;
        }

        const currentCategory = categoryWidget.value;
        nameWidget.value = "";
        textWidget.value = "";
        setSelectedPrompts(node, []);

        node.isNewUnsavedPrompt = true;
        node.newPromptCategory = currentCategory;
        node.newPromptName = null;
        node.composerLastSavedState = null;

        if (typeof node.updateComposerSelectorDisplay === "function") {
            node.updateComposerSelectorDisplay();
        }
        refreshComposerMultiUiState(node);
        app.graph.setDirtyCanvas(true, true);
    });

    const moreBtn = createComposerDropdownButton("More ▼", () => {
        const items = [
            {
                label: "Delete Prompt",
                action: async () => {
                    const category = categoryWidget.value;
                    const name = nameWidget.value;
                    if (!name) {
                        await showInfo("Error", "No prompt selected to delete.");
                        return;
                    }
                    const confirmed = await showConfirm(
                        "Delete Prompt",
                        `Are you sure you want to delete "${name}" from "${category}"? This cannot be undone.`,
                        "Delete",
                        "#c00"
                    );
                    if (confirmed) {
                        await deleteSourcePrompt(node, getSourceValue(node), category, name);
                        nameWidget.value = "";
                        textWidget.value = "";
                        setSelectedPrompts(node, []);
                        if (typeof node.updateComposerSelectorDisplay === "function") {
                            node.updateComposerSelectorDisplay();
                        }
                        refreshComposerMultiUiState(node);
                        app.graph.setDirtyCanvas(true, true);
                    }
                }
            },
            { divider: true },
            {
                label: "Save JSON",
                action: () => savePromptBrowserJSON(node),
            },
            {
                label: "Open JSON",
                action: () => openPromptBrowserJSON(node),
            },
            {
                label: "Merge JSON",
                action: () => mergePromptBrowserJSON(node),
            },
        ];
        if (getSourceValue(node) === SOURCE_SYSTEM_PROMPTS) {
            items.push({ divider: true });
            items.push({
                label: "Re-Import Default System Prompts",
                action: () => reimportBasicSystemPrompts(node),
            });
        }
        return items;
    });

    buttonContainer.appendChild(saveBtn);
    buttonContainer.appendChild(newBtn);
    buttonContainer.appendChild(moreBtn);

    if (node._composerRoot?.buttonsSection) {
        node._composerRoot.buttonsSection.innerHTML = "";
        node._composerRoot.buttonsSection.appendChild(buttonContainer);
    } else {
        const widget = node.addDOMWidget("composer_buttons", "div", buttonContainer);
        widget.computeSize = function(width) {
            return [width, 40];
        };
    }
    node._composerButtonBarContainer = buttonContainer;
    node._composerSaveButton = saveBtn;
    node.refreshComposerMultiUiState = () => refreshComposerMultiUiState(node);
    refreshComposerMultiUiState(node);
}

function hasComposerUnsavedChanges(node) {
    const categoryWidget = node.widgets.find((w) => w.name === "category");
    const nameWidget = node.widgets.find((w) => w.name === "name");
    const textWidget = node.widgets.find((w) => w.name === "text");
    const usePromptInputWidget = node.widgets.find((w) => w.name === "use_prompt_input");
    if (!categoryWidget || !nameWidget || !textWidget) return false;

    // Mirror Prompt Manager Advanced: input-driven text is read-only and not treated as dirty edits.
    if (usePromptInputWidget?.value === true) {
        return false;
    }

    // Multi-select cannot directly edit a specific prompt text in this node UI.
    if (getSelectedPrompts(node).length > 1) {
        return false;
    }

    const category = categoryWidget.value || "";
    const name = nameWidget.value || "";
    const text = textWidget.value || "";

    if (node.isNewUnsavedPrompt) {
        return text.trim().length > 0 || name.trim().length > 0;
    }

    if (!node.composerLastSavedState) {
        const entry = getNodeEntry(node, category, name);
        node.composerLastSavedState = entry ? { category, name, text: entry.prompt || "" } : null;
    }

    if (!node.composerLastSavedState) return false;
    return (
        node.composerLastSavedState.category !== category ||
        node.composerLastSavedState.name !== name ||
        node.composerLastSavedState.text !== text
    );
}

function updateComposerLastSavedState(node) {
    const categoryWidget = node.widgets.find((w) => w.name === "category");
    const nameWidget = node.widgets.find((w) => w.name === "name");
    const textWidget = node.widgets.find((w) => w.name === "text");
    if (!categoryWidget || !nameWidget || !textWidget) return;
    node.composerLastSavedState = {
        category: categoryWidget.value || "",
        name: nameWidget.value || "",
        text: textWidget.value || "",
    };
}

function getSourceExportFilename(node) {
    const source = getSourceValue(node);
    if (source === SOURCE_PROMPT) return "prompt_manager_data.json";
    if (source === SOURCE_SYSTEM_PROMPTS) return "prompt_generator_data.json";
    return "prompt_composer_data.json";
}

function getSourceImportEndpointPrefix(node) {
    return getEndpointPrefixForSource(getSourceValue(node));
}

function getSourceReplaceEndpoint(node) {
    const endpointPrefix = getSourceImportEndpointPrefix(node);
    if (endpointPrefix === COMPOSER_ENDPOINT_PREFIX) {
        return `${endpointPrefix}/replace-prompts`;
    }
    return `${endpointPrefix}/import-prompts`;
}

function analyzePromptBrowserImportConflicts(node, importedData) {
    const conflicts = {
        duplicatePrompts: [],
        duplicateCategorySettings: [],
    };
    const source = getSourceValue(node);

    for (const [category, entries] of Object.entries(importedData || {})) {
        if (category === "__meta__" || !entries || typeof entries !== "object") continue;
        const existingCategory = findExistingCategoryName(node, category);
        if (existingCategory && entries.__meta__ && typeof entries.__meta__ === "object") {
            conflicts.duplicateCategorySettings.push({ category, existingCategory });
        }

        for (const [name] of Object.entries(getCategoryPromptEntriesForSource(entries, source))) {
            const existingPrompt = findExistingPromptName(node, category, name);
            if (existingPrompt) {
                conflicts.duplicatePrompts.push({
                    category,
                    existingCategory: existingCategory || category,
                    name,
                    existingPrompt,
                });
            }
        }
    }

    return conflicts;
}

async function loadPromptLibraryFile(endpointPrefix, filePath) {
    const response = await api.fetchApi(`${endpointPrefix}/load-prompts-file`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: filePath }),
    });
    const result = await response.json();
    if (!response.ok || !result?.success) {
        throw new Error(result?.error || "Failed to load JSON file.");
    }
    return result.data;
}

async function exportPromptLibraryFile(endpointPrefix, filePath, data) {
    const response = await api.fetchApi(`${endpointPrefix}/export-prompts-file`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: filePath, data }),
    });
    const result = await response.json();
    if (!response.ok || !result?.success) {
        throw new Error(result?.error || "Failed to save JSON file.");
    }
    return result;
}

async function refreshPromptBrowserAfterLibraryChange(node, options = {}) {
    const clearSelection = options.clearSelection === true;
    await loadActivePrompts(node);

    if (clearSelection) {
        const categoryWidget = node.widgets?.find((w) => w.name === "category");
        const nameWidget = node.widgets?.find((w) => w.name === "name");
        const textWidget = node.widgets?.find((w) => w.name === "text");
        if (categoryWidget) categoryWidget.value = "";
        if (nameWidget) nameWidget.value = "";
        if (textWidget) textWidget.value = "";
        setSelectedPrompts(node, []);
    }

    if (typeof node.updateComposerSelectorDisplay === "function") {
        node.updateComposerSelectorDisplay();
    }
    if (typeof node.updateComposerPromptEditor === "function") {
        node.updateComposerPromptEditor();
    }
    if (typeof node.updateComposerPreview === "function") {
        node.updateComposerPreview();
    }
    updateComposerLastSavedState(node);
    refreshComposerPromptInputGhosting(node);
    if (typeof node.refreshComposerMultiUiState === "function") {
        node.refreshComposerMultiUiState();
    }
    app.graph.setDirtyCanvas(true, true);
}

async function mergePromptBrowserLibraryData(node, libraryData) {
    const source = getSourceValue(node);
    const endpointPrefix = getSourceImportEndpointPrefix(node);
    const conflicts = source === SOURCE_COMPOSE
        ? analyzeComposerImportConflicts(node, libraryData)
        : analyzePromptBrowserImportConflicts(node, libraryData);

    let importMode = "skip_existing";
    if (conflicts.duplicatePrompts.length || conflicts.duplicateCategorySettings.length) {
        importMode = await showComposerImportModeDialog({
            duplicatePromptCount: conflicts.duplicatePrompts.length,
            duplicateCategorySettingsCount: conflicts.duplicateCategorySettings.length,
        });
        if (importMode === "cancel") {
            return false;
        }
    }

    const response = await api.fetchApi(`${endpointPrefix}/import-prompts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: libraryData, mode: importMode }),
    });
    const result = await response.json();
    if (!response.ok || !result?.success) {
        throw new Error(result?.error || "Failed to merge JSON library.");
    }

    await refreshPromptBrowserAfterLibraryChange(node);

    const imported = Number(result?.imported_prompts || 0);
    const skipped = Number(result?.skipped_prompts || 0);
    const createdCategories = Number(result?.created_categories || 0);
    const summaryParts = [];
    if (imported > 0) {
        summaryParts.push(`Imported ${imported} prompt${imported === 1 ? "" : "s"}`);
    }
    if (createdCategories > 0) {
        summaryParts.push(`created ${createdCategories} categor${createdCategories === 1 ? "y" : "ies"}`);
    }
    if (skipped > 0) {
        summaryParts.push(`kept ${skipped} existing prompt${skipped === 1 ? "" : "s"}`);
    }
    await showInfo("Merge Complete", summaryParts.length ? `${summaryParts.join(", ")}.` : "JSON library merged.");
    return true;
}

async function savePromptBrowserJSON(node) {
    try {
        const endpointPrefix = getSourceImportEndpointPrefix(node);
        const savePath = await showPromptLibraryBrowser({
            mode: "save",
            title: "Save Prompt Browser JSON",
            confirmLabel: "Save Here",
            defaultFilename: getSourceExportFilename(node),
        });
        if (!savePath) return;

        const data = node.prompts || node.composerPrompts || {};
        await exportPromptLibraryFile(endpointPrefix, savePath, data);
    } catch (err) {
        console.error("[PromptBrowser] Save JSON error:", err);
        await showInfo("Save Failed", err.message || "Unknown error");
    }
}

async function openPromptBrowserJSON(node) {
    const confirmed = await showConfirm(
        "Open JSON",
        "This will replace the current library for the active source with the selected JSON file. Continue?",
        "Open & Replace",
        PMA_THEME.accent
    );
    if (!confirmed) return;

    try {
        const endpointPrefix = getSourceImportEndpointPrefix(node);
        const filePath = await showPromptLibraryBrowser({
            mode: "open",
            title: "Open Prompt Browser JSON",
            confirmLabel: "Open",
        });
        if (!filePath) return;

        const libraryData = await loadPromptLibraryFile(endpointPrefix, filePath);
        const replaceEndpoint = getSourceReplaceEndpoint(node);
        const body = replaceEndpoint.endsWith("/replace-prompts")
            ? { data: libraryData }
            : { data: libraryData, mode: "replace" };
        const response = await api.fetchApi(replaceEndpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const result = await response.json();
        if (!response.ok || !result?.success) {
            throw new Error(result?.error || "Failed to open JSON library.");
        }

        await refreshPromptBrowserAfterLibraryChange(node, { clearSelection: true });
    } catch (err) {
        console.error("[PromptBrowser] Open JSON error:", err);
        await showInfo("Open Failed", err.message || "Unknown error");
    }
}

async function reimportBasicSystemPrompts(node) {
    const source = getSourceValue(node);
    if (source !== SOURCE_SYSTEM_PROMPTS) {
        await showInfo("Re-Import", "This action is only available for System Prompts.");
        return;
    }
    const confirmed = await showConfirm(
        "Re-Import Default System Prompts",
        "This will add any missing built-in system prompts without overwriting your custom ones. Continue?",
        "Re-Import",
        PMA_THEME.accent
    );
    if (!confirmed) return;
    try {
        const resp = await api.fetchApi("/prompt-generator/reimport-default-prompts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
        });
        const data = await resp.json();
        if (data.success) {
            await showInfo("Re-Import Complete", "Added back missing system prompts.");
            await loadActivePrompts(node);
            if (typeof node.updateComposerSelectorDisplay === "function") {
                node.updateComposerSelectorDisplay();
            }
            if (typeof node.updateComposerPreview === "function") {
                node.updateComposerPreview();
            }
            app.graph.setDirtyCanvas(true, true);
        } else {
            await showInfo("Re-Import Failed", data.error || "Failed to re-import default system prompts.");
        }
    } catch (err) {
        console.error("[PromptBrowser] Error re-importing basic system prompts:", err);
        await showInfo("Re-Import Failed", err.message || "Unknown error");
    }
}

async function mergePromptBrowserJSON(node) {
    try {
        const endpointPrefix = getSourceImportEndpointPrefix(node);
        const filePath = await showPromptLibraryBrowser({
            mode: "open",
            title: "Merge Prompt Browser JSON",
            confirmLabel: "Merge",
        });
        if (!filePath) return;

        const libraryData = await loadPromptLibraryFile(endpointPrefix, filePath);
        await mergePromptBrowserLibraryData(node, libraryData);
    } catch (err) {
        console.error("[PromptBrowser] Merge JSON error:", err);
        await showInfo("Merge Failed", err.message || "Unknown error");
    }
}

app.registerExtension({
    name: "PromptBrowserNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        setupPromptBrowserNode(nodeType, nodeData);
    },
});
