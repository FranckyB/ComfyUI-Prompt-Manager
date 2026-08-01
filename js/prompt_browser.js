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
let showThumbnailContextMenu = () => {};
let _attachThumbnailDropHandlers = () => {};
let ensureThumbnailRenderSelection = async () => null;
let resolveThumbnailFallbackBase = async () => null;
let generateThumbnailForPrompt = async () => null;
let showThumbnailRenderPicker = async () => null;
let saveThumbnailRenderSelection = () => {};
let _thumbnailLeafName = (path) => String(path || "");
let getThumbnailRenderState = null;

let _thumbnailRenderFamily = null;
let _thumbnailRenderModel = null;
let _thumbnailRenderLora1 = null;
let _thumbnailRenderLora2 = null;

function _syncThumbnailRenderState() {
    if (typeof getThumbnailRenderState !== "function") return;
    const state = getThumbnailRenderState() || {};
    _thumbnailRenderFamily = state.family || null;
    _thumbnailRenderModel = state.model || null;
    _thumbnailRenderLora1 = state.lora1 || null;
    _thumbnailRenderLora2 = state.lora2 || null;
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
    if (typeof deps.showThumbnailContextMenu === "function") showThumbnailContextMenu = deps.showThumbnailContextMenu;
    if (typeof deps.attachThumbnailDropHandlers === "function") _attachThumbnailDropHandlers = deps.attachThumbnailDropHandlers;
    if (typeof deps.ensureThumbnailRenderSelection === "function") ensureThumbnailRenderSelection = deps.ensureThumbnailRenderSelection;
    if (typeof deps.resolveThumbnailFallbackBase === "function") resolveThumbnailFallbackBase = deps.resolveThumbnailFallbackBase;
    if (typeof deps.generateThumbnailForPrompt === "function") generateThumbnailForPrompt = deps.generateThumbnailForPrompt;
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

// Browser session state persists during a UI session and resets on restart.
let _sessionHideNSFW = null;
let _sessionViewModeByScope = {
    manager: null,
    mixer: null,
};
let _sessionEnableThumbnailPreview = null;
let _sessionBrowserContentFilter = null;
let _sessionPromptColumnWidthByScope = {
    manager: null,
    mixer: null,
};

function normalizeBrowserPrefScope(scope) {
    return scope === "mixer" ? "mixer" : "manager";
}

function getViewModeStorageKey(scope) {
    return normalizeBrowserPrefScope(scope) === "mixer"
        ? "PromptManager.BrowserViewMode.Mixer"
        : "PromptManager.BrowserViewMode";
}

function getPromptColumnWidthStorageKey(scope) {
    return normalizeBrowserPrefScope(scope) === "mixer"
        ? "PromptManager.ListPromptColumnWidth.Mixer"
        : "PromptManager.ListPromptColumnWidth";
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
    return (saved === "prompt" || saved === "recipe" || saved === "all") ? saved : "all";
}

export function setBrowserContentFilter(value) {
    _sessionBrowserContentFilter = value;
}

export function hasWorkflowDataPayload(rawWorkflowData) {
    return (
        (typeof rawWorkflowData === "string" && rawWorkflowData.trim().length > 0) ||
        (rawWorkflowData && typeof rawWorkflowData === "object" && Object.keys(rawWorkflowData).length > 0)
    );
}

export function hasPromptPresetPayload(promptData) {
    if (!promptData || typeof promptData !== "object") return false;
    const promptText = String(promptData.prompt || "").trim();
    const negativeText = String(promptData.negative_prompt || "").trim();
    const hasLorasA = Array.isArray(promptData.loras_a) && promptData.loras_a.some((lora) => String(lora?.name || "").trim().length > 0);
    const hasLorasB = Array.isArray(promptData.loras_b) && promptData.loras_b.some((lora) => String(lora?.name || "").trim().length > 0);
    const hasLorasC = Array.isArray(promptData.loras_c) && promptData.loras_c.some((lora) => String(lora?.name || "").trim().length > 0);
    const hasLorasD = Array.isArray(promptData.loras_d) && promptData.loras_d.some((lora) => String(lora?.name || "").trim().length > 0);
    return promptText.length > 0 || negativeText.length > 0 || hasLorasA || hasLorasB || hasLorasC || hasLorasD;
}

export function getPromptNamesForCategory(node, category, options = {}) {
    const hideNSFW = options.hideNSFW === true;
    const workflowOnly = options.workflowOnly === true;
    const contentFilter = String(options.contentFilter || "all").toLowerCase();
    const categoryPrompts = node?.prompts?.[category];
    if (!categoryPrompts || typeof categoryPrompts !== "object") return [];

    let promptNames = Object.keys(categoryPrompts)
        .filter((k) => k !== "__meta__")
        .sort((a, b) => a.localeCompare(b));

    if (hideNSFW) {
        promptNames = promptNames.filter((name) => categoryPrompts?.[name]?.nsfw !== true);
    }

    if (workflowOnly || contentFilter !== "all") {
        promptNames = promptNames.filter((name) => {
            const entry = categoryPrompts?.[name];
            const hasRecipeData = hasWorkflowDataPayload(entry?.workflow_data);
            const hasPromptData = hasPromptPresetPayload(entry);

            if (workflowOnly && !(hasRecipeData || hasPromptData)) return false;
            if (contentFilter === "prompt") return hasPromptData;
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

    const categories = Object.keys(node?.prompts || {})
        .filter((c) => c !== "__meta__")
        .sort((a, b) => a.localeCompare(b));

    if (!filterEmptyCategories) return categories;

    return categories.filter((category) => {
        if (keepCategory && category === keepCategory) return true;
        const names = getPromptNamesForCategory(node, category, { hideNSFW, workflowOnly, contentFilter });
        return names.length > 0;
    });
}
async function standaloneShowThumbnailBrowser(node, currentCategory, currentPrompt, options = {}) {
    // Reload prompts to ensure we have the latest data. Callers can pass a custom
    // loader (e.g. Prompt Mixer) so the browser uses the correct JSON store.
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
    const clearSelectionOnCategorySwitch = options?.clearSelectionOnCategorySwitch === true;
    const endpointPrefix = typeof options?.endpointPrefix === "string" ? options.endpointPrefix : "/prompt-manager-advanced";
    const promptOnly = options?.promptOnly === true;
    const browserPrefScope = normalizeBrowserPrefScope(
        options?.preferenceScope || (promptOnly ? "mixer" : "manager")
    );
    let updateSelectButton = () => {};

    const filterAllowedCategories = (categories) => {
        if (!allowedCategorySet || !Array.isArray(categories)) return categories;
        return categories.filter((cat) => allowedCategorySet.has(String(cat).toLowerCase()));
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
        
        let selectedCategory = currentCategory;
        let lastSelectedName = currentPrompt;
        let selectedNames = new Set(multiSelect && Array.isArray(options?.selectedPrompts)
            ? options.selectedPrompts
            : (currentPrompt ? [currentPrompt] : []));
        let multiSelectAnchorName = selectedNames.size > 0 ? Array.from(selectedNames)[0] : "";

        const clearMultiSelection = () => {
            if (!multiSelect || selectedNames.size === 0) return;
            selectedNames.clear();
            multiSelectAnchorName = "";
            updateSelectButton();
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

        const compactBrowser = Boolean(app.ui.settings.getSettingValue("PromptManager.CompactPromptBrowser"));
        const browserLayout = compactBrowser
            ? { width: 654, height: 680, cols: 5, itemWidth: 120, gap: 4, thumbWidth: 100, thumbHeight: 132 }
            : {
                width: 1304, height: 985, iconHeight: 1185,
                cols: 5, itemWidth: 240, gap: 16, thumbWidth: 200, thumbHeight: 264,
                iconCols: 10, iconItemWidth: 120, iconGap: 7, iconThumbWidth: 100, iconThumbHeight: 132,
            };

        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: ${UI.panel};
            border: 1px solid ${UI.panelBorder};
            border-radius: 12px;
            padding: 16px;
            z-index: 10000;
            width: ${browserLayout.width}px;
            max-height: 80vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
            user-select: none;
            -webkit-user-select: none;
        `;

        // Prevent default context menu on dialog (thumbnail cards have their own)
        dialog.addEventListener("contextmenu", (e) => e.preventDefault());

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
            renderContent("");
        };
        const origOninput = searchInput.oninput;
        searchInput.addEventListener("input", () => {
            clearBtn.style.display = searchInput.value ? "" : "none";
        });
        searchWrapper.appendChild(searchInput);
        searchWrapper.appendChild(clearBtn);

        // Prompt/Recipe/All filter button
        let contentFilterState = getBrowserContentFilter(app);
        const contentFilterBtn = document.createElement("button");

        // For prompt-only stores (e.g. Prompt Mixer) the content filter has no meaning.
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
            } else if (contentFilterState === "recipe") {
                contentFilterBtn.textContent = "Type: Recipe";
                contentFilterBtn.style.cssText = btnStyle + `background: rgba(235, 140, 35, 0.18); border-color: rgba(235, 140, 35, 0.8); color: #ffe7c2;`;
                contentFilterBtn.title = "Showing recipe entries only";
            } else {
                contentFilterBtn.textContent = "Type: All";
                contentFilterBtn.style.cssText = btnStyle;
                contentFilterBtn.title = "Showing prompts and recipes";
            }
        };
        updateContentFilterBtn();
        updateNsfwBtn();
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

        controlsBar.appendChild(searchWrapper);
        if (!allowedCategories && !promptOnly) {
            controlsBar.appendChild(contentFilterBtn);
        }
        controlsBar.appendChild(nsfwBtn);
        controlsBar.appendChild(viewModeBtn);

        // Category selector
        const categoryContainer = document.createElement("div");
        categoryContainer.style.cssText = `
            display: flex;
            gap: 6px;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid ${UI.sectionBorder};
            flex-wrap: wrap;
        `;

        let categories = filterAllowedCategories(getVisibleCategories(node, {
            hideNSFW: hideNSFWState,
            workflowOnly,
            contentFilter: contentFilterState,
        }));
        let selectedSaveName = initialSaveName;
        let categoryButtons = [];

        const isCategoryNSFW = (cat) => {
            return node.prompts?.[cat]?.["__meta__"]?.nsfw === true;
        };

        const ensureSelectedCategory = () => {
            const previousCategory = selectedCategory;
            categories = filterAllowedCategories(getVisibleCategories(node, {
                hideNSFW: hideNSFWState,
                workflowOnly,
                contentFilter: contentFilterState,
            }));

            if (!Array.isArray(categories) || categories.length === 0) {
                selectedCategory = "";
                return "";
            }

            if (!selectedCategory || !categories.includes(selectedCategory)) {
                selectedCategory = categories[0];
            }

            if (hideNSFWState && isCategoryNSFW(selectedCategory)) {
                const firstVisible = categories.find(c => !isCategoryNSFW(c));
                if (firstVisible) {
                    selectedCategory = firstVisible;
                }
            }

            if (
                clearSelectionOnCategorySwitch &&
                multiSelect &&
                previousCategory &&
                selectedCategory !== previousCategory
            ) {
                clearMultiSelection();
            }

            return selectedCategory;
        };

        ensureSelectedCategory();

        const updateCategoryButtons = () => {
            ensureSelectedCategory();

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
            });

            // If selected category is now hidden/empty under current filters, switch.
            ensureSelectedCategory();
        };

        // Category context menu for NSFW toggle
        const showCategoryContextMenu = (event, cat) => {
            const existing = document.querySelector('.category-context-menu');
            if (existing) existing.remove();

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
                        body: JSON.stringify({ type: "category", category: cat })
                    });
                    const result = await resp.json();
                    if (result.success) {
                        node.prompts = result.prompts;
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
            renameItem.textContent = "✏️ Rename";
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
                const result = await showRenameCategoryDialog(
                    "Rename Category",
                    "Enter new category name:",
                    [cat],
                    cat
                );
                if (result && result.newCategory && result.newCategory.trim()) {
                    const newCat = result.newCategory.trim();
                    if (newCat === cat) return;
                    try {
                        const resp = await fetch(`${endpointPrefix}/rename-category`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ old_category: cat, new_category: newCat })
                        });
                        const data = await resp.json();
                        if (data.success) {
                            node.prompts = data.prompts;
                            if (selectedCategory === cat) {
                                const previousCategory = selectedCategory;
                                selectedCategory = data.new_category;
                                if (previousCategory !== selectedCategory && clearSelectionOnCategorySwitch && multiSelect) {
                                    clearMultiSelection();
                                }
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

            // Delete Category
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
                if (await showConfirm("Delete Category", `Are you sure you want to delete category "${cat}" and all its prompts?`)) {
                    try {
                        const resp = await fetch(`${endpointPrefix}/delete-category`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ category: cat })
                        });
                        const data = await resp.json();
                        if (data.success) {
                            node.prompts = data.prompts;
                            if (selectedCategory === cat) {
                                const previousCategory = selectedCategory;
                                const cats = Object.keys(node.prompts).filter(c => c !== "__meta__");
                                selectedCategory = cats[0] || "";
                                if (previousCategory !== selectedCategory && clearSelectionOnCategorySwitch && multiSelect) {
                                    clearMultiSelection();
                                }
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

            const runBatchGeneration = async (names, title) => {
                if (names.length === 0) {
                    await showInfo("No Thumbnails to Generate", `All prompts in "${cat}" already have thumbnails.`);
                    return;
                }

                if (!await showConfirm(title, `Generate thumbnails for ${names.length} prompt(s) in "${cat}"?`, "Generate", "#4CAF50")) {
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
                            await generateThumbnailForPrompt(node, cat, pName, () => {
                                renderContent(searchInput.value);
                            }, {
                                queue: true,
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
                const missing = Object.keys(catPrompts).filter(name => {
                    const data = catPrompts[name];
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

                const all = Object.keys(catPrompts).filter(name => {
                    const data = catPrompts[name];
                    return data && typeof data === "object";
                });

                await runBatchGeneration(all, "Re-Generate All Thumbnails");
            };
            menu.appendChild(regenerateItem);

            // Select Thumbnail Family + Model
            const modelItem = document.createElement("div");
            const selectedLorasForLabel = [_thumbnailRenderLora1, _thumbnailRenderLora2].filter(Boolean);
            const modelLabel = (_thumbnailRenderFamily && _thumbnailRenderModel)
                ? `🔧 ${_thumbnailRenderFamily} : ${_thumbnailLeafName(_thumbnailRenderModel)}${selectedLorasForLabel.length ? ` + ${selectedLorasForLabel.length} LoRA` + (selectedLorasForLabel.length > 1 ? "s" : "") : ""}`
                : "🔧 Select Thumbnail Family + Model";
            modelItem.textContent = modelLabel;
            modelItem.style.cssText = `
                padding: 8px 16px;
                color: #ccc;
                cursor: pointer;
                font-size: 13px;
            `;
            modelItem.onmouseover = () => modelItem.style.background = '#3a3a3a';
            modelItem.onmouseout = () => modelItem.style.background = 'transparent';
            modelItem.onclick = async () => {
                menu.remove();
                const picked = await showThumbnailRenderPicker(
                    _thumbnailRenderFamily,
                    _thumbnailRenderModel,
                    _thumbnailRenderLora1,
                    _thumbnailRenderLora2,
                );
                if (picked) {
                    saveThumbnailRenderSelection(picked);
                }
            };
            menu.appendChild(modelItem);

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
            categories = filterAllowedCategories(getVisibleCategories(node, {
                hideNSFW: hideNSFWState,
                workflowOnly,
                contentFilter: contentFilterState,
            }));
            ensureSelectedCategory();
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
                `;

                btn.textContent = cat;

                btn.onclick = () => {
                    const categoryChanged = selectedCategory !== cat;
                    selectedCategory = cat;
                    if (categoryChanged && clearSelectionOnCategorySwitch && multiSelect) {
                        clearMultiSelection();
                    }
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
            `;
            addBtn.onmouseover = () => { addBtn.style.background = '#38414c'; addBtn.style.color = '#fff'; };
            addBtn.onmouseout = () => { addBtn.style.background = '#313843'; addBtn.style.color = '#aaa'; };
            addBtn.onclick = async () => {
                const result = await showNewCategoryDialog();
                if (result && result.name && result.name.trim()) {
                    const categoryName = result.name.trim();
                    const existingCategories = Object.keys(node.prompts || {});
                    const existingCategoryName = existingCategories.find(cat => cat.toLowerCase() === categoryName.toLowerCase());
                    if (existingCategoryName) {
                        await showInfo("Category Exists", `Category already exists as "${existingCategoryName}".`);
                        return;
                    }
                    try {
                        const resp = await fetch(`${endpointPrefix}/save-category`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ category_name: categoryName, nsfw: result.nsfw })
                        });
                        const data = await resp.json();
                        if (data.success) {
                            node.prompts = data.prompts;
                            const categoryChanged = selectedCategory !== categoryName;
                            selectedCategory = categoryName;
                            if (categoryChanged && clearSelectionOnCategorySwitch && multiSelect) {
                                clearMultiSelection();
                            }
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
            categoryContainer.appendChild(addBtn);

            updateCategoryButtons();
        };
        rebuildCategoryList();

        // When the caller locks the browser to a single category, hide the tabs.
        if (allowedCategories && allowedCategories.length === 1) {
            categoryContainer.style.display = "none";
        }

        // Content container - fixed size
        const gridContainer = document.createElement("div");
        gridContainer.className = "thumbnail-grid-container";
        gridContainer.style.cssText = `
            overflow-y: auto;
            height: ${currentViewMode === "icon" ? browserLayout.iconHeight : browserLayout.height}px;
            scrollbar-width: none;
            -ms-overflow-style: none;
        `;
        // Hide scrollbar for webkit browsers
        const style = document.createElement("style");
        style.textContent = `.thumbnail-grid-container::-webkit-scrollbar { display: none; }`;
        document.head.appendChild(style);

        const grid = document.createElement("div");

        // Helper: get filtered prompt names for the selected category
        const getFilteredPrompts = (filter = "") => {
            let promptNames = getPromptNamesForCategory(node, selectedCategory, {
                hideNSFW: hideNSFWState,
                workflowOnly,
                contentFilter: contentFilterState,
            });

            // Filter by search
            if (filter) {
                promptNames = promptNames.filter(name => name.toLowerCase().includes(filter.toLowerCase()));
            }
            return promptNames;
        };

        const applyMultiSelectInteraction = (promptName, filteredPrompts, event) => {
            if (!multiSelect) return "none";

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
                    updateSelectButton();
                    return "rerender";
                }
            }

            if (selectedNames.has(promptName)) {
                selectedNames.delete(promptName);
            } else {
                selectedNames.add(promptName);
            }
            multiSelectAnchorName = promptName;
            updateSelectButton();
            return "single";
        };

        // Shared right-click handler for prompt items (works in both grid and list view)
        const promptContextMenu = (e, promptName) => {
            e.preventDefault();
            showThumbnailContextMenu(e, node, selectedCategory, promptName, () => {
                renderContent(searchInput.value);
            }, endpointPrefix);
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

            if (filteredPrompts.length === 0) {
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

            filteredPrompts.forEach(promptName => {
                const promptData = categoryPrompts[promptName];
                const thumbnail = promptData?.thumbnail || DEFAULT_THUMBNAIL;
                const isSelected = multiSelect ? selectedNames.has(promptName) : promptName === currentPrompt;
                const isNSFW = promptData?.nsfw === true || isCategoryNSFW(selectedCategory);
                const rawWorkflowData = promptData?.workflow_data;
                const hasWorkflowData = !promptOnly && (
                    (typeof rawWorkflowData === "string" && rawWorkflowData.trim().length > 0) ||
                    (rawWorkflowData && typeof rawWorkflowData === "object" && Object.keys(rawWorkflowData).length > 0)
                );
                const hasPromptPayload = !promptOnly && hasPromptPresetPayload(promptData);

                const card = document.createElement("div");
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
                const showWorkflowBadge = !workflowOnly && hasWorkflowData;
                const showPromptBadge = workflowOnly && !hasWorkflowData && hasPromptPayload;
                if (isNSFW || showWorkflowBadge || showPromptBadge) {
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
                    if (mode === "save") {
                        selectedSaveName = promptName;
                        if (saveNameInput) {
                            saveNameInput.value = promptName;
                            saveNameInput.focus();
                            saveNameInput.select();
                        }
                        currentPrompt = promptName;
                        renderContent(searchInput.value);
                        return;
                    }

                    if (multiSelect) {
                        const action = applyMultiSelectInteraction(promptName, filteredPrompts, e);
                        if (action === "rerender") {
                            renderContent(searchInput.value);
                            return;
                        }
                        updateCardSelection(card, promptName);
                        return;
                    }

                    resolve({ category: selectedCategory, prompt: promptName });
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
                } else if (multiSelect) {
                    card.ondblclick = () => {
                        resolve({ category: selectedCategory, prompt: promptName, prompts: [promptName] });
                        cleanup();
                    };
                }

                card.oncontextmenu = (e) => promptContextMenu(e, promptName);

                grid.appendChild(card);
            });
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

            if (filteredPrompts.length === 0) {
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

            filteredPrompts.forEach(promptName => {
                const promptData = categoryPrompts[promptName];
                const thumbnail = promptData?.thumbnail || DEFAULT_THUMBNAIL;
                const isSelected = multiSelect ? selectedNames.has(promptName) : promptName === currentPrompt;
                const isNSFW = promptData?.nsfw === true || isCategoryNSFW(selectedCategory);
                const rawWorkflowData = promptData?.workflow_data;
                const hasWorkflowData = !promptOnly && (
                    (typeof rawWorkflowData === "string" && rawWorkflowData.trim().length > 0) ||
                    (rawWorkflowData && typeof rawWorkflowData === "object" && Object.keys(rawWorkflowData).length > 0)
                );
                const hasPromptPayload = !promptOnly && hasPromptPresetPayload(promptData);

                const card = document.createElement("div");
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
                const showWorkflowBadge = !workflowOnly && hasWorkflowData;
                const showPromptBadge = workflowOnly && !hasWorkflowData && hasPromptPayload;
                if (isNSFW || showWorkflowBadge || showPromptBadge) {
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
                    if (mode === "save") {
                        selectedSaveName = promptName;
                        if (saveNameInput) {
                            saveNameInput.value = promptName;
                            saveNameInput.focus();
                            saveNameInput.select();
                        }
                        currentPrompt = promptName;
                        renderContent(searchInput.value);
                        return;
                    }

                    if (multiSelect) {
                        const action = applyMultiSelectInteraction(promptName, filteredPrompts, e);
                        if (action === "rerender") {
                            renderContent(searchInput.value);
                            return;
                        }
                        updateCardSelection(card, promptName);
                        return;
                    }

                    resolve({ category: selectedCategory, prompt: promptName });
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
                } else if (multiSelect) {
                    card.ondblclick = () => {
                        resolve({ category: selectedCategory, prompt: promptName, prompts: [promptName] });
                        cleanup();
                    };
                }

                card.oncontextmenu = (e) => promptContextMenu(e, promptName);

                grid.appendChild(card);
            });
        };

        // ---- List View ----
        const renderListView = (filter = "") => {
            grid.style.cssText = `
                display: flex;
                flex-direction: column;
                gap: 0;
                padding: 4px 0;
                position: relative;
            `;
            grid.innerHTML = "";

            const categoryPrompts = node.prompts[selectedCategory] || {};
            const filteredPrompts = getFilteredPrompts(filter);

            if (filteredPrompts.length === 0) {
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
                updateSelectButton();
            };

            const listViewportWidth = Math.max(
                520,
                (gridContainer.clientWidth > 0 ? gridContainer.clientWidth : browserLayout.width) - 18
            );
            const defaultPromptColumnWidth = promptOnly
                ? Math.round(listViewportWidth * (browserPrefScope === "mixer" ? 0.68 : 0.5))
                : Math.round(listViewportWidth * 0.43);
            let promptColumnWidth = getPromptColumnWidth(
                browserPrefScope,
                promptOnly,
                defaultPromptColumnWidth
            );
            const buildListColumns = () => {
                const promptCol = clampPromptColumnWidth(promptColumnWidth, promptOnly);
                return promptOnly
                    ? `44px minmax(80px, 1fr) ${promptCol}px`
                    : `44px minmax(150px, 1fr) ${promptCol}px 70px 70px 70px`;
            };

            // Column headers
            const headerRow = document.createElement("div");
            const listColumns = buildListColumns();
            headerRow.style.cssText = `
                display: grid;
                grid-template-columns: ${listColumns};
                gap: 0;
                padding: 4px 8px 6px 8px;
                border-bottom: 1px solid rgba(148,163,184,0.2);
                margin-bottom: 4px;
                align-items: center;
                position: relative;
            `;
            const headers = promptOnly
                ? ["", "Name", "Prompt"]
                : ["", "Name", "Prompt", "LoRAs A", "LoRAs B", "Triggers"];
            headers.forEach((h, i) => {
                const hDiv = document.createElement("div");
                hDiv.textContent = h;
                hDiv.style.cssText = `
                    font-size: 10px;
                    color: #888;
                    font-weight: bold;
                    text-transform: uppercase;
                    text-align: ${i === 0 ? 'center' : (!promptOnly && i >= 3 ? 'center' : 'left')};
                `;

                if (i > 0) {
                    hDiv.style.paddingLeft = "12px";
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
                                ? `44px minmax(80px, 1fr) ${clampedWidth}px`
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
                const promptData = categoryPrompts[promptName];
                const thumbnail = promptData?.thumbnail || DEFAULT_THUMBNAIL;
                const isSelected = multiSelect ? selectedNames.has(promptName) : promptName === currentPrompt;
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
                `;
                row.onmouseenter = () => { if (!selectedNames.has(promptName)) row.style.background = '#2a2a2a'; };
                row.onmouseleave = () => { updateRowSelection(row, promptName); };

                // Thumbnail icon (using div with background-image to avoid browser extension interference)
                const thumbWrap = document.createElement("div");
                thumbWrap.style.cssText = `
                    width: 36px;
                    height: 36px;
                    position: relative;
                    flex-shrink: 0;
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

                const showWorkflowBadge = !workflowOnly && hasWorkflowData;
                const showPromptBadge = workflowOnly && !hasWorkflowData && hasPromptPayload;
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
                    padding: 4px 0 4px 12px;
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
                    padding: 4px 0 4px 12px;
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
                        padding: 4px 0;
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
                    if (mode === "save") {
                        selectedSaveName = promptName;
                        if (saveNameInput) {
                            saveNameInput.value = promptName;
                            saveNameInput.focus();
                            saveNameInput.select();
                        }
                        currentPrompt = promptName;
                        renderContent(searchInput.value);
                        return;
                    }

                    if (multiSelect) {
                        const action = applyMultiSelectInteraction(promptName, filteredPrompts, e);
                        if (action === "rerender") {
                            renderContent(searchInput.value);
                            return;
                        }
                        updateRowSelection(row, promptName);
                        return;
                    }

                    resolve({ category: selectedCategory, prompt: promptName });
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
                } else if (multiSelect) {
                    row.ondblclick = () => {
                        resolve({ category: selectedCategory, prompt: promptName, prompts: [promptName] });
                        cleanup();
                    };
                }

                row.oncontextmenu = (e) => promptContextMenu(e, promptName);

                grid.appendChild(row);
            });

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
            renderContent(searchInput.value);
        };

        viewModeBtn.onclick = () => {
            if (currentViewMode === "grid") {
                currentViewMode = "icon";
            } else if (currentViewMode === "icon") {
                currentViewMode = "list";
            } else {
                currentViewMode = "grid";
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

            const existing = node.prompts?.[category]?.[name];
            let overwrite = false;
            if (existing) {
                overwrite = await showConfirm(
                    "Overwrite Prompt",
                    `Prompt "${name}" already exists in category "${category}". Do you want to replace it?`,
                    "Replace",
                    "#c44",
                    false
                );
                if (!overwrite) {
                    return;
                }
            }

            const saveResult = await onSave({ category, name, overwrite });
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
            renderContent(searchInput.value);
        };

        gridContainer.appendChild(grid);

        // Footer with hint
        const footer = document.createElement("div");
        footer.style.cssText = `
            margin-top: 6px;
            margin-bottom: 0;
            padding-top: 8px;
            border-top: 1px solid ${UI.sectionBorder};
            font-size: 12px;
            line-height: 1.35;
            color: #8a95a6;
            text-align: center;
        `;
        footer.textContent = mode === "save"
            ? "Right-click a prompt or category for more options (thumbnails, NSFW, delete). Single-click fills name; double-click replaces."
            : (multiSelect
                ? "Click prompts to select/deselect. Shift+click selects all between the previously selected prompt and the one you click. Double-click picks one. Click Select to confirm."
                : "Right-click a prompt or category for more options (thumbnails, NSFW, delete)");

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
        } else if (multiSelect) {
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
            clearSelectionBtn.textContent = "Clear Selection";
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
                multiSelectAnchorName = "";
                updateSelectButton();
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
                updateSelectButton();
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

            const selectBtn = document.createElement("button");
            selectBtn.textContent = `Select (${selectedNames.size})`;
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
                selectBtn.textContent = `Select (${selectedNames.size})`;
            };
            selectBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                const result = {
                    category: selectedCategory,
                    prompt: selectedNames.size > 0 ? Array.from(selectedNames)[0] : "",
                    prompts: Array.from(selectedNames),
                };
                resolve(result);
                cleanup();
            });

            selectionTools.appendChild(clearSelectionBtn);
            selectionTools.appendChild(selectAllBtn);
            actionButtons.appendChild(cancelBtn);
            actionButtons.appendChild(selectBtn);

            saveBar.appendChild(selectionTools);
            saveBar.appendChild(actionButtons);
        }

        dialog.appendChild(header);
        dialog.appendChild(controlsBar);
        dialog.appendChild(categoryContainer);
        dialog.appendChild(gridContainer);
        dialog.appendChild(footer);
        if (mode === "save" || multiSelect) {
            dialog.appendChild(saveBar);
        }

        const cleanup = () => {
            clearTimeout(hoverTimer);
            clearTimeout(hideTimer);
            clearTimeout(resetTimer);
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

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
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
    });
}



export async function showThumbnailBrowser(node, currentCategory, currentPrompt, options = {}) {
    _syncThumbnailRenderState();
    return standaloneShowThumbnailBrowser(node, currentCategory, currentPrompt, options);
}

export async function openPromptBrowserForSave(options = {}) {
    return standaloneOpenPromptBrowserForSave(options);
}
