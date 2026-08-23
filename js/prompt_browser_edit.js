import { saveComposerCategorySettings } from "./prompt_composer_common.js";
import { app } from "../../scripts/app.js";

// Placeholder thumbnail. Defined locally (NOT imported from prompt_manager_advanced.js)
// to break the module cycle:
//   prompt_manager_advanced -> prompt_browser -> prompt_browser_edit -> prompt_manager_advanced
// which left prompt_browser's bindings in the temporal dead zone and intermittently
// prevented the PromptBrowser node's JS from registering.
const DEFAULT_THUMBNAIL = new URL("./placeholder.png", import.meta.url).href;

const SETTING_COMPOSER_EXTRA_TYPES = "PromptManager.ComposerExtraPromptTypes";

// Per-prompt category for system prompts (Prompt Generator). Stored on each
// prompt entry as "category"; distinct from the category-level _prompt_type_.
const SYSTEM_PROMPT_CATEGORIES = ["Audio", "Image", "Video", "Other"];

const PROMPT_TYPE_CHOICES = [
    { value: "", label: "None" },
    { value: "animal", label: "Animal" },
    { value: "attire", label: "Attire" },
    { value: "background", label: "Background" },
    { value: "camera", label: "Camera" },
    { value: "character", label: "Character" },
    { value: "color_palette", label: "Color Palette" },
    { value: "composition", label: "Composition" },
    { value: "dialogue", label: "Dialogue" },
    { value: "expression", label: "Expression" },
    { value: "lens_artifact", label: "Lens Artifact" },
    { value: "lighting", label: "Lighting" },
    { value: "mood", label: "Mood" },
    { value: "motion", label: "Motion" },
    { value: "pose", label: "Pose" },
    { value: "scene", label: "Scene" },
    { value: "soundscape", label: "Soundscape" },
    { value: "style", label: "Style" },
    { value: "subject", label: "Subject" },
    { value: "temporal_flow", label: "Temporal Flow" },
    { value: "weather", label: "Weather" },
];

export function getPromptTypeChoices() {
    const choices = [...PROMPT_TYPE_CHOICES];
    const seen = new Set(choices.map((c) => String(c.value || "").trim().toLowerCase()));

    const raw = String(app?.ui?.settings?.getSettingValue?.(SETTING_COMPOSER_EXTRA_TYPES) || "");
    if (!raw.trim()) return choices;

    const extras = raw
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);

    for (const item of extras) {
        const key = item.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        choices.push({ value: item, label: item });
    }

    return choices;
}

const STYLE = {
    panel: "hsl(216 11% 15%)",
    panelBorder: "hsl(216 20% 65% / 0.24)",
    sectionBorder: "hsl(216 20% 65% / 0.20)",
    inputBg: "hsl(220 15% 10%)",
    inputBorder: "hsl(218 10% 41%)",
    buttonBg: "hsl(219 16% 18%)",
    textPrimary: "hsl(0 0% 87%)",
    textMuted: "hsl(0 0% 67%)",
    accent: "hsl(208 73% 57% / 0.9)",
    accentSoft: "hsl(208 73% 57% / 0.16)",
};

function el(tag, styles = {}, text = "") {
    const element = document.createElement(tag);
    if (text) element.textContent = text;
    Object.assign(element.style, styles);
    return element;
}

function createInput(value, placeholder, styles = {}) {
    const input = document.createElement("input");
    input.type = "text";
    input.value = value || "";
    input.placeholder = placeholder || "";
    input.style.cssText = `
        width: 100%;
        padding: 7px 10px;
        background: ${STYLE.inputBg};
        border: 1px solid ${STYLE.inputBorder};
        border-radius: 4px;
        color: #fff;
        font-size: 13px;
        box-sizing: border-box;
        outline: none;
    `;
    Object.assign(input.style, styles);
    input.addEventListener("focus", () => { input.style.borderColor = STYLE.accent; });
    input.addEventListener("blur", () => { input.style.borderColor = STYLE.inputBorder; });
    return input;
}

function createTextarea(value, placeholder, styles = {}) {
    const textarea = document.createElement("textarea");
    textarea.value = value || "";
    textarea.placeholder = placeholder || "";
    textarea.style.cssText = `
        width: 100%;
        min-height: 120px;
        resize: none;
        background: ${STYLE.inputBg};
        border: 1px solid ${STYLE.inputBorder};
        border-radius: 4px;
        color: #fff;
        font-size: 14px;
        padding: 8px;
        box-sizing: border-box;
        outline: none;
        font-family: inherit;
    `;
    Object.assign(textarea.style, styles);
    textarea.addEventListener("focus", () => { textarea.style.borderColor = STYLE.accent; });
    textarea.addEventListener("blur", () => { textarea.style.borderColor = STYLE.inputBorder; });
    return textarea;
}

function createButton(text, onClick, styles = {}) {
    const button = document.createElement("button");
    button.textContent = text;
    button.style.cssText = `
        background: ${STYLE.buttonBg};
        border: 1px solid ${STYLE.inputBorder};
        color: ${STYLE.textPrimary};
        padding: 6px 12px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        white-space: nowrap;
    `;
    Object.assign(button.style, styles);
    button.addEventListener("click", onClick);
    return button;
}

function createSelect(value, options, styles = {}) {
    const select = document.createElement("select");
    select.style.cssText = `
        width: 100%;
        padding: 7px 10px;
        background: ${STYLE.inputBg};
        border: 1px solid ${STYLE.inputBorder};
        border-radius: 4px;
        color: #fff;
        font-size: 13px;
        box-sizing: border-box;
        outline: none;
    `;
    Object.assign(select.style, styles);
    for (const opt of options) {
        const option = document.createElement("option");
        option.value = opt.value;
        option.textContent = opt.label;
        if (opt.value === value) option.selected = true;
        select.appendChild(option);
    }
    select.addEventListener("focus", () => { select.style.borderColor = STYLE.accent; });
    select.addEventListener("blur", () => { select.style.borderColor = STYLE.inputBorder; });
    return select;
}

export function createPromptBrowserEditPanel(options) {
    const {
        node,
        endpointPrefix = COMPOSER_ENDPOINT_PREFIX,
        showInfo,
        showConfirm,
        generateThumbnail,
        savePrompt,
        loadPrompts,
        onChange,
        compact,
    } = options || {};
    const isCompact = Boolean(compact);
    const EDIT_PANEL_WIDTH = isCompact ? 280 : 320;
    const isSystemPromptsSource = String(endpointPrefix) === "/prompt-generator";
    const isPromptManagerSource = String(endpointPrefix) === "/prompt-manager" || String(endpointPrefix) === "/prompt-manager-advanced";

    const _showInfo = typeof showInfo === "function" ? showInfo : async () => {};
    const _showConfirm = typeof showConfirm === "function" ? showConfirm : async () => false;
    const _generateThumbnail = typeof generateThumbnail === "function" ? generateThumbnail : async () => {};
    const _savePrompt = typeof savePrompt === "function" ? savePrompt : async () => ({ success: false });
    const _loadPrompts = typeof loadPrompts === "function" ? loadPrompts : async () => {};
    const _selectPrompt = typeof options?.selectPrompt === "function" ? options.selectPrompt : null;
    const _onChange = typeof onChange === "function" ? onChange : () => {};

    let currentCategory = "";
    let currentPromptName = "";
    let pendingThumbnail = null;
    let loadedThumbnail = null;
    let loadedPromptText = "";

    const root = el("div", {
        display: "flex",
        flexDirection: "column",
        width: `${EDIT_PANEL_WIDTH}px`,
        flexShrink: "0",
        borderLeft: `1px solid ${STYLE.sectionBorder}`,
        background: STYLE.panel,
        padding: "0 0 8px 12px",
        gap: "8px",
        boxSizing: "border-box",
        overflow: "hidden",
        marginTop: "-1px",
    });

    // Category settings collapsible (top)
    const settingsHeader = el("div", {
        display: "flex",
        alignItems: "center",
        gap: "6px",
        padding: "6px 0 8px 0",
        cursor: "pointer",
        userSelect: "none",
        color: STYLE.textPrimary,
        fontSize: "13px",
        fontWeight: "bold",
    }, "Category Settings");

    const settingsArrow = el("span", {
        display: "inline-block",
        width: "12px",
        transition: "color 0.2s ease",
    }, "▶");
    settingsHeader.prepend(settingsArrow);

    const settingsBody = el("div", {
        display: "none",
        flexDirection: "column",
        gap: "10px",
        paddingBottom: "10px",
        borderBottom: `1px solid ${STYLE.sectionBorder}`,
        flexShrink: "0",
    });

    const promptHeader = el("div", {
        display: "flex",
        alignItems: "center",
        gap: "6px",
        padding: "6px 0 8px 0",
        cursor: "pointer",
        userSelect: "none",
        color: STYLE.textPrimary,
        fontSize: "13px",
        fontWeight: "bold",
    }, "Prompt Settings");

    const promptArrow = el("span", {
        display: "inline-block",
        width: "12px",
        transition: "color 0.2s ease",
    }, "▶");
    promptHeader.prepend(promptArrow);

    const promptBody = el("div", {
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        flex: "1",
        minHeight: "0",
        overflowY: "auto",
    });

    const toolsHeader = el("div", {
        display: "flex",
        alignItems: "center",
        gap: "6px",
        padding: "6px 0 8px 0",
        cursor: "pointer",
        userSelect: "none",
        color: STYLE.textPrimary,
        fontSize: "13px",
        fontWeight: "bold",
    }, "Tools");

    const toolsArrow = el("span", {
        display: "inline-block",
        width: "12px",
        transition: "color 0.2s ease",
    }, "▶");
    toolsHeader.prepend(toolsArrow);

    const toolsBody = el("div", {
        display: "none",
        flexDirection: "column",
        gap: "10px",
        paddingBottom: "10px",
        flexShrink: "0",
    });

    let settingsOpen = false;
    let promptOpen = true;
    let toolsOpen = false;

    const syncSectionVisibility = () => {
        settingsBody.style.display = settingsOpen ? "flex" : "none";
        promptBody.style.display = promptOpen ? "flex" : "none";
        toolsBody.style.display = toolsOpen ? "flex" : "none";
        settingsArrow.textContent = settingsOpen ? "▼" : "▶";
        promptArrow.textContent = promptOpen ? "▼" : "▶";
        toolsArrow.textContent = toolsOpen ? "▼" : "▶";
    };

    // Category Settings / Tools are only meaningful for the compose source;
    // hide them for System Prompts and Prompt Manager prompts.
    const showCategorySections = !isSystemPromptsSource && !isPromptManagerSource;
    settingsHeader.style.display = showCategorySections ? "flex" : "none";
    settingsBody.style.display = showCategorySections ? settingsBody.style.display : "none";
    toolsHeader.style.display = showCategorySections ? "flex" : "none";
    toolsBody.style.display = showCategorySections ? toolsBody.style.display : "none";
    if (!showCategorySections) {
        settingsOpen = false;
        toolsOpen = false;
    }

    settingsHeader.addEventListener("click", () => {
        if (settingsOpen) {
            settingsOpen = false;
        } else {
            settingsOpen = true;
            promptOpen = false;
            toolsOpen = false;
        }
        syncSectionVisibility();
    });

    promptHeader.addEventListener("click", () => {
        if (promptOpen) {
            promptOpen = false;
        } else {
            promptOpen = true;
            settingsOpen = false;
            toolsOpen = false;
        }
        syncSectionVisibility();
    });

    toolsHeader.addEventListener("click", () => {
        if (toolsOpen) {
            toolsOpen = false;
        } else {
            toolsOpen = true;
            settingsOpen = false;
            promptOpen = false;
        }
        syncSectionVisibility();
    });

    root.appendChild(settingsHeader);
    root.appendChild(settingsBody);
    root.appendChild(promptHeader);
    root.appendChild(promptBody);
    root.appendChild(toolsHeader);
    root.appendChild(toolsBody);
    syncSectionVisibility();

    const typeLabel = el("label", { color: STYLE.textMuted, fontSize: "12px" }, "Prompt Type");
    settingsBody.appendChild(typeLabel);

    const typeSelect = createSelect("", getPromptTypeChoices());
    settingsBody.appendChild(typeSelect);

    // Auto-save prompt type when changed (no confirmation; prompt-level Save handles the prompt itself).
    typeSelect.addEventListener("change", async () => {
        const category = currentCategory;
        if (!category) return;
        const result = await saveComposerCategorySettings(category, {
            basePrompt: baseTextArea.value,
            promptType: typeSelect.value,
        });
        if (result?.success) {
            node.prompts = result.prompts;
            _onChange();
        }
    });

    const baseLabel = el("label", { color: STYLE.textMuted, fontSize: "12px" }, "Base Prompt (for thumbnails)");
    settingsBody.appendChild(baseLabel);

    const baseTextArea = createTextarea("", "Base prompt used when generating thumbnails for this category");
    baseTextArea.style.minHeight = isCompact ? "140px" : "260px";

    settingsBody.appendChild(baseTextArea);

    const saveSettingsBtn = createButton("Save Category Settings", async () => {
        const category = currentCategory;
        if (!category) {
            await _showInfo("Missing Category", "Please select a category first.");
            return;
        }

        const categoryData = node?.prompts?.[category];
        const promptKeys = (categoryData && typeof categoryData === "object")
            ? Object.keys(categoryData).filter((k) => !String(k || "").startsWith("_"))
            : [];
        const hasExistingPrompts = promptKeys.length > 0;
        const hasExistingCategorySettings = !!(
            categoryData
            && typeof categoryData === "object"
            && (String(categoryData._base_prompt_ || "").trim() || String(categoryData._prompt_type_ || "").trim())
        );

        if (hasExistingPrompts || hasExistingCategorySettings) {
            const confirmed = await _showConfirm(
                "Overwrite Category Settings",
                `Category settings for "${category}" already exist. Replace them?`,
                "Replace",
                "#c44"
            );
            if (!confirmed) return;
        }

        const result = await saveComposerCategorySettings(category, {
            basePrompt: baseTextArea.value,
            promptType: typeSelect.value,
        });
        if (result?.success) {
            node.prompts = result.prompts;
            _onChange();
        } else {
            await _showInfo("Save Failed", result?.error || "Failed to save category settings.");
        }
    }, { background: "#2b6d3a", borderColor: "#4a9158", color: "#fff" });
    settingsBody.appendChild(saveSettingsBtn);

    // Prompt editor section
    const promptNameInput = createInput("", "Prompt name");
    promptBody.appendChild(promptNameInput);

    // Per-prompt category selector is intentionally hidden for system prompts.
    // With the category-bucket model, the category is the bucket itself, so a
    // separate per-prompt category field is no longer needed.
    const promptCategoryWrap = el("div", {
        display: "none",
        flexDirection: "column",
        gap: "4px",
    });
    const promptCategoryLabel = el("label", { color: STYLE.textMuted, fontSize: "12px" }, "Prompt Category");
    const promptCategorySelect = createSelect("Other", SYSTEM_PROMPT_CATEGORIES.map((c) => ({ value: c, label: c })));
    promptCategoryWrap.appendChild(promptCategoryLabel);
    promptCategoryWrap.appendChild(promptCategorySelect);
    promptBody.appendChild(promptCategoryWrap);

    const promptTextArea = createTextarea("", "Prompt text");
    promptTextArea.style.minHeight = isCompact ? "60px" : "120px";
    promptTextArea.style.flex = "1";
    promptBody.appendChild(promptTextArea);

    const editorButtonRow = el("div", {
        display: "flex",
        gap: "8px",
        justifyContent: "flex-end",
    });

    const clearBtn = createButton("Clear", async () => {
        const cleared = await clearPrompt();
        if (cleared) {
            _onChange();
        }
    });

    const saveBtn = createButton("Save", async () => {
        await doSavePrompt(false);
    }, { background: "#2b6d3a", borderColor: "#4a9158", color: "#fff" });

    editorButtonRow.appendChild(clearBtn);
    editorButtonRow.appendChild(saveBtn);
    promptBody.appendChild(editorButtonRow);

    const thumbnailWrap = el("div", {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "0",
        background: STYLE.inputBg,
        border: `1px solid ${STYLE.inputBorder}`,
        borderRadius: "4px",
        width: "100%",
        aspectRatio: "1 / 1",
        flexShrink: "0",
        alignSelf: "stretch",
        boxSizing: "border-box",
        overflow: "hidden",
        position: "relative",
    });

    const thumbnailImg = document.createElement("img");
    thumbnailImg.style.cssText = `
        width: 100%;
        height: 100%;
        object-fit: contain;
        object-position: center;
        border-radius: 4px;
        display: block;
    `;
    thumbnailWrap.appendChild(thumbnailImg);

    promptBody.appendChild(thumbnailWrap);

    const generateBtn = createButton("Generate Thumbnail", async () => {
        const category = currentCategory;
        const name = String(promptNameInput.value || "").trim();
        if (!category || !name) {
            await _showInfo("Missing Prompt", "Please enter a category and prompt name first.");
            return;
        }

        generateBtn.disabled = true;
        generateBtn.textContent = "Generating...";
        try {
            const thumbnail = await _generateThumbnail(category, name);
            pendingThumbnail = thumbnail || null;
            updateThumbnailDisplay(pendingThumbnail);
            _onChange();
        } catch (err) {
            console.error("[PromptBrowserEdit] Thumbnail generation failed:", err);
            await _showInfo("Generation Failed", String(err?.message || err));
        } finally {
            generateBtn.disabled = false;
            generateBtn.textContent = "Generate Thumbnail";
        }
    });
    promptBody.appendChild(generateBtn);

    const bulkPromptBtn = createButton("Bulk Prompt Builder", () => {
        openBulkPromptDialog();
    });
    toolsBody.appendChild(bulkPromptBtn);

    function updateThumbnailDisplay(thumbnail) {
        thumbnailImg.src = thumbnail || DEFAULT_THUMBNAIL;
    }

    async function doSavePrompt(autoFromGeneration) {
        const category = currentCategory;
        const name = String(promptNameInput.value || "").trim();
        const text = String(promptTextArea.value || "").trim();

        if (!category) {
            await _showInfo("Missing Category", "Please select a category first.");
            return { success: false };
        }
        if (!name) {
            await _showInfo("Missing Name", "Please enter a prompt name.");
            promptNameInput.focus();
            return { success: false };
        }

        const categoryData = node?.prompts?.[category];
        let existing = null;
        let existingFound = false;
        if (categoryData && typeof categoryData === "object") {
            if (Object.prototype.hasOwnProperty.call(categoryData, name)) {
                existing = categoryData[name];
                existingFound = true;
            } else {
                const target = name.toLowerCase();
                for (const [entryName, entryValue] of Object.entries(categoryData)) {
                    if (String(entryName || "").startsWith("_")) continue;
                    if (String(entryName).toLowerCase() === target) {
                        existing = entryValue;
                        existingFound = true;
                        break;
                    }
                }
            }
        }
        let overwrite = false;
        if (existingFound) {
            if (autoFromGeneration && (!existing || !existing.thumbnail)) {
                overwrite = true;
            } else {
                overwrite = await _showConfirm(
                    "Overwrite Prompt",
                    `Prompt "${name}" already exists in category "${category}". Replace it?`,
                    "Replace",
                    "#c44"
                );
            }
            if (!overwrite) return { success: false };
        }

        const thumbnail = pendingThumbnail || loadedThumbnail;
        const savePayload = { category, name, text, thumbnail };
        const result = await _savePrompt(savePayload);
        if (result?.success) {
            await _loadPrompts(node);
            currentPromptName = name;
            pendingThumbnail = null;
            const entry = node?.prompts?.[category]?.[name];
            loadedPromptText = entry?.prompt || "";
            updateThumbnailDisplay(entry?.thumbnail || null);
            _onChange();
        } else {
            if (!autoFromGeneration) {
                await _showInfo("Save Failed", result?.error || "Failed to save prompt.");
            }
        }
        return result || { success: false };
    }

    function hasUnsavedChanges() {
        const currentName = String(promptNameInput.value || "").trim();
        const currentText = String(promptTextArea.value || "").trim();

        if (!currentCategory) {
            return false;
        }

        const entry = node?.prompts?.[currentCategory]?.[currentPromptName];
        if (currentPromptName && entry && typeof entry === "object") {
            const originalText = String(entry.prompt || "").trim();
            const originalThumbnail = entry.thumbnail || null;
            if (currentName !== currentPromptName) return true;
            if (currentText !== originalText) return true;
            if (pendingThumbnail !== null && pendingThumbnail !== originalThumbnail) return true;
            return false;
        }

        if (currentName || currentText || pendingThumbnail !== null) {
            return true;
        }
        return false;
    }

    async function confirmDiscardChanges() {
        if (!hasUnsavedChanges()) return true;
        const confirmed = await _showConfirm(
            "Unsaved Changes",
            "You have unsaved changes. Discard them?",
            "Discard",
            "#c44"
        );
        return confirmed;
    }

    async function loadPrompt(category, promptName) {
        const canProceed = await confirmDiscardChanges();
        if (!canProceed) return false;

        currentCategory = category || "";
        currentPromptName = promptName || "";
        promptNameInput.value = currentPromptName;
        pendingThumbnail = null;

        // Selecting a prompt should always focus Prompt Settings.
        settingsOpen = false;
        promptOpen = true;
        if (!showCategorySections) {
            toolsOpen = false;
        }
        syncSectionVisibility();

        const entry = node?.prompts?.[category]?.[promptName];
        if (entry && typeof entry === "object") {
            promptTextArea.value = entry.prompt || "";
            loadedPromptText = entry.prompt || "";
            loadedThumbnail = entry.thumbnail || null;
            updateThumbnailDisplay(loadedThumbnail);
        } else {
            promptTextArea.value = "";
            loadedPromptText = "";
            loadedThumbnail = null;
            updateThumbnailDisplay(null);
        }

        loadCategorySettings(category);
        return true;
    }

    function loadCategorySettings(category) {
        currentCategory = category || currentCategory || "";
        const catData = node?.prompts?.[category];
        if (catData && typeof catData === "object") {
            const promptType = catData._prompt_type_ || "";
            typeSelect.value = promptType;
            baseTextArea.value = catData._base_prompt_ || "";
        } else {
            typeSelect.value = "";
            baseTextArea.value = "";
        }
    }

    function showCategorySettings() {
        if (!showCategorySections) {
            showPromptSettings();
            return;
        }
        settingsOpen = true;
        promptOpen = false;
        syncSectionVisibility();
    }

    function showPromptSettings() {
        settingsOpen = false;
        promptOpen = true;
        syncSectionVisibility();
    }

    async function clearPrompt() {
        const canProceed = await confirmDiscardChanges();
        if (!canProceed) return false;
        promptNameInput.value = "";
        promptTextArea.value = "";
        loadedPromptText = "";
        pendingThumbnail = null;
        loadedThumbnail = null;
        updateThumbnailDisplay(null);
        currentPromptName = "";
        return true;
    }

    function openBulkPromptDialog() {
        const category = currentCategory;
        if (!category) {
            _showInfo("Missing Category", "Please select a category first.");
            return;
        }

        const overlay = el("div", {
            position: "fixed",
            top: "0",
            left: "0",
            right: "0",
            bottom: "0",
            background: "rgba(0, 0, 0, 0.65)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: "10005",
        });

        const dialog = el("div", {
            background: STYLE.panel,
            border: `1px solid ${STYLE.panelBorder}`,
            borderRadius: "6px",
            padding: "16px",
            width: "480px",
            maxWidth: "90vw",
            maxHeight: "90vh",
            display: "flex",
            flexDirection: "column",
            gap: "12px",
            boxShadow: "0 8px 32px rgba(0, 0, 0, 0.45)",
        });

        const title = el("h3", {
            margin: "0",
            color: STYLE.textPrimary,
            fontSize: "15px",
            fontWeight: "bold",
        }, "Bulk Prompt Builder");

        const subtext = el("p", {
            margin: "0",
            color: STYLE.textMuted,
            fontSize: "12px",
        }, `Quickly create multiple prompts in "${category}".`);

        const textarea = createTextarea("", "Enter one tag per line...");
        textarea.style.minHeight = "auto";
        textarea.style.height = "auto";
        textarea.style.resize = "vertical";
        textarea.rows = 20;

        const buttonRow = el("div", {
            display: "flex",
            gap: "8px",
            justifyContent: "flex-end",
        });

        const cancelBtn = createButton("Cancel", () => {
            if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
        });

        const createBtn = createButton("Create", async () => {
            const raw = String(textarea.value || "");
            const lines = raw.split(/\r?\n/).map((s) => s.trim()).filter((s) => s.length > 0);
            if (lines.length === 0) {
                await _showInfo("No Prompts", "Please enter at least one line.");
                return;
            }

            createBtn.disabled = true;
            createBtn.textContent = "Creating...";
            let created = 0;
            let failed = 0;
            const seen = new Set();

            for (const line of lines) {
                const key = line.toLowerCase();
                if (seen.has(key)) continue;
                seen.add(key);
                try {
                    const result = await _savePrompt({ category, name: line, text: line, thumbnail: null });
                    if (result?.success) {
                        created++;
                    } else {
                        failed++;
                    }
                } catch (err) {
                    console.error("[PromptBrowserEdit] Batch save failed:", err);
                    failed++;
                }
            }

            await _loadPrompts(node);
            _onChange();
            if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
            if (created > 0) {
                const firstName = lines.find((line) => line.length > 0);
                if (firstName) {
                    _selectPrompt?.(category, firstName);
                }
            }
            await _showInfo("Prompts Created", `${created} prompt(s) created in "${category}".${failed ? ` ${failed} failed.` : ""}`);
        }, { background: "#2b6d3a", borderColor: "#4a9158", color: "#fff" });

        buttonRow.appendChild(cancelBtn);
        buttonRow.appendChild(createBtn);

        dialog.appendChild(title);
        dialog.appendChild(subtext);
        dialog.appendChild(textarea);
        dialog.appendChild(buttonRow);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);

        textarea.focus();

        const closeOnEscape = (e) => {
            if (e.key === "Escape") {
                document.removeEventListener("keydown", closeOnEscape);
                if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
            }
        };
        document.addEventListener("keydown", closeOnEscape);

        overlay.addEventListener("click", (e) => {
            if (e.target === overlay) {
                document.removeEventListener("keydown", closeOnEscape);
                if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
            }
        });
    }

    // Initialize the thumbnail area with the placeholder so it never starts empty.
    updateThumbnailDisplay(null);

    return {
        element: root,
        loadPrompt,
        loadCategorySettings,
        showCategorySettings,
        showPromptSettings,
        clearPrompt,
        getCurrentPromptName: () => String(promptNameInput.value || "").trim(),
    };
}
