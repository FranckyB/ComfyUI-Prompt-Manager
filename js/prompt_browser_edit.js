import { saveComposerCategorySettings, saveComposerTypeSettings } from "./prompt_composer_common.js";
import { api } from "../../scripts/api.js";
import { getCategoryPromptEntries, getCategoryPromptEntriesForEndpoint, getCategoryPromptEntryForEndpoint } from "./prompt_store_adapters.js";
import { mediaFileUrl } from "./path_browser.js";

// Placeholder thumbnail. Defined locally (NOT imported from prompt_manager_advanced.js)
// to break the module cycle:
//   prompt_manager_advanced -> prompt_browser -> prompt_browser_edit -> prompt_manager_advanced
// which left prompt_browser's bindings in the temporal dead zone and intermittently
// prevented the PromptBrowser node's JS from registering.
const DEFAULT_THUMBNAIL = new URL("./placeholder.png", import.meta.url).href;
const IMAGE_PREVIEW_EXTS = [".png", ".jpg", ".jpeg", ".webp"];

// Per-prompt category for system prompts (Prompt Generator). Stored on each
// prompt entry as "category"; distinct from the category-level _prompt_type_.
const SYSTEM_PROMPT_CATEGORIES = ["Audio", "Image", "Video", "Other"];
const COMPOSER_CATEGORY_KEY_SEPARATOR = "::";

const COMPOSER_SUBJECT_KIND_CHOICES = [
    { value: "character", label: "Character" },
    { value: "animal", label: "Animal" },
    { value: "environment", label: "Environment" },
    { value: "other", label: "Other" },
];

function normalizeComposerSubjectKind(value, fallback = "other") {
    const normalized = String(value || "").trim().toLowerCase();
    if (COMPOSER_SUBJECT_KIND_CHOICES.some((choice) => choice.value === normalized)) {
        return normalized;
    }
    if (normalized === "person") return "character";
    return fallback;
}

function defaultComposerSubjectKind(subjectType) {
    return String(subjectType || "").trim().toLowerCase() === "new_subject" ? "character" : "other";
}

function addPromptTypeChoice(choices, seen, value, label = null) {
    const normalized = String(value || "").trim();
    if (!normalized) return;
    const key = normalized.toLowerCase();
    const normalizedLabel = String(label || normalized).trim() || normalized;
    if (seen.has(key)) {
        const existing = choices.find((choice) => String(choice?.value || "").trim().toLowerCase() === key);
        if (existing) existing.label = normalizedLabel;
        return;
    }
    seen.add(key);
    choices.push({ value: normalized, label: normalizedLabel });
}

function collectPromptTypesFromData(promptsData) {
    if (!promptsData || typeof promptsData !== "object") return [];
    const discovered = [];
    const seen = new Set();
    for (const categoryData of Object.values(promptsData)) {
        if (!categoryData || typeof categoryData !== "object" || Array.isArray(categoryData)) continue;
        const promptType = String(categoryData._prompt_type_ || "").trim();
        if (promptType) {
            const key = promptType.toLowerCase();
            if (seen.has(key)) continue;
            seen.add(key);
            discovered.push({
                value: promptType,
                label: String(categoryData._type_name_ || promptType).trim() || promptType,
            });
        }
    }
    return discovered;
}

export function getPromptTypeChoices(promptsData = null) {
    const choices = [];
    const seen = new Set();

    for (const item of collectPromptTypesFromData(promptsData)) {
        addPromptTypeChoice(choices, seen, item.value, item.label);
    }

    return choices;
}

function getPromptTypeMetadata(promptsData, promptType, composerLibrary = null) {
    const normalizedPromptType = String(promptType || "").trim();
    if (!normalizedPromptType) {
        return {
            exists: false,
            promptType: "",
            typeFile: "",
            typeName: "",
            subjectType: "subject",
            subjectKind: "other",
            promptPrefix: "",
            basePrompt: "",
        };
    }

    const normalizedKey = normalizedPromptType.toLowerCase();
    const canonicalTypes = composerLibrary && typeof composerLibrary === "object" && !Array.isArray(composerLibrary)
        ? composerLibrary._types_
        : null;
    if (canonicalTypes && typeof canonicalTypes === "object" && !Array.isArray(canonicalTypes)) {
        for (const [typeFile, typeData] of Object.entries(canonicalTypes)) {
            if (!typeData || typeof typeData !== "object" || Array.isArray(typeData)) continue;
            const typeStem = String(typeFile || "").replace(/\.json$/i, "").trim().toLowerCase();
            if (typeStem !== normalizedKey) continue;
            return {
                exists: true,
                promptType: typeStem || normalizedPromptType,
                typeFile: String(typeFile || `${normalizedKey}.json`).trim(),
                typeName: String(typeData.name || normalizedPromptType).trim() || normalizedPromptType,
                subjectType: String(typeData.subject_type || "subject").trim().toLowerCase() || "subject",
                subjectKind: normalizeComposerSubjectKind(
                    typeData.subject_kind,
                    defaultComposerSubjectKind(typeData.subject_type || "subject")
                ),
                promptPrefix: String(typeData.prefix || ""),
                basePrompt: String(typeData.base_prompt || ""),
            };
        }
    }

    const fallbackChoice = getPromptTypeChoices(promptsData)
        .find((choice) => String(choice?.value || "").trim().toLowerCase() === normalizedKey);

    return {
        exists: false,
        promptType: normalizedPromptType,
        typeFile: `${normalizedKey}.json`,
        typeName: String(fallbackChoice?.label || normalizedPromptType).trim() || normalizedPromptType,
        subjectType: "subject",
        subjectKind: defaultComposerSubjectKind("subject"),
        promptPrefix: "",
        basePrompt: "",
    };
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

function getCategoryDisplayName(promptsData, category) {
    const direct = promptsData && typeof promptsData === "object" ? promptsData?.[category] : null;
    const metadataName = String(direct?._category_name_ || "").trim();
    if (metadataName) return metadataName;
    return parseComposerCategoryKey(category).categoryName || String(category || "").trim();
}

function getComposerCategoryMetadata(composerLibrary, category, promptType = "") {
    const parsedCategory = parseComposerCategoryKey(category);
    const normalizedCategory = String(parsedCategory.categoryName || category || "").trim();
    const normalizedTypeFile = String(parsedCategory.typeFile || "").trim().toLowerCase();
    if (!normalizedCategory) {
        return {
            exists: false,
            categoryName: "",
            promptType: "",
            typeFile: "",
            typeName: "",
            subjectType: "subject",
            subjectKind: "other",
            typePromptPrefix: "",
            typeBasePrompt: "",
            categoryPromptPrefix: "",
            categoryBasePrompt: "",
        };
    }

    const canonicalTypes = composerLibrary && typeof composerLibrary === "object" && !Array.isArray(composerLibrary)
        ? composerLibrary._types_
        : null;
    if (!canonicalTypes || typeof canonicalTypes !== "object" || Array.isArray(canonicalTypes)) {
        return {
            exists: false,
            categoryName: normalizedCategory,
            promptType: String(promptType || "").trim().toLowerCase(),
            typeFile: "",
            typeName: "",
            subjectType: "subject",
            subjectKind: "other",
            typePromptPrefix: "",
            typeBasePrompt: "",
            categoryPromptPrefix: "",
            categoryBasePrompt: "",
        };
    }

    const normalizedPromptType = String(promptType || "").trim().toLowerCase();
    let fallbackMatch = null;
    for (const [typeFile, typeData] of Object.entries(canonicalTypes)) {
        if (!typeData || typeof typeData !== "object" || Array.isArray(typeData)) continue;
        if (normalizedTypeFile && String(typeFile || "").trim().toLowerCase() !== normalizedTypeFile) continue;
        const typeStem = String(typeFile || "").replace(/\.json$/i, "").trim().toLowerCase();
        if (normalizedPromptType && typeStem !== normalizedPromptType) continue;
        const categories = typeData.categories;
        if (!categories || typeof categories !== "object" || Array.isArray(categories)) continue;
        for (const [categoryName, categoryData] of Object.entries(categories)) {
            if (String(categoryName || "").trim().toLowerCase() !== normalizedCategory.toLowerCase()) continue;
            const match = {
                exists: true,
                categoryName: String(categoryName || normalizedCategory).trim() || normalizedCategory,
                promptType: typeStem,
                typeFile: String(typeFile || "").trim(),
                typeName: String(typeData.name || typeStem || normalizedCategory).trim() || typeStem || normalizedCategory,
                subjectType: String(typeData.subject_type || "subject").trim().toLowerCase() || "subject",
                subjectKind: normalizeComposerSubjectKind(
                    typeData.subject_kind,
                    defaultComposerSubjectKind(typeData.subject_type || "subject")
                ),
                typePromptPrefix: String(typeData.prefix || ""),
                typeBasePrompt: String(typeData.base_prompt || ""),
                categoryPromptPrefix: String(categoryData?.prefix || ""),
                categoryBasePrompt: String(categoryData?.base_prompt || ""),
            };
            if (normalizedPromptType) return match;
            if (!fallbackMatch) fallbackMatch = match;
        }
    }

    return fallbackMatch || {
        exists: false,
        categoryName: normalizedCategory,
        promptType: normalizedPromptType,
        typeFile: "",
        typeName: "",
        subjectType: "subject",
        subjectKind: "other",
        typePromptPrefix: "",
        typeBasePrompt: "",
        categoryPromptPrefix: "",
        categoryBasePrompt: "",
    };
}

const STYLE = {
    panel: "hsl(216 11% 15%)",
    panelBorder: "hsl(216 20% 65% / 0.24)",
    sectionBorder: "hsl(216 20% 65% / 0.20)",
    inputBg: "hsl(220 15% 10%)",
    inputBorder: "hsl(218 10% 41%)",
    buttonBg: "hsl(219 16% 18%)",
    cardBg: "hsl(219 16% 18%)",
    textPrimary: "hsl(0 0% 87%)",
    textMuted: "hsl(0 0% 67%)",
    accent: "hsl(208 73% 57% / 0.9)",
    accentBorder: "hsl(208 73% 57% / 0.65)",
    accentSoft: "hsl(208 73% 57% / 0.16)",
};

let _composerLoraPickerCache = null;
let _composerLoraPickerPromise = null;
let _composerRefModPickerCache = null;
let _composerRefModPickerPromise = null;

function normalizeBrowserPath(value) {
    return String(value || "").replace(/\\/g, "/").replace(/\/+$/g, "");
}

function stripKnownAssetExtension(value) {
    return String(value || "").replace(/\.(safetensors|ckpt|pt|bin|pth)$/i, "");
}

function previewUrlCandidatesForAbsoluteStem(pathStem) {
    const stem = String(pathStem || "").trim();
    if (!stem) return [];
    return IMAGE_PREVIEW_EXTS.map((ext) => mediaFileUrl(`${stem}${ext}`));
}

function loraValueFromAbsolutePath(absPath, rootPath) {
    const normalizedPath = normalizeBrowserPath(stripKnownAssetExtension(absPath));
    const normalizedRoot = normalizeBrowserPath(rootPath);
    if (normalizedRoot && normalizedPath.toLowerCase().startsWith(`${normalizedRoot.toLowerCase()}/`)) {
        return normalizedPath.substring(normalizedRoot.length + 1);
    }
    const parts = splitFolderAndName(normalizedPath);
    return parts.subfolder ? `${parts.subfolder}/${parts.name}` : parts.name;
}

function createAssetPromptLabel(value) {
    return String(value || "").trim();
}

function setImagePreviewCandidates(img, candidates) {
    const queue = Array.isArray(candidates)
        ? candidates.map((item) => String(item || "").trim()).filter(Boolean)
        : [];
    let index = 0;
    const loadNext = () => {
        if (index >= queue.length) {
            img.onerror = null;
            img.src = DEFAULT_THUMBNAIL;
            return;
        }
        img.src = queue[index++];
    };
    img.onerror = loadNext;
    loadNext();
}

function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(reader.error || new Error("Failed to read blob"));
        reader.readAsDataURL(blob);
    });
}

async function fetchImageAsDataUrl(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch preview (${response.status})`);
    }
    return await blobToDataUrl(await response.blob());
}

async function previewCandidatesToThumbnail(candidates) {
    const queue = Array.isArray(candidates)
        ? candidates.map((item) => String(item || "").trim()).filter(Boolean)
        : [];
    for (const candidate of queue) {
        try {
            const thumbnail = await fetchImageAsDataUrl(candidate);
            if (thumbnail) return thumbnail;
        } catch {
            // Try the next candidate.
        }
    }
    return null;
}

async function fetchLoraBrowserRoot() {
    try {
        const response = await api.fetchApi("/fbnodes/lora-browser/root");
        if (!response?.ok) return "";
        const data = await response.json();
        return data?.ok ? String(data.root || "") : "";
    } catch (err) {
        console.warn("[PromptBrowserEdit] Failed to get LoRA browser root:", err);
        return "";
    }
}

async function listLoraBrowserFolder(path, rootPath) {
    const query = path ? `?path=${encodeURIComponent(path)}` : "";
    const response = await api.fetchApi(`/fbnodes/lora-browser/list${query}`);
    if (!response?.ok) {
        let message = `Failed to browse LoRAs (${response?.status || "request"})`;
        try {
            const payload = await response.json();
            if (payload?.error) message = String(payload.error);
        } catch {
            // ignore
        }
        throw new Error(message);
    }
    const data = await response.json();
    const currentPath = String(data?.current_path || path || rootPath || "");
    const files = Array.isArray(data?.files) ? data.files : [];
    return {
        root: String(rootPath || data?.root || ""),
        currentPath,
        parentPath: data?.parent_path ? String(data.parent_path) : null,
        dirs: Array.isArray(data?.dirs) ? data.dirs : [],
        assets: files.map((item) => {
            const absPath = String(item?.path || "").trim();
            const promptName = createAssetPromptLabel(stripKnownAssetExtension(String(item?.name || "")));
            const stem = stripKnownAssetExtension(absPath);
            return {
                id: absPath,
                path: absPath,
                assetValue: loraValueFromAbsolutePath(absPath, rootPath),
                promptName,
                promptText: promptName,
                previewCandidates: previewUrlCandidatesForAbsoluteStem(stem),
                meta: absPath,
            };
        }),
    };
}

async function listRefModBrowserFolder(path) {
    const query = path ? `?path=${encodeURIComponent(path)}` : "";
    const response = await api.fetchApi(`/h3refmods/refmod-browser/list${query}`);
    if (!response?.ok) {
        let message = `Failed to browse RefMods (${response?.status || "request"})`;
        try {
            const payload = await response.json();
            if (payload?.error) message = String(payload.error);
        } catch {
            // ignore
        }
        throw new Error(message);
    }
    const data = await response.json();
    const currentPath = String(data?.current_path || path || "");
    const root = String(data?.root || "");
    const mods = Array.isArray(data?.mods) ? data.mods : [];
    return {
        root,
        currentPath,
        parentPath: data?.parent_path ? String(data.parent_path) : null,
        dirs: Array.isArray(data?.dirs) ? data.dirs : [],
        assets: mods.map((item) => {
            const absPath = String(item?.path || "").trim();
            const promptName = createAssetPromptLabel(stripKnownAssetExtension(String(item?.name || "")));
            const normalizedRoot = normalizeBrowserPath(root);
            const normalizedPath = normalizeBrowserPath(stripKnownAssetExtension(absPath));
            const assetValue = normalizedRoot && normalizedPath.toLowerCase().startsWith(`${normalizedRoot.toLowerCase()}/`)
                ? normalizedPath.substring(normalizedRoot.length + 1)
                : stripKnownAssetExtension(String(item?.name || ""));
            return {
                id: absPath,
                path: absPath,
                assetValue,
                promptName,
                promptText: promptName,
                previewCandidates: item?.preview_path
                    ? [api.apiURL(`/h3refmods/refmod-browser/file?path=${encodeURIComponent(String(item.preview_path))}`)]
                    : [],
                meta: String(item?.description || item?.concept_type || absPath || ""),
            };
        }),
    };
}

function showVisualAssetImportPicker({ title, rootPath, loadFolder, emptyMessage = "No assets found.", createButtonLabel = "Create Prompts" }) {
    return new Promise((resolve) => {
        const selectedAssets = new Map();
        const assetCardEls = new Map();
        let currentListing = null;
        let currentPath = String(rootPath || "");
        let requestToken = 0;
        let selectionAnchorId = "";

        const overlay = el("div", {
            position: "fixed",
            inset: "0",
            background: "rgba(0, 0, 0, 0.75)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: "10010",
        });

        const dialog = el("div", {
            background: STYLE.panel,
            border: `1px solid ${STYLE.panelBorder}`,
            borderRadius: "8px",
            width: "1100px",
            maxWidth: "94vw",
            height: "860px",
            maxHeight: "90vh",
            display: "flex",
            flexDirection: "column",
            gap: "10px",
            padding: "16px",
            boxSizing: "border-box",
            boxShadow: "0 12px 36px rgba(0, 0, 0, 0.4)",
        });

        const titleEl = el("div", { color: STYLE.textPrimary, fontSize: "16px", fontWeight: "600" }, title || "Import Assets");
        const controlsRow = el("div", { display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" });
        const rootBtn = createButton("Root", () => { void loadPath(rootPath); });
        const upBtn = createButton("Up", () => {
            if (currentListing?.parentPath) {
                void loadPath(currentListing.parentPath);
            }
        });
        const pathLabel = el("div", {
            flex: "1",
            minWidth: "240px",
            color: STYLE.textMuted,
            fontSize: "12px",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
        }, "");
        const filterInput = createInput("", "Filter assets...", { maxWidth: "260px" });
        controlsRow.append(rootBtn, upBtn, pathLabel, filterInput);

        const selectionInfo = el("div", { color: STYLE.textMuted, fontSize: "12px" }, "0 selected");
        const content = el("div", {
            display: "flex",
            flexDirection: "column",
            gap: "12px",
            overflowY: "auto",
            flex: "1",
            minHeight: "0",
            paddingRight: "4px",
        });
        const itemsSection = el("div", {
            display: "flex",
            flexDirection: "column",
            gap: "8px",
            flex: "1",
            minHeight: "0",
        });
        const itemsTitle = el("div", {
            color: STYLE.textMuted,
            fontSize: "12px",
            fontWeight: "600",
            letterSpacing: "0.02em",
            textTransform: "uppercase",
        }, "Items");
        const itemsWrap = el("div", {
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
            gap: "12px",
        });
        const emptyState = el("div", {
            color: STYLE.textMuted,
            fontSize: "13px",
            padding: "24px 0",
            textAlign: "center",
        }, emptyMessage);
        itemsSection.append(itemsTitle, itemsWrap, emptyState);
        content.append(itemsSection);

        const buttonRow = el("div", { display: "flex", justifyContent: "flex-end", gap: "8px" });
        const selectionTools = el("div", {
            display: "flex",
            alignItems: "center",
            gap: "8px",
            marginRight: "auto",
        });
        const selectToggleBtn = createButton("Select All", () => {
            const visibleAssets = getVisibleAssets();
            const allVisibleSelected = visibleAssets.length > 0 && visibleAssets.every((asset) => selectedAssets.has(asset.id));
            if (allVisibleSelected) {
                for (const asset of visibleAssets) {
                    selectedAssets.delete(asset.id);
                }
                selectionAnchorId = "";
            } else {
                for (const asset of visibleAssets) {
                    selectedAssets.set(asset.id, asset);
                }
                if (visibleAssets.length > 0) {
                    selectionAnchorId = visibleAssets[0].id;
                }
            }
            updateSelectionInfo();
            syncAssetCardSelectionStates();
        });
        selectionTools.appendChild(selectToggleBtn);
        const cancelBtn = createButton("Cancel", () => close(null));
        const createBtn = createButton(createButtonLabel, () => close([...selectedAssets.values()]), {
            background: "#2b6d3a",
            borderColor: "#4a9158",
            color: "#fff",
        });
        buttonRow.append(selectionTools, cancelBtn, createBtn);

        dialog.append(titleEl, controlsRow, selectionInfo, content, buttonRow);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);

        const close = (value) => {
            document.removeEventListener("keydown", onKeyDown, true);
            if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
            resolve(value);
        };

        const updateSelectionInfo = () => {
            const count = selectedAssets.size;
            selectionInfo.textContent = `${count} selected`;
            createBtn.disabled = count === 0;
            createBtn.style.opacity = count === 0 ? "0.55" : "1";
            createBtn.style.cursor = count === 0 ? "not-allowed" : "pointer";
            const visibleAssets = getVisibleAssets();
            const allVisibleSelected = visibleAssets.length > 0 && visibleAssets.every((asset) => selectedAssets.has(asset.id));
            selectToggleBtn.textContent = allVisibleSelected ? "Select None" : "Select All";
        };

        const matchesFilter = (asset) => {
            const tokens = String(filterInput.value || "").trim().toLowerCase().split(/\s+/).filter(Boolean);
            if (!tokens.length) return true;
            const haystack = `${asset?.promptName || ""} ${asset?.meta || ""} ${asset?.assetValue || ""}`.toLowerCase();
            return tokens.every((token) => haystack.includes(token));
        };

        const getVisibleAssets = () => {
            return (Array.isArray(currentListing?.assets) ? currentListing.assets : []).filter(matchesFilter);
        };

        const updateAssetCardSelectionStyles = (card, selected) => {
            if (!card) return;
            card.style.background = selected ? STYLE.accentSoft : STYLE.cardBg;
            card.style.border = `2px solid ${selected ? STYLE.accentBorder : STYLE.inputBorder}`;
        };

        const syncAssetCardSelectionStates = () => {
            for (const [assetId, card] of assetCardEls.entries()) {
                updateAssetCardSelectionStyles(card, selectedAssets.has(assetId));
            }
        };

        const applyShiftSelection = (targetAsset, visibleAssets) => {
            if (!selectionAnchorId) return false;
            const anchorIndex = visibleAssets.findIndex((asset) => asset.id === selectionAnchorId);
            const targetIndex = visibleAssets.findIndex((asset) => asset.id === targetAsset.id);
            if (anchorIndex < 0 || targetIndex < 0) return false;
            const start = Math.min(anchorIndex, targetIndex);
            const end = Math.max(anchorIndex, targetIndex);
            for (let i = start; i <= end; i++) {
                const asset = visibleAssets[i];
                selectedAssets.set(asset.id, asset);
            }
            return true;
        };

        const render = () => {
            pathLabel.textContent = currentListing?.currentPath || currentPath || rootPath || "";
            itemsWrap.innerHTML = "";
            assetCardEls.clear();
            const dirs = Array.isArray(currentListing?.dirs) ? currentListing.dirs : [];
            const assets = getVisibleAssets();

            for (const dir of dirs) {
                const dirCard = el("button", {
                    display: "flex",
                    flexDirection: "column",
                    gap: "10px",
                    width: "100%",
                    padding: "10px",
                    background: STYLE.cardBg,
                    border: `2px solid ${STYLE.inputBorder}`,
                    borderRadius: "8px",
                    color: STYLE.textPrimary,
                    cursor: "pointer",
                    textAlign: "left",
                    minHeight: "220px",
                    boxSizing: "border-box",
                });
                dirCard.type = "button";
                dirCard.addEventListener("click", () => {
                    void loadPath(String(dir?.path || ""));
                });
                const dirPreview = el("div", {
                    width: "100%",
                    aspectRatio: "3 / 4",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    borderRadius: "6px",
                    background: STYLE.inputBg,
                    border: `1px solid ${STYLE.inputBorder}`,
                });
                const dirIcon = el("div", {
                    fontSize: "42px",
                    lineHeight: "1",
                }, "📁");
                const dirName = el("div", {
                    fontSize: "13px",
                    fontWeight: "600",
                    maxWidth: "100%",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    textAlign: "center",
                }, String(dir?.name || "Folder"));
                dirPreview.appendChild(dirIcon);
                dirCard.append(dirPreview, dirName);
                itemsWrap.appendChild(dirCard);
            }

            assets.forEach((asset) => {
                const selected = selectedAssets.has(asset.id);
                const card = el("button", {
                    display: "flex",
                    flexDirection: "column",
                    gap: "10px",
                    width: "100%",
                    padding: "10px",
                    background: selected ? STYLE.accentSoft : STYLE.cardBg,
                    border: `2px solid ${selected ? STYLE.accentBorder : STYLE.inputBorder}`,
                    borderRadius: "8px",
                    cursor: "pointer",
                    color: STYLE.textPrimary,
                    textAlign: "left",
                    minHeight: "220px",
                    boxSizing: "border-box",
                });
                card.type = "button";
                card.addEventListener("click", (event) => {
                    if (event.shiftKey && applyShiftSelection(asset, assets)) {
                        updateSelectionInfo();
                        syncAssetCardSelectionStates();
                        return;
                    }
                    if (selectedAssets.has(asset.id)) {
                        selectedAssets.delete(asset.id);
                    } else {
                        selectedAssets.set(asset.id, asset);
                    }
                    selectionAnchorId = asset.id;
                    updateSelectionInfo();
                    syncAssetCardSelectionStates();
                });

                const preview = document.createElement("img");
                preview.style.cssText = `
                    width: 100%;
                    aspect-ratio: 3 / 4;
                    object-fit: cover;
                    border-radius: 6px;
                    background: ${STYLE.inputBg};
                    border: 1px solid ${STYLE.inputBorder};
                `;
                setImagePreviewCandidates(preview, asset.previewCandidates);

                const name = el("div", {
                    color: STYLE.textPrimary,
                    fontSize: "13px",
                    fontWeight: "600",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    textAlign: "center",
                }, asset.promptName || asset.assetValue || "Asset");

                card.append(preview, name);
                assetCardEls.set(asset.id, card);
                itemsWrap.appendChild(card);
            });

            emptyState.style.display = itemsWrap.childElementCount === 0 ? "block" : "none";
        };

        const loadPath = async (nextPath) => {
            const token = ++requestToken;
            currentPath = String(nextPath || rootPath || "");
            pathLabel.textContent = `Loading ${currentPath || rootPath || "assets"}...`;
            itemsWrap.innerHTML = "";
            emptyState.style.display = "block";
            emptyState.textContent = "Loading...";
            try {
                const listing = await loadFolder(currentPath);
                if (token !== requestToken) return;
                currentListing = listing;
                emptyState.textContent = emptyMessage;
                render();
            } catch (err) {
                console.error("[PromptBrowserEdit] Asset browser load failed:", err);
                if (token !== requestToken) return;
                currentListing = { currentPath, parentPath: null, dirs: [], assets: [] };
                emptyState.textContent = String(err?.message || err || "Failed to load assets.");
                render();
            }
        };

        const onKeyDown = (event) => {
            if (event.key === "Escape") {
                event.preventDefault();
                close(null);
            }
        };

        filterInput.addEventListener("input", () => render());
        overlay.addEventListener("click", (event) => {
            if (event.target === overlay) close(null);
        });
        document.addEventListener("keydown", onKeyDown, true);
        updateSelectionInfo();
        void loadPath(currentPath || rootPath || "");
        filterInput.focus();
    });
}

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

function replaceSelectOptions(select, options, value) {
    if (!select) return;
    const nextValue = String(value ?? "");
    select.innerHTML = "";
    for (const opt of Array.isArray(options) ? options : []) {
        const option = document.createElement("option");
        option.value = opt.value;
        option.textContent = opt.label;
        if (String(opt.value) === nextValue) option.selected = true;
        select.appendChild(option);
    }
    if (nextValue && !Array.from(select.options).some((option) => option.value === nextValue)) {
        const fallback = document.createElement("option");
        fallback.value = nextValue;
        fallback.textContent = nextValue;
        fallback.selected = true;
        select.appendChild(fallback);
    }
    if (!select.value && select.options.length) {
        select.value = select.options[0].value;
    }
}

function createNumberInput(value, options = {}, styles = {}) {
    const input = document.createElement("input");
    input.type = "number";
    input.value = String(value ?? "1");
    if (options.min !== undefined && options.min !== null) input.min = String(options.min);
    if (options.max !== undefined && options.max !== null) input.max = String(options.max);
    if (options.step !== undefined && options.step !== null) input.step = String(options.step);
    input.style.cssText = `
        width: 84px;
        padding: 7px 8px;
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

function createPickerTrigger(placeholder = "(None)", styles = {}) {
    const button = document.createElement("button");
    button.type = "button";
    button.value = "";
    button.title = placeholder;
    button.style.cssText = `
        width: 100%;
        min-height: 34px;
        padding: 7px 10px;
        background: ${STYLE.inputBg};
        border: 1px solid ${STYLE.inputBorder};
        border-radius: 4px;
        color: #fff;
        font-size: 13px;
        box-sizing: border-box;
        outline: none;
        cursor: pointer;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    `;
    Object.assign(button.style, styles);
    button.dataset.placeholder = placeholder;
    button.setSelectedItem = (item) => {
        const value = String(item?.value || "").trim();
        const label = String(item?.pickerLabel || item?.label || item?.name || value || placeholder).trim();
        button.value = value;
        button.textContent = value ? label : placeholder;
        button.title = value ? `${label}${item?.pickerMeta ? `\n${item.pickerMeta}` : ""}` : placeholder;
        button.style.color = value ? "#fff" : STYLE.textMuted;
    };
    button.getSelectedValue = () => String(button.value || "").trim();
    button.setSelectedItem(null);
    button.addEventListener("focus", () => { button.style.borderColor = STYLE.accent; });
    button.addEventListener("blur", () => { button.style.borderColor = STYLE.inputBorder; });
    return button;
}

function normalizeAssetWeight(value, defaultValue = 1.0, minimum = null, maximum = null) {
    const numeric = Number(value);
    let next = Number.isFinite(numeric) ? numeric : Number(defaultValue);
    if (minimum !== null) next = Math.max(Number(minimum), next);
    if (maximum !== null) next = Math.min(Number(maximum), next);
    return next;
}

function splitFolderAndName(relativePath) {
    const normalized = String(relativePath || "").replace(/\\/g, "/").trim();
    if (!normalized) return { subfolder: "", name: "" };
    const idx = normalized.lastIndexOf("/");
    return {
        subfolder: idx >= 0 ? normalized.substring(0, idx) : "",
        name: idx >= 0 ? normalized.substring(idx + 1) : normalized,
    };
}

function findPickerItemByValue(items, value, fallbackLabel = "") {
    const normalizedValue = String(value || "").trim();
    if (!normalizedValue) return null;
    const list = Array.isArray(items) ? items : [];
    return list.find((item) => String(item?.value || "").trim() === normalizedValue) || {
        value: normalizedValue,
        label: fallbackLabel || normalizedValue,
        pickerLabel: fallbackLabel || normalizedValue,
    };
}

function normalizePickerItems(items, emptyLabel = "(None)") {
    const list = Array.isArray(items) ? items : [];
    return [
        { value: "", label: emptyLabel, pickerLabel: emptyLabel, pickerMeta: "" },
        ...list.map((item) => ({
            ...item,
            value: String(item?.value || "").trim(),
            label: String(item?.label || item?.name || item?.value || "").trim(),
            pickerLabel: String(item?.pickerLabel || item?.name || item?.label || item?.value || "").trim(),
            pickerMeta: String(item?.pickerMeta || item?.label || item?.value || "").trim(),
        })),
    ];
}

function filterPickerItems(items, query) {
    const tokens = String(query || "").trim().toLowerCase().split(/\s+/).filter(Boolean);
    if (!tokens.length) return [...items];
    return items.filter((item) => {
        if (!item?.value) return true;
        const haystack = `${item.pickerLabel || ""} ${item.pickerMeta || ""} ${item.label || ""} ${item.value || ""}`.toLowerCase();
        return tokens.every((token) => haystack.includes(token));
    });
}

function groupPickerItems(items) {
    const ungrouped = [];
    const groups = new Map();
    for (const item of items) {
        if (!item?.value) {
            ungrouped.push(item);
            continue;
        }
        const groupKey = String(item?.group || "").trim();
        if (!groupKey) {
            ungrouped.push(item);
            continue;
        }
        if (!groups.has(groupKey)) groups.set(groupKey, []);
        groups.get(groupKey).push(item);
    }
    const sortByLabel = (a, b) => String(a?.pickerLabel || a?.label || a?.value || "")
        .localeCompare(String(b?.pickerLabel || b?.label || b?.value || ""), undefined, { sensitivity: "base" });
    ungrouped.sort(sortByLabel);
    const sortedGroups = [...groups.entries()]
        .sort((a, b) => a[0].localeCompare(b[0], undefined, { sensitivity: "base" }))
        .map(([group, entries]) => [group, [...entries].sort(sortByLabel)]);
    return { ungrouped, sortedGroups };
}

async function fetchAvailableComposerLoras() {
    if (Array.isArray(_composerLoraPickerCache)) {
        return [..._composerLoraPickerCache];
    }
    if (_composerLoraPickerPromise) {
        return await _composerLoraPickerPromise;
    }
    _composerLoraPickerPromise = (async () => {
        try {
            let rawPaths = [];
            try {
                const resp = await api.fetchApi("/object_info/LoraLoader");
                if (resp?.ok) {
                    const data = await resp.json();
                    const options = data?.LoraLoader?.input?.required?.lora_name?.[0];
                    if (Array.isArray(options)) {
                        rawPaths = [...new Set(options.map((x) => String(x || "").trim()).filter(Boolean))];
                    }
                }
            } catch {
                rawPaths = [];
            }

            if (!rawPaths.length) {
                const resp = await fetch("/prompt-manager-advanced/available-loras");
                const data = await resp.json();
                if (resp.ok && data?.success && Array.isArray(data?.loras)) {
                    rawPaths = [...new Set(data.loras.map((x) => String(x || "").trim()).filter(Boolean))];
                }
            }

            const stripKnownExt = (value) => String(value || "").replace(/\.(safetensors|ckpt|pt|bin|pth)$/i, "");

            const items = [];
            const seen = new Set();
            for (const relPath of rawPaths) {
                const normalizedValue = stripKnownExt(String(relPath || "").replace(/\\/g, "/").trim());
                if (!normalizedValue) continue;
                const key = normalizedValue.toLowerCase();
                if (seen.has(key)) continue;
                seen.add(key);
                const meta = splitFolderAndName(normalizedValue);
                items.push({
                    value: normalizedValue,
                    name: meta.name || normalizedValue,
                    subfolder: meta.subfolder,
                    label: meta.subfolder ? `${meta.subfolder}/${meta.name}` : (meta.name || normalizedValue),
                    pickerLabel: meta.name || normalizedValue,
                    pickerMeta: meta.subfolder ? `${meta.subfolder}/${meta.name}` : normalizedValue,
                    group: meta.subfolder ? meta.subfolder.replace(/\//g, " - ") : "",
                });
            }

            items.sort((a, b) => String(a.label || a.value).localeCompare(String(b.label || b.value), undefined, { sensitivity: "base" }));
            _composerLoraPickerCache = items;
            return [...items];
        } catch (err) {
            console.warn("[PromptBrowserEdit] Failed to load LoRAs:", err);
            _composerLoraPickerCache = [];
            return [];
        } finally {
            _composerLoraPickerPromise = null;
        }
    })();
    return await _composerLoraPickerPromise;
}

async function fetchAvailableComposerRefMods() {
    if (Array.isArray(_composerRefModPickerCache)) {
        return [..._composerRefModPickerCache];
    }
    if (_composerRefModPickerPromise) {
        return await _composerRefModPickerPromise;
    }
    _composerRefModPickerPromise = (async () => {
        try {
            const resp = await api.fetchApi("/object_info/H3RefModLoader");
            if (!resp?.ok) return [];
            const data = await resp.json();
            const options = data?.H3RefModLoader?.input?.required?.mod?.[0];
            const list = Array.isArray(options) ? options : [];
            const items = [...new Set(list.map((x) => String(x || "").trim()).filter(Boolean))]
                .map((value) => {
                    const meta = splitFolderAndName(value);
                    return {
                        value,
                        label: value,
                        name: meta.name || value,
                        pickerLabel: meta.name || value,
                        pickerMeta: value,
                        group: meta.subfolder ? meta.subfolder.replace(/\//g, " - ") : "",
                    };
                })
                .sort((a, b) => a.label.localeCompare(b.label, undefined, { sensitivity: "base" }));
            _composerRefModPickerCache = items;
            return [...items];
        } catch (err) {
            console.warn("[PromptBrowserEdit] Failed to load RefMods:", err);
            _composerRefModPickerCache = [];
            return [];
        } finally {
            _composerRefModPickerPromise = null;
        }
    })();
    return await _composerRefModPickerPromise;
}

function showTextAssetPicker({ title, items, initialValue = "", emptyLabel = "(None)", filterPlaceholder = "Filter..." }) {
    return new Promise((resolve) => {
        const overlay = el("div", {
            position: "fixed",
            inset: "0",
            background: "rgba(0, 0, 0, 0.75)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: "10010",
        });

        const dialog = el("div", {
            background: STYLE.panel,
            border: `1px solid ${STYLE.panelBorder}`,
            borderRadius: "8px",
            width: "560px",
            maxWidth: "92vw",
            maxHeight: "84vh",
            display: "flex",
            flexDirection: "column",
            gap: "10px",
            padding: "16px",
            boxSizing: "border-box",
            boxShadow: "0 12px 36px rgba(0, 0, 0, 0.4)",
        });

        const titleEl = el("div", { color: STYLE.textPrimary, fontSize: "16px", fontWeight: "600" }, title || "Select Item");
        const filterInput = createInput("", filterPlaceholder);
        const list = document.createElement("select");
        list.size = 16;
        list.style.cssText = `
            width: 100%;
            min-height: 280px;
            background: ${STYLE.inputBg};
            border: 1px solid ${STYLE.inputBorder};
            border-radius: 4px;
            color: #fff;
            padding: 6px;
            box-sizing: border-box;
        `;
        list.multiple = false;
        const buttonRow = el("div", { display: "flex", justifyContent: "flex-end", gap: "8px" });
        const cancelBtn = createButton("Cancel", () => close(null));
        const selectBtn = createButton("Select", () => close(list.value), { background: "#2b6d3a", borderColor: "#4a9158", color: "#fff" });
        buttonRow.append(cancelBtn, selectBtn);
        dialog.append(titleEl, filterInput, list, buttonRow);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);

        const allItems = normalizePickerItems(items, emptyLabel);
        const initialItem = findPickerItemByValue(allItems, initialValue, initialValue);
        const render = () => {
            const filtered = filterPickerItems(allItems, filterInput.value);
            const { ungrouped, sortedGroups } = groupPickerItems(filtered);
            const desiredValue = String(list.value || initialItem?.value || "").trim();
            list.innerHTML = "";

            for (const item of ungrouped) {
                const option = document.createElement("option");
                option.value = item.value;
                option.textContent = item.pickerLabel || item.label || item.value || emptyLabel;
                option.title = item.pickerMeta || option.textContent;
                list.appendChild(option);
            }

            for (const [groupLabel, entries] of sortedGroups) {
                const group = document.createElement("optgroup");
                group.label = groupLabel;
                list.appendChild(group);
                for (const item of entries) {
                    const option = document.createElement("option");
                    option.value = item.value;
                    option.textContent = item.pickerLabel || item.label || item.value;
                    option.title = item.pickerMeta || option.textContent;
                    group.appendChild(option);
                }
            }

            if ([...list.querySelectorAll("option")].some((option) => option.value === desiredValue)) {
                list.value = desiredValue;
            } else {
                list.selectedIndex = 0;
            }
        };

        const close = (value) => {
            document.removeEventListener("keydown", onKeyDown, true);
            if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
            resolve(value);
        };

        const onKeyDown = (event) => {
            if (event.key === "Escape") {
                event.preventDefault();
                close(null);
            } else if (event.key === "Enter") {
                event.preventDefault();
                close(list.value);
            }
        };

        list.addEventListener("dblclick", () => close(list.value));
        filterInput.addEventListener("input", render);
        overlay.addEventListener("click", (event) => {
            if (event.target === overlay) close(null);
        });
        document.addEventListener("keydown", onKeyDown, true);
        render();
        filterInput.focus();
    });
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
        syncPromptSelection,
        onChange,
        compact,
        width,
    } = options || {};
    const isCompact = Boolean(compact);
    const requestedWidth = Number(width);
    const EDIT_PANEL_WIDTH = Number.isFinite(requestedWidth) && requestedWidth > 0
        ? Math.round(requestedWidth)
        : (isCompact ? 280 : 320);
    const isSystemPromptsSource = String(endpointPrefix) === "/prompt-generator";
    const isPromptManagerSource = String(endpointPrefix) === "/prompt-manager" || String(endpointPrefix) === "/prompt-manager-advanced";
    const isComposerSource = !isSystemPromptsSource && !isPromptManagerSource;

    const _showInfo = typeof showInfo === "function" ? showInfo : async () => {};
    const _showConfirm = typeof showConfirm === "function" ? showConfirm : async () => false;
    const _generateThumbnail = typeof generateThumbnail === "function" ? generateThumbnail : async () => {};
    const _savePrompt = typeof savePrompt === "function" ? savePrompt : async () => ({ success: false });
    const _loadPrompts = typeof loadPrompts === "function" ? loadPrompts : async () => {};
    const _syncPromptSelection = typeof syncPromptSelection === "function" ? syncPromptSelection : () => {};
    const _selectPrompt = typeof options?.selectPrompt === "function" ? options.selectPrompt : null;
    const _pickThumbnailModel = typeof options?.pickThumbnailModel === "function" ? options.pickThumbnailModel : async () => false;
    const _getThumbnailModelLabel = typeof options?.getThumbnailModelLabel === "function"
        ? options.getThumbnailModelLabel
        : (() => ({ shortLabel: "🔧 Model", iconLabel: "🔧", fullLabel: "Select thumbnail model" }));
    const _onChange = typeof onChange === "function" ? onChange : () => {};
    const _onCategorySettingsSaved = typeof options?.onCategorySettingsSaved === "function" ? options.onCategorySettingsSaved : () => {};

    let currentCategory = "";
    let currentPromptName = "";
    let pendingThumbnail = null;
    let loadedThumbnail = null;
    let loadedPromptText = "";
    let loadedImageLora = "";
    let loadedImageLoraStrength = 1.0;
    let loadedVideoLora = "";
    let loadedVideoLoraStrength = 1.0;
    let loadedRefMod = "";
    let loadedRefModWeight = 1.0;
    let loadedGroupPromptType = "";
    let loadedGroupSubjectType = "subject";
    let loadedGroupSubjectKind = "other";
    let loadedGroupPromptPrefix = "";
    let loadedGroupBasePrompt = "";
    let loadedCategoryPromptType = "";
    let loadedCategoryBasePrompt = "";
    let loadedCategoryPromptPrefix = "";

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

    // Prompt group settings collapsible (top)
    const groupHeader = el("div", {
        display: "flex",
        alignItems: "center",
        gap: "6px",
        padding: "6px 0 8px 0",
        cursor: "pointer",
        userSelect: "none",
        color: STYLE.textPrimary,
        fontSize: "13px",
        fontWeight: "bold",
    }, "Prompt Group Settings");

    const groupArrow = el("span", {
        display: "inline-block",
        width: "12px",
        transition: "color 0.2s ease",
    }, "▶");
    groupHeader.prepend(groupArrow);

    const groupBody = el("div", {
        display: "none",
        flexDirection: "column",
        gap: "10px",
        paddingBottom: "10px",
        borderBottom: `1px solid ${STYLE.sectionBorder}`,
        flexShrink: "0",
    });

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

    let groupOpen = false;
    let settingsOpen = false;
    let promptOpen = true;
    let toolsOpen = false;

    const syncSectionVisibility = () => {
        groupBody.style.display = groupOpen ? "flex" : "none";
        settingsBody.style.display = settingsOpen ? "flex" : "none";
        promptBody.style.display = promptOpen ? "flex" : "none";
        toolsBody.style.display = toolsOpen ? "flex" : "none";
        groupArrow.textContent = groupOpen ? "▼" : "▶";
        settingsArrow.textContent = settingsOpen ? "▼" : "▶";
        promptArrow.textContent = promptOpen ? "▼" : "▶";
        toolsArrow.textContent = toolsOpen ? "▼" : "▶";
    };

    // Category Settings / Tools are only meaningful for the compose source;
    // hide them for System Prompts and Prompt Manager prompts.
    const showCategorySections = isComposerSource;
    groupHeader.style.display = showCategorySections ? "flex" : "none";
    groupBody.style.display = showCategorySections ? groupBody.style.display : "none";
    settingsHeader.style.display = showCategorySections ? "flex" : "none";
    settingsBody.style.display = showCategorySections ? settingsBody.style.display : "none";
    toolsHeader.style.display = showCategorySections ? "flex" : "none";
    toolsBody.style.display = showCategorySections ? toolsBody.style.display : "none";
    if (!showCategorySections) {
        groupOpen = false;
        settingsOpen = false;
        toolsOpen = false;
    }

    groupHeader.addEventListener("click", () => {
        if (groupOpen) {
            groupOpen = false;
        } else {
            groupOpen = true;
            settingsOpen = false;
            promptOpen = false;
            toolsOpen = false;
        }
        syncSectionVisibility();
    });

    settingsHeader.addEventListener("click", () => {
        if (settingsOpen) {
            settingsOpen = false;
        } else {
            groupOpen = false;
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
            groupOpen = false;
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
            groupOpen = false;
            settingsOpen = false;
            promptOpen = false;
        }
        syncSectionVisibility();
    });

    root.appendChild(groupHeader);
    root.appendChild(groupBody);
    root.appendChild(settingsHeader);
    root.appendChild(settingsBody);
    root.appendChild(promptHeader);
    root.appendChild(promptBody);
    root.appendChild(toolsHeader);
    root.appendChild(toolsBody);
    syncSectionVisibility();

    const groupNameLabel = el("label", { color: STYLE.textMuted, fontSize: "12px" }, "Prompt Group");
    groupBody.appendChild(groupNameLabel);

    const groupNameValue = el("div", {
        minHeight: "34px",
        padding: "7px 10px",
        background: STYLE.inputBg,
        border: `1px solid ${STYLE.inputBorder}`,
        borderRadius: "4px",
        color: STYLE.textPrimary,
        fontSize: "13px",
        boxSizing: "border-box",
        display: "flex",
        alignItems: "center",
    }, "");
    groupBody.appendChild(groupNameValue);

    const groupSubjectTypeInfoText = "Controls how prompts from this group behave in subject building.\nNew Subject starts a fresh subject block.\nNon Subject applies styling or modifiers without creating a subject.\nSubject adds details to the current subject.";
    const groupSubjectTypeLabelRow = el("div", {
        display: "flex",
        alignItems: "center",
        gap: "6px",
    });
    const groupSubjectTypeLabel = el("label", { color: STYLE.textMuted, fontSize: "12px" }, "Subject Type");
    groupSubjectTypeLabel.title = groupSubjectTypeInfoText;
    const groupSubjectTypeHint = el("span", {
        color: STYLE.textMuted,
        fontSize: "12px",
        userSelect: "none",
    }, "ⓘ");
    groupSubjectTypeHint.title = groupSubjectTypeInfoText;
    groupSubjectTypeLabelRow.append(groupSubjectTypeLabel, groupSubjectTypeHint);
    groupBody.appendChild(groupSubjectTypeLabelRow);

    const groupSubjectTypeSelect = createSelect("subject", [
        { value: "new_subject", label: "New Subject" },
        { value: "non_subject", label: "Non Subject" },
        { value: "subject", label: "Subject" },
    ]);
    groupSubjectTypeSelect.addEventListener("change", () => _onChange());
    groupBody.appendChild(groupSubjectTypeSelect);

    const groupSubjectKindWrap = el("div", {
        display: "none",
        flexDirection: "column",
        gap: "4px",
    });
    const groupSubjectKindInfoText = "Only used for New Subject groups. Character and Animal each get their own First/Second numbering. Environment stays separate from character numbering.";
    const groupSubjectKindLabelRow = el("div", {
        display: "flex",
        alignItems: "center",
        gap: "6px",
    });
    const groupSubjectKindLabel = el("label", { color: STYLE.textMuted, fontSize: "12px" }, "New Subject Kind");
    groupSubjectKindLabel.title = groupSubjectKindInfoText;
    const groupSubjectKindHint = el("span", {
        color: STYLE.textMuted,
        fontSize: "12px",
        userSelect: "none",
    }, "ⓘ");
    groupSubjectKindHint.title = groupSubjectKindInfoText;
    groupSubjectKindLabelRow.append(groupSubjectKindLabel, groupSubjectKindHint);
    const groupSubjectKindSelect = createSelect("character", COMPOSER_SUBJECT_KIND_CHOICES);
    groupSubjectKindSelect.addEventListener("change", () => _onChange());
    groupSubjectKindWrap.append(groupSubjectKindLabelRow, groupSubjectKindSelect);
    groupBody.appendChild(groupSubjectKindWrap);

    function syncGroupSubjectKindVisibility() {
        const isNewSubject = String(groupSubjectTypeSelect.value || "subject").trim().toLowerCase() === "new_subject";
        groupSubjectKindWrap.style.display = isNewSubject ? "flex" : "none";
    }

    groupSubjectTypeSelect.addEventListener("change", () => {
        if (String(groupSubjectTypeSelect.value || "subject").trim().toLowerCase() === "new_subject") {
            groupSubjectKindSelect.value = normalizeComposerSubjectKind(
                groupSubjectKindSelect.value,
                defaultComposerSubjectKind(groupSubjectTypeSelect.value)
            );
        } else {
            groupSubjectKindSelect.value = "other";
        }
        syncGroupSubjectKindVisibility();
    });
    syncGroupSubjectKindVisibility();

    const groupBasePromptInfoText = "Thumbnail prompt for this prompt group.";
    const groupBasePromptLabelRow = el("div", {
        display: "flex",
        alignItems: "center",
        gap: "6px",
    });
    const groupBasePromptLabel = el("label", { color: STYLE.textMuted, fontSize: "12px" }, "Thumbnail Prompt");
    groupBasePromptLabel.title = groupBasePromptInfoText;
    const groupBasePromptHint = el("span", {
        color: STYLE.textMuted,
        fontSize: "12px",
        userSelect: "none",
    }, "ⓘ");
    groupBasePromptHint.title = groupBasePromptInfoText;
    groupBasePromptLabelRow.append(groupBasePromptLabel, groupBasePromptHint);
    groupBody.appendChild(groupBasePromptLabelRow);

    const groupBasePromptInput = createTextarea("", "A shared thumbnail prompt for this prompt group");
    groupBasePromptInput.style.minHeight = "112px";
    groupBasePromptInput.style.resize = "vertical";
    groupBasePromptInput.addEventListener("input", () => _onChange());
    groupBody.appendChild(groupBasePromptInput);

    const groupPrefixInfoText = "Prefix prompt to add before prompt.\nex: \"She wears\" \"a green dress\"";
    const groupPrefixLabelRow = el("div", {
        display: "flex",
        alignItems: "center",
        gap: "6px",
    });
    const groupPrefixLabel = el("label", { color: STYLE.textMuted, fontSize: "12px" }, "Generation Prefix");
    groupPrefixLabel.title = groupPrefixInfoText;
    const groupPrefixHint = el("span", {
        color: STYLE.textMuted,
        fontSize: "12px",
        userSelect: "none",
    }, "ⓘ");
    groupPrefixHint.title = groupPrefixInfoText;
    groupPrefixLabelRow.append(groupPrefixLabel, groupPrefixHint);
    groupBody.appendChild(groupPrefixLabelRow);

    const groupPrefixInput = createInput("", "");
    groupPrefixInput.addEventListener("input", () => _onChange());
    groupBody.appendChild(groupPrefixInput);

    const saveGroupSettingsBtn = createButton("Save Prompt Group Settings", async () => {
        const promptType = String(loadedGroupPromptType || "").trim();
        if (!promptType) {
            await _showInfo("Missing Prompt Group", "Please select a prompt group first.");
            return;
        }

        const groupMeta = getPromptTypeMetadata(node?.prompts, promptType, node?.composerPromptLibrary);
        if (!groupMeta.exists || !String(groupMeta.typeFile || "").trim()) {
            await _showInfo("Missing Prompt Group Data", `Could not resolve canonical data for prompt group "${promptType}".`);
            return;
        }
        const confirmed = await _showConfirm(
            "Save Prompt Group Settings",
            `Save prompt group settings for "${groupMeta.typeName || promptType}"?`,
            "Save",
            "#2b6d3a"
        );
        if (!confirmed) return;

        const result = await saveComposerTypeSettings({
            typeFile: groupMeta.typeFile,
            promptType,
            typeName: groupMeta.typeName || promptType,
            subjectType: groupSubjectTypeSelect.value,
            subjectKind: groupSubjectKindSelect.value,
            promptPrefix: groupPrefixInput.value,
            basePrompt: groupBasePromptInput.value,
        });
        if (result?.success) {
            if (result.library && typeof result.library === "object") {
                node.composerPromptLibrary = result.library;
            }
            if (result.prompts && typeof result.prompts === "object") {
                node.composerPrompts = result.prompts;
                node.prompts = result.prompts;
            }
            loadTypeSettings(promptType);
            if (currentCategory) {
                const currentCategoryType = String(node?.prompts?.[currentCategory]?._prompt_type_ || "").trim();
                loadCategorySettings(currentCategory);
                if (promptType && promptType !== currentCategoryType) {
                    loadTypeSettings(promptType);
                }
            }
            _onChange();
        } else {
            await _showInfo("Save Failed", result?.error || "Failed to save prompt group settings.");
        }
    }, { background: "#2b6d3a", borderColor: "#4a9158", color: "#fff" });
    groupBody.appendChild(saveGroupSettingsBtn);

    const categoryBasePromptInfoText = "Thumbnail prompt for this category.\nIf empty, the prompt group thumbnail prompt is used.";
    const categoryBasePromptLabelRow = el("div", {
        display: "flex",
        alignItems: "center",
        gap: "6px",
    });
    const categoryBasePromptLabel = el("label", { color: STYLE.textMuted, fontSize: "12px" }, "Thumbnail Prompt");
    categoryBasePromptLabel.title = categoryBasePromptInfoText;
    const categoryBasePromptHint = el("span", {
        color: STYLE.textMuted,
        fontSize: "12px",
        userSelect: "none",
    }, "ⓘ");
    categoryBasePromptHint.title = categoryBasePromptInfoText;
    categoryBasePromptLabelRow.append(categoryBasePromptLabel, categoryBasePromptHint);
    settingsBody.appendChild(categoryBasePromptLabelRow);

    const categoryBasePromptInput = createTextarea("", "A thumbnail prompt for this category");
    categoryBasePromptInput.style.minHeight = "96px";
    categoryBasePromptInput.style.resize = "vertical";
    categoryBasePromptInput.addEventListener("input", () => _onChange());
    settingsBody.appendChild(categoryBasePromptInput);

    const prefixInfoText = "Prefix prompt to add before prompt.\nex: \"She wears\" \"a green dress\"\nIf empty, the prompt group prefix is used.";
    const prefixLabelRow = el("div", {
        display: "flex",
        alignItems: "center",
        gap: "6px",
    });
    const prefixLabel = el("label", { color: STYLE.textMuted, fontSize: "12px" }, "Generation Prefix");
    prefixLabel.title = prefixInfoText;
    const prefixHint = el("span", {
        color: STYLE.textMuted,
        fontSize: "12px",
        userSelect: "none",
    }, "ⓘ");
    prefixHint.title = prefixInfoText;
    prefixLabelRow.append(prefixLabel, prefixHint);
    settingsBody.appendChild(prefixLabelRow);

    const prefixInput = createInput("", "");
    prefixInput.addEventListener("input", () => _onChange());
    settingsBody.appendChild(prefixInput);

    const saveSettingsBtn = createButton("Save Category Settings", async () => {
        const category = currentCategory;
        const categoryLabel = getCategoryDisplayName(node?.prompts, category);
        if (!category) {
            await _showInfo("Missing Category", "Please select a category first.");
            return;
        }

        const categoryData = node?.prompts?.[category];
        const promptKeys = Object.keys(getCategoryPromptEntries(categoryData, "composer"));
        const hasExistingPrompts = promptKeys.length > 0;
        const hasExistingCategorySettings = !!(
            categoryData
            && typeof categoryData === "object"
            && (
                String(categoryData._category_base_prompt_ || "").trim()
                || String(categoryData._category_prefix_ || "").trim()
            )
        );

        if (hasExistingPrompts || hasExistingCategorySettings) {
            const confirmed = await _showConfirm(
                "Update Category Settings",
                `Category settings for "${categoryLabel}" already exist. Update them?`,
                "Update",
                "#c44"
            );
            if (!confirmed) return;
        }

        const previousPromptType = loadedCategoryPromptType;
        const categoryMeta = getComposerCategoryMetadata(node?.composerPromptLibrary, category, loadedCategoryPromptType);
        if (!categoryMeta.exists || !String(categoryMeta.typeFile || "").trim()) {
            await _showInfo("Missing Category Data", `Could not resolve canonical data for category "${categoryLabel}".`);
            return;
        }
        const result = await saveComposerCategorySettings(categoryMeta.categoryName, {
            typeFile: categoryMeta.typeFile,
            promptType: categoryMeta.promptType,
            basePrompt: categoryBasePromptInput.value,
            promptPrefix: prefixInput.value,
        });
        if (result?.success) {
            if (result.library && typeof result.library === "object") {
                node.composerPromptLibrary = result.library;
            }
            node.prompts = result.prompts;
            loadCategorySettings(category);
            _onChange();
            await _onCategorySettingsSaved({
                category,
                previousPromptType,
                promptType: String(loadedCategoryPromptType || "").trim(),
                promptPrefix: String(prefixInput.value || ""),
            });
        } else {
            await _showInfo("Save Failed", result?.error || "Failed to save category settings.");
        }
    }, { background: "#2b6d3a", borderColor: "#4a9158", color: "#fff" });
    settingsBody.appendChild(saveSettingsBtn);

    // Prompt editor section
    const promptNameInput = createInput("", "Prompt name");
    promptBody.appendChild(promptNameInput);
    promptNameInput.addEventListener("input", () => {
        const nextName = String(promptNameInput.value || "").trim();
        if (!isEditingExistingPrompt() || nextName === String(currentPromptName || "").trim()) {
            _syncPromptSelection(currentCategory, nextName);
        }
        updateEditorActionButtons();
        _onChange();
    });

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

    const imageLoraRow = el("div", {
        display: isComposerSource ? "flex" : "none",
        flexDirection: "column",
        gap: "4px",
    });
    const imageLoraLabel = el("label", { color: STYLE.textMuted, fontSize: "12px" }, "Image LoRA");
    const imageLoraControls = el("div", { display: "flex", gap: "8px", alignItems: "center" });
    const imageLoraTrigger = createPickerTrigger("(None)", { flex: "1" });
    imageLoraTrigger.addEventListener("click", async () => {
        const loras = await fetchAvailableComposerLoras();
        const selected = await showTextAssetPicker({
            title: "Select Image LoRA",
            items: loras,
            initialValue: imageLoraTrigger.getSelectedValue(),
            emptyLabel: "(None)",
            filterPlaceholder: "Filter LoRAs by one or more keywords...",
        });
        if (selected !== null) {
            imageLoraTrigger.setSelectedItem(findPickerItemByValue(loras, selected));
            _onChange();
        }
    });
    const imageLoraStrengthInput = createNumberInput(1.0, { step: 0.05 }, { width: "72px" });
    imageLoraStrengthInput.title = "Image LoRA strength";
    imageLoraStrengthInput.addEventListener("input", () => _onChange());
    imageLoraControls.append(imageLoraTrigger, imageLoraStrengthInput);
    imageLoraRow.append(imageLoraLabel, imageLoraControls);
    promptBody.appendChild(imageLoraRow);

    const videoLoraRow = el("div", {
        display: isComposerSource ? "flex" : "none",
        flexDirection: "column",
        gap: "4px",
    });
    const videoLoraLabel = el("label", { color: STYLE.textMuted, fontSize: "12px" }, "Video LoRA");
    const videoLoraControls = el("div", { display: "flex", gap: "8px", alignItems: "center" });
    const videoLoraTrigger = createPickerTrigger("(None)", { flex: "1" });
    videoLoraTrigger.addEventListener("click", async () => {
        const loras = await fetchAvailableComposerLoras();
        const selected = await showTextAssetPicker({
            title: "Select Video LoRA",
            items: loras,
            initialValue: videoLoraTrigger.getSelectedValue(),
            emptyLabel: "(None)",
            filterPlaceholder: "Filter LoRAs by one or more keywords...",
        });
        if (selected !== null) {
            videoLoraTrigger.setSelectedItem(findPickerItemByValue(loras, selected));
            _onChange();
        }
    });
    const videoLoraStrengthInput = createNumberInput(1.0, { step: 0.05 }, { width: "72px" });
    videoLoraStrengthInput.title = "Video LoRA strength";
    videoLoraStrengthInput.addEventListener("input", () => _onChange());
    videoLoraControls.append(videoLoraTrigger, videoLoraStrengthInput);
    videoLoraRow.append(videoLoraLabel, videoLoraControls);
    promptBody.appendChild(videoLoraRow);

    const refModRow = el("div", {
        display: isComposerSource ? "flex" : "none",
        flexDirection: "column",
        gap: "4px",
    });
    const refModLabel = el("label", { color: STYLE.textMuted, fontSize: "12px" }, "RefMod");
    const refModControls = el("div", { display: "flex", gap: "8px", alignItems: "center" });
    const refModTrigger = createPickerTrigger("(None)", { flex: "1" });
    refModTrigger.addEventListener("click", async () => {
        const refmods = await fetchAvailableComposerRefMods();
        const selected = await showTextAssetPicker({
            title: "Select RefMod",
            items: refmods,
            initialValue: refModTrigger.getSelectedValue(),
            emptyLabel: "(None)",
            filterPlaceholder: "Filter RefMods by one or more keywords...",
        });
        if (selected !== null) {
            refModTrigger.setSelectedItem(findPickerItemByValue(refmods, selected));
            _onChange();
        }
    });
    const refModWeightInput = createNumberInput(1.0, { min: 0, max: 10, step: 0.05 }, { width: "72px" });
    refModWeightInput.title = "RefMod weight";
    refModWeightInput.addEventListener("input", () => _onChange());
    refModControls.append(refModTrigger, refModWeightInput);
    refModRow.append(refModLabel, refModControls);
    promptBody.appendChild(refModRow);

    const readCurrentImageLoraStrength = () => normalizeAssetWeight(imageLoraStrengthInput.value, 1.0);
    const readCurrentVideoLoraStrength = () => normalizeAssetWeight(videoLoraStrengthInput.value, 1.0);
    const readCurrentRefModWeight = () => normalizeAssetWeight(refModWeightInput.value, 1.0, 0.0, 10.0);

    async function refreshComposerAssetChoices(nextValues = null) {
        if (!isComposerSource) return;
        const currentImageLoraValue = String(nextValues?.loraImage ?? imageLoraTrigger.getSelectedValue() ?? loadedImageLora ?? "").trim();
        const currentVideoLoraValue = String(nextValues?.loraVideo ?? videoLoraTrigger.getSelectedValue() ?? loadedVideoLora ?? "").trim();
        const currentRefModValue = String(nextValues?.refmod ?? refModTrigger.getSelectedValue() ?? loadedRefMod ?? "").trim();
        const [loras, refmods] = await Promise.all([
            fetchAvailableComposerLoras(),
            fetchAvailableComposerRefMods(),
        ]);
        imageLoraTrigger.setSelectedItem(findPickerItemByValue(loras, currentImageLoraValue));
        videoLoraTrigger.setSelectedItem(findPickerItemByValue(loras, currentVideoLoraValue));
        refModTrigger.setSelectedItem(findPickerItemByValue(refmods, currentRefModValue));
    }

    const editorButtonRow = el("div", {
        display: "flex",
        gap: "8px",
        justifyContent: "flex-end",
    });

    const saveNewBtn = createButton("Save New", async () => {
        await doSavePrompt(false);
    }, { background: "#313843", borderColor: "#5f6773", color: STYLE.textPrimary });

    const saveBtn = createButton("Save", async () => {
        await doPrimaryPromptAction(false);
    }, { background: "#2b6d3a", borderColor: "#4a9158", color: "#fff" });
    saveNewBtn.style.display = "none";
    editorButtonRow.appendChild(saveNewBtn);
    editorButtonRow.appendChild(saveBtn);
    promptBody.appendChild(editorButtonRow);

    function getLoadedPromptEntry(category = currentCategory, promptName = currentPromptName) {
        return getCategoryPromptEntryForEndpoint(node?.prompts?.[category], promptName, endpointPrefix);
    }

    function isEditingExistingPrompt() {
        return !!(currentCategory && currentPromptName && getLoadedPromptEntry(currentCategory, currentPromptName));
    }

    function hasPromptNameChanged() {
        if (!isEditingExistingPrompt()) return false;
        const currentName = String(promptNameInput.value || "").trim();
        return currentName.length > 0 && currentName !== String(currentPromptName || "").trim();
    }

    function findPromptNameConflict(category, name, excludedName = "") {
        const targetName = String(name || "").trim().toLowerCase();
        const skipName = String(excludedName || "").trim().toLowerCase();
        if (!category || !targetName) return "";
        const promptEntries = getCategoryPromptEntriesForEndpoint(node?.prompts?.[category], endpointPrefix);
        for (const entryName of Object.keys(promptEntries)) {
            const normalized = String(entryName || "").trim().toLowerCase();
            if (!normalized || normalized === skipName) continue;
            if (normalized === targetName) return entryName;
        }
        return "";
    }

    function buildCurrentPromptSavePayload(options = {}) {
        const includeOriginalIdentity = options.includeOriginalIdentity === true;
        const payload = {
            category: currentCategory,
            name: String(promptNameInput.value || "").trim(),
            text: String(promptTextArea.value || "").trim(),
            thumbnail: pendingThumbnail || loadedThumbnail,
        };
        if (includeOriginalIdentity && isEditingExistingPrompt()) {
            payload.old_name = String(currentPromptName || "").trim();
            payload.old_category = String(currentCategory || "").trim();
        }
        if (isComposerSource) {
            payload.lora_image = String(imageLoraTrigger.getSelectedValue() || "").trim();
            payload.lora_image_strength = readCurrentImageLoraStrength();
            payload.lora_video = String(videoLoraTrigger.getSelectedValue() || "").trim();
            payload.lora_video_strength = readCurrentVideoLoraStrength();
            payload.refmod = String(refModTrigger.getSelectedValue() || "").trim();
            payload.refmod_weight = readCurrentRefModWeight();
        }
        return payload;
    }

    async function applyPromptSaveResult(result, category, name) {
        if (!result?.success) return result || { success: false };
        if (result?.library && typeof result.library === "object") {
            node.composerPromptLibrary = result.library;
        }
        if (result?.prompts && typeof result.prompts === "object") {
            node.composerPrompts = result.prompts;
            node.prompts = result.prompts;
        } else {
            await _loadPrompts(node);
        }
        currentPromptName = name;
        pendingThumbnail = null;
        const entry = getCategoryPromptEntryForEndpoint(node?.prompts?.[category], name, endpointPrefix);
        loadedPromptText = entry?.prompt || "";
        loadedThumbnail = entry?.thumbnail || null;
        loadedImageLora = String(entry?.lora_image || entry?.lora || "").trim();
        loadedImageLoraStrength = normalizeAssetWeight(entry?.lora_image_strength ?? entry?.lora_strength, 1.0);
        loadedVideoLora = String(entry?.lora_video || "").trim();
        loadedVideoLoraStrength = normalizeAssetWeight(entry?.lora_video_strength, 1.0);
        loadedRefMod = String(entry?.refmod || "").trim();
        loadedRefModWeight = normalizeAssetWeight(entry?.refmod_weight, 1.0, 0.0, 10.0);
        imageLoraStrengthInput.value = String(loadedImageLoraStrength);
        videoLoraStrengthInput.value = String(loadedVideoLoraStrength);
        refModWeightInput.value = String(loadedRefModWeight);
        await refreshComposerAssetChoices({ loraImage: loadedImageLora, loraVideo: loadedVideoLora, refmod: loadedRefMod });
        updateThumbnailDisplay(entry?.thumbnail || null);
        updateEditorActionButtons();
        _onChange();
        return result;
    }

    async function saveCurrentPromptPayload(savePayload, options = {}) {
        const result = await _savePrompt(savePayload);
        if (result?.success) {
            return await applyPromptSaveResult(result, savePayload.category, savePayload.name);
        }
        if (options.showFailure !== false) {
            await _showInfo(options.failureTitle || "Save Failed", result?.error || options.failureMessage || "Failed to save prompt.");
        }
        return result || { success: false };
    }

    function updateEditorActionButtons() {
        const editingExisting = isEditingExistingPrompt();
        const nameChanged = hasPromptNameChanged();
        saveBtn.textContent = editingExisting ? (nameChanged ? "Rename" : "Update") : "Save";
        saveNewBtn.style.display = editingExisting && nameChanged ? "inline-flex" : "none";
    }

    async function doPrimaryPromptAction(autoFromGeneration) {
        if (isEditingExistingPrompt()) {
            return await doUpdatePrompt(autoFromGeneration);
        }
        return await doSavePrompt(autoFromGeneration);
    }

    async function doUpdatePrompt(autoFromGeneration) {
        const category = currentCategory;
        const categoryLabel = getCategoryDisplayName(node?.prompts, category);
        const originalName = String(currentPromptName || "").trim();
        const nextName = String(promptNameInput.value || "").trim();

        if (!category) {
            await _showInfo("Missing Category", "Please select a category first.");
            return { success: false };
        }
        if (!originalName || !getLoadedPromptEntry(category, originalName)) {
            return await doSavePrompt(autoFromGeneration);
        }
        if (!nextName) {
            await _showInfo("Missing Name", "Please enter a prompt name.");
            promptNameInput.focus();
            return { success: false };
        }

        const conflictingName = findPromptNameConflict(category, nextName, originalName);
        if (conflictingName) {
            await _showInfo("Update Failed", `A prompt named "${conflictingName}" already exists in category "${categoryLabel}".`);
            return { success: false };
        }

        const confirmed = await _showConfirm(
            "Update Prompt",
            nextName === originalName
                ? `Update prompt "${originalName}" in category "${categoryLabel}"?`
                : `Update prompt "${originalName}" and rename it to "${nextName}" in category "${categoryLabel}"?`,
            "Update",
            "#2b6d3a"
        );
        if (!confirmed) return { success: false };

        const savePayload = buildCurrentPromptSavePayload({ includeOriginalIdentity: true });
        const saveResult = await saveCurrentPromptPayload(savePayload, {
            showFailure: !autoFromGeneration,
            failureTitle: "Update Failed",
            failureMessage: "Failed to update prompt.",
        });
        return saveResult?.success ? saveResult : (saveResult || { success: false });
    }

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

    const thumbnailGenerateRow = el("div", {
        display: "flex",
        gap: "8px",
        alignItems: "flex-end",
    });
    const thumbnailControlHeight = "32px";
    const thumbnailSideControlWidth = "62px";
    const thumbnailModelBtn = createButton("Model", async () => {
        const picked = await _pickThumbnailModel();
        if (picked) {
            refreshThumbnailModelButton();
        }
    }, {
        flex: `0 0 ${thumbnailSideControlWidth}`,
        width: thumbnailSideControlWidth,
        height: thumbnailControlHeight,
        minHeight: thumbnailControlHeight,
        padding: "4px 10px",
        minWidth: "0",
        alignSelf: "flex-end",
    });
    thumbnailModelBtn.title = "Select thumbnail model";
    const thumbnailSeedWrap = el("div", {
        display: "flex",
        flexDirection: "column",
        gap: "4px",
        width: thumbnailSideControlWidth,
        flexShrink: "0",
    });
    const thumbnailSeedLabel = el("label", { color: STYLE.textMuted, fontSize: "12px", textAlign: "center" }, "Seed");
    const thumbnailSeedInput = createNumberInput(42, { step: 1 }, {
        width: "100%",
        height: thumbnailControlHeight,
        minHeight: thumbnailControlHeight,
        padding: "4px 8px",
    });
    thumbnailSeedInput.title = "Seed used for edit-panel thumbnail generation";
    thumbnailSeedInput.addEventListener("input", () => _onChange());
    thumbnailSeedWrap.append(thumbnailSeedLabel, thumbnailSeedInput);

    const generateBtn = createButton("Generate Thumbnail", async () => {
        const category = currentCategory;
        const categoryLabel = getCategoryDisplayName(node?.prompts, category);
        const name = String(promptNameInput.value || "").trim();
        const text = String(promptTextArea.value || "").trim();
        const requestedSeed = Number(thumbnailSeedInput.value);
        const thumbnailSeed = Number.isFinite(requestedSeed) ? Math.trunc(requestedSeed) : 42;
        if (!category || !name) {
            await _showInfo("Missing Prompt", "Please enter a category and prompt name first.");
            return;
        }
        if (!text) {
            await _showInfo("Missing Prompt Text", "Please enter prompt text before generating a thumbnail.");
            promptTextArea.focus();
            return;
        }

        generateBtn.disabled = true;
        generateBtn.textContent = "Generating...";
        try {
            const draftPromptData = {
                prompt: text,
                __pm_thumbnail_seed: thumbnailSeed,
            };
            const savedEntry = getCategoryPromptEntryForEndpoint(node?.prompts?.[category], name, endpointPrefix);
            if (savedEntry && typeof savedEntry === "object" && savedEntry.workflow_data) {
                draftPromptData.workflow_data = savedEntry.workflow_data;
            }
            if (isComposerSource) {
                draftPromptData.lora_image = String(imageLoraTrigger.getSelectedValue() || "").trim();
                draftPromptData.lora_image_strength = readCurrentImageLoraStrength();
                draftPromptData.lora_video = String(videoLoraTrigger.getSelectedValue() || "").trim();
                draftPromptData.lora_video_strength = readCurrentVideoLoraStrength();
                draftPromptData.refmod = String(refModTrigger.getSelectedValue() || "").trim();
                draftPromptData.refmod_weight = readCurrentRefModWeight();
            }
            const thumbnail = await _generateThumbnail(category, name, draftPromptData);
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
    generateBtn.style.flex = "1";
    generateBtn.style.height = thumbnailControlHeight;
    generateBtn.style.minHeight = thumbnailControlHeight;
    const refreshThumbnailModelButton = () => {
        const label = _getThumbnailModelLabel() || {};
        thumbnailModelBtn.textContent = String(label.iconLabel || label.shortLabel || "🔧");
        thumbnailModelBtn.title = String(label.fullLabel || "Select thumbnail model");
    };
    refreshThumbnailModelButton();
    thumbnailGenerateRow.append(thumbnailModelBtn, generateBtn, thumbnailSeedWrap);
    promptBody.appendChild(thumbnailGenerateRow);

    const bulkPromptBtn = createButton("Bulk Prompt Importer", () => {
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
            const promptEntries = getCategoryPromptEntriesForEndpoint(categoryData, endpointPrefix);
            if (Object.prototype.hasOwnProperty.call(promptEntries, name)) {
                existing = promptEntries[name];
                existingFound = true;
            } else {
                const target = name.toLowerCase();
                for (const [entryName, entryValue] of Object.entries(promptEntries)) {
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
                    `Prompt "${name}" already exists in category "${categoryLabel}". Replace it?`,
                    "Replace",
                    "#c44"
                );
            }
            if (!overwrite) return { success: false };
        }

        const thumbnail = pendingThumbnail || loadedThumbnail;
        const savePayload = buildCurrentPromptSavePayload();
        savePayload.category = category;
        savePayload.name = name;
        savePayload.text = text;
        savePayload.thumbnail = thumbnail;
        return await saveCurrentPromptPayload(savePayload, {
            showFailure: !autoFromGeneration,
            failureTitle: "Save Failed",
            failureMessage: "Failed to save prompt.",
        });
    }

    function hasUnsavedChanges() {
        const currentName = String(promptNameInput.value || "").trim();
        const currentText = String(promptTextArea.value || "").trim();
        const currentGroupSubjectType = String(groupSubjectTypeSelect.value || "subject").trim().toLowerCase() || "subject";
        const currentGroupSubjectKind = normalizeComposerSubjectKind(
            groupSubjectKindSelect.value,
            defaultComposerSubjectKind(currentGroupSubjectType)
        );
        const currentUsesSubjectKind = currentGroupSubjectType === "new_subject";
        const loadedUsesSubjectKind = String(loadedGroupSubjectType || "subject").trim().toLowerCase() === "new_subject";
        const currentGroupPromptPrefix = String(groupPrefixInput.value || "").trim();
        const currentGroupBasePrompt = String(groupBasePromptInput.value || "").trim();
        const currentCategoryBasePrompt = String(categoryBasePromptInput.value || "").trim();
        const currentCategoryPromptPrefix = String(prefixInput.value || "").trim();

        if (currentGroupSubjectType !== String(loadedGroupSubjectType || "subject").trim().toLowerCase()) return true;
        if ((currentUsesSubjectKind || loadedUsesSubjectKind)
            && currentGroupSubjectKind !== normalizeComposerSubjectKind(loadedGroupSubjectKind, defaultComposerSubjectKind(loadedGroupSubjectType))) return true;
        if (currentGroupPromptPrefix !== String(loadedGroupPromptPrefix || "").trim()) return true;
        if (currentGroupBasePrompt !== String(loadedGroupBasePrompt || "").trim()) return true;

        if (!currentCategory) {
            return false;
        }

        if (currentCategoryBasePrompt !== String(loadedCategoryBasePrompt || "").trim()) return true;
        if (currentCategoryPromptPrefix !== String(loadedCategoryPromptPrefix || "").trim()) return true;

        const entry = getCategoryPromptEntryForEndpoint(node?.prompts?.[currentCategory], currentPromptName, endpointPrefix);
        if (currentPromptName && entry && typeof entry === "object") {
            const originalText = String(entry.prompt || "").trim();
            const originalThumbnail = entry.thumbnail || null;
            if (currentName !== currentPromptName) return true;
            if (currentText !== originalText) return true;
            if (isComposerSource) {
                if (String(imageLoraTrigger.getSelectedValue() || "").trim() !== String(entry.lora_image || entry.lora || "").trim()) return true;
                if (readCurrentImageLoraStrength() !== normalizeAssetWeight(entry.lora_image_strength ?? entry.lora_strength, 1.0)) return true;
                if (String(videoLoraTrigger.getSelectedValue() || "").trim() !== String(entry.lora_video || "").trim()) return true;
                if (readCurrentVideoLoraStrength() !== normalizeAssetWeight(entry.lora_video_strength, 1.0)) return true;
                if (String(refModTrigger.getSelectedValue() || "").trim() !== String(entry.refmod || "").trim()) return true;
                if (readCurrentRefModWeight() !== normalizeAssetWeight(entry.refmod_weight, 1.0, 0.0, 10.0)) return true;
            }
            if (pendingThumbnail !== null && pendingThumbnail !== originalThumbnail) return true;
            return false;
        }

        if (isComposerSource) {
            if (String(imageLoraTrigger.getSelectedValue() || "").trim()) return true;
            if (String(videoLoraTrigger.getSelectedValue() || "").trim()) return true;
            if (String(refModTrigger.getSelectedValue() || "").trim()) return true;
            if (readCurrentImageLoraStrength() !== 1.0) return true;
            if (readCurrentVideoLoraStrength() !== 1.0) return true;
            if (readCurrentRefModWeight() !== 1.0) return true;
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
        groupOpen = false;
        settingsOpen = false;
        promptOpen = true;
        if (!showCategorySections) {
            toolsOpen = false;
        } else {
            toolsOpen = false;
        }
        syncSectionVisibility();

        const entry = getCategoryPromptEntryForEndpoint(node?.prompts?.[category], promptName, endpointPrefix);
        if (entry && typeof entry === "object") {
            promptTextArea.value = entry.prompt || "";
            loadedPromptText = entry.prompt || "";
            loadedThumbnail = entry.thumbnail || null;
            loadedImageLora = String(entry.lora_image || entry.lora || "").trim();
            loadedImageLoraStrength = normalizeAssetWeight(entry.lora_image_strength ?? entry.lora_strength, 1.0);
            loadedVideoLora = String(entry.lora_video || "").trim();
            loadedVideoLoraStrength = normalizeAssetWeight(entry.lora_video_strength, 1.0);
            loadedRefMod = String(entry.refmod || "").trim();
            loadedRefModWeight = normalizeAssetWeight(entry.refmod_weight, 1.0, 0.0, 10.0);
            imageLoraStrengthInput.value = String(loadedImageLoraStrength);
            videoLoraStrengthInput.value = String(loadedVideoLoraStrength);
            refModWeightInput.value = String(loadedRefModWeight);
            await refreshComposerAssetChoices({ loraImage: loadedImageLora, loraVideo: loadedVideoLora, refmod: loadedRefMod });
            updateThumbnailDisplay(loadedThumbnail);
            updateEditorActionButtons();
        } else {
            promptTextArea.value = "";
            loadedPromptText = "";
            loadedThumbnail = null;
            loadedImageLora = "";
            loadedImageLoraStrength = 1.0;
            loadedVideoLora = "";
            loadedVideoLoraStrength = 1.0;
            loadedRefMod = "";
            loadedRefModWeight = 1.0;
            imageLoraStrengthInput.value = "1";
            videoLoraStrengthInput.value = "1";
            refModWeightInput.value = "1";
            await refreshComposerAssetChoices({ loraImage: "", loraVideo: "", refmod: "" });
            updateThumbnailDisplay(null);
            updateEditorActionButtons();
        }

        loadCategorySettings(category);
        return true;
    }

    function loadTypeSettings(promptType) {
        const selectedPromptType = String(promptType || "").trim();
        if (!selectedPromptType || selectedPromptType === "__all__") {
            groupNameValue.textContent = "No prompt group selected";
            groupSubjectTypeSelect.value = "subject";
            groupSubjectKindSelect.value = "other";
            groupPrefixInput.value = "";
            groupBasePromptInput.value = "";
            loadedGroupPromptType = "";
            loadedGroupSubjectType = "subject";
            loadedGroupSubjectKind = "other";
            loadedGroupPromptPrefix = "";
            loadedGroupBasePrompt = "";
            syncGroupSubjectKindVisibility();
            return;
        }
        const typeMeta = getPromptTypeMetadata(node?.prompts, selectedPromptType || loadedGroupPromptType || "", node?.composerPromptLibrary);
        groupNameValue.textContent = String(typeMeta.typeName || typeMeta.promptType || "").trim() || "No prompt group selected";
        groupSubjectTypeSelect.value = String(typeMeta.subjectType || "subject").trim().toLowerCase() || "subject";
        groupSubjectKindSelect.value = normalizeComposerSubjectKind(
            typeMeta.subjectKind,
            defaultComposerSubjectKind(typeMeta.subjectType || "subject")
        );
        groupPrefixInput.value = String(typeMeta.promptPrefix || "");
        groupBasePromptInput.value = String(typeMeta.basePrompt || "");
        loadedGroupPromptType = String(typeMeta.promptType || "").trim();
        loadedGroupSubjectType = String(typeMeta.subjectType || "subject").trim().toLowerCase() || "subject";
        loadedGroupSubjectKind = normalizeComposerSubjectKind(
            typeMeta.subjectKind,
            defaultComposerSubjectKind(typeMeta.subjectType || "subject")
        );
        loadedGroupPromptPrefix = String(typeMeta.promptPrefix || "").trim();
        loadedGroupBasePrompt = String(typeMeta.basePrompt || "").trim();
        syncGroupSubjectKindVisibility();
    }

    function loadCategorySettings(category) {
        currentCategory = category || currentCategory || "";
        const catData = node?.prompts?.[category];
        if (catData && typeof catData === "object") {
            const promptType = String(catData._prompt_type_ || "").trim();
            const categoryMeta = getComposerCategoryMetadata(node?.composerPromptLibrary, category, promptType);
            loadTypeSettings(promptType);
            categoryBasePromptInput.value = categoryMeta.exists ? String(categoryMeta.categoryBasePrompt || "") : "";
            prefixInput.value = categoryMeta.exists ? String(categoryMeta.categoryPromptPrefix || "") : "";
            loadedCategoryPromptType = categoryMeta.exists
                ? String(categoryMeta.promptType || promptType).trim()
                : promptType;
            loadedCategoryBasePrompt = categoryMeta.exists ? String(categoryMeta.categoryBasePrompt || "").trim() : "";
            loadedCategoryPromptPrefix = categoryMeta.exists ? String(categoryMeta.categoryPromptPrefix || "").trim() : "";
        } else {
            loadTypeSettings("");
            categoryBasePromptInput.value = "";
            prefixInput.value = "";
            loadedCategoryPromptType = "";
            loadedCategoryBasePrompt = "";
            loadedCategoryPromptPrefix = "";
        }
    }

    function showTypeSettings() {
        if (!showCategorySections) {
            showPromptSettings();
            return;
        }
        groupOpen = true;
        settingsOpen = false;
        promptOpen = false;
        toolsOpen = false;
        syncSectionVisibility();
    }

    function showCategorySettings() {
        if (!showCategorySections) {
            showPromptSettings();
            return;
        }
        groupOpen = false;
        settingsOpen = true;
        promptOpen = false;
        toolsOpen = false;
        syncSectionVisibility();
    }

    function showPromptSettings() {
        groupOpen = false;
        settingsOpen = false;
        promptOpen = true;
        toolsOpen = false;
        syncSectionVisibility();
    }

    async function clearPrompt(options = {}) {
        const skipConfirm = options?.skipConfirm === true;
        const canProceed = skipConfirm ? true : await confirmDiscardChanges();
        if (!canProceed) return false;
        promptNameInput.value = "";
        promptTextArea.value = "";
        loadedPromptText = "";
        pendingThumbnail = null;
        loadedThumbnail = null;
        loadedImageLora = "";
        loadedImageLoraStrength = 1.0;
        loadedVideoLora = "";
        loadedVideoLoraStrength = 1.0;
        loadedRefMod = "";
        loadedRefModWeight = 1.0;
        imageLoraStrengthInput.value = "1";
        videoLoraStrengthInput.value = "1";
        refModWeightInput.value = "1";
        if (isComposerSource) {
            await refreshComposerAssetChoices({ loraImage: "", loraVideo: "", refmod: "" });
        }
        updateThumbnailDisplay(null);
        currentPromptName = "";
        updateEditorActionButtons();
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
        }, "Bulk Prompt Importer");

        const subtext = el("p", {
            margin: "0",
            color: STYLE.textMuted,
            fontSize: "12px",
        }, `Quickly create multiple prompts in "${category}".`);

        const textarea = createTextarea("", "Bomber Jacket - a cropped bomber jacket over a simple shirt");
        textarea.style.minHeight = "auto";
        textarea.style.height = "auto";
        textarea.style.resize = "vertical";
        textarea.rows = 20;

        const formatHint = el("p", {
            margin: "0",
            color: STYLE.textMuted,
            fontSize: "12px",
            lineHeight: "1.45",
            whiteSpace: "pre-line",
        }, "Each line must be: Title - Prompt text");

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
            let invalid = 0;
            const seen = new Set();
            let firstCreatedName = "";

            for (const line of lines) {
                const separatorIndex = line.indexOf("-");
                if (separatorIndex <= 0) {
                    invalid++;
                    continue;
                }
                const name = line.substring(0, separatorIndex).trim();
                const text = line.substring(separatorIndex + 1).trim();
                if (!name || !text) {
                    invalid++;
                    continue;
                }

                const key = name.toLowerCase();
                if (seen.has(key)) continue;
                seen.add(key);
                try {
                    const result = await _savePrompt({ category, name, text, thumbnail: null });
                    if (result?.success) {
                        created++;
                        if (!firstCreatedName) firstCreatedName = name;
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
            if (created > 0 && firstCreatedName) {
                _selectPrompt?.(category, firstCreatedName);
            }
            await _showInfo(
                "Prompts Created",
                `${created} prompt(s) created in "${category}".${failed ? ` ${failed} failed.` : ""}${invalid ? ` ${invalid} line(s) were skipped because they were not in the format \"Title - Prompt text\".` : ""}`
            );
        }, { background: "#2b6d3a", borderColor: "#4a9158", color: "#fff" });

        buttonRow.appendChild(cancelBtn);
        buttonRow.appendChild(createBtn);

        dialog.appendChild(title);
        dialog.appendChild(subtext);
        dialog.appendChild(formatHint);
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

    async function importAssetsViaBrowser(config) {
        const category = currentCategory;
        if (!category) {
            await _showInfo("Missing Category", "Please select a category first.");
            return;
        }

        const canProceed = await confirmDiscardChanges();
        if (!canProceed) return;

        const rootPath = await config.getRootPath();
        if (!rootPath) {
            await _showInfo("Browser Unavailable", config.browserUnavailableMessage || "Unable to open asset browser.");
            return;
        }

        const selectedAssets = await showVisualAssetImportPicker({
            title: config.dialogTitle,
            rootPath,
            loadFolder: (path) => config.loadFolder(path, rootPath),
            emptyMessage: config.emptyMessage,
            createButtonLabel: config.createButtonLabel,
        });

        if (!Array.isArray(selectedAssets) || selectedAssets.length === 0) return;

        let created = 0;
        let failed = 0;
        let firstCreatedName = "";
        const seen = new Set();

        for (const asset of selectedAssets) {
            const promptName = String(asset?.promptName || "").trim();
            if (!promptName) continue;
            const key = promptName.toLowerCase();
            if (seen.has(key)) continue;
            seen.add(key);
            try {
                const thumbnail = await previewCandidatesToThumbnail(asset.previewCandidates);
                const payload = config.buildSavePayload(asset, thumbnail, category);
                const result = await _savePrompt(payload);
                if (result?.success) {
                    created++;
                    if (!firstCreatedName) firstCreatedName = promptName;
                } else {
                    failed++;
                }
            } catch (err) {
                console.error("[PromptBrowserEdit] Asset import failed:", err);
                failed++;
            }
        }

        await _loadPrompts(node);
        _onChange();
        if (created > 0 && firstCreatedName) {
            _selectPrompt?.(category, firstCreatedName);
        }
        await _showInfo(
            config.resultTitle || "Import Complete",
            `${created} prompt(s) created in "${category}".${failed ? ` ${failed} failed.` : ""}`
        );
    }

    const importLorasBtn = createButton("Import LoRAs", async () => {
        await importAssetsViaBrowser({
            dialogTitle: "Import LoRAs As Prompts",
            getRootPath: fetchLoraBrowserRoot,
            loadFolder: listLoraBrowserFolder,
            emptyMessage: "No LoRAs found in this folder.",
            createButtonLabel: "Create Prompts",
            browserUnavailableMessage: "The FBnodes LoRA browser routes are unavailable.",
            resultTitle: "LoRAs Imported",
            buildSavePayload: (asset, thumbnail, category) => ({
                category,
                name: asset.promptName,
                text: asset.promptText,
                thumbnail,
                lora_image: String(asset.assetValue || "").trim(),
                lora_image_strength: 1.0,
                lora_video: String(asset.assetValue || "").trim(),
                lora_video_strength: 1.0,
                refmod: "",
                refmod_weight: 1.0,
            }),
        });
    });
    toolsBody.appendChild(importLorasBtn);

    const importRefModsBtn = createButton("Import RefMods", async () => {
        await importAssetsViaBrowser({
            dialogTitle: "Import RefMods As Prompts",
            getRootPath: async () => {
                try {
                    const listing = await listRefModBrowserFolder("");
                    return listing.root || "";
                } catch {
                    return "";
                }
            },
            loadFolder: (path) => listRefModBrowserFolder(path),
            emptyMessage: "No RefMods found in this folder.",
            createButtonLabel: "Create Prompts",
            browserUnavailableMessage: "The H3 RefMod browser routes are unavailable.",
            resultTitle: "RefMods Imported",
            buildSavePayload: (asset, thumbnail, category) => ({
                category,
                name: asset.promptName,
                text: asset.promptText,
                thumbnail,
                refmod: String(asset.assetValue || "").trim(),
                refmod_weight: 1.0,
            }),
        });
    });
    toolsBody.appendChild(importRefModsBtn);

    // Initialize the thumbnail area with the placeholder so it never starts empty.
    updateThumbnailDisplay(null);
    updateEditorActionButtons();
    if (isComposerSource) {
        void refreshComposerAssetChoices();
        loadTypeSettings("");
    }

    return {
        element: root,
        loadPrompt,
        loadTypeSettings,
        loadCategorySettings,
        showTypeSettings,
        showCategorySettings,
        showPromptSettings,
        refreshThumbnailModelButton,
        clearPrompt,
        confirmDiscardChanges,
        getCurrentPromptName: () => String(promptNameInput.value || "").trim(),
    };
}
