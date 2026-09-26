import { api } from "../../scripts/api.js";
import { COMPOSER_ENDPOINT_PREFIX, getCategoryPromptEntries } from "./prompt_store_adapters.js";

function normalizeComposerLibrary(library) {
    if (!library || typeof library !== "object" || Array.isArray(library)) {
        return { __meta__: { schema_version: 2, storage: "type_files" }, _types_: {} };
    }
    if (library._types_ && typeof library._types_ === "object" && !Array.isArray(library._types_)) {
        return library;
    }
    return { __meta__: { schema_version: 1, storage: "legacy_flat" }, _types_: {} };
}

function getComposerTypeOrder(typeData) {
    const numeric = Number(typeData?.order);
    return Number.isInteger(numeric) ? numeric : Number.POSITIVE_INFINITY;
}

function getOrderedComposerTypes(types) {
    const entries = Object.entries(types || {});
    const hasExplicitOrder = entries.some(([, typeData]) => Number.isInteger(Number(typeData?.order)));
    if (!hasExplicitOrder) {
        return entries.sort((a, b) => String(a[0] || "").localeCompare(String(b[0] || ""), undefined, { sensitivity: "base" }));
    }
    return entries.sort((a, b) => {
        const orderDiff = getComposerTypeOrder(a[1]) - getComposerTypeOrder(b[1]);
        if (orderDiff !== 0) return orderDiff;
        return String(a[0] || "").localeCompare(String(b[0] || ""), undefined, { sensitivity: "base" });
    });
}

function flattenComposerType(typeFile, typeData, flatData) {
    if (!typeData || typeof typeData !== "object") return;
    const typeKey = String(typeFile || "").replace(/\.json$/i, "").trim().toLowerCase();
    const typeName = String(typeData.name || "").trim() || typeKey || "Misc";
    const typePrefix = String(typeData.prefix || "");
    const typeBasePrompt = String(typeData.base_prompt || "");
    const typeSubjectType = String(typeData.subject_type || "subject").trim().toLowerCase() || "subject";
    const typeNsfw = typeData.nsfw === true;
    const categories = typeData.categories;
    if (!categories || typeof categories !== "object" || Array.isArray(categories)) return;

    for (const [categoryName, categoryData] of Object.entries(categories)) {
        if (!categoryName || !categoryData || typeof categoryData !== "object") continue;
        const promptEntries = categoryData._prompts_ && typeof categoryData._prompts_ === "object"
            ? categoryData._prompts_
            : {};
        const effectivePrefix = String(categoryData.prefix || typePrefix || "");
        const effectiveBasePrompt = String(categoryData.base_prompt || typeBasePrompt || "");
        const effectiveNsfw = categoryData.nsfw === true || typeNsfw;
        const flatCategory = {
            _prompts_: promptEntries,
            _prompt_type_: typeKey,
            _type_file_: String(typeFile || ""),
            _type_name_: typeName,
            _subject_type_: typeSubjectType,
        };
        if (effectivePrefix.trim()) flatCategory._prompt_prefix_ = effectivePrefix;
        if (effectiveBasePrompt.trim()) flatCategory._base_prompt_ = effectiveBasePrompt;
        if (String(categoryData.prefix || "").trim()) flatCategory._category_prefix_ = String(categoryData.prefix || "");
        if (String(categoryData.base_prompt || "").trim()) flatCategory._category_base_prompt_ = String(categoryData.base_prompt || "");
        if (typePrefix.trim()) flatCategory._type_prefix_ = typePrefix;
        if (typeBasePrompt.trim()) flatCategory._type_base_prompt_ = typeBasePrompt;
        if (typeNsfw) flatCategory._type_nsfw_ = true;
        if (effectiveNsfw) flatCategory.__meta__ = { nsfw: true };
        flatData[categoryName] = flatCategory;
    }
}

export function flattenComposerLibrary(library) {
    const normalized = normalizeComposerLibrary(library);
    if (!Object.keys(normalized._types_ || {}).length) {
        if (!library || typeof library !== "object" || Array.isArray(library)) return {};
        if (!library._types_) return library;
    }

    const flatData = { __meta__: normalized.__meta__ || { schema_version: 2, storage: "type_files" } };
    for (const [typeFile, typeData] of getOrderedComposerTypes(normalized._types_ || {})) {
        flattenComposerType(typeFile, typeData, flatData);
    }
    return flatData;
}

export async function loadComposerPrompts(node) {
    try {
        const resp = await fetch(`${COMPOSER_ENDPOINT_PREFIX}/get-prompts`, {
            cache: "no-store",
        });
        const library = await resp.json();
        node.composerPromptLibrary = normalizeComposerLibrary(library);
        node.composerPrompts = flattenComposerLibrary(node.composerPromptLibrary);
        // Make the shared browser see composer data instead of prompt_manager_data.json.
        node.prompts = node.composerPrompts;
    } catch (err) {
        console.error("[PromptComposer] Error loading composer prompts:", err);
        node.composerPromptLibrary = { __meta__: { schema_version: 2, storage: "type_files" }, _types_: {} };
        node.composerPrompts = {};
        node.prompts = {};
    }
    return node.composerPrompts;
}

function getComposerData(node) {
    // The shared browser mutates node.prompts; for composer nodes that is always composer data.
    return node.prompts || node.composerPrompts || {};
}

export function getComposerCategories(node) {
    const data = getComposerData(node);
    return Object.keys(data).filter((c) => c !== "__meta__").sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
}

export function getComposerNames(node, category) {
    const data = getComposerData(node);
    const catData = data[category];
    if (!catData || typeof catData !== "object") return [];
    return Object.keys(getCategoryPromptEntries(catData, "composer")).sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
}

export function getComposerEntry(node, category, name) {
    const data = getComposerData(node);
    const catData = data[category];
    if (!catData || typeof catData !== "object") return null;
    return getCategoryPromptEntries(catData, "composer")[name] || null;
}

export async function saveComposerCategorySettings(category, settings) {
    try {
        const resp = await fetch(`${COMPOSER_ENDPOINT_PREFIX}/save-category-settings`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                category,
                base_prompt: settings.basePrompt || "",
                prompt_type: settings.promptType || "",
                prefix: settings.promptPrefix || "",
            }),
        });
        return await resp.json();
    } catch (err) {
        console.error("[PromptComposer] Error saving category settings:", err);
        return { success: false, error: String(err) };
    }
}

export async function saveComposerTypeSettings(settings) {
    try {
        const resp = await fetch(`${COMPOSER_ENDPOINT_PREFIX}/save-type-settings`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                type_file: settings.typeFile || "",
                prompt_type: settings.promptType || "",
                type_name: settings.typeName || "",
                prefix: settings.promptPrefix || "",
                base_prompt: settings.basePrompt || "",
            }),
        });
        return await resp.json();
    } catch (err) {
        console.error("[PromptComposer] Error saving type settings:", err);
        return { success: false, error: String(err) };
    }
}

export { COMPOSER_ENDPOINT_PREFIX };
