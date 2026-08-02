import { api } from "../../scripts/api.js";

const COMPOSER_ENDPOINT_PREFIX = "/prompt-manager/mixer";

export async function loadComposerPrompts(node) {
    try {
        const resp = await fetch(`${COMPOSER_ENDPOINT_PREFIX}/get-prompts`);
        node.composerPrompts = await resp.json();
        // Make the shared browser see composer data instead of prompt_manager_data.json.
        node.prompts = node.composerPrompts;
    } catch (err) {
        console.error("[PromptComposer] Error loading composer prompts:", err);
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
    return Object.keys(catData).filter((n) => n !== "__meta__").sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
}

export function getComposerEntry(node, category, name) {
    const data = getComposerData(node);
    const catData = data[category];
    if (!catData || typeof catData !== "object") return null;
    return catData[name] || null;
}

export { COMPOSER_ENDPOINT_PREFIX };
