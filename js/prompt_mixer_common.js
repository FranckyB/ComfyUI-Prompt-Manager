import { api } from "../../scripts/api.js";

const MIXER_ENDPOINT_PREFIX = "/prompt-manager/mixer";

export async function loadMixerPrompts(node) {
    try {
        const resp = await fetch(`${MIXER_ENDPOINT_PREFIX}/get-prompts`);
        node.mixerPrompts = await resp.json();
        // Make the shared browser see mixer data instead of prompt_manager_data.json.
        node.prompts = node.mixerPrompts;
    } catch (err) {
        console.error("[PromptMixer] Error loading mixer prompts:", err);
        node.mixerPrompts = {};
        node.prompts = {};
    }
    return node.mixerPrompts;
}

function getMixerData(node) {
    // The shared browser mutates node.prompts; for mixer nodes that is always mixer data.
    return node.prompts || node.mixerPrompts || {};
}

export function getMixerCategories(node) {
    const data = getMixerData(node);
    return Object.keys(data).filter((c) => c !== "__meta__").sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
}

export function getMixerNames(node, category) {
    const data = getMixerData(node);
    const catData = data[category];
    if (!catData || typeof catData !== "object") return [];
    return Object.keys(catData).filter((n) => n !== "__meta__").sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
}

export function getMixerEntry(node, category, name) {
    const data = getMixerData(node);
    const catData = data[category];
    if (!catData || typeof catData !== "object") return null;
    return catData[name] || null;
}

export { MIXER_ENDPOINT_PREFIX };
