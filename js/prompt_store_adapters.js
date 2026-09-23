export const PROMPT_ENDPOINT_PREFIX = "/prompt-manager";
export const PROMPT_ADVANCED_ENDPOINT_PREFIX = "/prompt-manager-advanced";
export const SYSTEM_PROMPTS_ENDPOINT_PREFIX = "/prompt-generator";
export const COMPOSER_ENDPOINT_PREFIX = "/prompt-manager/compose";

export const SOURCE_COMPOSE = "Compose Data";
export const SOURCE_PROMPT = "Prompt Data";
export const SOURCE_SYSTEM_PROMPTS = "System Prompts";

export function isHiddenPromptEntryKey(name) {
    const normalized = String(name || "").trim().toLowerCase();
    return normalized === "__meta__" || normalized === "_base_prompt_" || normalized === "_prompt_prefix_" || normalized === "_prompt_type_" || normalized === "_prompts_";
}

export function getPromptStoreKindFromSource(source) {
    if (source === SOURCE_COMPOSE) return "composer";
    if (source === SOURCE_PROMPT) return "manager";
    return "generator";
}

export function getPromptStoreKindFromEndpoint(endpointPrefix) {
    if (endpointPrefix === COMPOSER_ENDPOINT_PREFIX) return "composer";
    if (endpointPrefix === SYSTEM_PROMPTS_ENDPOINT_PREFIX) return "generator";
    return "manager";
}

export function getEndpointPrefixForSource(source) {
    if (source === SOURCE_PROMPT) return PROMPT_ENDPOINT_PREFIX;
    if (source === SOURCE_SYSTEM_PROMPTS) return SYSTEM_PROMPTS_ENDPOINT_PREFIX;
    return COMPOSER_ENDPOINT_PREFIX;
}

export function getCategoryPromptEntries(categoryData, storeKind = "manager") {
    if (!categoryData || typeof categoryData !== "object" || Array.isArray(categoryData)) return {};
    if (storeKind === "composer") {
        const nested = categoryData._prompts_;
        if (nested && typeof nested === "object" && !Array.isArray(nested)) {
            return nested;
        }
    }
    return Object.fromEntries(
        Object.entries(categoryData).filter(([name]) => !isHiddenPromptEntryKey(name))
    );
}

export function getCategoryPromptEntriesForSource(categoryData, source) {
    return getCategoryPromptEntries(categoryData, getPromptStoreKindFromSource(source));
}

export function getCategoryPromptEntriesForEndpoint(categoryData, endpointPrefix) {
    return getCategoryPromptEntries(categoryData, getPromptStoreKindFromEndpoint(endpointPrefix));
}

export function getCategoryPromptEntryForEndpoint(categoryData, promptName, endpointPrefix) {
    return getCategoryPromptEntriesForEndpoint(categoryData, endpointPrefix)?.[promptName] || null;
}

export function buildSavePromptRequestBody(storeKind, payload) {
    const body = {
        category: payload?.category,
        name: payload?.name,
        text: payload?.text,
    };

    if (storeKind === "generator") {
        if (Object.prototype.hasOwnProperty.call(payload || {}, "thumbnail")) body.thumbnail = payload.thumbnail;
        if (Object.prototype.hasOwnProperty.call(payload || {}, "prompt_category")) body.prompt_category = payload.prompt_category;
        return body;
    }

    if (storeKind === "composer") {
        if (Object.prototype.hasOwnProperty.call(payload || {}, "thumbnail")) body.thumbnail = payload.thumbnail;
        if (Object.prototype.hasOwnProperty.call(payload || {}, "prompt_category")) body.prompt_category = payload.prompt_category;
        if (Object.prototype.hasOwnProperty.call(payload || {}, "lora")) body.lora = payload.lora;
        if (Object.prototype.hasOwnProperty.call(payload || {}, "lora_strength")) body.lora_strength = payload.lora_strength;
        if (Object.prototype.hasOwnProperty.call(payload || {}, "lora_image")) body.lora_image = payload.lora_image;
        if (Object.prototype.hasOwnProperty.call(payload || {}, "lora_image_strength")) body.lora_image_strength = payload.lora_image_strength;
        if (Object.prototype.hasOwnProperty.call(payload || {}, "lora_video")) body.lora_video = payload.lora_video;
        if (Object.prototype.hasOwnProperty.call(payload || {}, "lora_video_strength")) body.lora_video_strength = payload.lora_video_strength;
        if (Object.prototype.hasOwnProperty.call(payload || {}, "refmod")) body.refmod = payload.refmod;
        if (Object.prototype.hasOwnProperty.call(payload || {}, "refmod_weight")) body.refmod_weight = payload.refmod_weight;
        return body;
    }

    if (Object.prototype.hasOwnProperty.call(payload || {}, "thumbnail")) body.thumbnail = payload.thumbnail;
    if (Object.prototype.hasOwnProperty.call(payload || {}, "workflow_data")) body.workflow_data = payload.workflow_data;
    if (Object.prototype.hasOwnProperty.call(payload || {}, "negative_prompt")) body.negative_prompt = payload.negative_prompt;
    if (Object.prototype.hasOwnProperty.call(payload || {}, "loras_a")) body.loras_a = payload.loras_a;
    if (Object.prototype.hasOwnProperty.call(payload || {}, "loras_b")) body.loras_b = payload.loras_b;
    if (Object.prototype.hasOwnProperty.call(payload || {}, "loras_c")) body.loras_c = payload.loras_c;
    if (Object.prototype.hasOwnProperty.call(payload || {}, "loras_d")) body.loras_d = payload.loras_d;
    if (Object.prototype.hasOwnProperty.call(payload || {}, "trigger_words")) body.trigger_words = payload.trigger_words;
    return body;
}

export function buildSavePromptRequestBodyForSource(source, payload) {
    return buildSavePromptRequestBody(getPromptStoreKindFromSource(source), payload);
}

export function buildSavePromptRequestBodyForEndpoint(endpointPrefix, payload) {
    return buildSavePromptRequestBody(getPromptStoreKindFromEndpoint(endpointPrefix), payload);
}