/**
 * Recipe Model Picker
 *
 * Keeps Type/Model/VAE/CLIP widgets in sync with Builder family filtering
 * and emits a single model_data payload for Recipe Relay.
 */
import { app } from "../../scripts/app.js";

const DEFAULT_ASSET = "(Default)";
const DEFAULT_FAMILY = "sdxl";

function getWidget(node, name) {
    return node?.widgets?.find((w) => w.name === name) || null;
}

function setWidgetOptions(widget, values, fallback = "") {
    if (!widget?.options) return;
    const list = Array.isArray(values) && values.length > 0 ? values : [fallback];
    widget.options.values = list;
}

function ensureWidgetValue(widget, preferred, fallback = "") {
    if (!widget?.options) return;
    const values = Array.isArray(widget.options.values) ? widget.options.values : [];
    const candidate = (preferred != null && values.includes(preferred))
        ? preferred
        : (values.includes(widget.value) ? widget.value : null);
    widget.value = candidate != null ? candidate : (values[0] ?? fallback);
}

async function fetchJson(url) {
    const response = await fetch(url);
    return response.json();
}

function familyKeyFromTypeValue(typeValue, familiesMap) {
    const raw = String(typeValue || "").trim();
    if (!raw) return DEFAULT_FAMILY;

    if (familiesMap[raw]) return raw;

    for (const [key, label] of Object.entries(familiesMap)) {
        if (label === raw) return key;
    }

    return DEFAULT_FAMILY;
}

async function reloadPickerLists(node) {
    const typeWidget = getWidget(node, "type");
    const modelWidget = getWidget(node, "model");
    const vaeWidget = getWidget(node, "vae");
    const clipWidget = getWidget(node, "clip");
    if (!typeWidget || !modelWidget || !vaeWidget || !clipWidget) return;

    let families = node._recipePickerFamilies;
    if (!families) {
        try {
            const data = await fetchJson("/workflow-extractor/list-families");
            families = data?.families || {};
        } catch {
            families = {};
        }
        node._recipePickerFamilies = families;
    }

    const familyKey = familyKeyFromTypeValue(typeWidget.value, families);

    const [modelsData, vaesData, clipsData] = await Promise.all([
        fetchJson(`/workflow-extractor/list-models?family=${encodeURIComponent(familyKey)}`),
        fetchJson(`/workflow-extractor/list-vaes?family=${encodeURIComponent(familyKey)}`),
        fetchJson(`/workflow-extractor/list-clips?family=${encodeURIComponent(familyKey)}`),
    ]).catch(() => [{ models: [] }, { vaes: [] }, { clips: [] }]);

    const models = Array.isArray(modelsData?.models) ? modelsData.models : [];
    const vaes = Array.isArray(vaesData?.vaes) ? vaesData.vaes : [];
    const clips = Array.isArray(clipsData?.clips) ? clipsData.clips : [];

    const prevModel = modelWidget.value;
    const prevVae = vaeWidget.value;
    const prevClip = clipWidget.value;

    setWidgetOptions(modelWidget, models, "");
    setWidgetOptions(vaeWidget, [DEFAULT_ASSET, ...vaes], DEFAULT_ASSET);
    setWidgetOptions(clipWidget, [DEFAULT_ASSET, ...clips], DEFAULT_ASSET);

    ensureWidgetValue(modelWidget, prevModel, "");
    ensureWidgetValue(vaeWidget, prevVae, DEFAULT_ASSET);
    ensureWidgetValue(clipWidget, prevClip, DEFAULT_ASSET);

    if (typeof modelWidget.callback === "function") modelWidget.callback(modelWidget.value);
    if (typeof vaeWidget.callback === "function") vaeWidget.callback(vaeWidget.value);
    if (typeof clipWidget.callback === "function") clipWidget.callback(clipWidget.value);

    app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "prompt-manager.recipe-model-picker",
    async nodeCreated(node) {
        if (node.comfyClass !== "RecipeModelPicker") return;

        const typeWidget = getWidget(node, "type");
        if (!typeWidget) return;

        const originalTypeCallback = typeWidget.callback;
        typeWidget.callback = async function wrappedTypeCallback(value) {
            if (typeof originalTypeCallback === "function") {
                originalTypeCallback.call(this, value);
            }
            await reloadPickerLists(node);
        };

        requestAnimationFrame(() => {
            reloadPickerLists(node).catch(() => {
                // Keep node usable even if API endpoints are temporarily unavailable.
            });
        });
    },
});
