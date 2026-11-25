import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { mountCard } from "./fbui/card.js";
import { toast } from "./fbui/comfy.js";
import { button, helpBadge, ICONS } from "./fbui/controls.js";
import { chainCallback, el } from "./fbui/dom.js";
import { FONT, ROW_HEIGHT } from "./fbui/theme.js";
import { findWidget, hideInputsInDef, hideWidget } from "./fbui/widgets.js";
import { showThumbnailBrowser } from "./prompt_browser.js";
import { SYSTEM_PROMPTS_ENDPOINT_PREFIX, getCategoryPromptEntries } from "./prompt_store_adapters.js";

const NODE_NAME = "PromptGenerator";
const MIN_WIDTH = 380;
const TEXT_MIN_HEIGHT = 90;
const CSS_ID = "pg-css-v1";
const TWO = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr))";

const ENGINE_LABELS = { llama: "Llama", native: "Native" };
const ENGINE_TITLES = {
    llama: "A llama.cpp or Ollama server runs a GGUF model (Prompt Manager settings choose which).",
    native: "ComfyUI runs a .safetensors text encoder from models/text_encoders/LLM itself, as a connected CLIP would.",
};

const GUIDE = [
    "Pick a system prompt with ◀ ▶ or click its name to browse them.",
    "Write a prompt, connect an image, or both.",
    ["Llama", "a llama.cpp or Ollama server runs a GGUF model."],
    ["Native", "ComfyUI runs a .safetensors text encoder from models/text_encoders/LLM (the folder is a Prompt Manager setting)."],
    ["Unload after", "stop the server, or drop the native model, when the run ends."],
    ["Clear VRAM", "free VRAM before and after a Llama run."],
];

function ensureCss() {
    if (document.getElementById(CSS_ID)) return;
    const style = document.createElement("style");
    style.id = CSS_ID;
    style.textContent = `
.pg-name{flex:1 1 0;min-width:0;justify-content:flex-start;font:500 12px ${FONT}}
.pg-name span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pg-text.fbui-text{width:100%;height:100%;min-height:${TEXT_MIN_HEIGHT}px;padding:6px 8px;box-sizing:border-box;resize:none;pointer-events:auto}
`;
    document.head.append(style);
}

// ---------------------------------------------------------------- system prompt

const getChoices = (prompts) => Object.entries(prompts)
    .filter(([category, entries]) => !category.startsWith("_") && entries && typeof entries === "object")
    .flatMap(([category, entries]) => Object.entries(getCategoryPromptEntries(entries, "generator"))
        .filter(([name, entry]) => {
            const text = entry && typeof entry === "object" ? entry.prompt : entry;
            return !name.startsWith("_") && String(text ?? "").trim();
        })
        .map(([name]) => `${category} / ${name}`))
    .sort((left, right) => left.toLowerCase().localeCompare(right.toLowerCase()));

const matchChoice = (value, choices, allowFallback) => {
    const normalized = String(value ?? "").trim().toLowerCase();
    const entries = choices.map(choice => {
        const separator = choice.indexOf(" / ");
        return { value: choice, category: choice.slice(0, separator), name: choice.slice(separator + 3) };
    });
    const exact = entries.find(entry => entry.value.toLowerCase() === normalized);
    const legacy = entries.find(entry => `${entry.name} (${entry.category})`.toLowerCase() === normalized);
    // A prompt moved to another category keeps its name.
    const previousName = normalized.includes(" / ") ? normalized.slice(normalized.indexOf(" / ") + 3) : normalized;
    const names = entries.filter(entry => entry.name.toLowerCase() === previousName);
    const match = exact || legacy || (names.length === 1 ? names[0] : null);
    if (match || !allowFallback) return match;
    return entries.find(entry => entry.value.toLowerCase() === "image / enhance prompt") || entries[0] || null;
};

async function loadSystemPrompts(node) {
    const response = await api.fetchApi(`${SYSTEM_PROMPTS_ENDPOINT_PREFIX}/get-prompts`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Could not load system prompts (${response.status})`);
    const prompts = await response.json();
    node.prompts = prompts;
    // The combo list comes from page load; keep it current with prompts added or moved since.
    const widget = findWidget(node, "system_prompt");
    const choices = getChoices(prompts);
    if (choices.length && widget?.options && typeof widget.options.values !== "function") widget.options.values = choices;
    return prompts;
}

async function pickSystemPrompt(node, direction) {
    const pg = node._pg;
    if (pg.busy) return;
    pg.busy = true;
    try {
        const value = String(findWidget(node, "system_prompt")?.value ?? "");
        if (direction) {
            const choices = getChoices(await loadSystemPrompts(node));
            if (!choices.length) return;
            const index = choices.indexOf(value);
            const nextIndex = index < 0 ? (direction > 0 ? 0 : choices.length - 1) : (index + direction + choices.length) % choices.length;
            pg.card.setWidget("system_prompt", choices[nextIndex]);
        } else {
            const separator = value.indexOf(" / ");
            const selection = await showThumbnailBrowser(node, separator < 0 ? "" : value.slice(0, separator), separator < 0 ? value : value.slice(separator + 3), {
                title: "Select System Prompt",
                endpointPrefix: SYSTEM_PROMPTS_ENDPOINT_PREFIX,
                loadPromptsFn: () => loadSystemPrompts(node),
                preferenceScope: "system",
                promptOnly: true,
                allowMultiSelect: false,
            });
            if (selection?.prompt) pg.card.setWidget("system_prompt", `${selection.category} / ${selection.prompt}`);
        }
    } catch (error) {
        console.error("[PromptGenerator] System prompt selection failed:", error);
        toast(String(error.message || error), { severity: "error", summary: "System prompts unavailable" });
    } finally {
        pg.busy = false;
    }
}

// A saved name that no longer matches (a legacy label, a prompt moved to
// another category) becomes its current "Category / Name".
function migrateSystemPrompt(node, choices, allowFallback) {
    const widget = findWidget(node, "system_prompt");
    const match = widget && choices.length ? matchChoice(widget.value, choices, allowFallback) : null;
    if (!match || widget.value === match.value) return;
    console.log(`[PromptGenerator] Migrated system prompt '${widget.value}' to '${match.value}'.`);
    widget.value = match.value;
    node._pg.card.refresh();
}

function systemPromptBar(node) {
    ensureCss();
    // The system prompt as a button that opens the browser, ‹ › stepping through them.
    const bar = el("div", "fbui-field");
    const label = el("span");
    const name = button("", { primary: true, onClick: () => pickSystemPrompt(node, 0) });
    name.classList.add("pg-name");
    name.append(label);
    const back = button("", { icon: ICONS.chevronLeft, title: "Previous system prompt", onClick: () => pickSystemPrompt(node, -1) });
    const next = button("", { icon: ICONS.chevronRight, title: "Next system prompt", onClick: () => pickSystemPrompt(node, 1) });
    bar.append(name, back, next);
    return {
        element: bar,
        sync(values) {
            label.textContent = String(values.system_prompt ?? "").replace(" / ", " : ");
            name.title = `System prompt: ${values.system_prompt ?? ""}\nClick to browse the system prompts`;
        },
    };
}

// A multiline widget as the card's text area.
function textPanel(name, placeholder) {
    return (node, card) => {
        const area = el("textarea", "fbui-text pg-text");
        area.spellcheck = false;
        area.addEventListener("input", () => card.setWidget(name, area.value, { settle: false }));
        area.addEventListener("change", () => card.setWidget(name, area.value));
        // Typing and selecting here must not drag the node or reach the canvas.
        area.addEventListener("pointerdown", (event) => event.stopPropagation());
        area.addEventListener("keydown", (event) => event.stopPropagation());
        return {
            element: area,
            sync(values) {
                area.placeholder = typeof placeholder === "function" ? placeholder(values) : placeholder;
                const shown = String(values[name] ?? "");
                if (area.value !== shown && document.activeElement !== area) area.value = shown;
            },
        };
    };
}

// ---------------------------------------------------------------- chips

// A chip for a BOOLEAN widget: lit (the bright blue of an enabled choice)
// while on, greyed while `active()` says it has no effect.
function chip(node, card, name, { text, title, active = () => true }) {
    const element = button(text, { grow: true, title, onClick: () => card.setWidget(name, !findWidget(node, name)?.value) });
    return {
        element,
        sync() {
            const live = active();
            element.disabled = !live;
            element.classList.toggle("primary", live && Boolean(findWidget(node, name)?.value));
        },
    };
}

function chipPair(...chips) {
    return (node, card) => {
        const built = chips.map(([name, options]) => chip(node, card, name, options(node)));
        const field = el("div", "fbui-field");
        field.style.cssText = TWO;
        field.append(...built.map((item) => item.element));
        return { element: field, sync: () => built.forEach((item) => item.sync()) };
    };
}

// A model by its file name alone: no folder, no extension.
const modelLabel = (value) => String(value).split(/[\\/]/).pop().replace(/\.(gguf|safetensors)$/i, "");

const isNative = (node) => findWidget(node, "engine")?.value === "native";

// ---------------------------------------------------------------- node

function setupNode(node) {
    ensureCss();
    const pg = { busy: false };
    node._pg = pg;
    for (const name of ["system_prompt", "prompt"]) hideWidget(findWidget(node, name));
    pg.card = mountCard(node, {
        minWidth: MIN_WIDTH,
        rows: [
            // Set and forget: folded under Options.
            { group: "options", label: "Options" },
            { group: "options", widget: "engine", label: "Engine", kind: "segment", labels: ENGINE_LABELS, titles: ENGINE_TITLES },
            { group: "options", widget: "model", label: "Model", kind: "select", labels: modelLabel, when: (values) => values.engine !== "native" },
            { group: "options", widget: "text_encoder", label: "Model", kind: "select", labels: modelLabel, when: (values) => values.engine === "native" },
            {
                group: "options",
                label: "Output",
                widgets: ["format_as_json", "enable_thinking"],
                control: chipPair(
                    ["format_as_json", () => ({ text: "JSON", title: "Format the output as structured JSON with a scene breakdown" })],
                    ["enable_thinking", () => ({ text: "Thinking", title: "Let a model that can reason think first; its thoughts go to the thoughts output" })],
                ),
            },
            {
                group: "options",
                label: "Memory",
                widgets: ["stop_server_after", "clear_vram_on_run"],
                control: chipPair(
                    ["stop_server_after", () => ({ text: "Unload after", title: "Stop the llama server, or drop the native model, when the run ends" })],
                    ["clear_vram_on_run", (n) => ({ text: "Clear VRAM", title: "Free VRAM before and after a Llama run", active: () => !isNative(n) })],
                ),
            },
            { group: "options", pair: ["seed", "control_after_generate"], label: "Seed" },
            { panel: systemPromptBar, height: ROW_HEIGHT },
            { panel: textPanel("prompt", "Enter prompt…"), grow: true, height: TEXT_MIN_HEIGHT },
        ],
    });
    // The bar is a panel, so a new pick redraws the card here.
    chainCallback(findWidget(node, "system_prompt"), "callback", () => pg.card.refresh());
    helpBadge(node, GUIDE);
    chainCallback(node, "onConfigure", () => {
        // Legacy labels such as "Enhance Prompt (Image)" migrate at once so an immediate queue never sends them.
        const widget = findWidget(node, "system_prompt");
        if (!String(widget?.value ?? "").includes(" / ")) {
            const values = widget?.options?.values;
            migrateSystemPrompt(node, Array.isArray(values) ? values : [], false);
        }
        // The combo's choices date from page load, so the value is checked against the live list.
        loadSystemPrompts(node)
            .then((prompts) => migrateSystemPrompt(node, getChoices(prompts), true))
            .catch((error) => console.warn("[PromptGenerator] Could not refresh system prompts; keeping the saved selection.", error));
    });
}

// ---------------------------------------------------------------- options node

const OPTIONS_NAME = "PromptGenOptions";
const OPTIONS_WIDGETS = ["model", "system_prompt_mode", "system_prompt", "use_model_default_sampling", "temperature", "top_k", "top_p", "min_p", "repeat_penalty", "context_size", "max_length", "gpu_device", "show_everything_in_console"];
const OPTIONS_GUIDE = [
    "Connect it to the Prompt Generator's options input.",
    ["Instructions", "replace the system prompt with the text below, or add it after."],
    ["Model sampling", "on: the model's own settings; off: the ones shown below it."],
    ["Llama", "context size and GPU for llama.cpp."],
    ["Native", "the longest answer, in tokens, for a native text encoder."],
    "image2 … image5 add pictures after the Prompt Generator's images.",
];
const custom = (values) => values.use_model_default_sampling === false;

function setupOptionsNode(node) {
    ensureCss();
    // The model moved to the Prompt Generator; old workflows still pass theirs through.
    for (const name of ["model", "system_prompt"]) hideWidget(findWidget(node, name));
    mountCard(node, {
        minWidth: 320,
        rows: [
            { widget: "system_prompt_mode", label: "Instructions", kind: "segment", labels: { replace: "Replace", append: "Append" } },
            { panel: textPanel("system_prompt", (values) => (values.system_prompt_mode === "append" ? "Added after the system prompt…" : "Replaces the system prompt…")), grow: true, height: TEXT_MIN_HEIGHT },
            { widget: "use_model_default_sampling", label: "Model sampling" },
            { widget: "temperature", label: "Temperature", when: custom },
            { pair: ["top_k", "top_p"], label: "Top K · P", when: custom },
            { pair: ["min_p", "repeat_penalty"], label: "Min P · Rep", when: custom },
            { section: "Llama" },
            { widget: "context_size", label: "Context" },
            { widget: "gpu_device", label: "GPU", kind: "text", placeholder: "system default" },
            { section: "Native" },
            { widget: "max_length", label: "Max length" },
            { section: "Console" },
            { widget: "show_everything_in_console", label: "Show all" },
        ],
    });
    helpBadge(node, OPTIONS_GUIDE);
}

function migrateLegacyPromptGenOptionsWidgets(node) {
    if (!node?.widgets || node.widgets.length === 0) {
        return false;
    }

    const byName = (name) => node.widgets.find(w => w.name === name);
    const modeWidget = byName("system_prompt_mode");
    const systemPromptWidget = byName("system_prompt");
    const useDefaultWidget = byName("use_model_default_sampling");
    const temperatureWidget = byName("temperature");
    const topKWidget = byName("top_k");
    const topPWidget = byName("top_p");
    const minPWidget = byName("min_p");
    const repeatPenaltyWidget = byName("repeat_penalty");
    const contextSizeWidget = byName("context_size");
    const showConsoleWidget = byName("show_everything_in_console");

    if (!modeWidget || !systemPromptWidget || !useDefaultWidget || !temperatureWidget ||
        !topKWidget || !topPWidget || !minPWidget || !repeatPenaltyWidget ||
        !contextSizeWidget || !showConsoleWidget) {
        return false;
    }

    const modeIsValid = modeWidget.value === "replace" || modeWidget.value === "append";
    const looksLegacyShape = typeof systemPromptWidget.value === "boolean" && typeof useDefaultWidget.value === "number";

    if (modeIsValid && !looksLegacyShape) {
        return false;
    }

    const oldSystemPrompt = modeWidget.value;
    const oldUseDefaultSampling = systemPromptWidget.value;
    const oldTemperature = useDefaultWidget.value;
    const oldTopK = temperatureWidget.value;
    const oldTopP = topPWidget.value;
    const oldMinP = topPWidget.value;
    const oldRepeatPenalty = repeatPenaltyWidget.value;
    const oldContextSize = contextSizeWidget.value;
    const oldShowInConsole = showConsoleWidget.value;

    modeWidget.value = "replace";
    systemPromptWidget.value = typeof oldSystemPrompt === "string" ? oldSystemPrompt : "";
    useDefaultWidget.value = Boolean(oldUseDefaultSampling);
    temperatureWidget.value = Number(oldTemperature);
    topKWidget.value = Number.isFinite(Number(oldTopK)) ? Math.round(Number(oldTopK)) : oldTopK;
    topPWidget.value = Number(oldTopP);
    minPWidget.value = Number(oldMinP);
    repeatPenaltyWidget.value = Number(oldRepeatPenalty);
    contextSizeWidget.value = Number.isFinite(Number(oldContextSize)) ? Math.round(Number(oldContextSize)) : oldContextSize;
    showConsoleWidget.value = Boolean(oldShowInConsole);

    node.serialize_widgets = true;
    app.graph.setDirtyCanvas(true, true);
    console.log("[PromptManager] Migrated legacy PromptGenOptions widget order for workflow compatibility.");
    return true;
}

app.registerExtension({
    name: "comfyui-prompt-manager.generator",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === OPTIONS_NAME) {
            hideInputsInDef(nodeData, OPTIONS_WIDGETS);
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                migrateLegacyPromptGenOptionsWidgets(this);
                setupOptionsNode(this);
                this.setSize([380, 460]);
            });
            chainCallback(nodeType.prototype, "onConfigure", function () {
                migrateLegacyPromptGenOptionsWidgets(this);
            });
        }

        if (nodeData.name === NODE_NAME) {
            hideInputsInDef(nodeData, ["system_prompt", "prompt", "engine", "model", "text_encoder", "format_as_json", "enable_thinking", "stop_server_after", "clear_vram_on_run", "seed"]);
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                setupNode(this);
                this.setSize([420, 520]);
            });
        }
    }
});

console.log("[PromptManager] PromptGenerator extension loaded");
