/**
 * Prompt Manager (PromptManagerV2): a saved prompt with its LoRA stacks and
 * trigger words, on one fbui card with the legacy node's look: the prompt
 * bar, Save / New / More, a stack card per slot in use and the trigger words.
 *
 * The stock widgets stay hidden and hold the saved values: category, name,
 * use_prompt_input, lora_mode, text, and state, the node's edits
 * {loras: {slot: {name: {active, strength, removed}}}, order: {slot: [name]},
 * trigger_words: {text: {active, removed}}} (names in lower case). Before a
 * run the cards show the saved prompt's LoRAs; after one, the rows the node
 * ran with (its UI output, see nodes/prompt_manager_v2.py), connected LoRAs
 * included; those rows are kept in the node's properties, so a loaded
 * workflow shows them too.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { mountCard } from "./fbui/card.js";
import { notifyChange, toast } from "./fbui/comfy.js";
import { button, helpBadge, ICONS } from "./fbui/controls.js";
import { chainCallback, el } from "./fbui/dom.js";
import { confirm, openMenu } from "./fbui/modal.js";
import { FONT, ROW_HEIGHT } from "./fbui/theme.js";
import { findWidget, hideInputsInDef, hideWidget } from "./fbui/widgets.js";
import { DEFAULT_THUMBNAIL, exportPromptsJSON, getLoraManagerPreviewTooltip, PMA_THEME } from "./prompt_manager_advanced.js";
import { getHideNSFW, hasComposeLikePayload, hasLayoutPayload, hasRecipeLikePayload, openPromptBrowserForSave, showThumbnailBrowser } from "./prompt_browser.js";
import { buildSavePromptRequestBodyForEndpoint, isHiddenPromptEntryKey, PROMPT_ADVANCED_ENDPOINT_PREFIX as ENDPOINT } from "./prompt_store_adapters.js";

const NODE_NAME = "PromptManagerV2";
const SLOTS = ["a", "b", "c", "d"];
const MODE_LABELS = { "Prompt LoRAs Only": "Prompt only", "Combine LoRAs": "Combine", "Input LoRAs Only": "Input only" };
const MODE_TITLES = {
    "Prompt LoRAs Only": "The saved prompt's LoRAs only.",
    "Combine LoRAs": "The saved prompt's LoRAs, then the connected ones.",
    "Input LoRAs Only": "The connected LoRAs only.",
};
const PROMPT_ONLY = "Prompt LoRAs Only";
const INPUT_ONLY = "Input LoRAs Only";
const MIN_WIDTH = 460;
const TEXT_MIN_HEIGHT = 90;
// A stack card's padding and title bar, then one 24px line of tags per 28px.
const STACK_HEAD = 50;
const CSS_ID = "pm-v2-css-v2";
// The last run's panel rows, saved with the workflow.
const RUN_PROPERTY = "pm_last_run";

const GUIDE = [
    "Pick a prompt with ◀ ▶ or click its name to browse the library.",
    "LoRA stacks in:",
    ["lora_stacks", "one LoRA stack: connect it here, it is stack A."],
    ["Multi LoRA Stack", "stacks A-D from one node: connect its output here."],
    ["Multi LoRA Relay", "several separate stacks: wire them into its lora_stack_a…d, and its lora_stacks out here."],
    "LoRA stacks out: a plain LoRA stack when one stack has LoRAs, a multi LoRA stack when several do. A node that takes one plain stack (a LoRA loader) fails on a prompt with several; put Multi LoRA Relay in between and use its lora_stack_a…d.",
    "Click a LoRA or trigger word to switch it off or on, its number to set the strength; right-click to move or delete it. Save Prompt keeps these edits with the prompt.",
];

function ensureCss() {
    if (document.getElementById(CSS_ID)) return;
    const style = document.createElement("style");
    style.id = CSS_ID;
    style.textContent = `
.pm-v2-name{flex:1 1 0;min-width:0;justify-content:flex-start;gap:6px;font:500 12px ${FONT}}
.pm-v2-name .label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pm-v2-badge{flex-shrink:0;color:#fff;font-weight:bold}
.pm-v2-badge.nsfw{padding:0 4px;border-radius:2px;background:rgba(204,50,50,.85);font-size:8px;line-height:12px}
.pm-v2-badge.kind{width:14px;height:14px;display:inline-flex;align-items:center;justify-content:center;border-radius:50%;font-size:9px;line-height:1}
.pm-v2-thumb{position:fixed;display:none;z-index:10001;pointer-events:none}
.pm-v2-thumb img{display:block;max-width:300px;max-height:300px;object-fit:contain;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,.5)}
.pm-v2-text.fbui-text{width:100%;height:100%;min-height:${TEXT_MIN_HEIGHT}px;padding:6px 8px;box-sizing:border-box;resize:none;pointer-events:auto}
.pm-v2-text[readonly]{opacity:.5}
.pm-v2-stack{display:flex;flex-direction:column;gap:4px;padding:8px;box-sizing:border-box;width:100%;border-radius:6px;background:${PMA_THEME.cardBg};pointer-events:auto}
.pm-v2-stack .head{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;padding-bottom:4px;border-bottom:1px solid ${PMA_THEME.sectionBorder}}
.pm-v2-stack .title{color:${PMA_THEME.textMuted};font-size:14px;font-weight:bold}
.pm-v2-stack .tools{display:flex;gap:4px}
.pm-v2-mini{padding:2px 8px;border:1px solid ${PMA_THEME.inputBorder};border-radius:6px;background:${PMA_THEME.buttonBg};color:${PMA_THEME.textPrimary};font-size:12px;cursor:pointer}
.pm-v2-tags{display:flex;flex-wrap:wrap;gap:4px}
.pm-v2-tag{display:inline-flex;align-items:center;gap:6px;height:24px;padding:4px 10px;box-sizing:border-box;border:1px solid rgba(226,232,240,.2);border-radius:6px;background:rgba(45,55,72,.7);color:rgba(226,232,240,.6);font-size:14px;cursor:pointer;user-select:none;transition:background-color .15s ease,color .15s ease,border-color .15s ease}
.pm-v2-tag:hover{transform:translateY(-1px);box-shadow:0 2px 4px rgba(0,0,0,.2)}
.pm-v2-tag .name{flex-grow:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pm-v2-tag.lora{flex-shrink:0;width:min(200px,calc(50% - 2px))}
.pm-v2-tag.lora.on{background:rgba(66,153,225,.9);border-color:rgba(122,188,243,.9);color:#fff}
.pm-v2-tag.lora.input.on{background:rgba(50,112,163,.9);border-color:rgba(66,153,225,.9)}
.pm-v2-tag.lora.missing{background:rgba(220,53,69,.4);border-color:rgba(220,53,69,.9);color:rgba(255,200,200,.8)}
.pm-v2-tag.lora.missing.on{background:rgba(220,53,69,.9);color:#fff}
.pm-v2-tag.lora.missing .name{text-decoration:line-through;opacity:.8}
.pm-v2-tag .warn{margin-right:-2px;font-size:13px}
.pm-v2-strength{width:38px;padding:1px 4px;border:1px solid transparent;border-radius:999px;outline:none;background:rgba(255,255,255,.15);color:rgba(255,255,255,.9);font-size:12px;font-weight:600;text-align:center;cursor:text}
.pm-v2-strength:focus{border-color:rgba(66,153,225,.6);background:rgba(255,255,255,.25)}
.pm-v2-tag.word{justify-content:center;max-width:100%;white-space:nowrap;overflow:hidden}
.pm-v2-tag.word.on{background:rgba(72,187,120,.9);border-color:rgba(72,187,120,.9);color:#fff}
`;
    document.head.append(style);
}

// ---------------------------------------------------------------- library

async function loadPrompts(node) {
    const response = await api.fetchApi(`${ENDPOINT}/get-prompts`, { cache: "no-store" });
    // The prompt browser reads the library from node.prompts too.
    node.prompts = response.ok ? await response.json() : {};
    return node.prompts;
}

const selection = (node) => ({ category: findWidget(node, "category")?.value || "", name: findWidget(node, "name")?.value || "" });
// A run's rows stand for the prompt and LoRA mode they were made with.
const runKey = (category, name, mode) => `${category}\u0000${name}\u0000${mode}`;
function currentKey(node) {
    const { category, name } = selection(node);
    return runKey(category, name, findWidget(node, "lora_mode")?.value);
}

function entryOf(node) {
    const { category, name } = selection(node);
    const entry = name ? node.prompts?.[category]?.[name] : null;
    return entry && typeof entry === "object" ? entry : null;
}

function isNsfw(node, category, name) {
    const entries = node.prompts?.[category];
    return entries?.__meta__?.nsfw === true || entries?.[name]?.nsfw === true;
}

// Every prompt in library order, NSFW ones left out while they are hidden.
function allPrompts(node) {
    const hide = getHideNSFW(app) === true;
    const out = [];
    for (const [category, entries] of Object.entries(node.prompts ?? {})) {
        if (!entries || typeof entries !== "object" || isHiddenPromptEntryKey(category)) continue;
        for (const name of Object.keys(entries)) {
            if (isHiddenPromptEntryKey(name)) continue;
            if (hide && isNsfw(node, category, name)) continue;
            out.push({ category, name });
        }
    }
    return out;
}

// ---------------------------------------------------------------- state

function readState(node) {
    try {
        const state = JSON.parse(findWidget(node, "state")?.value || "{}");
        return state && typeof state === "object" ? state : {};
    } catch {
        return {};
    }
}

function writeState(node, state) {
    findWidget(node, "state").value = JSON.stringify(state);
    node._pm.card.refresh();
    notifyChange();
}

const hasEdits = (state) => ["loras", "order"].some((key) => Object.values(state[key] ?? {}).some((slot) => Object.keys(slot ?? {}).length)) || Object.keys(state.trigger_words ?? {}).length > 0;

// Rewrite one entry of an edits map; an empty entry is dropped.
function setEdit(map, key, change) {
    const next = change({ ...(map[key] ?? {}) });
    if (Object.keys(next).length) map[key] = next;
    else delete map[key];
}

function editLoras(node, slot, changes) {
    const state = readState(node);
    state.loras ??= {};
    state.loras[slot] ??= {};
    for (const [name, change] of changes) setEdit(state.loras[slot], name.toLowerCase(), change);
    writeState(node, state);
}

function editWords(node, changes) {
    const state = readState(node);
    state.trigger_words ??= {};
    for (const [text, change] of changes) setEdit(state.trigger_words, text.toLowerCase(), change);
    writeState(node, state);
}

// An on/off edit, dropped when it matches the row's own state.
const activeEdit = (row, active) => (edit) => {
    if (active === (row.base_active ?? true)) delete edit.active;
    else edit.active = active;
    return edit;
};

// ---------------------------------------------------------------- rows

// The rows the node runs with: the last run's when it ran this prompt in
// this mode, else as nodes/prompt_manager_v2.py builds them, from the saved
// prompt's LoRAs and the last run's connected ones (those do not depend on
// the prompt), a connected LoRA of a saved one's name left out.
function baseLoras(node, slot) {
    const pm = node._pm;
    if (pm.run?.key === currentKey(node)) return pm.run.loras?.[slot] ?? [];
    const mode = findWidget(node, "lora_mode")?.value;
    const saved = mode === INPUT_ONLY ? [] : savedLoras(node, slot);
    const connected = mode === PROMPT_ONLY ? [] : (pm.run?.loras?.[slot] ?? []).filter((row) => row.from_input);
    const names = new Set(saved.map((row) => row.name.toLowerCase()));
    return [...saved, ...connected.filter((row) => !names.has(row.name.toLowerCase()))];
}

function savedLoras(node, slot) {
    return (entryOf(node)?.[`loras_${slot}`] ?? [])
        .filter((lora) => lora && typeof lora === "object" && String(lora.name ?? "").trim())
        .map((lora) => {
            const strength = Number(lora.strength ?? lora.model_strength ?? 1);
            return {
                name: String(lora.name).trim(),
                path: lora.path ?? lora.name,
                strength,
                clip_strength: Number(lora.clip_strength ?? strength),
                base_strength: strength,
                base_active: lora.active !== false,
                available: node._pm.available.get(String(lora.name).trim().toLowerCase()) ?? true,
                from_input: false,
            };
        });
}

// Rows as shown: the base with the node's edits, removed ones left out, in the node's order.
function loraRows(node, slot) {
    const state = readState(node);
    const edits = state.loras?.[slot] ?? {};
    const order = state.order?.[slot] ?? [];
    const at = (row) => (order.includes(row.name.toLowerCase()) ? order.indexOf(row.name.toLowerCase()) : order.length);
    return baseLoras(node, slot)
        .filter((row) => edits[row.name.toLowerCase()]?.removed !== true)
        .map((row) => {
            const edit = edits[row.name.toLowerCase()] ?? {};
            const strength = typeof edit.strength === "number" ? edit.strength : row.base_strength ?? row.strength;
            return { ...row, strength, active: edit.active ?? row.base_active ?? row.active };
        })
        .sort((a, b) => at(a) - at(b));
}

// Trigger words the same way: saved ones, then the last run's connected ones.
function baseWords(node) {
    const pm = node._pm;
    if (pm.run?.key === currentKey(node)) return pm.run.trigger_words ?? [];
    const saved = (entryOf(node)?.trigger_words ?? []).map((word) => (typeof word === "string" ? { text: word, base_active: true } : { text: word?.text ?? "", base_active: word?.active !== false }));
    const texts = new Set(saved.map((word) => String(word.text).toLowerCase()));
    return [...saved, ...(pm.run?.trigger_words ?? []).filter((word) => word.from_input && !texts.has(word.text.toLowerCase()))];
}

function wordRows(node) {
    const base = baseWords(node);
    const edits = readState(node).trigger_words ?? {};
    return base
        .filter((word) => String(word.text ?? "").trim() && edits[word.text.toLowerCase()]?.removed !== true)
        .map((word) => ({ ...word, active: edits[word.text.toLowerCase()]?.active ?? word.base_active ?? word.active }));
}

// Which saved LoRAs are found locally, asked once per prompt shown.
async function checkAvailability(node) {
    const entry = entryOf(node);
    const names = SLOTS.flatMap((slot) => (entry?.[`loras_${slot}`] ?? []).map((lora) => String(lora?.name ?? "").trim())).filter(Boolean);
    if (!names.length) return;
    const response = await api.fetchApi(`${ENDPOINT}/check-loras`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lora_names: names }),
    });
    const result = response.ok ? await response.json() : null;
    for (const [name, found] of Object.entries(result?.results ?? {})) node._pm.available.set(name.toLowerCase(), found === true);
    node._pm.card.refresh();
}

// ---------------------------------------------------------------- selection

const isLinked = (node, name) => node.inputs?.find((input) => input.name === name)?.link != null;

function isDirty(node) {
    const entry = entryOf(node);
    const text = findWidget(node, "text")?.value ?? "";
    return hasEdits(readState(node)) || (entry ? text !== (entry.prompt ?? "") : text.trim() !== "");
}

async function mayLeave(node) {
    if (!isDirty(node) || app.ui.settings.getSettingValue("PromptManager.WarnUnsavedChanges") === false) return true;
    return confirm({ title: "Unsaved changes", message: "The prompt's text or LoRA edits are not saved. Discard them and switch?", confirmText: "Discard & Switch" });
}

// Show a prompt: its saved text and no edits. The last run's rows no longer
// stand for it (even a prompt just saved may differ from what ran), but its
// connected LoRAs and trigger words still show.
function select(node, category, name) {
    findWidget(node, "category").value = category;
    findWidget(node, "name").value = name;
    findWidget(node, "text").value = name ? node.prompts?.[category]?.[name]?.prompt ?? "" : "";
    if (node._pm.run) node._pm.run.key = null;
    writeState(node, {});
    checkAvailability(node);
}

async function step(node, delta) {
    const list = allPrompts(node);
    if (!list.length || !(await mayLeave(node))) return;
    const { category, name } = selection(node);
    const at = list.findIndex((item) => item.category === category && item.name === name);
    const next = list[(at + delta + list.length) % list.length] ?? list[0];
    select(node, next.category, next.name);
}

async function browse(node) {
    const { category, name } = selection(node);
    const picked = await showThumbnailBrowser(node, category, name, { endpointPrefix: ENDPOINT, loadPromptsFn: loadPrompts, filterEmptyCategories: true, showAllCategoriesToggle: true });
    if (!picked?.prompt || (picked.category === category && picked.prompt === name) || !(await mayLeave(node))) return;
    select(node, picked.category, picked.prompt);
}

async function newPrompt(node) {
    if (!(await mayLeave(node))) return;
    select(node, selection(node).category, "");
}

async function deletePrompt(node) {
    const { category, name } = selection(node);
    if (!name || !(await confirm({ title: "Delete prompt", message: `Delete "${name}" from ${category}? This cannot be undone.`, confirmText: "Delete", danger: true }))) return;
    const response = await api.fetchApi(`${ENDPOINT}/delete-prompt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category, name }),
    });
    const result = await response.json();
    if (!result?.success) return toast(result?.error || "Could not delete the prompt.", { severity: "error" });
    node.prompts = result.prompts;
    select(node, category, "");
}

function importPrompts(node, mode) {
    const input = el("input");
    input.type = "file";
    input.accept = ".json";
    input.addEventListener("change", async () => {
        const file = input.files?.[0];
        if (!file) return;
        let data;
        try {
            data = JSON.parse(await file.text());
        } catch {
            return toast(`${file.name} is not valid JSON.`, { severity: "error" });
        }
        if (mode === "replace" && !(await confirm({ title: "Replace prompts", message: `Replace the whole prompt library with ${file.name}?`, confirmText: "Replace", danger: true }))) return;
        const response = await api.fetchApi(`${ENDPOINT}/import-prompts`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ data, mode }),
        });
        const result = await response.json();
        if (!result?.success) return toast(result?.error || "Could not import the prompts.", { severity: "error" });
        node.prompts = result.prompts;
        node._pm.card.refresh();
        toast(mode === "replace" ? "Prompts replaced." : "Prompts imported.");
    });
    input.click();
}

// Save through the library's save dialog, edits included; the saved prompt
// then carries them, so the node's edits are cleared.
async function savePrompt(node) {
    const { category, name } = selection(node);
    const entry = entryOf(node);
    const initialName = name || "New Prompt";
    const loras = Object.fromEntries(SLOTS.map((slot) => [`loras_${slot}`, loraRows(node, slot).map((row) => ({
        name: row.name, path: row.path, strength: row.strength, clip_strength: row.clip_strength,
        active: row.active, available: row.available, fromInput: row.from_input === true,
    }))]));
    const words = wordRows(node).map((word) => ({ text: word.text, active: word.active }));
    await openPromptBrowserForSave({
        node,
        currentCategory: category,
        currentPrompt: name,
        title: "Save Prompt",
        saveButtonText: "Save",
        namePlaceholder: "Prompt name",
        initialName,
        saveWithEditPanel: true,
        saveMode: true,
        showNSFWOnOpen: Boolean(entry) && isNsfw(node, category, name),
        saveDraft: { category, name: initialName, text: findWidget(node, "text").value || "", thumbnail: entry?.thumbnail || null, workflow_data: entry?.workflow_data || null, ...loras, trigger_words: words },
        onSave: async ({ category: target, name: saveName, overwrite, text, thumbnail, ...edited }) => {
            const promptName = String(saveName || "").trim();
            const targetCategory = String(target || "").trim();
            if (!promptName || !targetCategory) return { success: false, error: "Category and prompt name are required." };
            const payload = { category: targetCategory, name: promptName, text: typeof text === "string" ? text : findWidget(node, "text").value || "", thumbnail };
            for (const slot of SLOTS) payload[`loras_${slot}`] = Array.isArray(edited[`loras_${slot}`]) ? edited[`loras_${slot}`] : loras[`loras_${slot}`];
            payload.trigger_words = Array.isArray(edited.trigger_words) ? edited.trigger_words : words;
            const response = await api.fetchApi(`${ENDPOINT}/save-prompt`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(buildSavePromptRequestBodyForEndpoint(ENDPOINT, payload)),
            });
            const result = await response.json();
            if (!result?.success) return { success: false, error: result?.error || "Could not save the prompt." };
            node.prompts = result.prompts;
            select(node, targetCategory, promptName);
            return { success: true, category: targetCategory, name: promptName, overwritten: overwrite === true };
        },
    });
}

// ---------------------------------------------------------------- prompt bar

// The saved prompt's thumbnail at its own size above the bar, while hovered.
function thumbnailHover(node, target) {
    const preview = el("div", "pm-v2-thumb");
    const image = el("img");
    preview.append(image);
    document.body.append(preview);
    chainCallback(node, "onRemoved", () => preview.remove());
    let timer = null;
    target.addEventListener("mouseenter", () => {
        timer = setTimeout(() => {
            const thumbnail = entryOf(node)?.thumbnail;
            if (!thumbnail || thumbnail === DEFAULT_THUMBNAIL) return;
            const probe = new Image();
            probe.onload = () => {
                image.src = thumbnail;
                const width = Math.min(probe.naturalWidth, 300);
                const height = Math.min(probe.naturalHeight, 300);
                const box = target.getBoundingClientRect();
                preview.style.left = `${Math.max(5, Math.min(box.left + box.width / 2 - width / 2, window.innerWidth - width - 5))}px`;
                preview.style.top = `${Math.max(5, Math.min(box.top - height - 8, window.innerHeight - height - 5))}px`;
                preview.style.display = "block";
            };
            probe.src = thumbnail;
        }, 300);
    });
    target.addEventListener("mouseleave", () => {
        clearTimeout(timer);
        preview.style.display = "none";
    });
}

function promptBar(node) {
    ensureCss();
    // The prompt as a button that opens the library (the suite's soft blue
    // action), ‹ › stepping through it.
    const bar = el("div", "fbui-field");
    const label = el("span", "label");
    const nsfw = el("span", "pm-v2-badge nsfw", "NSFW");
    const kind = el("span", "pm-v2-badge kind");
    const name = button("", { soft: true, onClick: () => browse(node) });
    name.classList.add("pm-v2-name");
    name.append(label, nsfw, kind);
    const back = button("", { icon: ICONS.chevronLeft, title: "Previous prompt", onClick: () => step(node, -1) });
    const next = button("", { icon: ICONS.chevronRight, title: "Next prompt", onClick: () => step(node, 1) });
    bar.append(name, back, next);
    bar.addEventListener("contextmenu", (event) => event.preventDefault());
    thumbnailHover(node, name);
    return {
        element: bar,
        sync() {
            const { category, name: prompt } = selection(node);
            label.textContent = `${category} : ${prompt || "new prompt"}`;
            name.title = `${label.textContent}\nClick to browse the prompt library`;
            const entry = entryOf(node);
            nsfw.hidden = !prompt || !isNsfw(node, category, prompt);
            // L, C or R: the entry also holds a layout, a Composer setup or a recipe.
            const compose = hasComposeLikePayload(entry);
            const layout = compose && hasLayoutPayload(entry);
            const recipe = !compose && hasRecipeLikePayload(entry);
            kind.hidden = !compose && !recipe;
            kind.textContent = layout ? "L" : compose ? "C" : "R";
            kind.style.background = layout ? "rgba(124,92,220,.95)" : compose ? "rgba(55,165,95,.95)" : "rgba(235,140,35,.95)";
        },
    };
}

function buttonBar(node) {
    const row = el("div", "fbui-field");
    const add = (text, title, onClick, icon = "") => row.append(button(text, { icon, soft: true, grow: true, title, onClick }));
    add("Save Prompt", "Save the prompt, its LoRAs and trigger words to the library", () => savePrompt(node));
    add("New Prompt", "Start a new prompt", () => newPrompt(node));
    add("More", "", (event) => {
        const box = event.currentTarget.getBoundingClientRect();
        openMenu(box.left, box.bottom + 4, [
            { label: "Delete Prompt", danger: true, disabled: !selection(node).name, action: () => deletePrompt(node) },
            { label: "Undo LoRA and trigger word edits", disabled: !hasEdits(readState(node)), action: () => writeState(node, {}) },
            null,
            { label: "Export JSON", action: () => exportPromptsJSON(node) },
            { label: "Import JSON", action: () => importPrompts(node, "merge") },
            { label: "Replace JSON", action: () => importPrompts(node, "replace") },
        ]);
    }, ICONS.chevronDown);
    row.addEventListener("contextmenu", (event) => event.preventDefault());
    return { element: row };
}

function textPanel(node) {
    const area = el("textarea", "fbui-text pm-v2-text");
    area.spellcheck = false;
    area.addEventListener("input", () => { findWidget(node, "text").value = area.value; });
    area.addEventListener("change", notifyChange);
    // Typing and selecting here must not drag the node or reach the canvas.
    area.addEventListener("pointerdown", (event) => event.stopPropagation());
    area.addEventListener("keydown", (event) => event.stopPropagation());
    return {
        element: area,
        sync() {
            // With the prompt input on, its text from the last run, read-only.
            const fromInput = findWidget(node, "use_prompt_input")?.value === true && isLinked(node, "prompt");
            area.readOnly = fromInput;
            const shown = fromInput ? node._pm.run?.prompt_input ?? "" : findWidget(node, "text").value ?? "";
            if (area.value !== shown && document.activeElement !== area) area.value = shown;
            area.placeholder = fromInput ? "From the prompt input (shown after a run)" : "Enter prompt text";
        },
    };
}

// ---------------------------------------------------------------- stack cards

// A titled card with buttons at the right and a wrap of tags; it measures its
// own height and asks the node's card to follow. Tags are rebuilt only when
// what they show changes, and never under an open strength field.
function stackCard(node, key, title, tools, renderTags) {
    const box = el("div", "pm-v2-stack");
    const head = el("div", "head");
    const buttons = el("div", "tools");
    for (const [text, hint, onClick] of tools) {
        const btn = el("button", "pm-v2-mini", text);
        btn.title = hint;
        btn.addEventListener("click", onClick);
        buttons.append(btn);
    }
    head.append(el("span", "title", title), buttons);
    const tags = el("div", "pm-v2-tags");
    box.append(head, tags);
    box.addEventListener("contextmenu", (event) => event.preventDefault());
    const measure = () => {
        const height = Math.ceil(box.scrollHeight);
        if (height && height !== node._pm.heights[key]) {
            node._pm.heights[key] = height;
            node._pm.card.refresh();
        }
    };
    new ResizeObserver(measure).observe(tags);
    let shown = null;
    return {
        element: box,
        sync() {
            // A strength being typed keeps its field; anything else (a button just clicked) redraws.
            if (box.contains(document.activeElement) && document.activeElement.classList.contains("pm-v2-strength")) return;
            const { signature, elements } = renderTags();
            if (signature === shown) return;
            shown = signature;
            tags.replaceChildren(...elements);
            requestAnimationFrame(measure);
        },
    };
}

function strengthField(node, slot, row) {
    const input = el("input", "pm-v2-strength");
    input.type = "text";
    input.value = Number(row.strength).toFixed(2);
    input.addEventListener("focus", () => input.select());
    input.addEventListener("blur", () => {
        const value = parseFloat(input.value);
        if (Number.isNaN(value) || value === row.strength) {
            input.value = Number(row.strength).toFixed(2);
            return;
        }
        editLoras(node, slot, [[row.name, (edit) => {
            if (value === row.base_strength) delete edit.strength;
            else edit.strength = value;
            return edit;
        }]]);
    });
    input.addEventListener("keydown", (event) => {
        event.stopPropagation();
        if (event.key === "Enter") input.blur();
        if (event.key === "Escape") {
            input.value = Number(row.strength).toFixed(2);
            input.blur();
        }
    });
    input.addEventListener("pointerdown", (event) => event.stopPropagation());
    input.addEventListener("click", (event) => event.stopPropagation());
    return input;
}

function moveLora(node, slot, index, delta) {
    const names = loraRows(node, slot).map((row) => row.name.toLowerCase());
    const to = index + delta;
    if (to < 0 || to >= names.length) return;
    names.splice(to, 0, names.splice(index, 1)[0]);
    const state = readState(node);
    state.order ??= {};
    state.order[slot] = names;
    writeState(node, state);
}

function loraTag(node, slot, row, index, count) {
    const missing = row.available === false;
    const tag = el("div", "pm-v2-tag lora");
    tag.classList.toggle("on", row.active);
    tag.classList.toggle("input", row.from_input === true);
    tag.classList.toggle("missing", missing);
    if (missing) tag.append(el("span", "warn", "⚠️"));
    tag.append(el("span", "name", row.name), strengthField(node, slot, row));
    tag.addEventListener("click", () => editLoras(node, slot, [[row.name, activeEdit(row, !row.active)]]));
    tag.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        event.stopPropagation();
        openMenu(event.clientX, event.clientY, [
            { label: "Move Up", disabled: index === 0, action: () => moveLora(node, slot, index, -1) },
            { label: "Move Down", disabled: index === count - 1, action: () => moveLora(node, slot, index, 1) },
            ...(missing ? [{ label: "🔍 Search on CivitAI", action: () => window.open(`https://civitai.com/search/models?sortBy=models_v9&query=${encodeURIComponent(row.name)}&modelType=LORA`, "_blank") }] : []),
            null,
            { label: "Delete", danger: true, action: () => editLoras(node, slot, [[row.name, (edit) => ({ ...edit, removed: true })]]) },
        ]);
    });
    tag.title = missing
        ? `${row.name}\n⚠️ NOT FOUND - This LoRA is missing from your system\nRight-click for options`
        : `${row.name}${row.from_input ? " (from the input)" : ""}\nStrength: ${Number(row.strength).toFixed(2)}\nClick to toggle on/off\nRight-click for options`;
    if (missing) return tag;
    // LoRA Manager's preview beside the tag when it is installed, in place of the plain tooltip.
    let timer = null;
    tag.addEventListener("mouseenter", () => {
        const box = tag.getBoundingClientRect();
        timer = setTimeout(async () => {
            const tooltip = await getLoraManagerPreviewTooltip();
            if (!tooltip) return;
            tag.removeAttribute("title");
            tooltip.show(row.name, box.right, box.top);
        }, 300);
    });
    tag.addEventListener("mouseleave", () => {
        clearTimeout(timer);
        getLoraManagerPreviewTooltip().then((tooltip) => tooltip?.hide());
    });
    return tag;
}

function loraCard(node, slot) {
    return stackCard(node, slot, `LoRAs Stack ${slot.toUpperCase()}`, [
        ["Reset Strength", "Reset all strength values to their original values", () => {
            editLoras(node, slot, loraRows(node, slot).map((row) => [row.name, (edit) => {
                delete edit.strength;
                return edit;
            }]));
        }],
        ["Toggle All", "Switch all of this stack's LoRAs on, or off when they all are", () => {
            const rows = loraRows(node, slot);
            const active = !rows.every((row) => row.active);
            editLoras(node, slot, rows.map((row) => [row.name, activeEdit(row, active)]));
        }],
    ], () => {
        const rows = loraRows(node, slot);
        return {
            signature: JSON.stringify(rows),
            elements: rows.map((row, index) => loraTag(node, slot, row, index, rows.length)),
        };
    });
}

function wordTag(node, word) {
    const tag = el("div", "pm-v2-tag word");
    tag.classList.toggle("on", word.active);
    tag.append(el("span", "name", word.text));
    tag.title = `${word.text}${word.from_input ? " (from the input)" : ""}\nClick to toggle on/off\nRight-click for options`;
    tag.addEventListener("click", () => editWords(node, [[word.text, activeEdit(word, !word.active)]]));
    tag.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        event.stopPropagation();
        openMenu(event.clientX, event.clientY, [{ label: "Delete", danger: true, action: () => editWords(node, [[word.text, (edit) => ({ ...edit, removed: true })]]) }]);
    });
    return tag;
}

function wordsCard(node) {
    return stackCard(node, "words", "Trigger Words", [
        ["Toggle All", "Switch all trigger words on, or off when they all are", () => {
            const words = wordRows(node);
            const active = !words.every((word) => word.active);
            editWords(node, words.map((word) => [word.text, activeEdit(word, active)]));
        }],
    ], () => {
        const words = wordRows(node);
        return { signature: JSON.stringify(words), elements: words.map((word) => wordTag(node, word)) };
    });
}

// ---------------------------------------------------------------- node

function setupNode(node) {
    ensureCss();
    const pm = { run: null, available: new Map(), heights: { a: STACK_HEAD + 28, b: STACK_HEAD + 28, c: STACK_HEAD + 28, d: STACK_HEAD + 28, words: STACK_HEAD + 28 } };
    node._pm = pm;
    // The card stands in for these; use_prompt_input and lora_mode are card rows.
    for (const name of ["category", "name", "text", "state"]) hideWidget(findWidget(node, name));
    pm.card = mountCard(node, {
        minWidth: MIN_WIDTH,
        rows: [
            { widget: "use_prompt_input", label: "Prompt input" },
            { widget: "lora_mode", label: "LoRA input", kind: "segment", labels: MODE_LABELS, titles: MODE_TITLES },
            // The prompt's name, then its text, then what to do with it.
            { panel: promptBar, height: ROW_HEIGHT },
            { panel: textPanel, grow: true, height: TEXT_MIN_HEIGHT },
            { panel: buttonBar, height: ROW_HEIGHT },
            ...SLOTS.map((slot) => ({ panel: (n) => loraCard(n, slot), height: () => pm.heights[slot], when: (_values, n) => loraRows(n, slot).length > 0 })),
            { panel: wordsCard, height: () => pm.heights.words, when: (_values, n) => wordRows(n).length > 0 },
        ],
    });
    helpBadge(node, GUIDE);
    chainCallback(node, "onRemoved", () => getLoraManagerPreviewTooltip().then((tooltip) => tooltip?.hide()));
    chainCallback(node, "onExecuted", (output) => {
        const panels = output?.prompt_manager?.[0];
        if (!panels) return;
        pm.run = { ...panels, key: runKey(panels.category, panels.name, panels.lora_mode) };
        node.properties[RUN_PROPERTY] = pm.run;
        pm.card.refresh();
    });
    chainCallback(node, "onConfigure", () => {
        pm.configured = true;
        pm.run = node.properties?.[RUN_PROPERTY] ?? null;
        loadPrompts(node).then(() => {
            pm.card.refresh();
            checkAvailability(node);
        });
    });
    queueMicrotask(() => {
        if (pm.configured) return;
        loadPrompts(node).then(() => pm.card.refresh());
    });
}

app.registerExtension({
    name: "PromptManager.PromptManagerV2",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        hideInputsInDef(nodeData, ["category", "name", "use_prompt_input", "lora_mode", "text", "state"]);
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            setupNode(this);
        });
    },
});
