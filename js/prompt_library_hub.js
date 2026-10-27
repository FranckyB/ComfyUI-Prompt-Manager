import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { PM_UI_PALETTE as UI } from "./ui_palette.js";
import { showInfo, showConfirm } from "./prompt_manager_advanced.js";
import { showThumbnailBrowser, getPromptNamesForCategory } from "./prompt_browser.js";
import {
    exportComposerJsonLibrary,
    mergeComposerJsonLibrary,
    openComposerJsonLibrary,
    showComposerImportModeDialog,
    triggerBrowserDownload,
} from "./prompt_composer.js";
import { loadComposerPrompts } from "./prompt_composer_common.js";
import {
    COMPOSER_ENDPOINT_PREFIX,
    PROMPT_ADVANCED_ENDPOINT_PREFIX,
    PROMPT_ENDPOINT_PREFIX,
    SYSTEM_PROMPTS_ENDPOINT_PREFIX,
    getCategoryPromptEntries,
    isHiddenPromptEntryKey,
} from "./prompt_store_adapters.js";

const HUB_TITLE = "Prompt Library Hub";
const STYLE_ID = "pm-library-hub-style";
const KIND_STORAGE_KEY = "pm-library-hub-kind";

// Saved prompt kinds share prompt_manager_data.json and are told apart by the browser's Type filter.
const PROMPT_KINDS = [
    { id: "prompt", label: "Prompt Manager", badge: "P", color: "56, 130, 246", hint: "Saved from Prompt Manager or Library Manager." },
    { id: "compose", label: "Prompt Composer", badge: "C", color: "47, 146, 72", hint: "Prompt Composer setups saved with Library Manager." },
    { id: "layout", label: "Prompt Layout", badge: "L", color: "124, 92, 220", hint: "Prompt Layout Builder layouts saved with Library Manager." },
    { id: "recipe", label: "Prompt Recipe", badge: "R", color: "235, 140, 35", hint: "Recipes saved with Library Manager." },
];

const SECTIONS = [
    { id: "prompts", title: "Saved Prompts", store: "manager", kinds: PROMPT_KINDS, unit: "entr" },
    { id: "snippets", title: "Saved Snippets", browseTitle: "Saved Composer Snippets", store: "composer", hint: "Prompt snippets used by Prompt Composer.", unit: "snippet" },
    { id: "system", title: "Saved System Prompts", store: "generator", hint: "System prompts used by Prompt Generator.", unit: "prompt" },
];

const plural = (count, unit) => `${count} ${unit === "entr" ? (count === 1 ? "entry" : "entries") : `${unit}${count === 1 ? "" : "s"}`}`;

function libraryFor(section, kind) {
    if (!kind) return { id: section.id, title: section.browseTitle || section.title, store: section.store, unit: section.unit };
    return { id: `${section.id}_${kind.id}`, title: `${section.title} - ${kind.label}`, store: section.store, filter: kind.id, unit: section.unit };
}

const STORE_ENDPOINTS = {
    generator: SYSTEM_PROMPTS_ENDPOINT_PREFIX,
    composer: COMPOSER_ENDPOINT_PREFIX,
    manager: PROMPT_ADVANCED_ENDPOINT_PREFIX,
};

// Stand-in "node" objects: the shared browser and Composer helpers keep their loaded data on these.
const hosts = {
    generator: { prompts: {} },
    composer: { prompts: {} },
    manager: { prompts: {} },
};

async function fetchStore(host, endpointPrefix) {
    const response = await api.fetchApi(`${endpointPrefix}/get-prompts`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Could not load ${endpointPrefix} (${response.status})`);
    host.prompts = await response.json();
    return host.prompts;
}

const loadSystemPrompts = (host) => fetchStore(host, SYSTEM_PROMPTS_ENDPOINT_PREFIX);
const loadManagerPrompts = (host) => fetchStore(host, PROMPT_ADVANCED_ENDPOINT_PREFIX);

function categoryNames(data) {
    return Object.keys(data || {})
        .filter((category) => category !== "__meta__" && data[category] && typeof data[category] === "object")
        .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
}

function entryNames(library, category) {
    const data = hosts[library.store].prompts;
    if (library.store === "manager") {
        return getPromptNamesForCategory(hosts.manager, category, { contentFilter: library.filter, endpointPrefix: PROMPT_ADVANCED_ENDPOINT_PREFIX });
    }
    return Object.keys(getCategoryPromptEntries(data?.[category], library.store));
}

function summarize(library) {
    const data = hosts[library.store].prompts;
    let entries = 0;
    let categories = 0;
    for (const category of categoryNames(data)) {
        const count = entryNames(library, category).length;
        entries += count;
        if (count) categories += 1;
    }
    const groups = library.store === "composer" ? Object.keys(hosts.composer.composerPromptLibrary?._types_ || {}).length : 0;
    return { entries, categories, groups };
}

function timestamp() {
    return new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
}

function pickJsonFile() {
    return new Promise((resolve) => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".json,application/json";
        input.onchange = () => resolve(input.files?.[0] || null);
        input.oncancel = () => resolve(null);
        input.click();
    });
}

async function readLibraryFile() {
    const file = await pickJsonFile();
    if (!file) return null;
    let data;
    try {
        data = JSON.parse(await file.text());
    } catch {
        throw new Error(`"${file.name}" is not valid JSON.`);
    }
    if (!data || typeof data !== "object" || Array.isArray(data)) {
        throw new Error(`"${file.name}" is not a prompt library (expected categories of entries).`);
    }
    return { file, data };
}

function makeButton(label, { primary = false, title = "" } = {}) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.title = title;
    button.className = primary ? "pm-hub-btn pm-hub-btn-primary" : "pm-hub-btn";
    return button;
}

// Category picker modelled on Prompt Composer's "Save JSONs" dialog.
function showCategoryExportDialog(library, categories) {
    return new Promise((resolve) => {
        const selected = new Set(categories.map((item) => item.name));
        const overlay = document.createElement("div");
        overlay.className = "pm-hub-overlay";
        const dialog = document.createElement("div");
        dialog.className = "pm-hub-dialog";

        const header = document.createElement("div");
        header.className = "pm-hub-dialog-header";
        const heading = document.createElement("div");
        heading.className = "pm-hub-dialog-title";
        heading.textContent = `Export ${library.title}`;
        const hint = document.createElement("div");
        hint.className = "pm-hub-muted";
        hint.textContent = "Select the categories to export as one JSON file.";
        header.append(heading, hint);

        const body = document.createElement("div");
        body.className = "pm-hub-dialog-grid";
        const cards = new Map();
        for (const item of categories) {
            const card = document.createElement("button");
            card.type = "button";
            card.className = "pm-hub-pick";
            const name = document.createElement("div");
            name.className = "pm-hub-pick-name";
            name.textContent = item.name;
            const count = document.createElement("div");
            count.className = "pm-hub-muted";
            count.textContent = `${item.count} entr${item.count === 1 ? "y" : "ies"}`;
            card.append(name, count);
            card.onclick = () => {
                if (selected.has(item.name)) selected.delete(item.name);
                else selected.add(item.name);
                sync();
            };
            cards.set(item.name, card);
            body.appendChild(card);
        }

        const footer = document.createElement("div");
        footer.className = "pm-hub-dialog-footer";
        const info = document.createElement("div");
        info.className = "pm-hub-muted";
        info.style.marginRight = "auto";
        const clearButton = makeButton("Clear");
        const allButton = makeButton("Select All");
        const cancelButton = makeButton("Cancel");
        const saveButton = makeButton("Export JSON", { primary: true });
        footer.append(info, clearButton, allButton, cancelButton, saveButton);

        function sync() {
            cards.forEach((card, name) => card.classList.toggle("active", selected.has(name)));
            const total = categories.filter((item) => selected.has(item.name)).reduce((sum, item) => sum + item.count, 0);
            info.textContent = `${selected.size} categor${selected.size === 1 ? "y" : "ies"} · ${total} entr${total === 1 ? "y" : "ies"}`;
            saveButton.disabled = selected.size === 0;
        }

        const finish = (value) => {
            overlay.remove();
            resolve(value);
        };
        clearButton.onclick = () => { selected.clear(); sync(); };
        allButton.onclick = () => { categories.forEach((item) => selected.add(item.name)); sync(); };
        cancelButton.onclick = () => finish(null);
        saveButton.onclick = () => finish(categories.map((item) => item.name).filter((name) => selected.has(name)));
        overlay.onclick = (event) => { if (event.target === overlay) finish(null); };

        dialog.append(header, body, footer);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
        sync();
    });
}

async function exportLibrary(library) {
    if (library.store === "composer") {
        await loadComposerPrompts(hosts.composer);
        await exportComposerJsonLibrary(hosts.composer);
        return;
    }
    const data = hosts[library.store].prompts;
    const categories = categoryNames(data)
        .map((name) => ({ name, count: entryNames(library, name).length }))
        .filter((item) => item.count > 0);
    if (!categories.length) {
        await showInfo("Nothing to Export", `${library.title} has no entries yet.`);
        return;
    }
    const picked = await showCategoryExportDialog(library, categories);
    if (!picked?.length) return;

    const out = {};
    for (const category of picked) {
        const source = data[category];
        const names = new Set(entryNames(library, category));
        out[category] = Object.fromEntries(Object.entries(source).filter(([key]) => names.has(key) || isHiddenPromptEntryKey(key)));
    }
    const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
    triggerBrowserDownload(blob, `${library.id}_${timestamp()}.json`);
}

function showChoiceDialog(title, choices) {
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.className = "pm-hub-overlay";
        const dialog = document.createElement("div");
        dialog.className = "pm-hub-dialog pm-hub-dialog-small";
        const header = document.createElement("div");
        header.className = "pm-hub-dialog-header";
        const heading = document.createElement("div");
        heading.className = "pm-hub-dialog-title";
        heading.textContent = title;
        header.appendChild(heading);
        const body = document.createElement("div");
        body.className = "pm-hub-dialog-list";
        const finish = (value) => {
            overlay.remove();
            resolve(value);
        };
        for (const choice of choices) {
            const option = document.createElement("button");
            option.type = "button";
            option.className = "pm-hub-pick";
            const name = document.createElement("div");
            name.className = "pm-hub-pick-name";
            name.textContent = choice.label;
            const description = document.createElement("div");
            description.className = "pm-hub-muted";
            description.textContent = choice.description;
            option.append(name, description);
            option.onclick = () => finish(choice.value);
            body.appendChild(option);
        }
        const footer = document.createElement("div");
        footer.className = "pm-hub-dialog-footer";
        const cancelButton = makeButton("Cancel");
        cancelButton.style.marginLeft = "auto";
        cancelButton.onclick = () => finish(null);
        footer.appendChild(cancelButton);
        overlay.onclick = (event) => { if (event.target === overlay) finish(null); };
        dialog.append(header, body, footer);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
    });
}

async function importLibrary(library) {
    if (library.store === "manager") {
        await importJsonFile(library, false);
        return;
    }
    const choices = [
        { value: "merge", label: "Merge a file", description: "Add the file's entries; you choose what happens to entries that already exist." },
        { value: "replace", label: "Replace with a file", description: `Replace all ${library.title} with the file's contents.` },
    ];
    if (library.store === "generator") {
        choices.push({ value: "defaults", label: "Restore built-in defaults", description: "Add back missing built-in system prompts without touching yours." });
    }
    const action = await showChoiceDialog(`Import ${library.title}`, choices);
    if (!action) return;
    if (action === "defaults") {
        await restoreDefaultSystemPrompts();
    } else if (library.store === "composer") {
        await loadComposerPrompts(hosts.composer);
        await (action === "replace" ? openComposerJsonLibrary : mergeComposerJsonLibrary)(hosts.composer);
    } else {
        await importJsonFile(library, action === "replace");
    }
}

async function importJsonFile(library, replace) {
    const picked = await readLibraryFile();
    if (!picked) return;
    if (replace) {
        const confirmed = await showConfirm(
            `Replace ${library.title}`,
            `Replace ALL ${library.title} with "${picked.file.name}"? This cannot be undone.`,
            "Replace",
            UI.accent,
        );
        if (!confirmed) return;
    }

    let mode = replace ? "replace" : "skip_existing";
    if (!replace) {
        const current = hosts[library.store].prompts;
        let duplicates = 0;
        for (const category of categoryNames(picked.data)) {
            const existing = getCategoryPromptEntries(current?.[category], library.store);
            for (const name of Object.keys(getCategoryPromptEntries(picked.data[category], library.store))) {
                if (Object.prototype.hasOwnProperty.call(existing, name)) duplicates += 1;
            }
        }
        if (duplicates) {
            mode = await showComposerImportModeDialog({ duplicatePromptCount: duplicates });
            if (mode === "cancel") return;
        }
    }

    // Prompt Manager's file is shared by all saved kinds; entries keep their kind through their data blocks.
    const endpointPrefix = library.store === "manager" ? PROMPT_ENDPOINT_PREFIX : STORE_ENDPOINTS[library.store];
    const response = await api.fetchApi(`${endpointPrefix}/import-prompts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: picked.data, mode }),
    });
    const result = await response.json();
    if (!response.ok || !result?.success) throw new Error(result?.error || "Import failed.");
    if (replace) {
        await showInfo("Replace Complete", `${library.title} replaced with "${picked.file.name}".`);
        return;
    }
    const imported = Number(result.imported_prompts || 0);
    const skipped = Number(result.skipped_prompts || 0);
    const created = Number(result.created_categories || 0);
    const parts = [`Imported ${imported} entr${imported === 1 ? "y" : "ies"}`];
    if (created) parts.push(`created ${created} categor${created === 1 ? "y" : "ies"}`);
    if (skipped) parts.push(`kept ${skipped} existing entr${skipped === 1 ? "y" : "ies"}`);
    await showInfo("Import Complete", `${parts.join(", ")}.`);
}

async function restoreDefaultSystemPrompts() {
    const confirmed = await showConfirm(
        "Restore Default System Prompts",
        "This adds any missing built-in system prompts without overwriting your custom ones. Continue?",
        "Restore",
        UI.accent,
    );
    if (!confirmed) return;
    const response = await api.fetchApi(`${SYSTEM_PROMPTS_ENDPOINT_PREFIX}/reimport-default-prompts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
    });
    const result = await response.json();
    if (!response.ok || !result?.success) throw new Error(result?.error || "Failed to restore default system prompts.");
    await showInfo("Restore Complete", "Added back any missing built-in system prompts.");
}

async function browseLibrary(library) {
    const title = library.title;
    if (library.store === "generator") {
        await showThumbnailBrowser(hosts.generator, "", "", {
            title,
            endpointPrefix: SYSTEM_PROMPTS_ENDPOINT_PREFIX,
            loadPromptsFn: loadSystemPrompts,
            promptOnly: true,
            allowMultiSelect: false,
            editMode: true,
        });
    } else if (library.store === "composer") {
        await showThumbnailBrowser(hosts.composer, "", "", {
            title,
            endpointPrefix: COMPOSER_ENDPOINT_PREFIX,
            loadPromptsFn: loadComposerPrompts,
            preferenceScope: "composer",
            promptOnly: true,
            allowMultiSelect: false,
            editMode: true,
        });
    } else {
        await showThumbnailBrowser(hosts.manager, "", "", {
            title,
            loadPromptsFn: loadManagerPrompts,
            contentFilter: library.filter,
            hideContentFilterControl: true,
            filterEmptyCategories: true,
            allowMultiSelect: false,
            editMode: true,
        });
    }
}

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .pm-hub { display: flex; flex-direction: column; gap: 12px; height: 100%; overflow: auto; padding: 12px;
            box-sizing: border-box; color: ${UI.textPrimary}; font: 14px/1.4 sans-serif; }
        .pm-hub-header { display: flex; align-items: center; gap: 8px; }
        .pm-hub-title { font-size: 16px; font-weight: 700; color: ${UI.textHeading}; margin-right: auto; }
        .pm-hub-muted { color: ${UI.textMuted}; font-size: 13px; }
        .pm-hub-section { border: 1px solid ${UI.panelBorder}; border-radius: 8px; background: ${UI.panel}; padding: 12px;
            display: flex; flex-direction: column; gap: 10px; }
        .pm-hub-section-head { display: flex; align-items: baseline; gap: 8px; }
        .pm-hub-name { font-size: 15px; font-weight: 700; color: ${UI.textHeading}; }
        .pm-hub-count { margin-left: auto; color: ${UI.textHint}; font-size: 13px; white-space: nowrap; }
        .pm-hub-kinds { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 6px; }
        .pm-hub-kind { display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: 6px; cursor: pointer;
            border: 1px solid ${UI.inputBorder}; background: ${UI.inputBg}; color: ${UI.textPrimary}; font: inherit; font-size: 13px; text-align: left; }
        .pm-hub-kind:hover { border-color: rgba(var(--pm-hub-rgb), 0.8); }
        .pm-hub-kind.active { border-color: rgba(var(--pm-hub-rgb), 0.95); background: rgba(var(--pm-hub-rgb), 0.18); }
        .pm-hub-kind-count { margin-left: auto; color: ${UI.textHint}; }
        .pm-hub-badge { flex: 0 0 20px; height: 20px; border-radius: 5px; display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 12px; color: #fff; background: rgba(var(--pm-hub-rgb), 0.9); }
        .pm-hub-actions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
        .pm-hub-btn { padding: 6px 10px; border-radius: 6px; border: 1px solid ${UI.inputBorder}; background: ${UI.buttonBg};
            color: ${UI.textPrimary}; font: inherit; font-size: 13px; cursor: pointer; white-space: nowrap; }
        .pm-hub-btn:hover:not(:disabled) { border-color: ${UI.accentBorder}; }
        .pm-hub-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .pm-hub-btn-primary { border-color: ${UI.accentBorder}; background: hsl(208 73% 57% / 0.2); color: #dbeafe; }
        .pm-hub-overlay { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.72); z-index: 9999; display: flex;
            align-items: center; justify-content: center; }
        .pm-hub-dialog { width: min(860px, calc(100vw - 48px)); max-height: min(760px, calc(100vh - 48px)); background: ${UI.panel};
            border: 1px solid ${UI.panelBorder}; border-radius: 10px; box-shadow: 0 10px 32px rgba(0, 0, 0, 0.45);
            color: ${UI.textPrimary}; display: flex; flex-direction: column; overflow: hidden; font: 14px/1.4 sans-serif; }
        .pm-hub-dialog-small { width: min(480px, calc(100vw - 48px)); }
        .pm-hub-dialog-header { padding: 16px 18px 10px; border-bottom: 1px solid ${UI.sectionBorder}; }
        .pm-hub-dialog-title { font-size: 16px; font-weight: 700; margin-bottom: 6px; }
        .pm-hub-dialog-grid { padding: 16px 18px; overflow: auto; display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
        .pm-hub-dialog-list { padding: 16px 18px; overflow: auto; display: flex; flex-direction: column; gap: 10px; }
        .pm-hub-dialog-footer { padding: 12px 18px 18px; border-top: 1px solid ${UI.sectionBorder}; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
        .pm-hub-pick { text-align: left; min-height: 64px; border: 1px solid ${UI.inputBorder}; border-radius: 10px; background: ${UI.cardBg};
            color: ${UI.textPrimary}; padding: 12px; cursor: pointer; display: flex; flex-direction: column; gap: 6px; font: inherit; }
        .pm-hub-pick:hover { border-color: ${UI.accentBorder}; }
        .pm-hub-pick.active { background: hsl(208 73% 57% / 0.16); border-color: ${UI.accentBorder}; }
        .pm-hub-pick-name { font-size: 15px; font-weight: 700; word-break: break-word; }
    `;
    document.head.appendChild(style);
}

function renderHub(container) {
    ensureStyle();
    container.replaceChildren();
    const root = document.createElement("div");
    root.className = "pm-hub";
    container.appendChild(root);

    const header = document.createElement("div");
    header.className = "pm-hub-header";
    const title = document.createElement("div");
    title.className = "pm-hub-title";
    title.textContent = HUB_TITLE;
    const refreshButton = makeButton("Refresh", { title: "Reload all libraries" });
    header.append(title, refreshButton);
    const status = document.createElement("div");
    status.className = "pm-hub-muted";
    root.append(header);

    let selectedKind = PROMPT_KINDS.find((kind) => kind.id === localStorage.getItem(KIND_STORAGE_KEY)) || PROMPT_KINDS[0];
    let busy = false;
    const updaters = [];
    const update = () => updaters.forEach((fn) => fn());

    const refresh = async () => {
        status.textContent = "Loading libraries…";
        const results = await Promise.allSettled([
            loadManagerPrompts(hosts.manager),
            loadComposerPrompts(hosts.composer),
            loadSystemPrompts(hosts.generator),
        ]);
        const failed = results.filter((result) => result.status === "rejected");
        failed.forEach((result) => console.error("[PromptLibraryHub]", result.reason));
        status.textContent = failed.length ? "Some libraries could not be loaded; see the console." : "";
        update();
    };

    const run = async (task) => {
        if (busy) return;
        busy = true;
        try {
            await task();
        } catch (error) {
            console.error("[PromptLibraryHub]", error);
            await showInfo(HUB_TITLE, error?.message || String(error));
        } finally {
            busy = false;
            await refresh();
        }
    };

    for (const section of SECTIONS) {
        const card = document.createElement("div");
        card.className = "pm-hub-section";
        const head = document.createElement("div");
        head.className = "pm-hub-section-head";
        const name = document.createElement("div");
        name.className = "pm-hub-name";
        name.textContent = section.title;
        const count = document.createElement("div");
        count.className = "pm-hub-count";
        head.append(name, count);
        const hint = document.createElement("div");
        hint.className = "pm-hub-muted";
        card.append(head);

        const current = () => libraryFor(section, section.kinds ? selectedKind : null);

        if (section.kinds) {
            const kinds = document.createElement("div");
            kinds.className = "pm-hub-kinds";
            for (const kind of section.kinds) {
                const chip = document.createElement("button");
                chip.type = "button";
                chip.className = "pm-hub-kind";
                chip.style.setProperty("--pm-hub-rgb", kind.color);
                const badge = document.createElement("span");
                badge.className = "pm-hub-badge";
                badge.textContent = kind.badge;
                const label = document.createElement("span");
                label.textContent = kind.label;
                const kindCount = document.createElement("span");
                kindCount.className = "pm-hub-kind-count";
                chip.append(badge, label, kindCount);
                chip.onclick = () => {
                    selectedKind = kind;
                    localStorage.setItem(KIND_STORAGE_KEY, kind.id);
                    update();
                };
                updaters.push(() => {
                    chip.classList.toggle("active", kind === selectedKind);
                    kindCount.textContent = String(summarize(libraryFor(section, kind)).entries);
                });
                kinds.appendChild(chip);
            }
            card.appendChild(kinds);
        }
        card.appendChild(hint);

        const actions = document.createElement("div");
        actions.className = "pm-hub-actions";
        const browse = makeButton("Browse & Edit", { primary: true });
        browse.onclick = () => run(() => browseLibrary(current()));
        const importButton = makeButton("Import");
        importButton.onclick = () => run(() => importLibrary(current()));
        const exportButton = makeButton("Export");
        exportButton.onclick = () => run(() => exportLibrary(current()));
        actions.append(browse, importButton, exportButton);
        card.appendChild(actions);

        updaters.push(() => {
            const library = current();
            const { entries, categories, groups } = summarize(library);
            const parts = [plural(entries, library.unit), `${categories} categor${categories === 1 ? "y" : "ies"}`];
            if (groups) parts.push(`${groups} group${groups === 1 ? "" : "s"}`);
            count.textContent = parts.join(" · ");
            hint.textContent = section.kinds ? selectedKind.hint : section.hint;
            browse.title = `Open ${library.title} in the Prompt Browser with editing on`;
            importButton.title = section.store === "manager"
                ? "Merge a JSON file into Saved Prompts (entries keep their own type)"
                : `Merge or replace ${library.title} from a file`;
            exportButton.title = section.store === "composer" ? "Export selected prompt groups as a ZIP" : "Export selected categories as JSON";
        });

        root.appendChild(card);
    }
    root.appendChild(status);

    refreshButton.onclick = () => run(async () => {});
    update();
    refresh();
}

const SHOW_SIDEBAR_SETTING = "PromptManager.LibraryHub.ShowSidebar";
const SIDEBAR_TAB_ID = "prompt-library-hub";
let sidebarTabRegistered = false;

function syncSidebarTab(show) {
    const manager = app.extensionManager;
    if (show && !sidebarTabRegistered && manager?.registerSidebarTab) {
        manager.registerSidebarTab({
            id: SIDEBAR_TAB_ID,
            icon: "pi pi-book",
            title: HUB_TITLE,
            tooltip: HUB_TITLE,
            label: "Prompts",
            type: "custom",
            render: (container) => renderHub(container),
        });
        sidebarTabRegistered = true;
    } else if (!show && sidebarTabRegistered && manager?.unregisterSidebarTab) {
        manager.unregisterSidebarTab(SIDEBAR_TAB_ID);
        sidebarTabRegistered = false;
    }
}

app.registerExtension({
    name: "PromptManager.LibraryHub",
    settings: [
        {
            id: SHOW_SIDEBAR_SETTING,
            category: ["Prompt Manager", "4. Advanced Preferences", "Show Prompts Sidebar Button"],
            name: "Show the Prompts button (Prompt Library Hub) in the sidebar",
            tooltip: "Adds a Prompts button to ComfyUI's left sidebar that opens the Prompt Library Hub.",
            type: "boolean",
            defaultValue: true,
            onChange: (value) => syncSidebarTab(value !== false),
        },
    ],
    setup() {
        const value = app.extensionManager?.setting?.get?.(SHOW_SIDEBAR_SETTING)
            ?? app.ui?.settings?.getSettingValue?.(SHOW_SIDEBAR_SETTING);
        syncSidebarTab(value !== false);
    },
});


