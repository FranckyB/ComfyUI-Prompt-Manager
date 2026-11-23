/**
 * Prompt Manager's side of the shared file browser (file_browser.js, the
 * same file as in FBnodes): its routes and Custom-folder setting, root
 * resolution, and the input/output/absolute selection classifier, so the
 * Prompt Extractor can browse anywhere while keeping old workflows working.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

export const SETTING_CUSTOM_PATH = "PromptManager.PromptExtractorCustomPath";

// This pack's routes for the shared file browser.
export const BROWSER_ROUTES = {
    list: "/prompt-extractor/path-browser/list",
    file: "/prompt-extractor/path-browser/file",
    remove: "/prompt-extractor/path-browser/delete",
    videoFrame: "/prompt-extractor/video-frame",
};

export function getCustomPresetPath() {
    const pref = app.ui?.settings?.getSettingValue(SETTING_CUSTOM_PATH) || "";
    return typeof pref === "string" ? pref.trim() : "";
}

/** Store the Custom folder (the same setting as in Preferences). */
export function setCustomPresetPath(path) {
    return app.ui.settings.setSettingValueAsync(SETTING_CUSTOM_PATH, String(path || "").trim());
}

/** URL that streams a media file from any path (for previews of out-of-tree files). */
export function mediaFileUrl(absPath) {
    return api.apiURL(`${BROWSER_ROUTES.file}?path=${encodeURIComponent(absPath)}`);
}

function normalizeSlashes(p) {
    return String(p || "").replace(/\\/g, "/");
}

/** True when `child` is inside `parent` (both absolute), case-insensitive on Windows. */
function isUnder(child, parent) {
    if (!child || !parent) return false;
    let c = normalizeSlashes(child).replace(/\/+$/, "");
    let p = normalizeSlashes(parent).replace(/\/+$/, "");
    if (navigator.platform.startsWith("Win")) {
        c = c.toLowerCase();
        p = p.toLowerCase();
    }
    return c === p || c.startsWith(p + "/");
}

/**
 * Given a selected absolute path and the input/output roots, work out how to
 * store it so old workflows keep working:
 *   - inside input  -> { sourceFolder:"input",  value:<relative> }
 *   - inside output -> { sourceFolder:"output", value:<relative> }
 *   - elsewhere     -> { sourceFolder:null,     value:<absolute> }
 */
export function classifySelection(absPath, roots) {
    const inputRoot = roots?.input || "";
    const outputRoot = roots?.output || "";
    const norm = normalizeSlashes(absPath);

    if (inputRoot && isUnder(absPath, inputRoot)) {
        const rel = norm.slice(normalizeSlashes(inputRoot).replace(/\/+$/, "").length + 1);
        return { sourceFolder: "input", value: rel, absPath };
    }
    if (outputRoot && isUnder(absPath, outputRoot)) {
        const rel = norm.slice(normalizeSlashes(outputRoot).replace(/\/+$/, "").length + 1);
        return { sourceFolder: "output", value: rel, absPath };
    }
    return { sourceFolder: null, value: absPath, absPath };
}

async function fetchBrowser(path, kind) {
    const params = [];
    if (path) params.push(`path=${encodeURIComponent(path)}`);
    if (kind) params.push(`kind=${encodeURIComponent(kind)}`);
    const query = params.length ? `?${params.join("&")}` : "";
    const resp = await api.fetchApi(`${BROWSER_ROUTES.list}${query}`);
    if (!resp.ok) {
        let msg = `Request failed (${resp.status})`;
        try {
            const err = await resp.json();
            if (err?.error) msg = err.error;
        } catch {
            // ignore
        }
        throw new Error(msg);
    }
    return await resp.json();
}

/** Resolve the input/output preset roots once (backend "roots" mode). */
export async function getMediaRoots() {
    try {
        const data = await fetchBrowser("", "");
        const roots = Array.isArray(data?.roots) ? data.roots : [];
        return { input: roots[0] || "", output: roots[1] || "" };
    } catch {
        return { input: "", output: "" };
    }
}

// Register the shared Custom-path preference once.
app.registerExtension({
    name: "PromptManager.PromptExtractorPathBrowser",
    settings: [
        {
            id: SETTING_CUSTOM_PATH,
            category: ["Prompt Manager", "Prompt Extractor", "Custom Browser Path"],
            name: "Prompt Extractor Custom Browser Path",
            tooltip: "The folder the Custom button opens in the Prompt Extractor's file browser.",
            type: "text",
            defaultValue: "",
        },
    ],
});

