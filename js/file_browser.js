/**
 * File browser modal for the packs' loaders and lists (FBnodes' Load+
 * nodes, Stitchers and LoRA pickers, Prompt Manager's Prompt Extractor):
 * browse any folder on disk, preview thumbnails, pick a file (or several
 * with multiSelect). Built on the FB UI kit (js/fbui/).
 * The same file in every pack (tools/sync_fbui.py copies it): the pack's own
 * path_browser.js names its routes and its Custom-folder setting.
 *
 * createFileBrowserModal(currentFile, onFileSelect, sourceFolder, options)
 *   currentFile     the value now in the node (relative to initialPath, or absolute)
 *   onFileSelect    (absPath, { absPath, dir, roots }) when a file is picked
 *   sourceFolder    "input" | "output": the root to open when nothing else says
 *   options:
 *     initialPath, selectedAbsPath     the folder to open, the file to highlight
 *     navKind                          what the backend lists: media | video | audiovideo | lora ...
 *     allowedTypes                     types shown: image | video | audio | json | other
 *     filterTypeOptions, defaultFilter the Type filter's choices and its default
 *     showListKindSelector             the Type filter offers All / Audio / Video
 *     multiSelect, onMultiSelect([{ value, meta }])
 *     places                           quick folders instead of Input / Output:
 *                                      [{ label, title, path() }] (path may be async)
 *     custom                           the Custom folder: { get(), set(path), message };
 *                                      by default the pack's own; null: no Custom button
 *     canDelete                        false: no "Delete file…" (a file that is more than
 *                                      one file on disk)
 */

import { api } from "../../scripts/api.js";
import { BROWSER_ROUTES, getCustomPresetPath, mediaFileUrl, setCustomPresetPath } from "./path_browser.js";
import { toast } from "./fbui/comfy.js";
import { button, ICONS } from "./fbui/controls.js";
import { el } from "./fbui/dom.js";
import { confirm, openMenu, openModal, prompt } from "./fbui/modal.js";
import { FONT, RADIUS, RADIUS_CARD, ROW_HEIGHT, T } from "./fbui/theme.js";

const PLACEHOLDER = new URL("./placeholder.png", import.meta.url).href;
const VIEW_KEY = "FBnodes.FileBrowser.ViewMode";
const VIEWS = {
    grid: { icon: ICONS.grid, title: "Large thumbnails" },
    icon: { icon: ICONS.tiles, title: "Small thumbnails" },
    list: { icon: ICONS.list, title: "Details list" },
};
// Video thumbnails are kept in localStorage; "Regenerate cache" clears every
// key with the shorter prefix (older releases used it too).
const THUMB_KEY = "video_thumb_pyav_";
const THUMB_PREFIX = "video_thumb_";

// The types the loaders accept (gif, mkv, ogg ... are listed by the backend
// for some kinds but read as "other", so allowedTypes keeps them out).
const EXTENSIONS = {
    image: ["png", "jpg", "jpeg", "webp"],
    video: ["mp4", "webm", "mov", "avi"],
    audio: ["wav", "flac", "mp3", "m4a"],
    json: ["json"],
};
const LORA_EXTENSIONS = ["safetensors", "ckpt", "pt", "bin", "pth"];
const TYPE_LABELS = { image: "Image", video: "Video", audio: "Audio", json: "JSON", other: "File" };
const FILTER_LABELS = { all: "All files", image: "Images", video: "Videos", audio: "Audio", json: "JSON", other: "Other files" };

const extensionOf = (path) => String(path || "").split(".").pop().toLowerCase();
const basename = (path) => String(path || "").replace(/\\/g, "/").replace(/\/+$/, "").split("/").pop();
const isAbsolute = (path) => /^([a-zA-Z]:[\\/]|\\\\|\/)/.test(path || "");
const pathKey = (path) => String(path || "").replace(/\\/g, "/").replace(/\/+/g, "/").replace(/\/$/, "").toLowerCase();
const samePath = (a, b) => pathKey(a) === pathKey(b);
const joinPath = (dir, leaf) => `${String(dir).replace(/[\\/]+$/, "")}/${String(leaf).replace(/^[\\/]+/, "")}`;
const parentOf = (path) => String(path || "").replace(/[\\/][^\\/]*$/, "");
const compareText = (a, b) => String(a || "").localeCompare(String(b || ""), undefined, { numeric: true, sensitivity: "base" });

function fileType(path) {
    const ext = extensionOf(path);
    for (const [type, list] of Object.entries(EXTENSIONS)) if (list.includes(ext)) return type;
    return "other";
}

function formatSize(bytes) {
    const n = Number(bytes);
    if (!Number.isFinite(n) || n < 0) return "—";
    if (n < 1024) return `${n} B`;
    if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
    return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function toMillis(raw) {
    if (raw == null || raw === "") return null;
    const ms = new Date(typeof raw === "number" && raw > 0 && raw < 1e12 ? raw * 1000 : raw).getTime();
    return Number.isFinite(ms) ? ms : null;
}

function formatDate(raw) {
    const ms = toMillis(raw);
    if (ms === null) return "—";
    const date = new Date(ms);
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

function formatDuration(raw) {
    const n = Number(raw);
    if (!Number.isFinite(n) || n < 0) return "";
    const total = Math.round(n);
    const pad = (value) => String(value).padStart(2, "0");
    const hours = Math.floor(total / 3600);
    const rest = `${pad(Math.floor((total % 3600) / 60))}:${pad(total % 60)}`;
    return hours > 0 ? `${pad(hours)}:${rest}` : rest;
}

function fileDetails(file) {
    if (file.details) return file.details;
    const duration = formatDuration(file.duration);
    if (file.type === "video") return [duration, file.width && file.height ? `${file.width}×${file.height}` : ""].filter(Boolean).join(" · ") || "—";
    if (file.type === "audio") return duration || "—";
    return "—";
}

function normalizeFile(entry) {
    const path = typeof entry === "string" ? entry : entry?.path || entry?.filename || entry?.name || "";
    const data = typeof entry === "object" && entry ? entry : {};
    return {
        path,
        name: data.name || basename(path),
        type: fileType(path),
        size: data.size ?? data.bytes ?? null,
        modified: data.modified ?? data.mtime ?? data.date ?? data.timestamp ?? null,
        duration: data.duration ?? null,
        width: data.width ?? null,
        height: data.height ?? null,
        // Set by listings that know more about their files (RefMods).
        preview: data.preview ?? null,
        details: data.details ?? null,
    };
}

// Backend roots come as ["input", "output"] paths or as [{ id, path }].
function rootPaths(roots) {
    if (!Array.isArray(roots)) return { input: "", output: "" };
    if (typeof roots[0] === "string") return { input: roots[0] || "", output: roots[1] || "" };
    const byId = (id, index) => roots.find((root) => root?.id === id)?.path || roots[index]?.path || "";
    return { input: byId("input", 0), output: byId("output", 1) };
}

async function fetchListing(path, kind) {
    const params = new URLSearchParams();
    if (path) params.set("path", path);
    if (kind) params.set("kind", kind);
    const response = await api.fetchApi(`${BROWSER_ROUTES.list}?${params}`);
    const data = await response.json().catch(() => null);
    if (!response.ok) throw new Error(data?.error || `Request failed (${response.status})`);
    return data;
}

function storedView() {
    try {
        const value = localStorage.getItem(VIEW_KEY);
        // "large" / "medium" / "detail" are the names older releases stored.
        return { large: "grid", medium: "icon", detail: "list" }[value] ?? (VIEWS[value] ? value : "grid");
    } catch {
        return "grid";
    }
}

// ---------------------------------------------------------------- thumbnails

const thumbKey = (path) => `${THUMB_KEY}${String(path).replace(/[\/\\]/g, "_")}`;

function showImage(holder, src, { fallbacks = [] } = {}) {
    const img = el("img");
    img.loading = "lazy";
    img.decoding = "async";
    img.draggable = false;
    const queue = [...fallbacks];
    img.onerror = () => {
        const next = queue.shift();
        img.src = next ?? PLACEHOLDER;
        if (!next) img.onerror = null;
    };
    img.src = src;
    holder.replaceChildren(img);
}

// The first frame, decoded by the backend (works for H265 / yuv444 too),
// shrunk and cached so a folder opens instantly next time.
function showVideoThumb(holder, path, { refresh = false } = {}) {
    const key = thumbKey(path);
    try {
        if (refresh) localStorage.removeItem(key);
        const cached = localStorage.getItem(key);
        if (cached) return showImage(holder, cached);
    } catch {
        // No localStorage: decode every time.
    }
    holder.replaceChildren(el("span", "spin"));
    const img = new Image();
    img.onload = () => {
        const scale = Math.min(1, 320 / img.naturalWidth, 240 / img.naturalHeight);
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
        canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
        canvas.getContext("2d", { alpha: false }).drawImage(img, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
        showImage(holder, dataUrl);
        try {
            localStorage.setItem(key, dataUrl);
        } catch {
            // Storage full: the thumbnail still shows, it is just not kept.
        }
    };
    img.onerror = () => holder.replaceChildren(iconFor("video"), el("span", "note", "Preview unavailable"));
    img.src = api.apiURL(`${BROWSER_ROUTES.videoFrame}?${new URLSearchParams({ filename: path, position: "0" })}`);
}

// Resting on a video plays it in its tile, muted and looping (as AusBoss's
// media picker does); one the browser cannot play keeps its still.
const HOVER_PLAY_DELAY = 150;
function playOnHover(thumb, path) {
    let timer = 0;
    let player = null;
    const stop = () => {
        clearTimeout(timer);
        if (!player) return;
        player.removeAttribute("src");
        player.load();
        player.remove();
        player = null;
    };
    thumb.addEventListener("pointerenter", () => {
        timer = setTimeout(() => {
            player = Object.assign(el("video", "hover-play"), { muted: true, loop: true, autoplay: true, playsInline: true });
            player.addEventListener("error", stop);
            player.src = mediaFileUrl(path);
            thumb.append(player);
        }, HOVER_PLAY_DELAY);
    });
    thumb.addEventListener("pointerleave", stop);
}

function clearThumbCache() {
    try {
        const keys = [];
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key?.startsWith(THUMB_PREFIX)) keys.push(key);
        }
        for (const key of keys) localStorage.removeItem(key);
    } catch {
        // Nothing cached.
    }
}

function iconFor(type) {
    const holder = el("span", "glyph");
    holder.innerHTML = { folder: ICONS.folder, up: ICONS.up, video: ICONS.film, audio: ICONS.music }[type] ?? ICONS.file;
    return holder;
}

function fillThumb(holder, file) {
    // A listing may name a file's own preview picture (RefMods do).
    if (file.preview) return showImage(holder, mediaFileUrl(file.preview));
    if (file.type === "image") return showImage(holder, mediaFileUrl(file.path));
    if (file.type === "video") return showVideoThumb(holder, file.path);
    // A LoRA shows the picture saved beside it with the same base name.
    if (LORA_EXTENSIONS.includes(extensionOf(file.path))) {
        const base = file.path.slice(0, -(extensionOf(file.path).length + 1));
        const [first, ...rest] = ["png", "jpg", "jpeg", "webp"].map((ext) => mediaFileUrl(`${base}.${ext}`));
        return showImage(holder, first, { fallbacks: rest });
    }
    holder.replaceChildren(iconFor(file.type));
}

// ---------------------------------------------------------------- styles

const CSS_ID = "fbnodes-browser-css-v2";

function ensureBrowserCss() {
    if (document.getElementById(CSS_ID)) return;
    const style = document.createElement("style");
    style.id = CSS_ID;
    style.textContent = `
.fbnodes-browser .fbui-modal-body{display:flex;flex-direction:column;padding:0;overflow:hidden}
.fbnodes-browser-bar{flex:none;display:flex;align-items:center;gap:6px;padding:0 12px 8px}
.fbnodes-browser-bar .fbui-text{flex:1 1 220px;min-width:0}
.fbnodes-browser-search{position:relative;flex:1 1 220px;min-width:0;display:flex}
.fbnodes-browser-search .fbui-text{padding:0 28px}
.fbnodes-browser-search>.lead{position:absolute;left:9px;top:50%;transform:translateY(-50%);color:${T.faint};pointer-events:none;display:flex}
.fbnodes-browser-search>.fbui-btn{position:absolute;right:3px;top:3px;height:20px;width:20px;background:transparent}
.fbnodes-browser-bar>.fbui-select{width:auto;flex:none;min-width:120px}
.fbnodes-browser-view{flex:none;width:96px}
.fbnodes-browser-view button{display:grid;place-items:center}
.fbnodes-browser-items{flex:1 1 auto;min-height:0;overflow:auto;display:grid;gap:12px;padding:4px 12px 12px;align-content:start}
.fbnodes-browser-items.grid{grid-template-columns:repeat(auto-fill,minmax(260px,1fr));--thumb:220px}
.fbnodes-browser-items.icon{grid-template-columns:repeat(auto-fill,minmax(150px,1fr));--thumb:120px}
.fbnodes-browser-items.list{display:block;padding-top:0}
.fbnodes-browser-card{display:flex;flex-direction:column;gap:6px;padding:6px;border:1px solid ${T.plateEdge};border-radius:${RADIUS_CARD}px;background:${T.plate};cursor:pointer;min-width:0}
.fbnodes-browser-card:hover{border-color:${T.brand}}
.fbnodes-browser-card .thumb{position:relative;height:var(--thumb);border-radius:${RADIUS}px;background:${T.card};display:flex;align-items:center;justify-content:center;flex-direction:column;gap:6px;overflow:hidden;color:${T.faint}}
.fbnodes-browser-card .thumb img{width:100%;height:100%;object-fit:contain;display:block}
.fbnodes-browser-card .thumb .hover-play{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:${T.card}}
.fbnodes-browser-card .glyph{display:flex;color:${T.label}}
.fbnodes-browser-card .glyph svg{width:42px;height:42px;stroke-width:1.1}
.fbnodes-browser-items.icon .glyph svg{width:30px;height:30px}
.fbnodes-browser-card .chip-time{position:absolute;z-index:1;right:6px;bottom:6px;padding:2px 6px;border-radius:999px;background:rgba(0,0,0,.65);color:${T.ink};font:600 10px ${FONT};font-variant-numeric:tabular-nums}
.fbnodes-browser-card .name{overflow:hidden;white-space:nowrap;text-overflow:ellipsis;text-align:center;color:${T.inkSoft};font:12px ${FONT}}
.fbnodes-browser-items .note{font:11px ${FONT};color:${T.faint}}
.fbnodes-browser-items .spin{width:18px;height:18px;border-radius:50%;border:2px solid ${T.fill};border-top-color:${T.brand};animation:fbnodes-browser-spin .8s linear infinite}
@keyframes fbnodes-browser-spin{to{transform:rotate(360deg)}}
.fbnodes-browser-row{display:grid;grid-template-columns:24px minmax(0,1fr) 70px 150px 80px 140px;align-items:center;gap:10px;height:32px;padding:0 8px;border-bottom:1px solid ${T.rule};cursor:pointer;color:${T.label};font:11px ${FONT};font-variant-numeric:tabular-nums}
.fbnodes-browser-row:hover{background:${T.fill}}
.fbnodes-browser-row>*{overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.fbnodes-browser-row .name{color:${T.inkSoft};font-size:12px}
.fbnodes-browser-row .size{text-align:right}
.fbnodes-browser-row .thumb{width:24px;height:24px;border-radius:4px;background:${T.card};display:grid;place-items:center;overflow:hidden;color:${T.label}}
.fbnodes-browser-row .thumb img{width:100%;height:100%;object-fit:cover}
.fbnodes-browser-row .thumb svg{width:14px;height:14px}
.fbnodes-browser-row .thumb .spin{width:10px;height:10px;border-width:1.5px}
.fbnodes-browser-head{position:sticky;top:0;z-index:1;height:30px;background:${T.panel};cursor:default;color:${T.faint};font:600 10px ${FONT};letter-spacing:.08em;text-transform:uppercase}
.fbnodes-browser-head:hover{background:${T.panel}}
.fbnodes-browser-head button{all:unset;cursor:pointer;overflow:hidden;text-overflow:ellipsis}
.fbnodes-browser-head button:hover,.fbnodes-browser-head button.on{color:${T.ink}}
.fbnodes-browser-head .size{text-align:right}
.fbnodes-browser-items .selected{border-color:${T.brand};background:${T.brandSoft};box-shadow:0 0 0 1px ${T.brand}}
.fbnodes-browser-items .selected .name{color:${T.ink}}
.fbnodes-browser-row.selected{box-shadow:inset 3px 0 0 ${T.brand}}
.fbnodes-browser-items .cursor{outline:2px solid ${T.brandInk};outline-offset:1px}
.fbnodes-browser-items .playing{box-shadow:inset 0 0 0 1px ${T.brand}}
.fbnodes-browser-items .came-from{border-color:${T.brandInk}}
.fbnodes-browser-empty{grid-column:1/-1;padding:48px 12px;text-align:center;color:${T.faint};font:12px ${FONT}}
.fbnodes-browser-empty.error{color:${T.dangerInk}}
.fbnodes-browser .count{margin-right:auto;align-self:center;color:${T.label};font:12px ${FONT}}
`;
    document.head.append(style);
}

// ---------------------------------------------------------------- the modal

// Firefox on Linux turns the mouse's side buttons into the browser's own
// Back / Forward without sending the page any mouse event; all the page sees
// is the history move. So while the browser is open the page stands on a
// history entry of its own, with one entry behind it and one ahead: a move
// either way is read as that button (onPress(true) for Back), swallowed
// before ComfyUI's router hears of it, and undone. Release steps back to
// the entry the page started on, once any move still in flight has landed.
// The entries reuse the router's own state, so the URL and the router's
// position never change.
function trapHistoryButtons(onPress) {
    const MARK = "fbnodesBrowser";
    const base = history.state;
    let pending = 1; // the back() below, onto the middle entry
    let closing = false;
    let finalStepDone = false;
    const onPop = (event) => {
        event.stopImmediatePropagation();
        if (pending > 0) {
            pending -= 1;
            if (pending > 0 || !closing) return;
            if (!finalStepDone) {
                finalStepDone = true;
                pending = 1;
                history.back();
                return;
            }
            window.removeEventListener("popstate", onPop, true);
            return;
        }
        if (closing) return;
        pending = 1;
        if (event.state?.[MARK] === "ahead") {
            history.back();
            onPress(false);
        } else {
            history.forward();
            onPress(true);
        }
    };
    window.addEventListener("popstate", onPop, true);
    history.pushState({ ...base, [MARK]: "middle" }, "");
    history.pushState({ ...base, [MARK]: "ahead" }, "");
    history.back();
    return () => {
        closing = true;
        if (pending > 0) return; // the last move lands first, then steps back
        finalStepDone = true;
        pending = 1;
        history.back();
    };
}

export function createFileBrowserModal(currentFile, onFileSelect, sourceFolder, options = {}) {
    ensureBrowserCss();
    const multi = options.multiSelect === true;
    const allowed = Array.isArray(options.allowedTypes) && options.allowedTypes.length
        ? options.allowedTypes.map((type) => String(type).toLowerCase())
        : null;
    const defaultFilters = options.showListKindSelector ? ["all", "audio", "video"] : ["all", "image", "video", "audio", "json"];
    const filters = (Array.isArray(options.filterTypeOptions) ? options.filterTypeOptions : defaultFilters)
        .map((type) => String(type).toLowerCase())
        .filter((type, index, list) => FILTER_LABELS[type] && list.indexOf(type) === index);

    let current = isAbsolute(options.selectedAbsPath) ? options.selectedAbsPath : currentFile || "";
    const state = {
        dir: options.initialPath || (isAbsolute(current) ? parentOf(current) : ""),
        parent: null,
        roots: { input: "", output: "" },
        kind: options.navKind || "media",
        view: storedView(),
        sort: { key: "name", dir: 1 },
        filter: options.defaultFilter && filters.includes(options.defaultFilter) ? options.defaultFilter : filters[0] ?? "all",
        search: "",
        dirs: [],
        files: [],
        items: [], // { kind: "dir" | "file", entry, element }
        cursor: null,
        cameFrom: "",
        picked: new Map(), // multiSelect: path -> meta
        down: [], // folders left by going up, newest last (Forward side button)
        firstLoad: true,
        request: 0,
        audio: null,
        playing: null,
    };
    // A relative current value lives in the folder the browser opens on.
    if (current && !isAbsolute(current) && state.dir) current = joinPath(state.dir, current);

    const metaFor = (path) => ({ absPath: path, dir: parentOf(path) || state.dir, roots: { ...state.roots } });

    // ---- toolbar: where
    const pathField = el("input", "fbui-text");
    pathField.type = "text";
    pathField.spellcheck = false;
    pathField.placeholder = "Paste a folder path and press Enter";
    pathField.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        const next = pathField.value.trim();
        if (next) go(next);
    });
    const custom = options.custom === null ? null : options.custom ?? {
        get: getCustomPresetPath,
        set: setCustomPresetPath,
        message: "The folder the Custom button opens, in every Load+ browser.",
    };
    const customButton = button("Custom", { onClick: () => {
        const path = custom.get();
        if (path) go(path);
        else askCustomPath();
    } });
    customButton.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        event.stopPropagation();
        askCustomPath();
    });
    const updateCustomTitle = () => {
        if (!custom) return;
        const path = custom.get();
        customButton.title = path ? `${path}\nRight-click to choose another folder` : "Click to choose your Custom folder";
    };
    // Asks for the Custom folder (this one, to start with), keeps it in its
    // setting and opens it.
    async function askCustomPath() {
        const path = await prompt({
            title: "Custom folder",
            message: custom.message,
            value: custom.get() || state.dir,
            placeholder: "/path/to/folder",
            confirmText: "Set and open",
        });
        if (!path) return;
        try {
            await custom.set(path);
        } catch (error) {
            toast(`Could not save the Custom folder: ${error.message}`, { severity: "error" });
        }
        updateCustomTitle();
        go(path);
    }
    updateCustomTitle();

    const places = options.places ?? [
        { label: "Input", title: "ComfyUI's input folder", path: () => state.roots.input },
        { label: "Output", title: "ComfyUI's output folder", path: () => state.roots.output },
    ];
    const whereBar = el("div", "fbnodes-browser-bar");
    whereBar.append(
        button("", { icon: ICONS.up, title: "Up one folder (Backspace, mouse Back button)", onClick: () => goUp() }),
        button("", { icon: ICONS.refresh, title: "Reload this folder", onClick: () => load() }),
        ...places.map((place) => button(place.label, { title: place.title, onClick: async () => {
            const path = await place.path();
            if (path) go(path);
        } })),
        ...(custom ? [customButton] : []),
        pathField,
        ...(state.kind === "lora" ? [] : [button("Regenerate cache", { icon: ICONS.film, title: "Rebuild every video thumbnail", onClick: () => { clearThumbCache(); load(); } })]),
    );

    // ---- toolbar: what
    const searchField = el("input", "fbui-text");
    searchField.type = "text";
    searchField.spellcheck = false;
    searchField.placeholder = "Search files…";
    const clearSearch = button("", { icon: ICONS.close, small: true, title: "Clear search", onClick: () => {
        searchField.value = "";
        state.search = "";
        applyFilter();
        searchField.focus();
    } });
    searchField.addEventListener("input", () => {
        state.search = searchField.value;
        applyFilter();
    });
    const search = el("div", "fbnodes-browser-search");
    const lead = el("span", "lead");
    lead.innerHTML = ICONS.search;
    search.append(lead, searchField, clearSearch);

    const typeSelect = el("select", "fbui-select");
    for (const type of filters) {
        const option = el("option", "", FILTER_LABELS[type]);
        option.value = type;
        typeSelect.append(option);
    }
    typeSelect.value = state.filter;
    typeSelect.addEventListener("change", () => {
        state.filter = typeSelect.value;
        applyFilter();
    });

    const viewSwitch = el("div", "fbui-seg fbnodes-browser-view");
    const viewButtons = {};
    for (const [mode, view] of Object.entries(VIEWS)) {
        const choice = el("button");
        choice.type = "button";
        choice.title = view.title;
        choice.innerHTML = view.icon;
        choice.addEventListener("click", () => setView(mode));
        viewButtons[mode] = choice;
        viewSwitch.append(choice);
    }

    const whatBar = el("div", "fbnodes-browser-bar");
    whatBar.append(
        button("", { icon: ICONS.chevronUp, title: "Previous folder beside this one (Ctrl+↑)", onClick: () => goSibling(-1) }),
        button("", { icon: ICONS.chevronDown, title: "Next folder beside this one (Ctrl+↓)", onClick: () => goSibling(1) }),
        search,
        typeSelect,
        viewSwitch,
    );

    const items = el("div", "fbnodes-browser-items");

    // ---- footer (multiSelect)
    let footer = null;
    let countLabel = null;
    let addButton = null;
    if (multi) {
        countLabel = el("span", "count");
        addButton = button("Add selected", { primary: true, onClick: () => addPicked() });
        footer = [countLabel, button("Cancel", { onClick: () => modal.close() }), addButton];
    }

    const modal = openModal({
        title: multi ? "Select files" : "Select a file",
        body: [whereBar, whatBar, items],
        footer,
        width: "min(1304px, 95vw)",
        height: "min(1185px, 92vh)",
        onClose: () => {
            stopAudio();
            document.removeEventListener("keydown", onKey, true);
            for (const type of sideButtonEvents) window.removeEventListener(type, onSideButton, true);
            releaseHistory();
        },
    });
    modal.overlay.classList.add("fbnodes-browser");
    modal.modal.style.maxWidth = "none";
    modal.modal.style.maxHeight = "none";
    // Our own menus only; the browser's would offer "Back" over a file.
    modal.overlay.addEventListener("contextmenu", (event) => event.preventDefault());
    // The mouse's side buttons: Back goes up a folder, Forward goes back down
    // into the folder last left that way. Chromium browsers send them as
    // mouse events (button 3 / 4); acting on pointerup matters because a page
    // handler that cancels pointerdown stops mouseup from being sent at all.
    // Every side-button event is swallowed, so the page never navigates.
    // Firefox on Linux sends no mouse event at all, only a history move,
    // which trapHistoryButtons catches.
    let lastSidePress = 0;
    const sideAction = (back) => {
        if (!modal.isTop()) return;
        if (back) goUp();
        else goDown();
    };
    const sideButtonEvents = ["pointerdown", "pointerup", "mousedown", "mouseup", "auxclick"];
    const onSideButton = (event) => {
        if (event.button !== 3 && event.button !== 4) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.type !== "pointerup") return;
        lastSidePress = performance.now();
        sideAction(event.button === 3);
    };
    for (const type of sideButtonEvents) window.addEventListener(type, onSideButton, true);
    const releaseHistory = trapHistoryButtons((back) => {
        // A press the pointer events already handled.
        if (performance.now() - lastSidePress > 500) sideAction(back);
    });

    // ---- picking
    function updateCount() {
        if (!multi) return;
        const n = state.picked.size;
        countLabel.textContent = `${n} selected`;
        addButton.disabled = n === 0;
        addButton.querySelector("span").textContent = n ? `Add ${n} clip${n === 1 ? "" : "s"}` : "Add selected";
    }

    function addPicked() {
        if (!state.picked.size) return;
        const picked = [...state.picked].map(([value, meta]) => ({ value, meta }));
        if (typeof options.onMultiSelect === "function") options.onMultiSelect(picked);
        else onFileSelect(picked.map((item) => item.value), { multi: true, items: picked });
        modal.close();
    }

    function isPicked(path) {
        return multi ? state.picked.has(path) : samePath(path, current);
    }

    function openItem(item) {
        if (item.kind === "dir") return go(item.entry.path);
        if (item.kind === "up") return goUp();
        const path = item.entry.path;
        if (multi) {
            if (state.picked.has(path)) state.picked.delete(path);
            else state.picked.set(path, metaFor(path));
            item.element.classList.toggle("selected", state.picked.has(path));
            updateCount();
            return;
        }
        onFileSelect(path, metaFor(path));
        modal.close();
    }

    // ---- audio preview: one click plays or stops, a double click picks
    function stopAudio() {
        if (state.audio) {
            state.audio.pause();
            state.audio.removeAttribute("src");
        }
        state.playing?.classList.remove("playing");
        state.playing = null;
    }

    async function toggleAudio(item) {
        const wasPlaying = state.playing === item.element;
        stopAudio();
        if (wasPlaying) return;
        state.audio ??= Object.assign(new Audio(), { preload: "metadata", onended: stopAudio });
        state.playing = item.element;
        item.element.classList.add("playing");
        try {
            state.audio.src = mediaFileUrl(item.entry.path);
            await state.audio.play();
        } catch (error) {
            stopAudio();
            toast(`Could not play ${item.entry.name}: ${error.message}`, { severity: "warn" });
        }
    }

    // ---- context menu
    function showMenu(event, item) {
        const file = item.entry;
        const entries = [];
        if (file.type === "video") {
            entries.push({ label: "Refresh thumbnail", action: () => showVideoThumb(item.element.querySelector(".thumb"), file.path, { refresh: true }) });
        }
        if (options.canDelete !== false) entries.push({ label: "Delete file…", danger: true, action: () => deleteFile(file) });
        if (entries.length) openMenu(event.clientX, event.clientY, entries);
    }

    async function deleteFile(file) {
        const sure = await confirm({ title: "Delete file", message: `Delete "${file.name}"? This cannot be undone.`, confirmText: "Delete", danger: true });
        if (!sure) return;
        try {
            const response = await api.fetchApi(BROWSER_ROUTES.remove, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: file.path }),
            });
            const data = await response.json().catch(() => ({}));
            if (!data.ok) throw new Error(data.error || `Request failed (${response.status})`);
            try { localStorage.removeItem(thumbKey(file.path)); } catch { /* nothing cached */ }
            state.picked.delete(file.path);
            updateCount();
            load();
        } catch (error) {
            toast(`Could not delete ${file.name}: ${error.message}`, { severity: "error" });
        }
    }

    // ---- building the items
    function makeCard(item) {
        const { kind, entry } = item;
        const card = el("div", "fbnodes-browser-card");
        const thumb = el("div", "thumb");
        if (kind === "file") {
            fillThumb(thumb, entry);
            const time = (entry.type === "video" || entry.type === "audio") && formatDuration(entry.duration);
            if (time) thumb.append(el("span", "chip-time", time));
            if (entry.type === "video") playOnHover(thumb, entry.path);
        } else {
            thumb.append(iconFor(kind === "up" ? "up" : "folder"));
        }
        const name = el("div", "name", kind === "up" ? "Up" : entry.name);
        name.title = entry.path || "";
        card.append(thumb, name);
        return card;
    }

    function makeRow(item) {
        const { kind, entry } = item;
        const row = el("div", "fbnodes-browser-row");
        const thumb = el("div", "thumb");
        if (kind === "file") {
            if (entry.type === "image") showImage(thumb, mediaFileUrl(entry.path));
            else if (entry.type === "video") showVideoThumb(thumb, entry.path);
            else thumb.append(iconFor(entry.type));
        } else {
            thumb.append(iconFor(kind === "up" ? "up" : "folder"));
        }
        const name = el("span", "name", kind === "up" ? ".." : entry.name);
        name.title = entry.path || "";
        const isFile = kind === "file";
        row.append(
            thumb,
            name,
            el("span", "", isFile ? TYPE_LABELS[entry.type] : kind === "up" ? "Parent" : "Folder"),
            el("span", "", isFile ? fileDetails(entry) : "—"),
            el("span", "size", isFile ? formatSize(entry.size) : "—"),
            el("span", "", isFile ? formatDate(entry.modified) : "—"),
        );
        return row;
    }

    function makeHeader() {
        const head = el("div", "fbnodes-browser-row fbnodes-browser-head");
        head.append(el("span"));
        for (const [key, label] of [["name", "Name"], ["type", "Type"], ["details", "Details"], ["size", "Size"], ["modified", "Modified"]]) {
            const cell = el("button", key === "size" ? "size" : "", label + (state.sort.key === key ? (state.sort.dir > 0 ? " ▲" : " ▼") : ""));
            cell.type = "button";
            cell.classList.toggle("on", state.sort.key === key);
            cell.addEventListener("click", () => {
                state.sort = { key, dir: state.sort.key === key ? -state.sort.dir : 1 };
                render();
            });
            head.append(cell);
        }
        return head;
    }

    function compareFiles(a, b) {
        const byName = compareText(a.name, b.name);
        const number = (x, y) => {
            const okX = Number.isFinite(x);
            const okY = Number.isFinite(y);
            if (!okX || !okY) return okX ? -1 : okY ? 1 : 0;
            return x - y;
        };
        let result = 0;
        switch (state.sort.key) {
            case "type": result = compareText(TYPE_LABELS[a.type], TYPE_LABELS[b.type]); break;
            case "size": result = number(Number(a.size), Number(b.size)); break;
            case "modified": result = number(toMillis(a.modified) ?? NaN, toMillis(b.modified) ?? NaN); break;
            case "details": result = a.type === b.type && (a.type === "video" || a.type === "audio") ? number(Number(a.duration), Number(b.duration)) : 0; break;
        }
        return (result || byName) * state.sort.dir;
    }

    function render() {
        stopAudio();
        const list = state.view === "list";
        items.className = `fbnodes-browser-items ${state.view}`;
        items.replaceChildren();
        if (list) items.append(makeHeader());
        state.items = [];
        const add = (item) => {
            item.element = list ? makeRow(item) : makeCard(item);
            wire(item);
            state.items.push(item);
            items.append(item.element);
        };
        if (state.parent && list) add({ kind: "up", entry: { path: state.parent, name: ".." } });
        for (const dir of state.dirs) add({ kind: "dir", entry: dir });
        for (const file of [...state.files].sort(compareFiles)) add({ kind: "file", entry: file });
        applyFilter();
    }

    function wire(item) {
        const element = item.element;
        if (item.kind === "dir" && samePath(item.entry.path, state.cameFrom)) element.classList.add("came-from");
        if (item.kind !== "file") {
            element.addEventListener("click", () => openItem(item));
            return;
        }
        element.classList.toggle("selected", isPicked(item.entry.path));
        element.addEventListener("click", () => {
            setCursor(item);
            if (item.entry.type === "audio" && !multi) toggleAudio(item);
            else openItem(item);
        });
        // Audio plays on a click, so a double click is what picks it.
        if (item.entry.type === "audio" && !multi) {
            element.addEventListener("dblclick", (event) => {
                event.preventDefault();
                openItem(item);
            });
        }
        element.addEventListener("contextmenu", (event) => {
            event.preventDefault();
            event.stopPropagation();
            showMenu(event, item);
        });
    }

    function applyFilter() {
        clearSearch.hidden = !state.search;
        const needle = state.search.toLowerCase();
        let shown = 0;
        for (const item of state.items) {
            if (item.kind !== "file") continue;
            const type = item.entry.type;
            const visible = (!needle || item.entry.name.toLowerCase().includes(needle))
                && (state.filter === "all" || type === state.filter)
                && (!allowed || allowed.includes(type));
            item.element.hidden = !visible;
            if (visible) shown += 1;
        }
        items.querySelector(".fbnodes-browser-empty")?.remove();
        if (!shown && !state.dirs.length) {
            items.append(el("div", "fbnodes-browser-empty", state.files.length ? "No files match the search or filter" : "This folder has no files to show"));
        }
        if (state.cursor?.element.hidden) setCursor(null);
    }

    function setView(mode) {
        state.view = mode;
        try { localStorage.setItem(VIEW_KEY, mode); } catch { /* not kept */ }
        for (const [key, choice] of Object.entries(viewButtons)) choice.classList.toggle("on", key === mode);
        render();
        revealCurrent();
    }

    // ---- loading a folder
    async function load() {
        const request = ++state.request;
        items.className = `fbnodes-browser-items ${state.view}`;
        items.replaceChildren(el("div", "fbnodes-browser-empty", "Loading…"));
        try {
            let data = await fetchListing(state.dir, state.kind);
            if (data.mode === "roots") {
                state.roots = rootPaths(data.roots);
                state.dir = (sourceFolder === "output" && state.roots.output) || state.roots.input || state.roots.output;
                if (!state.dir) throw new Error("No folders available");
                data = await fetchListing(state.dir, state.kind);
            }
            if (request !== state.request) return;
            state.dir = data.current_path || state.dir;
            state.parent = data.parent_path || null;
            if (Array.isArray(data.roots)) state.roots = rootPaths(data.roots);
            state.dirs = (data.dirs || []).slice().sort((a, b) => compareText(a?.name, b?.name));
            state.files = (data.files || []).map(normalizeFile);
            state.firstLoad = false;
        } catch (error) {
            if (request !== state.request) return;
            // A remembered folder that no longer exists opens the input folder instead.
            if (state.firstLoad) {
                state.firstLoad = false;
                state.dir = "";
                return load();
            }
            items.replaceChildren(el("div", "fbnodes-browser-empty error", error.message || "Could not open this folder"));
            return;
        }
        if (document.activeElement !== pathField) pathField.value = state.dir;
        render();
        revealCurrent();
    }

    // Every folder change goes through here. Going up remembers the folder
    // left, so the Forward side button can go back down into it; going
    // anywhere else forgets that trail.
    function navigate(path, { cameFrom = "", trail = false } = {}) {
        if (!path) return;
        if (!trail) state.down = [];
        state.cameFrom = cameFrom;
        state.dir = path;
        load();
    }

    function go(path) {
        navigate(path);
    }

    function goUp() {
        if (!state.parent) return;
        state.down.push(state.dir);
        navigate(state.parent, { cameFrom: state.dir, trail: true });
    }

    function goDown() {
        if (state.down.length) navigate(state.down.pop(), { trail: true });
    }

    async function goSibling(direction) {
        if (!state.parent) return;
        try {
            const listing = await fetchListing(state.parent, state.kind);
            const siblings = (listing.dirs || []).slice().sort((a, b) => compareText(a?.name, b?.name));
            const index = siblings.findIndex((dir) => samePath(dir.path, state.dir));
            if (index < 0 || siblings.length < 2) return;
            go(siblings[(index + direction + siblings.length) % siblings.length].path);
        } catch {
            // The parent cannot be read: stay here.
        }
    }

    // Scroll to the file the node holds, or to the folder we just came up from.
    function revealCurrent() {
        const target = state.items.find((item) => item.kind === "file" && samePath(item.entry.path, current))
            ?? state.items.find((item) => item.kind === "dir" && samePath(item.entry.path, state.cameFrom));
        if (!target) return;
        setCursor(target);
        requestAnimationFrame(() => target.element.scrollIntoView({ block: "center" }));
    }

    // ---- keyboard
    function setCursor(item) {
        state.cursor?.element.classList.remove("cursor");
        state.cursor = item;
        item?.element.classList.add("cursor");
    }

    // Arrows move one item (one row up / down in a grid), Page Up / Down move
    // a screenful of rows, Home / End jump to the first / last item.
    const NAV_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End"]);

    function moveCursor(key) {
        const visible = state.items.filter((item) => !item.element.hidden);
        if (!visible.length) return;
        const index = visible.indexOf(state.cursor);
        let next = null;
        if (key === "Home") next = visible[0];
        else if (key === "End") next = visible[visible.length - 1];
        else if (index < 0) next = visible[0];
        else if (key === "ArrowLeft" || key === "ArrowRight" || (state.view === "list" && key.startsWith("Arrow"))) {
            next = visible[index + (key === "ArrowLeft" || key === "ArrowUp" ? -1 : 1)];
        } else {
            // The card nearest the cursor's column, in the row one step (arrows)
            // or one screen (pages) above or below; a list is a grid of one column.
            const here = state.cursor.element;
            const centre = here.offsetLeft + here.offsetWidth / 2;
            const rows = [...new Set(visible.map((item) => item.element.offsetTop))].sort((a, b) => a - b);
            const at = rows.indexOf(here.offsetTop);
            const down = key === "ArrowDown" || key === "PageDown";
            let target;
            if (key.startsWith("Arrow")) {
                target = rows[at + (down ? 1 : -1)];
            } else {
                const page = Math.max(here.offsetHeight, items.clientHeight - here.offsetHeight);
                const wanted = here.offsetTop + (down ? page : -page);
                target = rows.reduce((best, row) => (Math.abs(row - wanted) < Math.abs(best - wanted) ? row : best), rows[0]);
                if (target === here.offsetTop) target = rows[at + (down ? 1 : -1)];
            }
            if (target === undefined) return;
            next = visible
                .filter((item) => item.element.offsetTop === target)
                .reduce((best, item) => {
                    const distance = Math.abs(item.element.offsetLeft + item.element.offsetWidth / 2 - centre);
                    return !best || distance < best.distance ? { item, distance } : best;
                }, null)?.item;
        }
        if (!next) return;
        setCursor(next);
        next.element.scrollIntoView({ block: "nearest" });
    }

    function onKey(event) {
        // A dialog opened over the browser (Delete) takes its own keys.
        if (!modal.isTop()) return;
        if (event.ctrlKey && !event.shiftKey && !event.altKey && !event.metaKey && (event.key === "ArrowUp" || event.key === "ArrowDown")) {
            event.preventDefault();
            event.stopPropagation();
            goSibling(event.key === "ArrowUp" ? -1 : 1);
            return;
        }
        const editing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target?.tagName ?? "");
        if (editing) return;
        if (NAV_KEYS.has(event.key)) {
            event.preventDefault();
            event.stopPropagation();
            moveCursor(event.key);
        } else if (event.key === "Enter") {
            event.preventDefault();
            event.stopPropagation();
            // With files picked, Enter adds them; otherwise it opens what the cursor is on.
            if (multi && state.picked.size) addPicked();
            else if (state.cursor) openItem(state.cursor);
        } else if (event.key === "Backspace") {
            event.preventDefault();
            goUp();
        }
    }
    document.addEventListener("keydown", onKey, true);

    for (const [key, choice] of Object.entries(viewButtons)) choice.classList.toggle("on", key === state.view);
    clearSearch.hidden = true;
    updateCount();
    load();
}
