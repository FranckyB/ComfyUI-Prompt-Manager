/**
 * File Browser Modal for FBnodes LoadImagePlus
 * Shows thumbnails of files in input directory
 */

import { mediaFileUrl, getCustomPresetPath } from "./path_browser.js";

// Track current subfolder for navigation
let currentSubfolder = '';

// ---- Arbitrary-path ("browse anywhere") navigation state ----
// 'legacy' = input/output relative browsing; 'abs' = absolute-path browsing.
let currentNavMode = 'legacy';
let currentAbsDir = '';
let currentAbsParent = null;
let currentAbsRoots = { input: '', output: '' };
let currentNavKind = 'media';
let currentAbsFocusName = '';

function isAbsBrowserPath(p) {
    return /^([a-zA-Z]:[\\/]|\\\\|\/)/.test(p || '');
}

function browserBasename(p) {
    if (!p) return p;
    const norm = String(p).replace(/\\/g, '/');
    const idx = norm.lastIndexOf('/');
    return idx >= 0 ? norm.slice(idx + 1) : norm;
}

function normalizePathKey(p) {
    return String(p || "").replace(/\\/g, "/").replace(/\/+/g, "/").toLowerCase();
}

function pathsEqual(a, b) {
    return normalizePathKey(a) === normalizePathKey(b);
}

async function fetchAbsListing(path, kind) {
    const params = [];
    if (path) params.push(`path=${encodeURIComponent(path)}`);
    if (kind) params.push(`kind=${encodeURIComponent(kind)}`);
    const query = params.length ? `?${params.join('&')}` : '';
    const resp = await fetch(`/prompt-extractor/path-browser/list${query}`);
    if (!resp.ok) {
        let msg = `Request failed (${resp.status})`;
        try {
            const err = await resp.json();
            if (err?.error) msg = err.error;
        } catch { /* ignore */ }
        throw new Error(msg);
    }
    return await resp.json();
}

function extractRootPaths(roots) {
    const out = { input: '', output: '' };
    if (!Array.isArray(roots)) return out;

    // Supports both legacy string roots and object roots with ids.
    if (roots.length > 0 && typeof roots[0] === 'string') {
        out.input = roots[0] || '';
        out.output = roots[1] || '';
        return out;
    }

    const inputRoot = roots.find(r => r?.id === 'input');
    const outputRoot = roots.find(r => r?.id === 'output');
    out.input = inputRoot?.path || '';
    out.output = outputRoot?.path || '';
    if (!out.input && roots[0]?.path) out.input = roots[0].path;
    if (!out.output && roots[1]?.path) out.output = roots[1].path;
    return out;
}

// Track current source folder (input or output)
let currentSourceFolder = 'input';
// Track selected list mode and backend query kind.
let currentListMode = 'media';
let currentListKind = 'media';
let currentAllowedTypes = null;
let currentViewMode = 'medium';
let currentDetailSortKey = 'name';
let currentDetailSortDir = 'asc';
// Track active audio preview in modal
let activeAudioPreview = null;
let activeAudioPreviewFilename = null;
let activeAudioPreviewItem = null;

function normalizeListMode(mode) {
    if (mode === 'all' || mode === 'audio' || mode === 'video' || mode === 'media') {
        return mode;
    }
    return 'media';
}

function setListMode(mode) {
    currentListMode = normalizeListMode(mode);
    if (currentListMode === 'all') {
        currentListKind = 'all';
    } else if (currentListMode === 'audio') {
        currentListKind = 'audio';
    } else {
        // Backend has media/all/audio. Video mode uses media + client-side filter.
        currentListKind = 'media';
    }
}

function normalizeAllowedTypes(types) {
    if (!Array.isArray(types) || types.length === 0) return null;
    const allowed = ['image', 'video', 'audio', 'json', 'other'];
    const normalized = types
        .map(t => String(t || '').toLowerCase())
        .filter(t => allowed.includes(t));
    return normalized.length > 0 ? normalized : null;
}

function normalizeFilterTypeOptions(types, fallback) {
    const allowed = ['all', 'image', 'video', 'audio', 'json'];
    if (!Array.isArray(types) || types.length === 0) return fallback;

    const normalized = [];
    for (const t of types) {
        const value = String(t || '').toLowerCase();
        if (allowed.includes(value) && !normalized.includes(value)) {
            normalized.push(value);
        }
    }

    return normalized.length > 0 ? normalized : fallback;
}

function normalizeViewMode(mode) {
    const m = String(mode || '').toLowerCase();
    if (m === 'large' || m === 'detail' || m === 'medium') {
        return m;
    }
    return 'medium';
}

function normalizeFileEntry(entry) {
    if (typeof entry === 'string') {
        return { path: entry, name: browserBasename(entry), size: null, modified: null, duration: null, width: null, height: null };
    }
    if (entry && typeof entry === 'object') {
        const path = entry.path || entry.filename || entry.name || '';
        const size = entry.size != null ? entry.size : (entry.bytes != null ? entry.bytes : null);
        const modified = entry.modified != null
            ? entry.modified
            : (entry.mtime != null
                ? entry.mtime
                : (entry.date != null ? entry.date : (entry.timestamp != null ? entry.timestamp : null)));
        return {
            path,
            name: entry.name || browserBasename(path),
            size,
            modified,
            duration: entry.duration != null ? entry.duration : null,
            width: entry.width != null ? entry.width : null,
            height: entry.height != null ? entry.height : null,
        };
    }
    return { path: '', name: '', size: null, modified: null, duration: null, width: null, height: null };
}

function formatFileSize(size) {
    const n = Number(size);
    if (!Number.isFinite(n) || n < 0) return '-';
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
    return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatFileDate(raw) {
    if (raw == null || raw === '') return '-';
    let ts = raw;
    if (typeof ts === 'number' && ts > 0 && ts < 1e12) {
        ts *= 1000;
    }
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return '-';
    return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
}

function formatDuration(raw) {
    const n = Number(raw);
    if (!Number.isFinite(n) || n < 0) return '-';
    const total = Math.round(n);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h > 0) {
        return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function getCardPreviewHeight() {
    return currentViewMode === 'large' ? 250 : 150;
}

function detailSortIndicator(key) {
    if (currentDetailSortKey !== key) return '';
    return currentDetailSortDir === 'asc' ? ' ▲' : ' ▼';
}

function parseSortDate(raw) {
    if (raw == null || raw === '') return null;
    let ts = raw;
    if (typeof ts === 'number' && ts > 0 && ts < 1e12) ts *= 1000;
    const n = new Date(ts).getTime();
    return Number.isFinite(n) ? n : null;
}

function compareText(a, b) {
    return String(a || '').localeCompare(String(b || ''), undefined, { numeric: true, sensitivity: 'base' });
}

function compareMaybeNumber(a, b) {
    const an = Number(a);
    const bn = Number(b);
    const aOk = Number.isFinite(an);
    const bOk = Number.isFinite(bn);
    if (!aOk && !bOk) return 0;
    if (!aOk) return 1;
    if (!bOk) return -1;
    return an - bn;
}

function getTypeLabel(fileType) {
    if (fileType === 'image') return 'Image';
    if (fileType === 'video') return 'Video';
    if (fileType === 'audio') return 'Audio';
    if (fileType === 'json') return 'JSON';
    return 'File';
}

function compareFilesForDetailSort(a, b) {
    let result = 0;
    const aType = getFileType(a.path || '');
    const bType = getFileType(b.path || '');

    if (currentDetailSortKey === 'type') {
        result = compareText(getTypeLabel(aType), getTypeLabel(bType));
        if (result === 0) result = compareText(browserBasename(a.path), browserBasename(b.path));
    } else if (currentDetailSortKey === 'size') {
        result = compareMaybeNumber(a.size, b.size);
        if (result === 0) result = compareText(browserBasename(a.path), browserBasename(b.path));
    } else if (currentDetailSortKey === 'modified') {
        result = compareMaybeNumber(parseSortDate(a.modified), parseSortDate(b.modified));
        if (result === 0) result = compareText(browserBasename(a.path), browserBasename(b.path));
    } else if (currentDetailSortKey === 'details') {
        const aTypeSort = getFileType(a.path || '');
        const bTypeSort = getFileType(b.path || '');
        if (aTypeSort === 'video' && bTypeSort === 'video') {
            result = compareMaybeNumber(a.duration, b.duration);
        } else if (aTypeSort === 'audio' && bTypeSort === 'audio') {
            result = compareMaybeNumber(a.duration, b.duration);
        } else {
            result = compareText(browserBasename(a.path), browserBasename(b.path));
        }
    } else {
        result = compareText(browserBasename(a.path), browserBasename(b.path));
    }

    return currentDetailSortDir === 'asc' ? result : -result;
}

function toggleDetailSort(key) {
    if (currentDetailSortKey === key) {
        currentDetailSortDir = currentDetailSortDir === 'asc' ? 'desc' : 'asc';
    } else {
        currentDetailSortKey = key;
        currentDetailSortDir = 'asc';
    }
}

function createDetailHeader(container, currentFile, onFileSelect, overlay, breadcrumbElement) {
    if (currentViewMode !== 'detail') return;

    const header = document.createElement('div');
    header.className = 'detail-header-row';
    header.style.cssText = `
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 10px;
        border-bottom: 1px solid rgba(226, 232, 240, 0.18);
        color: #aeb8c8;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        position: sticky;
        top: 0;
        background: rgba(26, 31, 40, 0.96);
        z-index: 2;
    `;

    const makeHeaderCell = (label, width, key, alignRight = false) => {
        const cell = document.createElement('button');
        cell.type = 'button';
        cell.textContent = `${label}${detailSortIndicator(key)}`;
        cell.style.cssText = `
            background: transparent;
            border: 0;
            color: #aeb8c8;
            cursor: pointer;
            padding: 2px 0;
            font-size: 11px;
            text-align: ${alignRight ? 'right' : 'left'};
            ${width ? `width: ${width};` : 'flex: 1; min-width: 0;'}
        `;
        cell.onclick = () => {
            toggleDetailSort(key);
            loadFileThumbnails(container, currentFile, onFileSelect, overlay, breadcrumbElement);
        };
        return cell;
    };

    const iconSpacer = document.createElement('div');
    iconSpacer.style.cssText = 'width: 20px; min-width: 20px;';
    header.appendChild(iconSpacer);
    header.appendChild(makeHeaderCell('Name', '', 'name'));
    header.appendChild(makeHeaderCell('Type', '110px', 'type'));
    header.appendChild(makeHeaderCell('Details', '190px', 'details'));
    header.appendChild(makeHeaderCell('Size', '110px', 'size', true));
    header.appendChild(makeHeaderCell('Modified', '170px', 'modified'));
    container.appendChild(header);
}

function createDetailFolderRow(name, typeLabel, iconText) {
    const item = document.createElement('div');
    item.className = 'folder-item';
    item.style.cssText = `
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 10px;
        border-bottom: 1px solid rgba(226, 232, 240, 0.12);
        background: transparent;
        cursor: pointer;
        transition: background 0.15s ease;
    `;

    const icon = document.createElement('div');
    icon.style.cssText = `
        width: 20px;
        height: 20px;
        min-width: 20px;
        border-radius: 3px;
        background: rgba(0, 0, 0, 0.45);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
    `;
    icon.textContent = iconText;

    const nameCol = document.createElement('div');
    nameCol.style.cssText = 'flex: 1; min-width: 0; font-size: 12px; color: #d3dbe7; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;';
    nameCol.textContent = name;
    nameCol.title = name;

    const typeCol = document.createElement('div');
    typeCol.style.cssText = 'width: 110px; font-size: 11px; color: rgba(178, 191, 208, 0.92);';
    typeCol.textContent = typeLabel;

    const detailCol = document.createElement('div');
    detailCol.style.cssText = 'width: 190px; font-size: 11px; color: rgba(178, 191, 208, 0.92);';
    detailCol.textContent = '-';

    const sizeCol = document.createElement('div');
    sizeCol.style.cssText = 'width: 110px; font-size: 11px; color: rgba(178, 191, 208, 0.92); text-align: right;';
    sizeCol.textContent = '-';

    const dateCol = document.createElement('div');
    dateCol.style.cssText = 'width: 170px; font-size: 11px; color: rgba(178, 191, 208, 0.92); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;';
    dateCol.textContent = '-';

    item.appendChild(icon);
    item.appendChild(nameCol);
    item.appendChild(typeCol);
    item.appendChild(detailCol);
    item.appendChild(sizeCol);
    item.appendChild(dateCol);

    item.onmouseenter = () => {
        item.style.background = 'rgba(50, 112, 163, 0.28)';
    };
    item.onmouseleave = () => {
        item.style.background = 'transparent';
    };

    return item;
}

function applyViewLayout(container) {
    if (!container) return;
    if (currentViewMode === 'detail') {
        container.style.display = 'block';
        container.style.gridTemplateColumns = '';
        container.style.gap = '0';
        return;
    }

    const minW = currentViewMode === 'large' ? 300 : 180;
    container.style.display = 'grid';
    container.style.gridTemplateColumns = `repeat(auto-fill, minmax(${minW}px, 1fr))`;
    container.style.gap = '15px';
}

function buildViewUrl(filename) {
    if (isAbsBrowserPath(filename)) {
        return mediaFileUrl(filename);
    }
    let subfolder = '';
    let basename = filename;
    if (filename.includes('/')) {
        const lastSlash = filename.lastIndexOf('/');
        subfolder = filename.substring(0, lastSlash);
        basename = filename.substring(lastSlash + 1);
    }

    let url = `/view?filename=${encodeURIComponent(basename)}&type=${currentSourceFolder}`;
    if (subfolder) {
        url += `&subfolder=${encodeURIComponent(subfolder)}`;
    }
    return url;
}

function clearPreviewItemState() {
    if (!activeAudioPreviewItem) return;
    activeAudioPreviewItem.style.boxShadow = '';
    activeAudioPreviewItem = null;
}

function stopAudioPreview() {
    if (activeAudioPreview) {
        try {
            activeAudioPreview.pause();
            activeAudioPreview.currentTime = 0;
        } catch (error) {
            // Ignore preview stop errors.
        }
    }
    activeAudioPreviewFilename = null;
    clearPreviewItemState();
}

function closeBrowserModal(overlay) {
    stopAudioPreview();
    if (overlay && overlay._keydownHandler) {
        document.removeEventListener('keydown', overlay._keydownHandler, true);
        overlay._keydownHandler = null;
    }
    if (overlay && overlay.parentNode) {
        overlay.parentNode.removeChild(overlay);
    }
}

function setThumbnailSelected(item, selected) {
    if (!item) return;
    item.dataset.selected = selected ? '1' : '0';
    const isDetail = item.dataset.viewMode === 'detail';
    if (isDetail) {
        item.style.background = selected ? 'rgba(50, 112, 163, 0.28)' : 'transparent';
    } else {
        item.style.borderColor = selected ? 'rgba(66, 153, 225, 0.9)' : 'rgba(226, 232, 240, 0.2)';
        item.style.background = selected ? 'rgba(50, 112, 163, 0.28)' : 'rgba(45, 55, 72, 0.7)';
        item.style.transform = 'translateY(0)';
    }
}

function selectThumbnailItem(container, item, shouldScroll = false) {
    if (!container || !item) return;
    const current = container.querySelectorAll('.thumbnail-item[data-selected="1"]');
    current.forEach((it) => setThumbnailSelected(it, false));
    setThumbnailSelected(item, true);
    if (shouldScroll) {
        item.scrollIntoView({ block: 'nearest', behavior: 'instant' });
    }
}

function getVisibleThumbnailItems(container) {
    if (!container) return [];
    return Array.from(container.querySelectorAll('.thumbnail-item')).filter((item) => {
        return item.style.display !== 'none';
    });
}

function buildThumbnailRows(items) {
    const rows = [];
    const tolerance = 10;
    for (const item of items) {
        const top = item.offsetTop;
        let row = rows.find((r) => Math.abs(r.top - top) <= tolerance);
        if (!row) {
            row = { top, items: [] };
            rows.push(row);
        }
        row.items.push(item);
    }
    rows.sort((a, b) => a.top - b.top);
    return rows.map((r) => r.items);
}

function findNearestByX(items, targetX) {
    if (!items.length) return null;
    let best = items[0];
    let bestDist = Math.abs((best.offsetLeft + best.offsetWidth / 2) - targetX);
    for (let i = 1; i < items.length; i++) {
        const it = items[i];
        const cx = it.offsetLeft + it.offsetWidth / 2;
        const dist = Math.abs(cx - targetX);
        if (dist < bestDist) {
            best = it;
            bestDist = dist;
        }
    }
    return best;
}

function moveThumbnailSelection(container, key) {
    const items = getVisibleThumbnailItems(container);
    if (!items.length) return false;

    let selected = container.querySelector('.thumbnail-item[data-selected="1"]');
    if (!selected || !items.includes(selected)) {
        selectThumbnailItem(container, items[0], true);
        return true;
    }

    const currentView = selected.dataset.viewMode || currentViewMode;
    const index = items.indexOf(selected);

    if (currentView === 'detail') {
        if (key === 'ArrowUp' && index > 0) {
            selectThumbnailItem(container, items[index - 1], true);
            return true;
        }
        if (key === 'ArrowDown' && index < items.length - 1) {
            selectThumbnailItem(container, items[index + 1], true);
            return true;
        }
        return false;
    }

    const rows = buildThumbnailRows(items);
    let rowIndex = -1;
    let colIndex = -1;
    for (let r = 0; r < rows.length; r++) {
        const c = rows[r].indexOf(selected);
        if (c >= 0) {
            rowIndex = r;
            colIndex = c;
            break;
        }
    }
    if (rowIndex < 0 || colIndex < 0) return false;

    if (key === 'ArrowLeft') {
        if (index <= 0) return false;
        selectThumbnailItem(container, items[index - 1], true);
        return true;
    }

    if (key === 'ArrowRight') {
        if (index >= items.length - 1) return false;
        selectThumbnailItem(container, items[index + 1], true);
        return true;
    }

    const targetRowIndex = key === 'ArrowUp' ? rowIndex - 1 : (key === 'ArrowDown' ? rowIndex + 1 : rowIndex);
    if (targetRowIndex < 0 || targetRowIndex >= rows.length) return false;

    const targetX = selected.offsetLeft + selected.offsetWidth / 2;
    const targetItem = findNearestByX(rows[targetRowIndex], targetX);
    if (!targetItem) return false;
    selectThumbnailItem(container, targetItem, true);
    return true;
}

function openSelectedThumbnail(container) {
    const items = getVisibleThumbnailItems(container);
    if (!items.length) return false;

    let selected = container.querySelector('.thumbnail-item[data-selected="1"]');
    if (!selected || !items.includes(selected)) {
        selected = items[0];
        selectThumbnailItem(container, selected, true);
    }

    if (selected && typeof selected._openFile === 'function') {
        selected._openFile();
        return true;
    }
    return false;
}

function splitFolderPath(path) {
    const normalized = String(path || '').replace(/\\/g, '/').replace(/\/+$/, '');
    const idx = normalized.lastIndexOf('/');
    if (idx <= 0) {
        return { parent: '', name: normalized };
    }
    return { parent: normalized.slice(0, idx), name: normalized.slice(idx + 1) };
}

async function navigateSiblingFolder(direction, container, currentFile, onFileSelect, overlay, breadcrumbElement) {
    const dir = direction >= 0 ? 1 : -1;

    if (currentNavMode === 'abs') {
        const currentPath = String(currentAbsDir || '').replace(/\/+$/, '');
        if (!currentPath) return false;

        let parentPath = currentAbsParent;
        if (!parentPath) {
            try {
                const here = await fetchAbsListing(currentPath, currentNavKind);
                parentPath = here?.parent_path || null;
            } catch {
                return false;
            }
        }
        if (!parentPath) return false;

        let parentListing;
        try {
            parentListing = await fetchAbsListing(parentPath, currentNavKind);
        } catch {
            return false;
        }

        const siblings = (parentListing?.dirs || []).slice().sort((a, b) => compareText(a?.name, b?.name));
        if (!siblings.length) return false;

        let index = siblings.findIndex((d) => pathsEqual(d?.path, currentPath));
        if (index < 0) {
            const currentName = browserBasename(currentPath);
            index = siblings.findIndex((d) => d?.name === currentName);
        }
        if (index < 0) return false;

        const next = siblings[(index + dir + siblings.length) % siblings.length];
        if (!next?.path || pathsEqual(next.path, currentPath)) return false;

        currentAbsFocusName = '';
        currentAbsDir = next.path;
        await loadFileThumbnails(container, currentFile, onFileSelect, overlay, breadcrumbElement);
        return true;
    }

    if (!currentSubfolder) return false;

    try {
        const response = await fetch(`/prompt-extractor/list-files?source=${encodeURIComponent(currentSourceFolder)}&kind=${encodeURIComponent(currentListKind)}&include_meta=1`);
        const data = await response.json();
        const allFiles = (data.files || [])
            .filter(f => (typeof f === 'string' ? f !== '(none)' : (f?.path || f?.filename || f?.name) !== '(none)'))
            .map(normalizeFileEntry)
            .map((f) => f.path);

        const parts = splitFolderPath(currentSubfolder);
        const parentPrefix = parts.parent ? `${parts.parent}/` : '';
        const siblingSet = new Set();

        allFiles.forEach((filepath) => {
            if (!filepath.startsWith(parentPrefix)) return;
            const remainder = filepath.substring(parentPrefix.length);
            const slashIndex = remainder.indexOf('/');
            if (slashIndex > 0) {
                siblingSet.add(remainder.substring(0, slashIndex));
            }
        });

        const siblings = Array.from(siblingSet).sort(compareText);
        if (!siblings.length) return false;

        const idx = siblings.indexOf(parts.name);
        if (idx < 0) return false;

        const nextName = siblings[(idx + dir + siblings.length) % siblings.length];
        const nextPath = parts.parent ? `${parts.parent}/${nextName}` : nextName;
        if (nextPath === currentSubfolder) return false;

        currentSubfolder = nextPath;
        await loadFileThumbnails(container, currentFile, onFileSelect, overlay, breadcrumbElement);
        return true;
    } catch {
        return false;
    }
}

async function toggleAudioPreview(filename, item) {
    const sourceUrl = `${buildViewUrl(filename)}&${Date.now()}`;

    if (!activeAudioPreview) {
        activeAudioPreview = document.createElement('audio');
        activeAudioPreview.preload = 'metadata';
        activeAudioPreview.style.display = 'none';
        activeAudioPreview.onended = () => {
            activeAudioPreviewFilename = null;
            clearPreviewItemState();
        };
        document.body.appendChild(activeAudioPreview);
    }

    if (activeAudioPreviewFilename === filename && !activeAudioPreview.paused) {
        stopAudioPreview();
        return;
    }

    stopAudioPreview();
    activeAudioPreviewFilename = filename;
    activeAudioPreviewItem = item;
    activeAudioPreviewItem.style.boxShadow = 'inset 0 0 0 1px rgba(64, 192, 255, 0.85)';

    try {
        activeAudioPreview.src = sourceUrl;
        await activeAudioPreview.play();
    } catch (error) {
        console.warn('[FileBrowser] Failed to preview audio:', error);
        stopAudioPreview();
    }
}

export function createFileBrowserModal(currentFile, onFileSelect, sourceFolder, options) {
    const opts = options || {};
    const joinBrowserPath = (base, leaf) => {
        if (!base || !leaf) return leaf || base || "";
        const cleanBase = String(base).replace(/[\\/]+$/, "");
        const cleanLeaf = String(leaf).replace(/^[\\/]+/, "");
        return `${cleanBase}/${cleanLeaf}`;
    };
    const showListKindSelector = opts.showListKindSelector === true;
    const onViewModeChange = typeof opts.onViewModeChange === 'function' ? opts.onViewModeChange : null;
    // Store source folder
    currentSourceFolder = sourceFolder || 'input';
    setListMode(opts.listKind || 'media');
    currentAllowedTypes = normalizeAllowedTypes(opts.allowedTypes);
    currentViewMode = normalizeViewMode(opts.viewMode || 'medium');

    // Arbitrary-path navigation mode ("browse anywhere").
    currentNavMode = opts.enableNavigation ? 'abs' : 'legacy';
    currentNavKind = opts.navKind || 'media';
    if (currentNavMode === 'abs') {
        if (opts.selectedAbsPath && isAbsBrowserPath(opts.selectedAbsPath)) {
            currentFile = opts.selectedAbsPath;
        }
        currentAbsDir = opts.initialPath || (isAbsBrowserPath(currentFile) ? currentFile.replace(/[\\/][^\\/]*$/, '') : '');
        currentAbsParent = null;
        currentAbsRoots = { input: '', output: '' };
        currentAbsFocusName = '';

        // If loader passes a relative selected file, resolve it against initialPath so
        // absolute-mode listing can highlight/scroll to that file on open.
        if (currentFile && !isAbsBrowserPath(currentFile) && currentAbsDir) {
            currentFile = joinBrowserPath(currentAbsDir, currentFile);
        }
    }

    // If a file is currently selected and lives in a subfolder, open in that folder
    if (currentNavMode === 'legacy' && currentFile && currentFile.includes('/')) {
        currentSubfolder = currentFile.substring(0, currentFile.lastIndexOf('/'));
    } else {
        currentSubfolder = '';
    }

    // Create modal overlay
    const overlay = document.createElement('div');
    overlay.className = 'prompt-extractor-browser-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    `;
    overlay._initialAbsLoadPending = currentNavMode === 'abs';

    // Create modal container (match PromptManagerAdvanced styling)
    const modal = document.createElement('div');
    modal.className = 'prompt-extractor-browser-modal';
    modal.style.cssText = `
        background: rgba(40, 44, 52, 0.98);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 6px;
        width: 90%;
        max-width: 1200px;
        height: 80%;
        max-height: 800px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    `;

    // Create header
    const header = document.createElement('div');
    header.style.cssText = `
        padding: 15px 20px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        display: flex;
        flex-direction: column;
        gap: 8px;
    `;
    
    const topRow = document.createElement('div');
    topRow.style.cssText = 'display: flex; justify-content: space-between; align-items: center;';
    const headerTitle = currentNavMode === 'abs'
        ? 'Select File'
        : `Select File from ${currentSourceFolder === 'output' ? 'Output' : 'Input'} Folder`;
    topRow.innerHTML = `
        <h3 style="margin: 0; color: #aaa;">${headerTitle}</h3>
        <div style="display: flex; gap: 10px; align-items: center;">
            <button class="close-btn" style="
                background: none;
                border: none;
                color: #aaa;
                font-size: 24px;
                cursor: pointer;
                padding: 0;
                width: 30px;
                height: 30px;
            ">×</button>
        </div>
    `;
    
    const breadcrumb = document.createElement('div');
    breadcrumb.className = 'folder-breadcrumb';
    breadcrumb.style.cssText = 'font-size: 12px; color: #888; cursor: pointer;';
    breadcrumb.textContent = `${currentSourceFolder}/`;
    breadcrumb.onclick = () => {
        if (currentNavMode === 'legacy' && currentSubfolder) {
            currentSubfolder = '';
            loadFileThumbnails(gridContainer, currentFile, onFileSelect, overlay, breadcrumb);
        }
    };

    header.appendChild(topRow);

    // Navigation toolbar (browse-anywhere mode): Up / Refresh / path box + presets.
    if (currentNavMode === 'abs') {
        const navRow = document.createElement('div');
        navRow.style.cssText = 'display:flex; gap:8px; align-items:center; flex-wrap:wrap;';

        const mkBtn = (label) => {
            const b = document.createElement('button');
            b.textContent = label;
            b.style.cssText = 'background:#2e3b4a;border:1px solid rgba(255,255,255,0.2);border-radius:6px;color:#dce6f2;padding:5px 12px;cursor:pointer;font-size:12px;';
            b.onmouseenter = () => { b.style.background = '#3a4a5c'; };
            b.onmouseleave = () => { b.style.background = '#2e3b4a'; };
            return b;
        };

        const upBtn = mkBtn('Up');
        const refreshBtn = mkBtn('Refresh');
        const pathInput = document.createElement('input');
        pathInput.type = 'text';
        pathInput.placeholder = 'Paste folder path and press Enter';
        pathInput.style.cssText = 'flex:1;min-width:240px;font-size:12px;color:#dce6f2;background:#222a33;border:1px solid rgba(255,255,255,0.2);border-radius:6px;padding:6px 8px;';

        const regenerateCacheBtn = document.createElement('button');
        regenerateCacheBtn.className = 'regenerate-cache-btn';
        regenerateCacheBtn.textContent = '\u267B\uFE0F Regenerate Cache';
        regenerateCacheBtn.style.cssText = 'background:#333;border:1px solid #555;border-radius:6px;color:#ccc;padding:5px 12px;cursor:pointer;font-size:12px;';

        const inputPreset = mkBtn('Input');
        const outputPreset = mkBtn('Output');
        const customPreset = mkBtn('Custom');

        const goUp = () => {
            if (!currentAbsParent) return;
            currentAbsFocusName = browserBasename(currentAbsDir);
            currentAbsDir = currentAbsParent;
            loadFileThumbnails(gridContainer, currentFile, onFileSelect, overlay, breadcrumb);
        };

        upBtn.onclick = () => {
            goUp();
        };
        refreshBtn.onclick = () => loadFileThumbnails(gridContainer, currentFile, onFileSelect, overlay, breadcrumb);
        pathInput.addEventListener('keydown', (e) => {
            if (e.key !== 'Enter') return;
            const next = pathInput.value.trim();
            if (!next) return;
            currentAbsDir = next;
            loadFileThumbnails(gridContainer, currentFile, onFileSelect, overlay, breadcrumb);
        });
        inputPreset.onclick = () => {
            if (currentAbsRoots.input) { currentAbsDir = currentAbsRoots.input; loadFileThumbnails(gridContainer, currentFile, onFileSelect, overlay, breadcrumb); }
        };
        outputPreset.onclick = () => {
            if (currentAbsRoots.output) { currentAbsDir = currentAbsRoots.output; loadFileThumbnails(gridContainer, currentFile, onFileSelect, overlay, breadcrumb); }
        };
        customPreset.onclick = () => {
            const p = getCustomPresetPath();
            if (p) { currentAbsDir = p; loadFileThumbnails(gridContainer, currentFile, onFileSelect, overlay, breadcrumb); }
        };
        if (!getCustomPresetPath()) customPreset.style.display = 'none';

        navRow.appendChild(upBtn);
        navRow.appendChild(refreshBtn);
        navRow.appendChild(inputPreset);
        navRow.appendChild(outputPreset);
        navRow.appendChild(customPreset);
        navRow.appendChild(pathInput);
        navRow.appendChild(regenerateCacheBtn);
        header.appendChild(navRow);

        // Expose the path box so the loader can keep it in sync.
        overlay._navPathInput = pathInput;
        overlay._regenerateCacheBtn = regenerateCacheBtn;
        overlay._goUp = goUp;
    } else {
        // Keep the legacy placement when absolute navigation is disabled.
        const legacyRegenerateBtn = document.createElement('button');
        legacyRegenerateBtn.className = 'regenerate-cache-btn';
        legacyRegenerateBtn.textContent = '\u267B\uFE0F Regenerate Cache';
        legacyRegenerateBtn.style.cssText = 'background:#333;border:1px solid #555;border-radius:6px;color:#ccc;padding:5px 12px;cursor:pointer;font-size:12px;';
        topRow.querySelector('div').prepend(legacyRegenerateBtn);
    }

    // Breadcrumb row intentionally hidden: path input on the toolbar already shows current location.


    // Create search/filter bar
    const filterBar = document.createElement('div');
    filterBar.style.cssText = `
        padding: 10px 20px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        display: flex;
        gap: 10px;
    `;

    const defaultFilterValues = showListKindSelector
        ? ['all', 'audio', 'video']
        : ['all', 'image', 'video', 'audio', 'json'];
    const filterValues = normalizeFilterTypeOptions(opts.filterTypeOptions, defaultFilterValues);
    const filterLabels = {
        all: 'All Files',
        image: 'Images',
        video: 'Videos',
        audio: 'Audio',
        json: 'JSON',
    };
    const typeFilterOptions = filterValues
        .map(value => `<option value="${value}">${filterLabels[value] || value}</option>`)
        .join('');

    filterBar.innerHTML = `
        <button class="sibling-prev" title="Previous sibling folder (Ctrl+Up)" style="
            width: 34px;
            min-width: 34px;
            height: 34px;
            padding: 0;
            background: rgba(45, 55, 72, 0.7);
            border: 1px solid rgba(226, 232, 240, 0.2);
            border-radius: 6px;
            color: #ccc;
            cursor: pointer;
            font-size: 16px;
            line-height: 1;
        ">^</button>
        <button class="sibling-next" title="Next sibling folder (Ctrl+Down)" style="
            width: 34px;
            min-width: 34px;
            height: 34px;
            padding: 0;
            background: rgba(45, 55, 72, 0.7);
            border: 1px solid rgba(226, 232, 240, 0.2);
            border-radius: 6px;
            color: #ccc;
            cursor: pointer;
            font-size: 16px;
            line-height: 1;
        ">v</button>
        <input type="text" placeholder="Search files..." class="search-input" style="
            flex: 1;
            padding: 8px 12px;
            background: rgba(45, 55, 72, 0.7);
            border: 1px solid rgba(226, 232, 240, 0.2);
            border-radius: 6px;
            color: #ccc;
        ">
        <select class="filter-type" style="
            padding: 8px 12px;
            background: rgba(45, 55, 72, 0.7);
            border: 1px solid rgba(226, 232, 240, 0.2);
            border-radius: 6px;
            color: #ccc;
        ">
            ${typeFilterOptions}
        </select>
        <select class="view-mode" style="
            padding: 8px 12px;
            background: rgba(45, 55, 72, 0.7);
            border: 1px solid rgba(226, 232, 240, 0.2);
            border-radius: 6px;
            color: #ccc;
        ">
            <option value="large">Large</option>
            <option value="medium">Medium</option>
            <option value="detail">Detail</option>
        </select>
    `;

    // Create thumbnail grid container
    const gridContainer = document.createElement('div');
    gridContainer.className = 'thumbnail-grid';
    gridContainer.style.cssText = `
        flex: 1;
        overflow-y: auto;
        padding: 20px;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 15px;
        align-content: start;
    `;
    applyViewLayout(gridContainer);

    // Create loading indicator
    const loading = document.createElement('div');
    loading.style.cssText = `
        text-align: center;
        padding: 40px;
        color: #aaa;
    `;
    loading.textContent = 'Loading files...';
    gridContainer.appendChild(loading);

    // Assemble modal
    modal.appendChild(header);
    modal.appendChild(filterBar);
    modal.appendChild(gridContainer);
    overlay.appendChild(modal);

    // Prevent right-click context menu everywhere in the browser
    // (video thumbnails have their own handler that will show the refresh menu)
    overlay.oncontextmenu = (e) => {
        e.preventDefault();
        return false;
    };

    // Close button handler
    const closeBtn = topRow.querySelector('.close-btn');
    closeBtn.onclick = () => closeBrowserModal(overlay);
    overlay.onclick = (e) => {
        if (e.target === overlay) closeBrowserModal(overlay);
    };
    overlay._keydownHandler = (e) => {
        if (e.ctrlKey && !e.shiftKey && !e.altKey && !e.metaKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
            e.preventDefault();
            e.stopPropagation();
            if (typeof overlay._navigateSibling === 'function') {
                const dir = e.key === 'ArrowUp' ? -1 : 1;
                overlay._navigateSibling(dir);
            }
            return;
        }

        const target = e.target;
        const targetTag = target && target.tagName ? target.tagName.toUpperCase() : '';
        const isEditing = target && (targetTag === 'INPUT' || targetTag === 'TEXTAREA' || targetTag === 'SELECT' || target.isContentEditable);

        if (!isEditing && (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
            e.preventDefault();
            e.stopPropagation();
            moveThumbnailSelection(gridContainer, e.key);
            return;
        }

        if (!isEditing && e.key === 'Enter') {
            e.preventDefault();
            e.stopPropagation();
            openSelectedThumbnail(gridContainer);
            return;
        }

        if (e.key === 'Escape') {
            e.preventDefault();
            e.stopPropagation();
            closeBrowserModal(overlay);
        }
    };
    document.addEventListener('keydown', overlay._keydownHandler, true);
    // Mouse "Back" button should behave like Up while browsing.
    overlay.onmouseup = (e) => {
        if (e.button === 3 && typeof overlay._goUp === 'function') {
            e.preventDefault();
            e.stopPropagation();
            overlay._goUp();
        }
    };

    // Regenerate cache button handler
    const regenerateCacheBtn = overlay._regenerateCacheBtn || topRow.querySelector('.regenerate-cache-btn');
    regenerateCacheBtn.onclick = () => {
        clearThumbnailCache();
        // Reload thumbnails to regenerate them
        loadFileThumbnails(gridContainer, currentFile, onFileSelect, overlay, breadcrumb);
    };

    // Add to DOM
    document.body.appendChild(overlay);

    // Setup search/filter
    const searchInput = filterBar.querySelector('.search-input');
    const filterType = filterBar.querySelector('.filter-type');
    const viewMode = filterBar.querySelector('.view-mode');
    const siblingPrevBtn = filterBar.querySelector('.sibling-prev');
    const siblingNextBtn = filterBar.querySelector('.sibling-next');

    overlay._navigateSibling = async (dir) => {
        await navigateSiblingFolder(dir, gridContainer, currentFile, onFileSelect, overlay, breadcrumb);
        filterThumbnails(gridContainer, searchInput.value, filterType.value);
    };

    siblingPrevBtn.onclick = () => {
        overlay._navigateSibling(-1);
    };
    siblingNextBtn.onclick = () => {
        overlay._navigateSibling(1);
    };

    // Apply default filter if specified
    if (opts.defaultFilter && filterValues.includes(opts.defaultFilter)) {
        filterType.value = opts.defaultFilter;
    } else if (showListKindSelector) {
        filterType.value = currentListMode === 'audio' ? 'audio'
            : currentListMode === 'video' ? 'video'
            : 'all';
    } else if (filterValues.length > 0) {
        filterType.value = filterValues[0];
    }

    if (showListKindSelector) {
        setListMode(filterType.value || 'all');
    }
    viewMode.value = currentViewMode;
    
    searchInput.oninput = () => filterThumbnails(gridContainer, searchInput.value, filterType.value);
    filterType.onchange = () => {
        if (showListKindSelector) {
            setListMode(filterType.value || 'all');
            loadFileThumbnails(gridContainer, currentFile, onFileSelect, overlay, breadcrumb).then(() => {
                filterThumbnails(gridContainer, searchInput.value, filterType.value);
            });
            return;
        }
        filterThumbnails(gridContainer, searchInput.value, filterType.value);
    };

    viewMode.onchange = () => {
        currentViewMode = normalizeViewMode(viewMode.value);
        if (onViewModeChange) {
            onViewModeChange(currentViewMode);
        }
        applyViewLayout(gridContainer);
        loadFileThumbnails(gridContainer, currentFile, onFileSelect, overlay, breadcrumb).then(() => {
            filterThumbnails(gridContainer, searchInput.value, filterType.value);
        });
    };

    // Load files then apply initial filter
    loadFileThumbnails(gridContainer, currentFile, onFileSelect, overlay, breadcrumb).then(() => {
        filterThumbnails(gridContainer, searchInput.value, filterType.value);
    });
}

async function loadFileThumbnails(container, currentFile, onFileSelect, overlay, breadcrumbElement) {
    if (currentNavMode === 'abs') {
        return loadFileThumbnailsAbs(container, currentFile, onFileSelect, overlay, breadcrumbElement);
    }
    try {
        // Update breadcrumb
        if (!currentSubfolder) {
            breadcrumbElement.textContent = `${currentSourceFolder}/`;
        } else {
            breadcrumbElement.textContent = `${currentSourceFolder}/${currentSubfolder}/`;
        }
        
        // Fetch file list from server
        const response = await fetch(`/prompt-extractor/list-files?source=${encodeURIComponent(currentSourceFolder)}&kind=${encodeURIComponent(currentListKind)}&include_meta=1`);
        const data = await response.json();
        
        container.innerHTML = '';
        
        if (!data.files || data.files.length === 0) {
            container.innerHTML = '<div style="text-align: center; padding: 40px; color: #aaa;">No files found in ' + currentSourceFolder + ' directory</div>';
            return;
        }

        // Get all files with their full paths
        const allFiles = data.files
            .filter(f => (typeof f === 'string' ? f !== '(none)' : (f?.path || f?.filename || f?.name) !== '(none)'))
            .map(normalizeFileEntry);
        
        // Filter files/folders for current directory
        const currentFiles = [];
        const subfolders = new Set();
        
        const prefix = currentSubfolder ? currentSubfolder + '/' : '';
        
        allFiles.forEach(fileEntry => {
            const filepath = fileEntry.path;
            if (filepath.startsWith(prefix)) {
                const remainder = filepath.substring(prefix.length);
                const slashIndex = remainder.indexOf('/');
                
                if (slashIndex === -1) {
                    // It's a file in current directory
                    currentFiles.push(fileEntry);
                } else {
                    // It's in a subdirectory - extract folder name
                    const folderName = remainder.substring(0, slashIndex);
                    subfolders.add(folderName);
                }
            }
        });

        const sortedFolders = Array.from(subfolders).sort((a, b) => compareText(a, b));
        const sortedFiles = currentFiles.slice().sort(compareFilesForDetailSort);

        createDetailHeader(container, currentFile, onFileSelect, overlay, breadcrumbElement);
        
        // Add "back" button if in subfolder
        if (currentSubfolder) {
            const backItem = createBackItem(container, currentFile, onFileSelect, overlay, breadcrumbElement);
            container.appendChild(backItem);
        }
        
        // Add folder items
        sortedFolders.forEach(folderName => {
            const folderItem = createFolderItem(folderName, container, currentFile, onFileSelect, overlay, breadcrumbElement);
            container.appendChild(folderItem);
        });

        // Create thumbnail for each file
        sortedFiles.forEach(fileEntry => {
            const item = createThumbnailItem(fileEntry, currentFile, onFileSelect, overlay);
            container.appendChild(item);
        });

        // Scroll to the currently selected file
        if (currentFile) {
            const selectedItem = container.querySelector(`.thumbnail-item[data-filename="${CSS.escape(currentFile)}"]`) ||
                Array.from(container.querySelectorAll('.thumbnail-item')).find(it => pathsEqual(it.dataset.filename, currentFile));
            if (selectedItem) {
                // Use requestAnimationFrame to ensure layout is complete before scrolling
                requestAnimationFrame(() => {
                    selectedItem.scrollIntoView({ block: 'center', behavior: 'instant' });
                });
            }
        }

    } catch (error) {
        console.error('[FileBrowser] Error loading files:', error);
        container.innerHTML = '<div style="text-align: center; padding: 40px; color: rgba(220, 53, 69, 0.9);">Error loading files</div>';
    }
}

async function loadFileThumbnailsAbs(container, currentFile, onFileSelect, overlay, breadcrumbElement) {
    container.innerHTML = '<div style="text-align: center; padding: 40px; color: #888;">Loading...</div>';
    try {
        let data = await fetchAbsListing(currentAbsDir, currentNavKind);

        // No directory yet -> backend returns the available roots; jump into the first one.
        if (data.mode === 'roots') {
            const roots = data.roots || [];
            currentAbsRoots = extractRootPaths(roots);
            const firstPath = (currentSourceFolder === 'output' && currentAbsRoots.output)
                ? currentAbsRoots.output
                : (currentAbsRoots.input || currentAbsRoots.output);
            if (!firstPath) {
                container.innerHTML = '<div style="text-align: center; padding: 40px; color: #888;">No locations available</div>';
                return;
            }
            currentAbsDir = firstPath;
            data = await fetchAbsListing(currentAbsDir, currentNavKind);
        }

        currentAbsDir = data.current_path || currentAbsDir;
        currentAbsParent = data.parent_path || null;
        if (Array.isArray(data.roots)) {
            currentAbsRoots = extractRootPaths(data.roots);
        }

        if (breadcrumbElement) breadcrumbElement.textContent = currentAbsDir;
        if (overlay && overlay._navPathInput && document.activeElement !== overlay._navPathInput) {
            overlay._navPathInput.value = currentAbsDir;
        }

        container.innerHTML = '';

        createDetailHeader(container, currentFile, onFileSelect, overlay, breadcrumbElement);

        // Sub-directories
        const dirs = (data.dirs || []).slice().sort((a, b) => compareText(a?.name, b?.name));
        let focusItem = null;
        for (const dir of dirs) {
            const item = createAbsFolderItem(dir.name, dir.path, container, currentFile, onFileSelect, overlay, breadcrumbElement);
            if (currentAbsFocusName && dir.name === currentAbsFocusName) {
                focusItem = item;
                item.style.borderColor = 'rgba(66, 153, 225, 0.9)';
                item.style.boxShadow = '0 0 0 2px rgba(66, 153, 225, 0.35)';
            }
            container.appendChild(item);
        }
        if (focusItem) {
            requestAnimationFrame(() => {
                focusItem.scrollIntoView({ block: 'center', behavior: 'instant' });
            });
            currentAbsFocusName = '';
        }

        // Files (absolute paths; thumbnail/url helpers detect abs paths)
        const files = (data.files || []).map(normalizeFileEntry).sort(compareFilesForDetailSort);
        let selectedItem = null;
        for (const file of files) {
            const thumb = createThumbnailItem(file, currentFile, onFileSelect, overlay);
            if (thumb) {
                container.appendChild(thumb);
                if (currentFile && pathsEqual(file.path, currentFile)) selectedItem = thumb;
            }
        }

        if (!dirs.length && !files.length && !currentAbsParent) {
            const empty = document.createElement('div');
            empty.style.cssText = 'text-align: center; padding: 40px; color: #888; grid-column: 1 / -1;';
            empty.textContent = 'This folder is empty';
            container.appendChild(empty);
        }

        if (selectedItem) {
            requestAnimationFrame(() => {
                selectedItem.scrollIntoView({ block: 'center', behavior: 'instant' });
            });
        }

        if (overlay && overlay._initialAbsLoadPending) {
            overlay._initialAbsLoadPending = false;
        }
    } catch (error) {
        // On first open, invalid remembered paths should fall back to Comfy's input root.
        if (overlay && overlay._initialAbsLoadPending) {
            overlay._initialAbsLoadPending = false;
            try {
                const rootsData = await fetchAbsListing('', currentNavKind);
                const fallbackRoots = extractRootPaths(rootsData?.roots || []);
                const fallbackPath = fallbackRoots.input || fallbackRoots.output;
                if (fallbackPath && !pathsEqual(currentAbsDir, fallbackPath)) {
                    currentAbsRoots = fallbackRoots;
                    currentAbsDir = fallbackPath;
                    return loadFileThumbnailsAbs(container, currentFile, onFileSelect, overlay, breadcrumbElement);
                }
            } catch {
                // Ignore fallback probe failures and show original error below.
            }
        }

        console.error('[FileBrowser] Error loading absolute path:', error);
        container.innerHTML = `<div style="text-align: center; padding: 40px; color: rgba(220, 53, 69, 0.9);">${error.message || 'Error loading files'}</div>`;
    }
}

function createAbsFolderItem(name, absPath, container, currentFile, onFileSelect, overlay, breadcrumbElement) {
    if (currentViewMode === 'detail') {
        const row = createDetailFolderRow(name, 'Folder', '📁');
        row.onclick = () => {
            currentAbsFocusName = '';
            currentAbsDir = absPath;
            loadFileThumbnails(container, currentFile, onFileSelect, overlay, breadcrumbElement);
        };
        return row;
    }

    const item = document.createElement('div');
    item.className = 'folder-item';
    item.style.cssText = `
        background: rgba(45, 55, 72, 0.7);
        border: 1px solid rgba(226, 232, 240, 0.2);
        border-radius: 6px;
        padding: 8px;
        cursor: pointer;
        transition: all 0.15s ease;
        display: flex;
        flex-direction: column;
        gap: 8px;
    `;

    const preview = document.createElement('div');
    preview.style.cssText = `
        width: 100%;
        height: ${getCardPreviewHeight()}px;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 64px;
    `;
    preview.textContent = '\uD83D\uDCC1';

    const label = document.createElement('div');
    label.textContent = name;
    label.style.cssText = `
        font-size: 12px;
        color: #ccc;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        text-align: center;
    `;
    label.title = name;

    item.appendChild(preview);
    item.appendChild(label);

    item.onmouseenter = () => {
        item.style.borderColor = 'rgba(66, 153, 225, 0.9)';
        item.style.transform = 'translateY(-2px)';
        item.style.background = 'rgba(50, 112, 163, 0.5)';
    };
    item.onmouseleave = () => {
        item.style.borderColor = 'rgba(226, 232, 240, 0.2)';
        item.style.transform = 'translateY(0)';
        item.style.background = 'rgba(45, 55, 72, 0.7)';
    };

    item.onclick = () => {
        currentAbsFocusName = '';
        currentAbsDir = absPath;
        loadFileThumbnails(container, currentFile, onFileSelect, overlay, breadcrumbElement);
    };
    return item;
}

function createBackItem(container, currentFile, onFileSelect, overlay, breadcrumbElement) {
    if (currentViewMode === 'detail') {
        const row = createDetailFolderRow('..', 'Parent Folder', '←');
        row.onclick = () => {
            const parts = currentSubfolder.split('/');
            parts.pop();
            currentSubfolder = parts.join('/');
            loadFileThumbnails(container, currentFile, onFileSelect, overlay, breadcrumbElement);
        };
        return row;
    }

    const item = document.createElement('div');
    item.className = 'folder-item';
    item.style.cssText = `
        background: rgba(45, 55, 72, 0.7);
        border: 1px solid rgba(226, 232, 240, 0.2);
        border-radius: 6px;
        padding: 8px;
        cursor: pointer;
        transition: all 0.15s ease;
        display: flex;
        flex-direction: column;
        gap: 8px;
    `;
    
    const preview = document.createElement('div');
    preview.style.cssText = `
        width: 100%;
        height: ${getCardPreviewHeight()}px;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 64px;
    `;
    preview.textContent = '←';
    
    const label = document.createElement('div');
    label.textContent = 'Back';
    label.style.cssText = `
        font-size: 12px;
        color: #ccc;
        text-align: center;
    `;
    
    item.appendChild(preview);
    item.appendChild(label);
    
    item.onmouseenter = () => {
        item.style.borderColor = 'rgba(66, 153, 225, 0.9)';
        item.style.transform = 'translateY(-2px)';
        item.style.background = 'rgba(50, 112, 163, 0.5)';
    };
    item.onmouseleave = () => {
        item.style.borderColor = 'rgba(226, 232, 240, 0.2)';
        item.style.transform = 'translateY(0)';
        item.style.background = 'rgba(45, 55, 72, 0.7)';
    };
    
    item.onclick = () => {
        // Go up one level
        const parts = currentSubfolder.split('/');
        parts.pop();
        currentSubfolder = parts.join('/');
        loadFileThumbnails(container, currentFile, onFileSelect, overlay, breadcrumbElement);
    };
    
    return item;
}

function createFolderItem(folderName, container, currentFile, onFileSelect, overlay, breadcrumbElement) {
    if (currentViewMode === 'detail') {
        const row = createDetailFolderRow(folderName, 'Folder', '📁');
        row.onclick = () => {
            currentSubfolder = currentSubfolder ? `${currentSubfolder}/${folderName}` : folderName;
            loadFileThumbnails(container, currentFile, onFileSelect, overlay, breadcrumbElement);
        };
        return row;
    }

    const item = document.createElement('div');
    item.className = 'folder-item';
    item.style.cssText = `
        background: rgba(45, 55, 72, 0.7);
        border: 1px solid rgba(226, 232, 240, 0.2);
        border-radius: 6px;
        padding: 8px;
        cursor: pointer;
        transition: all 0.15s ease;
        display: flex;
        flex-direction: column;
        gap: 8px;
    `;
    
    const preview = document.createElement('div');
    preview.style.cssText = `
        width: 100%;
        height: ${getCardPreviewHeight()}px;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 64px;
    `;
    preview.textContent = '📁';
    
    const label = document.createElement('div');
    label.textContent = folderName;
    label.style.cssText = `
        font-size: 12px;
        color: #ccc;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        text-align: center;
    `;
    label.title = folderName;
    
    item.appendChild(preview);
    item.appendChild(label);
    
    item.onmouseenter = () => {
        item.style.borderColor = 'rgba(66, 153, 225, 0.9)';
        item.style.transform = 'translateY(-2px)';
        item.style.background = 'rgba(50, 112, 163, 0.5)';
    };
    item.onmouseleave = () => {
        item.style.borderColor = 'rgba(226, 232, 240, 0.2)';
        item.style.transform = 'translateY(0)';
        item.style.background = 'rgba(45, 55, 72, 0.7)';
    };
    
    item.onclick = () => {
        // Navigate into folder
        currentSubfolder = currentSubfolder ? `${currentSubfolder}/${folderName}` : folderName;
        loadFileThumbnails(container, currentFile, onFileSelect, overlay, breadcrumbElement);
    };
    
    return item;
}

function createThumbnailItem(fileEntryInput, currentFile, onFileSelect, overlay) {
    const fileEntry = normalizeFileEntry(fileEntryInput);
    const filename = fileEntry.path;
    const item = document.createElement('div');
    item.className = 'thumbnail-item';
    item.dataset.filename = filename;
    item.dataset.type = getFileType(filename);
    item.dataset.viewMode = currentViewMode;
    const fileType = item.dataset.type;
    
    const isSelected = pathsEqual(filename, currentFile);
    
    if (currentViewMode === 'detail') {
        item.style.cssText = `
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 10px;
            border-bottom: 1px solid rgba(226, 232, 240, 0.12);
            background: transparent;
            cursor: pointer;
            transition: background 0.15s ease;
        `;
    } else {
        item.style.cssText = `
        background: rgba(45, 55, 72, 0.7);
        border: 1px solid rgba(226, 232, 240, 0.2);
        border-radius: 6px;
        padding: 8px;
        cursor: pointer;
        transition: all 0.15s ease;
        display: flex;
        flex-direction: column;
        gap: 8px;
    `;
    }

    // Thumbnail preview
    const preview = document.createElement('div');
    preview.style.cssText = currentViewMode === 'detail'
        ? `
            width: 20px;
            height: 20px;
            min-width: 20px;
            min-height: 20px;
            background: rgba(0, 0, 0, 0.45);
            border-radius: 3px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        `
        : `
            width: 100%;
            height: ${getCardPreviewHeight()}px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        `;

    const ext = filename.split('.').pop().toLowerCase();
    const imageExts = ['png', 'jpg', 'jpeg', 'webp'];
    const videoExts = ['mp4', 'webm', 'mov', 'avi'];
    const audioExts = ['wav', 'flac', 'mp3', 'm4a'];

    if (imageExts.includes(ext)) {
        const img = document.createElement('img');
        if (isAbsBrowserPath(filename)) {
            img.src = mediaFileUrl(filename);
        } else {
            // Handle subfolder paths: split filename into folder and basename
            let subfolder = '';
            let basename = filename;
            if (filename.includes('/')) {
                const lastSlash = filename.lastIndexOf('/');
                subfolder = filename.substring(0, lastSlash);
                basename = filename.substring(lastSlash + 1);
            }
            img.src = `/view?filename=${encodeURIComponent(basename)}&type=${currentSourceFolder}&subfolder=${encodeURIComponent(subfolder)}`;
        }
        img.style.cssText = 'max-width: 100%; max-height: 100%; object-fit: contain;';
        img.onerror = () => {
            // Use placeholder image
            img.src = new URL("./placeholder.png", import.meta.url).href;
        };
        preview.appendChild(img);
    } else if (audioExts.includes(ext) && (currentListMode === 'audio' || !videoExts.includes(ext))) {
        const audioBadge = document.createElement('div');
        audioBadge.style.cssText = `
            color: #9ec5fe;
            font-size: 12px;
            letter-spacing: 0.08em;
            border: 1px solid rgba(158, 197, 254, 0.4);
            border-radius: 999px;
            padding: 4px 10px;
            font-family: sans-serif;
        `;
        audioBadge.textContent = 'AUDIO';
        preview.appendChild(audioBadge);
    } else if (videoExts.includes(ext)) {
        // Extract video thumbnail via server-side PyAV
        extractVideoThumbnailServer(filename, preview);
    } else if (ext === 'json') {
        // Use placeholder for JSON
        const img = document.createElement('img');
        img.src = new URL("./placeholder.png", import.meta.url).href;
        img.style.cssText = 'max-width: 100%; max-height: 100%; object-fit: contain;';
        preview.appendChild(img);
    } else {
        // Use placeholder for unknown types
        const img = document.createElement('img');
        img.src = new URL("./placeholder.png", import.meta.url).href;
        img.style.cssText = 'max-width: 100%; max-height: 100%; object-fit: contain;';
        preview.appendChild(img);
    }

    item.appendChild(preview);

    if (currentViewMode === 'detail') {
        const nameCol = document.createElement('div');
        nameCol.style.cssText = `
            flex: 1;
            min-width: 0;
            font-size: 12px;
            color: ${isSelected ? '#fff' : '#d3dbe7'};
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        `;
        nameCol.textContent = browserBasename(filename);
        nameCol.title = filename;

        const typeCol = document.createElement('div');
        typeCol.style.cssText = `
            width: 110px;
            font-size: 11px;
            color: rgba(178, 191, 208, 0.92);
        `;
        typeCol.textContent = getTypeLabel(fileType);

        const detailsCol = document.createElement('div');
        detailsCol.style.cssText = `
            width: 190px;
            font-size: 11px;
            color: rgba(178, 191, 208, 0.92);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        `;
        if (fileType === 'video') {
            const dims = (fileEntry.width && fileEntry.height) ? `${fileEntry.width}x${fileEntry.height}` : '-';
            detailsCol.textContent = `${formatDuration(fileEntry.duration)}  ${dims}`;
        } else if (fileType === 'audio') {
            detailsCol.textContent = `${formatDuration(fileEntry.duration)}`;
        } else {
            detailsCol.textContent = '-';
        }

        const sizeCol = document.createElement('div');
        sizeCol.style.cssText = `
            flex-shrink: 0;
            width: 110px;
            font-size: 11px;
            color: rgba(178, 191, 208, 0.92);
            text-align: right;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        `;
        sizeCol.textContent = formatFileSize(fileEntry.size);

        const modifiedCol = document.createElement('div');
        modifiedCol.style.cssText = `
            width: 170px;
            font-size: 11px;
            color: rgba(178, 191, 208, 0.92);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        `;
        modifiedCol.textContent = formatFileDate(fileEntry.modified);

        item.appendChild(nameCol);
        item.appendChild(typeCol);
        item.appendChild(detailsCol);
        item.appendChild(sizeCol);
        item.appendChild(modifiedCol);
    } else {
        // Filename label (show only basename, not full path)
        const label = document.createElement('div');
        const basename = browserBasename(filename);
        label.textContent = basename;
        label.style.cssText = `
            font-size: 12px;
            color: ${isSelected ? '#fff' : '#ccc'};
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            text-align: center;
        `;
        label.title = filename;
        item.appendChild(label);
    }

    // Hover effect
    item.onmouseenter = () => {
        if (!isSelected) {
            if (currentViewMode === 'detail') {
                item.style.background = 'rgba(50, 112, 163, 0.28)';
            } else {
                item.style.borderColor = 'rgba(66, 153, 225, 0.9)';
                item.style.transform = 'translateY(-2px)';
                item.style.background = 'rgba(50, 112, 163, 0.5)';
            }
        }
    };
    item.onmouseleave = () => {
        if (!isSelected) {
            if (currentViewMode === 'detail') {
                item.style.background = 'transparent';
            } else {
                item.style.borderColor = 'rgba(226, 232, 240, 0.2)';
                item.style.transform = 'translateY(0)';
                item.style.background = 'rgba(45, 55, 72, 0.7)';
            }
        }
    };

    const selectFile = () => {
        if (currentNavMode === 'abs') {
            onFileSelect(filename, { absPath: filename, dir: currentAbsDir, roots: { ...currentAbsRoots } });
        } else {
            onFileSelect(filename);
        }
    };

    item._openFile = () => {
        selectFile();
        closeBrowserModal(overlay);
    };

    setThumbnailSelected(item, isSelected);

    // Click handler
    if (item.dataset.type === 'audio') {
        item.onclick = async (e) => {
            // Don't preview if clicking to close context menu
            if (document.querySelector('.thumbnail-context-menu')) {
                return;
            }
            selectThumbnailItem(item.parentElement, item);
            await toggleAudioPreview(filename, item);
        };

        item.ondblclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            item._openFile();
        };
    } else {
        item.onclick = (e) => {
            // Don't select if clicking to close context menu
            if (document.querySelector('.thumbnail-context-menu')) {
                return;
            }
            item._openFile();
        };

        item.ondblclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            item._openFile();
        };
    }

    // Right-click context menu - only for video thumbnails
    if (videoExts.includes(ext)) {
        item.oncontextmenu = (e) => {
            e.preventDefault();
            e.stopPropagation();
            showThumbnailContextMenu(e, filename, preview);
            return false;
        };
    }

    return item;
}

function getFileType(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const imageExts = ['png', 'jpg', 'jpeg', 'webp'];
    const videoExts = ['mp4', 'webm', 'mov', 'avi'];
    const audioExts = ['wav', 'flac', 'mp3', 'm4a'];
    
    if (audioExts.includes(ext) && (currentListMode === 'audio' || !videoExts.includes(ext))) return 'audio';
    if (imageExts.includes(ext)) return 'image';
    if (videoExts.includes(ext)) return 'video';
    if (ext === 'json') return 'json';
    return 'other';
}

/**
 * Show context menu for thumbnail with refresh option
 */
function showThumbnailContextMenu(event, filename, previewElement) {
    // Remove any existing context menu
    const existingMenu = document.querySelector('.thumbnail-context-menu');
    if (existingMenu) {
        existingMenu.remove();
    }

    // Create context menu
    const menu = document.createElement('div');
    menu.className = 'thumbnail-context-menu';
    menu.style.cssText = `
        position: fixed;
        left: ${event.pageX}px;
        top: ${event.pageY}px;
        background: rgba(30, 30, 30, 0.98);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 6px;
        padding: 4px;
        z-index: 10001;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        min-width: 160px;
    `;

    // Refresh option
    const refreshBtn = document.createElement('div');
    refreshBtn.textContent = '🔄 Refresh Thumbnail';
    refreshBtn.style.cssText = `
        padding: 8px 12px;
        color: #ccc;
        cursor: pointer;
        border-radius: 4px;
        font-size: 13px;
        transition: background 0.15s ease;
    `;
    refreshBtn.onmouseenter = () => {
        refreshBtn.style.background = 'rgba(66, 153, 225, 0.3)';
    };
    refreshBtn.onmouseleave = () => {
        refreshBtn.style.background = 'transparent';
    };
    refreshBtn.onclick = () => {
        refreshIndividualThumbnail(filename, previewElement);
        menu.remove();
    };

    menu.appendChild(refreshBtn);
    document.body.appendChild(menu);

    // Close menu when clicking outside
    const closeMenu = (e) => {
        if (!menu.contains(e.target)) {
            e.stopPropagation(); // Prevent click from going through to elements below
            menu.remove();
            document.removeEventListener('click', closeMenu, true); // Remove from capture phase
        }
    };
    // Use capture phase to catch the click before it reaches other elements
    setTimeout(() => {
        document.addEventListener('click', closeMenu, true);
    }, 10);
}

/**
 * Refresh thumbnail for a single file
 */
function refreshIndividualThumbnail(filename, previewElement) {
    console.log(`[FileBrowser] Refreshing thumbnail for: ${filename}`);
    
    // Clear cache for this specific file
    const cacheKey = `video_thumb_${filename.replace(/[\/\\]/g, '_')}`;
    
    try {
        localStorage.removeItem(cacheKey);
        console.log(`[FileBrowser] Cleared cache for: ${filename}`);
    } catch (error) {
        console.error('[FileBrowser] Error clearing cache:', error);
    }
    
    // Re-extract thumbnail
    extractVideoThumbnail(filename, previewElement, cacheKey);
}

function filterThumbnails(container, searchText, fileType) {
    const items = container.querySelectorAll('.thumbnail-item');
    const search = searchText.toLowerCase();
    
    items.forEach(item => {
        const filename = item.dataset.filename.toLowerCase();
        const type = item.dataset.type;
        
        const matchesSearch = !searchText || filename.includes(search);
        const matchesType = fileType === 'all' || type === fileType;
        const matchesAllowedTypes = !currentAllowedTypes || currentAllowedTypes.includes(type);
        
        item.style.display = (matchesSearch && matchesType && matchesAllowedTypes) ? 'flex' : 'none';
    });
}

/**
 * Extract thumbnail from video file with caching
 * Caches thumbnails in browser's localStorage
 */
async function extractVideoThumbnailCached(filename, previewElement) {
    // Use full path as cache key to avoid conflicts between subfolders
    const cacheKey = `video_thumb_${filename.replace(/[\/\\]/g, '_')}`;
    
    try {
        // Check if we have a cached thumbnail
        const cachedThumb = localStorage.getItem(cacheKey);
        
        // If cache exists, use it (trust localStorage until manual regeneration)
        if (cachedThumb) {
            const img = document.createElement('img');
            img.style.cssText = 'max-width: 100%; max-height: 100%; object-fit: contain;';
            img.src = cachedThumb;
            previewElement.innerHTML = '';
            previewElement.appendChild(img);
            return;
        }
    } catch (error) {
        console.log('[FileBrowser] Cache check failed, generating new thumbnail:', error);
    }
    
    // No cache, extract new thumbnail
    extractVideoThumbnail(filename, previewElement, cacheKey);
}

/**
 * Extract thumbnail from video file
 * Fast, lightweight version for preview purposes only
 */
async function extractVideoThumbnail(filename, previewElement, cacheKey = null) {
    // Show placeholder while loading
    const placeholderImg = document.createElement('img');
    placeholderImg.src = new URL("./placeholder.png", import.meta.url).href;
    placeholderImg.style.cssText = 'max-width: 100%; max-height: 100%; object-fit: contain;';
    previewElement.innerHTML = '';
    previewElement.appendChild(placeholderImg);
    
    // Create video element - append to DOM for reliable codec detection
    const video = document.createElement('video');
    video.crossOrigin = 'anonymous';
    video.preload = 'auto';
    video.muted = true;
    video.playsInline = true;
    video.style.cssText = 'position:fixed;top:-9999px;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none;';

    const cleanupVideo = () => {
        video.onloadedmetadata = null;
        video.onseeked = null;
        video.onerror = null;
        try { video.src = ''; video.load(); } catch (e) { /* ignore */ }
        if (video.parentNode) video.parentNode.removeChild(video);
    };
    
    video.onloadedmetadata = () => {
        // Use absolute first frame (no seeking = fastest)
        video.currentTime = 0;
    };
    
    video.onseeked = async () => {
        try {
            // Create small canvas for fast rendering (thumbnail size)
            const maxWidth = 180;
            const maxHeight = 150;
            const aspectRatio = video.videoWidth / video.videoHeight;
            
            let width, height;
            if (aspectRatio > maxWidth / maxHeight) {
                width = maxWidth;
                height = maxWidth / aspectRatio;
            } else {
                height = maxHeight;
                width = maxHeight * aspectRatio;
            }
            
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d', { alpha: false });
            
            // Draw video frame to small canvas
            ctx.drawImage(video, 0, 0, width, height);
            
            // Create image from canvas with lower quality for speed
            const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
            const img = document.createElement('img');
            img.style.cssText = 'max-width: 100%; max-height: 100%; object-fit: contain;';
            img.src = dataUrl;
            
            // Replace placeholder with thumbnail (display first, cache later)
            previewElement.innerHTML = '';
            previewElement.appendChild(img);
            
            // Cache the thumbnail asynchronously after display (non-blocking)
            if (cacheKey) {
                setTimeout(() => {
                    try {
                        localStorage.setItem(cacheKey, dataUrl);
                        console.log(`[FileBrowser] Cached thumbnail for: ${filename}`);
                    } catch (cacheError) {
                        console.log('[FileBrowser] Failed to cache thumbnail:', cacheError);
                    }
                }, 0);
            }
            
            // Clean up
            cleanupVideo();
        } catch (error) {
            console.error('[FileBrowser] Error extracting video thumbnail:', error);
            // Keep placeholder on error
            cleanupVideo();
        }
    };
    
    video.onerror = () => {
        console.log('[FileBrowser] Browser cannot decode video, trying server-side extraction:', filename);
        cleanupVideo();
        // Fall back to server-side frame extraction (PyAV handles H265/yuv444)
        const frameUrl = `/prompt-extractor/video-frame?filename=${encodeURIComponent(filename)}&source=${currentSourceFolder}&position=0`;
        const img = document.createElement('img');
        img.style.cssText = 'max-width: 100%; max-height: 100%; object-fit: contain;';
        img.onload = () => {
            previewElement.innerHTML = '';
            previewElement.appendChild(img);
            // Cache the server-extracted thumbnail
            if (cacheKey) {
                try {
                    const canvas = document.createElement('canvas');
                    canvas.width = img.naturalWidth;
                    canvas.height = img.naturalHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
                    localStorage.setItem(cacheKey, dataUrl);
                } catch (e) {
                    console.log('[FileBrowser] Failed to cache server thumbnail:', e);
                }
            }
        };
        img.onerror = () => {
            // Show "can't display" message
            previewElement.innerHTML = `
                <div style="text-align: center; color: #888; font-size: 11px; padding: 10px;">
                    <div style="font-size: 24px; margin-bottom: 6px;">🎬</div>
                    <div>Preview unavailable</div>
                    <div style="font-size: 10px; opacity: 0.7; margin-top: 2px;">H265/yuv444 format</div>
                </div>`;
        };
        img.src = frameUrl;
    };
    
    // Load video from input directory
    // Handle subfolder paths: split filename into folder and basename
    let subfolder = '';
    let basename = filename;
    if (filename.includes('/')) {
        const lastSlash = filename.lastIndexOf('/');
        subfolder = filename.substring(0, lastSlash);
        basename = filename.substring(lastSlash + 1);
    }
    document.body.appendChild(video);
    video.src = `/view?filename=${encodeURIComponent(basename)}&type=${currentSourceFolder}&subfolder=${encodeURIComponent(subfolder)}&${Date.now()}`;
}

/**
 * Extract thumbnail using server-side PyAV for all videos.
 * Fetches a JPEG frame from /prompt-extractor/video-frame and caches to localStorage.
 */
function extractVideoThumbnailServer(filename, previewElement) {
    const cacheKey = `video_thumb_pyav_${filename.replace(/[\/\\]/g, '_')}`;
    try {
        const cached = localStorage.getItem(cacheKey);
        if (cached) {
            const img = document.createElement('img');
            // Stretch to preview box size so large mode shows larger cached thumbs.
            img.style.cssText = 'width: 100%; height: 100%; object-fit: contain;';
            img.src = cached;
            previewElement.innerHTML = '';
            previewElement.appendChild(img);
            return;
        }
    } catch (e) { /* ignore */ }

    // Show placeholder while loading
    const placeholderImg = document.createElement('img');
    placeholderImg.src = new URL("./placeholder.png", import.meta.url).href;
    placeholderImg.style.cssText = 'max-width: 100%; max-height: 100%; object-fit: contain;';
    previewElement.innerHTML = '';
    previewElement.appendChild(placeholderImg);

    const frameUrl = `/prompt-extractor/video-frame?filename=${encodeURIComponent(filename)}&source=${currentSourceFolder}&position=0`;
    const img = document.createElement('img');
    img.onload = () => {
        // Scale down to thumbnail size
        const maxW = 180, maxH = 150;
        const ar = img.naturalWidth / img.naturalHeight;
        let w, h;
        if (ar > maxW / maxH) { w = maxW; h = maxW / ar; }
        else { h = maxH; w = maxH * ar; }
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d', { alpha: false });
        ctx.drawImage(img, 0, 0, w, h);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
        const result = document.createElement('img');
        // Stretch to preview box size so medium/large use the same cached asset at different display sizes.
        result.style.cssText = 'width: 100%; height: 100%; object-fit: contain;';
        result.src = dataUrl;
        previewElement.innerHTML = '';
        previewElement.appendChild(result);
        try { localStorage.setItem(cacheKey, dataUrl); } catch (e) { /* ignore */ }
    };
    img.onerror = () => {
        previewElement.innerHTML = `
            <div style="text-align: center; color: #888; font-size: 11px; padding: 10px;">
                <div style="font-size: 24px; margin-bottom: 6px;">🎬</div>
                <div>Preview unavailable</div>
            </div>`;
    };
    img.src = frameUrl;
}

/**
 * Clear all video thumbnail cache
 */
function clearThumbnailCache() {
    try {
        const cachePrefix = 'video_thumb_';
        const keysToRemove = [];
        
        // Find all cache keys in localStorage
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && key.startsWith(cachePrefix)) {
                keysToRemove.push(key);
            }
        }
        
        // Remove all cache entries
        keysToRemove.forEach(key => {
            localStorage.removeItem(key);
        });
        
        console.log(`[FileBrowser] Cleared ${keysToRemove.length} cache entries`);
    } catch (error) {
        console.error('[FileBrowser] Error clearing cache:', error);
    }
}

/**
 * Clean up orphaned thumbnail cache entries
 * Removes cache for files that no longer exist in the input directory
 */
function cleanupOrphanedThumbnailCache(currentFiles) {
    try {
        const cachePrefix = 'video_thumb_';
        const timePrefix = 'video_thumb_time_';
        const keysToRemove = [];
        
        // Find all cache keys in localStorage
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            
            // Check if it's a video thumbnail cache key
            if (key && key.startsWith(cachePrefix)) {
                // Extract filename from cache key
                const filename = key.substring(cachePrefix.length);
                
                // If this filename is not in current files list, mark for removal
                if (!currentFiles.includes(filename)) {
                    keysToRemove.push(key);
                    keysToRemove.push(timePrefix + filename);
                }
            }
        }
        
        // Remove orphaned cache entries
        keysToRemove.forEach(key => {
            localStorage.removeItem(key);
        });
        
        if (keysToRemove.length > 0) {
            console.log(`[FileBrowser] Cleaned up ${keysToRemove.length / 2} orphaned thumbnail cache entries`);
        }
    } catch (error) {
        console.error('[FileBrowser] Error cleaning up cache:', error);
    }
}

