/**
 * File Browser Modal for PromptExtractor
 * Shows thumbnails of files in input directory
 */
import { PM_UI_PALETTE as UI } from './ui_palette.js';

// Track current subfolder for navigation
let currentSubfolder = '';
// Track current source folder (input or output)
let currentSourceFolder = 'input';

export function createFileBrowserModal(currentFile, onFileSelect, sourceFolder) {
    // Store source folder
    currentSourceFolder = sourceFolder || 'input';
    // If a file is currently selected and lives in a subfolder, open in that folder
    if (currentFile && currentFile.includes('/')) {
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
        background: ${UI.overlay};
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    `;

    // Create modal container (match PromptManagerAdvanced styling)
    const modal = document.createElement('div');
    modal.className = 'prompt-extractor-browser-modal';
    modal.style.cssText = `
        background: ${UI.panel};
        border: 1px solid ${UI.panelBorder};
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
        border-bottom: 1px solid ${UI.sectionBorder};
        display: flex;
        flex-direction: column;
        gap: 8px;
    `;
    
    const topRow = document.createElement('div');
    topRow.style.cssText = 'display: flex; justify-content: space-between; align-items: center;';
    topRow.innerHTML = `
        <h3 style="margin: 0; color: #d3d7de;">Select File from ${currentSourceFolder === 'output' ? 'Output' : 'Input'} Folder</h3>
        <div style="display: flex; gap: 10px; align-items: center;">
            <button class="regenerate-cache-btn" style="
                background: ${UI.buttonBg};
                border: 1px solid ${UI.inputBorder};
                border-radius: 6px;
                color: #ccc;
                padding: 5px 12px;
                cursor: pointer;
                font-size: 12px;
            ">Regenerate Cache</button>
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
    breadcrumb.style.cssText = `font-size: 12px; color: ${UI.textHint}; cursor: pointer;`;
    breadcrumb.textContent = `${currentSourceFolder}/`;
    breadcrumb.onclick = () => {
        if (currentSubfolder) {
            currentSubfolder = '';
            loadFileThumbnails(gridContainer, currentFile, onFileSelect, overlay, breadcrumb);
        }
    };
    
    header.appendChild(topRow);
    header.appendChild(breadcrumb);

    // Create search/filter bar
    const filterBar = document.createElement('div');
    filterBar.style.cssText = `
        padding: 10px 20px;
        border-bottom: 1px solid ${UI.sectionBorder};
        display: flex;
        gap: 10px;
    `;
    filterBar.innerHTML = `
        <input type="text" placeholder="Search files..." class="search-input" style="
            flex: 1;
            padding: 8px 12px;
            background: ${UI.inputBg};
            border: 1px solid ${UI.inputBorder};
            border-radius: 6px;
            color: #ccc;
        ">
        <select class="filter-type" style="
            padding: 8px 12px;
            background: ${UI.inputBg};
            border: 1px solid ${UI.inputBorder};
            border-radius: 6px;
            color: #ccc;
        ">
            <option value="all">All Files</option>
            <option value="image">Images</option>
            <option value="video">Videos</option>
            <option value="json">JSON</option>
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
    closeBtn.onclick = () => document.body.removeChild(overlay);
    overlay.onclick = (e) => {
        if (e.target === overlay) document.body.removeChild(overlay);
    };

    // Regenerate cache button handler
    const regenerateCacheBtn = topRow.querySelector('.regenerate-cache-btn');
    regenerateCacheBtn.onclick = () => {
        clearThumbnailCache();
        // Reload thumbnails to regenerate them
        loadFileThumbnails(gridContainer, currentFile, onFileSelect, overlay, breadcrumb);
    };

    // Add to DOM
    document.body.appendChild(overlay);

    // Load files
    loadFileThumbnails(gridContainer, currentFile, onFileSelect, overlay, breadcrumb);

    // Setup search/filter
    const searchInput = filterBar.querySelector('.search-input');
    const filterType = filterBar.querySelector('.filter-type');
    
    searchInput.oninput = () => filterThumbnails(gridContainer, searchInput.value, filterType.value);
    filterType.onchange = () => filterThumbnails(gridContainer, searchInput.value, filterType.value);
}

async function loadFileThumbnails(container, currentFile, onFileSelect, overlay, breadcrumbElement) {
    try {
        // Update breadcrumb
        if (!currentSubfolder) {
            breadcrumbElement.textContent = `${currentSourceFolder}/`;
        } else {
            breadcrumbElement.textContent = `${currentSourceFolder}/${currentSubfolder}/`;
        }
        
        // Fetch file list from server
        const response = await fetch(`/prompt-extractor/list-files?source=${encodeURIComponent(currentSourceFolder)}`);
        const data = await response.json();
        
        container.innerHTML = '';
        
        if (!data.files || data.files.length === 0) {
            container.innerHTML = '<div style="text-align: center; padding: 40px; color: #aaa;">No files found in ' + currentSourceFolder + ' directory</div>';
            return;
        }

        // Get all files with their full paths
        const allFiles = data.files.filter(f => f !== '(none)');
        
        // Filter files/folders for current directory
        const currentFiles = [];
        const subfolders = new Set();
        
        const prefix = currentSubfolder ? currentSubfolder + '/' : '';
        
        allFiles.forEach(filepath => {
            if (filepath.startsWith(prefix)) {
                const remainder = filepath.substring(prefix.length);
                const slashIndex = remainder.indexOf('/');
                
                if (slashIndex === -1) {
                    // It's a file in current directory
                    currentFiles.push(filepath);
                } else {
                    // It's in a subdirectory - extract folder name
                    const folderName = remainder.substring(0, slashIndex);
                    subfolders.add(folderName);
                }
            }
        });
        
        // Add "back" button if in subfolder
        if (currentSubfolder) {
            const backItem = createBackItem(container, currentFile, onFileSelect, overlay, breadcrumbElement);
            container.appendChild(backItem);
        }
        
        // Add folder items
        Array.from(subfolders).sort().forEach(folderName => {
            const folderItem = createFolderItem(folderName, container, currentFile, onFileSelect, overlay, breadcrumbElement);
            container.appendChild(folderItem);
        });

        // Create thumbnail for each file
        currentFiles.forEach(filename => {
            const item = createThumbnailItem(filename, currentFile, onFileSelect, overlay);
            container.appendChild(item);
        });

        // Scroll to the currently selected file
        if (currentFile) {
            const selectedItem = container.querySelector(`.thumbnail-item[data-filename="${CSS.escape(currentFile)}"]`);
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

function createBackItem(container, currentFile, onFileSelect, overlay, breadcrumbElement) {
    const item = document.createElement('div');
    item.className = 'folder-item';
    item.style.cssText = `
        background: ${UI.cardBg};
        border: 1px solid ${UI.panelBorder};
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
        height: 150px;
        background: rgba(16, 18, 22, 0.6);
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
        item.style.borderColor = UI.accentBorder;
        item.style.transform = 'translateY(-2px)';
        item.style.background = UI.accentSoft;
    };
    item.onmouseleave = () => {
        item.style.borderColor = UI.panelBorder;
        item.style.transform = 'translateY(0)';
        item.style.background = UI.cardBg;
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
        height: 150px;
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

function createThumbnailItem(filename, currentFile, onFileSelect, overlay) {
    const item = document.createElement('div');
    item.className = 'thumbnail-item';
    item.dataset.filename = filename;
    item.dataset.type = getFileType(filename);
    
    const isSelected = filename === currentFile;
    
    item.style.cssText = `
        background: ${UI.cardBg};
        border: 1px solid ${isSelected ? UI.accent : UI.panelBorder};
        border-radius: 6px;
        padding: 8px;
        cursor: pointer;
        transition: all 0.15s ease;
        display: flex;
        flex-direction: column;
        gap: 8px;
    `;

    // Thumbnail preview
    const preview = document.createElement('div');
    preview.style.cssText = `
        width: 100%;
        height: 150px;
        background: rgba(16, 18, 22, 0.6);
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

    if (imageExts.includes(ext)) {
        const img = document.createElement('img');
        // Handle subfolder paths: split filename into folder and basename
        let subfolder = '';
        let basename = filename;
        if (filename.includes('/')) {
            const lastSlash = filename.lastIndexOf('/');
            subfolder = filename.substring(0, lastSlash);
            basename = filename.substring(lastSlash + 1);
        }
        img.src = `/view?filename=${encodeURIComponent(basename)}&type=${currentSourceFolder}&subfolder=${encodeURIComponent(subfolder)}`;
        img.style.cssText = 'max-width: 100%; max-height: 100%; object-fit: contain;';
        img.onerror = () => {
            // Use placeholder image
            img.src = new URL("./placeholder.png", import.meta.url).href;
        };
        preview.appendChild(img);
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

    // Filename label (show only basename, not full path)
    const label = document.createElement('div');
    const basename = filename.includes('/') ? filename.split('/').pop() : filename;
    label.textContent = basename;
    label.style.cssText = `
        font-size: 12px;
        color: ${isSelected ? '#fff' : '#ccc'};
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        text-align: center;
    `;
    label.title = filename; // Full path in tooltip

    item.appendChild(preview);
    item.appendChild(label);

    // Hover effect
    item.onmouseenter = () => {
        if (!isSelected) {
            item.style.borderColor = UI.accentBorder;
            item.style.transform = 'translateY(-2px)';
            item.style.background = UI.accentSoft;
        }
    };
    item.onmouseleave = () => {
        if (!isSelected) {
            item.style.borderColor = UI.panelBorder;
            item.style.transform = 'translateY(0)';
            item.style.background = UI.cardBg;
        }
    };

    // Click handler
    item.onclick = (e) => {
        // Don't select if clicking to close context menu
        if (document.querySelector('.thumbnail-context-menu')) {
            return;
        }
        onFileSelect(filename);
        document.body.removeChild(overlay);
    };

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
        background: ${UI.panel};
        border: 1px solid ${UI.panelBorder};
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
        refreshBtn.style.background = UI.accentSoft;
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
    const cacheKey = `video_thumb_pyav_${filename.replace(/[\/\\]/g, '_')}`;
    
    try {
        localStorage.removeItem(cacheKey);
        console.log(`[FileBrowser] Cleared cache for: ${filename}`);
    } catch (error) {
        console.error('[FileBrowser] Error clearing cache:', error);
    }
    
    // Re-extract thumbnail via server-side PyAV
    extractVideoThumbnailServer(filename, previewElement);
}

function filterThumbnails(container, searchText, fileType) {
    const items = container.querySelectorAll('.thumbnail-item');
    const search = searchText.toLowerCase();
    
    items.forEach(item => {
        const filename = item.dataset.filename.toLowerCase();
        const type = item.dataset.type;
        
        const matchesSearch = !searchText || filename.includes(search);
        const matchesType = fileType === 'all' || type === fileType;
        
        item.style.display = (matchesSearch && matchesType) ? 'flex' : 'none';
    });
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
            img.style.cssText = 'max-width: 100%; max-height: 100%; object-fit: contain;';
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
        result.style.cssText = 'max-width: 100%; max-height: 100%; object-fit: contain;';
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

