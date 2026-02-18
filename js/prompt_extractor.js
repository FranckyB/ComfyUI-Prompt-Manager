/**
 * PromptExtractor Extension for ComfyUI
 * Adds image preview functionality for the extractor node
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createFileBrowserModal } from "./file_browser.js";

// Placeholder image path - loaded from static PNG file
const PLACEHOLDER_IMAGE_PATH = new URL("./placeholder.png", import.meta.url).href;

/**
 * Extract metadata from PNG file
 * Reads tEXt/iTXt chunks for prompt and workflow (ComfyUI native approach)
 */
async function getPNGMetadata(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            const pngData = new Uint8Array(event.target.result);
            const dataView = new DataView(pngData.buffer);
            const decoder = new TextDecoder();

            // Verify PNG signature
            if (dataView.getUint32(0) !== 0x89504E47 || dataView.getUint32(4) !== 0x0D0A1A0A) {
                resolve(null);
                return;
            }

            let prompt = null;
            let workflow = null;
            let parameters = null; // A1111/Forge parameters that ComfyUI can convert to workflow
            let offset = 8; // Skip PNG signature

            // Parse PNG chunks
            while (offset < pngData.length - 12) {
                const chunkLength = dataView.getUint32(offset);
                const chunkType = String.fromCharCode(
                    pngData[offset + 4],
                    pngData[offset + 5],
                    pngData[offset + 6],
                    pngData[offset + 7]
                );

                // Check for tEXt or iTXt chunks
                if (chunkType === 'tEXt' || chunkType === 'iTXt') {
                    const chunkData = pngData.slice(offset + 8, offset + 8 + chunkLength);

                    // Find null terminator for keyword
                    let keywordEnd = 0;
                    while (keywordEnd < chunkData.length && chunkData[keywordEnd] !== 0) {
                        keywordEnd++;
                    }

                    const keyword = decoder.decode(chunkData.slice(0, keywordEnd));
                    let text = '';

                    if (chunkType === 'tEXt') {
                        text = decoder.decode(chunkData.slice(keywordEnd + 1));
                    } else if (chunkType === 'iTXt') {
                        // iTXt format: keyword\0compression\0language\0translated\0text
                        const compression = chunkData[keywordEnd + 1];
                        let textStart = keywordEnd + 2;
                        // Skip language and translated keyword (find two more nulls)
                        let nullCount = 0;
                        while (textStart < chunkData.length && nullCount < 2) {
                            if (chunkData[textStart] === 0) nullCount++;
                            textStart++;
                        }
                        text = decoder.decode(chunkData.slice(textStart));
                    }

                    // Check for ComfyUI metadata or A1111 parameters
                    if (keyword === 'prompt') {
                        try {
                            prompt = JSON.parse(text);
                        } catch (e) {
                            console.error('[PromptExtractor] Failed to parse prompt metadata:', e);
                        }
                    } else if (keyword === 'workflow') {
                        try {
                            workflow = JSON.parse(text);
                        } catch (e) {
                            console.error('[PromptExtractor] Failed to parse workflow metadata:', e);
                        }
                    } else if (keyword === 'parameters') {
                        // A1111/Forge generation parameters (ComfyUI can load workflow from this)
                        parameters = text;
                    }
                }

                // Move to next chunk (length + type + data + CRC)
                offset += 12 + chunkLength;

                // Stop if we found metadata or reached IEND
                if ((prompt && workflow) || parameters || chunkType === 'IEND') {
                    break;
                }
            }

            // Return metadata if found (including A1111 parameters)
            if (prompt || workflow || parameters) {
                const metadata = { prompt, workflow, parameters };
                
                // If we have A1111 parameters, parse them for easier access
                if (parameters && !workflow) {
                    metadata.parsed_parameters = parseA1111Parameters(parameters);
                }
                
                resolve(metadata);
            } else {
                resolve(null);
            }
        };
        reader.readAsArrayBuffer(file);
    });
}

/**
 * Parse A1111/Forge parameters format
 * Extracts prompt, negative, and LoRAs
 */
function parseA1111Parameters(parametersText) {
    if (!parametersText) return null;

    const result = {
        prompt: '',
        negative_prompt: '',
        loras: []
    };

    // Split by "Negative prompt:" to separate positive and negative
    const parts = parametersText.split(/Negative prompt:\s*/i);
    let positivePrompt = parts[0].trim();
    let remainder = parts[1] || '';

    // Extract LoRAs from positive prompt using pattern: <lora:name:strength> or <lora:name:model_strength:clip_strength>
    const loraRegex = /<lora:([^:>]+):([^:>]+)(?::([^:>]+))?>/gi;
    let loraMatch;
    const loras = [];
    
    while ((loraMatch = loraRegex.exec(positivePrompt)) !== null) {
        const loraName = loraMatch[1].trim();
        const strength1 = parseFloat(loraMatch[2]);
        const strength2 = loraMatch[3] ? parseFloat(loraMatch[3]) : strength1;
        
        loras.push({
            name: loraName,
            model_strength: strength1,
            clip_strength: strength2
        });
    }

    // Remove LoRA tags from prompt
    positivePrompt = positivePrompt.replace(loraRegex, '').trim();
    result.prompt = positivePrompt;
    result.loras = loras;

    // Extract negative prompt (before any "Steps:" line if present)
    const settingsMatch = remainder.match(/^(.*?)[\r\n]+Steps:/s);
    if (settingsMatch) {
        result.negative_prompt = settingsMatch[1].trim();
    } else {
        result.negative_prompt = remainder.trim();
    }

    return result;
}

/**
 * Extract metadata from JPEG/WebP file
 * Reads EXIF UserComment field (0x9286) for workflow metadata
 */
async function getJPEGMetadata(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            const imageData = new Uint8Array(event.target.result);
            const dataView = new DataView(imageData.buffer);
            const decoder = new TextDecoder();

            // Check for JPEG signature (0xFFD8)
            if (dataView.getUint16(0) !== 0xFFD8) {
                resolve(null);
                return;
            }

            // Search for APP1 marker (EXIF) - 0xFFE1
            let offset = 2;
            while (offset < imageData.length - 4) {
                const marker = dataView.getUint16(offset);
                const segmentLength = dataView.getUint16(offset + 2);

                if (marker === 0xFFE1) {
                    // Check for EXIF header
                    const exifHeader = String.fromCharCode(...imageData.slice(offset + 4, offset + 10));
                    if (exifHeader === 'Exif\x00\x00') {
                        // Parse TIFF header
                        const tiffOffset = offset + 10;
                        const byteOrder = dataView.getUint16(tiffOffset);
                        const littleEndian = byteOrder === 0x4949;

                        // Get IFD0 offset
                        const ifd0Offset = tiffOffset + dataView.getUint32(tiffOffset + 4, littleEndian);

                        // Search for UserComment tag (0x9286)
                        const metadata = parseIFD(imageData, ifd0Offset, tiffOffset, littleEndian, decoder);
                        if (metadata) {
                            resolve(metadata);
                            return;
                        }
                    }
                }

                // Move to next marker
                if (marker >= 0xFF00) {
                    offset += 2 + segmentLength;
                } else {
                    break;
                }
            }

            resolve(null);
        };
        reader.readAsArrayBuffer(file);
    });
}

/**
 * Parse EXIF IFD (Image File Directory) for UserComment
 */
function parseIFD(imageData, ifdOffset, tiffOffset, littleEndian, decoder) {
    const dataView = new DataView(imageData.buffer);
    const numEntries = dataView.getUint16(ifdOffset, littleEndian);

    for (let i = 0; i < numEntries; i++) {
        const entryOffset = ifdOffset + 2 + (i * 12);
        const tag = dataView.getUint16(entryOffset, littleEndian);

        // UserComment tag (0x9286)
        if (tag === 0x9286) {
            const format = dataView.getUint16(entryOffset + 2, littleEndian);
            const count = dataView.getUint32(entryOffset + 4, littleEndian);
            const valueOffset = dataView.getUint32(entryOffset + 8, littleEndian);

            // Get actual data offset
            const dataOffset = count > 4 ? tiffOffset + valueOffset : entryOffset + 8;

            // Read comment data
            const commentData = imageData.slice(dataOffset, dataOffset + count);
            let comment = decoder.decode(commentData);

            // Remove ASCII/UNICODE prefix and null bytes
            comment = comment.replace(/^(ASCII|UNICODE)\x00*/, '').replace(/\x00/g, '');

            // Try to parse as JSON
            try {
                const json = JSON.parse(comment);
                return json;
            } catch (e) {
                console.error('[PromptExtractor] Failed to parse JPEG EXIF metadata:', e);
            }
        }

        // Check for EXIF SubIFD (tag 0x8769)
        if (tag === 0x8769) {
            const subIfdOffset = tiffOffset + dataView.getUint32(entryOffset + 8, littleEndian);
            const metadata = parseIFD(imageData, subIfdOffset, tiffOffset, littleEndian, decoder);
            if (metadata) return metadata;
        }
    }

    return null;
}

/**
 * Extract metadata from JSON file
 */
async function getJSONMetadata(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const json = JSON.parse(event.target.result);
                resolve(json);
            } catch (e) {
                console.error('[PromptExtractor] Failed to parse JSON file:', e);
                resolve(null);
            }
        };
        reader.readAsText(file);
    });
}

/**
 * Extract metadata from video file (WebM/MKV/MP4)
 * Reads container metadata directly
 */
async function getVideoMetadata(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            const videoData = new Uint8Array(event.target.result);
            const dataView = new DataView(videoData.buffer);
            const decoder = new TextDecoder();

            // Check for WebM/MKV (EBML/Matroska format) - magic: 0x1A45DFA3
            if (dataView.getUint32(0) === 0x1A45DFA3) {
                // Look for COMMENT tag (0x4487) which contains workflow metadata
                let offset = 4 + 8;
                while (offset < videoData.length - 16) {
                    if (dataView.getUint16(offset) === 0x4487) {
                        const name = String.fromCharCode(...videoData.slice(offset - 7, offset));
                        if (name === "COMMENT") {
                            let vint = dataView.getUint32(offset + 2);
                            let n_octets = Math.clz32(vint) + 1;
                            if (n_octets < 4) {
                                let length = (vint >> (8 * (4 - n_octets))) & ~(1 << (7 * n_octets));
                                const content = decoder.decode(videoData.slice(offset + 2 + n_octets, offset + 2 + n_octets + length));
                                try {
                                    const json = JSON.parse(content);
                                    resolve(json);
                                    return;
                                } catch (e) {
                                    console.error("[PromptExtractor] Failed to parse WebM/MKV metadata:", e);
                                }
                            }
                        }
                    }
                    offset += 1;
                }
            }
            // Check for MP4 (ISO Media format) - ftyp: 0x66747970, isom: 0x69736F6D
            else if (dataView.getUint32(4) === 0x66747970 && dataView.getUint32(8) === 0x69736F6D) {
                // Look for 'cmt' (comment) data tag in MP4
                let offset = videoData.length - 4;
                while (offset > 16) {
                    if (dataView.getUint32(offset) === 0x64617461) { // 'data' tag
                        if (dataView.getUint32(offset - 8) === 0xa9636d74) { // 'cmt' tag
                            let size = dataView.getUint32(offset - 4) - 4 * 4;
                            const content = decoder.decode(videoData.slice(offset + 12, offset + 12 + size));
                            try {
                                const json = JSON.parse(content);
                                resolve(json);
                                return;
                            } catch (e) {
                                console.error("[PromptExtractor] Failed to parse MP4 metadata:", e);
                            }
                        }
                    }
                    offset -= 1;
                }
            }

            resolve(null);
        };
        reader.readAsArrayBuffer(file);
    });
}

/**
 * Request Python to extract video metadata using ffprobe (fallback)
 */
async function extractVideoMetadataWithPython(filename) {
    try {
        console.log(`[PromptExtractor] Requesting Python ffprobe extraction for: ${filename}`);
        const response = await api.fetchApi("/prompt-extractor/extract-video-metadata", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename })
        });

        if (response.ok) {
            const result = await response.json();
            if (result.success && result.metadata) {
                console.log(`[PromptExtractor] Python extracted video metadata successfully for: ${filename}`);
                console.log(`[PromptExtractor] Metadata:`, result.metadata);
                return result.metadata;
            } else if (result.warning && result.error === 'ffprobe_not_found') {
                console.warn(`[PromptExtractor] WARNING: Video metadata fallback unavailable - ffprobe not found.`);
                console.warn(`[PromptExtractor] Install FFmpeg to enable metadata extraction for all video formats.`);
            } else {
                console.log(`[PromptExtractor] Python extraction returned no metadata for: ${filename}`);
            }
        } else {
            console.error(`[PromptExtractor] Python extraction failed with status: ${response.status}`);
        }
        return null;
    } catch (error) {
        console.error("[PromptExtractor] Error requesting Python video extraction:", error);
        return null;
    }
}

/**
 * Send file metadata to Python backend for caching
 */
async function cacheFileMetadata(filename, metadata) {
    try {
        const response = await api.fetchApi("/prompt-extractor/cache-file-metadata", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename, metadata })
        });

        if (response.ok) {
            console.log(`[PromptExtractor] Cached metadata for: ${filename}`);
        } else {
            console.error("[PromptExtractor] Failed to cache metadata:", response.status);
        }
    } catch (error) {
        console.error("[PromptExtractor] Error caching metadata:", error);
    }
}

/**
 * Send video frame to Python backend for caching
 */
async function cacheVideoFrame(filename, frameData, framePosition) {
    try {
        const response = await api.fetchApi("/prompt-extractor/cache-video-frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename, frame: frameData, frame_position: framePosition })
        });

        if (response.ok) {
            console.log(`[PromptExtractor] Cached video frame at position ${framePosition.toFixed(2)} for: ${filename}`);
        } else {
            console.error("[PromptExtractor] Failed to cache video frame:", response.status);
        }
    } catch (error) {
        console.error("[PromptExtractor] Error caching video frame:", error);
    }
}

/**
 * Create and show image preview modal
 */
function showImagePreviewModal(filename) {
    // Build image URL
    let actualFilename = filename;
    let subfolder = "";
    
    if (filename.includes('/')) {
        const lastSlash = filename.lastIndexOf('/');
        subfolder = filename.substring(0, lastSlash);
        actualFilename = filename.substring(lastSlash + 1);
    }
    
    let imageUrl = `/view?filename=${encodeURIComponent(actualFilename)}&type=input`;
    if (subfolder) {
        imageUrl += `&subfolder=${encodeURIComponent(subfolder)}`;
    }

    // Create modal overlay
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.85);
        z-index: 10000;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    `;

    // Create header with filename and close button
    const header = document.createElement('div');
    header.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        padding: 15px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(0, 0, 0, 0.5);
    `;

    const title = document.createElement('span');
    title.textContent = filename;
    title.style.cssText = `
        color: #fff;
        font-size: 14px;
        font-family: sans-serif;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: calc(100% - 50px);
    `;

    const closeBtn = document.createElement('button');
    closeBtn.textContent = '✕';
    closeBtn.style.cssText = `
        background: rgba(255, 255, 255, 0.1);
        border: none;
        color: #fff;
        font-size: 20px;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.2s;
    `;
    closeBtn.onmouseover = () => closeBtn.style.background = 'rgba(255, 255, 255, 0.2)';
    closeBtn.onmouseout = () => closeBtn.style.background = 'rgba(255, 255, 255, 0.1)';
    closeBtn.onclick = () => overlay.remove();

    header.appendChild(title);
    header.appendChild(closeBtn);

    // Create image container
    const imageContainer = document.createElement('div');
    imageContainer.style.cssText = `
        max-width: 90%;
        max-height: 80%;
        display: flex;
        align-items: center;
        justify-content: center;
    `;

    // Create image element
    const img = document.createElement('img');
    img.src = imageUrl;
    img.style.cssText = `
        max-width: 100%;
        max-height: 80vh;
        border-radius: 8px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    `;

    // Error handling
    img.onerror = () => {
        imageContainer.innerHTML = `
            <div style="color: #ff6666; font-family: sans-serif; text-align: center;">
                <div style="font-size: 48px; margin-bottom: 10px;">⚠️</div>
                <div>Failed to load image</div>
                <div style="font-size: 12px; margin-top: 5px; opacity: 0.7;">${filename}</div>
            </div>
        `;
    };

    imageContainer.appendChild(img);

    // Create keyboard hint
    const hint = document.createElement('div');
    hint.textContent = 'Press ESC or click outside to close';
    hint.style.cssText = `
        position: absolute;
        bottom: 20px;
        color: rgba(255, 255, 255, 0.5);
        font-size: 12px;
        font-family: sans-serif;
    `;

    overlay.appendChild(header);
    overlay.appendChild(imageContainer);
    overlay.appendChild(hint);

    // Close on overlay click (but not image click)
    overlay.onclick = (e) => {
        if (e.target === overlay) {
            overlay.remove();
        }
    };

    // Close on ESC key
    const handleKeydown = (e) => {
        if (e.key === 'Escape') {
            overlay.remove();
            document.removeEventListener('keydown', handleKeydown);
        }
    };
    document.addEventListener('keydown', handleKeydown);

    // Add to document
    document.body.appendChild(overlay);
}

/**
 * Create and show video preview modal
 */
function showVideoPreviewModal(filename) {
    // Build video URL
    let actualFilename = filename;
    let subfolder = "";
    
    if (filename.includes('/')) {
        const lastSlash = filename.lastIndexOf('/');
        subfolder = filename.substring(0, lastSlash);
        actualFilename = filename.substring(lastSlash + 1);
    }
    
    let videoUrl = `/view?filename=${encodeURIComponent(actualFilename)}&type=input`;
    if (subfolder) {
        videoUrl += `&subfolder=${encodeURIComponent(subfolder)}`;
    }

    // Create modal overlay
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.85);
        z-index: 10000;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    `;

    // Create header with filename and close button
    const header = document.createElement('div');
    header.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        padding: 15px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(0, 0, 0, 0.5);
    `;

    const title = document.createElement('span');
    title.textContent = filename;
    title.style.cssText = `
        color: #fff;
        font-size: 14px;
        font-family: sans-serif;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: calc(100% - 50px);
    `;

    const closeBtn = document.createElement('button');
    closeBtn.textContent = '✕';
    closeBtn.style.cssText = `
        background: rgba(255, 255, 255, 0.1);
        border: none;
        color: #fff;
        font-size: 20px;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.2s;
    `;
    closeBtn.onmouseover = () => closeBtn.style.background = 'rgba(255, 255, 255, 0.2)';
    closeBtn.onmouseout = () => closeBtn.style.background = 'rgba(255, 255, 255, 0.1)';
    closeBtn.onclick = () => overlay.remove();

    header.appendChild(title);
    header.appendChild(closeBtn);

    // Create video container
    const videoContainer = document.createElement('div');
    videoContainer.style.cssText = `
        max-width: 90%;
        max-height: 80%;
        display: flex;
        align-items: center;
        justify-content: center;
    `;

    // Create video element
    const video = document.createElement('video');
    video.src = videoUrl;
    video.controls = true;
    video.autoplay = true;
    video.loop = true;
    video.style.cssText = `
        max-width: 100%;
        max-height: 80vh;
        border-radius: 8px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    `;

    // Error handling
    video.onerror = () => {
        videoContainer.innerHTML = `
            <div style="color: #ff6666; font-family: sans-serif; text-align: center;">
                <div style="font-size: 48px; margin-bottom: 10px;">⚠️</div>
                <div>Failed to load video</div>
                <div style="font-size: 12px; margin-top: 5px; opacity: 0.7;">${filename}</div>
            </div>
        `;
    };

    videoContainer.appendChild(video);

    // Create keyboard hint
    const hint = document.createElement('div');
    hint.textContent = 'Press ESC or click outside to close';
    hint.style.cssText = `
        position: absolute;
        bottom: 20px;
        color: rgba(255, 255, 255, 0.5);
        font-size: 12px;
        font-family: sans-serif;
    `;

    overlay.appendChild(header);
    overlay.appendChild(videoContainer);
    overlay.appendChild(hint);

    // Close on overlay click (but not video click)
    overlay.onclick = (e) => {
        if (e.target === overlay) {
            overlay.remove();
        }
    };

    // Close on ESC key
    const handleKeydown = (e) => {
        if (e.key === 'Escape') {
            overlay.remove();
            document.removeEventListener('keydown', handleKeydown);
        }
    };
    document.addEventListener('keydown', handleKeydown);

    // Add to document
    document.body.appendChild(overlay);

    // Focus video for keyboard controls
    video.focus();
}

/**
 * Check if filename is a video file
 */
function isVideoFile(filename) {
    if (!filename) return false;
    const ext = filename.split('.').pop().toLowerCase();
    return ['mp4', 'webm', 'mov', 'avi', 'mkv', 'm4v', 'wmv'].includes(ext);
}

/**
 * Check if filename is a previewable file (image or video)
 */
function isPreviewableFile(filename) {
    if (!filename || filename === '(none)') return false;
    const ext = filename.split('.').pop().toLowerCase();
    const imageExtensions = ['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'];
    const videoExtensions = ['mp4', 'webm', 'mov', 'avi', 'mkv', 'm4v', 'wmv'];
    return imageExtensions.includes(ext) || videoExtensions.includes(ext);
}

app.registerExtension({
    name: "PromptExtractor",

    async setup() {
        // Listen for frame extraction requests from Python backend
        api.addEventListener("prompt-extractor-extract-frame", async (event) => {
            const { filename, frame_position } = event.detail;
            console.log(`[PromptExtractor] Received extraction request for ${filename} at position ${frame_position}`);
            
            // Find the node with this filename
            if (app.graph && app.graph._nodes) {
                for (const node of app.graph._nodes) {
                    if (node.type === "PromptExtractor") {
                        const imageWidget = node.widgets?.find(w => w.name === "image");
                        const frameWidget = node.widgets?.find(w => w.name === "frame_position");
                        
                        if (imageWidget && imageWidget.value === filename) {
                            // Update frame position if provided
                            if (frameWidget && frame_position !== undefined) {
                                frameWidget.value = frame_position;
                            }
                            
                            // Trigger extraction
                            await loadAndDisplayImage(node, filename);
                            console.log(`[PromptExtractor] Extracted and cached frame for ${filename}`);
                            break;
                        }
                    }
                }
            }
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PromptExtractor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                const node = this;

                // Track workflow metadata status for indicator
                node.hasWorkflow = false;
                // Track the currently loaded image filename and frame position to prevent unnecessary reloads
                node._loadedImageFilename = null;
                node._loadedFramePosition = null;

                // Find the frame_position widget (slider) early so we can reference it
                const framePositionWidget = this.widgets?.find(w => w.name === "frame_position");

                // Find the image widget (combo dropdown)
                const imageWidget = this.widgets?.find(w => w.name === "image");
                if (imageWidget) {
                    // Track the last known value to detect changes
                    let lastImageValue = imageWidget.value;
                    let extractionInProgress = false;
                    
                    // Store original callback
                    const originalCallback = imageWidget.callback;

                    // Override callback to load and display image
                    imageWidget.callback = function(value) {
                        if (originalCallback) {
                            originalCallback.apply(this, arguments);
                        }

                        // Load and display the image/video (handles all display logic)
                        loadAndDisplayImage(node, value);
                    };
                    
                    // Watch for value changes to update metadata indicator
                    const originalOnDrawForeground = node.onDrawForeground;
                    node.onDrawForeground = function(ctx) {
                        if (originalOnDrawForeground) {
                            originalOnDrawForeground.call(this, ctx);
                        }
                        
                        // Check if image value changed and not already extracting
                        if (imageWidget.value !== lastImageValue && !extractionInProgress) {
                            extractionInProgress = true;
                            lastImageValue = imageWidget.value;
                            extractAndUpdateMetadata(node, imageWidget.value).finally(() => {
                                extractionInProgress = false;
                            });
                        }
                    };
                    
                    // Add custom "Browse Files" button AFTER the image widget
                    const imageWidgetIndex = this.widgets.indexOf(imageWidget);
                    const browseButton = {
                        type: "button",
                        name: "📁 Browse Files",
                        value: null,
                        callback: () => {
                            const currentFile = imageWidget.value === "(none)" ? null : imageWidget.value;
                            
                            // Open file browser modal
                            createFileBrowserModal(currentFile, (selectedFile) => {
                                // Update the dropdown value
                                imageWidget.value = selectedFile;
                                
                                // Trigger the callback to load/display the file
                                if (imageWidget.callback) {
                                    imageWidget.callback(selectedFile);
                                }
                                
                                // Mark node as needing update
                                node.setDirtyCanvas(true);
                            });
                        },
                        serialize: false // Don't save button state
                    };
                    
                    // Insert button right after the image widget
                    this.widgets.splice(imageWidgetIndex + 1, 0, browseButton);
                    Object.defineProperty(browseButton, "node", { value: node });

                    // Track if current file is a video for UI behavior
                    node._isVideoFile = false;

                    // Function to update video-specific UI visibility
                    const updateVideoUIVisibility = () => {
                        const isVideo = isVideoFile(imageWidget.value);
                        node._isVideoFile = isVideo;
                        
                        // Show/hide frame_position widget based on file type
                        if (framePositionWidget) {
                            framePositionWidget.hidden = !isVideo;
                        }
                        
                        node.setDirtyCanvas(true);
                    };

                    // Wrap the image widget callback to also update visibility
                    const wrappedCallback = imageWidget.callback;
                    imageWidget.callback = function(value) {
                        if (wrappedCallback) {
                            wrappedCallback.apply(this, arguments);
                        }
                        updateVideoUIVisibility();
                    };

                    // Initial visibility check
                    setTimeout(updateVideoUIVisibility, 100);
                }

                // Configure frame_position widget behavior
                if (framePositionWidget) {
                    // Ensure the widget is serialized (saved in workflows)
                    framePositionWidget.serialize = true;
                    
                    // Initially hide frame_position (will be shown if video is loaded)
                    framePositionWidget.hidden = true;
                    
                    // Store original callback
                    const originalFrameCallback = framePositionWidget.callback;
                    
                    // Debounce timer to prevent excessive frame extraction during slider drag
                    let frameUpdateTimer = null;

                    // Override callback to reload video frame when position changes
                    framePositionWidget.callback = function(value) {
                        if (originalFrameCallback) {
                            originalFrameCallback.apply(this, arguments);
                        }

                        // Clear any pending frame update
                        if (frameUpdateTimer) {
                            clearTimeout(frameUpdateTimer);
                        }

                        // Debounce: only extract frame after user stops moving slider for 300ms
                        frameUpdateTimer = setTimeout(() => {
                            // If a video is currently loaded, reload it with new frame position
                            if (imageWidget && imageWidget.value) {
                                const filename = imageWidget.value;
                                const ext = filename.split('.').pop().toLowerCase();
                                const videoExtensions = ['mp4', 'webm', 'mov', 'avi'];
                                
                                if (videoExtensions.includes(ext)) {
                                    loadVideoFrame(node, filename);
                                }
                            }
                        }, 300);
                    };
                }

                // Hook into workflow restoration to load preview for restored values
                const onConfigure = node.onConfigure;
                node.onConfigure = function(info) {
                    const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;

                    // Restore frame_position if it exists in the workflow data
                    if (info && info.widgets_values && framePositionWidget) {
                        // Find the frame_position widget's index
                        const frameWidgetIndex = this.widgets.findIndex(w => w.name === "frame_position");
                        if (frameWidgetIndex >= 0 && info.widgets_values[frameWidgetIndex] !== undefined) {
                            framePositionWidget.value = info.widgets_values[frameWidgetIndex];
                        }
                    }

                    // After workflow is configured, load the preview if a file is selected
                    // Only reload if we're not already displaying this exact image/video frame (prevents refresh on queue)
                    if (imageWidget && imageWidget.value && imageWidget.value !== "(none)") {
                        const isVideo = isVideoFile(imageWidget.value);
                        const currentFramePos = framePositionWidget ? framePositionWidget.value : 0.0;
                        
                        // Check if we already have this exact image/frame loaded
                        const alreadyLoaded = node._loadedImageFilename === imageWidget.value &&
                            (!isVideo || node._loadedFramePosition === currentFramePos);
                        
                        if (!alreadyLoaded) {
                            loadAndDisplayImage(node, imageWidget.value);
                        }
                    }

                    return result;
                };

                // Load initial image if widget has a value on creation
                if (imageWidget) {
                    // Use setTimeout to ensure node is fully initialized
                    setTimeout(() => {
                        if (imageWidget.value && imageWidget.value !== "(none)") {
                            // Only load if not already loaded
                            if (node._loadedImageFilename !== imageWidget.value) {
                                loadAndDisplayImage(node, imageWidget.value);
                            }
                        } else {
                            // Show placeholder for (none) or empty
                            showPlaceholder(node);
                        }
                    }, 10);
                }

                // Add drag-and-drop support for file uploads
                node.onDragOver = function(e) {
                    if (e.dataTransfer && e.dataTransfer.items) {
                        e.preventDefault();
                        e.stopPropagation();
                        return true; // Allow drop
                    }
                    return false;
                };

                node.onDragDrop = async function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    if (!e.dataTransfer || !e.dataTransfer.files || e.dataTransfer.files.length === 0) {
                        return false;
                    }

                    const file = e.dataTransfer.files[0];
                    const filename = file.name;
                    const ext = filename.split('.').pop().toLowerCase();
                    const supportedExtensions = ['png', 'jpg', 'jpeg', 'webp', 'json', 'mp4', 'webm', 'mov', 'avi'];

                    if (!supportedExtensions.includes(ext)) {
                        console.warn(`[PromptExtractor] Unsupported file type: ${ext}`);
                        return false;
                    }

                    // Upload file to input directory
                    const formData = new FormData();
                    formData.append('image', file);
                    formData.append('subfolder', '');
                    formData.append('type', 'input');

                    try {
                        const response = await api.fetchApi('/upload/image', {
                            method: 'POST',
                            body: formData
                        });

                        if (response.ok) {
                            const data = await response.json();
                            const uploadedFilename = data.name;

                            // Update widget with new file
                            if (imageWidget) {
                                // Fetch updated file list from server
                                try {
                                    const listResponse = await api.fetchApi('/prompt-extractor/list-files');
                                    if (listResponse.ok) {
                                        const result = await listResponse.json();
                                        if (result.files && result.files.length > 0) {
                                            // Update widget options with fresh file list
                                            imageWidget.options.values = result.files;
                                        }
                                    }
                                } catch (err) {
                                    console.warn('[PromptExtractor] Could not fetch file list:', err);
                                }
                                
                                // Set the value and trigger callback
                                imageWidget.value = uploadedFilename;
                                
                                // Trigger the original callback to load and display
                                if (imageWidget.callback) {
                                    imageWidget.callback(uploadedFilename);
                                }
                            }

                            console.log(`[PromptExtractor] Uploaded file: ${uploadedFilename}`);
                        } else {
                            console.error('[PromptExtractor] Upload failed:', response.status);
                        }
                    } catch (error) {
                        console.error('[PromptExtractor] Error uploading file:', error);
                    }

                    return true;
                };

                // Add workflow status indicator light and preview button
                const onDrawForeground = node.onDrawForeground;
                node.onDrawForeground = function(ctx) {
                    const result = onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;

                    // Draw status indicator if we have an image and node is not minimized
                    if (node.imgs && node.imgs.length > 0 && !(node.flags && node.flags.collapsed)) {
                        const radius = 7;
                        const titleHeight = LiteGraph.NODE_TITLE_HEIGHT || 30;
                        // Position in top-right, raised 20px to be in title bar
                        const x = node.size[0] - radius - 8;
                        const y = (titleHeight / 2) - 30;

                        // Draw indicator circle
                        ctx.beginPath();
                        ctx.arc(x, y, radius, 0, Math.PI * 2);
                        ctx.fillStyle = node.hasWorkflow ? '#00ff00' : '#ff3333';
                        ctx.fill();
                        ctx.strokeStyle = node.hasWorkflow ? '#054405' : '#550505';
                        ctx.lineWidth = 1;
                        ctx.stroke();

                        // Draw preview play icon (simple triangle) left of workflow indicator - only for images/videos
                        const imageWidget = node.widgets?.find(w => w.name === "image");
                        const currentFile = imageWidget?.value;
                        
                        if (isPreviewableFile(currentFile)) {
                            const playX = node.size[0] - radius - 8 - 34;
                            const playY = (titleHeight / 2) - 30;
                            const triSize = 8;
                            
                            ctx.beginPath();
                            ctx.moveTo(playX - triSize, playY - triSize);
                            ctx.lineTo(playX - triSize, playY + triSize);
                            ctx.lineTo(playX + triSize, playY);
                            ctx.closePath();
                            ctx.fillStyle = node._hoverPreviewIcon ? '#ffffff' : 'rgba(255, 255, 255, 0.7)';
                            ctx.fill();
                            
                            // Store play icon bounds for click detection
                            node._previewIconBounds = {
                                x: playX - triSize - 3,
                                y: playY - triSize - 3,
                                width: triSize * 2 + 6,
                                height: triSize * 2 + 6
                            };
                        } else {
                            node._previewIconBounds = null;
                        }
                    } else {
                        node._previewIconBounds = null;
                    }

                    return result;
                };

                // Add tooltip on hover
                const onMouseMove = node.onMouseMove;
                node.onMouseMove = function(e, localPos, canvas) {
                    const result = onMouseMove ? onMouseMove.apply(this, arguments) : undefined;
                    
                    // Check if hovering over indicator
                    const radius = 7;
                    const titleHeight = LiteGraph.NODE_TITLE_HEIGHT || 30;
                    // Position raised 20px to be in title bar
                    const indicatorX = node.size[0] - radius - 8;
                    const indicatorY = (titleHeight / 2) - 30;
                    
                    const dist = Math.sqrt(
                        Math.pow(localPos[0] - indicatorX, 2) + 
                        Math.pow(localPos[1] - indicatorY, 2)
                    );
                    
                    if (dist <= radius) {
                        canvas.canvas.title = node.hasWorkflow ? 
                            'Workflow metadata found' : 
                            'No workflow metadata';
                        node._hoverPreviewIcon = false;
                    } else if (node._previewIconBounds) {
                        // Check if hovering over preview icon
                        const bounds = node._previewIconBounds;
                        if (localPos[0] >= bounds.x && localPos[0] <= bounds.x + bounds.width &&
                            localPos[1] >= bounds.y && localPos[1] <= bounds.y + bounds.height) {
                            canvas.canvas.style.cursor = 'pointer';
                            canvas.canvas.title = 'Click to preview';
                            node._hoverPreviewIcon = true;
                            node.setDirtyCanvas(true);
                        } else {
                            if (node._hoverPreviewIcon) {
                                node._hoverPreviewIcon = false;
                                node.setDirtyCanvas(true);
                            }
                            canvas.canvas.style.cursor = '';
                        }
                    }
                    
                    return result;
                };
                
                // Handle click on preview icon
                const onMouseDown = node.onMouseDown;
                node.onMouseDown = function(e, localPos, canvas) {
                    // Check if clicking on preview icon
                    if (node._previewIconBounds && node.imgs && node.imgs.length > 0) {
                        const bounds = node._previewIconBounds;
                        if (localPos[0] >= bounds.x && localPos[0] <= bounds.x + bounds.width &&
                            localPos[1] >= bounds.y && localPos[1] <= bounds.y + bounds.height) {
                            // Open preview modal
                            const imageWidget = node.widgets?.find(w => w.name === "image");
                            if (imageWidget && imageWidget.value) {
                                if (node._isVideoFile) {
                                    showVideoPreviewModal(imageWidget.value);
                                } else {
                                    showImagePreviewModal(imageWidget.value);
                                }
                            }
                            return true; // Consume the event
                        }
                    }
                    
                    return onMouseDown ? onMouseDown.apply(this, arguments) : undefined;
                };

                return result;
            };
        }
    }
});

/**
 * Extract metadata and update workflow indicator (without affecting display)
 */
async function extractAndUpdateMetadata(node, filename) {
    if (!filename || filename === "(none)") {
        node.hasWorkflow = false;
        node.setDirtyCanvas(true, true);
        return;
    }

    try {
        const ext = filename.split('.').pop().toLowerCase();
        
        // Split path into filename and subfolder for proper URL handling
        let actualFilename = filename;
        let subfolder = "";
        
        if (filename.includes('/')) {
            const lastSlash = filename.lastIndexOf('/');
            subfolder = filename.substring(0, lastSlash);
            actualFilename = filename.substring(lastSlash + 1);
        }
        
        // Build URL with subfolder parameter if present
        let fileUrl = `/view?filename=${encodeURIComponent(actualFilename)}&type=input`;
        if (subfolder) {
            fileUrl += `&subfolder=${encodeURIComponent(subfolder)}`;
        }
        
        // Fetch the file to extract metadata
        const response = await fetch(fileUrl);
        if (!response.ok) {
            console.warn(`[PromptExtractor] Failed to fetch file for metadata: ${filename}`);
            node.hasWorkflow = false;
            node.setDirtyCanvas(true, true);
            return;
        }
        
        const fileBlob = await response.blob();
        let metadata = null;

        // Extract based on file type
        if (ext === 'png') {
            metadata = await getPNGMetadata(fileBlob);
        } else if (['jpg', 'jpeg', 'webp'].includes(ext)) {
            metadata = await getJPEGMetadata(fileBlob);
        } else if (ext === 'json') {
            metadata = await getJSONMetadata(fileBlob);
        } else if (['mp4', 'webm', 'mov', 'avi'].includes(ext)) {
            console.log(`[PromptExtractor] Attempting JavaScript video metadata extraction for: ${filename}`);
            metadata = await getVideoMetadata(fileBlob);
            
            // If JavaScript parser failed, ask Python to try with ffprobe
            if (metadata === null) {
                console.log(`[PromptExtractor] JavaScript parser returned null, trying Python ffprobe for: ${filename}`);
                metadata = await extractVideoMetadataWithPython(filename);
                
                if (metadata) {
                    console.log(`[PromptExtractor] Successfully got metadata from Python for: ${filename}`);
                } else {
                    console.log(`[PromptExtractor] Python also returned no metadata for: ${filename}`);
                }
            } else {
                console.log(`[PromptExtractor] JavaScript successfully extracted metadata for: ${filename}`);
            }
        }

        // Cache metadata for Python backend (only if we extracted it and Python doesn't already have it)
        if (metadata !== null) {
            await cacheFileMetadata(filename, metadata);
        }

        // Update workflow status
        if (ext === 'json') {
            node.hasWorkflow = !!(metadata && (metadata.workflow || (metadata.nodes && metadata.links)));
        } else {
            // Check for ComfyUI workflow or A1111 parameters
            node.hasWorkflow = !!(metadata && (metadata.workflow || metadata.parameters));
        }

        // Force canvas redraw to update indicator
        node.setDirtyCanvas(true, true);
        app.graph.setDirtyCanvas(true, true);
    } catch (error) {
        console.error("[PromptExtractor] Error extracting metadata:", error);
        node.hasWorkflow = false;
        node.setDirtyCanvas(true, true);
    }
}

/**
 * Load and display an image in the node
 */
async function loadAndDisplayImage(node, filename) {
    if (!filename) {
        // Show placeholder for empty
        showPlaceholder(node);
        return;
    }

    // Check file extension to determine if it's an image or video
    const ext = filename.split('.').pop().toLowerCase();
    const imageExtensions = ['png', 'jpg', 'jpeg', 'webp'];
    const videoExtensions = ['mp4', 'webm', 'mov', 'avi'];

    if (videoExtensions.includes(ext)) {
        // It's a video - extract and display frame at specified position
        loadVideoFrame(node, filename);
        return;
    }

    if (!imageExtensions.includes(ext)) {
        // Handle JSON files
        if (ext === 'json') {
            loadJSONFile(node, filename);
        } else {
            // Unknown file type - show placeholder
            showPlaceholder(node);
        }
        return;
    }

    // It's an image - load, display, and extract metadata
    loadImageFile(node, filename);
}

/**
 * Load an image file, display it, and extract metadata
 */
async function loadImageFile(node, filename) {
    try {
        // Fetch the image file to extract metadata
        const imageBlob = await fetch(`/view?filename=${encodeURIComponent(filename)}&type=input`)
            .then(res => res.blob());

        // Extract metadata from image file (PNG or JPEG/WebP)
        const ext = filename.split('.').pop().toLowerCase();
        let metadata = null;

        if (ext === 'png') {
            metadata = await getPNGMetadata(imageBlob);
        } else if (['jpg', 'jpeg', 'webp'].includes(ext)) {
            metadata = await getJPEGMetadata(imageBlob);
        }

        // Cache metadata (or lack thereof) for Python backend
        await cacheFileMetadata(filename, metadata);

        // Update workflow status flag - check for workflow or parameters
        node.hasWorkflow = !!(metadata && (metadata.workflow || metadata.parameters));
        
        // Force canvas redraw to update indicator immediately
        node.setDirtyCanvas(true, true);
        app.graph.setDirtyCanvas(true, true);

        // Load and display the image
        const img = new Image();
        img.onload = () => {
            node.imgs = [img];
            node.imageIndex = 0;
            // Track that this image is now loaded
            node._loadedImageFilename = filename;

            // Resize node to fit image (like Load Image does)
            const targetWidth = Math.max(node.size[0], 256);
            const targetHeight = Math.max(node.size[1], img.naturalHeight * (targetWidth / img.naturalWidth) + 100);
            node.setSize([targetWidth, targetHeight]);

            node.setDirtyCanvas(true, true);
            app.graph.setDirtyCanvas(true, true);
        };

        img.onerror = () => {
            console.error(`[PromptExtractor] Failed to load image: ${filename}`);
            showPlaceholder(node);
        };

        // Load from input directory
        img.src = `/view?filename=${encodeURIComponent(filename)}&type=input&${Date.now()}`;
    } catch (error) {
        console.error("[PromptExtractor] Error loading image:", error);
        showPlaceholder(node);
    }
}

/**
 * Load a JSON file and extract metadata
 */
async function loadJSONFile(node, filename) {
    try {
        // Fetch the JSON file
        const jsonBlob = await fetch(`/view?filename=${encodeURIComponent(filename)}&type=input`)
            .then(res => res.blob());

        // Extract metadata from JSON file
        const metadata = await getJSONMetadata(jsonBlob);
        // Cache metadata (or lack thereof) for Python backend
        await cacheFileMetadata(filename, metadata);

        // Update workflow status flag
        // Check if metadata has workflow property OR if metadata itself is a workflow (has nodes/links)
        node.hasWorkflow = !!(metadata && (metadata.workflow || (metadata.nodes && metadata.links)));
        // Track that this JSON is now loaded
        node._loadedImageFilename = filename;
        
        // Force canvas redraw to update indicator immediately
        node.setDirtyCanvas(true, true);
        app.graph.setDirtyCanvas(true, true);

        // Show placeholder for JSON files (no visual preview)
        showPlaceholder(node);
    } catch (error) {
        console.error("[PromptExtractor] Error loading JSON:", error);
        showPlaceholder(node);
    }
}

/**
 * Load frame from a video file at specified position and extract metadata
 */
async function loadVideoFrame(node, filename) {
    try {
        // Get frame position from the widget (0.0 to 1.0)
        const framePositionWidget = node.widgets?.find(w => w.name === "frame_position");
        const framePosition = framePositionWidget ? framePositionWidget.value : 0.0;

        // Split path into filename and subfolder for ComfyUI's /view endpoint
        let actualFilename = filename;
        let subfolder = "";
        
        if (filename.includes('/')) {
            const lastSlash = filename.lastIndexOf('/');
            subfolder = filename.substring(0, lastSlash);
            actualFilename = filename.substring(lastSlash + 1);
        }
        
        // Build URL with subfolder parameter if present
        let videoUrl = `/view?filename=${encodeURIComponent(actualFilename)}&type=input`;
        if (subfolder) {
            videoUrl += `&subfolder=${encodeURIComponent(subfolder)}`;
        }

        // Fetch the video file to extract metadata
        const videoBlob = await fetch(videoUrl)
            .then(res => res.blob());

        // Extract metadata from video file
        const metadata = await getVideoMetadata(videoBlob);
        // Cache metadata (or lack thereof) for Python backend
        await cacheFileMetadata(filename, metadata);

        // Update workflow status flag - check for workflow or parameters
        node.hasWorkflow = !!(metadata && (metadata.workflow || metadata.parameters));
        
        // Force canvas redraw to update indicator immediately
        node.setDirtyCanvas(true, true);
        app.graph.setDirtyCanvas(true, true);

        // Create a video element for frame extraction
        const video = document.createElement('video');
        video.crossOrigin = 'anonymous';
        video.preload = 'metadata';

        video.onloadedmetadata = () => {
            // Calculate frame time based on position (0.0 = start, 1.0 = end)
            const frameTime = framePosition * Math.max(0, video.duration - 0.1);
            video.currentTime = frameTime;
        };

        video.onseeked = () => {
            // Create canvas to draw the video frame
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');

            // Draw the video frame to canvas
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert canvas to image
            const img = new Image();
            img.onload = () => {
                // Display the frame
                node.imgs = [img];
                node.imageIndex = 0;
                // Track that this video frame is now loaded at this specific position
                node._loadedImageFilename = filename;
                node._loadedFramePosition = framePosition;

                // Cache frame as base64 for Python backend
                const frameData = canvas.toDataURL('image/png');
                cacheVideoFrame(filename, frameData, framePosition);

                // Resize node to fit image
                const targetWidth = Math.max(node.size[0], 256);
                const targetHeight = Math.max(node.size[1], img.naturalHeight * (targetWidth / img.naturalWidth) + 100);
                node.setSize([targetWidth, targetHeight]);

                node.setDirtyCanvas(true, true);
                app.graph.setDirtyCanvas(true, true);
            };

            img.onerror = () => {
                console.error(`[PromptExtractor] Failed to create image from video frame at position ${framePosition.toFixed(2)}`);
                showPlaceholder(node);
            };

            img.src = canvas.toDataURL('image/png');
        };

        video.onerror = () => {
            console.error(`[PromptExtractor] Failed to load video: ${filename}`);
            showPlaceholder(node);
        };

        // Load video from input directory (using same URL with subfolder support)
        video.src = videoUrl + `&${Date.now()}`;
    } catch (error) {
        console.error("[PromptExtractor] Error loading video:", error);
        showPlaceholder(node);
    }
}

/**
 * Show placeholder image for non-image files
 */
function showPlaceholder(node) {
    // Clear the loaded filename and frame position since we're showing a placeholder
    node._loadedImageFilename = null;
    node._loadedFramePosition = null;
    
    const placeholderImg = new Image();
    placeholderImg.src = PLACEHOLDER_IMAGE_PATH;
    placeholderImg.onload = () => {
        node.imgs = [placeholderImg];
        node.imageIndex = 0;
        
        // Resize node to fit placeholder image
        const targetWidth = Math.max(node.size[0], 256);
        const targetHeight = Math.max(node.size[1], placeholderImg.naturalHeight * (targetWidth / placeholderImg.naturalWidth) + 100);
        node.setSize([targetWidth, targetHeight]);
        
        node.setDirtyCanvas(true, true);
        app.graph.setDirtyCanvas(true, true);
    };
}

console.log("[PromptExtractor] Extension loaded");
