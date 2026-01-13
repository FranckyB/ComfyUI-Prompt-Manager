/**
 * PromptExtractor Extension for ComfyUI
 * Adds image preview functionality for the extractor node
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

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

                    // Check for ComfyUI metadata
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
                    }
                }

                // Move to next chunk (length + type + data + CRC)
                offset += 12 + chunkLength;

                // Stop if we found both or reached IEND
                if ((prompt && workflow) || chunkType === 'IEND') {
                    break;
                }
            }

            // Return metadata if found
            if (prompt || workflow) {
                resolve({ prompt, workflow });
            } else {
                resolve(null);
            }
        };
        reader.readAsArrayBuffer(file);
    });
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

app.registerExtension({
    name: "PromptExtractor",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PromptExtractor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                const node = this;

                // Track workflow metadata status for indicator
                node.hasWorkflow = false;

                // Find the image widget (combo dropdown)
                const imageWidget = this.widgets?.find(w => w.name === "image");
                if (imageWidget) {
                    // Store original callback
                    const originalCallback = imageWidget.callback;

                    // Override callback to load and display image
                    imageWidget.callback = function(value) {
                        if (originalCallback) {
                            originalCallback.apply(this, arguments);
                        }

                        // Load and display the image
                        loadAndDisplayImage(node, value);
                    };
                }

                // Find the frame_position widget (slider)
                const framePositionWidget = this.widgets?.find(w => w.name === "frame_position");
                if (framePositionWidget) {
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

                    // After workflow is configured, load the preview if a file is selected
                    if (imageWidget && imageWidget.value) {
                        loadAndDisplayImage(node, imageWidget.value);
                    }

                    return result;
                };

                // Load initial image if widget has a value on creation
                if (imageWidget && imageWidget.value) {
                    // Use setTimeout to ensure node is fully initialized
                    setTimeout(() => {
                        loadAndDisplayImage(node, imageWidget.value);
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

                // Add workflow status indicator light
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
                    }
                    
                    return result;
                };

                return result;
            };
        }
    }
});

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

        // Update workflow status flag
        node.hasWorkflow = !!(metadata && metadata.workflow);

        // Load and display the image
        const img = new Image();
        img.onload = () => {
            node.imgs = [img];
            node.imageIndex = 0;

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

        // Fetch the video file to extract metadata
        const videoBlob = await fetch(`/view?filename=${encodeURIComponent(filename)}&type=input`)
            .then(res => res.blob());

        // Extract metadata from video file
        const metadata = await getVideoMetadata(videoBlob);
        // Cache metadata (or lack thereof) for Python backend
        await cacheFileMetadata(filename, metadata);

        // Update workflow status flag
        node.hasWorkflow = !!(metadata && metadata.workflow);

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

        // Load video from input directory
        video.src = `/view?filename=${encodeURIComponent(filename)}&type=input&${Date.now()}`;
    } catch (error) {
        console.error("[PromptExtractor] Error loading video:", error);
        showPlaceholder(node);
    }
}

/**
 * Show placeholder image for non-image files
 */
function showPlaceholder(node) {
    const placeholderImg = new Image();
    placeholderImg.src = PLACEHOLDER_IMAGE_PATH;
    placeholderImg.onload = () => {
        node.imgs = [placeholderImg];
        node.imageIndex = 0;
        node.setDirtyCanvas(true, true);
        app.graph.setDirtyCanvas(true, true);
    };
}

console.log("[PromptExtractor] Extension loaded");
