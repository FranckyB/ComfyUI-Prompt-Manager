/**
 * Video Metadata Reader - VHS Compatible
 * Enables drag-and-drop workflow loading from videos saved by VideoHelperSuite
 * without needing VHS installed.
 * 
 * Based on VideoHelperSuite's videoinfo.js implementation.
 */

import { app } from '../../../scripts/app.js'

/**
 * Parse video file bytes to extract embedded workflow/prompt metadata.
 * Supports webm/mkv (EBML/Matroska) and mp4 (QuickTime) formats.
 */
function getVideoMetadata(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            const videoData = new Uint8Array(event.target.result);
            const dataView = new DataView(videoData.buffer);
            const decoder = new TextDecoder();

            // Check for known valid magic bytes
            if (dataView.getUint32(0) === 0x1A45DFA3) {
                // webm/mkv - EBML/Matroska format
                // Metadata stored in COMMENT tags
                // See: http://wiki.webmproject.org/webm-metadata/global-metadata
                // See: https://www.matroska.org/technical/elements.html
                let offset = 4 + 8; // Skip header, COMMENT is 7 chars + 1 to realign

                while (offset < videoData.length - 16) {
                    // Check for text tags (0x4487 = TagString element)
                    if (dataView.getUint16(offset) === 0x4487) {
                        // Check that name of tag is COMMENT
                        const name = String.fromCharCode(...videoData.slice(offset - 7, offset));
                        if (name === "COMMENT") {
                            // Parse EBML variable-length integer
                            // See: https://github.com/ietf-wg-cellar/ebml-specification/blob/master/specification.markdown
                            let vint = dataView.getUint32(offset + 2);
                            let n_octets = Math.clz32(vint) + 1;
                            if (n_octets < 4) { // 250MB sanity cutoff
                                let length = (vint >> (8 * (4 - n_octets))) & ~(1 << (7 * n_octets));
                                const content = decoder.decode(videoData.slice(offset + 2 + n_octets, offset + 2 + n_octets + length));
                                try {
                                    const json = JSON.parse(content);
                                    resolve(json);
                                    return;
                                } catch (e) {
                                    console.warn("[VideoMetadata] Failed to parse COMMENT JSON:", e);
                                }
                            }
                        }
                    }
                    offset += 1;
                }
            } else if (dataView.getUint32(4) === 0x66747970) {
                // mp4 - QuickTime/ISO format
                // See: https://developer.apple.com/documentation/quicktime-file-format
                // Metadata can be in various locations, search from end

                // First try: Look for ©cmt (comment) atom - VHS style
                let offset = videoData.length - 4;
                while (offset > 16) {
                    if (dataView.getUint32(offset) === 0x64617461) { // 'data' tag
                        if (dataView.getUint32(offset - 8) === 0xa9636d74) { // '©cmt' (comment) tag
                            let size = dataView.getUint32(offset - 4) - 4 * 4;
                            const content = decoder.decode(videoData.slice(offset + 12, offset + 12 + size));
                            try {
                                const json = JSON.parse(content);
                                resolve(json);
                                return;
                            } catch (e) {
                                console.warn("[VideoMetadata] Failed to parse ©cmt JSON:", e);
                            }
                        }
                    }
                    offset -= 1;
                }

                // Second try: Look for 'prompt' or 'workflow' tags in metadata
                // These are stored by our SaveVideoH26x node using movflags=use_metadata_tags
                offset = 0;
                while (offset < videoData.length - 100) {
                    // Look for 'prompt' or 'workflow' strings followed by JSON
                    const slice = decoder.decode(videoData.slice(offset, offset + 20));
                    if (slice.includes('prompt') || slice.includes('workflow')) {
                        // Try to find JSON starting after the key
                        for (let i = 0; i < 50 && offset + i < videoData.length - 1; i++) {
                            if (videoData[offset + i] === 0x7B) { // '{'
                                // Found potential JSON start, try to parse
                                let jsonEnd = offset + i;
                                let braceCount = 0;
                                for (let j = jsonEnd; j < Math.min(jsonEnd + 1000000, videoData.length); j++) {
                                    if (videoData[j] === 0x7B) braceCount++;
                                    if (videoData[j] === 0x7D) braceCount--;
                                    if (braceCount === 0) {
                                        jsonEnd = j + 1;
                                        break;
                                    }
                                }
                                const jsonStr = decoder.decode(videoData.slice(offset + i, jsonEnd));
                                try {
                                    const json = JSON.parse(jsonStr);
                                    if (json.workflow || json.prompt || json.nodes) {
                                        resolve(json.workflow ? json : { workflow: json });
                                        return;
                                    }
                                } catch (e) {
                                    // Not valid JSON, continue searching
                                }
                            }
                        }
                    }
                    offset += 100; // Skip ahead in larger chunks
                }
            }

            // No metadata found
            resolve(undefined);
        };

        reader.onerror = () => {
            console.error("[VideoMetadata] Failed to read file");
            resolve(undefined);
        };

        reader.readAsArrayBuffer(file);
    });
}

/**
 * Check if file is a video by extension
 */
function isVideoFile(file) {
    const videoExtensions = ['.webm', '.mp4', '.mkv', '.mov', '.avi'];
    const name = file?.name?.toLowerCase() || '';
    return videoExtensions.some(ext => name.endsWith(ext));
}

// Store original handler
let originalHandleFile = null;

/**
 * Custom file handler that intercepts video files to extract workflow metadata
 */
async function handleFile(file) {
    // Check if this is a video file
    if (file?.type?.startsWith("video/") || isVideoFile(file)) {
        console.log("[VideoMetadata] Processing video file:", file.name);
        
        try {
            const videoInfo = await getVideoMetadata(file);
            
            if (videoInfo?.workflow) {
                console.log("[VideoMetadata] Found workflow in video, loading...");
                await app.loadGraphData(videoInfo.workflow);
                return;
            } else {
                console.log("[VideoMetadata] No workflow metadata found in video");
            }
        } catch (e) {
            console.error("[VideoMetadata] Error extracting metadata:", e);
        }
    }
    
    // Fall through to original handler
    if (originalHandleFile) {
        return await originalHandleFile.apply(this, arguments);
    }
}

// Register extension
app.registerExtension({
    name: "PromptManager.VideoMetadata",
    
    async setup() {
        // Only hijack if not already done by VHS
        if (!window._vhsVideoMetadataRegistered) {
            // Store and replace the file handler
            originalHandleFile = app.handleFile;
            app.handleFile = handleFile;
            
            // Add video formats to file input accept attribute
            const fileInput = document.getElementById("comfy-file-input");
            if (fileInput && !fileInput.accept.includes("video/")) {
                fileInput.accept += ",video/webm,video/mp4,video/x-matroska";
            }
            
            // Mark as registered to avoid double-registration if VHS loads later
            window._vhsVideoMetadataRegistered = true;
            
            console.log("[VideoMetadata] VHS-compatible video metadata reader registered");
        } else {
            console.log("[VideoMetadata] Video metadata handler already registered (likely by VHS)");
        }
    }
});
