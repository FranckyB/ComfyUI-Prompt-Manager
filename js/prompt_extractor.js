/**
 * PromptExtractor Extension for ComfyUI
 * Adds image preview functionality for the extractor node
 */

import { app } from "../../scripts/app.js";

// Placeholder image path - loaded from static PNG file
const PLACEHOLDER_IMAGE_PATH = new URL("./placeholder.png", import.meta.url).href;

app.registerExtension({
    name: "PromptExtractor",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PromptExtractor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                
                const node = this;
                
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
                    
                    // Load initial image if one is selected
                    if (imageWidget.value) {
                        loadAndDisplayImage(node, imageWidget.value);
                    }
                }
                
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
        // It's a video - extract and display first frame
        loadVideoFirstFrame(node, filename);
        return;
    }
    
    if (!imageExtensions.includes(ext)) {
        // Not an image or video (json, etc) - show placeholder
        showPlaceholder(node);
        return;
    }
    
    // It's an image - load and display it
    try {
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
 * Load first frame from a video file
 */
async function loadVideoFirstFrame(node, filename) {
    try {
        // Create a video element
        const video = document.createElement('video');
        video.crossOrigin = 'anonymous';
        video.preload = 'metadata';
        
        video.onloadedmetadata = () => {
            // Seek to first frame
            video.currentTime = 0;
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
                node.imgs = [img];
                node.imageIndex = 0;
                
                // Resize node to fit image
                const targetWidth = Math.max(node.size[0], 256);
                const targetHeight = Math.max(node.size[1], img.naturalHeight * (targetWidth / img.naturalWidth) + 100);
                node.setSize([targetWidth, targetHeight]);
                
                node.setDirtyCanvas(true, true);
                app.graph.setDirtyCanvas(true, true);
            };
            
            img.onerror = () => {
                console.error(`[PromptExtractor] Failed to create image from video frame`);
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
