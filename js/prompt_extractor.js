/**
 * PromptExtractor Extension for ComfyUI
 * Adds file browser functionality for selecting images/videos/workflows
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
                
                // Set initial size (compact)
                this.setSize([300, 160]);
                
                // Find the file_path widget and enhance it
                const filePathWidget = this.widgets?.find(w => w.name === "file_path");
                if (filePathWidget) {
                    // Store original callback
                    const originalCallback = filePathWidget.callback;
                    
                    // Add browse button after the widget
                    // Use requestAnimationFrame for proper timing after node is fully created
                    requestAnimationFrame(() => {
                        addBrowseButton(node, filePathWidget);
                    });
                }
                
                return result;
            };
            
            // Enforce minimum node size
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                size[0] = Math.max(250, size[0]);
                size[1] = Math.max(150, size[1]);
                return onResize ? onResize.apply(this, arguments) : size;
            };
        }
    }
});

/**
 * Add a browse button to the file path widget
 */
function addBrowseButton(node, filePathWidget) {
    // Create button container with proper styling for DOM widget
    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        gap: 4px;
        padding: 2px 4px;
        align-items: center;
        width: 100%;
        box-sizing: border-box;
        position: relative;
    `;
    
    // Browse button - styled to match Save/New buttons in PromptManager
    const browseBtn = document.createElement("button");
    browseBtn.textContent = "Browse";
    browseBtn.style.cssText = `
        flex: 1;
        min-width: 80px;
        padding: 6px 8px;
        cursor: pointer;
        background-color: #222;
        color: #fff;
        border: 1px solid #444;
        border-radius: 6px;
        font-size: 11px;
        font-weight: normal;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: normal;
        text-align: center;
        height: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    `;
    browseBtn.onclick = () => openFilePicker(node, filePathWidget);
    
    // Prevent right-click context menu
    container.addEventListener("contextmenu", (e) => e.preventDefault());
    
    container.appendChild(browseBtn);
    
    // Add as DOM widget with proper options to avoid clipping issues
    const widget = node.addDOMWidget("browse_button", "div", container, {
        hideOnZoom: true,
        serialize: false
    });
    widget.computeSize = function(width) {
        return [width, 26];
    };
    
    // Setup drag and drop on the node
    setupDragAndDrop(node, filePathWidget);
    
    // Force canvas and node to redraw so widget appears immediately
    node.setDirtyCanvas(true, true);
    app.canvas.setDirty(true, true);
    
    // Schedule another redraw to ensure DOM widget is visible
    requestAnimationFrame(() => {
        app.canvas.draw(true, true);
    });
}

/**
 * Setup drag and drop functionality for the node
 */
function setupDragAndDrop(node, filePathWidget) {
    const canvas = app.canvas.canvas;
    
    // Track if we're dragging over our node
    let isDraggingOver = false;
    
    // Store original draw function to add visual feedback
    const originalDrawBackground = node.onDrawBackground;
    node.onDrawBackground = function(ctx) {
        if (originalDrawBackground) {
            originalDrawBackground.call(this, ctx);
        }
        
        // Draw drop zone highlight when dragging over
        if (isDraggingOver) {
            ctx.fillStyle = "rgba(66, 153, 225, 0.2)";
            ctx.strokeStyle = "rgba(66, 153, 225, 0.8)";
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.fillRect(0, 0, this.size[0], this.size[1]);
            ctx.strokeRect(0, 0, this.size[0], this.size[1]);
            ctx.setLineDash([]);
            
            // Draw drop text
            ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
            ctx.font = "14px Arial";
            ctx.textAlign = "center";
            ctx.fillText("Drop file here", this.size[0] / 2, this.size[1] / 2);
        }
    };
    
    // Helper to check if mouse is over this node
    const isOverNode = (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / app.canvas.ds.scale - app.canvas.ds.offset[0];
        const y = (e.clientY - rect.top) / app.canvas.ds.scale - app.canvas.ds.offset[1];
        
        return x >= node.pos[0] && x <= node.pos[0] + node.size[0] &&
               y >= node.pos[1] && y <= node.pos[1] + node.size[1];
    };
    
    // Drag over handler
    const handleDragOver = (e) => {
        if (isOverNode(e)) {
            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
            if (!isDraggingOver) {
                isDraggingOver = true;
                app.graph.setDirtyCanvas(true, true);
            }
        } else if (isDraggingOver) {
            isDraggingOver = false;
            app.graph.setDirtyCanvas(true, true);
        }
    };
    
    // Drag leave handler
    const handleDragLeave = (e) => {
        if (isDraggingOver && !isOverNode(e)) {
            isDraggingOver = false;
            app.graph.setDirtyCanvas(true, true);
        }
    };
    
    // Drop handler
    const handleDrop = (e) => {
        if (isOverNode(e)) {
            e.preventDefault();
            e.stopPropagation();
            isDraggingOver = false;
            app.graph.setDirtyCanvas(true, true);
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                const ext = file.name.split('.').pop().toLowerCase();
                const validExtensions = ['png', 'jpg', 'jpeg', 'webp', 'json', 'mp4', 'webm', 'mov'];
                
                if (validExtensions.includes(ext)) {
                    uploadFile(file, node, filePathWidget);
                } else {
                    console.warn(`[PromptExtractor] Invalid file type: ${ext}`);
                    alert(`Invalid file type. Supported: ${validExtensions.join(', ')}`);
                }
            }
        }
    };
    
    // Add event listeners to canvas
    canvas.addEventListener("dragover", handleDragOver);
    canvas.addEventListener("dragleave", handleDragLeave);
    canvas.addEventListener("drop", handleDrop);
    
    // Store handlers for cleanup
    node._dragHandlers = { handleDragOver, handleDragLeave, handleDrop };
    
    // Cleanup on node removal
    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function() {
        if (node._dragHandlers) {
            canvas.removeEventListener("dragover", node._dragHandlers.handleDragOver);
            canvas.removeEventListener("dragleave", node._dragHandlers.handleDragLeave);
            canvas.removeEventListener("drop", node._dragHandlers.handleDrop);
        }
        if (originalOnRemoved) {
            originalOnRemoved.call(this);
        }
    };
}

/**
 * Open file picker dialog
 */
function openFilePicker(node, filePathWidget) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".png,.jpg,.jpeg,.webp,.json,.mp4,.webm,.mov";
    
    input.onchange = (e) => {
        const file = e.target.files[0];
        if (file) {
            // For local files, we need to upload or use the path
            // In a browser context, we can't get the full path directly
            // So we'll upload the file to ComfyUI's input directory
            uploadFile(file, node, filePathWidget);
        }
    };
    
    input.click();
}

/**
 * Upload file to ComfyUI input directory
 */
async function uploadFile(file, node, filePathWidget) {
    try {
        // First check if file already exists in input directory
        const checkResponse = await fetch(`/view?filename=${encodeURIComponent(file.name)}&type=input`);
        
        if (checkResponse.ok) {
            // File already exists in input directory, use it directly (no upload needed)
            console.log(`[PromptExtractor] File already exists in input: ${file.name}`);
            filePathWidget.value = file.name;
            
            if (filePathWidget.callback) {
                filePathWidget.callback(file.name);
            }
            
            // Set placeholder image
            const placeholderImg = new Image();
            placeholderImg.src = PLACEHOLDER_IMAGE_PATH;
            placeholderImg.onload = () => {
                node.imgs = [placeholderImg];
                node.imageIndex = 0;
                // Only grow the node if needed, never shrink
                const newSize = node.computeSize();
                const currentSize = node.size;
                const targetWidth = Math.max(currentSize[0], newSize[0]);
                const targetHeight = Math.max(currentSize[1], newSize[1]);
                if (targetWidth > currentSize[0] || targetHeight > currentSize[1]) {
                    node.setSize([targetWidth, targetHeight]);
                }
                node.setDirtyCanvas(true, true);
                app.graph.setDirtyCanvas(true, true);
                requestAnimationFrame(() => {
                    app.canvas.draw(true, true);
                });
            };
            
            node.serialize_widgets = true;
            return;
        }
        
        // File doesn't exist in input, upload to temp directory
        const formData = new FormData();
        formData.append("image", file);
        formData.append("type", "temp");  // Upload to temp since it's not in input
        formData.append("overwrite", "true");  // Overwrite if already in temp
        const response = await fetch("/upload/image", {
            method: "POST",
            body: formData
        });
        
        const result = await response.json();
        
        if (result.name) {
            // Update the file path widget
            filePathWidget.value = result.name;
            
            // Trigger callback if exists
            if (filePathWidget.callback) {
                filePathWidget.callback(result.name);
            }
            
            // Set placeholder image instead of clearing to avoid naturalWidth errors
            const placeholderImg = new Image();
            placeholderImg.src = PLACEHOLDER_IMAGE_PATH;
            placeholderImg.onload = () => {
                node.imgs = [placeholderImg];
                node.imageIndex = 0;
                // Only grow the node if needed, never shrink
                const newSize = node.computeSize();
                const currentSize = node.size;
                const targetWidth = Math.max(currentSize[0], newSize[0]);
                const targetHeight = Math.max(currentSize[1], newSize[1]);
                if (targetWidth > currentSize[0] || targetHeight > currentSize[1]) {
                    node.setSize([targetWidth, targetHeight]);
                }
                node.setDirtyCanvas(true, true);
                app.graph.setDirtyCanvas(true, true);
                requestAnimationFrame(() => {
                    app.canvas.draw(true, true);
                });
            };
            
            node.serialize_widgets = true;
            
            console.log(`[PromptExtractor] File uploaded: ${result.name} - showing placeholder, will refresh on execution`);
        } else {
            console.error("[PromptExtractor] Upload failed:", result);
            alert("Failed to upload file");
        }
    } catch (error) {
        console.error("[PromptExtractor] Upload error:", error);
        alert("Error uploading file: " + error.message);
    }
}

console.log("[PromptExtractor] Extension loaded");
