import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Latent Preview Extension for Prompt Manager
 * Provides animated previews during video model sampling.
 * Compatible with VideoHelperSuite - will defer to VHS if installed.
 */

// Track nodes that have latent preview widgets
let latentPreviewNodes = new Set();

// Store preview images per node
let previewImagesDict = {};

// Animation intervals for each node
let animateIntervals = {};

// Text decoder for binary messages
const textDecoder = new TextDecoder();

/**
 * Get a node by ID, handling subgraphs
 */
function getNodeById(id) {
    // Try main graph first
    let node = app.graph.getNodeById(id);
    if (node) return node;
    
    // Try subgraphs if available
    if (app.graph.subgraphs) {
        for (const subgraph of app.graph.subgraphs.values()) {
            node = subgraph.getNodeById(id);
            if (node) return node;
        }
    }
    return null;
}

/**
 * Fit the node height to accommodate the preview widget
 */
function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize()[1]]);
    node.graph?.setDirtyCanvas(true, true);
}

/**
 * Allow drag from the preview widget
 */
function allowDragFromWidget(widget) {
    widget.element.addEventListener('pointerdown', (e) => {
        if (e.button === 0) {
            // Prevent default to allow canvas interaction
            e.stopPropagation();
        }
    });
}

/**
 * Get or create the canvas context for latent preview
 */
function getLatentPreviewCtx(id, width, height) {
    const node = getNodeById(id);
    if (!node) {
        return undefined;
    }

    let previewWidget = node.widgets?.find((w) => w.name === "pmlatentpreview");
    
    if (!previewWidget) {
        // Check for and remove any native preview
        const nativePreview = node.widgets?.findIndex((w) => w.name === '$$canvas-image-preview');
        if (nativePreview >= 0) {
            node.imgs = [];
            node.widgets.splice(nativePreview, 1);
        }
        
        // Create canvas element
        const canvasEl = document.createElement("canvas");
        canvasEl.style.width = "100%";
        
        previewWidget = node.addDOMWidget("pmlatentpreview", "pmcanvas", canvasEl, {
            serialize: false,
            hideOnZoom: false,
        });
        previewWidget.serialize = false;
        
        allowDragFromWidget(previewWidget);
        
        // Forward mouse events to canvas for interaction
        const forwardEvent = (eventName, handler) => {
            canvasEl.addEventListener(eventName, (e) => {
                e.preventDefault();
                return handler(e);
            }, true);
        };
        
        forwardEvent('contextmenu', (e) => app.canvas._mousedown_callback(e));
        forwardEvent('pointerdown', (e) => app.canvas._mousedown_callback(e));
        forwardEvent('mousewheel', (e) => app.canvas._mousewheel_callback(e));
        forwardEvent('pointermove', (e) => app.canvas._mousemove_callback(e));
        forwardEvent('pointerup', (e) => app.canvas._mouseup_callback(e));

        previewWidget.computeSize = function(width) {
            if (this.aspectRatio) {
                let height = (node.size[0] - 20) / this.aspectRatio + 10;
                if (!(height > 0)) {
                    height = 0;
                }
                this.computedHeight = height + 10;
                return [width, height];
            }
            return [width, -4]; // No loaded src, widget should not display
        };
    }
    
    const canvasEl = previewWidget.element;
    if (!previewWidget.ctx || canvasEl.width !== width || canvasEl.height !== height) {
        previewWidget.aspectRatio = width / height;
        canvasEl.width = width;
        canvasEl.height = height;
        fitHeight(node);
    }
    
    return canvasEl.getContext("2d");
}

/**
 * Begin animated latent preview for a node
 */
function beginLatentPreview(id, previewImages, rate) {
    latentPreviewNodes.add(id);
    
    if (animateIntervals[id]) {
        clearInterval(animateIntervals[id]);
    }
    
    let displayIndex = 0;
    const node = getNodeById(id);
    
    // Initialize progress to avoid race condition
    if (node) {
        node.progress = 0;
    }
    
    animateIntervals[id] = setInterval(() => {
        const currentNode = getNodeById(id);
        if (!currentNode?.progress && currentNode?.progress !== 0) {
            clearInterval(animateIntervals[id]);
            delete animateIntervals[id];
            return;
        }
        
        // Check if we're still on the right graph
        if (app.canvas.graph.rootGraph !== currentNode.graph?.rootGraph) {
            clearInterval(animateIntervals[id]);
            delete animateIntervals[id];
            return;
        }
        
        if (!previewImages[displayIndex]) {
            return;
        }
        
        const ctx = getLatentPreviewCtx(
            id,
            previewImages[displayIndex].width,
            previewImages[displayIndex].height
        );
        ctx?.drawImage?.(previewImages[displayIndex], 0, 0);
        
        displayIndex = (displayIndex + 1) % previewImages.length;
    }, 1000 / rate);
}

/**
 * Check if VideoHelperSuite is providing latent preview
 */
function isVHSLatentPreviewActive() {
    // Check if VHS extension is registered and has latent preview enabled
    const vhsExtension = app.extensions.find(ext => ext.name === "VideoHelperSuite.Core");
    if (vhsExtension) {
        try {
            const vhsSetting = app.ui.settings.getSettingValue("VHS.LatentPreview");
            return vhsSetting === true;
        } catch (e) {
            // Setting doesn't exist or error
        }
    }
    return false;
}

app.registerExtension({
    name: "PromptManager.LatentPreview",
    settings: [
        {
            id: "PromptManager.LatentPreview",
            category: ["Prompt Manager", "5. Video Sampling", "Animated Latent Preview"],
            name: "Display animated previews when sampling",
            tooltip: "Enable animated preview during video model sampling (KSampler). Will be disabled if VideoHelperSuite provides this feature.",
            type: "boolean",
            defaultValue: false,
            onChange(value) {
                if (!value) {
                    // Remove any preview widgets when disabled
                    for (const id of latentPreviewNodes) {
                        const node = app.graph?.getNodeById(id);
                        const widgetIndex = node?.widgets?.findIndex((w) => w.name === 'pmlatentpreview');
                        if (widgetIndex >= 0) {
                            const widget = node.widgets.splice(widgetIndex, 1)[0];
                            widget.onRemove?.();
                        }
                    }
                    latentPreviewNodes = new Set();
                }
            },
        },
        {
            id: "PromptManager.LatentPreviewRate",
            category: ["Prompt Manager", "5. Video Sampling", "Playback Rate Override"],
            name: "Playback rate override",
            tooltip: "Force a specific frame rate for latent preview playback. Set to 0 for auto-detect based on video model. This is the preview FPS, not the output video FPS.",
            type: "number",
            attrs: {
                min: 0,
                step: 1,
                max: 60,
            },
            defaultValue: 0,
        },
    ],
    
    async setup() {
        // Hook into graphToPrompt to pass our settings to the backend
        const originalGraphToPrompt = app.graphToPrompt;
        
        app.graphToPrompt = async function() {
            const res = await originalGraphToPrompt.apply(this, arguments);
            
            // Check if VHS is handling latent preview
            const vhsActive = isVHSLatentPreviewActive();
            
            if (!vhsActive) {
                // Add our settings to the workflow extra data
                res.workflow.extra['PM_latentpreview'] = app.ui.settings.getSettingValue("PromptManager.LatentPreview");
                res.workflow.extra['PM_latentpreviewrate'] = app.ui.settings.getSettingValue("PromptManager.LatentPreviewRate");
            }
            
            return res;
        };
        
        console.log("[PromptManager] Latent preview extension loaded");
    },
    
    async init() {
        // Clear preview nodes on execution complete
        api.addEventListener('executing', ({ detail }) => {
            if (detail === null) {
                // Execution complete - clean up progress indicators
                for (const id of latentPreviewNodes) {
                    const node = getNodeById(id);
                    if (node) {
                        delete node.progress;
                    }
                }
            }
        });
        
        // Listen for latent preview initialization from backend
        api.addEventListener('PM_latentpreview', ({ detail }) => {
            if (detail.id == null) {
                return;
            }
            
            // Skip if VHS is active
            if (isVHSLatentPreviewActive()) {
                return;
            }
            
            const previewImages = previewImagesDict[detail.id] = [];
            previewImages.length = detail.length;

            // Handle node ID parts (for subgraphs)
            const idParts = detail.id.split(':');
            for (let i = 1; i <= idParts.length; i++) {
                const id = idParts.slice(0, i).join(':');
                beginLatentPreview(id, previewImages, detail.rate);
            }
        });
        
        // Listen for binary preview images
        api.addEventListener('b_preview', async (e) => {
            // Only handle if we have active animations and VHS isn't handling it
            if (Object.keys(animateIntervals).length === 0 || isVHSLatentPreviewActive()) {
                return;
            }
            
            e.preventDefault();
            e.stopImmediatePropagation();
            e.stopPropagation();
            
            const dv = new DataView(await e.detail.slice(0, 24).arrayBuffer());
            const index = dv.getUint32(4);
            const idlen = dv.getUint8(8);
            const id = textDecoder.decode(dv.buffer.slice(9, 9 + idlen));
            
            // Only process if this is our preview (PM_ prefix in the workflow extra)
            if (previewImagesDict[id]) {
                previewImagesDict[id][index] = await window.createImageBitmap(e.detail.slice(24));
            }
            
            return false;
        }, true);
    },
});
