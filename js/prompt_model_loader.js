/**
 * PromptModelLoader Extension for ComfyUI
 * Displays model type and name info on the node, hides CLIP/VAE slots for non-checkpoint models.
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "PromptManager.ModelLoader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PromptModelLoader") return;

        // Hook into onExecuted to receive UI data from Python
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origOnExecuted?.apply(this, arguments);

            const info = message?.model_info?.[0];
            if (!info) return;

            this._applyModelInfo(info.model_type || "Unknown", info.model_name || "");
        };

        // Central method to apply model info (used by both onExecuted and onConfigure)
        nodeType.prototype._applyModelInfo = function (modelType, modelName) {
            const isCheckpoint = modelType === "Checkpoint";

            // Store for display and serialization
            this._modelType = modelType;
            this._modelName = modelName;

            // Update title
            this.title = `Prompt Model Loader [${modelType}]`;

            // Show/hide CLIP and VAE outputs
            this._updateOutputSlots(isCheckpoint);

            // Create info widget
            this._ensureInfoWidget();
            this.setDirtyCanvas(true, true);
        };

        // Save state into the workflow JSON
        const origOnSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            origOnSerialize?.apply(this, arguments);
            if (this._modelType && this._modelName) {
                o._modelLoaderState = {
                    modelType: this._modelType,
                    modelName: this._modelName
                };
            }
        };

        // Restore state when workflow is loaded / tab is switched back
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            origOnConfigure?.apply(this, arguments);
            const state = info?._modelLoaderState;
            if (state && state.modelType && state.modelName) {
                // Defer to next frame so the node is fully set up
                requestAnimationFrame(() => {
                    this._applyModelInfo(state.modelType, state.modelName);
                });
            }
        };

        // Manage output slot visibility
        nodeType.prototype._updateOutputSlots = function (isCheckpoint) {
            if (!this._originalOutputs) {
                // Save original outputs on first run
                this._originalOutputs = this.outputs.map(o => ({ ...o }));
            }

            if (isCheckpoint) {
                // Restore all 3 outputs
                while (this.outputs.length < 3) {
                    const orig = this._originalOutputs[this.outputs.length];
                    this.addOutput(orig.name, orig.type);
                }
            } else {
                // Remove CLIP (index 1) and VAE (index 2) if present
                // Disconnect them first to avoid dangling links
                for (let i = this.outputs.length - 1; i >= 1; i--) {
                    if (this.outputs[i].links && this.outputs[i].links.length > 0) {
                        for (const linkId of [...this.outputs[i].links]) {
                            this.disconnectOutput(i);
                        }
                    }
                    this.removeOutput(i);
                }
            }
            this.computeSize();
        };

        // Info widget helper
        nodeType.prototype._ensureInfoWidget = function () {
            let widget = this.widgets?.find(w => w.name === "_model_info_display");
            if (!widget) {
                widget = {
                    name: "_model_info_display",
                    type: "custom",
                    y: 0,
                    computeSize: () => [0, 44],
                    draw: function (ctx, node, widgetWidth, y) {
                        if (!node._modelName) return;

                        const padding = 8;

                        // Type badge colors
                        const typeColors = {
                            "Checkpoint": "#4CAF50",
                            "Diffusion": "#2196F3",
                            "GGUF": "#FF9800",
                            "NOT FOUND": "#F44336",
                            "Unknown": "#9E9E9E"
                        };
                        const typeText = node._modelType || "Unknown";
                        const badgeColor = typeColors[typeText] || typeColors["Unknown"];

                        // Badge pill
                        ctx.font = "bold 11px Arial";
                        const typeWidth = ctx.measureText(typeText).width + 14;
                        ctx.fillStyle = badgeColor;
                        ctx.beginPath();
                        ctx.roundRect(padding, y + 4, typeWidth, 16, 8);
                        ctx.fill();

                        // Badge text
                        ctx.fillStyle = "#FFFFFF";
                        ctx.textAlign = "left";
                        ctx.textBaseline = "middle";
                        ctx.fillText(typeText, padding + 7, y + 12);

                        // Model name below badge
                        ctx.font = "12px Arial";
                        ctx.fillStyle = "#CCCCCC";
                        const maxNameWidth = widgetWidth - padding * 2;
                        let displayName = node._modelName;
                        while (ctx.measureText(displayName).width > maxNameWidth && displayName.length > 5) {
                            displayName = displayName.slice(0, -4) + "...";
                        }
                        ctx.fillText(displayName, padding, y + 34);
                    }
                };
                if (!this.widgets) this.widgets = [];
                this.widgets.push(widget);
                this.computeSize();
            }
        };

        // Reset display on clear/reconnect
        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            origOnConnectionsChange?.apply(this, arguments);
            // Reset info when input connection changes
            if (arguments[1] === 0 && arguments[0] === 1) {
                this._modelType = null;
                this._modelName = null;
                this.title = "Prompt Model Loader";
                // Restore all outputs
                if (this._originalOutputs) {
                    while (this.outputs.length < 3) {
                        const orig = this._originalOutputs[this.outputs.length];
                        this.addOutput(orig.name, orig.type);
                    }
                }
                // Remove info widget
                if (this.widgets) {
                    const idx = this.widgets.findIndex(w => w.name === "_model_info_display");
                    if (idx >= 0) {
                        this.widgets.splice(idx, 1);
                        this.computeSize();
                    }
                }
                this.setDirtyCanvas(true, true);
            }
        };
    }
});
