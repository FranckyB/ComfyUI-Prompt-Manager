/**
 * PromptModelLoader Extension for ComfyUI
 * Displays model type and name info on the node.
 * Dynamically shows/hides Model B outputs when two models are found.
 * Hides CLIP/VAE slots for non-checkpoint models.
 */

import { app } from "../../scripts/app.js";

// --- Draw model info (shared between single and dual) ---
const TYPE_COLORS = {
    "Checkpoint": "#4CAF50",
    "Diffusion": "#2196F3",
    "GGUF": "#FF9800",
    "NOT FOUND": "#F44336",
    "Unknown": "#9E9E9E"
};

function drawModelRow(ctx, typeText, modelName, yOffset, label, widgetWidth) {
    const padding = 8;
    const badgeColor = TYPE_COLORS[typeText] || TYPE_COLORS["Unknown"];
    const badgeLabel = label ? `${label} ${typeText}` : typeText;

    // Badge pill
    ctx.font = "bold 11px Arial";
    const typeWidth = ctx.measureText(badgeLabel).width + 14;
    ctx.fillStyle = badgeColor;
    ctx.beginPath();
    ctx.roundRect(padding, yOffset + 4, typeWidth, 16, 8);
    ctx.fill();

    // Badge text
    ctx.fillStyle = "#FFFFFF";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(badgeLabel, padding + 7, yOffset + 12);

    // Model name to the right of badge
    ctx.font = "12px Arial";
    ctx.fillStyle = typeText === "NOT FOUND" ? "#F44336" : "#CCCCCC";
    const nameX = padding + typeWidth + 8;
    const maxNameWidth = widgetWidth - nameX - padding;
    let displayName = modelName || "";
    while (ctx.measureText(displayName).width > maxNameWidth && displayName.length > 5) {
        displayName = displayName.slice(0, -4) + "...";
    }
    ctx.fillText(displayName, nameX, yOffset + 12);
}

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

            this._applyModelInfo(info);
        };

        // Central method to apply model info
        nodeType.prototype._applyModelInfo = function (info) {
            const typeA = info.model_type_a || "Unknown";
            const nameA = info.model_name_a || "";
            const hasB = !!info.has_model_b;
            const typeB = info.model_type_b || "";
            const nameB = info.model_name_b || "";
            const isCheckpointA = typeA === "Checkpoint";
            const isCheckpointB = typeB === "Checkpoint";

            // Store for display and serialization
            this._modelInfo = info;

            // Update title
            this.title = hasB
                ? `Prompt Model Loader [${typeA} + ${typeB}]`
                : `Prompt Model Loader [${typeA}]`;

            // Update output slots
            this._updateOutputSlots(hasB, isCheckpointA, isCheckpointB);

            // Create/update info widget
            this._ensureInfoWidget();
            this.setDirtyCanvas(true, true);
        };

        // Save state into the workflow JSON
        const origOnSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            origOnSerialize?.apply(this, arguments);
            if (this._modelInfo) {
                o._modelLoaderState = this._modelInfo;
            }
        };

        // Restore state when workflow is loaded / tab is switched back
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            origOnConfigure?.apply(this, arguments);
            const state = info?._modelLoaderState;
            if (state) {
                requestAnimationFrame(() => {
                    this._applyModelInfo(state);
                });
            }
        };

        // Manage output slot visibility
        // Full outputs: model_a(0), clip_a(1), vae_a(2), model_b(3), clip_b(4), vae_b(5)
        nodeType.prototype._updateOutputSlots = function (hasB, isCheckpointA, isCheckpointB) {
            if (!this._originalOutputs) {
                this._originalOutputs = this.outputs.map(o => ({ ...o }));
            }

            // Build desired output list
            const desired = [];

            // Model A always present
            desired.push(this._originalOutputs[0]); // model_a

            // CLIP A / VAE A only for checkpoints
            if (isCheckpointA) {
                desired.push(this._originalOutputs[1]); // clip_a
                desired.push(this._originalOutputs[2]); // vae_a
            }

            // Model B outputs only when two models found
            if (hasB) {
                desired.push(this._originalOutputs[3]); // model_b
                if (isCheckpointB) {
                    desired.push(this._originalOutputs[4]); // clip_b
                    desired.push(this._originalOutputs[5]); // vae_b
                }
            }

            // Disconnect outputs that will be removed
            for (let i = this.outputs.length - 1; i >= desired.length; i--) {
                if (this.outputs[i]?.links?.length > 0) {
                    for (const linkId of [...this.outputs[i].links]) {
                        this.disconnectOutput(i);
                    }
                }
                this.removeOutput(i);
            }

            // Add/update outputs to match desired
            for (let i = 0; i < desired.length; i++) {
                if (i < this.outputs.length) {
                    this.outputs[i].name = desired[i].name;
                    this.outputs[i].type = desired[i].type;
                } else {
                    this.addOutput(desired[i].name, desired[i].type);
                }
            }

            this.computeSize();
        };

        // Info widget helper
        nodeType.prototype._ensureInfoWidget = function () {
            const info = this._modelInfo;
            if (!info) return;

            const hasB = !!info.has_model_b;
            const widgetHeight = hasB ? 50 : 26;

            let widget = this.widgets?.find(w => w.name === "_model_info_display");
            if (!widget) {
                widget = {
                    name: "_model_info_display",
                    type: "custom",
                    y: 0,
                    computeSize: () => [0, widgetHeight],
                    draw: function (ctx, node, widgetWidth, y) {
                        const info = node._modelInfo;
                        if (!info) return;
                        const hasB = !!info.has_model_b;
                        if (hasB) {
                            drawModelRow(ctx, info.model_type_a || "Unknown", info.model_name_a || "", y, "A:", widgetWidth);
                            drawModelRow(ctx, info.model_type_b || "Unknown", info.model_name_b || "", y + 24, "B:", widgetWidth);
                        } else {
                            drawModelRow(ctx, info.model_type_a || "Unknown", info.model_name_a || "", y, "", widgetWidth);
                        }
                    }
                };
                if (!this.widgets) this.widgets = [];
                this.widgets.push(widget);
            }
            // Update height for single vs dual model
            widget.computeSize = () => [0, widgetHeight];
            this.computeSize();
        };

        // Reset display on clear/reconnect
        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            origOnConnectionsChange?.apply(this, arguments);
            // Reset info when input connection changes
            if (arguments[1] === 0 && arguments[0] === 1) {
                this._modelInfo = null;
                this.title = "Prompt Model Loader";
                // Restore all outputs
                if (this._originalOutputs) {
                    while (this.outputs.length < this._originalOutputs.length) {
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
