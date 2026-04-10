/**
 * Workflow Generator JS Extension
 *
 * Minimal UI: just shows a status badge when workflow_dict is connected
 * (auto mode) vs manual mode.  All configuration widgets are native.
 *
 * State persistence: handled by native ComfyUI widget serialization.
 * No custom state needed — all widgets are standard.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const C = {
    bgCard:   "rgba(40, 44, 52, 0.7)",
    accent:   "rgba(66, 153, 225, 0.9)",
    success:  "#6c6",
    warning:  "#da3",
    text:     "#ccc",
    textMuted: "#999",
    border:   "rgba(226, 232, 240, 0.15)",
};

function makeEl(tag, style, text) {
    const el = document.createElement(tag);
    if (style) Object.assign(el.style, style);
    if (text !== undefined) el.textContent = text;
    return el;
}

app.registerExtension({
    name: "FBnodes.WorkflowGenerator",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WorkflowGenerator") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated?.apply(this, arguments);
            const node = this;

            // Build a small status badge as a DOM widget at the top of the node
            const badge = makeEl("div", {
                padding: "4px 10px",
                borderRadius: "6px",
                background: C.bgCard,
                border: `1px solid ${C.border}`,
                fontSize: "11px",
                fontFamily: "Inter, system-ui, sans-serif",
                display: "flex",
                alignItems: "center",
                gap: "8px",
            });
            const indicator = makeEl("span", {
                width: "8px", height: "8px", borderRadius: "50%",
                background: C.textMuted, flexShrink: "0", display: "inline-block",
            });
            const modeLabel = makeEl("span", { color: C.textMuted }, "Manual mode");
            badge.appendChild(indicator);
            badge.appendChild(modeLabel);

            const domW = node.addDOMWidget("wg_status", "div", badge, {
                hideOnZoom: false,
                serialize: false,
            });
            domW.computeSize = (w) => [w, 28];

            node._wgStatusDot   = indicator;
            node._wgStatusLabel = modeLabel;
            node._wgDictConnected = false;

            // Poll the connection state on each draw
            const origDraw = node.onDrawForeground;
            node.onDrawForeground = function (ctx) {
                origDraw?.apply(this, arguments);
                // Check if workflow_dict input is connected
                const inputs = node.inputs || [];
                const dictInput = inputs.find(inp => inp.name === "workflow_dict");
                const isConnected = !!(dictInput?.link);
                if (isConnected !== node._wgDictConnected) {
                    node._wgDictConnected = isConnected;
                    if (isConnected) {
                        indicator.style.background = C.success;
                        modeLabel.textContent = "Auto mode — workflow dict connected";
                        modeLabel.style.color = C.success;
                    } else {
                        indicator.style.background = C.textMuted;
                        modeLabel.textContent = "Manual mode";
                        modeLabel.style.color = C.textMuted;
                    }
                }
            };

            return r;
        };

        // ── Lazy-load family-filtered model lists ─────────────────────────
        // This hooks the family widget to filter model_a / model_b dropdowns
        // when the user changes the family selector.
        const origExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            origExecuted?.apply(this, arguments);
            const node = this;
            if (output?.generated?.[0]) {
                const info = output.generated[0];
                if (node._wgStatusLabel) {
                    node._wgStatusLabel.textContent =
                        `✓ Generated — ${info.family || ""} — ${info.model || ""}`;
                    node._wgStatusLabel.style.color = C.success;
                }
                if (node._wgStatusDot) {
                    node._wgStatusDot.style.background = C.success;
                }
            }
        };
    },
});
