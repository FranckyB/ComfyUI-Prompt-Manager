/**
 * SwitchAny Extension for ComfyUI
 * - Parses the "names" widget (comma/semicolon separated) to populate the "select" dropdown
 * - Shows/hides input slots based on "num_inputs" slider
 * - Strips non-selected inputs from the execution payload so upstream nodes are never evaluated
 */

import { app } from "../../scripts/app.js";

const MAX_INPUTS = 10;

/**
 * Parse a names string the same way Python does.
 */
function parseNames(namesStr, count) {
    const parts = namesStr.split(/[;,]/).map(s => s.trim()).filter(Boolean);
    const result = [];
    for (let i = 0; i < count; i++) {
        result.push(i < parts.length ? parts[i] : `Input ${i + 1}`);
    }
    return result;
}

app.registerExtension({
    name: "SwitchAny",

    setup() {
        // Strip non-selected inputs from execution payload so ComfyUI never
        // validates or executes upstream nodes that aren't the active selection.
        const originalGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function (...args) {
            const result = await originalGraphToPrompt.apply(this, args);
            if (result?.output) {
                for (const nodeData of Object.values(result.output)) {
                    if (nodeData.class_type !== "SwitchAny") continue;

                    const selected = nodeData.inputs.select;
                    const numInputs = nodeData.inputs.num_inputs ?? 2;
                    const namesStr = nodeData.inputs.names ?? "";
                    const names = parseNames(namesStr, numInputs);

                    let selectedIndex = -1;
                    for (let i = 0; i < names.length; i++) {
                        if (names[i] === selected) {
                            selectedIndex = i + 1;
                            break;
                        }
                    }
                    // Remove every input_* except the selected one
                    for (let i = 1; i <= MAX_INPUTS; i++) {
                        if (i !== selectedIndex) {
                            delete nodeData.inputs[`input_${i}`];
                        }
                    }
                }
            }
            return result;
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SwitchAny") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            node.setSize([260, node.size[1]]);

            const selectWidget = node.widgets.find(w => w.name === "select");
            const numWidget = node.widgets.find(w => w.name === "num_inputs");
            const namesWidget = node.widgets.find(w => w.name === "names");

            node._refreshSwitchAny = refreshNode;

            function refreshNode() {
                if (!selectWidget || !numWidget || !namesWidget) return;
                const count = numWidget.value ?? 2;
                const names = parseNames(namesWidget.value || "", count);

                // Update dropdown choices
                const prev = selectWidget.value;
                selectWidget.options.values = names;
                selectWidget.value = names.includes(prev) ? prev : names[0];

                // Show/hide input slots
                if (node.inputs) {
                    for (let i = 0; i < MAX_INPUTS; i++) {
                        const slot = node.inputs.find(inp => inp.name === `input_${i + 1}`);
                        if (slot) {
                            // Rename the visible slot to the custom name
                            slot.label = i < count ? names[i] : `input_${i + 1}`;
                            // Hide slots beyond num_inputs
                            if (i >= count) {
                                slot.hidden = true;
                            } else {
                                slot.hidden = false;
                            }
                        }
                    }
                }

                node.setDirtyCanvas(true, true);
            }

            // Hook callbacks
            const origNum = numWidget?.callback;
            if (numWidget) {
                numWidget.callback = function (value) {
                    if (origNum) origNum.apply(this, arguments);
                    refreshNode();
                };
            }

            const origNames = namesWidget?.callback;
            if (namesWidget) {
                namesWidget.callback = function (value) {
                    if (origNames) origNames.apply(this, arguments);
                    refreshNode();
                };
            }

            // Initial sync
            setTimeout(refreshNode, 50);

            return result;
        };

        // Refresh names every time the node is executed
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (data) {
            const result = onExecuted?.apply(this, arguments);
            if (this._refreshSwitchAny) this._refreshSwitchAny();
            return result;
        };

        // Restore state on workflow load / page reload
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = onConfigure?.apply(this, arguments);
            const node = this;

            setTimeout(() => {
                const selectWidget = node.widgets?.find(w => w.name === "select");
                const numWidget = node.widgets?.find(w => w.name === "num_inputs");
                const namesWidget = node.widgets?.find(w => w.name === "names");
                if (!selectWidget || !numWidget || !namesWidget) return;

                const count = numWidget.value ?? 2;
                const names = parseNames(namesWidget.value || "", count);

                selectWidget.options.values = names;
                // Restore saved selection
                if (info.widgets_values) {
                    const idx = node.widgets.indexOf(selectWidget);
                    const saved = info.widgets_values[idx];
                    if (saved && names.includes(saved)) {
                        selectWidget.value = saved;
                    }
                }

                // Show/hide + rename input slots
                if (node.inputs) {
                    for (let i = 0; i < MAX_INPUTS; i++) {
                        const slot = node.inputs.find(inp => inp.name === `input_${i + 1}`);
                        if (slot) {
                            slot.label = i < count ? names[i] : `input_${i + 1}`;
                            slot.hidden = i >= count;
                        }
                    }
                }

                node.setDirtyCanvas(true, true);
            }, 50);

            return result;
        };
    }
});
