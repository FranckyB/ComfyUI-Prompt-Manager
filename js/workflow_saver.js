import { app } from "../../scripts/app.js";
import { openPromptBrowserForSave } from "./prompt_manager_advanced.js";

function notify(message) {
    try {
        app.ui?.dialog?.show?.(message);
    } catch (_) {
        console.log("[WorkflowSaver]", message);
    }
}

async function openSaveDialog(node) {
    const nodeId = String(node.id);

    const snapshotResp = await fetch(`/workflow-saver/snapshot?node_id=${encodeURIComponent(nodeId)}`);
    const snapshotData = await snapshotResp.json();
    if (!snapshotData?.success) {
        notify(snapshotData?.error || "No workflow data available. Queue once, then save.");
        return;
    }

    await openPromptBrowserForSave({
        node,
        currentCategory: "Default",
        title: "Save Workflow",
        saveButtonText: "Save",
        namePlaceholder: "Prompt name",
        onSave: async ({ category, name, overwrite }) => {
            try {
                const response = await fetch("/workflow-saver/save", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        node_id: nodeId,
                        category,
                        name,
                        overwrite: overwrite === true,
                    }),
                });
                const data = await response.json();
                return {
                    success: data?.success === true,
                    error: data?.error,
                    category: data?.category || category,
                    name: data?.name || name,
                    overwritten: data?.overwritten === true,
                };
            } catch (error) {
                return {
                    success: false,
                    error: error?.message || "Failed to save workflow.",
                    category,
                    name,
                    overwritten: overwrite === true,
                };
            }
        },
    });
}

app.registerExtension({
    name: "WorkflowSaver",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WorkflowSaver") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);
            const node = this;

            node.addWidget("button", "Save Workflow", "", async () => {
                try {
                    await openSaveDialog(node);
                } catch (e) {
                    notify(`Workflow Saver error: ${e?.message || e}`);
                }
            });

            return r;
        };
    },
});
