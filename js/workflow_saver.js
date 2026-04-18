import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { openPromptBrowserForSave } from "./prompt_manager_advanced.js";

function notify(message) {
    try {
        app.ui?.dialog?.show?.(message);
    } catch (_) {
        console.log("[WorkflowSaver]", message);
    }
}

function setPreviewThumbnail(node, thumbnailDataUrl) {
    const ui = node?._workflowSaverPreview;
    if (!ui) return;

    if (typeof thumbnailDataUrl === "string" && thumbnailDataUrl.trim()) {
        ui.image.src = thumbnailDataUrl;
        ui.image.style.display = "block";
        ui.emptyLabel.style.display = "none";
    } else {
        ui.image.removeAttribute("src");
        ui.image.style.display = "none";
        ui.emptyLabel.style.display = "block";
        ui.emptyLabel.textContent = "No image preview in latest snapshot";
    }

    node.setDirtyCanvas?.(true, true);
}

async function fetchWorkflowSaverSnapshot(nodeId) {
    const snapshotResp = await fetch(`/workflow-saver/snapshot?node_id=${encodeURIComponent(nodeId)}`);
    let snapshotData = null;
    try {
        snapshotData = await snapshotResp.json();
    } catch {
        snapshotData = null;
    }

    return {
        ok: snapshotResp.ok,
        data: snapshotData,
    };
}

async function refreshNodePreviewFromSnapshot(node, { silent = true } = {}) {
    const nodeId = String(node.id);
    try {
        const snapshot = await fetchWorkflowSaverSnapshot(nodeId);
        if (!snapshot?.data?.success) {
            setPreviewThumbnail(node, null);
            if (!silent) {
                notify(snapshot?.data?.error || "No workflow data available. Queue once, then save.");
            }
            return null;
        }

        const thumb = snapshot.data?.snapshot?.thumbnail || null;
        setPreviewThumbnail(node, thumb);
        return snapshot.data;
    } catch (e) {
        setPreviewThumbnail(node, null);
        if (!silent) {
            notify(`Workflow Saver error: ${e?.message || e}`);
        }
        return null;
    }
}

function addPreviewWidget(node) {
    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 6px;
        background: #1e1e1e;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        padding: 8px;
        box-sizing: border-box;
        overflow: hidden;
    `;

    const title = document.createElement("div");
    title.textContent = "Incoming Image Preview";
    title.style.cssText = `
        font-size: 11px;
        color: #9aa;
        font-weight: 600;
        letter-spacing: 0.2px;
    `;

    const previewBox = document.createElement("div");
    previewBox.style.cssText = `
        position: relative;
        width: 100%;
        aspect-ratio: 1 / 1;
        border-radius: 6px;
        background: #111;
        border: 1px solid #333;
        overflow: hidden;
        box-sizing: border-box;
    `;

    const image = document.createElement("img");
    image.style.cssText = `
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
        object-position: center center;
        display: none;
    `;

    const emptyLabel = document.createElement("div");
    emptyLabel.textContent = "No image preview in latest snapshot";
    emptyLabel.style.cssText = `
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-size: 11px;
        color: #777;
        background: #141414;
        border: 1px dashed rgba(110, 110, 110, 0.5);
        border-radius: 5px;
        padding: 8px;
        box-sizing: border-box;
    `;

    container.appendChild(title);
    previewBox.appendChild(image);
    previewBox.appendChild(emptyLabel);
    container.appendChild(previewBox);

    const widget = node.addDOMWidget("workflow_saver_preview", "div", container, {
        hideOnZoom: false,
    });
    widget.computeSize = (width) => [Math.max(width || 260, 260), 252];

    node._workflowSaverPreview = {
        container,
        image,
        emptyLabel,
        widget,
    };
}

async function openSaveDialog(node) {
    const nodeId = String(node.id);

    const snapshotData = await refreshNodePreviewFromSnapshot(node, { silent: true });
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

            addPreviewWidget(node);

            node.addWidget("button", "Save Workflow", "", async () => {
                try {
                    await openSaveDialog(node);
                } catch (e) {
                    notify(`Workflow Saver error: ${e?.message || e}`);
                }
            });

            // Refresh preview when this node executes and updates its snapshot cache.
            const onExecuted = (e) => {
                if (String(e?.detail?.node || "") !== String(node.id)) return;
                refreshNodePreviewFromSnapshot(node, { silent: true });
            };
            api.addEventListener("executed", onExecuted);
            node._workflowSaverOnExecuted = onExecuted;

            // Try an initial refresh in case this node already has a cached snapshot.
            refreshNodePreviewFromSnapshot(node, { silent: true });

            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                try {
                    if (node._workflowSaverOnExecuted) {
                        api.removeEventListener("executed", node._workflowSaverOnExecuted);
                        node._workflowSaverOnExecuted = null;
                    }
                } catch (_) {}
                return origOnRemoved?.apply(this, arguments);
            };

            return r;
        };
    },
});
