/**
 * Workflow Renderer — Set initial node size to match Workflow Builder width.
 */
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "prompt-manager.workflow-renderer",
    async nodeCreated(node) {
        if (node.comfyClass !== "WorkflowRenderer") return;
        requestAnimationFrame(() => {
            node.setSize([320, node.size[1]]);
            app.graph.setDirtyCanvas(true, true);
        });
    },
});
