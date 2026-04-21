/**
 * Recipe Renderer — Set initial node size to match Recipe Builder width.
 */
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "prompt-manager.recipe-renderer",
    async nodeCreated(node) {
        if (node.comfyClass !== "RecipeRenderer") return;
        requestAnimationFrame(() => {
            node.setSize([320, node.size[1]]);
            app.graph.setDirtyCanvas(true, true);
        });
    },
});
