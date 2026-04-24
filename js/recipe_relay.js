/**
 * Recipe Relay — set a compact default width for newly created nodes.
 */
import { app } from "../../scripts/app.js";

const RELAY_DEFAULT_WIDTH = 220;

app.registerExtension({
    name: "prompt-manager.recipe-relay",
    async nodeCreated(node) {
        if (node.comfyClass !== "RecipeRelay") return;

        requestAnimationFrame(() => {
            node.setSize([RELAY_DEFAULT_WIDTH, node.size[1]]);
            app.graph.setDirtyCanvas(true, true);
        });
    },
});
