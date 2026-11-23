// Helpers that need the ComfyUI app.
import { app } from "../../../scripts/app.js";

export function isVueNodes() {
  return globalThis.LiteGraph?.vueNodesMode === true;
}

// One toast for every message; never alert(). Falls back to the console and
// never throws, so a message cannot break the action that raised it.
export function toast(detail, { severity = "info", summary = "", life = 5000 } = {}) {
  try {
    const store = app?.extensionManager?.toast;
    if (typeof store?.add === "function") {
      store.add({ severity, summary, detail, life });
      return;
    }
  } catch {
    // Fall through to the console.
  }
  (severity === "error" || severity === "warn" ? console.warn : console.log)(`${summary ? `${summary}: ` : ""}${detail}`);
}

// The undo/dirty tracker snapshots on canvas mouse-up, but our DOM panels
// commit values a phase later. Call once per user gesture, after the write,
// and never from a workflow-load path.
let changeTimer = null;
export function notifyChange() {
  clearTimeout(changeTimer);
  changeTimer = setTimeout(() => {
    changeTimer = null;
    try {
      const workflow = app?.extensionManager?.workflow?.activeWorkflow;
      const tracker = workflow?.changeTracker;
      if (typeof tracker?.captureCanvasState === "function") tracker.captureCanvasState();
      else tracker?.checkState?.();
    } catch {
      // The write itself already succeeded.
    }
  }, 120);
}
