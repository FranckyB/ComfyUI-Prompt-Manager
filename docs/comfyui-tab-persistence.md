# ComfyUI Tab Switch & State Persistence

## How ComfyUI handles tab switches

When a user switches browser tabs (or ComfyUI workflow tabs) and returns, ComfyUI **destroys and recreates all nodes**. The sequence is:

1. `onNodeCreated` fires (synchronous — rebuilds all DOM, widgets, state variables from scratch)
2. `onConfigure` fires immediately after (synchronous — receives serialized `info` with `properties` and `widgets_values`)

This means **any JS variable stored on the node object is lost**. Only data serialized into the workflow JSON survives.

## What gets serialized in workflow JSON

Each node in the workflow JSON has:
- `properties` — an object directly on the node, serialized as-is
- `widgets_values` — an array of widget values (index-based, fragile if widget order changes)
- `pos`, `size`, `flags` — layout info

## The correct pattern (node.properties)

### Store durable state in `node.properties`

```js
// In syncHidden (called on every user interaction):
node.properties ||= {};
node.properties.my_state_key = JSON.stringify(stateObject);
```

`node.properties` is **directly serialized** in the workflow JSON `"properties"` field. It survives:
- Tab switches
- Page refreshes  
- Workflow save/load
- Copy/paste

### Restore in onConfigure (synchronously)

```js
nodeType.prototype.onConfigure = function (info) {
    origConfigure?.apply(this, arguments);
    const node = this;
    
    // Set flag SYNCHRONOUSLY — blocks onNodeCreated's async callbacks
    node._configuredFromWorkflow = true;
    
    // Read from node.properties (already restored by LiteGraph)
    const props = node.properties || {};
    const savedState = props.my_state_key;
    
    // Restore UI synchronously — no setTimeout, no API calls
    if (savedState) {
        const state = JSON.parse(savedState);
        restoreUI(node, state);
    }
    
    // Recalculate heights AFTER restoring UI
    if (node._weRecalc) node._weRecalc();
    node.setDirtyCanvas(true, true);
};
```

### Guard onNodeCreated's async work

```js
nodeType.prototype.onNodeCreated = function () {
    // ... build DOM, create widgets ...
    
    // Async init — only for FRESH node creation
    const initImg = getWidgetValue("image");
    if (initImg) {
        setTimeout(() => {
            // _configuredFromWorkflow was set synchronously by onConfigure
            // before this setTimeout fires
            if (!node._configuredFromWorkflow) doInitialLoad(initImg);
        }, 300);
    }
};
```

## Critical rules

1. **NEVER use setTimeout in onConfigure** — it creates race conditions with onNodeCreated's async work
2. **NEVER make API/network calls in onConfigure** — restore purely from serialized data
3. **NEVER reload images/thumbnails in onConfigure** — causes DOM flash and height reset
4. **ALWAYS call recalcHeight after restore** — otherwise the node stays at its empty initial size
5. **ALWAYS persist state on every user interaction** — syncHidden must write to node.properties immediately
6. **ALWAYS persist state after doExtract** — so even before user changes anything, the extracted data is saved
7. **Use a permanent flag** (`_configuredFromWorkflow`) — set it in onConfigure, never clear it. It only needs to block the initial async load in onNodeCreated.

## Why widgets_values is unreliable

- Index-based: if you splice widgets (e.g., adding a Browse button), indices shift
- DOM widgets with `serialize: false` still take a slot
- Widget order depends on Python INPUT_TYPES + JS modifications
- `node.properties` is a named key-value store — no index alignment issues

## What we store in properties

For the Workflow Extractor node:
- `we_override_data` — JSON of all user overrides (model, sampler, prompts, section states, family)
- `we_lora_state` — JSON of LoRA toggle states  
- `we_extracted_cache` — JSON of the full extracted data (avoids API call on restore)

We also write to the hidden widgets (`override_data`, `lora_state`, `extracted_cache`) for Python execution, but `node.properties` is the source of truth for tab persistence.

## Reference: PM Advanced pattern

The Prompt Manager Advanced node uses this exact pattern:
- `_configuredFromWorkflow = true` set synchronously in onConfigure
- State restored from `info.widgets_values` (works for PM Advanced because it doesn't splice widgets)
- DOM widgets reattached synchronously
- Only async work is data loading (not state restoration)
- The flag permanently blocks onNodeCreated's async `loadPromptData` callback

## Do NOT use visibilitychange

The `visibilitychange` event is NOT needed. PM Advanced doesn't use it. ComfyUI's own lifecycle (onNodeCreated → onConfigure) handles everything. Adding `visibilitychange` listeners causes extra work that fights against the restore flow.

## DOM Widget Sizing — Critical Rules

When a node uses `addDOMWidget` with a container holding sections/content, sizing is the #1 source of bugs. Follow these rules:

### The Problem: Flex Shrinking

LiteGraph constrains widget containers to the node's current height. If your root container uses `display: flex; flex-direction: column`, child elements will **shrink to fit** inside the constrained container. Then `scrollHeight` / `offsetHeight` report the *compressed* height, `computeSize` returns that small value to LiteGraph, and the node collapses to nothing in a feedback loop.

### The Fix: `flexShrink: "0"` on all children

Every direct child of the root flex container (sections, cards, etc.) must have `flexShrink: "0"`. This prevents compression, forces overflow, and makes `scrollHeight` report the **true** content height.

```js
// In makeSection():
const wrap = makeEl("div", {
    borderRadius: "6px", overflow: "hidden", marginTop: "2px",
    backgroundColor: C.bgCard, flexShrink: "0",  // ← CRITICAL
});
```

### computeSize must use scrollHeight, not offsetHeight

- `offsetHeight` — reports the *clipped* height (what the parent container allows)
- `scrollHeight` — reports the full *intrinsic* content height (what the content actually needs)

```js
domW.computeSize = function (nodeWidth) {
    const h = root.scrollHeight || 800;
    return [nodeWidth, h];
};
```

### Use requestAnimationFrame, not setTimeout

After DOM mutations (adding LoRA tags, showing/hiding rows), call `_resizeNode` via `requestAnimationFrame` — not `setTimeout`. RAF guarantees the browser has reflowed the DOM, so `scrollHeight` is accurate.

```js
if (node._weRecalc) requestAnimationFrame(() => node._weRecalc());
```

### Reference: PMA's approach

PMA avoids this problem entirely by using **multiple separate DOM widgets** (one per section), each with their own `computeSize`. LiteGraph sums them naturally. WG uses a single DOM widget with a flex container, so the `flexShrink: "0"` + `scrollHeight` approach is required instead.

### _resizeNode pattern

```js
function _resizeNode() {
    const domH = root.scrollHeight || 800;
    const needed = domH + NATIVE_H;
    if (node.size) {
        const w = Math.max(MIN_W, node.size[0]);
        const h = Math.max(needed, MIN_H);
        if (node.size[0] !== w || node.size[1] !== h) {
            node.setSize([w, h]);
        }
    }
    app.graph.setDirtyCanvas(true, true);
}
```

### onResize for enforcing minimums

```js
const origOnResize = node.onResize;
node.onResize = function (size) {
    const domH = root.scrollHeight || 800;
    size[0] = Math.max(MIN_W, size[0]);
    size[1] = Math.max(domH + NATIVE_H, MIN_H, size[1]);
    if (origOnResize) return origOnResize.apply(this, arguments);
};
```
