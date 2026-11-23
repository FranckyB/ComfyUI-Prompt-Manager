// How a DOM panel claims its share of a node's height, in both renderers.
// No imports, so node:test can load it.
//
// The classic frontend arranges widgets in one pass:
//   if (w.computeSize)            -> fixed height, kept out of the split
//   else if (w.computeLayoutSize) -> shares the node's free height
// so a growing panel must not carry a computeSize, and a fixed one should.
//
// DomWidgets insets an element by options.margin per side (default 10), so
// the element gets this much less height than the layout hands the widget.
export const WIDGET_FRAME = 20;

const resolve = (value) => {
  const number = Number(typeof value === "function" ? value() : value);
  return Number.isFinite(number) ? Math.max(0, number) : 0;
};

// The frontend pads a DOM widget's minWidth before it becomes the node's
// floor (the room a number widget's value box takes). Measure that padding
// once from the frontend itself instead of copying its constants.
const PROBE_WIDTH = 100000;
let probing = false;
let widthPadding = null;

function measureWidthPadding(node, widget) {
  if (probing || typeof node?.computeSize !== "function") return null;
  const own = widget.computeLayoutSize;
  probing = true;
  widget.computeLayoutSize = () => ({ minWidth: PROBE_WIDTH, minHeight: 0 });
  let width = NaN;
  try {
    width = Number(node.computeSize()?.[0]);
  } catch {
    // Measured next time.
  } finally {
    widget.computeLayoutSize = own;
    probing = false;
  }
  return Number.isFinite(width) && width >= PROBE_WIDTH ? width - PROBE_WIDTH : null;
}

// A node can carry several panels; its width floor is the widest of them.
const FLOORS = Symbol.for("fbui.widthFloors");

function nodeWidthFloor(node) {
  let floor = 0;
  for (const value of node?.[FLOORS]?.values() ?? []) floor = Math.max(floor, resolve(value));
  return floor;
}

// Nodes 2.0 lays a node out by its content and has no layout API for
// minimums: lend it the floors as inline CSS on the node element and panel.
function holdVueMinWidth(panel, width) {
  const host = panel?.closest?.("[data-node-id]");
  if (!host?.style || !(width > 0)) return;
  const css = `${Math.round(width)}px`;
  if (host.style.minWidth !== css) host.style.minWidth = css;
}

function holdVueMinHeight(panel, height) {
  if (!panel?.style) return;
  const inside = Boolean(panel.closest?.("[data-node-id]"));
  const css = inside && height > 0 ? `${Math.round(height)}px` : "";
  if ((panel.style.minHeight || "") !== css) panel.style.minHeight = css;
}

// One ResizeObserver per panel; the Vue node element only exists around a
// panel once it has been laid out inside it.
const WATCHED = Symbol.for("fbui.layoutWatch");
function whenLaidOut(panel, name, task) {
  if (!panel || typeof ResizeObserver !== "function") return;
  if (!panel[WATCHED]) {
    const tasks = new Map();
    panel[WATCHED] = tasks;
    new ResizeObserver(() => { for (const run of tasks.values()) run(); }).observe(panel);
  }
  panel[WATCHED].set(name, task);
}

// Make `minWidth` a real floor for the node in both renderers: the classic
// corner drag stops at node.computeSize(), Nodes 2.0 at the host's min-width.
export function holdNodeMinWidth(widget, minWidth) {
  const node = widget?.node;
  if (!node) return widget;
  if (!node[FLOORS]) {
    node[FLOORS] = new Map();
    const inherited = node.computeSize;
    node.computeSize = function (out) {
      const size = inherited.call(this, out);
      const floor = nodeWidthFloor(this);
      if (size && size[0] < floor) size[0] = floor;
      return size;
    };
  }
  node[FLOORS].set(widget, minWidth);
  whenLaidOut(widget.element, "width", () => holdVueMinWidth(widget.element, nodeWidthFloor(node)));
  return widget;
}

// A panel that takes whatever height the node has left over (a preview).
export function fillNodeHeight(widget, { minWidth = 0, minHeight = 0 } = {}) {
  if (!widget) return widget;
  const panel = widget.element;
  const holdHeight = (height) => holdVueMinHeight(panel, height - WIDGET_FRAME);
  delete widget.computeSize;
  widget.computeLayoutSize = (node) => {
    const height = resolve(minHeight);
    holdHeight(height);
    if (widthPadding === null) widthPadding = measureWidthPadding(node ?? widget.node, widget);
    return { minWidth: Math.max(0, resolve(minWidth) - (widthPadding ?? 0)), minHeight: height };
  };
  holdNodeMinWidth(widget, minWidth);
  whenLaidOut(panel, "height", () => holdHeight(resolve(minHeight)));
  return widget;
}

// A panel that keeps its own height (a card whose height is its rows).
// Nodes 2.0 gives any widget with a computeLayoutSize a share of the spare
// height, and every DOM widget inherits one, so shadow it.
export function pinHeight(widget, height, minWidth = 0) {
  if (!widget) return widget;
  widget.computeSize = (width) => [Math.max(minWidth, Number(width) || 0), resolve(height)];
  widget.computeLayoutSize = undefined;
  if (minWidth) holdNodeMinWidth(widget, minWidth);
  return widget;
}

// Grow a node to the height its widgets ask for: a workflow saved before a
// panel existed carries the node's old, smaller size.
export function ensureNodeMinHeight(node) {
  const min = Number(node?.computeSize?.()?.[1]);
  const current = Number(node?.size?.[1]);
  if (!Number.isFinite(min) || !Number.isFinite(current) || current >= min) return false;
  node.setSize?.([node.size[0], min]);
  return true;
}
