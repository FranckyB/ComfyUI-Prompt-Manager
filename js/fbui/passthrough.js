// Graph zoom and pan over our panels, and files dropped on them.
//
// A panel that takes the mouse also swallows the wheel, so the graph stopped
// zooming the moment the pointer crossed a card or preview. One listener hands
// those gestures back to the canvas, unless the element under the pointer
// scrolls that way. In the classic renderer a file dropped on a panel
// missed its node; the drop listeners hand it to the node under the pointer.
//
// Every pack carrying the kit calls installPassthrough(); the first one wins,
// and it serves the `fbui-` panels of all of them.
import { app } from "../../../scripts/app.js";

const INSTALLED = Symbol.for("fbui.passthrough");
const WRAPPERS = ".dom-widget, .lg-node-widgets";
const isOurs = (element) => typeof element?.className === "string" && element.className.includes("fbui-");

function panelRoot(target) {
  const host = target?.closest?.(WRAPPERS);
  if (!host) return null;
  let root = null;
  for (let element = target; element && element !== host; element = element.parentElement) if (isOurs(element)) root = element;
  if (!root && target === host && host.matches(".dom-widget")) root = Array.from(host.children).find(isOurs) ?? null;
  return root;
}

// An element with more content than room keeps the wheel, even at its
// ends, as a native scroll box: the graph only zooms where nothing scrolls.
function canScroll(element, delta, horizontal) {
  if (!delta) return false;
  const size = horizontal ? element.scrollWidth : element.scrollHeight;
  const visible = horizontal ? element.clientWidth : element.clientHeight;
  return size - visible > 1;
}

function panelKeepsWheel(target, event, root) {
  if (event.ctrlKey || event.metaKey) return false;
  for (let element = target; element && element !== root.parentElement; element = element.parentElement) {
    const style = getComputedStyle(element);
    if ((element.tagName === "TEXTAREA" || /(auto|scroll)/.test(style.overflowY)) && canScroll(element, event.deltaY, false)) return true;
    if (/(auto|scroll)/.test(style.overflowX) && canScroll(element, event.deltaX, true)) return true;
  }
  return false;
}

let forwardedPointer = null;

// Nodes 2.0 hands the wheel over a node to the canvas unless a focused
// element takes it: a panel that scrolls keeps it, focused or not.
function keepScroll(event) {
  const root = panelRoot(event.target);
  if (root && panelKeepsWheel(event.target, event, root)) event.stopPropagation();
}

function onWheel(event) {
  const root = panelRoot(event.target);
  if (!root || !app.canvas || event.defaultPrevented || panelKeepsWheel(event.target, event, root)) return;
  event.preventDefault();
  app.canvas.processMouseWheel(event);
}

// Middle-button pan starts over a panel too.
function onPointerDown(event) {
  if (!app.canvas || event.button !== 1 || !panelRoot(event.target)) return;
  forwardedPointer = event.pointerId;
  event.preventDefault();
  event.stopPropagation();
  app.canvas.processMouseDown(event);
}

function onPointerMove(event) {
  if (forwardedPointer === event.pointerId) app.canvas?.processMouseMove(event);
}

function onPointerEnd(event) {
  if (forwardedPointer !== event.pointerId) return;
  forwardedPointer = null;
  app.canvas?.processMouseUp(event);
}

function dropTaker(event) {
  const types = Array.from(event.dataTransfer?.types ?? []);
  if (!types.includes("Files") || !event.target?.closest?.(".dom-widget") || !panelRoot(event.target)) return null;
  const canvas = app.canvas;
  if (typeof canvas?.adjustMouseEvent !== "function") return null;
  canvas.adjustMouseEvent(event);
  const node = canvas.graph?.getNodeOnPos?.(event.canvasX, event.canvasY);
  return typeof node?.onDragDrop === "function" && node.onDragOver?.(event) ? node : null;
}

function onDragOver(event) {
  if (!dropTaker(event)) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = "copy";
}

async function onDrop(event) {
  const node = dropTaker(event);
  if (!node) return;
  // The node takes the file; ComfyUI's own handler would add a new node.
  event.preventDefault();
  event.stopPropagation();
  await node.onDragDrop(event);
}

export function installPassthrough() {
  if (globalThis[INSTALLED]) return;
  globalThis[INSTALLED] = true;
  window.addEventListener("wheel", keepScroll, { capture: true, passive: true });
  window.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("pointerdown", onPointerDown, true);
  window.addEventListener("pointermove", onPointerMove, true);
  window.addEventListener("pointerup", onPointerEnd, true);
  window.addEventListener("pointercancel", onPointerEnd, true);
  window.addEventListener("dragover", onDragOver, true);
  window.addEventListener("drop", onDrop, true);
}
