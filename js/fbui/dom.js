// Plain DOM and LiteGraph-object helpers. No imports, so node:test can load
// everything that builds on them.

export function el(tag, className = "", text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Wrap a callback without clobbering whoever hooked it first. Other packs
// patch the same objects, so never assign `obj[name] = fn` directly.
export function chainCallback(obj, name, fn) {
  const prior = obj[name];
  obj[name] = function (...args) {
    const result = prior?.apply(this, args);
    fn.apply(this, args);
    return result;
  };
}

// chainCallback for handlers whose truthy return means "consumed".
export function chainHandler(obj, name, fn) {
  const prior = obj[name];
  obj[name] = function (...args) {
    const result = prior?.apply(this, args);
    const handled = fn.apply(this, args);
    return result || handled;
  };
}

// The frontend sizes a DOM widget's wrapper as `widget.width ?? node.width`,
// and LiteGraph plants widget.width during draws, after which the wrapper no
// longer follows the node when it is resized narrower. Discard those writes.
export function keepDomWidgetWidthAuto(widget) {
  if (!widget) return widget;
  try {
    Object.defineProperty(widget, "width", { get: () => undefined, set: () => {}, configurable: true });
  } catch {
    // A frozen widget keeps the frontend's behaviour.
  }
  return widget;
}

// App shortcuts (Ctrl+Enter, Ctrl+S) go on to ComfyUI; every other key typed
// in one of our fields stays in the field so the canvas never acts on it.
const FIELD_EDIT_KEYS = new Set(["a", "c", "v", "x", "z", "y", "backspace", "delete", "arrowleft", "arrowright", "arrowup", "arrowdown", "home", "end"]);

export function keepKeyInField(event) {
  const appShortcut = (event.ctrlKey || event.metaKey) && !FIELD_EDIT_KEYS.has(String(event.key ?? "").toLowerCase());
  if (!appShortcut) event.stopPropagation();
}
