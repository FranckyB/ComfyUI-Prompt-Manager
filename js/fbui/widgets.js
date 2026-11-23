// Show and hide stock widgets in place. A hidden widget stays in
// node.widgets, so widgets_values keeps its order and saved workflows load
// unchanged; only its rendering collapses.
//
// The classic canvas lays widgets out by computeSize / computeLayoutSize and
// paints them with draw, so those are zeroed. Nodes 2.0 filters its widget
// list on `widget.options.hidden`, so that flag is set too.

const STASH = Symbol.for("fbui.hiddenWidget");
const PAINT_KEYS = ["computeSize", "computeLayoutSize", "draw"];

export function hideWidget(widget) {
  if (!widget) return false;
  if (widget[STASH]) {
    if (widget.options) widget.options.hidden = true;
    return false;
  }
  let options = widget.options ?? {};
  widget[STASH] = {
    own: Object.fromEntries(PAINT_KEYS.map((key) => [key, Object.hasOwn(widget, key)])),
    paint: Object.fromEntries(PAINT_KEYS.map((key) => [key, widget[key]])),
    hidden: widget.hidden,
    optionsHidden: options.hidden,
  };
  widget.hidden = true;
  options.hidden = true;
  // The frontend swaps a widget's options object after creation (a workflow
  // restore does), which would drop the flag; keep re-applying it.
  try {
    Object.defineProperty(widget, "options", {
      configurable: true,
      enumerable: true,
      get: () => options,
      set: (next) => {
        options = next ?? {};
        options.hidden = true;
      },
    });
  } catch {
    widget.options = options;
  }
  widget.computeSize = () => [0, -4];
  widget.computeLayoutSize = () => ({ minWidth: 0, minHeight: 0 });
  widget.draw = () => {};
  if (widget.element) widget.element.style.display = "none";
  return true;
}

export function showWidget(widget) {
  const stash = widget?.[STASH];
  if (!stash) return false;
  for (const key of PAINT_KEYS) {
    if (stash.own[key]) widget[key] = stash.paint[key];
    else delete widget[key];
  }
  widget.hidden = stash.hidden;
  const options = widget.options;
  delete widget.options;
  widget.options = options;
  if (stash.optionsHidden === undefined) delete options.hidden;
  else options.hidden = stash.optionsHidden;
  if (widget.element) widget.element.style.display = "";
  widget[STASH] = undefined;
  return true;
}

export function isWidgetHidden(widget) {
  return Boolean(widget?.[STASH]);
}

// Mark inputs hidden in the node definition, before any widget exists. The
// frontend rebuilds widget options from this spec on restore and on the
// Nodes 2.0 mount, so a flag on the live widget alone is not enough.
export function hideInputsInDef(nodeData, names) {
  for (const name of names) {
    for (const group of ["required", "optional"]) {
      const spec = nodeData?.input?.[group]?.[name];
      if (!Array.isArray(spec)) continue;
      spec[1] = { ...(spec[1] ?? {}), hidden: true };
    }
  }
}

export function findWidget(node, name) {
  return node?.widgets?.find((widget) => widget.name === name) ?? null;
}

export function comboValues(widget) {
  let values = widget?.options?.values;
  if (typeof values === "function") values = values(widget);
  return Array.isArray(values) ? values : [];
}

// Write a value the way LiteGraph's own widgets do: set it, run the widget's
// callback, then tell the node.
export function commitWidgetValue(node, widget, value) {
  const previous = widget.value;
  widget.value = value;
  widget.callback?.(value, globalThis.app?.canvas, node);
  node.onWidgetChanged?.(widget.name, value, previous, widget);
}

export function widgetSlot(node, name) {
  return node?.inputs?.find((input) => input?.widget?.name === name) ?? null;
}

export function widgetLinked(node, name) {
  const slot = widgetSlot(node, name);
  return slot?.link !== null && slot?.link !== undefined;
}

// A widget's hover text: its own tooltip, else the node definition's.
export function widgetTooltip(node, name) {
  const widget = findWidget(node, name);
  const own = widget?.options?.tooltip ?? widget?.tooltip;
  if (typeof own === "string" && own) return own;
  const inputs = node?.constructor?.nodeData?.input;
  const spec = inputs?.required?.[name] ?? inputs?.optional?.[name];
  return typeof spec?.[1]?.tooltip === "string" ? spec[1].tooltip : "";
}

// For a node that draws its own preview: keep the frontend's run-output
// picture off it in both renderers. Nodes 2.0 reads hideOutputImages; the
// classic canvas draws node.imgs through a "$$canvas-image-preview" widget.
const CORE_PREVIEW = "$$canvas-image-preview";
export function suppressCorePreview(node) {
  node.hideOutputImages = true;
  const index = node.widgets?.findIndex((widget) => widget.name === CORE_PREVIEW) ?? -1;
  if (index >= 0) {
    node.widgets[index].onRemove?.();
    node.widgets.splice(index, 1);
  }
  if (Object.getOwnPropertyDescriptor(node, "imgs")?.get) return;
  const add = node.addCustomWidget;
  if (typeof add === "function") {
    node.addCustomWidget = function (widget) {
      if (widget?.name === CORE_PREVIEW) hideWidget(widget);
      return add.call(this, widget);
    };
  }
  Object.defineProperty(node, "imgs", { configurable: true, enumerable: true, get: () => undefined, set: () => {} });
}
