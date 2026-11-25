// The card: one DOM panel standing in for a node's stock widgets. Every row
// mirrors a hidden widget (or a pair of them), so the widgets stay the source
// of truth: save/load, undo, API format and links all keep working through
// them. The card writes widget.value, runs the widget's callback, and re-reads
// everything on configure, on connection changes and after any callback.
//
// Rows:
//   { widget, label?, kind?, suffix?, labels?, title?, info?, when?, group?, button? }
//       info: help lines shown from an (i) beside the label (see controls.info)
//       kind: scrub | segment | select | switch | text (default: from the widget)
//       labels: { value: text } or (value, values, node) => text
//       button: { text, icon?, title?, onClick(node, card) } (or a list) after the control
//       groups: (value, values, node) => heading, for a select's <optgroup>s
//   { pair: [a, b], label, sep?, when?, group? }
//   { section: "Caption", when? }
//   { label, control: (node, card) => ({ element, label?, sync?(values) }), widgets?, when?, group? }
//       a labelled row holding the node's own control; `widgets` names the
//       hidden widgets it drives; a returned `label` element replaces the text
//   { panel: (node, card) => ({ element, sync?(values) }), height, grow?, when? }
//       grow takes the node's spare height; a height function is read again
//       on every refresh (a panel whose content wraps)
//   { group: "more", label: "More", tools? }   later rows with group: "more" fold under it
//       tools: (node, card) => ({ element, sync?(values) }) at the header's right
// Options: minWidth, first (place before the node's other DOM widgets).
import { chainCallback, el, keepDomWidgetWidthAuto, keepKeyInField } from "./dom.js";
import { notifyChange } from "./comfy.js";
import { button as makeButton, info as infoIcon, segmented, toggleSwitch } from "./controls.js";
import { cardHeight, choiceLabel, isGroupHeader, rowHeight, rowKind, rowTops, socketWidgetY, visibleRows } from "./card_math.js";
import { ensureNodeMinHeight, fillNodeHeight, pinHeight, WIDGET_FRAME } from "./layout.js";
import { makeScrub, scrubSteps } from "./scrub.js";
import { ensureKitCss, FONT, GROUP_HEIGHT, ROW_HEIGHT, UNIT_SLOT_WIDTH } from "./theme.js";
import { comboValues, commitWidgetValue, findWidget, hideWidget, widgetLinked, widgetSlot, widgetTooltip } from "./widgets.js";

export const CARD_WIDGET = "fbui_card";
const WRAPPER_TOP = 10;

function isFloatWidget(widget) {
  const precision = widget?.options?.precision;
  if (precision !== undefined) return Number(precision) > 0;
  const step = widget?.options?.step2 ?? widget?.options?.round;
  return step !== undefined ? !Number.isInteger(Number(step)) : !Number.isInteger(Number(widget?.value));
}

export function mountCard(node, { rows, minWidth = 280, first = false } = {}) {
  ensureKitCss();
  const root = el("div", "fbui-card");
  const card = { node, root, entries: [], height: 0, widget: null, rowTop: new Map(), rowSpan: new Map(), disposed: false };
  const managed = new Set();

  const values = () => Object.fromEntries((node.widgets ?? []).map((widget) => [widget.name, widget.value]));
  const groupKey = (name) => `fbui_open_${name}`;
  const groupOpen = (name) => Boolean(node.properties?.[groupKey(name)]);

  const setWidget = (name, value, { settle = true } = {}) => {
    const widget = findWidget(node, name);
    if (!widget || widget.value === value) return;
    card.byHand = true;
    try {
      commitWidgetValue(node, widget, value);
    } finally {
      card.byHand = false;
    }
    node.graph?.setDirtyCanvas?.(true, true);
    if (settle) notifyChange();
  };
  card.setWidget = setWidget;

  const builders = {
    scrub(name, row, into) {
      const widget = findWidget(node, name);
      const isFloat = isFloatWidget(widget);
      const steps = scrubSteps(widget?.options, isFloat, row.suffix === "px");
      const control = makeScrub({
        value: Number(widget?.value) || 0,
        min: Number(widget?.options?.min ?? -Infinity),
        max: Number(widget?.options?.max ?? Infinity),
        step: row.step ?? steps.step,
        fineStep: steps.fineStep,
        decimals: row.decimals ?? steps.decimals,
        unit: row.suffix ?? "",
        unitWidth: row.pair ? 0 : UNIT_SLOT_WIDTH,
        title: row.title ?? widgetTooltip(node, name),
        onChange: (next) => setWidget(name, isFloat ? next : Math.round(next), { settle: false }),
        onSettle: notifyChange,
      });
      into.append(control.root);
      return (vals) => control.set(Number(vals[name]) || 0);
    },
    segment(name, row, into) {
      const widget = findWidget(node, name);
      // The choices can change (one widget's list following another's value):
      // the pill is rebuilt whenever they do.
      let shown = null;
      let seg = null;
      return (vals) => {
        const options = comboValues(widget);
        const key = options.join("\n");
        if (key !== shown) {
          shown = key;
          const next = segmented(options.map((value) => ({ value, label: choiceLabel(row, value, options, node), title: row.titles?.[value] })), { onPick: (value) => setWidget(name, value) });
          if (seg) seg.root.replaceWith(next.root);
          else into.append(next.root);
          seg = next;
        }
        seg.set(vals[name]);
      };
    },
    switch(name, row, into) {
      const words = row.offText || row.onText ? [row.offText ?? "off", row.onText ?? "on"] : null;
      const control = toggleSwitch({ words, title: row.title ?? widgetTooltip(node, name), onPick: (value) => setWidget(name, value) });
      into.append(control.root);
      return (vals) => control.set(Boolean(vals[name]));
    },
    select(name, row, into) {
      const widget = findWidget(node, name);
      const select = el("select", "fbui-select");
      select.title = row.title ?? widgetTooltip(node, name);
      let shown = null;
      const fill = (current) => {
        const options = comboValues(widget);
        const list = options.includes(current) || current === undefined || current === "" ? options : [...options, current];
        const key = list.join("\n");
        if (key !== shown) {
          shown = key;
          select.textContent = "";
          // row.groups(value) names a value's group: runs of one group sit
          // under an <optgroup> heading.
          let group = null;
          for (const value of list) {
            const option = el("option", "", choiceLabel(row, value, options, node));
            option.value = String(value);
            const name = row.groups?.(value, options, node) ?? null;
            if (name && name !== group?.label) {
              group = el("optgroup");
              group.label = name;
              select.append(group);
            } else if (!name) group = null;
            (group ?? select).append(option);
          }
        }
        select.value = String(current ?? "");
        select.title = String(current ?? "") || widgetTooltip(node, name);
      };
      select.addEventListener("change", () => setWidget(name, select.value));
      // The list may change between refreshes (an upload, a folder change).
      select.addEventListener("pointerdown", () => fill(findWidget(node, name)?.value));
      into.append(select);
      return (vals) => fill(vals[name]);
    },
    text(name, row, into) {
      const input = el("input", "fbui-text");
      input.type = "text";
      input.spellcheck = false;
      input.placeholder = row.placeholder ?? "";
      input.title = row.title ?? widgetTooltip(node, name);
      const commit = () => {
        if (input.value !== String(findWidget(node, name)?.value ?? "")) setWidget(name, input.value);
      };
      input.addEventListener("change", commit);
      input.addEventListener("keydown", (event) => {
        keepKeyInField(event);
        if (event.key === "Enter") input.blur();
        if (event.key === "Escape") {
          input.value = String(findWidget(node, name)?.value ?? "");
          input.blur();
        }
      });
      into.append(input);
      return (vals) => {
        if (document.activeElement !== input) input.value = String(vals[name] ?? "");
      };
    },
  };

  let currentGroup = null;
  for (const row of rows) {
    if (row.section !== undefined) {
      const caption = el("div", "fbui-section", row.section);
      card.entries.push({ row: { ...row, inGroup: currentGroup }, element: caption, syncs: [] });
      root.append(caption);
      continue;
    }
    if (row.panel) {
      const line = el("div", "fbui-row panel");
      if (row.grow) {
        line.classList.add("grow");
        line.style.minHeight = `${rowHeight(row)}px`;
      } else {
        line.style.height = `${rowHeight(row)}px`;
      }
      const panel = row.panel(node, card);
      line.append(panel.element);
      card.entries.push({ row: { ...row, inGroup: row.group ?? null }, element: line, syncs: panel.sync ? [panel.sync] : [] });
      root.append(line);
      continue;
    }
    if (row.control) {
      // `widgets` names the hidden widgets the control drives: their sockets
      // sit on this row and a link dims it, as for a widget row.
      const names = row.widgets ?? [];
      for (const name of names) {
        const widget = findWidget(node, name);
        if (widget) {
          hideWidget(widget);
          managed.add(name);
        }
      }
      const line = el("div", "fbui-row");
      const control = el("div", "fbui-control");
      const built = row.control(node, card);
      control.append(built.element);
      line.append(built.label ?? el("span", "fbui-label", row.label ?? ""), control);
      card.entries.push({ row: { ...row, inGroup: row.group ?? null }, element: line, names, syncs: built.sync ? [built.sync] : [] });
      root.append(line);
      continue;
    }
    if (isGroupHeader(row)) {
      currentGroup = row.group;
      const header = el("div", "fbui-group");
      const toggle = el("button", "toggle");
      toggle.type = "button";
      toggle.append(el("span", "glyph", "▸"), el("span", "", row.label ?? row.group));
      const tools = row.tools?.(node, card);
      header.append(toggle, ...(tools ? [tools.element] : []));
      toggle.addEventListener("click", () => {
        node.properties ??= {};
        node.properties[groupKey(row.group)] = !groupOpen(row.group);
        card.byHand = true;
        try {
          refresh();
        } finally {
          card.byHand = false;
        }
      });
      card.entries.push({ row, element: header, header: true, syncs: tools?.sync ? [tools.sync] : [] });
      root.append(header);
      continue;
    }
    const names = row.pair ?? [row.widget];
    for (const name of names) {
      const widget = findWidget(node, name);
      if (widget) {
        hideWidget(widget);
        managed.add(name);
      }
    }
    const line = el("div", "fbui-row");
    const label = el("span", "fbui-label");
    label.title = row.title ?? widgetTooltip(node, names[0]);
    if (row.info) {
      label.classList.add("has-info");
      label.append(el("span", "", row.label ?? names[0]), infoIcon(row.info));
    } else {
      label.textContent = row.label ?? names[0];
    }
    const control = el("div", "fbui-control");
    const syncs = [];
    const fields = new Map();
    names.forEach((name, index) => {
      const widget = findWidget(node, name);
      if (!widget) return;
      if (index > 0) control.append(el("span", "fbui-sep", row.sep ?? "·"));
      if (row.prefixes?.[index]) control.append(el("span", "fbui-sep", row.prefixes[index]));
      const field = el("div", "fbui-field");
      const kind = rowKind(row, widget, comboValues(widget));
      syncs.push(builders[kind](name, row, field));
      control.append(field);
      fields.set(name, field);
    });
    for (const { text, onClick, ...rest } of [row.button ?? []].flat()) {
      control.append(makeButton(text, { ...rest, onClick: () => onClick?.(node, card) }));
    }
    line.append(label, control);
    card.entries.push({ row: { ...row, inGroup: row.group ?? null }, element: line, names, syncs, fields });
    root.append(line);
  }

  // Clicks on controls stay out of the graph's drag; empty card space falls
  // through. A stage stops the events it consumes itself.
  root.addEventListener("pointerdown", (event) => {
    if (event.target.closest("button,input,select,textarea,.fbui-scrub")) event.stopPropagation();
  });

  // The label column fits the longest label (an (i) adds its icon), rather
  // than a share of the width that leaves a gap on a wide node.
  const measure = document.createElement("canvas").getContext("2d");
  measure.font = `11px ${FONT}`;
  const labelWidth = Math.max(0, ...[...root.querySelectorAll(":scope > .fbui-row > .fbui-label")].map((label) => measure.measureText(label.textContent).width + (label.querySelector(".fbui-info") ? 20 : 0)));
  if (labelWidth > 0) root.style.setProperty("--fbui-label-w", `${Math.ceil(labelWidth) + 2}px`);

  const widget = node.addDOMWidget(CARD_WIDGET, CARD_WIDGET, root, { serialize: false, hideOnZoom: false });
  keepDomWidgetWidthAuto(widget);
  // options.serialize only keeps it out of the prompt; saves skip a widget
  // whose own serialize is false.
  widget.serialize = false;
  card.widget = widget;
  const declared = () => card.height + WIDGET_FRAME;
  if (rows.some((row) => row.grow)) fillNodeHeight(widget, { minWidth, minHeight: declared });
  else pinHeight(widget, declared, minWidth);
  if (first && Array.isArray(node.widgets)) {
    const index = node.widgets.indexOf(widget);
    const firstDom = node.widgets.findIndex((item) => item !== widget && item.element);
    if (index >= 0 && firstDom >= 0 && firstDom < index) {
      node.widgets.splice(index, 1);
      node.widgets.splice(firstDom, 0, widget);
    }
  }

  // Each hidden widget with an input slot reports the y of its row, so the
  // frontend draws and hit-tests its socket there. A folded row hands its
  // socket to its group header, or to the card's top edge.
  const socketY = (name) => {
    const cardY = Number(card.widget?.y);
    if (card.disposed || !Number.isFinite(cardY)) return null;
    const top = card.rowTop.get(name);
    if (top === undefined) return cardY + WRAPPER_TOP;
    return socketWidgetY(cardY, WRAPPER_TOP, top, card.rowSpan.get(name) ?? ROW_HEIGHT);
  };
  const socketed = [];
  for (const name of managed) {
    const target = findWidget(node, name);
    if (!target || !widgetSlot(node, name) || Object.getOwnPropertyDescriptor(target, "y")?.get) continue;
    let stored = target.y;
    try {
      Object.defineProperty(target, "y", { configurable: true, enumerable: true, get: () => socketY(name) ?? stored, set: (value) => { stored = value; } });
      socketed.push(() => { delete target.y; target.y = stored; });
    } catch {
      // A sealed widget keeps its socket where the frontend puts it.
    }
  }

  // While a link is dragged, the row under the pointer stands in for its
  // widget, so a drop anywhere on the row lands on that widget's input.
  const originalGetWidgetOnPos = node.getWidgetOnPos;
  if (typeof originalGetWidgetOnPos === "function") {
    node.getWidgetOnPos = function (canvasX, canvasY, includeDisabled) {
      const found = originalGetWidgetOnPos.call(this, canvasX, canvasY, includeDisabled);
      if (card.disposed || found !== card.widget || !globalThis.app?.canvas?.linkConnector?.isConnecting) return found;
      const localY = canvasY - this.pos[1] - Number(card.widget.y) - WRAPPER_TOP;
      for (const entry of card.entries) {
        const name = entry.names?.[0];
        const top = card.rowTop.get(name);
        if (top !== undefined && localY >= top && localY <= top + (card.rowSpan.get(name) ?? ROW_HEIGHT)) return findWidget(this, name) ?? found;
      }
      return found;
    };
  }

  function refresh() {
    if (card.disposed) return;
    const vals = values();
    const open = {};
    for (const entry of card.entries) if (entry.header) open[entry.row.group] = groupOpen(entry.row.group);
    const shown = visibleRows(card.entries.map((entry) => entry.row), vals, open, node);
    const shownSet = new Set(shown);
    const tops = rowTops(shown);
    card.rowTop = new Map();
    card.rowSpan = new Map();
    shown.forEach((row, index) => {
      const entry = card.entries.find((item) => item.row === row);
      for (const name of entry.names ?? []) {
        card.rowTop.set(name, tops[index]);
        card.rowSpan.set(name, rowHeight(row));
      }
      if (entry.header) card.rowTop.set(`group:${row.group}`, tops[index]);
    });
    for (const entry of card.entries) {
      if (!entry.names || shownSet.has(entry.row) || !entry.row.inGroup) continue;
      const headerTop = card.rowTop.get(`group:${entry.row.inGroup}`);
      if (headerTop === undefined) continue;
      for (const name of entry.names) {
        card.rowTop.set(name, headerTop);
        card.rowSpan.set(name, GROUP_HEIGHT);
      }
    }
    for (const entry of card.entries) {
      const show = shownSet.has(entry.row);
      entry.element.classList.toggle("hidden", !show);
      if (entry.header) entry.element.classList.toggle("open", open[entry.row.group]);
      if (!show) continue;
      if (entry.row.panel && !entry.row.grow && typeof entry.row.height === "function") entry.element.style.height = `${rowHeight(entry.row)}px`;
      const linkedNames = (entry.names ?? []).filter((name) => widgetLinked(node, name));
      const linked = linkedNames.length > 0 && linkedNames.length === entry.names.length;
      entry.element.classList.toggle("linked", linked);
      for (const [name, field] of entry.fields ?? []) field.classList.toggle("linked", linkedNames.includes(name));
      for (const sync of entry.syncs) sync(vals, linked);
    }
    const height = cardHeight(shown);
    if (height !== card.height) {
      const change = card.height ? height - card.height : 0;
      card.height = height;
      // A row opened by hand moves the node's height with it, so a preview
      // under the card keeps its size; anything else keeps the node's height,
      // lifted to the floor.
      const current = Number(node.size?.[1]) || 0;
      const target = card.byHand && !card.restoring ? current + change : current;
      const floor = Number(node.computeSize?.()?.[1]) || 0;
      node.setSize?.([Math.max(minWidth, node.size?.[0] || minWidth), Math.max(floor, target)]);
      node.graph?.setDirtyCanvas?.(true, true);
    }
    node._widgetSlotsDirty = true;
  }
  card.refresh = refresh;
  // For a node that changes which rows show from its own controls (a tool
  // switch): the node's height follows the card, as for a group opened by hand.
  card.refreshByHand = () => {
    card.byHand = true;
    try {
      refresh();
    } finally {
      card.byHand = false;
    }
  };

  for (const name of managed) {
    const target = findWidget(node, name);
    if (target) chainCallback(target, "callback", () => refresh());
  }
  chainCallback(node, "onConfigure", () => {
    card.restoring = true;
    queueMicrotask(() => {
      try {
        for (const name of managed) hideWidget(findWidget(node, name));
        refresh();
      } finally {
        card.restoring = false;
      }
      // A workflow saved before the card existed sized the node for the
      // stock widgets; give the card and its panels their floors.
      ensureNodeMinHeight(node);
    });
  });
  chainCallback(node, "onConnectionsChange", () => queueMicrotask(refresh));
  chainCallback(node, "onRemoved", () => {
    card.disposed = true;
    for (const restore of socketed) restore();
    if (typeof originalGetWidgetOnPos === "function") delete node.getWidgetOnPos;
  });

  refresh();
  return card;
}
