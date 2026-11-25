// Stand-alone controls for panels that are not card rows (a toolbar over a
// preview, a modal footer). Card rows build on the same classes.
import { chainCallback, chainHandler, el } from "./dom.js";

// button("Browse", { icon, title, primary, soft, danger, small, grow, onClick })
// soft: the suite's blue action button (Save, New, More).
export function button(label, { icon = "", title = "", primary = false, soft = false, danger = false, small = false, grow = false, onClick = null } = {}) {
  const node = el("button", "fbui-btn");
  node.type = "button";
  node.classList.toggle("primary", primary);
  node.classList.toggle("soft", soft);
  node.classList.toggle("danger", danger);
  node.classList.toggle("small", small);
  node.classList.toggle("grow", grow);
  // An icon alone is a square button the height of a row.
  node.classList.toggle("icon", Boolean(icon) && !label);
  if (icon) node.insertAdjacentHTML("beforeend", icon);
  if (label) node.append(el("span", "", label));
  if (title) node.title = title;
  if (onClick) node.addEventListener("click", (event) => onClick(event));
  return node;
}

// The help popover: `lines` under (or over) `anchor`, a client-pixel box
// { left, top, bottom }. It stays until a click outside it (or Esc), so the
// text can be selected and copied; a click on a token copies that token.
// `lines` are strings, or [code, text] pairs shown as a two-column list
// (tokens and what they do). Returns close().
export function openHelp(lines, anchor, { keep = null, onClose = null } = {}) {
  const pop = el("div", "fbui-pop");
  pop.tabIndex = -1;
  for (const line of lines) {
    if (Array.isArray(line)) {
      const row = el("div", "pair");
      const code = el("code", "", line[0]);
      code.title = "Click to copy";
      code.addEventListener("click", () => {
        navigator.clipboard?.writeText(line[0]).then(() => {
          code.classList.add("copied");
          setTimeout(() => code.classList.remove("copied"), 900);
        });
      });
      row.append(code, el("span", "", line[1]));
      pop.append(row);
    } else {
      pop.append(el("div", "", line));
    }
  }
  const close = () => {
    pop.remove();
    document.removeEventListener("pointerdown", outside, true);
    document.removeEventListener("keydown", onEscape, true);
    onClose?.();
  };
  // A press on `keep` (the control that opened it) is that control's to handle.
  const outside = (event) => {
    if (!pop.contains(event.target) && !keep?.contains(event.target)) close();
  };
  const onEscape = (event) => {
    if (event.key === "Escape") close();
  };
  // Keys typed here (Ctrl+C on a selection) are the popover's, not the canvas's.
  pop.addEventListener("keydown", (event) => event.stopPropagation());
  document.body.append(pop);
  pop.style.left = `${Math.max(8, Math.min(window.innerWidth - pop.offsetWidth - 8, anchor.left))}px`;
  const below = anchor.bottom + 6;
  pop.style.top = `${below + pop.offsetHeight > window.innerHeight - 8 ? Math.max(8, anchor.top - pop.offsetHeight - 6) : below}px`;
  pop.focus({ preventScroll: true });
  document.addEventListener("pointerdown", outside, true);
  document.addEventListener("keydown", onEscape, true);
  return close;
}

// A small (i) beside a label; a click opens `lines` (see openHelp).
export function info(lines) {
  const icon = el("span", "fbui-info");
  icon.innerHTML = ICONS.info;
  icon.title = "Help";
  let close = null;
  icon.addEventListener("click", (event) => {
    event.stopPropagation();
    if (close) return close();
    icon.classList.add("on");
    close = openHelp(lines, icon.getBoundingClientRect(), {
      keep: icon,
      onClose: () => {
        close = null;
        icon.classList.remove("on");
      },
    });
  });
  return icon;
}

// The node's help as a "?" at the right of its title bar, where most packs
// put it; a click opens `lines` (see openHelp). Nodes 2.0 draws no title
// badges, so there it is "About this node" in the node's menu.
const BADGE_INSET = 16;
const BADGE_MIN_WIDTH = 140;

function badgeShown(node) {
  return !node.flags?.collapsed && (node.size?.[0] ?? 0) >= BADGE_MIN_WIDTH;
}

export function helpBadge(node, lines) {
  const centerY = -(globalThis.LiteGraph?.NODE_TITLE_HEIGHT ?? 30) / 2;
  chainCallback(node, "onDrawForeground", function (ctx) {
    if (!badgeShown(this)) return;
    ctx.save();
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.font = "bold 19px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("?", this.size[0] - BADGE_INSET, centerY + 1);
    ctx.restore();
  });
  chainHandler(node, "onMouseDown", function (event, pos) {
    if (!badgeShown(this) || !pos) return false;
    if (Math.abs(pos[0] - (this.size[0] - BADGE_INSET)) > 10 || Math.abs(pos[1] - centerY) > 11) return false;
    openHelp(lines, { left: event.clientX - 150, top: event.clientY, bottom: event.clientY + 6 });
    // Consumed, so the click does not start a drag.
    return true;
  });
  chainCallback(node, "getExtraMenuOptions", function (_canvas, options) {
    if (!globalThis.LiteGraph?.vueNodesMode) return;
    options.unshift({ content: "About this node", callback: () => openHelp(lines, { left: window.innerWidth / 2 - 150, top: 80, bottom: 80 }) });
  });
}

// A row of pill choices: segmented([{ value, label, title }], { onPick, bool })
// -> { root, set(value) }.
export function segmented(choices, { onPick, bool = false } = {}) {
  const root = el("div", bool ? "fbui-seg fbui-bool" : "fbui-seg");
  const buttons = new Map();
  for (const choice of choices) {
    const item = el("button", "", choice.label ?? String(choice.value));
    item.type = "button";
    item.title = choice.title ?? String(choice.label ?? choice.value);
    item.addEventListener("click", () => onPick?.(choice.value));
    root.append(item);
    buttons.set(choice.value, item);
  }
  return {
    root,
    set(value) {
      for (const [key, item] of buttons) item.classList.toggle("on", key === value);
    },
  };
}

// A slide handle (↔ or ↕) at the start of a row's fields: dragging it
// reports the pointer's sideways travel for either axis, as every slider
// here moves sideways (right is right, or down).
export function slideHandle({ axis = "x", title = "", onStart, onSlide, onEnd } = {}) {
  const handle = el("span", "fbui-slide");
  handle.insertAdjacentHTML("beforeend", axis === "x" ? ICONS.slideX : ICONS.slideY);
  handle.title = title;
  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    handle.setPointerCapture(event.pointerId);
    handle.classList.add("on");
    const zoom = handle.getBoundingClientRect().width / Math.max(1, handle.offsetWidth) || 1;
    const start = event.clientX;
    const move = (e) => onSlide?.((e.clientX - start) / zoom);
    const end = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", end);
      handle.removeEventListener("pointercancel", end);
      handle.classList.remove("on");
      onEnd?.();
    };
    onStart?.();
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
  });
  return handle;
}

// A timeline rail with a playhead, for picking a frame: press or drag
// anywhere on it. rail({ onSeek(fraction, settled) }) -> { root, set(fraction), setEnabled(on) }.
// `fraction` is 0..1 along the rail; the caller maps it to its own frames.
export function rail({ onSeek, title = "" } = {}) {
  const root = el("div", "fbui-rail");
  const fill = el("div", "fbui-rail-fill");
  const head = el("div", "fbui-rail-head");
  root.append(fill, head);
  if (title) root.title = title;
  let enabled = true;
  const at = (event) => {
    const box = root.getBoundingClientRect();
    return Math.max(0, Math.min(1, (event.clientX - box.left) / Math.max(1, box.width)));
  };
  root.addEventListener("pointerdown", (event) => {
    if (!enabled || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    root.setPointerCapture(event.pointerId);
    onSeek?.(at(event), false);
  });
  root.addEventListener("pointermove", (event) => {
    if (root.hasPointerCapture(event.pointerId)) onSeek?.(at(event), false);
  });
  const end = (event) => {
    if (!root.hasPointerCapture(event.pointerId)) return;
    root.releasePointerCapture(event.pointerId);
    onSeek?.(at(event), true);
  };
  root.addEventListener("pointerup", end);
  root.addEventListener("pointercancel", end);
  return {
    root,
    set(fraction) {
      const pct = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
      head.style.left = pct;
      fill.style.width = pct;
    },
    setEnabled(on) {
      enabled = on;
      root.classList.toggle("disabled", !on);
    },
  };
}

// A rail with in and out edges and a playhead, for trimming: press near an
// edge to move it, anywhere else to move the playhead.
// rangeRail({ onEdge(edge, fraction, settled), onSeek(fraction, settled) })
//   -> { root, set({ start, stop, head }) } with edge "in" | "out", all 0..1.
export function rangeRail({ onEdge, onSeek, title = "" } = {}) {
  const root = el("div", "fbui-rail fbui-range");
  const fill = el("div", "fbui-range-fill");
  const edgeIn = el("div", "fbui-range-edge");
  const edgeOut = el("div", "fbui-range-edge");
  const head = el("div", "fbui-rail-head");
  root.append(fill, edgeIn, edgeOut, head);
  if (title) root.title = title;
  let shown = { start: 0, stop: 1, head: 0 };
  let drag = null;
  const at = (event) => {
    const box = root.getBoundingClientRect();
    return Math.max(0, Math.min(1, (event.clientX - box.left) / Math.max(1, box.width)));
  };
  const send = (event, settled) => {
    if (drag === "head") onSeek?.(at(event), settled);
    else onEdge?.(drag, at(event), settled);
  };
  root.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    root.setPointerCapture(event.pointerId);
    // The nearer edge within 8px; on top of each other, the side pressed.
    const width = root.getBoundingClientRect().width;
    const x = at(event);
    const toIn = Math.abs(x - shown.start) * width;
    const toOut = Math.abs(x - shown.stop) * width;
    drag = Math.min(toIn, toOut) > 8 ? "head" : toIn < toOut || (toIn === toOut && x < shown.start) ? "in" : "out";
    send(event, false);
  });
  root.addEventListener("pointermove", (event) => {
    if (drag && root.hasPointerCapture(event.pointerId)) send(event, false);
  });
  const end = (event) => {
    if (!drag || !root.hasPointerCapture(event.pointerId)) return;
    root.releasePointerCapture(event.pointerId);
    send(event, true);
    drag = null;
  };
  root.addEventListener("pointerup", end);
  root.addEventListener("pointercancel", end);
  const pct = (fraction) => `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  return {
    root,
    set(next) {
      shown = { ...shown, ...next };
      fill.style.left = pct(shown.start);
      fill.style.width = pct(shown.stop - shown.start);
      edgeIn.style.left = pct(shown.start);
      edgeOut.style.left = pct(shown.stop);
      head.style.left = pct(shown.head);
    },
  };
}

// Small stroke icons, drawn in currentColor so they follow the button state.
const svg = (body) => `<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
export const ICONS = {
  folder: svg('<path d="M1.5 4.5v8a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1v-6a1 1 0 0 0-1-1H8L6.5 3.5h-4a1 1 0 0 0-1 1z"/>'),
  play: svg('<path d="M5 3.5v9l7-4.5z" fill="currentColor"/>'),
  pause: svg('<path d="M5 3.5v9M11 3.5v9"/>'),
  expand: svg('<path d="M10 2.5h3.5V6M6 13.5H2.5V10M13.5 2.5 9 7M2.5 13.5 7 9"/>'),
  brush: svg('<path d="M10.5 2.5l3 3-6 6-3-3zM4.5 8.5l-2 5 5-2"/>'),
  crop: svg('<path d="M4 1.5V12h10.5M1.5 4H12v10.5"/>'),
  feather: svg('<path d="M13.5 2.5C8 2.5 4.5 6 4.5 11.5h2c5.5 0 7-4 7-9z"/><path d="M2.5 13.5l7-7"/>'),
  ratio: svg('<rect x="1.5" y="3.5" width="13" height="9" rx="1"/><path d="M4.5 8.5v-2H7M11.5 7.5v2H9"/>'),
  slideX: svg('<path d="M1.5 8h13M4.5 5 1.5 8l3 3M11.5 5l3 3-3 3"/>'),
  slideY: svg('<path d="M8 1.5v13M5 4.5 8 1.5l3 3M5 11.5l3 3 3-3"/>'),
  centerX: svg('<path d="M2 3v10M14 3v10M8 5v6M5 8H4M12 8h-1"/>'),
  centerY: svg('<path d="M3 2h10M3 14h10M5 8h6M8 5V4M8 12v-1"/>'),
  pad: svg('<rect x="5.5" y="5.5" width="5" height="5" rx=".5"/><path d="M1.5 5V1.5H5M11 1.5h3.5V5M14.5 11v3.5H11M5 14.5H1.5V11"/>'),
  rotate: svg('<path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9M13.5 2.5v3h-3"/><rect x="6" y="6" width="4" height="4" rx=".5"/>'),
  rotateLeft: svg('<path d="M2.5 5.5h7a4 4 0 0 1 0 8h-2M5 3 2.5 5.5 5 8"/>'),
  rotateRight: svg('<path d="M13.5 5.5h-7a4 4 0 0 0 0 8h2M11 3l2.5 2.5L11 8"/>'),
  cropTurn: svg('<path d="M4 1.5V10h6M1.5 4H10v2"/><path d="M14.5 11a3.5 3.5 0 1 1-1-2.5M14.5 7v2h-2"/>'),
  save: svg('<path d="M2.5 2.5h9l2 2v9h-11z"/><path d="M5 2.5v3.5h5V2.5M5 13.5V9.5h6v4"/>'),
  scale: svg('<rect x="1.5" y="8.5" width="6" height="6" rx=".5"/><path d="M9.5 1.5h5v5M14.5 1.5 8.5 7.5"/>'),
  eraser: svg('<path d="M6 13.5h7.5M2.5 10l6-6.5 5 5-5 5h-3z"/>'),
  undo: svg('<path d="M4.5 6.5h6a3 3 0 0 1 0 6h-3M6.5 4l-2.5 2.5L6.5 9"/>'),
  redo: svg('<path d="M11.5 6.5h-6a3 3 0 0 0 0 6h3M9.5 4l2.5 2.5L9.5 9"/>'),
  trash: svg('<path d="M2.5 4.5h11M6 4.5V3h4v1.5M4 4.5l.7 9h6.6l.7-9"/>'),
  flipX: svg('<path d="M8 1.5v13M6 4 2 12h4zM10 4l4 8h-4z"/>'),
  flipY: svg('<path d="M1.5 8h13M4 6 12 2v4zM4 10l8 4v-4z"/>'),
  reset: svg('<path d="M2.5 8a5.5 5.5 0 1 0 1.6-3.9M2.5 2.5v3h3"/>'),
  close: svg('<path d="M3.5 3.5l9 9M12.5 3.5l-9 9"/>'),
  invert: svg('<circle cx="8" cy="8" r="5.5"/><path d="M8 2.5v11a5.5 5.5 0 0 0 0-11z" fill="currentColor"/>'),
  eye: svg('<path d="M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z"/><circle cx="8" cy="8" r="1.8"/>'),
  info: svg('<circle cx="8" cy="8" r="6.5"/><path d="M8 7.2V11.5M8 4.6v.1"/>'),
  open: svg('<path d="M9.5 2.5h4v4M13.5 2.5 8 8M11.5 9.5v3a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1h3"/>'),
  up: svg('<path d="M8 13.5v-11M3.5 7 8 2.5 12.5 7"/>'),
  refresh: svg('<path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9M13.5 2.5v3h-3"/>'),
  chevronUp: svg('<path d="M3.5 10 8 5.5l4.5 4.5"/>'),
  chevronLeft: svg('<path d="M10 3.5 5.5 8l4.5 4.5"/>'),
  chevronRight: svg('<path d="M6 3.5 10.5 8 6 12.5"/>'),
  check: svg('<path d="M3 8.5 6.5 12 13 4.5"/>'),
  grip: svg('<g fill="currentColor" stroke="none"><circle cx="6" cy="4" r="1.1"/><circle cx="10" cy="4" r="1.1"/><circle cx="6" cy="8" r="1.1"/><circle cx="10" cy="8" r="1.1"/><circle cx="6" cy="12" r="1.1"/><circle cx="10" cy="12" r="1.1"/></g>'),
  scissors: svg('<circle cx="4" cy="4.5" r="2"/><circle cx="4" cy="11.5" r="2"/><path d="M5.7 5.6 13.5 12M5.7 10.4 13.5 4"/>'),
  timer: svg('<circle cx="8" cy="9" r="5.5"/><path d="M8 6v3l2 1.5M6.5 1.5h3"/>'),
  chevronDown: svg('<path d="M3.5 6 8 10.5 12.5 6"/>'),
  grid: svg('<rect x="2" y="2" width="5" height="5" rx="1"/><rect x="9" y="2" width="5" height="5" rx="1"/><rect x="2" y="9" width="5" height="5" rx="1"/><rect x="9" y="9" width="5" height="5" rx="1"/>'),
  tiles: svg('<path d="M2 2h3v3H2zM6.5 2h3v3h-3zM11 2h3v3h-3zM2 6.5h3v3H2zM6.5 6.5h3v3h-3zM11 6.5h3v3h-3zM2 11h3v3H2zM6.5 11h3v3h-3zM11 11h3v3h-3z"/>'),
  list: svg('<path d="M5.5 4h8M5.5 8h8M5.5 12h8M2.5 4h.5M2.5 8h.5M2.5 12h.5"/>'),
  search: svg('<circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5 14 14"/>'),
  film: svg('<rect x="2" y="3" width="12" height="10" rx="1.5"/><path d="M5 3v10M11 3v10M2 6h3M2 10h3M11 6h3M11 10h3"/>'),
  music: svg('<path d="M6 12V3.5l7.5-1.5V10.5"/><circle cx="4.5" cy="12" r="1.8"/><circle cx="12" cy="10.5" r="1.8"/>'),
  file: svg('<path d="M4 1.5h5.5L13 5v9.5H4zM9.5 1.5V5H13"/>'),
};
