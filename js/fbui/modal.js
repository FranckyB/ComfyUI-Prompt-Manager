// Modals and context menus in the kit's look. One backdrop, one z-index
// scale, Escape and a click outside close, and keys typed inside stay out of
// the canvas.
import { el } from "./dom.js";
import { button, ICONS } from "./controls.js";
import { ensureKitCss, T, Z_MODAL } from "./theme.js";

// Open modals, newest last: Escape closes only the top one.
const stack = [];

// Whether a kit modal, menu or viewer is open: node shortcuts stand aside.
export const modalOpen = () => stack.length > 0;

// openModal({ title, body, footer, width, height, closable, onClose }) -> { overlay, modal, body, close, isTop }.
// `body` and `footer` are elements (or arrays of elements). closable: false
// takes away the close button, Escape and the backdrop click, for a dialog
// that must end with one of its own buttons.
export function openModal({ title = "", body = null, footer = null, width = null, height = null, closable = true, onClose = null } = {}) {
  ensureKitCss();
  const overlay = el("div", "fbui-overlay");
  const modal = el("div", "fbui-modal");
  const size = (value) => (typeof value === "number" ? `${value}px` : value);
  if (width) modal.style.width = size(width);
  if (height) modal.style.height = size(height);
  const head = el("div", "fbui-modal-head");
  head.append(el("span", "title", title));
  const close = () => {
    if (!overlay.isConnected) return;
    overlay.remove();
    stack.splice(stack.indexOf(close), 1);
    document.removeEventListener("keydown", onKey, true);
    onClose?.();
  };
  if (closable) head.append(button("", { icon: ICONS.close, small: true, title: "Close (Esc)", onClick: close }));
  modal.append(head);
  const content = el("div", "fbui-modal-body");
  for (const item of [body].flat()) if (item) content.append(item);
  modal.append(content);
  if (footer) {
    const foot = el("div", "fbui-modal-foot");
    for (const item of [footer].flat()) if (item) foot.append(item);
    modal.append(foot);
  }
  overlay.append(modal);
  const onKey = (event) => {
    if (!closable || event.key !== "Escape" || stack[stack.length - 1] !== close) return;
    event.stopPropagation();
    close();
  };
  overlay.addEventListener("pointerdown", (event) => {
    if (closable && event.target === overlay) close();
  });
  overlay.addEventListener("keydown", (event) => event.stopPropagation());
  document.addEventListener("keydown", onKey, true);
  document.body.append(overlay);
  stack.push(close);
  return { overlay, modal, body: content, close, isTop: () => stack[stack.length - 1] === close };
}

// prompt({ title, message, value, placeholder, confirmText }) -> Promise<string | null>:
// the trimmed text on OK or Enter, null on Cancel or Escape.
export function prompt({ title = "", message = "", value = "", placeholder = "", confirmText = "OK" } = {}) {
  return new Promise((resolve) => {
    let answer = null;
    const field = el("input", "fbui-text");
    field.type = "text";
    field.spellcheck = false;
    field.value = value;
    field.placeholder = placeholder;
    field.style.width = "min(520px, 70vw)";
    const text = el("div", "", message);
    text.style.cssText = "margin-bottom:8px;max-width:520px;line-height:1.45;color:" + T.label;
    const accept = () => {
      answer = field.value.trim() || null;
      dialog.close();
    };
    field.addEventListener("keydown", (event) => {
      if (event.key === "Enter") accept();
    });
    const dialog = openModal({
      title,
      body: [message ? text : null, field],
      footer: [button("Cancel", { onClick: () => dialog.close() }), button(confirmText, { primary: true, onClick: accept })],
      onClose: () => resolve(answer),
    });
    dialog.overlay.style.zIndex = String(Z_MODAL + stack.length);
    field.focus();
    field.select();
  });
}

// confirm({ title, message, confirmText, danger }) -> Promise<boolean>.
export function confirm({ title = "Are you sure?", message = "", confirmText = "OK", danger = false } = {}) {
  return new Promise((resolve) => {
    let answer = false;
    const text = el("div", "", message);
    text.style.cssText = "max-width:420px;line-height:1.45";
    const cancel = button("Cancel", { onClick: () => dialog.close() });
    const ok = button(confirmText, { primary: !danger, danger, onClick: () => { answer = true; dialog.close(); } });
    if (danger) ok.style.background = T.danger;
    const dialog = openModal({ title, body: text, footer: [cancel, ok], onClose: () => resolve(answer) });
    dialog.overlay.style.zIndex = String(Z_MODAL + stack.length);
    ok.focus();
  });
}

// openMenu(clientX, clientY, [{ label, action, disabled, danger }]) at the
// pointer, kept inside the viewport; closes on pick or any outside press.
export function openMenu(x, y, items) {
  ensureKitCss();
  document.querySelector(".fbui-menu")?.remove();
  const menu = el("div", "fbui-menu");
  const close = () => {
    menu.remove();
    document.removeEventListener("pointerdown", outside, true);
  };
  const outside = (event) => {
    if (menu.contains(event.target)) return;
    close();
    // The press that closes the menu must not also click what is under it.
    const swallow = (click) => {
      click.stopPropagation();
      click.preventDefault();
    };
    document.addEventListener("click", swallow, { capture: true, once: true });
    setTimeout(() => document.removeEventListener("click", swallow, true), 400);
  };
  for (const item of items) {
    const entry = el("button", item.danger ? "danger" : "", item.label);
    entry.type = "button";
    entry.disabled = Boolean(item.disabled);
    if (item.disabled) entry.style.opacity = ".45";
    entry.addEventListener("click", () => {
      close();
      item.action?.();
    });
    menu.append(entry);
  }
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
  document.body.append(menu);
  const rect = menu.getBoundingClientRect();
  if (rect.right > window.innerWidth) menu.style.left = `${Math.max(0, x - rect.width)}px`;
  if (rect.bottom > window.innerHeight) menu.style.top = `${Math.max(0, y - rect.height)}px`;
  setTimeout(() => document.addEventListener("pointerdown", outside, true), 0);
  return { menu, close };
}

// openViewer([{ url, title, video?, fallback?, where? }], index, { step? }):
// the picture (or video) as large as the screen allows over a darkened
// ComfyUI. ←/→ and the ‹ › buttons step through several (while a video
// plays the keys seek it instead), Esc or a click on the dark closes. `fallback` is an image URL shown when the browser cannot
// play a video. With `step(direction)` the caller does the stepping instead
// and returns the item to show (null to stay); `where` labels its position.
export function openViewer(items, index = 0, { step = null } = {}) {
  ensureKitCss();
  if (!items.length) return;
  const root = el("div", "fbui-viewer");
  const stage = el("div", "stage");
  const name = el("div", "name");
  const where = el("div", "where");
  const close = () => {
    root.remove();
    stack.splice(stack.indexOf(close), 1);
    document.removeEventListener("keydown", onKey, true);
  };
  const prev = button("", { icon: ICONS.chevronLeft, title: "Previous (←)", onClick: () => move(-1) });
  const next = button("", { icon: ICONS.chevronRight, title: "Next (→)", onClick: () => move(1) });
  prev.classList.add("nav", "prev");
  next.classList.add("nav", "next");
  const shut = button("", { icon: ICONS.close, title: "Close (Esc)", onClick: close });
  shut.classList.add("shut");
  let at = 0;
  function show(next) {
    at = (next + items.length) % items.length;
    const item = items[at];
    let media;
    if (item.video) {
      media = el("video");
      Object.assign(media, { controls: true, autoplay: true, loop: true, playsInline: true, src: item.url });
      if (item.fallback) {
        media.addEventListener("error", () => {
          const still = el("img");
          still.src = item.fallback;
          media.replaceWith(still);
        });
      }
    } else {
      media = el("img");
      media.src = item.url;
      media.draggable = false;
    }
    stage.replaceChildren(media);
    // Focused, a video takes ← / → itself (seeking) and Space.
    if (item.video) media.focus({ preventScroll: true });
    name.textContent = item.title ?? "";
    name.title = item.title ?? "";
    where.textContent = item.where ?? `${at + 1} / ${items.length}`;
    where.hidden = !where.textContent || (!step && items.length < 2);
  }
  function move(direction) {
    if (!step) return show(at + direction);
    const item = step(direction);
    if (!item) return;
    items = [item];
    show(0);
  }
  const onKey = (event) => {
    if (stack[stack.length - 1] !== close) return;
    if (event.key === "Escape") close();
    else if (stage.firstChild?.tagName === "VIDEO" && (event.key === "ArrowLeft" || event.key === "ArrowRight")) return;
    else if ((step || items.length > 1) && (event.key === "ArrowLeft" || event.key === "ArrowRight")) move(event.key === "ArrowRight" ? 1 : -1);
    else return;
    event.preventDefault();
    event.stopPropagation();
  };
  root.addEventListener("pointerdown", (event) => {
    if (event.target === root || event.target === stage) close();
  });
  prev.hidden = next.hidden = !step && items.length < 2;
  root.append(stage, name, where, shut, prev, next);
  document.body.append(root);
  document.addEventListener("keydown", onKey, true);
  stack.push(close);
  show(index);
  return { close };
}
