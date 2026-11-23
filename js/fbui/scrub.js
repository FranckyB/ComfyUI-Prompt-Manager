// The kit's number control: drag left/right to scrub, click to type a value
// or a sum (1024*1.5), chevrons or ArrowUp/Down to step, Shift for the fine
// step. Pure math first (node:test), DOM factory below.
import { keepKeyInField } from "./dom.js";

export const SCRUB_DEAD_ZONE = 3;
export const SCRUB_PIXELS_PER_STEP = 4;
// Shift on a box with no finer step slows the drag by this much instead.
export const SCRUB_SLOW_FACTOR = 4;

export function quantize(value, { min = -Infinity, max = Infinity, decimals = 2 } = {}) {
  const number = Number(value);
  if (!Number.isFinite(number)) return quantize(min === -Infinity ? 0 : min, { min, max, decimals });
  const factor = 10 ** Math.max(0, decimals);
  return Math.round(Math.max(min, Math.min(max, number)) * factor) / factor;
}

// A sum typed into a number box: + - * / %, brackets and signs. Returns the
// answer, or null when the text is not a sum that comes to a finite number.
const SUM_NUMBER = /(?:\d+\.?\d*|\.\d+)/y;
export function sumValue(text) {
  const source = String(text ?? "");
  if (source.length > 200) return null;
  let at = 0;
  const skip = () => { while (source[at] === " ") at += 1; };
  const atom = () => {
    skip();
    const mark = source[at];
    if (mark === "+" || mark === "-") {
      at += 1;
      const value = atom();
      return value === null ? null : mark === "-" ? -value : value;
    }
    if (mark === "(") {
      at += 1;
      const value = sum();
      skip();
      if (value === null || source[at] !== ")") return null;
      at += 1;
      return value;
    }
    SUM_NUMBER.lastIndex = at;
    const match = SUM_NUMBER.exec(source);
    if (!match) return null;
    at += match[0].length;
    return Number(match[0]);
  };
  const product = () => {
    let value = atom();
    while (value !== null) {
      skip();
      const mark = source[at];
      if (mark !== "*" && mark !== "/" && mark !== "%") break;
      at += 1;
      const next = atom();
      if (next === null) return null;
      value = mark === "*" ? value * next : mark === "/" ? value / next : value % next;
    }
    return value;
  };
  function sum() {
    let value = product();
    while (value !== null) {
      skip();
      const mark = source[at];
      if (mark !== "+" && mark !== "-") break;
      at += 1;
      const next = product();
      if (next === null) return null;
      value = mark === "+" ? value + next : value - next;
    }
    return value;
  }
  const value = sum();
  skip();
  return value === null || at !== source.length || !Number.isFinite(value) ? null : value;
}

// Steps from a widget's options: the fine step is the widget's own
// increment; a 0..1 float moves by 0.05, a wide float by ten increments, a
// whole number by its increment (a wide pixel size by 8).
export function scrubSteps(options = {}, isFloat = true, pixels = false) {
  const fine = Number(options.step2 ?? (options.step ? options.step / 10 : null) ?? (isFloat ? 0.01 : 1)) || (isFloat ? 0.01 : 1);
  const range = Number(options.max ?? Infinity) - Number(options.min ?? -Infinity);
  let step;
  if (!isFloat) step = pixels && Number.isFinite(range) && range > 2048 ? 8 : Math.max(1, fine);
  else if (Number.isFinite(range) && range <= 2) step = Math.max(fine, 0.05);
  else step = fine * 10;
  const decimals = isFloat ? Number(options.precision ?? Math.max(0, -Math.floor(Math.log10(fine)))) : 0;
  return { step, fineStep: fine, decimals };
}

export function scrubbedValue(start, deltaX, fine, { step = 1, fineStep = null, ...range } = {}) {
  if (Math.abs(deltaX) <= SCRUB_DEAD_ZONE) return quantize(start, range);
  const finer = fineStep != null && fineStep < step;
  const size = fine && finer ? fineStep : step;
  const pixels = fine && !finer ? SCRUB_PIXELS_PER_STEP * SCRUB_SLOW_FACTOR : SCRUB_PIXELS_PER_STEP;
  const steps = Math.round((deltaX - Math.sign(deltaX) * SCRUB_DEAD_ZONE) / pixels);
  return quantize(start + steps * size, range);
}

function chevron(up) {
  const points = up ? "1.5,4 4.5,1 7.5,4" : "1.5,1 4.5,4 7.5,1";
  return `<svg width="9" height="5" viewBox="0 0 9 5" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="${points}"/></svg>`;
}

// makeScrub({ value, min, max, step, fineStep, decimals, unit, unitWidth, title, onChange, onSettle })
// -> { root, get, set, setRange }. onChange fires on every committed change,
// onSettle once a gesture finishes (the undo point).
export function makeScrub(options = {}) {
  const opts = { value: 0, min: -Infinity, max: Infinity, step: 1, fineStep: null, decimals: 2, unit: "", unitWidth: 0, title: "", ...options };
  let current = quantize(opts.value, opts);
  const format = (value) => Number(value).toFixed(Math.max(0, opts.decimals));
  const stepped = (direction, fine) => current + direction * (fine ? (opts.fineStep ?? opts.step) : opts.step);

  const box = document.createElement("div");
  box.className = "fbui-scrub";
  const input = document.createElement("input");
  input.className = "fbui-scrub-input";
  input.type = "text";
  input.inputMode = "decimal";
  input.readOnly = true;
  input.value = format(current);
  input.title = `${opts.title ? `${opts.title} ` : ""}Hold and drag right/up to raise, left/down to lower; click to type; Shift = fine.`;

  const commit = (value) => {
    const next = quantize(value, opts);
    const changed = next !== current;
    current = next;
    input.value = format(current);
    if (changed) opts.onChange?.(current);
  };
  const settle = () => opts.onSettle?.();

  // Press and drag either way: right or up raises the value, left or down
  // lowers it. The axis is the one the drag starts along.
  let drag = null;
  const travel = (event) => (drag.axis === "y" ? drag.y - event.clientY : event.clientX - drag.x);
  input.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !input.readOnly) return;
    drag = { x: event.clientX, y: event.clientY, start: current, scrubbed: false, fine: event.shiftKey, axis: "x" };
    input.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  input.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (!drag.scrubbed && Math.max(Math.abs(dx), Math.abs(dy)) > SCRUB_DEAD_ZONE) {
      drag.scrubbed = true;
      drag.axis = Math.abs(dy) > Math.abs(dx) ? "y" : "x";
      drag.fine = event.shiftKey;
    }
    if (!drag.scrubbed) return;
    if (event.shiftKey !== drag.fine) {
      // Pressing or releasing Shift carries on from here at the other speed.
      const moved = travel(event);
      const offset = Math.sign(moved || 1) * (SCRUB_DEAD_ZONE + 1);
      drag.start = current;
      if (drag.axis === "y") drag.y = event.clientY + offset;
      else drag.x = event.clientX - offset;
      drag.fine = event.shiftKey;
    }
    commit(scrubbedValue(drag.start, travel(event), event.shiftKey, opts));
  });
  const endDrag = (event) => {
    if (!drag) return;
    try { input.releasePointerCapture(event.pointerId); } catch { /* mouse fallback */ }
    const scrubbed = drag.scrubbed;
    drag = null;
    if (scrubbed) {
      settle();
      return;
    }
    input.readOnly = false;
    input.focus();
    input.select();
    // The canvas can grab focus during the click sequence; win that race once.
    requestAnimationFrame(() => {
      if (document.activeElement === input) return;
      input.readOnly = false;
      input.focus();
      input.select();
    });
  };
  input.addEventListener("pointerup", endDrag);
  input.addEventListener("pointercancel", endDrag);
  input.addEventListener("keydown", (event) => {
    keepKeyInField(event);
    if (event.key === "Enter") input.blur();
    else if (event.key === "Escape") { input.value = format(current); input.blur(); }
    else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      event.preventDefault();
      commit(stepped(event.key === "ArrowUp" ? 1 : -1, event.shiftKey));
      input.select();
      settle();
    }
  });
  input.addEventListener("blur", () => {
    if (!input.readOnly) {
      const number = Number(input.value);
      const typed = Number.isFinite(number) && input.value.trim() !== "" ? number : sumValue(input.value);
      if (typed !== null) { commit(typed); settle(); }
      input.readOnly = true;
    }
    input.value = format(current);
  });

  const steppers = document.createElement("div");
  steppers.className = "fbui-scrub-step";
  for (const direction of [1, -1]) {
    const button = document.createElement("button");
    button.type = "button";
    button.tabIndex = -1;
    button.innerHTML = chevron(direction > 0);
    button.addEventListener("click", (event) => { commit(stepped(direction, event.shiftKey)); settle(); });
    steppers.append(button);
  }

  box.append(input);
  if (opts.unit || opts.unitWidth) {
    // A fixed unit slot keeps every number in a column on one centre line.
    const unit = document.createElement("span");
    unit.className = "fbui-scrub-unit";
    unit.textContent = opts.unit;
    if (opts.unitWidth) unit.style.width = `${opts.unitWidth}px`;
    else unit.style.padding = "0 3px 0 1px";
    box.append(unit);
  }
  box.append(steppers);

  return {
    root: box,
    get: () => current,
    set: (value) => {
      current = quantize(value, opts);
      if (input.readOnly) input.value = format(current);
    },
    setRange: (range = {}) => {
      for (const key of ["min", "max", "step", "fineStep", "decimals"]) if (range[key] !== undefined) opts[key] = range[key];
      current = quantize(current, opts);
      if (input.readOnly) input.value = format(current);
    },
  };
}
