// Pure decisions behind card.js: which rows show, how tall the card is,
// where each row sits and what control a widget gets. No DOM, for node:test.
import { CARD_PADDING, GROUP_HEIGHT, ROW_GAP, ROW_HEIGHT, SECTION_HEIGHT } from "./theme.js";

// A widget input socket is drawn at the widget's y plus half a slot.
export const SLOT_OFFSET = 10;
const SEGMENT_MAX_OPTIONS = 4;
const SEGMENT_MAX_CHARS = 26;

export const isGroupHeader = (row) => row.group !== undefined && row.widget === undefined && row.pair === undefined && !row.panel && !row.control && row.section === undefined;

// Rows the current values leave visible. `open` maps a group to its state.
export function visibleRows(rows, values, open = {}, node = null) {
  return rows.filter((row) => {
    if (row.when && row.when(values, node) === false) return false;
    if (isGroupHeader(row)) return true;
    return !(row.inGroup && !open[row.inGroup]);
  });
}

export function rowHeight(row) {
  if (row.section !== undefined) return SECTION_HEIGHT;
  if (row.panel) return Number(typeof row.height === "function" ? row.height() : row.height) || ROW_HEIGHT;
  if (isGroupHeader(row)) return GROUP_HEIGHT;
  return ROW_HEIGHT;
}

export function cardHeight(rows) {
  if (!rows.length) return CARD_PADDING * 2;
  return CARD_PADDING * 2 + rows.reduce((sum, row) => sum + rowHeight(row), 0) + ROW_GAP * (rows.length - 1);
}

export function rowTops(rows) {
  let y = CARD_PADDING;
  return rows.map((row) => {
    const top = y;
    y += rowHeight(row) + ROW_GAP;
    return top;
  });
}

// The y a hidden widget reports so its socket sits level with its row.
export function socketWidgetY(cardY, insetTop, rowTop, height) {
  return cardY + insetTop + rowTop + Math.min(height / 2, 13) - SLOT_OFFSET;
}

export function segmentFits(labels) {
  if (labels.length < 2 || labels.length > SEGMENT_MAX_OPTIONS) return false;
  return labels.reduce((sum, label) => sum + String(label).length, 0) <= SEGMENT_MAX_CHARS;
}

export function choiceLabel(row, value, values = [], node = null) {
  const own = typeof row?.labels === "function" ? row.labels(value, values, node) : row?.labels?.[value];
  return own ?? String(value);
}

// Row kind from the widget it mirrors, unless the row names one.
export function rowKind(row, widget, values = []) {
  if (row.kind) return row.kind;
  const type = String(widget?.type ?? "");
  if (type === "number" || type === "slider" || type === "INT" || type === "FLOAT") return "scrub";
  if (type === "combo" || type === "COMBO") return segmentFits(values.map((value) => choiceLabel(row, value, values))) ? "segment" : "select";
  if (type === "toggle" || type === "BOOLEAN") return "switch";
  return "text";
}
