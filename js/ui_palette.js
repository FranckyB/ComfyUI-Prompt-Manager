// Prompt Manager's modal palette, now the FB UI kit's colours (js/fbui/theme.js)
// under the names the modals already use. Selected and hover states follow
// the kit: soft blue fill, solid blue edge.
import { T } from "./fbui/theme.js";

export const PM_UI_PALETTE = {
    // Backdrop behind modal dialogs
    overlay: T.overlay,
    // Main modal surface
    panel: T.panel,
    // Outer border of modal containers
    panelBorder: T.plateEdge,
    // Dividers between modal sections (header/body/footer)
    sectionBorder: T.rule,

    // Text input/select control surface
    inputBg: T.card,
    // Input/select/button neutral stroke
    inputBorder: T.plateEdge,
    // Neutral button/chip background, and its hover
    buttonBg: T.fill,
    buttonHover: T.fillHover,

    // Primary foreground text
    textPrimary: T.inkSoft,
    // Titles and important labels
    textHeading: T.ink,
    // Secondary text
    textMuted: T.label,
    // Hints and helper text
    textHint: T.faint,

    // Card/tile background (thumbnails, prompt cards)
    cardBg: T.plate,
    // Card default border
    cardBorder: T.plateEdge,

    // The accent (focus/selected), dimmed, as a soft fill, as an edge
    accent: T.brand,
    accentDim: T.brandInk,
    accentSoft: T.brandSoftHover,
    accentBorder: T.brand,
    // Text on a solid accent fill
    onAccent: T.onBrand,

    // Destructive actions
    danger: T.danger,
    dangerSoft: T.dangerSoft,
    dangerInk: T.dangerInk,
};
