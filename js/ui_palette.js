// Shared UI palette used by Prompt Manager modals.
// Keep this aligned with Workflow Builder's neutral + blue accent styling.
export const PM_UI_PALETTE = {
    // Backdrop behind modal dialogs
    overlay: "hsl(0 0% 0% / 0.8)",
    // Main modal surface
    panel: "hsl(216 11% 15%)",
    // Outer border of modal containers
    panelBorder: "hsl(216 20% 65% / 0.24)",
    // Dividers between modal sections (header/body/footer)
    sectionBorder: "hsl(216 20% 65% / 0.20)",

    // Text input/select control surface
    inputBg: "hsl(220 15% 10%)",
    // Input/select/button neutral stroke
    inputBorder: "hsl(218 10% 41%)",
    // Neutral button/chip background
    buttonBg: "hsl(219 16% 18%)",

    // Primary foreground text
    textPrimary: "hsl(0 0% 87%)",
    // Titles and important labels
    textHeading: "hsl(220 13% 85%)",
    // Secondary text
    textMuted: "hsl(0 0% 67%)",
    // Hints and helper text
    textHint: "hsl(216 15% 65%)",

    // Card/tile background (thumbnails, prompt cards)
    cardBg: "hsl(219 16% 18%)",
    // Card default border
    cardBorder: "hsl(217 12% 22%)",

    // Exact Builder accent family (from workflow_builder.js)
    // Primary blue accent (focus/selected)
    accent: "hsl(208 73% 57% / 0.9)",
    // Slightly dimmed accent (hover/secondary accent)
    accentDim: "hsl(208 73% 57% / 0.7)",
    // Soft accent fill for selected backgrounds
    accentSoft: "hsl(208 73% 57% / 0.16)",
    // Accent border used on selected elements
    accentBorder: "hsl(208 73% 57% / 0.65)",
};
