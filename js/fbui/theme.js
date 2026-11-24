// FB UI kit: the shared look of FBnodes, Prompt Manager and H3RefModPicker.
// Every pack ships an identical copy of js/fbui/ (tools/sync_fbui.py), so each
// pack works alone and all of them look the same when installed together.
// Layout and controls adapted from ComfyUI-AusBoss (MIT, (c) 2026 ausboss).
//
// No imports: the node:test suites load this file without the ComfyUI app.

// Bump once per release that changes the CSS below (against the last pushed
// value, not per local edit). Two packs shipping the same version inject one
// stylesheet; different versions both inject and the later wins.
export const KIT_VERSION = 13;

export const T = Object.freeze({
  brand: "#4299e1",          // the one accent: picked segment, switch on, focus ring
  onBrand: "#08182a",        // text on a brand fill
  brandSoft: "rgba(66,153,225,.16)",
  brandSoftHover: "rgba(66,153,225,.28)",
  brandInk: "#8cc4f2",       // chip text on brandSoft

  card: "#101317",           // card ground
  fill: "#2b3036",           // every control: solid, no border
  fillHover: "#363c43",
  rule: "#2a2f35",           // hairline after a section label
  switchOff: "#5f666e",      // the "off" half of a boolean pill when picked

  ink: "#f5f7f9",            // values
  inkSoft: "#dfe3e7",        // default card text
  inkHover: "#ffffff",
  label: "#b2b9c0",          // row labels, unselected segments
  faint: "#959ca4",          // units, captions, placeholders

  danger: "#e0605a",
  dangerInk: "#ffd9d6",
  dangerSoft: "#3a1c1b",
  warn: "#e0a040",
  warnInk: "#ffdca8",
  warnSoft: "#3a2a10",
  latent: "#c840c0",          // a lossless .latent file beside a clip

  overlay: "rgba(0,0,0,.72)", // modal backdrop
  panel: "#171b20",           // modal surface
  shade: "rgba(8,10,12,.62)", // outside a crop
  maskInk: "#ff3b3b",         // painted mask, drawn at half strength
  plate: "#1d232b",           // media frame and footnote bar: neutral slate
  plateEdge: "#323a45",       // their 1px trim
});

export const FONT = `system-ui,-apple-system,"Segoe UI",Roboto,sans-serif`;
export const MONO = `ui-monospace,"Cascadia Mono",Menlo,Consolas,monospace`;

export const ROW_HEIGHT = 26;
// The footnote bar under a picture or video (card panel row height).
export const CAPTION_HEIGHT = 24;
export const ROW_GAP = 8;
export const SECTION_HEIGHT = 17;
export const GROUP_HEIGHT = 22;
export const CARD_PADDING = 10;
export const UNIT_SLOT_WIDTH = 22;
export const RADIUS_XS = 4;
export const RADIUS = 6;
export const RADIUS_CARD = 10;

// Modals stack above ComfyUI's own dialogs; menus above modals.
export const Z_MODAL = 10000;
export const Z_MENU = 10010;

const CHEVRON = `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M1 1l4 4 4-4' fill='none' stroke='%23b2b9c0' stroke-width='1.5'/%3E%3C/svg%3E")`;

export function kitCss() {
  return `
.fbui-card{position:relative;box-sizing:border-box;width:100%;height:100%;display:flex;flex-direction:column;gap:${ROW_GAP}px;padding:${CARD_PADDING}px 12px;overflow:hidden;border-radius:${RADIUS_CARD}px;background:${T.card};font:12px/1.3 ${FONT};color:${T.inkSoft};pointer-events:none}
.fbui-row{display:grid;grid-template-columns:minmax(52px,24%) 1fr;align-items:center;gap:10px;height:${ROW_HEIGHT}px;flex:none;pointer-events:auto}
.fbui-row.wide{grid-template-columns:1fr}
.fbui-card .hidden{display:none!important}
.fbui-row.linked{opacity:.5}
.fbui-row.linked .fbui-control{pointer-events:none}
.fbui-row.panel{display:block}
.fbui-row.panel.grow{flex:1 1 auto;height:auto}
.fbui-label{overflow:hidden;white-space:nowrap;text-overflow:ellipsis;color:${T.label};font-size:11px;user-select:none}
.fbui-label.has-info{display:flex;align-items:center;gap:4px}
.fbui-label.has-info>span{overflow:hidden;text-overflow:ellipsis}
.fbui-info{flex:none;display:inline-flex;color:${T.faint};cursor:pointer}
.fbui-info:hover,.fbui-info.on{color:${T.brandInk}}
.fbui-info svg{width:13px;height:13px}
.fbui-pop{position:fixed;z-index:${Z_MENU};max-width:460px;padding:14px 16px;border:1px solid ${T.plateEdge};border-radius:${RADIUS_CARD}px;background:${T.panel};box-shadow:0 10px 32px rgba(0,0,0,.6);color:${T.inkSoft};font:13px/1.55 ${FONT};user-select:text;cursor:auto;outline:none}
.fbui-pop>div+div{margin-top:6px}
.fbui-pop .pair{display:grid;grid-template-columns:minmax(150px,auto) 1fr;gap:14px;align-items:baseline}
.fbui-pop code{justify-self:start;padding:1px 6px;border-radius:4px;background:${T.card};color:${T.brandInk};font:600 13px ${MONO};cursor:copy}
.fbui-pop code:hover{background:${T.fill}}
.fbui-pop code.copied{background:${T.brand};color:${T.onBrand}}
.fbui-control{display:flex;align-items:center;gap:6px;min-width:0;height:100%}
.fbui-field{display:flex;align-items:center;gap:6px;flex:1 1 0;min-width:0;height:100%}
.fbui-field.linked{opacity:.5;pointer-events:none}
.fbui-sep{flex:none;color:${T.faint};font-size:11px}
.fbui-section{height:${SECTION_HEIGHT}px;flex:none;display:flex;align-items:flex-end;gap:8px;color:${T.faint};font:600 10px/1 ${FONT};letter-spacing:.1em;text-transform:uppercase;user-select:none}
.fbui-section::after{content:"";flex:1;height:1px;margin-bottom:4px;background:${T.rule}}
.fbui-group{height:${GROUP_HEIGHT}px;flex:none;display:flex;align-items:center;gap:6px;padding:0 0 0 4px;color:${T.faint};font:600 10px/1 ${FONT};letter-spacing:.1em;text-transform:uppercase;pointer-events:auto}
.fbui-group .toggle{flex:1;min-width:0;height:100%;display:flex;align-items:center;gap:6px;padding:0;border:none;border-radius:${RADIUS}px;background:transparent;color:inherit;font:inherit;letter-spacing:inherit;text-transform:inherit;cursor:pointer;text-align:left}
.fbui-group .toggle:hover{color:${T.inkHover}}
.fbui-group .glyph{display:inline-block;width:10px;transition:transform .12s}
.fbui-group.open .glyph{transform:rotate(90deg)}
.fbui-seg{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;width:100%;height:${ROW_HEIGHT}px;padding:2px;box-sizing:border-box;border-radius:${RADIUS}px;background:${T.fill}}
.fbui-seg button{min-width:0;border:none;border-radius:${RADIUS_XS}px;padding:0 2px;background:transparent;color:${T.label};font:600 11px/1 ${FONT};cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fbui-seg button:hover{color:${T.inkHover}}
.fbui-seg button.on{background:${T.brand};color:${T.onBrand}}
.fbui-bool button.on:first-child{background:${T.switchOff};color:${T.ink}}
.fbui-select{width:100%;height:${ROW_HEIGHT}px;padding:0 24px 0 8px;box-sizing:border-box;border:1px solid transparent;border-radius:${RADIUS}px;background:${T.fill} ${CHEVRON} no-repeat right 9px center;color:${T.ink};font:500 12px ${FONT};appearance:none;-webkit-appearance:none;cursor:pointer;overflow:hidden;text-overflow:ellipsis;min-width:0}
.fbui-select:hover{background-color:${T.fillHover}}
.fbui-select:focus-visible{outline:2px solid ${T.brand};outline-offset:1px}
.fbui-select option{background:${T.panel};color:${T.ink}}
.fbui-text{width:100%;height:${ROW_HEIGHT}px;padding:0 8px;box-sizing:border-box;border:1px solid transparent;border-radius:${RADIUS}px;background:${T.fill};color:${T.ink};font:12px ${FONT};min-width:0}
.fbui-text:focus{outline:none;border-color:${T.brand}}
textarea.fbui-text{background:${T.card};border-color:${T.plateEdge};font-size:13px;line-height:1.45}
.fbui-text.entry{flex:none;height:30px;background:${T.card};border-color:${T.plateEdge};font-size:13px}
.comfy-multiline-input.fbui-native-text,.lg-node-widgets textarea[data-fbui-text]{background:${T.card};border:1px solid ${T.plateEdge};color:${T.ink}}
.fbui-text::placeholder{color:${T.faint}}
.fbui-btn{flex:none;display:inline-flex;align-items:center;justify-content:center;gap:5px;height:${ROW_HEIGHT}px;padding:0 10px;box-sizing:border-box;border:1px solid transparent;border-radius:${RADIUS}px;background:${T.fill};color:${T.inkSoft};font:600 11px/1 ${FONT};cursor:pointer;white-space:nowrap;pointer-events:auto}
.fbui-btn:hover{background:${T.fillHover};color:${T.inkHover}}
.fbui-btn:focus-visible{outline:2px solid ${T.brand};outline-offset:1px}
.fbui-btn:disabled{opacity:.45;cursor:default}
.fbui-card [hidden],.fbui-overlay [hidden],.fbui-btn[hidden]{display:none!important}
.fbui-btn.grow{flex:1 1 0;min-width:0}
/* soft: the suite's blue action button; on: a tool that is enabled but not the one in use. Same look. */
.fbui-btn.on,.fbui-btn.soft{background:${T.brandSoft};color:${T.brandInk}}
.fbui-btn.soft:hover{background:${T.brandSoftHover};color:${T.ink}}
.fbui-btn.primary{background:${T.brand};color:${T.onBrand}}
.fbui-btn.primary:hover{filter:brightness(1.1)}
.fbui-btn.danger:hover{background:${T.dangerSoft};color:${T.dangerInk}}
.fbui-btn.small{height:20px;padding:0 7px;font-size:10px}
.fbui-btn.icon{width:${ROW_HEIGHT}px;padding:0}
.fbui-btn.icon.small{width:20px}
.fbui-slide{display:flex;align-items:center;gap:5px;align-self:stretch;margin:2px 0;padding:0 4px;border-radius:${RADIUS}px;touch-action:none}
.fbui-slide>span{overflow:hidden;text-overflow:ellipsis}
.fbui-slide svg{flex:none;color:${T.faint}}
.fbui-slide:hover,.fbui-slide.on{background:${T.fill};color:${T.ink}}
.fbui-slide:hover svg,.fbui-slide.on svg{color:${T.brand}}
.fbui-rail{position:relative;height:24px;margin:0 7px;touch-action:none;cursor:pointer;user-select:none;pointer-events:auto}
.fbui-rail::before{content:"";position:absolute;left:0;right:0;top:10px;height:5px;border-radius:3px;background:${T.fill}}
.fbui-rail.disabled{opacity:.45;cursor:default}
.fbui-rail-fill{position:absolute;left:0;top:10px;height:5px;border-radius:3px;background:${T.brandSoftHover};pointer-events:none}
.fbui-rail-head{position:absolute;top:0;bottom:0;width:2px;margin-left:-1px;background:${T.ink};box-shadow:0 0 0 1px rgba(0,0,0,.55);pointer-events:none}
.fbui-rail-head::before{content:"";position:absolute;top:-1px;left:50%;transform:translateX(-50%);border:5px solid transparent;border-top:6px solid ${T.ink};filter:drop-shadow(0 0 1px rgba(0,0,0,.7))}
.fbui-range-fill{position:absolute;top:10px;height:5px;background:${T.brand};opacity:.6;pointer-events:none}
.fbui-range-edge{position:absolute;top:4px;width:7px;height:17px;margin-left:-3.5px;border-radius:3px;background:${T.brand};box-shadow:0 0 0 1px rgba(0,0,0,.5);pointer-events:none}
.fbui-readout{flex:1 1 0;min-width:0;color:${T.faint};font:11px ${FONT};font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fbui-readout b{color:${T.ink};font-weight:600}
.fbui-tag{flex:none;color:${T.brand};font:650 10px ${FONT};letter-spacing:.08em;text-transform:uppercase;user-select:none}
.fbui-btn svg{flex:none}
.fbui-chip{height:20px;flex:none;display:inline-flex;align-items:center;padding:0 9px;border:none;border-radius:999px;background:${T.brandSoft};color:${T.brandInk};font:600 10px/1 ${FONT};letter-spacing:.02em;font-variant-numeric:tabular-nums;white-space:nowrap}
.fbui-chip.warn{background:${T.warnSoft};color:${T.warnInk}}
.fbui-chip.danger{background:${T.dangerSoft};color:${T.dangerInk}}
.fbui-scrub{display:flex;width:100%;height:${ROW_HEIGHT}px;box-sizing:border-box;border:1px solid transparent;border-radius:${RADIUS}px;background:${T.fill};overflow:hidden;flex:1 1 0;min-width:0;color:${T.ink};font:500 12px ${FONT};font-variant-numeric:tabular-nums}
.fbui-scrub:focus-within{border-color:${T.brand}}
.fbui-scrub>.fbui-scrub-input{flex:1 1 auto;min-width:0;width:100%;height:100%;border:none;padding:0;margin:0;background:transparent;color:inherit;text-align:center;cursor:ew-resize;user-select:none;font:inherit;outline:none;border-radius:0}
.fbui-scrub>.fbui-scrub-input:focus{cursor:text;user-select:text}
.fbui-scrub>.fbui-scrub-unit{flex:none;align-self:center;color:${T.faint};font:600 10px ${FONT};letter-spacing:.04em;text-transform:uppercase;text-align:center;user-select:none;pointer-events:none}
.fbui-scrub>.fbui-scrub-step{flex:none;width:14px;display:flex;flex-direction:column;background:${T.fillHover}}
.fbui-scrub>.fbui-scrub-step>button{flex:1 1 0;min-height:0;border:none;background:transparent;color:${T.label};cursor:pointer;padding:0;margin:0;display:grid;place-items:center;border-radius:0}
.fbui-scrub>.fbui-scrub-step>button:hover{color:${T.brand};background:rgba(255,255,255,.05)}
.fbui-stage{position:relative;width:100%;height:100%;min-height:0;box-sizing:border-box;border:1px solid ${T.plateEdge};border-radius:${RADIUS_CARD}px;background:${T.plate};overflow:hidden;pointer-events:auto}
.fbui-stage>img,.fbui-stage>video,.fbui-stage>canvas.fbui-fit{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:block}
.fbui-stage-empty{position:absolute;inset:0;display:grid;place-items:center;padding:12px;color:${T.faint};font:11px ${FONT};text-align:center;pointer-events:none}
.fbui-stage-tools{position:absolute;top:6px;right:6px;display:flex;gap:4px;z-index:2}
.fbui-stage-tools .fbui-btn{height:20px;padding:0 7px;font-size:10px;letter-spacing:.04em;background:rgba(16,19,23,.82)}
.fbui-stage-tools .fbui-btn:hover{background:${T.fillHover}}
.fbui-stage-tools .fbui-btn.on{background:${T.brand};color:${T.onBrand}}
.fbui-caption{box-sizing:border-box;width:100%;height:${CAPTION_HEIGHT}px;display:flex;align-items:center;justify-content:center;gap:6px;padding:0 10px;border:1px solid ${T.plateEdge};border-radius:${RADIUS + 2}px;background:${T.plate};color:${T.label};font:600 11px ${FONT};font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:auto}
.fbui-overlay{position:fixed;inset:0;z-index:${Z_MODAL};display:flex;align-items:center;justify-content:center;background:${T.overlay};font:12px/1.4 ${FONT};color:${T.inkSoft}}
.fbui-modal{display:flex;flex-direction:column;max-width:92vw;max-height:90vh;min-width:320px;border-radius:${RADIUS_CARD}px;background:${T.panel};box-shadow:0 18px 60px rgba(0,0,0,.6);overflow:hidden}
.fbui-modal-head{flex:none;display:flex;align-items:center;gap:8px;padding:10px 12px;color:${T.ink};font:600 13px ${FONT}}
.fbui-modal-head .title{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fbui-modal-body{flex:1 1 auto;min-height:0;overflow:auto;padding:0 12px 12px}
.fbui-modal-foot{flex:none;display:flex;justify-content:flex-end;gap:6px;padding:10px 12px;background:${T.card}}
.fbui-viewer{position:fixed;inset:0;z-index:${Z_MODAL};background:rgba(5,6,8,.94);font:12px ${FONT};color:${T.inkSoft}}
.fbui-viewer .stage{position:absolute;inset:48px 64px 24px;display:flex;align-items:center;justify-content:center}
.fbui-viewer .stage img,.fbui-viewer .stage video{max-width:100%;max-height:100%;object-fit:contain;border-radius:${RADIUS}px;box-shadow:0 12px 48px rgba(0,0,0,.6);display:block}
.fbui-viewer .name{position:absolute;left:20px;top:14px;max-width:60vw;padding:4px 10px;border-radius:999px;background:${T.panel};color:${T.ink};font:600 12px ${FONT};white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fbui-viewer .where{position:absolute;left:50%;top:14px;transform:translateX(-50%);padding:4px 10px;border-radius:999px;background:${T.panel};color:${T.label};font:600 12px ${FONT};font-variant-numeric:tabular-nums}
.fbui-viewer .shut{position:absolute;right:16px;top:10px}
.fbui-viewer .nav{position:absolute;top:50%;transform:translateY(-50%);width:44px;height:44px;border-radius:50%;background:rgba(16,19,23,.8)}
.fbui-viewer .nav svg{width:20px;height:20px}
.fbui-viewer .nav.prev{left:12px}
.fbui-viewer .nav.next{right:12px}
.fbui-menu{position:fixed;z-index:${Z_MENU};min-width:160px;padding:4px;border-radius:${RADIUS}px;background:${T.panel};box-shadow:0 8px 28px rgba(0,0,0,.55);font:12px ${FONT};color:${T.inkSoft}}
.fbui-menu button{display:block;width:100%;padding:6px 10px;border:none;border-radius:${RADIUS_XS}px;background:transparent;color:inherit;font:inherit;text-align:left;cursor:pointer}
.fbui-menu button:hover{background:${T.fillHover};color:${T.inkHover}}
.fbui-menu button.danger:hover{background:${T.dangerSoft};color:${T.dangerInk}}
.fbui-menu .sep{height:1px;margin:4px 6px;background:${T.rule}}
`;
}

export function ensureKitCss() {
  const id = `fbui-css-v${KIT_VERSION}`;
  if (document.getElementById(id)) return;
  const style = document.createElement("style");
  style.id = id;
  style.textContent = kitCss();
  document.head.append(style);
}
