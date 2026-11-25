// Solid node colours for every node of the pack that installs this, the one
// thing that makes the packs read as one family on the canvas. Nodes without
// any JS of their own get the colours too.
//
// A node follows the setting while it carries no colours or exactly one
// scheme's pair; a colour picked in ComfyUI's own colour menu is the user's
// and wins.
import { app } from "../../../scripts/app.js";
import { chainCallback } from "./dom.js";

export const NODE_COLOR_SCHEMES = [
  { name: "FB Blue", colors: { title: "#1b3350", body: "#161b22" } },
  { name: "Theme default", colors: null },
  { name: "Graphite", colors: { title: "#242424", body: "#2f2f2f" } },
  { name: "Slate", colors: { title: "#1f2c38", body: "#2a3b4a" } },
  { name: "Teal", colors: { title: "#14332f", body: "#1c453f" } },
  { name: "Plum", colors: { title: "#2e1f38", body: "#3d2a4a" } },
  { name: "Rust", colors: { title: "#3a2317", body: "#4c2f20" } },
];
const NAMES = NODE_COLOR_SCHEMES.map((scheme) => scheme.name);
const norm = (value) => (typeof value === "string" ? value.trim().toLowerCase() : "");
const schemeColors = (name) => NODE_COLOR_SCHEMES.find((scheme) => scheme.name === name)?.colors ?? null;
const SCHEME_TITLES = new Set(NODE_COLOR_SCHEMES.flatMap((scheme) => (scheme.colors ? [scheme.colors.title] : [])));

// Our scheme titles are dark: the classic renderer's grey title text gets
// lost on them. Marked on the node by nodeCreated (never saved).
const MARK = Symbol.for("fbui.packNode");

function followsSetting(node) {
  const color = norm(node.color);
  const bgcolor = norm(node.bgcolor);
  if (!color && !bgcolor) return true;
  return NODE_COLOR_SCHEMES.some((scheme) => scheme.colors && color === scheme.colors.title && bgcolor === scheme.colors.body);
}

function apply(node, colors) {
  // Assigning undefined (never `delete`) keeps Nodes 2.0 reactive.
  node.color = colors ? colors.title : undefined;
  node.bgcolor = colors ? colors.body : undefined;
}

function allNodes(graph, seen = new Set()) {
  if (!graph || seen.has(graph)) return [];
  seen.add(graph);
  const out = [];
  for (const node of graph._nodes ?? graph.nodes ?? []) {
    out.push(node);
    if (node.subgraph) out.push(...allNodes(node.subgraph, seen));
  }
  return out;
}

// Wrapping drawTitleText is the one place a prototype is patched directly: a
// draw method must return through the original. Shared by every pack.
function installTitleInk() {
  const proto = globalThis.LiteGraph?.LGraphNode?.prototype;
  if (typeof proto?.drawTitleText !== "function" || proto.drawTitleText[MARK]) return;
  const original = proto.drawTitleText;
  function drawTitleText(ctx, options, ...rest) {
    if (options && this?.[MARK] && SCHEME_TITLES.has(norm(this.color))) options = { ...options, default_title_color: "#ffffff" };
    return original.call(this, ctx, options, ...rest);
  }
  drawTitleText[MARK] = true;
  proto.drawTitleText = drawTitleText;
}

// installAppearance({ settingsPrefix: "FBnodes", category: "FBnodes" }) from
// the pack's fbui_setup.js. The pack is found from this file's own URL
// (/extensions/<folder>/fbui/...), which matches the nodes' python_module
// whatever the install folder is called.
export function installAppearance({ settingsPrefix, category }) {
  const folder = decodeURIComponent(new URL(import.meta.url).pathname.split("/extensions/")[1]?.split("/")[0] ?? "");
  const module = `custom_nodes.${folder}`;
  const settingId = `${settingsPrefix}.Appearance.NodeColor`;
  const ours = (node) => node?.constructor?.nodeData?.python_module === module;
  let active = NAMES[0];

  app.registerExtension({
    name: `${settingsPrefix}.Appearance`,
    settings: [
      {
        id: settingId,
        category: [category, "Appearance", "Node color"],
        name: "Node color",
        type: "combo",
        defaultValue: NAMES[0],
        options: NAMES,
        tooltip: "Colour scheme for this pack's nodes. Nodes you coloured by hand keep their own colours.",
        onChange(value) {
          active = NAMES.includes(value) ? value : NAMES[0];
          for (const node of allNodes(app.graph)) if (ours(node) && followsSetting(node)) apply(node, schemeColors(active));
          app.graph?.setDirtyCanvas?.(true, true);
        },
      },
    ],
    setup() {
      const stored = app.ui?.settings?.getSettingValue?.(settingId);
      if (NAMES.includes(stored)) active = stored;
      installTitleInk();
      // ComfyUI's colour menus list LGraphCanvas.node_colors: the suite's
      // blue joins them, so a node coloured by hand can go back to it. Its
      // menu name comes from the pack's locales/<lang>/main.json (color.fbBlue).
      const colors = app.canvas?.constructor?.node_colors;
      const blue = NODE_COLOR_SCHEMES[0].colors;
      if (colors && !colors.fbBlue) colors.fbBlue = { color: blue.title, bgcolor: blue.body, groupcolor: "#2f5f8a" };
    },
    nodeCreated(node) {
      if (!ours(node)) return;
      node[MARK] = true;
      if (followsSetting(node)) apply(node, schemeColors(active));
      // Native text boxes wear the kit's text-area colour: Classic draws the
      // widget's own textarea, Nodes 2.0 one carrying the widget's options.
      for (const widget of node.widgets ?? []) {
        if (widget.element?.tagName !== "TEXTAREA") continue;
        widget.element.classList.add("fbui-native-text");
        widget.options = { ...widget.options, "data-fbui-text": "true" };
      }
      // Saved colours arrive after construction: a workflow saved before the
      // pack had colours, or with another scheme, follows the setting now.
      chainCallback(node, "onConfigure", function () {
        if (followsSetting(this)) apply(this, schemeColors(active));
      });
    },
  });
}
