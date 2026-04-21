/**
 * RecipeModelLoader extension.
 * Shows color-coded loaded assets (Model/VAE/CLIP) from workflow_data.
 */

import { app } from "../../scripts/app.js";

const MODEL_CHOICES = ["model_a", "model_b"];
const WEIGHT_DTYPE_CHOICES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"];

function _trimMiddle(text, maxLen) {
    const s = String(text || "");
    if (s.length <= maxLen) return s;
    const left = Math.ceil((maxLen - 3) / 2);
    const right = Math.floor((maxLen - 3) / 2);
    return `${s.slice(0, left)}...${s.slice(s.length - right)}`;
}

function _shortName(name) {
    const raw = String(name || "").trim();
    if (!raw) return "";
    const normalized = raw.replace(/\\/g, "/");
    const base = normalized.split("/").pop() || normalized;
    return base.replace(/\.(safetensors|ckpt|pt|pth|bin|gguf|sft)$/i, "");
}

function _shortList(names) {
    const list = Array.isArray(names) ? names : [];
    return list.map(_shortName).filter(Boolean);
}

const COLORS = {
    modelCheckpoint: "#4CAF50",
    modelDiffusion: "#2196F3",
    modelGguf: "#FF9800",
    vae: "#FF6B6B",
    clip: "#C4A300",
    notFound: "#F44336",
    muted: "#B0B0B0",
    missing: "#E57373",
};

function _buildRows(info) {
    const status = String(info.status || "").toLowerCase();
    if (!info || status === "idle") {
        return [
            { label: "Model", labelColor: COLORS.modelDiffusion, value: "(awaiting execution)", valueColor: COLORS.muted },
            { label: "Clip", labelColor: COLORS.clip, value: "(awaiting execution)", valueColor: COLORS.muted },
            { label: "Vae", labelColor: COLORS.vae, value: "(awaiting execution)", valueColor: COLORS.muted },
        ];
    }

    const modelKind = String(info.model_kind || info.loader_type || "").toLowerCase();
    const modelColor = modelKind === "checkpoint"
        ? COLORS.modelCheckpoint
        : (modelKind === "gguf" ? COLORS.modelGguf : COLORS.modelDiffusion);

    const rows = [];
    const modelShort = _shortName(info.model_name || "");
    rows.push({
        label: "Model",
        labelColor: modelColor,
        value: modelShort || "(none)",
        valueColor: modelShort ? "#CCCCCC" : COLORS.missing,
    });

    const loaderType = String(info.loader_type || "").toLowerCase();
    const vaeShort = _shortName(info.vae_name || "");
    const clipShort = _shortList(info.clip_names);

    if (loaderType === "checkpoint") {
        // Keep all tags visible; show source explicitly for checkpoint-provided assets.
        const usingInternalVae = !!info.uses_checkpoint_vae;
        const usingInternalClip = !!info.uses_checkpoint_clip;
        rows.push({
            label: "Clip",
            labelColor: COLORS.clip,
            value: usingInternalClip ? "From Checkpoint" : (clipShort.length > 0 ? clipShort.join(", ") : "(none provided)"),
            valueColor: usingInternalClip ? COLORS.muted : (clipShort.length > 0 ? "#CCCCCC" : COLORS.missing),
        });
        rows.push({
            label: "Vae",
            labelColor: COLORS.vae,
            value: usingInternalVae ? "From Checkpoint" : (vaeShort || "(none provided)"),
            valueColor: usingInternalVae ? COLORS.muted : (vaeShort ? "#CCCCCC" : COLORS.missing),
        });
    } else {
        // Diffusion/UNet: always show Clip + Vae lines.
        rows.push({
            label: "Clip",
            labelColor: COLORS.clip,
            value: clipShort.length > 0 ? clipShort.join(", ") : "(none provided)",
            valueColor: clipShort.length > 0 ? "#CCCCCC" : COLORS.missing,
        });
        rows.push({
            label: "Vae",
            labelColor: COLORS.vae,
            value: vaeShort || "(none provided)",
            valueColor: vaeShort ? "#CCCCCC" : COLORS.missing,
        });
    }

    return rows;
}

function _drawPillRow(ctx, row, y, widgetWidth) {
    const left = 8;
    const badgeText = String(row.label || "");
    const value = String(row.value || "");
    const badgeW = 68;
    const valueX = left + badgeW + 10;
    const valueMax = Math.max(16, Math.floor((widgetWidth - valueX - 8) / 6));
    const displayValue = _trimMiddle(value, valueMax);

    ctx.font = "bold 11px Arial";
    const badgeH = 16;
    const badgeY = y - Math.floor(badgeH / 2);

    ctx.fillStyle = row.labelColor || COLORS.muted;
    ctx.beginPath();
    ctx.roundRect(left, badgeY, badgeW, badgeH, 8);
    ctx.fill();

    ctx.fillStyle = "#FFFFFF";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(badgeText, left + 8, y + 0.5);

    ctx.font = "12px Arial";
    ctx.fillStyle = row.valueColor || "#CCCCCC";
    ctx.fillText(displayValue, valueX, y + 0.5);
}

function _getWidgetValue(node, name, fallback = "") {
    const w = node?.widgets?.find((x) => x.name === name);
    if (!w || w.value == null) return fallback;
    return String(w.value);
}

function _widgetByNames(node, names) {
    const arr = Array.isArray(names) ? names : [names];
    return node?.widgets?.find((w) => arr.includes(w.name));
}

function _sanitizeChoice(value, allowed, fallback) {
    const v = String(value ?? "");
    return allowed.includes(v) ? v : fallback;
}

function _selectedModelSlot(node) {
    return _sanitizeChoice(
        _getWidgetValue(node, "model", "model_a"),
        MODEL_CHOICES,
        "model_a"
    );
}

function _readWidgetState(node) {
    const modelWidget = _widgetByNames(node, ["model"]);
    const dtypeWidget = _widgetByNames(node, ["weight_dtype"]);
    return {
        model: _sanitizeChoice(modelWidget?.value, MODEL_CHOICES, "model_a"),
        weight_dtype: _sanitizeChoice(dtypeWidget?.value, WEIGHT_DTYPE_CHOICES, "default"),
    };
}

function _applyWidgetState(node, state, { persist = true } = {}) {
    const modelWidget = _widgetByNames(node, ["model"]);
    const dtypeWidget = _widgetByNames(node, ["weight_dtype"]);
    const modelVal = _sanitizeChoice(state?.model, MODEL_CHOICES, "model_a");
    const dtypeVal = _sanitizeChoice(state?.weight_dtype, WEIGHT_DTYPE_CHOICES, "default");

    if (modelWidget) modelWidget.value = modelVal;
    if (dtypeWidget) dtypeWidget.value = dtypeVal;

    if (persist) {
        node.properties = node.properties || {};
        node.properties.wfml_model = modelVal;
        node.properties.wfml_weight_dtype = dtypeVal;
    }
}

function _restoreOrNormalizeWidgetState(node) {
    node.properties = node.properties || {};
    const live = _readWidgetState(node);

    const savedModel = _sanitizeChoice(node.properties.wfml_model, MODEL_CHOICES, "");
    const savedDtype = _sanitizeChoice(node.properties.wfml_weight_dtype, WEIGHT_DTYPE_CHOICES, "");

    _applyWidgetState(node, {
        model: savedModel || live.model || "model_a",
        weight_dtype: savedDtype || live.weight_dtype || "default",
    }, { persist: true });
}

function _isRerouteNode(n) {
    const t = String(n?.type || "").toLowerCase();
    const c = String(n?.comfyClass || "").toLowerCase();
    return t.includes("reroute") || c.includes("reroute");
}

function _resolveWorkflowUpstreamLink(node) {
    const wfInputIdx = (node.inputs || []).findIndex((i) => i?.name === "workflow_data");
    if (wfInputIdx < 0) return null;
    let linkId = node.inputs[wfInputIdx]?.link;
    if (linkId == null) return null;

    const graph = node.graph;
    const seen = new Set();
    for (let hop = 0; hop < 24; hop++) {
        if (linkId == null || seen.has(linkId)) break;
        seen.add(linkId);

        const li = graph?.links?.[linkId];
        if (!li) break;
        const src = graph?.getNodeById?.(li.origin_id);
        if (!src) break;
        if (!_isRerouteNode(src)) {
            return { sourceNode: src, sourceSlot: li.origin_slot };
        }
        const in0 = src.inputs?.[0];
        linkId = in0?.link ?? null;
    }
    return null;
}

function _tryParseWorkflowData(value) {
    if (!value) return null;
    if (typeof value === "object") return value;
    if (typeof value === "string") {
        try {
            const p = JSON.parse(value);
            if (p && typeof p === "object") return p;
        } catch {
            return null;
        }
    }
    return null;
}

function _clipListFromWf(wf) {
    const c = wf?.clip;
    if (Array.isArray(c)) return c.filter(Boolean);
    if (typeof c === "string" && c) return [c];
    if (c && Array.isArray(c.names)) return c.names.filter(Boolean);
    return [];
}

function _vaeFromWf(wf) {
    const v = wf?.vae;
    if (typeof v === "string") return v;
    if (v && typeof v === "object" && typeof v.name === "string") return v.name;
    return "";
}

function _inferInfoFromWorkflowData(node, wf) {
    const modelKey = _selectedModelSlot(node);
    const modelName = typeof wf?.[modelKey] === "string" ? wf[modelKey] : "";
    if (!modelName) return null;

    const loaderType = String(wf?.loader_type || "").toLowerCase();
    const selectedLower = modelName.toLowerCase();
    const modelKind = loaderType === "checkpoint"
        ? "checkpoint"
        : (selectedLower.endsWith(".gguf") ? "gguf" : "diffusion");

    const vaeName = _vaeFromWf(wf);
    const clipNames = _clipListFromWf(wf);
    return {
        model: modelKey,
        model_name: _shortName(modelName),
        loader_type: loaderType,
        model_kind: modelKind,
        vae_name: vaeName,
        clip_names: clipNames,
        status: "ok",
        error_message: "",
        missing_name: "",
        uses_checkpoint_vae: (loaderType === "checkpoint" && !vaeName),
        uses_checkpoint_clip: (loaderType === "checkpoint" && clipNames.length === 0),
    };
}

function _refreshInfoFromUpstream(node) {
    const resolved = _resolveWorkflowUpstreamLink(node);
    if (!resolved) return false;
    const { sourceNode, sourceSlot } = resolved;

    const out = sourceNode?.outputs?.[sourceSlot];
    const wfFromOutput = _tryParseWorkflowData(out?._data) || _tryParseWorkflowData(out?.value);
    const wfFromBuilderCache = _tryParseWorkflowData(sourceNode?.properties?.we_extracted_cache);
    const wf = wfFromOutput || wfFromBuilderCache;
    if (!wf) return false;

    const inferred = _inferInfoFromWorkflowData(node, wf);
    if (!inferred) return false;
    node._applyRecipeModelLoaderInfo(inferred);
    return true;
}

function _defaultInfoFromWidgets(node) {
    const selected = _selectedModelSlot(node);
    return {
        model: selected,
        model_name: "",
        loader_type: "",
        model_kind: "",
        vae_name: "",
        clip_names: [],
        status: "idle",
        error_message: "",
        missing_name: "",
        uses_checkpoint_vae: false,
        uses_checkpoint_clip: false,
    };
}

app.registerExtension({
    name: "PromptManager.RecipeModelLoader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RecipeModelLoader") return;

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origOnExecuted?.apply(this, arguments);
            const info = message?.workflow_model_loader_info?.[0];
            if (!info) return;
            this._applyRecipeModelLoaderInfo(info);
        };

        nodeType.prototype._applyRecipeModelLoaderInfo = function (info) {
            const selected = _sanitizeChoice(info.model, MODEL_CHOICES, "model_a");
            this._recipeModelLoaderInfo = {
                model: selected,
                model_name: info.model_name || "",
                loader_type: info.loader_type || "",
                model_kind: info.model_kind || "",
                vae_name: info.vae_name || "",
                clip_names: Array.isArray(info.clip_names) ? info.clip_names : [],
                status: info.status || "ok",
                error_message: info.error_message || "",
                missing_name: info.missing_name || "",
                uses_checkpoint_vae: !!info.uses_checkpoint_vae,
                uses_checkpoint_clip: !!info.uses_checkpoint_clip,
            };
            this._ensureRecipeModelInfoWidget();
            this.setDirtyCanvas(true, true);
        };

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);
            _restoreOrNormalizeWidgetState(this);
            if (!this._recipeModelLoaderInfo) {
                this._recipeModelLoaderInfo = _defaultInfoFromWidgets(this);
            }
            this._ensureRecipeModelInfoWidget();
            _refreshInfoFromUpstream(this);
            this.setDirtyCanvas(true, true);
            return r;
        };

        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = origOnConnectionsChange?.apply(this, arguments);
            _refreshInfoFromUpstream(this);
            this.setDirtyCanvas(true, true);
            return r;
        };

        const origOnWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name, value, oldValue, w) {
            const r = origOnWidgetChanged?.apply(this, arguments);
            if (name === "model" || name === "weight_dtype") {
                _restoreOrNormalizeWidgetState(this);
                _refreshInfoFromUpstream(this);
                this.setDirtyCanvas(true, true);
            }
            return r;
        };

        const origOnSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            origOnSerialize?.apply(this, arguments);
            const ws = _readWidgetState(this);
            this.properties = this.properties || {};
            this.properties.wfml_model = ws.model;
            this.properties.wfml_weight_dtype = ws.weight_dtype;
            if (this._recipeModelLoaderInfo) {
                o._recipeModelLoaderInfo = this._recipeModelLoaderInfo;
                o._workflowModelLoaderInfo = this._recipeModelLoaderInfo;
            }
            if (o) {
                o.properties = o.properties || {};
                o.properties.wfml_model = ws.model;
                o.properties.wfml_weight_dtype = ws.weight_dtype;
            }
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            origOnConfigure?.apply(this, arguments);
            _restoreOrNormalizeWidgetState(this);
            const state = info?._recipeModelLoaderInfo || info?._workflowModelLoaderInfo;
            if (state) {
                requestAnimationFrame(() => {
                    this._applyRecipeModelLoaderInfo(state);
                    _refreshInfoFromUpstream(this);
                });
            } else {
                requestAnimationFrame(() => {
                    this._recipeModelLoaderInfo = _defaultInfoFromWidgets(this);
                    this._ensureRecipeModelInfoWidget();
                    _refreshInfoFromUpstream(this);
                    this.setDirtyCanvas(true, true);
                });
            }
        };

        nodeType.prototype._ensureRecipeModelInfoWidget = function () {
            let widget = this.widgets?.find(w => w.name === "_recipe_model_loader_info");
            if (!widget) {
                widget = {
                    name: "_recipe_model_loader_info",
                    type: "custom",
                    y: 0,
                    computeSize: () => {
                        const rows = _buildRows(this._recipeModelLoaderInfo || {});
                        const h = Math.max(26, (rows.length * 22) + 6);
                        return [0, h];
                    },
                    draw: function (ctx, node, widgetWidth, y) {
                        const info = node._recipeModelLoaderInfo;
                        if (!info) return;

                        const rows = _buildRows(info);

                        ctx.save();
                        ctx.textAlign = "left";
                        ctx.textBaseline = "middle";
                        for (let i = 0; i < rows.length; i++) {
                            const rowY = y + 12 + (i * 22);
                            const row = rows[i];
                            _drawPillRow(ctx, row, rowY, widgetWidth);
                        }
                        ctx.restore();
                    },
                };
                if (!this.widgets) this.widgets = [];
                this.widgets.push(widget);
            }
            this.computeSize();
        };
    },
});
