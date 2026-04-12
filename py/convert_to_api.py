#!/usr/bin/env python3
"""
convert_to_api.py — Convert ComfyUI UI-format workflow JSON to API format + _map.json

Usage:
    python tools/convert_to_api.py
    # Reads from workflows/*.json, writes to workflows/api/*_api.json + *_map.json

API format: dict keyed by node ID (string), each value = {"class_type": ..., "inputs": {...}}
Map format: dict of semantic field -> [node_id_str, input_field_name]
"""

import json
import os
import sys

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)
SRC_DIR      = os.path.join(REPO_ROOT, "workflows")
DST_DIR      = os.path.join(REPO_ROOT, "workflows", "api")
os.makedirs(DST_DIR, exist_ok=True)


# ── UI → API conversion ──────────────────────────────────────────────────────

def ui_to_api(ui_data):
    """
    Convert a UI-format workflow dict to API-format.

    UI format: { nodes: [...], links: [...], ... }
      - nodes have id, type, inputs (with link IDs), outputs (with link IDs), widgets_values
    API format: { "NODE_ID": { class_type, inputs: {field: value_or_[src_id, slot]} } }

    Widget values are embedded directly; linked inputs reference the source node+slot.
    """
    nodes   = ui_data.get("nodes", [])
    links   = ui_data.get("links", [])

    # Build link map: link_id -> (src_node_id, src_slot_index)
    # Links: [link_id, src_node_id, src_slot, dst_node_id, dst_slot, type]
    link_map = {}
    for lk in links:
        if len(lk) >= 4:
            link_map[lk[0]] = (str(lk[1]), lk[2])

    api = {}
    for node in nodes:
        node_id     = str(node["id"])
        class_type  = node["type"]
        inputs_def  = node.get("inputs", [])   # [{name, type, link, widget?}]
        widgets_val = node.get("widgets_values", [])

        # We need to figure out which inputs are widget-backed vs link-backed.
        # ComfyUI heuristic:
        #   - If input has a "widget" key, it is widget-backed (value from widgets_values)
        #   - If input has a "link" key (non-null), it is link-backed
        #   - Some inputs can be either (widget overridable by link)

        api_inputs = {}
        widget_idx = 0  # index into widgets_values

        for inp in inputs_def:
            name = inp["name"]
            link = inp.get("link")
            has_widget = "widget" in inp

            if link is not None:
                # Link wins
                src = link_map.get(link)
                if src:
                    api_inputs[name] = [src[0], src[1]]
                else:
                    # Dangling link — fall back to widget value if available
                    if has_widget and widget_idx < len(widgets_val):
                        api_inputs[name] = widgets_val[widget_idx]
                        widget_idx += 1
            elif has_widget:
                if widget_idx < len(widgets_val):
                    api_inputs[name] = widgets_val[widget_idx]
                    widget_idx += 1
            # else: pure output-only input — skip

        # Any remaining widgets_values beyond named widget inputs are for
        # implicit widget-only nodes (e.g. seed control extra slot "control_after_generate")
        # We inject them by matching known extra widget names per class_type
        _inject_extra_widgets(class_type, api_inputs, widgets_val, widget_idx)

        api[node_id] = {
            "class_type": class_type,
            "inputs":     api_inputs,
        }

    return api


def _inject_extra_widgets(class_type, api_inputs, widgets_val, start_idx):
    """
    Some nodes have implicit extra widget slots not reflected in the inputs list.
    The most common: KSampler seed has a hidden 'control_after_generate' slot
    inserted after noise_seed in widgets_values, but it is NOT in the inputs
    definition — so our widget_idx counter goes off by 1 after consuming seed.

    We fully re-map widgets_values for known node types that have hidden slots,
    overwriting anything the loop may have set incorrectly.
    """
    # KSampler: [seed, control_after_generate, steps, cfg, sampler_name, scheduler, denoise]
    # inputs list has: seed, steps, cfg, sampler_name, scheduler, denoise  (no control_after_generate)
    if class_type == "KSampler" and len(widgets_val) >= 7:
        api_inputs["seed"]         = widgets_val[0]
        # widgets_val[1] = control_after_generate -- skip
        api_inputs["steps"]        = widgets_val[2]
        api_inputs["cfg"]          = widgets_val[3]
        api_inputs["sampler_name"] = widgets_val[4]
        api_inputs["scheduler"]    = widgets_val[5]
        api_inputs["denoise"]      = widgets_val[6]

    # KSamplerAdvanced: [add_noise, noise_seed, control_after_generate, steps, cfg,
    #                    sampler_name, scheduler, start_at_step, end_at_step, return_with_leftover_noise]
    elif class_type == "KSamplerAdvanced" and len(widgets_val) >= 10:
        api_inputs["add_noise"]                  = widgets_val[0]
        api_inputs["noise_seed"]                 = widgets_val[1]
        # widgets_val[2] = control_after_generate -- skip
        api_inputs["steps"]                      = widgets_val[3]
        api_inputs["cfg"]                        = widgets_val[4]
        api_inputs["sampler_name"]               = widgets_val[5]
        api_inputs["scheduler"]                  = widgets_val[6]
        api_inputs["start_at_step"]              = widgets_val[7]
        api_inputs["end_at_step"]                = widgets_val[8]
        api_inputs["return_with_leftover_noise"] = widgets_val[9]

    # RandomNoise: [noise_seed, control_after_generate]
    elif class_type == "RandomNoise" and len(widgets_val) >= 1:
        api_inputs["noise_seed"] = widgets_val[0]
        # widgets_val[1] = control_after_generate -- skip


# ── Map generation ───────────────────────────────────────────────────────────

def build_map(api, family_hint):
    """
    Build a semantic field map for the given API workflow.
    Returns a dict: { field_name: [node_id_str, input_key] | null }

    The map is used by WorkflowRenderer to patch values before execution.
    """
    m = {}

    for nid, node in api.items():
        ct = node["class_type"]
        inp = node["inputs"]

        # ── Positive / Negative prompts ──────────────────────────────────
        if ct == "CLIPTextEncode":
            # Determine pos vs neg by looking at what this feeds into
            # We'll do a two-pass approach after the loop
            pass

        # ── Checkpoint loader ─────────────────────────────────────────────
        if ct in ("CheckpointLoaderSimple", "CheckpointLoader"):
            m["checkpoint"] = [nid, "ckpt_name"]

        # ── UNET loaders ─────────────────────────────────────────────────
        if ct in ("UNETLoader", "UNETLoaderGGUF"):
            if "model_a" not in m:
                m["model_a"] = [nid, "unet_name"]
            else:
                m["model_b"] = [nid, "unet_name"]

        # ── VAE ───────────────────────────────────────────────────────────
        if ct == "VAELoader":
            m["vae"] = [nid, "vae_name"]

        # ── CLIP / text encoder ───────────────────────────────────────────
        if ct in ("CLIPLoader", "DualCLIPLoader"):
            if ct == "DualCLIPLoader":
                m["clip_1"] = [nid, "clip_name1"]
                m["clip_2"] = [nid, "clip_name2"]
            else:
                m["clip"]   = [nid, "clip_name"]

        # ── Latent size ───────────────────────────────────────────────────
        if ct in ("EmptyLatentImage", "EmptySD3LatentImage", "EmptyHunyuanLatentVideo"):
            m["latent_width"]  = [nid, "width"]
            m["latent_height"] = [nid, "height"]
            if ct == "EmptyHunyuanLatentVideo":
                m["latent_length"] = [nid, "length"]

        # ── WanImageToVideo latent (i2v) — size comes from Resize node ───
        if ct == "WanImageToVideo":
            m["wan_i2v_latent"] = [nid, "start_image"]  # image input
            # width/height fed from resize node — mapped separately

        # ── Resize node (used in i2v for size control) ────────────────────
        if ct in ("ImageResizeKJv2", "ImageScale", "ImageResize"):
            if "resize_width" not in m:
                m["resize_width"]  = [nid, "width"]
                m["resize_height"] = [nid, "height"]

        # ── KSampler (standard) ───────────────────────────────────────────
        if ct == "KSampler":
            m["ksampler_seed"]      = [nid, "seed"]
            m["ksampler_steps"]     = [nid, "steps"]
            m["ksampler_cfg"]       = [nid, "cfg"]
            m["ksampler_sampler"]   = [nid, "sampler_name"]
            m["ksampler_scheduler"] = [nid, "scheduler"]
            m["ksampler_denoise"]   = [nid, "denoise"]

        # ── KSamplerAdvanced (WAN dual) ───────────────────────────────────
        if ct == "KSamplerAdvanced":
            if "ksampler_high_seed" not in m:
                m["ksampler_high_seed"]       = [nid, "noise_seed"]
                m["ksampler_high_steps"]      = [nid, "steps"]
                m["ksampler_high_start"]      = [nid, "start_at_step"]
                m["ksampler_high_end"]        = [nid, "end_at_step"]
                m["ksampler_high_cfg"]        = [nid, "cfg"]
                m["ksampler_high_sampler"]    = [nid, "sampler_name"]
                m["ksampler_high_scheduler"]  = [nid, "scheduler"]
            else:
                m["ksampler_low_seed"]        = [nid, "noise_seed"]
                m["ksampler_low_steps"]       = [nid, "steps"]
                m["ksampler_low_start"]       = [nid, "start_at_step"]
                m["ksampler_low_end"]         = [nid, "end_at_step"]
                m["ksampler_low_cfg"]         = [nid, "cfg"]
                m["ksampler_low_sampler"]     = [nid, "sampler_name"]
                m["ksampler_low_scheduler"]   = [nid, "scheduler"]

        # ── Flux / custom samplers ────────────────────────────────────────
        if ct == "KSamplerSelect":
            m["flux_sampler"] = [nid, "sampler_name"]
        if ct in ("BasicScheduler",):
            m["flux_scheduler"] = [nid, "scheduler"]
            m["flux_steps"]     = [nid, "steps"]
            m["flux_denoise"]   = [nid, "denoise"]
        if ct == "Flux2Scheduler":
            m["flux2_steps"]  = [nid, "steps"]
            m["flux2_width"]  = [nid, "width"]
            m["flux2_height"] = [nid, "height"]
        if ct == "CFGGuider":
            m["flux2_cfg"] = [nid, "cfg"]
        if ct == "FluxGuidance":
            m["flux_guidance"] = [nid, "guidance"]
        if ct == "RandomNoise":
            m["flux_seed"] = [nid, "noise_seed"]

        # ── PromptApplyLora ───────────────────────────────────────────────
        if ct == "PromptApplyLora":
            if "lora_stack_a" not in m:
                m["lora_stack_a"] = [nid, "lora_stack_text"]
            else:
                m["lora_stack_b"] = [nid, "lora_stack_text"]

        # ── VAEDecode (output image) ───────────────────────────────────────
        if ct == "VAEDecode":
            m["vae_decode"] = [nid, None]   # source of output IMAGE tensor

        # ── SaveImage → mark for removal ─────────────────────────────────
        if ct == "SaveImage":
            m.setdefault("_save_nodes", []).append(nid)

        # ── ModelSamplingSD3 (WAN shift) ──────────────────────────────────
        if ct == "ModelSamplingSD3":
            if "wan_shift_a" not in m:
                m["wan_shift_a"] = [nid, "shift"]
            else:
                m["wan_shift_b"] = [nid, "shift"]

    # ── Two-pass: resolve CLIPTextEncode pos/neg by tracing links ────────
    _resolve_clip_text_encode(api, m)

    return m


def _resolve_clip_text_encode(api, m):
    """
    For CLIPTextEncode nodes, determine which is positive and which is negative
    by checking what node/slot they feed into.
    """
    # Build reverse link map: (dst_node_id, dst_slot) -> src_node_id
    # We look at each node's inputs and find CLIPTextEncode sources
    text_encode_nodes = [nid for nid, n in api.items() if n["class_type"] == "CLIPTextEncode"]
    if not text_encode_nodes:
        return

    # For each CLIPTextEncode, check what uses its output
    # In API format, inputs reference sources as [src_node_id, src_slot]
    pos_node = None
    neg_node = None

    for nid, node in api.items():
        for field, val in node["inputs"].items():
            if isinstance(val, list) and len(val) == 2:
                src_nid = str(val[0])
                if src_nid in text_encode_nodes:
                    if field == "positive" and pos_node is None:
                        pos_node = src_nid
                    elif field == "negative" and neg_node is None:
                        neg_node = src_nid
                    # WanImageToVideo also has positive/negative
                    elif field == "positive":
                        pass
                    elif field == "negative":
                        pass

    # Also handle FluxGuidance — positive goes into conditioning input
    if pos_node is None:
        for nid, node in api.items():
            if node["class_type"] == "FluxGuidance":
                for field, val in node["inputs"].items():
                    if field == "conditioning" and isinstance(val, list):
                        src = str(val[0])
                        if src in text_encode_nodes:
                            pos_node = src
                            break

    # For ConditioningZeroOut — that's the negative for Flux
    for nid, node in api.items():
        if node["class_type"] == "ConditioningZeroOut":
            neg_node = nid  # The output of ConditioningZeroOut is the negative

    if pos_node:
        m["positive_prompt"] = [pos_node, "text"]
    if neg_node and api.get(neg_node, {}).get("class_type") == "CLIPTextEncode":
        m["negative_prompt"] = [neg_node, "text"]
    elif neg_node and api.get(neg_node, {}).get("class_type") == "ConditioningZeroOut":
        # For Flux, negative is zero — we patch the CLIPTextEncode feeding into it
        for nid, node in api.items():
            if node["class_type"] == "ConditioningZeroOut":
                for field, val in node["inputs"].items():
                    if isinstance(val, list):
                        src = str(val[0])
                        if api.get(src, {}).get("class_type") == "CLIPTextEncode":
                            m["negative_prompt"] = [src, "text"]
                            break

    # Fallback: if only one CLIPTextEncode, it's positive
    if pos_node is None and len(text_encode_nodes) >= 1:
        m["positive_prompt"] = [text_encode_nodes[0], "text"]
    if neg_node is None and len(text_encode_nodes) >= 2:
        m["negative_prompt"] = [text_encode_nodes[1], "text"]


# ── Strip SaveImage nodes from API ───────────────────────────────────────────

def strip_save_nodes(api, map_data):
    """Remove SaveImage nodes from the API dict (not needed for in-process execution)."""
    save_ids = set(map_data.pop("_save_nodes", []))
    for nid in save_ids:
        api.pop(nid, None)
    # Also remove any references to these nodes from other nodes' inputs
    for node in api.values():
        node["inputs"] = {
            k: v for k, v in node["inputs"].items()
            if not (isinstance(v, list) and len(v) == 2 and str(v[0]) in save_ids)
        }


# ── Main ─────────────────────────────────────────────────────────────────────

FAMILY_HINTS = {
    "sdxl":          "sdxl",
    "flux_1":        "flux1",
    "flux_2":        "flux2",
    "z_image":       "zimage",
    "wan_image":     "wan_image",
    "wan_video_t2v": "wan_video_t2v",
    "wan_video_i2v": "wan_video_i2v",
    "qwen_image":    "qwen_image",
}

def convert_file(src_path, dst_api_path, dst_map_path, family_hint):
    with open(src_path, "r", encoding="utf-8") as f:
        ui_data = json.load(f)

    api      = ui_to_api(ui_data)
    map_data = build_map(api, family_hint)
    strip_save_nodes(api, map_data)

    with open(dst_api_path, "w", encoding="utf-8") as f:
        json.dump(api, f, indent=2)

    with open(dst_map_path, "w", encoding="utf-8") as f:
        json.dump(map_data, f, indent=2)

    print(f"  {os.path.basename(src_path)} -> {os.path.basename(dst_api_path)}  (map: {os.path.basename(dst_map_path)})")
    return api, map_data


def main():
    print(f"Source:      {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    print()

    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]  # strip .json
        if stem in FAMILY_HINTS:
            hint = FAMILY_HINTS[stem]
        else:
            hint = stem

        src_path     = os.path.join(SRC_DIR, fname)
        dst_api_path = os.path.join(DST_DIR, f"{stem}_api.json")
        dst_map_path = os.path.join(DST_DIR, f"{stem}_map.json")

        try:
            api, map_data = convert_file(src_path, dst_api_path, dst_map_path, hint)
            # Print key mappings for review
            for k, v in sorted(map_data.items()):
                print(f"    {k}: {v}")
            print()
        except Exception as e:
            print(f"  ERROR converting {fname}: {e}")
            import traceback
            traceback.print_exc()

    print("Done.")


if __name__ == "__main__":
    main()
