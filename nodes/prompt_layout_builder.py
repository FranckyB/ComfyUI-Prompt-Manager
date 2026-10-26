"""
Prompt Layout Builder - draw a MOSS-Image layout (scene prompt + boxed regions) or load one from an LLM.

The editor stores boxes as canvas fractions {x, y, w, h, type, text, desc, palette} in the layout_data widget.
The prompt output is the native MOSS-Image layout JSON (bbox in pixels from the canvas centre, yxyx), the same
format as MOSS-Image Layout Prompt. import_json accepts that format, percent or 0-1000 yxyx layouts written by
an LLM, and the elements list from Create Bounding Boxes. The two converter nodes translate layouts to and from the
core BOUNDING_BOX / elements types so the builder can be chained with other box nodes.
"""
import json
import math
import re

import numpy as np
from PIL import Image

import comfy.utils
import nodes
from comfy_extras.nodes_bounding_boxes import (
    boxes_from_input,
    build_elements,
    fractions_to_pixels,
    pixels_to_fractions,
    render_preview,
)
from comfy_extras.nodes_resolution import ASPECT_RATIOS

from ..py.workflow_data_utils import build_v2_recipe_data_from_prompt


REF_RATIO = "Use Reference Ratio"
# compose_data key holding the editor state, next to Prompt Composer's "prompt_composer".
LAYOUT_RECIPE_KEY = "prompt_layout"
SIZE_MULTIPLE = 16
CANVASES = ("single", "multi_panel", "collage")
BOX_TYPES = ("object", "character", "text")
BOX_PALETTE_MAX = 8
SUMMARY_PALETTE_MAX = 16
HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")
RELATIVE_SPACES = {"percent": 100.0, "relative_1000": 1000.0}


def _resolution(aspect_ratio, megapixels, ref_size):
    if aspect_ratio != REF_RATIO:
        w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    else:
        w_ratio, h_ratio = ref_size or (1, 1)
    scale = math.sqrt(megapixels * 1024 * 1024 / (w_ratio * h_ratio))
    return tuple(max(SIZE_MULTIPLE, round(r * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE) for r in (w_ratio, h_ratio))


def _strs(value):
    if isinstance(value, str):
        value = re.split(r"[,\n]", value)
    return [str(v).strip() for v in (value if isinstance(value, list) else []) if str(v).strip()]


def _palette(value, limit):
    return [c.upper() for c in _strs(value) if HEX_COLOR.fullmatch(c)][:limit]


def _clean_box(raw):
    x, y, w, h = (float(raw.get(k) or 0.0) for k in ("x", "y", "w", "h"))
    x1, x2 = sorted((x, x + w))
    y1, y2 = sorted((y, y + h))
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2))
    box_type = raw.get("type") if raw.get("type") in BOX_TYPES else "object"
    box = {
        "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
        "type": box_type,
        "text": str(raw.get("text") or "") if box_type == "text" else "",
        "desc": str(raw.get("desc") or ""),
        "palette": _palette(raw.get("palette"), BOX_PALETTE_MAX),
    }
    ref = raw.get("prompt_ref")
    if isinstance(ref, dict) and ref.get("name"):
        box["prompt_ref"] = {"category": str(ref.get("category") or ""), "name": str(ref["name"])}
    return box


def _find_layout(text):
    # LLM output may wrap the JSON in prose or a code fence: take the first layout-shaped value.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[{\[]", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("items"), list):
            return value
        if isinstance(value, list) and value and all(isinstance(v, dict) and "bbox" in v and "type" in v for v in value):
            return {"items": value, "bbox_coord_space": "relative_1000"}
    raise ValueError('Prompt Layout Builder: import_json has no layout (an object with "items", or a list of elements).')


def _import_boxes(layout, width, height):
    space = str(layout.get("bbox_coord_space") or "abs_centralized")
    if space != "abs_centralized" and space not in RELATIVE_SPACES:
        raise ValueError(f"Prompt Layout Builder: unsupported bbox_coord_space {space!r} "
                         "(expected percent, relative_1000 or abs_centralized).")
    if str(layout.get("bbox_format") or "yxyx") != "yxyx":
        raise ValueError('Prompt Layout Builder: bbox_format must be "yxyx" (top, left, bottom, right).')
    boxes = []
    for index, item in enumerate(layout["items"], 1):
        if not isinstance(item, dict):
            continue
        try:
            top, left, bottom, right = (float(v) for v in item.get("bbox"))
        except (TypeError, ValueError):
            raise ValueError(f"Prompt Layout Builder: item {index} needs a bbox [top, left, bottom, right].")
        if space == "abs_centralized":
            top, bottom = ((v + height // 2) / height for v in (top, bottom))
            left, right = ((v + width // 2) / width for v in (left, right))
        else:
            top, left, bottom, right = (v / RELATIVE_SPACES[space] for v in (top, left, bottom, right))
        boxes.append(_clean_box({
            "x": left, "y": top, "w": right - left, "h": bottom - top,
            "type": item.get("label", item.get("type")),
            "text": item.get("text"),
            "desc": item.get("caption", item.get("desc")),
            "palette": item.get("palette", item.get("color_palette")),
        }))
    return boxes


def _moss_bbox(box, width, height):
    # Same rounding as MOSS-Image Layout Prompt: pixel edges enclosing the box, origin at the canvas centre.
    hx, hy = width // 2, height // 2
    return [math.floor(round(box["y"] * height, 9)) - hy, math.floor(round(box["x"] * width, 9)) - hx,
            math.ceil(round((box["y"] + box["h"]) * height, 9)) - hy, math.ceil(round((box["x"] + box["w"]) * width, 9)) - hx]


def _layout_prompt(prompt, meta, canvas, palette, boxes, width, height):
    items = []
    for box in boxes:
        item = {"label": box["type"], "caption": box["desc"]}
        if box["type"] == "text":
            item["text"] = box["text"]
        if box["palette"]:
            item["palette"] = box["palette"]
        item["bbox"] = _moss_bbox(box, width, height)
        items.append(item)
    if not items:
        return prompt
    summary = {}
    if meta:
        summary["meta"] = meta
    if canvas in CANVASES:
        summary["canvas"] = canvas
    summary["caption"] = prompt.strip()
    if palette:
        summary["palette"] = palette
    layout = {"summary": summary, "items": items, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def _fit_background(image, width, height):
    samples = image[:1, :, :, :3].movedim(-1, 1)
    return comfy.utils.common_upscale(samples, width, height, "lanczos", "center").movedim(1, -1)


def _compose_data(layout_prompt, state, width, height):
    data = build_v2_recipe_data_from_prompt(prompt_text=layout_prompt, source="PromptLayoutBuilder")
    data["models"]["model_a"]["resolution"].update(width=width, height=height)
    data[LAYOUT_RECIPE_KEY] = {
        **state,
        "width": width,
        "height": height,
        "composer_prompts": [
            {"box": index, **box["prompt_ref"]} for index, box in enumerate(state["boxes"], 1) if box.get("prompt_ref")
        ],
    }
    return data


def _pixel_frame(boxes, width, height):
    # Pixel boxes carrying the metadata Create Bounding Boxes and Vantage read back.
    frame = []
    for box in boxes:
        metadata = {"type": "text" if box["type"] == "text" else "obj", "text": box["text"], "desc": box["desc"],
                    "palette": box["palette"]}
        if box["type"] == "character":
            metadata["is_character"] = True
        frame.append({**fractions_to_pixels(box, width, height), "metadata": metadata})
    return [frame] if frame else []


def _percent_item(box):
    item = {"label": box["type"], "caption": box["desc"]}
    if box["type"] == "text":
        item["text"] = box["text"]
    if box["palette"]:
        item["palette"] = box["palette"]
    item["bbox"] = [round(v * 100, 1) for v in (box["y"], box["x"], box["y"] + box["h"], box["x"] + box["w"])]
    return item


_SIZE_INPUT = {"default": 1024, "min": 16, "max": 16384, "step": 8,
               "tooltip": "Canvas size the pixel boxes refer to, e.g. the layout builder's width/height outputs."}


class PromptLayoutToBBoxes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout_json": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Layout JSON: MOSS-native, percent or 0-1000 boxes (same formats as import_json).",
                }),
                "width": ("INT", _SIZE_INPUT),
                "height": ("INT", _SIZE_INPUT),
            },
        }

    RETURN_TYPES = ("BOUNDING_BOX", "ARRAY")
    RETURN_NAMES = ("bboxes", "elements")
    FUNCTION = "convert"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Convert a layout JSON to pixel bounding boxes and Create Bounding Boxes elements."

    def convert(self, layout_json, width, height):
        boxes = _import_boxes(_find_layout(layout_json), width, height)
        return (_pixel_frame(boxes, width, height), build_elements(boxes))


class BBoxesToPromptLayout:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": ("BOUNDING_BOX,ARRAY,STRING", {
                    "forceInput": True,
                    "tooltip": "Bounding boxes, Create Bounding Boxes elements, or their JSON.",
                }),
                "width": ("INT", _SIZE_INPUT),
                "height": ("INT", _SIZE_INPUT),
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False,
                    "tooltip": "Optional scene description, written as summary.caption.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("layout_json",)
    FUNCTION = "convert"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = "Convert bounding boxes or elements to a percent layout JSON for the layout builder's import_json."

    def convert(self, bboxes, width, height, prompt=""):
        items = []
        for raw in boxes_from_input(bboxes, width, height):
            metadata = raw.get("metadata") or {}
            box_type = "text" if metadata.get("type") == "text" else "character" if metadata.get("is_character") else "object"
            items.append(_percent_item(_clean_box({
                **pixels_to_fractions(raw, width, height),
                "type": box_type,
                "text": metadata.get("text"),
                "desc": metadata.get("desc"),
                "palette": metadata.get("palette"),
            })))
        layout = {"summary": {"caption": prompt.strip()}} if prompt.strip() else {}
        layout.update(items=items, bbox_coord_space="percent", bbox_format="yxyx")
        return (json.dumps(layout, ensure_ascii=False, indent=2),)


class PromptLayoutBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ([REF_RATIO, *(ratio.value for ratio in ASPECT_RATIOS)], {
                    "default": "1:1 (Square)",
                    "tooltip": "Canvas aspect ratio. 'Use Reference Ratio' follows a layout restored from compose_data, "
                               "otherwise the connected image.",
                }),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1,
                    "tooltip": "Target canvas size; width and height are rounded to multiples of 16.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Scene description (summary caption)",
                    "dynamicPrompts": False,
                    "tooltip": "Description of the whole image and its style.",
                }),
                "meta": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated style tags for summary.meta, e.g. \"poster, graphic design\" (blank = omitted).",
                }),
                "canvas": (["none", *CANVASES], {
                    "default": "none",
                    "tooltip": "summary.canvas: single image, multi_panel or collage (none = omitted).",
                }),
                "use_import_json": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Read the layout from import_json and lock the editor. Turn off to edit the imported boxes.",
                }),
                "layout_data": ("STRING", {"default": "{}", "tooltip": "Editor boxes (managed by the node UI)."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Background for the editor; also sets the ratio with 'Use Reference Ratio'."}),
                "import_json": ("STRING,COMPOSE_DATA", {
                    "forceInput": True,
                    "lazy": True,
                    "tooltip": "Layout JSON (e.g. from Prompt Generator: percent, 0-1000 or MOSS-native boxes) or a saved "
                               "layout's compose_data from Composition Manager. Only evaluated when use_import_json is on.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "INT", "INT", "COMPOSE_DATA")
    RETURN_NAMES = ("prompt", "preview", "width", "height", "compose_data")
    FUNCTION = "build"
    CATEGORY = "Prompt Manager"
    OUTPUT_NODE = True
    DESCRIPTION = "Draw MOSS-Image layout boxes over a scene prompt, or load them from an LLM layout JSON."

    def check_lazy_status(self, use_import_json, **kwargs):
        return ["import_json"] if use_import_json else []

    def build(self, aspect_ratio, megapixels, prompt, meta, canvas, use_import_json, layout_data,
              image=None, import_json=None):
        saved = None
        if use_import_json and isinstance(import_json, dict):
            saved = import_json.get(LAYOUT_RECIPE_KEY)
            if not isinstance(saved, dict):
                raise ValueError("Prompt Layout Builder: this compose_data has no layout (it is Prompt Composer data).")
        text_import = use_import_json and isinstance(import_json, str) and import_json.strip()
        if saved and saved.get("width") and saved.get("height"):
            ref_size = (int(saved["width"]), int(saved["height"]))
        elif image is not None:
            ref_size = (int(image.shape[2]), int(image.shape[1]))
        else:
            ref_size = None
        width, height = _resolution(aspect_ratio, megapixels, ref_size)

        state = saved if saved else json.loads(layout_data or "{}")
        boxes = [_clean_box(box) for box in state.get("boxes", []) if isinstance(box, dict)]
        palette = _palette(state.get("palette"), SUMMARY_PALETTE_MAX)
        meta_tags = _strs(meta)
        ui = {}

        if saved:
            prompt = str(saved.get("prompt") or "")
            meta_tags = _strs(saved.get("meta"))
            canvas = saved.get("canvas") if saved.get("canvas") in CANVASES else "none"
        elif text_import:
            layout = _find_layout(import_json)
            boxes = _import_boxes(layout, width, height)
            summary = layout.get("summary")
            if isinstance(summary, dict):
                prompt = str(summary.get("caption") or "")
                meta_tags = _strs(summary.get("meta"))
                canvas = summary.get("canvas") if summary.get("canvas") in CANVASES else "none"
                palette = _palette(summary.get("palette"), SUMMARY_PALETTE_MAX)
        if saved or text_import:
            ui["pm_layout_import"] = [{
                "boxes": boxes, "palette": palette, "prompt": prompt, "meta": ", ".join(meta_tags), "canvas": canvas,
            }]

        ui["pm_layout_ref_size"] = list(ref_size) if ref_size else []
        if image is None:
            preview_bg = None
            ui["pm_layout_background"] = []
        else:
            background = _fit_background(image, width, height)
            preview_bg = Image.fromarray((background[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            ui["pm_layout_background"] = nodes.PreviewImage().save_images(background, "pm.layout.background")["ui"]["images"]

        layout_prompt = _layout_prompt(prompt, meta_tags, canvas, palette, boxes, width, height)
        editor_state = {
            "boxes": boxes, "palette": palette, "prompt": prompt, "meta": meta_tags, "canvas": canvas,
            "aspect_ratio": aspect_ratio, "megapixels": megapixels,
        }
        return {
            "ui": ui,
            "result": (
                layout_prompt,
                render_preview(boxes, width, height, preview_bg),
                width,
                height,
                _compose_data(layout_prompt, editor_state, width, height),
            ),
        }
