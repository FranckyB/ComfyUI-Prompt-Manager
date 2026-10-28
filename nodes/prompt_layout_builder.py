"""
Prompt Layout Builder - draw a MOSS-Image layout (scene prompt + boxed regions) or load one from an LLM.

The editor stores boxes as canvas fractions {x, y, w, h, type, text, desc, palette} in the layout_data widget.
The prompt output follows output_format: the native MOSS-Image layout JSON (bbox in pixels from the canvas centre,
yxyx, the same format as MOSS-Image Layout Prompt), an Ideogram 4 JSON caption (0-1000 yxyx elements, as Build JSON
Prompt (Ideogram)) or the Krea 2 grounding lines of jimmycarter/krea2-turbo-bbox (x-first 0-1000 boxes).
import_json accepts the MOSS
format, percent or 0-1000 yxyx layouts written by
an LLM, Ideogram 4 captions, Krea 2 grounding lines, and the elements list from Create Bounding Boxes. The two converter nodes translate layouts to and from the
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
# manager_data key holding the editor state, next to Prompt Composer's "prompt_composer".
LAYOUT_RECIPE_KEY = "prompt_layout"
SIZE_MULTIPLE = 16
CANVASES = ("single", "multi_panel", "collage")
BOX_TYPES = ("object", "character", "text", "panel")
BOX_PALETTE_MAX = 8
SUMMARY_PALETTE_MAX = 16
HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")
RELATIVE_SPACES = {"percent": 100.0, "relative_1000": 1000.0}
OUTPUT_FORMATS = ("MOSS", "Ideogram", "Krea 2")
# Krea 2 character tags: person, anime character, furry / anthro character.
KREA_CHARACTER_TAGS = ("pe", "ac", "fc")
KREA_CHARACTER_ID = re.compile(r"[a-z0-9_]+")


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
    if box_type == "character":
        if raw.get("krea_tag") in KREA_CHARACTER_TAGS:
            box["krea_tag"] = raw["krea_tag"]
        char_id = str(raw.get("char_id") or "").strip().lower()
        if KREA_CHARACTER_ID.fullmatch(char_id):
            box["char_id"] = char_id
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


KREA_ELEMENT = re.compile(r"(p|o|t|pe|ac|fc)(?::([a-z0-9_]+))?\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]\s*(.*)")
KREA_TEXT = re.compile(r'"((?:[^"\\]|\\.)*)"\s*(.*)')
KREA_BOX_TYPES = {"p": "panel", "o": "object", "t": "text", "pe": "character", "ac": "character", "fc": "character"}


def _parse_krea(text):
    # Krea 2 grounding lines (krea2-turbo-bbox); None when the text has no element, @ or ~ line.
    fence = re.search(r"```[^\n]*\n(.*?)```", text, re.S)
    lines = [line.strip() for line in (fence.group(1) if fence else text).splitlines() if line.strip()]
    summary, fields, boxes, grounded = "", {}, [], False
    for line in lines:
        element = KREA_ELEMENT.fullmatch(line)
        if element:
            tag, char_id, rest = element.group(1), element.group(2), element.group(7)
            x0, y0, x1, y1 = (float(element.group(i)) / 1000 for i in range(3, 7))
            box = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "type": KREA_BOX_TYPES[tag], "desc": rest}
            if tag == "t":
                quoted = KREA_TEXT.match(rest)
                if quoted:
                    box["text"] = re.sub(r"\\(.)", lambda m: "\n" if m.group(1) == "n" else m.group(1), quoted.group(1))
                    box["desc"] = quoted.group(2)
            if box["type"] == "character":
                box.update(krea_tag=tag, char_id=char_id)
            boxes.append(_clean_box(box))
            grounded = True
        elif line.startswith("@"):
            head, sep, tail = line[1:].rpartition(";")
            fields["style"], fields["medium"] = (head.strip(), tail.strip()) if sep else (tail.strip(), "")
            grounded = True
        elif line.startswith("~"):
            fields["background"] = line[1:].strip()
            grounded = True
        elif not summary and not grounded:
            summary = line
    return {"prompt": summary, "krea": fields, "boxes": boxes} if grounded else None


def _parse_ideogram(text):
    # Ideogram 4 JSON caption (Build JSON Prompt (Ideogram)); None when the text has none.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            caption, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(caption, dict) and isinstance(caption.get("compositional_deconstruction"), dict):
            break
    else:
        return None
    scene = caption["compositional_deconstruction"]
    style = caption.get("style_description") if isinstance(caption.get("style_description"), dict) else {}
    text_of = lambda value: str(value or "").strip()
    fields = {"background": text_of(scene.get("background"))}
    if style:
        kind = "art_style" if "art_style" in style else "photo" if "photo" in style else (
            "photo" if text_of(style.get("medium")).lower() == "photograph" else "art_style")
        fields.update(style=kind, style_detail=text_of(style.get(kind)), aesthetics=text_of(style.get("aesthetics")),
                      lighting=text_of(style.get("lighting")), medium=text_of(style.get("medium")))
    elements = [e for e in scene.get("elements") or [] if isinstance(e, dict)]
    return {
        "prompt": text_of(caption.get("high_level_description")),
        "ideogram": fields,
        "palette": _palette(style.get("color_palette"), SUMMARY_PALETTE_MAX),
        "boxes": _import_boxes({"items": elements, "bbox_coord_space": "relative_1000"}, 1, 1),
    }

def _parse_text_layout(text, width, height, label):
    parsed = _parse_krea(text) or _parse_ideogram(text)
    if parsed:
        return parsed
    try:
        layout = _find_layout(text)
    except ValueError:
        received = " ".join(text.split())
        raise ValueError(f"{label} has no layout. Expected layout JSON (an object with \"items\" or a list of elements), "
                         "an Ideogram 4 caption or Krea 2 grounding lines; check the Prompt Generator system prompt. "
                         f"Received: {received[:200]!r}{'...' if len(received) > 200 else ''}") from None
    return {"layout": layout, "boxes": _import_boxes(layout, width, height)}


def _boxes_from_bboxes(bboxes, width, height):
    boxes = []
    for raw in boxes_from_input(bboxes, width, height):
        metadata = raw.get("metadata") or {}
        box_type = "text" if metadata.get("type") == "text" else "character" if metadata.get("is_character") else "object"
        boxes.append(_clean_box({
            **pixels_to_fractions(raw, width, height),
            "type": box_type,
            "text": metadata.get("text"),
            "desc": metadata.get("desc"),
            "palette": metadata.get("palette"),
        }))
    return boxes


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
        item = {"label": "object" if box["type"] == "panel" else box["type"], "caption": box["desc"]}
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


def _ideogram_prompt(prompt, palette, boxes, fields):
    # Same caption as Ideogram 4 Prompt Builder KJ: compact JSON, and every style key in order once a style is set.
    fields = fields if isinstance(fields, dict) else {}
    text = lambda key: str(fields.get(key) or "")
    caption = {}
    if prompt.strip():
        caption["high_level_description"] = prompt.strip()
    style = fields.get("style")
    if style in ("photo", "art_style"):
        style_description = {"aesthetics": text("aesthetics"), "lighting": text("lighting")}
        if style == "photo":
            style_description.update(photo=text("style_detail"), medium=text("medium"))
        else:
            style_description.update(medium=text("medium"), art_style=text("style_detail"))
        if palette:
            style_description["color_palette"] = palette
        caption["style_description"] = style_description
    caption["compositional_deconstruction"] = {"background": text("background"), "elements": build_elements(boxes)}
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))


def _format_prompt(output_format, prompt, meta, canvas, palette, boxes, width, height, ideogram=None, krea=None):
    if output_format == "Ideogram":
        return _ideogram_prompt(prompt, palette, boxes, ideogram)
    if output_format == "Krea 2":
        return _krea_prompt(prompt, boxes, krea)
    if not boxes:
        return prompt
    return _layout_prompt(prompt, meta, canvas, palette, boxes, width, height)


def _one_line(text):
    return " ".join(str(text or "").split())


def _krea_prompt(prompt, boxes, fields):
    # Grounding DSL from jimmycarter/krea2-turbo-bbox PROMPTING.md: summary, @style; medium, ~background, one line per box.
    fields = fields if isinstance(fields, dict) else {}
    lines = []
    if _one_line(prompt):
        lines.append(_one_line(prompt))
    style_line = "; ".join(part for part in (_one_line(fields.get("style")), _one_line(fields.get("medium"))) if part)
    if style_line:
        lines.append("@" + style_line)
    if _one_line(fields.get("background")):
        lines.append("~" + _one_line(fields.get("background")))
    # Panels first, as in the training data; the rest keep the editor order.
    for box in sorted(boxes, key=lambda b: b["type"] != "panel"):
        bbox = ",".join(str(round(v * 1000)) for v in (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]))
        if box["type"] == "character":
            head = box.get("krea_tag", "pe") + (f":{box['char_id']}" if box.get("char_id") else "")
        else:
            head = {"panel": "p", "text": "t"}.get(box["type"], "o")
        line = f"{head}[{bbox}]"
        if box["type"] == "text":
            escaped = box["text"].replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            line += f'"{escaped}"'
        if _one_line(box["desc"]):
            line += " " + _one_line(box["desc"])
        lines.append(line)
    return "\n".join(lines)


def _fit_background(image, width, height):
    samples = image[:1, :, :, :3].movedim(-1, 1)
    return comfy.utils.common_upscale(samples, width, height, "lanczos", "center").movedim(1, -1)


def _manager_data(layout_prompt, state, width, height):
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


class BboxConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", _SIZE_INPUT),
                "height": ("INT", _SIZE_INPUT),
                "use_bboxes": ("BOOLEAN", {
                    "default": False,
                    "label_on": "bboxes",
                    "label_off": "layout_json",
                    "tooltip": "Input to convert when both are connected (shown only then).",
                }),
            },
            "optional": {
                "layout_json": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Layout JSON (MOSS-native, percent or 0-1000 boxes), an Ideogram 4 caption or Krea 2 "
                               "grounding lines. Passed through to the layout_json output.",
                }),
                "bboxes": ("BOUNDING_BOX,ARRAY,STRING", {
                    "forceInput": True,
                    "tooltip": "Bounding boxes, Create Bounding Boxes elements, or their JSON.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "BOUNDING_BOX", "ARRAY")
    RETURN_NAMES = ("layout_json", "bboxes", "elements")
    FUNCTION = "convert"
    CATEGORY = "Prompt Manager"
    DESCRIPTION = ("Convert between a layout JSON and bounding boxes. Outputs the layout JSON (percent boxes when "
                   "converting bboxes), pixel bounding boxes and Create Bounding Boxes elements.")

    def convert(self, width, height, use_bboxes=False, layout_json=None, bboxes=None):
        if bboxes is not None and (use_bboxes or layout_json is None):
            boxes = _boxes_from_bboxes(bboxes, width, height)
            layout_json = json.dumps({"items": [_percent_item(box) for box in boxes], "bbox_coord_space": "percent",
                                      "bbox_format": "yxyx"}, ensure_ascii=False, indent=2)
        elif layout_json is not None:
            boxes = _parse_text_layout(layout_json, width, height, "Bbox Converter: layout_json")["boxes"]
        else:
            raise ValueError("Bbox Converter: connect layout_json or bboxes.")
        return (layout_json, _pixel_frame(boxes, width, height), build_elements(boxes))


class PromptLayoutBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ([REF_RATIO, *(ratio.value for ratio in ASPECT_RATIOS)], {
                    "default": "1:1 (Square)",
                    "tooltip": "Canvas aspect ratio. 'Use Reference Ratio' follows a layout restored from manager_data, "
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
                "output_format": (list(OUTPUT_FORMATS), {
                    "default": "MOSS",
                    "tooltip": "Prompt format: MOSS-Image layout JSON, Ideogram 4 JSON caption or Krea 2 grounding lines.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Background for the editor; also sets the ratio with 'Use Reference Ratio'."}),
                "import_json": ("STRING,MANAGER_DATA", {
                    "forceInput": True,
                    "lazy": True,
                    "tooltip": "Layout JSON (e.g. from Prompt Generator: percent, 0-1000 or MOSS-native boxes), an "
                               "Ideogram 4 caption, Krea 2 grounding lines, or a saved "
                               "layout's manager_data from Library Manager. Only evaluated when use_import_json is on.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "INT", "INT", "MANAGER_DATA")
    RETURN_NAMES = ("prompt", "preview", "width", "height", "manager_data")
    FUNCTION = "build"
    CATEGORY = "Prompt Manager"
    OUTPUT_NODE = True
    DESCRIPTION = ("Draw layout boxes over a scene prompt, or load them from an LLM layout. "
                   "Outputs a MOSS-Image, Ideogram 4 or Krea 2 prompt.")

    def check_lazy_status(self, use_import_json, **kwargs):
        return ["import_json"] if use_import_json else []

    def build(self, aspect_ratio, megapixels, prompt, meta, canvas, use_import_json, layout_data,
              image=None, import_json=None, output_format="MOSS"):
        saved = None
        if use_import_json and isinstance(import_json, dict):
            saved = import_json.get(LAYOUT_RECIPE_KEY)
            if not isinstance(saved, dict):
                raise ValueError("Prompt Layout Builder: this manager_data has no layout (it comes from Prompt Manager or Prompt Composer).")
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
        ideogram = state.get("ideogram") if isinstance(state.get("ideogram"), dict) else {}
        krea = state.get("krea") if isinstance(state.get("krea"), dict) else {}
        meta_tags = _strs(meta)
        ui = {}

        if saved:
            prompt = str(saved.get("prompt") or "")
            meta_tags = _strs(saved.get("meta"))
            canvas = saved.get("canvas") if saved.get("canvas") in CANVASES else "none"
        elif text_import:
            parsed = _parse_text_layout(import_json, width, height, "Prompt Layout Builder: import_json")
            prompt, boxes = parsed.get("prompt", prompt), parsed["boxes"]
            krea = parsed.get("krea", krea)
            ideogram = parsed.get("ideogram", ideogram)
            palette = parsed.get("palette", palette)
            summary = (parsed.get("layout") or {}).get("summary")
            if isinstance(summary, dict):
                prompt = str(summary.get("caption") or "")
                meta_tags = _strs(summary.get("meta"))
                canvas = summary.get("canvas") if summary.get("canvas") in CANVASES else "none"
                palette = _palette(summary.get("palette"), SUMMARY_PALETTE_MAX)
        if saved or text_import:
            ui["pm_layout_import"] = [{
                "boxes": boxes, "palette": palette, "prompt": prompt, "meta": ", ".join(meta_tags), "canvas": canvas,
                **({"ideogram": ideogram} if ideogram and (saved or text_import) else {}),
                **({"krea": krea} if krea and (saved or text_import) else {}),
            }]

        ui["pm_layout_ref_size"] = list(ref_size) if ref_size else []
        if image is None:
            preview_bg = None
            ui["pm_layout_background"] = []
        else:
            background = _fit_background(image, width, height)
            preview_bg = Image.fromarray((background[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            ui["pm_layout_background"] = nodes.PreviewImage().save_images(background, "pm.layout.background")["ui"]["images"]

        layout_prompt = _format_prompt(output_format, prompt, meta_tags, canvas, palette, boxes, width, height, ideogram, krea)
        editor_state = {
            "boxes": boxes, "palette": palette, "prompt": prompt, "meta": meta_tags, "canvas": canvas,
            "aspect_ratio": aspect_ratio, "megapixels": megapixels, "ideogram": ideogram, "krea": krea,
        }
        return {
            "ui": ui,
            "result": (
                layout_prompt,
                render_preview(boxes, width, height, preview_bg),
                width,
                height,
                _manager_data(layout_prompt, editor_state, width, height),
            ),
        }
