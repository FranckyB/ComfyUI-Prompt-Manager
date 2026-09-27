"""Move input files without embedded generation metadata into input/no_workflow.

Run this from the ComfyUI root directory.

Behavior:
- scans only files directly inside ./input
- does not recurse into subfolders
- keeps files that contain embedded prompt/workflow metadata
- moves files with no detectable metadata into ./input/no_workflow
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi"}
JSON_EXTS = {".json"}
SUPPORTED_EXTS = IMAGE_EXTS | VIDEO_EXTS | JSON_EXTS


def _json_load_maybe(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _has_workflow_payload(value) -> bool:
    data = _json_load_maybe(value)
    if isinstance(data, dict):
        if isinstance(data.get("nodes"), list):
            return True
        if isinstance(data.get("workflow"), dict):
            return _has_workflow_payload(data.get("workflow"))
    return False


def _has_prompt_payload(value) -> bool:
    data = _json_load_maybe(value)
    if isinstance(data, dict):
        if any(isinstance(v, dict) and "class_type" in v for v in data.values()):
            return True
        if any(key in data for key in ("prompt", "workflow", "positive", "negative", "loras")):
            return True
    return isinstance(data, str) and bool(data.strip())


def _extract_png_metadata(file_path: Path):
    if Image is None:
        return None, None

    with Image.open(file_path) as img:
        info = img.info or {}
        prompt_data = info.get("prompt") or info.get("Prompt") or info.get("parameters") or info.get("Comment")
        workflow_data = info.get("workflow") or info.get("Workflow")
        return prompt_data, workflow_data


def _extract_jpeg_metadata(file_path: Path):
    if Image is None:
        return None, None

    with Image.open(file_path) as img:
        prompt_data = None
        workflow_data = None

        exif = img.getexif()
        if exif:
            for tag_id in (0x010E, 0x010F):
                tag_val = exif.get(tag_id)
                if not tag_val:
                    continue
                if isinstance(tag_val, bytes):
                    tag_val = tag_val.decode("utf-8", errors="ignore")
                tag_val = str(tag_val).strip().rstrip("\x00")
                if tag_val.startswith("Workflow:"):
                    workflow_data = tag_val[len("Workflow:"):].strip()
                elif tag_val.startswith("Prompt:"):
                    prompt_data = tag_val[len("Prompt:"):].strip()

            if not prompt_data and not workflow_data:
                user_comment = exif.get(0x9286)
                if user_comment:
                    if isinstance(user_comment, bytes):
                        user_comment = user_comment.decode("utf-8", errors="ignore")
                    user_comment = str(user_comment)
                    if user_comment.startswith("UNICODE"):
                        user_comment = user_comment[7:].lstrip("\x00")
                    parsed = _json_load_maybe(user_comment)
                    if isinstance(parsed, dict):
                        prompt_data = parsed.get("prompt")
                        workflow_data = parsed.get("workflow")

        if prompt_data or workflow_data:
            return prompt_data, workflow_data

        info = getattr(img, "info", {}) or {}
        for key in ("prompt", "workflow", "parameters", "Comment"):
            if key not in info:
                continue
            parsed = _json_load_maybe(info[key])
            if isinstance(parsed, dict):
                return parsed.get("prompt", parsed), parsed.get("workflow")

        return None, None


def _extract_json_metadata(file_path: Path):
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "prompt" in data or "workflow" in data:
            return data.get("prompt"), data.get("workflow")
        if "nodes" in data:
            return None, data
        if any(isinstance(v, dict) and "class_type" in v for v in data.values()):
            return data, None
    return data, None


def _extract_video_metadata(file_path: Path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(file_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return None, None

    ffprobe_data = json.loads(result.stdout)
    tags = ffprobe_data.get("format", {}).get("tags", {})
    if not isinstance(tags, dict):
        return None, None

    prompt_val = tags.get("prompt")
    workflow_val = tags.get("workflow")

    if not prompt_val and not workflow_val and "comment" in tags:
        comment_data = _json_load_maybe(tags.get("comment"))
        if isinstance(comment_data, dict):
            prompt_val = comment_data.get("prompt")
            workflow_val = comment_data.get("workflow")

    return prompt_val, workflow_val


def has_embedded_generation_data(file_path: Path) -> bool:
    ext = file_path.suffix.lower()

    try:
        if ext == ".png":
            prompt_data, workflow_data = _extract_png_metadata(file_path)
        elif ext in {".jpg", ".jpeg", ".webp"}:
            prompt_data, workflow_data = _extract_jpeg_metadata(file_path)
        elif ext == ".json":
            prompt_data, workflow_data = _extract_json_metadata(file_path)
        elif ext in VIDEO_EXTS:
            prompt_data, workflow_data = _extract_video_metadata(file_path)
        else:
            return True
    except Exception as exc:
        print(f"ERROR reading {file_path.name}: {exc}")
        return False

    return _has_workflow_payload(workflow_data) or _has_prompt_payload(prompt_data)


def main() -> int:
    root = Path.cwd()
    input_dir = root / "input"
    no_workflow_dir = input_dir / "no_workflow"

    if not input_dir.exists() or not input_dir.is_dir():
        print("Error: run this from the ComfyUI root so ./input exists.", file=sys.stderr)
        return 1

    if Image is None:
        print("Error: Pillow is required to inspect image metadata.", file=sys.stderr)
        return 2

    no_workflow_dir.mkdir(exist_ok=True)

    moved = 0
    kept = 0
    skipped = 0

    for file_path in sorted(input_dir.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.parent == no_workflow_dir:
            continue

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            skipped += 1
            continue

        if has_embedded_generation_data(file_path):
            kept += 1
            print(f"KEEP  {file_path.name}")
            continue

        destination = no_workflow_dir / file_path.name
        if destination.exists():
            destination.unlink()
        shutil.move(str(file_path), str(destination))
        moved += 1
        print(f"MOVE  {file_path.name} -> input/no_workflow/")

    print(f"\nDone. kept={kept} moved={moved} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())