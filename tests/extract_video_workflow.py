"""
Extract embedded ComfyUI workflow JSON from a video file.

Usage:
  python extract_video_workflow.py <video_path>
  python extract_video_workflow.py <video_path> --out workflow.json

Notes:
- Assumes a standard ComfyUI video with embedded metadata.
- Tries ffprobe format tags first: "workflow" and "prompt".
- Falls back to JSON in "comment" tag if needed.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run_ffprobe(video_path: Path) -> dict:
    """Return ffprobe JSON for format metadata."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        err = result.stderr.strip() or "ffprobe failed"
        raise RuntimeError(err)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe returned invalid JSON: {e}") from e


def _json_load_maybe(value):
    """Parse JSON string if possible, otherwise return the input unchanged."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def extract_workflow(video_path: Path):
    """Extract workflow object from a Comfy video metadata payload."""
    ffprobe_data = _run_ffprobe(video_path)
    tags = ffprobe_data.get("format", {}).get("tags", {})

    # Preferred: direct ComfyUI workflow tag
    workflow = _json_load_maybe(tags.get("workflow"))
    if isinstance(workflow, dict):
        return workflow

    # Fallback: workflow nested in JSON comment payload
    comment = _json_load_maybe(tags.get("comment"))
    if isinstance(comment, dict) and "workflow" in comment:
        workflow = _json_load_maybe(comment.get("workflow"))
        if isinstance(workflow, dict):
            return workflow

    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract embedded ComfyUI workflow JSON from a video file"
    )
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument(
        "--out",
        dest="output_path",
        default=None,
        help="Optional path to save extracted workflow JSON",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists() or not video_path.is_file():
        print(f"Error: file not found: {video_path}", file=sys.stderr)
        return 1

    try:
        workflow = extract_workflow(video_path)
    except FileNotFoundError:
        print("Error: ffprobe is not installed or not in PATH.", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 3

    if workflow is None:
        print("No embedded workflow metadata found.", file=sys.stderr)
        return 4

    output_json = json.dumps(workflow, indent=2, ensure_ascii=False)

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.write_text(output_json, encoding="utf-8")
        print(f"Workflow extracted to: {output_path}")
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
