#!/usr/bin/env python3
"""
Organize Files by Workflow Presence
Standalone script to scan a folder and move files without workflow metadata
Defaults to "D:/ComfyUI/input" if no folder is provided.
"""

import json
import shutil
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("ERROR: PIL/Pillow is required!")
    print("Install with: pip install pillow")
    input("Press Enter to exit...")
    exit(1)


def check_png_workflow(file_path):
    """Check if PNG has workflow metadata"""
    try:
        with Image.open(file_path) as img:
            return 'workflow' in img.info or 'prompt' in img.info
    except:
        return False


def check_jpeg_workflow(file_path):
    """Check if JPEG/WebP has workflow metadata"""
    try:
        with Image.open(file_path) as img:
            exif = img.getexif()
            if exif and 0x9286 in exif:
                return True
            if hasattr(img, 'info'):
                return any(key in img.info for key in ['workflow', 'prompt', 'parameters'])
    except:
        return False


def check_json_workflow(file_path):
    """Check if JSON file contains workflow data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return 'nodes' in data or 'workflow' in data or any(
                isinstance(v, dict) and 'class_type' in v for v in data.values()
            )
    except Exception as e:
        print(f"    (JSON parse error: {e})")
        pass
    return False


def check_video_workflow(file_path):
    """Check if video has workflow metadata using ffprobe executable"""
    import subprocess

    try:
        # Use ffprobe to get video metadata in JSON format
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(file_path)],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'format' in data and 'tags' in data['format']:
                tags = data['format']['tags']
                # Check for common workflow metadata keys
                return any(key.lower() in ['comment', 'workflow', 'prompt', 'parameters'] for key in tags)
        return False

    except FileNotFoundError:
        # ffprobe not found in PATH
        print("    (Skipping - ffprobe not found in PATH)")
        return None  # Return None to skip
    except subprocess.TimeoutExpired:
        print("    (Timeout reading video)")
        return False
    except Exception as e:
        print(f"    (Video probe error: {e})")
        return False


def has_workflow_metadata(file_path):
    """Check if file has workflow metadata"""
    ext = file_path.suffix.lower()

    if ext == '.png':
        return check_png_workflow(file_path)
    elif ext in ['.jpg', '.jpeg', '.webp']:
        return check_jpeg_workflow(file_path)
    elif ext == '.json':
        return check_json_workflow(file_path)
    elif ext in ['.mp4', '.webm', '.mov', '.avi']:
        return check_video_workflow(file_path)

    return False


def main(folder_path=None):
    print("=" * 60)
    print("File Organizer - Separate files by workflow presence")
    print("=" * 60)

    if folder_path is None:
        # Get folder from user
        folder_input = input("\nEnter folder path to scan (default D:/ComfyUI/input): ").strip().strip('"')
        if folder_input == "":
            folder_path = Path("D:/ComfyUI/input")
        else:
            folder_path = Path(folder_input)

    if not folder_path.exists():
        print(f"\nERROR: Folder does not exist: {folder_path}")
        input("Press Enter to exit...")
        return

    if not folder_path.is_dir():
        print(f"\nERROR: Not a directory: {folder_path}")
        input("Press Enter to exit...")
        return

    # Create no_workflow subfolder
    no_workflow_path = folder_path / "no_workflow"
    no_workflow_path.mkdir(exist_ok=True)

    print(f"\nScanning: {folder_path}")
    print(f"Moving files without workflow to: {no_workflow_path}")
    print("=" * 60)

    # Supported extensions
    supported = {'.png', '.jpg', '.jpeg', '.webp', '.json', '.mp4', '.webm', '.mov', '.avi'}

    files_with_workflow = []
    files_without_workflow = []
    files_skipped = []

    # Scan only files in this directory (no subdirectories)
    for file_path in folder_path.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() in supported:
            print(f"Checking: {file_path.name}", end=' ... ')

            try:
                result = has_workflow_metadata(file_path)
                if result is None:
                    # Skipped (e.g., video without ffmpeg)
                    files_skipped.append(file_path)
                elif result:
                    print("✓ Has workflow")
                    files_with_workflow.append(file_path)
                else:
                    print("✗ No workflow")
                    files_without_workflow.append(file_path)
            except Exception as e:
                print(f"ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Files with workflow: {len(files_with_workflow)}")
    print(f"Files without workflow: {len(files_without_workflow)}")
    if files_skipped:
        print(f"Files skipped: {len(files_skipped)}")

    # Move files without workflow
    if files_without_workflow:
        print("\nMoving files without workflow...")
        for file_path in files_without_workflow:
            dest_path = no_workflow_path / file_path.name
            try:
                shutil.move(str(file_path), str(dest_path))
                print(f"  Moved: {file_path.name}")
            except Exception as e:
                print(f"  ERROR moving {file_path.name}: {e}")

    print("\nDone!")
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
