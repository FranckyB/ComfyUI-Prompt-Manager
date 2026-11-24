"""
Centralized rotating backup manager for Prompt Manager JSON data files.

Keeps up to MAX_BACKUPS numbered backups per data file in a shared
user/default/prompt_backups/ folder. A new backup is created once per day when
ComfyUI starts, not on every save. This gives us several daily recovery points
instead of many copies from the same session.
"""
import json
import os
import shutil
import time

import folder_paths

MAX_BACKUPS = 5
BACKUPS_DIR_NAME = "prompt_backups"


def _get_backups_dir():
    return os.path.join(folder_paths.get_user_directory(), "default", BACKUPS_DIR_NAME)


def _backup_path(data_path, index):
    """Return the path for the i-th numbered backup (1-based)."""
    filename = os.path.basename(data_path)
    return os.path.join(_get_backups_dir(), f"{filename}_bak_{index:02d}")


def _rotate_backups(data_path):
    """Shift existing backups 01 -> 02 -> ... and drop the oldest if over MAX."""
    for i in range(MAX_BACKUPS, 1, -1):
        older = _backup_path(data_path, i)
        newer = _backup_path(data_path, i - 1)
        if os.path.exists(older):
            try:
                os.remove(older)
            except Exception:
                pass
        if os.path.exists(newer):
            try:
                shutil.copy2(newer, older)
            except Exception:
                pass


def _read_json_dict(path):
    """The JSON object stored at path, or None when it is missing, unreadable or not an object."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _backup_is_from_today(backup_path):
    if not os.path.exists(backup_path):
        return False
    try:
        mtime = os.path.getmtime(backup_path)
        backup_day = time.strftime("%Y-%m-%d", time.localtime(mtime))
        today = time.strftime("%Y-%m-%d", time.localtime())
        return backup_day == today
    except Exception:
        return False


def load_with_fallback(data_path, log_prefix="PromptManager"):
    """
    Load JSON from data_path. If it is unreadable, return the newest valid
    *_bak_01..*_bak_05 instead. The main file is NOT overwritten here. None
    when there is no file and no backup.
    """
    data = _read_json_dict(data_path)
    if data is not None:
        return data
    exists = os.path.exists(data_path)
    for i in range(1, MAX_BACKUPS + 1):
        candidate = _backup_path(data_path, i)
        data = _read_json_dict(candidate)
        if data is not None:
            print(f"[{log_prefix}] {data_path} is unreadable; using the backup {candidate}")
            return data
    if exists:
        print(f"[{log_prefix}] {data_path} is unreadable and no backup could be read")
    return None


def atomic_save(data_path, data, log_prefix="PromptManager"):
    """
    Save data atomically. Returns True on success. Backups are NOT rotated here;
    they are managed daily by check_backup.
    """
    if not isinstance(data, dict):
        print(f"[{log_prefix}] Refusing to save non-dict data to {data_path}")
        return False

    tmp_path = data_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, data_path)
        return True
    except Exception as e:
        print(f"[{log_prefix}] Error saving {data_path}: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


def check_backup(data_path):
    """
    Called on startup for each registered data file: rotate backups and copy
    the current data file to *_bak_01 if it is valid and today has no backup
    yet. Not called on every load or save.
    """
    if _read_json_dict(data_path) is None:
        return
    if _backup_is_from_today(_backup_path(data_path, 1)):
        return
    _rotate_backups(data_path)
    try:
        os.makedirs(_get_backups_dir(), exist_ok=True)
        backup_path = _backup_path(data_path, 1)
        shutil.copy2(data_path, backup_path)
        # Record the backup creation time so _backup_is_from_today() works.
        os.utime(backup_path, None)
    except Exception:
        pass
