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


def _ensure_backups_dir():
    path = _get_backups_dir()
    os.makedirs(path, exist_ok=True)
    return path


def _backup_path(data_path, index):
    """Return the path for the i-th numbered backup (1-based)."""
    filename = os.path.basename(data_path)
    return os.path.join(_ensure_backups_dir(), f"{filename}_bak_{index:02d}")


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


def _is_valid_json_file(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict)
    except Exception:
        return False


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


def refresh_backup(data_path):
    """
    Rotate backups and copy the current data file to *_bak_01 if valid.
    This is the daily backup entry point; it should NOT be called on every save.
    """
    if not _is_valid_json_file(data_path):
        return
    if _backup_is_from_today(_backup_path(data_path, 1)):
        return
    _rotate_backups(data_path)
    try:
        _ensure_backups_dir()
        backup_path = _backup_path(data_path, 1)
        shutil.copy2(data_path, backup_path)
        # Record the backup creation time so _backup_is_from_today() works.
        os.utime(backup_path, None)
    except Exception:
        pass


def load_with_fallback(data_path, log_prefix="PromptManager"):
    """
    Load JSON from data_path. If corrupt, try *_bak_01..*_bak_05 and return
    the newest valid backup. The main file is NOT overwritten here.
    """
    if _is_valid_json_file(data_path):
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[{log_prefix}] Error loading {data_path}: {e}")
    else:
        print(f"[{log_prefix}] Data file missing or corrupt: {data_path}")

    for i in range(1, MAX_BACKUPS + 1):
        candidate = _backup_path(data_path, i)
        if not _is_valid_json_file(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[{log_prefix}] Recovered data from backup: {candidate}")
            return data
        except Exception as be:
            print(f"[{log_prefix}] Backup load failed ({candidate}): {be}")

    print(f"[{log_prefix}] No valid backup found for {data_path}.")
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
    Called on startup for each registered data file. Creates a daily backup if
    one does not already exist for today.
    """
    if not _is_valid_json_file(data_path):
        return
    refresh_backup(data_path)
