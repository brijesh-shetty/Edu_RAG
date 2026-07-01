"""
ingestion_tracker.py — Tracks processed files via SHA-256 hash.
Prevents re-ingesting unchanged files across sessions.
"""

import os
import json
import hashlib
import logging

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _tracker_path(collection_name: str) -> str:
    role_suffix = collection_name.replace("notego_", "")
    return os.path.join(_PROJECT_ROOT, "data", f"processed_files_{role_suffix}.json")


def _load_tracker(collection_name: str) -> dict:
    path = _tracker_path(collection_name)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_tracker(collection_name: str, data: dict):
    path = _tracker_path(collection_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def is_already_processed(file_path: str, collection_name: str) -> bool:
    """Check if a file (by content hash) has already been processed."""
    tracker = _load_tracker(collection_name)
    file_hash = compute_file_hash(file_path)
    filename = os.path.basename(file_path)

    entry = tracker.get(filename)
    if entry and entry.get("hash") == file_hash:
        return True
    return False


def mark_as_processed(file_path: str, collection_name: str):
    """Record a file as processed with its SHA-256 hash."""
    tracker = _load_tracker(collection_name)
    filename = os.path.basename(file_path)
    file_hash = compute_file_hash(file_path)

    tracker[filename] = {
        "hash": file_hash,
        "path": file_path,
    }
    _save_tracker(collection_name, tracker)
    logger.info("Marked '%s' as processed in '%s'", filename, collection_name)


def get_all_processed_filenames(collection_name: str) -> list[str]:
    """Return list of all filenames that have been processed for a collection."""
    tracker = _load_tracker(collection_name)
    return list(tracker.keys())


def clear_tracker(collection_name: str):
    """Delete the tracker file for a collection."""
    path = _tracker_path(collection_name)
    if os.path.exists(path):
        os.remove(path)
        logger.info("Cleared ingestion tracker for '%s'", collection_name)
