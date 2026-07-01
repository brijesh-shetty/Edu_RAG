"""
auth.py — Minimal user authentication registry.
"""

import json
import os
import hashlib

USERS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "users.json")


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        
        # Read from environment
        admin_pw = os.getenv("ADMIN_PASSWORD", "admin123")
        if not admin_pw or admin_pw == "your_admin_pw":
            admin_pw = "admin123"
            
        student_pw = os.getenv("STUDENT_PASSWORD", "student123")
        if not student_pw or student_pw == "your_student_pw":
            student_pw = "student123"
            
        # Default users
        defaults = {
            "admin": {"password_hash": _hash(admin_pw), "role": "admin"},
            "student": {"password_hash": _hash(student_pw), "role": "student"},
        }
        with open(USERS_FILE, "w") as f:
            json.dump(defaults, f, indent=2)
        return defaults
    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def authenticate(username: str, password: str) -> tuple[str | None, str | None]:
    """Returns (role, user_id) or (None, None)."""
    users = _load_users()
    user = users.get(username)
    if user and user["password_hash"] == _hash(password):
        return user["role"], username
    return None, None
