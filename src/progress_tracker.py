"""
progress_tracker.py — Simple JSON-backed per-user progress tracker.
"""

import json
import os
from datetime import datetime

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "progress")
os.makedirs(_DATA_DIR, exist_ok=True)


def _path(user_id: str) -> str:
    return os.path.join(_DATA_DIR, f"{user_id}.json")


def _load(user_id: str) -> dict:
    p = _path(user_id)
    if not os.path.exists(p):
        return {"questions": [], "quiz_scores": []}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {"questions": [], "quiz_scores": []}


def record_question(user_id: str, question: str, verdict: str):
    data = _load(user_id)
    data["questions"].append({
        "ts": datetime.now().isoformat(),
        "question": question,
        "verdict": verdict,
    })
    try:
        with open(_path(user_id), "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def record_quiz_score(user_id: str, topic: str, score: float, total: int):
    data = _load(user_id)
    data["quiz_scores"].append({
        "ts": datetime.now().isoformat(),
        "topic": topic,
        "score": score,
        "total": total,
    })
    try:
        with open(_path(user_id), "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def get_summary(user_id: str) -> dict:
    data = _load(user_id)
    qs = data["questions"]
    grounded = sum(1 for q in qs if q.get("verdict") == "GROUNDED")
    return {
        "total_questions": len(qs),
        "grounded_rate": grounded / len(qs) if qs else 0.0,
        "quiz_attempts": len(data["quiz_scores"]),
        "recent_topics": list({q["question"][:40] for q in qs[-10:]}),
    }
