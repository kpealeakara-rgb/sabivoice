"""Anonymised interaction log (local JSONL, optionally synced to a private HF dataset)."""
import os
import json
import uuid
import datetime
import threading
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG = DATA_DIR / "interactions.jsonl"
FEEDBACK = DATA_DIR / "feedback.jsonl"
_lock = threading.Lock()
_scheduler = None

repo = os.getenv("LOG_DATASET_REPO")
if repo and os.getenv("HF_TOKEN"):
    try:
        from huggingface_hub import CommitScheduler

        _scheduler = CommitScheduler(
            repo_id=repo, repo_type="dataset", folder_path=DATA_DIR,
            path_in_repo="logs", every=5, private=True, token=os.getenv("HF_TOKEN"),
        )
        _lock = _scheduler.lock
    except Exception as e:  # never break the app over logging
        print("log sync disabled:", e)


def _append(path: Path, row: dict):
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_interaction(**fields) -> str:
    iid = uuid.uuid4().hex[:10]
    row = {"id": iid, "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", **fields}
    _append(LOG, row)
    return iid


def log_feedback(iid: str, helpful: bool, comment: str = ""):
    _append(FEEDBACK, {"id": iid, "helpful": helpful, "comment": comment,
                       "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"})


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def interactions() -> list[dict]:
    return _read(LOG)


def feedback() -> list[dict]:
    return _read(FEEDBACK)
