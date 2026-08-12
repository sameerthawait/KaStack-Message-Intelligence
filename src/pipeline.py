"""
End-to-end pipeline: read messages.csv chronologically -> classify each
message -> extract task/event where applicable -> detect sensitive content
independently -> write three structured JSON output files.

Deliberately a plain function-based pipeline (no framework) — 900 rows is
small, single-machine, single-pass. A queue/worker system would be
over-engineering for this scale; see README "Scalability" section for how
this would change at higher volume.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from src.classifier import classify_message
from src.extractor import extract_task_or_event
from src.sensitive import detect_sensitive_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXTRACTABLE_CATEGORIES = {"action_required", "meeting_or_event"}


class MessageRow:
    __slots__ = ("message_id", "timestamp", "sender", "message")

    def __init__(self, message_id: str, timestamp: str, sender: str, message: str):
        self.message_id = message_id
        self.timestamp = timestamp
        self.sender = sender
        self.message = message


def load_messages(csv_path: Path) -> list[MessageRow]:
    """
    Load and validate the dataset. Rows are expected to already be in
    chronological order (per the assignment); we sort defensively by
    timestamp to guarantee it rather than assume it, since silently
    processing out of order would violate the "process in chronological
    order" rule if the input file were ever hand-edited.
    """
    rows: list[MessageRow] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required_cols = {"message_id", "timestamp", "sender", "message"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV missing required columns. Expected {required_cols}, "
                f"got {reader.fieldnames}"
            )
        for line_no, row in enumerate(reader, start=2):
            if not row.get("message_id") or not row.get("message"):
                # Log only which columns were missing, never the row content —
                # a malformed row could still contain a sensitive value in one
                # of its other fields, and that must never reach the logs.
                missing = [k for k in ("message_id", "message") if not row.get(k)]
                logger.warning("Skipping malformed row at line %d: missing %s", line_no, missing)
                continue
            rows.append(MessageRow(row["message_id"], row["timestamp"], row["sender"], row["message"]))

    rows.sort(key=lambda r: r.timestamp)
    return rows


def run_pipeline(csv_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_messages(csv_path)
    logger.info("Loaded %d messages", len(rows))

    classifications = []
    tasks_events = []
    sensitive_findings = []

    task_counter = 0
    for row in rows:
        classification = classify_message(row.message_id, row.sender, row.message)
        classifications.append(classification)

        if classification.category in EXTRACTABLE_CATEGORIES:
            task_counter += 1
            item_id = f"{'TASK' if classification.category == 'action_required' else 'EVENT'}_{task_counter:04d}"
            task_event = extract_task_or_event(item_id, row.message_id, row.message)
            if task_event:
                tasks_events.append(task_event)

        finding = detect_sensitive_info(row.message_id, row.message)
        if finding:
            sensitive_findings.append(finding)

    _write_json(output_dir / "classifications.json", [c.to_dict() for c in classifications])
    _write_json(output_dir / "tasks_events.json", [t.to_dict() for t in tasks_events])
    _write_json(output_dir / "sensitive_findings.json", [s.to_dict() for s in sensitive_findings])

    summary = {
        "total_messages": len(rows),
        "classification_counts": _count_by(classifications, lambda c: c.category),
        "tasks_extracted": sum(1 for t in tasks_events if t.type == "task"),
        "events_extracted": sum(1 for t in tasks_events if t.type == "event"),
        "sensitive_findings": len(sensitive_findings),
        "low_confidence_classifications": sum(1 for c in classifications if c.confidence < 0.7),
    }
    _write_json(output_dir / "summary.json", summary)
    logger.info("Pipeline complete: %s", summary)
    return summary


def _count_by(items, key_fn) -> dict:
    counts: dict[str, int] = {}
    for item in items:
        k = key_fn(item)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
