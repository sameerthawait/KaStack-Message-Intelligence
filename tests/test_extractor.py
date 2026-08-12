import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractor import extract_task_or_event


def test_extract_event_with_full_fields():
    te = extract_task_or_event(
        "EVENT_1", "MSG_0001",
        "For today: Calendar update: family dinner, 2026-09-19 at 10:00, the library.",
    )
    assert te.type == "event"
    assert te.deadline == "2026-09-19"
    assert te.time == "10:00"
    assert te.location == "the library"
    assert te.unresolved_fields == []


def test_extract_task_with_deadline():
    te = extract_task_or_event("TASK_1", "MSG_0002", "Can you review the privacy checklist before 2026-09-09?")
    assert te.type == "task"
    assert te.deadline == "2026-09-09"
    assert te.time is None
    assert te.unresolved_fields == []


def test_never_guesses_missing_fields():
    """'The review could be Friday afternoon.' has no resolvable date — must
    be left null and flagged, never inferred (e.g. must NOT invent a Friday
    date)."""
    te = extract_task_or_event("EVENT_2", "MSG_0037", "One more thing: The review could be Friday afternoon.")
    assert te.deadline is None
    assert "deadline" in te.unresolved_fields


def test_extract_person_from_call_task():
    te = extract_task_or_event("TASK_2", "MSG_X", "Please call Maya when you are free.")
    assert te.person == "Maya"
    assert te.deadline is None
    assert "deadline" in te.unresolved_fields


def test_extract_meeting_available_template():
    te = extract_task_or_event(
        "EVENT_3", "MSG_X",
        "Are you available for the technical interview at 13:00 on 2026-09-10? Location: the training hall.",
    )
    assert te.deadline == "2026-09-10"
    assert te.time == "13:00"
    assert te.location == "the training hall"
