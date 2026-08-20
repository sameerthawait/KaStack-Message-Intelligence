"""
Unit and integration tests for L2 features:
1. Priority and Action Engine (src/priority.py)
2. Related-Message Grouping Engine (src/grouping.py)
3. Privacy Router (src/privacy_router.py)
4. Intelligent Assistant and Semantic Search (src/assistant.py)
5. End-to-end pipeline execution with L2 datasets
"""

import tempfile
from pathlib import Path

import pytest

from src.classifier import classify_message
from src.extractor import extract_task_or_event
from src.sensitive import detect_sensitive_info
from src.priority import assess_priority
from src.grouping import group_messages, extract_topic_title
from src.privacy_router import route_privacy_policy
from src.assistant import IntelligentAssistant, LocalSemanticIndex
from src.pipeline import run_pipeline


# -----------------------------------------------------------------------------
# Part 1: Priority Engine Tests
# -----------------------------------------------------------------------------

def test_priority_critical_urgent_tomorrow():
    msg = "The deadline to confirm the interview slot is now tomorrow at 10 AM. This is urgent."
    cl = classify_message("DEMO_001", "Aarav", msg)
    te = extract_task_or_event("TASK_0001", "DEMO_001", msg)
    prio = assess_priority("DEMO_001", msg, "Aarav", "2026-10-04 09:41:00", cl, te)

    assert prio.priority == "critical"
    assert "urgent_keyword" in prio.signals
    assert prio.confidence >= 0.90


def test_priority_completed_is_low():
    msg = "Confirmed: email the signed document has been completed."
    cl = classify_message("DEMO_002", "Kabir", msg)
    prio = assess_priority("DEMO_002", msg, "Kabir", "2026-10-04 10:22:00", cl, None, status_override="completed")

    assert prio.priority == "low"
    assert "task_completed" in prio.signals


def test_priority_cancelled_is_low():
    msg = "Cancel update the project tracker; it is no longer needed."
    cl = classify_message("DEMO_003", "Maya", msg)
    prio = assess_priority("DEMO_003", msg, "Maya", "2026-10-04 11:03:00", cl, None, status_override="cancelled")

    assert prio.priority == "low"
    assert "task_cancelled" in prio.signals


def test_priority_high_authoritative_sender():
    msg = "New task: create a latency chart by 2026-10-03."
    cl = classify_message("MSG_1019", "Project Lead", msg)
    te = extract_task_or_event("TASK_0002", "MSG_1019", msg)
    prio = assess_priority("MSG_1019", msg, "Project Lead", "2026-09-30 17:14:00", cl, te)

    assert prio.priority in ("critical", "high")
    assert "authoritative_sender" in prio.signals


# -----------------------------------------------------------------------------
# Part 2: Related-Message Grouping Tests
# -----------------------------------------------------------------------------

def test_grouping_status_progression():
    messages = [
        {"message_id": "M1", "timestamp": "2026-09-01 10:00:00", "sender": "Aarav", "message": "Please confirm the interview slot by 2026-09-05."},
        {"message_id": "M2", "timestamp": "2026-09-26 13:25:00", "sender": "Aarav", "message": "Please confirm whether you started to confirm the interview slot."},
        {"message_id": "M3", "timestamp": "2026-10-04 09:41:00", "sender": "Aarav", "message": "The deadline to confirm the interview slot is now tomorrow at 10 AM. This is urgent."},
    ]
    te = [extract_task_or_event("TASK_001", "M1", messages[0]["message"])]
    groups = group_messages(messages, [t for t in te if t])

    assert len(groups) >= 1
    g = [grp for grp in groups if "Interview" in grp.title][0]
    assert len(g.related_message_ids) == 3
    assert g.status in ("in_progress", "pending", "rescheduled")
    assert g.confidence >= 0.85


def test_grouping_completion():
    messages = [
        {"message_id": "M1", "timestamp": "2026-09-01 10:00:00", "sender": "Kabir", "message": "Don't forget to email the signed document; deadline is 2026-09-04."},
        {"message_id": "M2", "timestamp": "2026-10-04 10:22:00", "sender": "Kabir", "message": "Confirmed: email the signed document has been completed."},
    ]
    groups = group_messages(messages, [])
    g = [grp for grp in groups if "Signed Document" in grp.title][0]
    assert g.status == "completed"


# -----------------------------------------------------------------------------
# Part 3: Privacy Router Tests
# -----------------------------------------------------------------------------

def test_privacy_router_blocked_otp():
    msg = "Your fictional OTP is 864219. Do not share it."
    sf = detect_sensitive_info("DEMO_012", msg)
    res = route_privacy_policy("DEMO_012", msg, sf)

    assert res.route == "blocked"
    assert "high-risk" in res.reason.lower() or "blocked" in res.reason.lower()


def test_privacy_router_confirm_address():
    msg = "Deliver the demo device to 22 Green Park Road, Chennai."
    sf = detect_sensitive_info("DEMO_014", msg)
    res = route_privacy_policy("DEMO_014", msg, sf)

    assert res.route == "confirm"
    assert res.requires_user_action is True


def test_privacy_router_local_standard_query():
    msg = "Which tasks have been completed?"
    res = route_privacy_policy("Q1", msg, None)

    assert res.route == "local"
    assert res.requires_user_action is False


# -----------------------------------------------------------------------------
# Part 4: Semantic Search & Assistant Tests
# -----------------------------------------------------------------------------

def test_assistant_demo_queries():
    messages = [
        {"message_id": "DEMO_001", "timestamp": "2026-10-04 09:41:00", "sender": "Aarav", "message": "The deadline to confirm the interview slot is now tomorrow at 10 AM. This is urgent."},
        {"message_id": "DEMO_002", "timestamp": "2026-10-04 10:22:00", "sender": "Kabir", "message": "Confirmed: email the signed document has been completed."},
        {"message_id": "DEMO_003", "timestamp": "2026-10-04 11:03:00", "sender": "Maya", "message": "Cancel update the project tracker; it is no longer needed."},
        {"message_id": "DEMO_007", "timestamp": "2026-10-04 13:47:00", "sender": "Ishaan", "message": "The internship orientation has moved to 2026-10-07 at 15:00."},
        {"message_id": "DEMO_009", "timestamp": "2026-10-04 15:09:00", "sender": "Ishaan", "message": "The date for internship orientation stays the same, but the time is now 17:30."},
        {"message_id": "DEMO_012", "timestamp": "2026-10-04 17:12:00", "sender": "Private", "message": "Your fictional OTP is 864219. Do not share it."},
        {"message_id": "DEMO_014", "timestamp": "2026-10-04 18:34:00", "sender": "Operations", "message": "Deliver the demo device to 22 Green Park Road, Chennai."},
        {"message_id": "DEMO_016", "timestamp": "2026-10-04 19:56:00", "sender": "Aarav", "message": "Confirm the interview slot might already be finished, but I cannot confirm it."},
        {"message_id": "DEMO_022", "timestamp": "2026-10-05 00:02:00", "sender": "Unknown", "message": "Was the compliance form approved by the finance director?"},
    ]

    classifications = [classify_message(m["message_id"], m["sender"], m["message"]) for m in messages]
    tasks_events = [extract_task_or_event(f"T{i}", m["message_id"], m["message"]) for i, m in enumerate(messages)]
    tasks_events = [t for t in tasks_events if t]
    sensitive_findings = [detect_sensitive_info(m["message_id"], m["message"]) for m in messages]
    sensitive_findings = [s for s in sensitive_findings if s]

    groups = group_messages(messages, tasks_events)
    priority_decisions = [
        assess_priority(m["message_id"], m["message"], m["sender"], m["timestamp"], classifications[i])
        for i, m in enumerate(messages)
    ]

    assistant = IntelligentAssistant(
        messages=messages,
        classifications=classifications,
        tasks_events=tasks_events,
        sensitive_findings=sensitive_findings,
        priority_decisions=priority_decisions,
        related_groups=groups,
    )

    # DQ01: Critical task in demo
    a1 = assistant.answer_query("Which existing task became critical in the demo data?")
    assert "DEMO_001" in a1.supporting_message_ids
    assert "interview slot" in a1.answer.lower()

    # DQ03: Rescheduled meeting
    a3 = assistant.answer_query("Which meeting was rescheduled and what is its latest schedule?")
    assert "17:30" in a3.answer or "2026-10-07" in a3.answer
    assert "DEMO_007" in a3.supporting_message_ids or "DEMO_009" in a3.supporting_message_ids

    # DQ05: Blocked messages
    a5 = assistant.answer_query("Which demo messages must be blocked from external processing?")
    assert "DEMO_012" in a5.supporting_message_ids

    # DQ06: Requires confirmation
    a6 = assistant.answer_query("Which message requires confirmation before processing?")
    assert "DEMO_014" in a6.supporting_message_ids

    # DQ08: Insufficient evidence
    a8 = assistant.answer_query("Was the compliance form approved by the finance director?")
    assert "insufficient evidence" in a8.answer.lower()


# -----------------------------------------------------------------------------
# Part 5: End-to-end Pipeline Execution
# -----------------------------------------------------------------------------

def test_pipeline_l2_outputs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_path = Path("L2/l2_demo_messages.csv")
        if not csv_path.exists():
            pytest.skip("L2/l2_demo_messages.csv not available")

        summary = run_pipeline(csv_path, tmp_path)

        assert (tmp_path / "classifications.json").exists()
        assert (tmp_path / "tasks_events.json").exists()
        assert (tmp_path / "sensitive_findings.json").exists()
        assert (tmp_path / "priority_decisions.json").exists()
        assert (tmp_path / "related_groups.json").exists()
        assert (tmp_path / "privacy_routing.json").exists()
        assert (tmp_path / "benchmark_report.json").exists()
        assert (tmp_path / "summary.json").exists()

        assert summary["total_messages"] > 0
        assert summary["related_groups_count"] > 0
        assert "priority_counts" in summary
