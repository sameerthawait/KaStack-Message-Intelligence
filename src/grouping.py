"""
Part 2: Related-Message Grouping Engine.

Connects messages that refer to the same task, meeting, event, request, or subject.
Maintains chronological topic threads and tracks the dynamic status lifecycle:
- pending (initial request or open task)
- in_progress (follow-up/status check)
- completed (explicit completion confirmation)
- rescheduled (deadline or meeting time moved)
- cancelled (explicit cancellation)
- unclear (conflicting, ambiguous, or tentative updates)

Outputs:
- group_id ("GROUP_001", "GROUP_002", ...)
- title (Clean descriptive subject)
- related_message_ids (chronological list of message IDs)
- related_task_or_event_ids (associated task/event IDs)
- status (current state after latest message)
- latest_deadline (most recent valid deadline or None)
- summary (chronological narrative summary)
- confidence (0.0 to 1.0)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from src.models import RelatedGroup, TaskEvent, Classification


# Canonical topic signatures mapped to standardized titles
TOPIC_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Interview Slot Confirmation", re.compile(r"\b(interview slot|confirm the interview slot|technical interview)\b", re.IGNORECASE)),
    ("Signed Document Submission", re.compile(r"\b(signed document|email the signed document)\b", re.IGNORECASE)),
    ("Project Tracker Update", re.compile(r"\b(project tracker|update the project tracker)\b", re.IGNORECASE)),
    ("Assignment Upload", re.compile(r"\b(upload the assignment|the assignment|concerning the assignment)\b", re.IGNORECASE)),
    ("Model Results Review", re.compile(r"\b(model results|review the model results|model-results review)\b", re.IGNORECASE)),
    ("Internship Orientation", re.compile(r"\b(internship orientation)\b", re.IGNORECASE)),
    ("Team Standup Meeting", re.compile(r"\b(team stand-up|team standup)\b", re.IGNORECASE)),
    ("Privacy Checklist Review", re.compile(r"\b(privacy checklist|review the privacy checklist)\b", re.IGNORECASE)),
    ("Onboarding Form Completion", re.compile(r"\b(onboarding form|complete the onboarding form)\b", re.IGNORECASE)),
    ("Python Exercise Completion", re.compile(r"\b(python exercise|complete the python exercise)\b", re.IGNORECASE)),
    ("Service Centre Call", re.compile(r"\b(service centre|call the service centre)\b", re.IGNORECASE)),
    ("Library Book Renewal", re.compile(r"\b(renew the library book|library book)\b", re.IGNORECASE)),
    ("Electricity Bill Payment", re.compile(r"\b(pay the electricity bill|electricity bill)\b", re.IGNORECASE)),
    ("Family Dinner Event", re.compile(r"\b(family dinner)\b", re.IGNORECASE)),
    ("Mentor Catch-up Meeting", re.compile(r"\b(mentor catch-up|mentor catchup)\b", re.IGNORECASE)),
    ("Client Email Reply", re.compile(r"\b(reply to the client email|client email)\b", re.IGNORECASE)),
    ("Test Cases Execution", re.compile(r"\b(finish the test cases|test cases)\b", re.IGNORECASE)),
    ("Revised Presentation Submission", re.compile(r"\b(revised presentation|send the revised presentation)\b", re.IGNORECASE)),
    ("Demo Video Preparation", re.compile(r"\b(prepare the demo video|demo video)\b", re.IGNORECASE)),
    ("Weekly Report Submission", re.compile(r"\b(submit the weekly report|weekly report|project report)\b", re.IGNORECASE)),
    ("Expense Receipt Submission", re.compile(r"\b(send the expense receipt|expense receipt)\b", re.IGNORECASE)),
    ("Project Files Backup", re.compile(r"\b(back up the project files|project files)\b", re.IGNORECASE)),
    ("Study Group Session", re.compile(r"\b(study-group session|study group)\b", re.IGNORECASE)),
    ("Client Discussion Meeting", re.compile(r"\b(client discussion)\b", re.IGNORECASE)),
    ("Product Demo Session", re.compile(r"\b(product demo)\b", re.IGNORECASE)),
    ("Doctor Appointment", re.compile(r"\b(doctor appointment)\b", re.IGNORECASE)),
    ("AI Workshop Session", re.compile(r"\b(ai workshop)\b", re.IGNORECASE)),
    ("Project Review Meeting", re.compile(r"\b(project review)\b", re.IGNORECASE)),
    ("Placement Briefing Session", re.compile(r"\b(placement briefing)\b", re.IGNORECASE)),
    ("Sprint Planning Meeting", re.compile(r"\b(sprint planning)\b", re.IGNORECASE)),
    ("Compliance Form Review", re.compile(r"\b(compliance form|approved by the finance director)\b", re.IGNORECASE)),
    ("Offline Inference Demo", re.compile(r"\b(offline inference demo|prepare the offline inference demo)\b", re.IGNORECASE)),
    ("Embedding Models Comparison", re.compile(r"\b(compare two embedding models|embedding models)\b", re.IGNORECASE)),
    ("Privacy-Routing Documentation", re.compile(r"\b(privacy-routing decisions|document privacy-routing)\b", re.IGNORECASE)),
    ("Latency Chart Creation", re.compile(r"\b(create a latency chart|latency chart)\b", re.IGNORECASE)),
    ("Quantization Results Review", re.compile(r"\b(quantization results|review the quantization results)\b", re.IGNORECASE)),
    ("Offline Assistant Testing", re.compile(r"\b(test the assistant without internet|test the optimized assistant in offline mode)\b", re.IGNORECASE)),
    ("Architecture Diagram Update", re.compile(r"\b(update the architecture diagram|architecture diagram)\b", re.IGNORECASE)),
    ("Memory Usage Measurement", re.compile(r"\b(measure memory usage|memory usage)\b", re.IGNORECASE)),
    ("L2 Presentation Preparation", re.compile(r"\b(prepare the l2 presentation|l2 presentation)\b", re.IGNORECASE)),
    ("Mandatory Queries Validation", re.compile(r"\b(validate the mandatory queries|mandatory queries)\b", re.IGNORECASE)),
]


def extract_topic_title(message_text: str) -> Optional[str]:
    """Extract standardized topic title from message text."""
    for title, pattern in TOPIC_PATTERNS:
        if pattern.search(message_text):
            return title
    return None


def _extract_date_from_text(text: str) -> Optional[str]:
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        return m.group(1)
    return None


def group_messages(
    messages: list[dict],
    tasks_events: list[TaskEvent],
) -> list[RelatedGroup]:
    """
    Cluster messages chronologically into related-message groups and track lifecycle status.
    """
    te_by_msg_id = {te.source_message_id: te for te in tasks_events}
    groups_by_title: dict[str, list[dict]] = defaultdict(list)

    # 1. Group messages by semantic topic
    for msg in messages:
        mid = msg.get("message_id", "")
        text = msg.get("message", "")
        topic = extract_topic_title(text)
        if topic:
            groups_by_title[topic].append(msg)

    result_groups: list[RelatedGroup] = []
    group_counter = 1

    for title, topic_msgs in groups_by_title.items():
        if not topic_msgs:
            continue

        # Sort chronologically
        topic_msgs.sort(key=lambda x: x.get("timestamp", ""))
        related_mids = [m["message_id"] for m in topic_msgs]
        related_te_ids = []

        status = "pending"
        latest_deadline: Optional[str] = None
        actions_seen: list[str] = []

        for m in topic_msgs:
            mid = m["message_id"]
            text = m.get("message", "")
            text_lower = text.lower()
            te = te_by_msg_id.get(mid)
            if te and te.item_id:
                related_te_ids.append(te.item_id)
                if te.deadline:
                    latest_deadline = te.deadline

            # Check for deadline updates in message text
            d = _extract_date_from_text(text)
            if d:
                latest_deadline = d

            # Lifecycle status progression
            if re.search(r"\b(completed successfully|has been completed|is done|already been handled|report has been submitted)\b", text_lower):
                status = "completed"
                actions_seen.append("confirmed completed")
            elif re.search(r"\b(cancel|no longer needed|no longer required|has been cancelled)\b", text_lower):
                status = "cancelled"
                actions_seen.append("cancelled")
            elif re.search(r"\b(moved to|rescheduled|time is now|new schedule|extended to)\b", text_lower):
                status = "rescheduled"
                actions_seen.append("rescheduled/extended")
            elif re.search(r"\b(might already be|cannot confirm|not completely sure|may move|wait for confirmation|wait for the official update|may be monday|could be)\b", text_lower):
                status = "unclear"
                actions_seen.append("marked ambiguous/conflicting")
            elif re.search(r"\b(following up|in progress|share an update|check the latest status|still needs attention|urgent|earlier than previously planned)\b", text_lower):
                if status not in ("completed", "cancelled"):
                    status = "in_progress"
                    actions_seen.append("followed up")
            else:
                if status == "pending":
                    actions_seen.append("initiated")

        # Generate concise narrative summary
        summary_parts = []
        summary_parts.append(f"Subject concerning '{title}' with {len(topic_msgs)} related messages.")
        if "initiated" in actions_seen:
            summary_parts.append("Initial request was logged.")
        if "followed up" in actions_seen:
            summary_parts.append("Follow-ups and status checks were communicated.")
        if "rescheduled/extended" in actions_seen:
            summary_parts.append(f"Schedule or deadline was updated (latest: {latest_deadline or 'see details'}).")
        if "marked ambiguous/conflicting" in actions_seen:
            summary_parts.append("Ambiguous or conflicting updates were noted awaiting confirmation.")
        if status == "completed":
            summary_parts.append("Final message confirmed successful completion.")
        elif status == "cancelled":
            summary_parts.append("Item was cancelled as no longer required.")

        summary_text = " ".join(summary_parts)
        group_id = f"GROUP_{group_counter:03d}"
        group_counter += 1

        result_groups.append(
            RelatedGroup(
                group_id=group_id,
                title=title,
                related_message_ids=related_mids,
                related_task_or_event_ids=list(dict.fromkeys(related_te_ids)),
                status=status,
                latest_deadline=latest_deadline,
                summary=summary_text,
                confidence=0.92 if len(topic_msgs) > 1 else 0.85,
            )
        )

    return result_groups
