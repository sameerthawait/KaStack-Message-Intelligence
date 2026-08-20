"""
Part 1: Priority and Action Engine.

Evaluates every actionable message, task, or event to assign an explainable
priority level: critical, high, medium, or low.

Priority factors:
1. Urgency keywords ("urgent", "asap", "immediately", "earlier than previously planned")
2. Deadline proximity (today/tomorrow, within 3 days, overdue, extended)
3. Status modifications (completed, cancelled, rescheduled, extended)
4. Sender authority (Project Lead, Mentor, HR Team, Director)
5. Sensitive data presence (high-risk credentials, OTPs)
6. Response requirements (explicit action/confirmation needed)

Every decision includes:
- message_id
- item_id (if associated with an extracted task/event)
- priority ("critical" | "high" | "medium" | "low")
- reason (human-readable explanation)
- signals (list of detected signals)
- confidence (0.0 - 1.0)
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from src.models import PriorityDecision, TaskEvent, Classification, SensitiveFinding


def assess_priority(
    message_id: str,
    raw_message: str,
    sender: str,
    timestamp: str,
    classification: Optional[Classification] = None,
    task_event: Optional[TaskEvent] = None,
    sensitive_finding: Optional[SensitiveFinding] = None,
    status_override: Optional[str] = None,
    updated_deadline: Optional[str] = None,
) -> PriorityDecision:
    """
    Compute explainable priority for a message and its associated task/event.
    """
    signals: list[str] = []
    text_lower = raw_message.lower()
    item_id = task_event.item_id if task_event else None

    # 1. Check for explicit completion or cancellation
    if status_override in ("completed", "cancelled") or re.search(r"\b(completed|cancelled|no longer (needed|required))\b", text_lower):
        if "completed" in text_lower or status_override == "completed":
            signals.append("task_completed")
            return PriorityDecision(
                message_id=message_id,
                item_id=item_id,
                priority="low",
                reason="Task or event has already been completed; no further urgent action required.",
                signals=signals,
                confidence=0.96,
            )
        elif "cancel" in text_lower or status_override == "cancelled":
            signals.append("task_cancelled")
            return PriorityDecision(
                message_id=message_id,
                item_id=item_id,
                priority="low",
                reason="Task or event has been cancelled or is no longer required.",
                signals=signals,
                confidence=0.96,
            )

    # 2. Urgency signals
    is_urgent = bool(re.search(r"\b(urgent|asap|immediately|treat this as urgent|earlier than previously planned)\b", text_lower))
    if is_urgent:
        signals.append("urgent_keyword")

    # 3. Authoritative sender signal
    sender_lower = sender.lower()
    if any(k in sender_lower for k in ("lead", "mentor", "hr", "director", "manager", "admin")):
        signals.append("authoritative_sender")

    # 4. Deadline proximity analysis
    deadline = updated_deadline or (task_event.deadline if task_event else None)
    if not deadline:
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", raw_message)
        if date_match:
            deadline = date_match.group(1)
    
    # Check text for relative deadline clues
    if re.search(r"\b(today|due today)\b", text_lower):
        signals.append("deadline_today")
    elif re.search(r"\b(tomorrow|by tomorrow)\b", text_lower):
        signals.append("deadline_tomorrow")
    elif re.search(r"\b(extended|rescheduled|moved to)\b", text_lower):
        signals.append("schedule_updated")

    # If ISO date is present, calculate days difference from message timestamp
    if deadline:
        try:
            msg_dt = datetime.strptime(timestamp[:10], "%Y-%m-%d")
            dl_dt = datetime.strptime(deadline[:10], "%Y-%m-%d")
            diff_days = (dl_dt - msg_dt).days
            if diff_days <= 0:
                signals.append("deadline_today_or_overdue")
            elif diff_days <= 1:
                signals.append("deadline_within_24h")
            elif diff_days <= 3:
                signals.append("deadline_near_term")
            else:
                signals.append("deadline_standard")
        except Exception:
            pass

    # 5. Sensitive data presence
    if sensitive_finding and sensitive_finding.risk == "high":
        signals.append("high_risk_sensitive_data")

    # 6. Response / Action required
    if classification and classification.category == "action_required":
        signals.append("action_required_category")
    elif classification and classification.category == "meeting_or_event":
        signals.append("event_category")

    if re.search(r"\b(new task|scheduled for|by \d{4}-\d{2}-\d{2})\b", text_lower):
        signals.append("action_required_category")

    # Decision Matrix
    # CRITICAL:
    # - Urgent + (today/tomorrow or near deadline or authoritative sender)
    # - High-risk sensitive item needing immediate action
    # - Same day / overdue actionable task
    if "deadline_today" in signals or "deadline_today_or_overdue" in signals or "deadline_within_24h" in signals:
        if "urgent_keyword" in signals:
            return PriorityDecision(
                message_id=message_id,
                item_id=item_id,
                priority="critical",
                reason="Deadline is imminent (today/tomorrow) and explicitly flagged as urgent.",
                signals=signals,
                confidence=0.96,
            )
        return PriorityDecision(
            message_id=message_id,
            item_id=item_id,
            priority="critical",
            reason="Deadline is imminent (due today or within 24 hours).",
            signals=signals,
            confidence=0.94,
        )

    if "urgent_keyword" in signals:
        priority = "critical" if "authoritative_sender" in signals or "action_required_category" in signals else "high"
        reason = "Message contains an explicit urgency indicator with updated action requirements." if priority == "critical" else "Message marked as urgent."
        return PriorityDecision(
            message_id=message_id,
            item_id=item_id,
            priority=priority,
            reason=reason,
            signals=signals,
            confidence=0.92,
        )

    # HIGH:
    # - Near-term deadline (within 3 days)
    # - Authoritative sender
    # - High risk sensitive finding
    if "deadline_near_term" in signals:
        return PriorityDecision(
            message_id=message_id,
            item_id=item_id,
            priority="high",
            reason="Deadline is approaching within the next 3 days.",
            signals=signals,
            confidence=0.90,
        )

    if "authoritative_sender" in signals:
        return PriorityDecision(
            message_id=message_id,
            item_id=item_id,
            priority="high",
            reason="Action item assigned by an authoritative stakeholder (Lead/Mentor/HR).",
            signals=signals,
            confidence=0.90,
        )

    if "high_risk_sensitive_data" in signals:
        return PriorityDecision(
            message_id=message_id,
            item_id=item_id,
            priority="high",
            reason="Contains high-risk credential or authentication data requiring protected handling.",
            signals=signals,
            confidence=0.92,
        )

    # LOW:
    # - Promotional or General Informational items with no deadlines
    if classification and classification.category in ("promotional", "general_information"):
        return PriorityDecision(
            message_id=message_id,
            item_id=item_id,
            priority="low",
            reason=f"Informational or promotional update ({classification.category}) with no action required.",
            signals=signals,
            confidence=0.90,
        )

    # MEDIUM (Default actionable baseline):
    return PriorityDecision(
        message_id=message_id,
        item_id=item_id,
        priority="medium",
        reason="Standard actionable item with regular timeframe and no immediate escalation signals.",
        signals=signals or ["standard_schedule"],
        confidence=0.85,
    )
