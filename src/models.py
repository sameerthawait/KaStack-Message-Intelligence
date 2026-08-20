"""
Typed data models for the pipeline.

Using dataclasses (not a heavier framework) keeps this dependency-light and
easy for a reviewer to read end-to-end. `to_dict()` gives us explicit control
over JSON field order and null handling instead of relying on a serializer's
defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Classification:
    message_id: str
    category: str
    confidence: float
    reason: str
    rule_id: str  # which rule fired — kept for auditability, not in spec but cheap and useful

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "category": self.category,
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "matched_rule": self.rule_id,
        }


@dataclass
class TaskEvent:
    item_id: str
    type: str  # "task" | "event"
    title: str
    description: Optional[str]
    deadline: Optional[str]  # ISO date (YYYY-MM-DD) or None
    time: Optional[str]  # HH:MM (24h) or None
    person: Optional[str]
    priority: str  # "high" | "medium" | "low"
    source_message_id: str
    location: Optional[str] = None
    unresolved_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "deadline": self.deadline,
            "time": self.time,
            "person": self.person,
            "priority": self.priority,
            "location": self.location,
            "source_message_id": self.source_message_id,
            "unresolved_fields": self.unresolved_fields,
        }


@dataclass
class SensitiveFinding:
    message_id: str
    sensitivity_type: str
    risk: str  # "high" | "medium" | "low"
    masked_text: str
    recommended_action: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "sensitivity_type": self.sensitivity_type,
            "risk": self.risk,
            "masked_text": self.masked_text,
            "recommended_action": self.recommended_action,
            "reason": self.reason,
        }


@dataclass
class PriorityDecision:
    message_id: str
    item_id: Optional[str]
    priority: str  # "critical" | "high" | "medium" | "low"
    reason: str
    signals: list[str]
    confidence: float

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "item_id": self.item_id,
            "priority": self.priority,
            "reason": self.reason,
            "signals": self.signals,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class RelatedGroup:
    group_id: str
    title: str
    related_message_ids: list[str]
    related_task_or_event_ids: list[str]
    status: str  # "pending" | "in_progress" | "completed" | "rescheduled" | "cancelled" | "unclear"
    latest_deadline: Optional[str]
    summary: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "title": self.title,
            "related_message_ids": self.related_message_ids,
            "related_task_or_event_ids": self.related_task_or_event_ids,
            "status": self.status,
            "latest_deadline": self.latest_deadline,
            "summary": self.summary,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class PrivacyRoutingResult:
    target_id: str  # message_id or query_id
    route: str  # "local" | "confirm" | "blocked"
    reason: str
    requires_user_action: bool
    sensitivity_type: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "route": self.route,
            "reason": self.reason,
            "requires_user_action": self.requires_user_action,
            "sensitivity_type": self.sensitivity_type,
        }


@dataclass
class AssistantAnswer:
    query: str
    answer: str
    supporting_message_ids: list[str]
    group_id: Optional[str]
    item_id: Optional[str]
    relevance_scores: list[float]
    reason: str

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "supporting_message_ids": self.supporting_message_ids,
            "group_id": self.group_id,
            "item_id": self.item_id,
            "relevance_scores": [round(s, 2) for s in self.relevance_scores],
            "reason": self.reason,
        }
