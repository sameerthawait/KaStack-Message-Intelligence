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
