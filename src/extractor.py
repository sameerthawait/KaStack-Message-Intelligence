"""
Part 2: Task and event extraction.

Runs only on messages already classified as `action_required` or
`meeting_or_event` (see pipeline.py). Each extraction pattern mirrors a
classifier template 1:1 so the "why did this get extracted this way" answer
is always traceable to one named pattern — same transparency principle as
classifier.py.

Hard rule from the assignment: never guess a missing field. Every pattern
below only fills a field it can literally read from the text; anything else
is left None and listed in `unresolved_fields`.
"""

from __future__ import annotations

import re
from typing import Optional

from src.models import TaskEvent
from src.text_utils import normalize_time, strip_filler_prefixes

_DATE = r"\d{4}-\d{2}-\d{2}"
_TIME = r"\d{1,2}:\d{2}"


def _priority_for(days_note: str = "") -> str:
    # The dataset gives no explicit priority signal (no "urgent"/"low priority"
    # wording anywhere in the 900 messages, verified during analysis), so every
    # extracted task/event defaults to "medium" rather than inventing a signal
    # that isn't in the text. This is a documented assumption, not a guess per
    # message.
    return "medium"


def _person_from_text(text: str) -> Optional[str]:
    m = re.search(r"\b(?:call|ask)\s+([A-Z][a-z]+)\b", text)
    if m:
        return m.group(1)
    m = re.search(r"^([A-Z][a-z]+) asked\b", text)
    if m:
        return m.group(1)
    return None


def extract_task_or_event(item_id: str, message_id: str, raw_message: str) -> Optional[TaskEvent]:
    core = strip_filler_prefixes(raw_message)
    unresolved: list[str] = []

    # ---- Meeting/event templates ----

    m = re.match(rf"^Calendar update: (.+?), ({_DATE}) at ({_TIME}), (.+?)\.?$", core)
    if m:
        title, date, time, location = m.groups()
        return TaskEvent(item_id, "event", title.strip().capitalize(), core, date,
                          normalize_time(time), None, _priority_for(), message_id,
                          location=location.strip())

    m = re.match(rf"^Reminder: (.+?) happens on ({_DATE}) at ({_TIME}) in (.+?)\.?$", core)
    if m:
        title, date, time, location = m.groups()
        return TaskEvent(item_id, "event", title.strip().capitalize(), core, date,
                          normalize_time(time), None, _priority_for(), message_id,
                          location=location.strip())

    m = re.match(rf"^Please join the (.+?) on ({_DATE}), ({_TIME}) at (.+?)\.?$", core)
    if m:
        title, date, time, location = m.groups()
        return TaskEvent(item_id, "event", title.strip().capitalize(), core, date,
                          normalize_time(time), None, _priority_for(), message_id,
                          location=location.strip())

    m = re.match(rf"^Are you available for the (.+?) at ({_TIME}) on ({_DATE})\? Location: (.+?)\.?$", core)
    if m:
        title, time, date, location = m.groups()
        return TaskEvent(item_id, "event", title.strip().capitalize(), core, date,
                          normalize_time(time), None, _priority_for(), message_id,
                          location=location.strip())

    m = re.match(rf"^(?:The )?(.+?) is scheduled for ({_DATE}) at ({_TIME}) in (.+?)\.?$", core)
    if m:
        title, date, time, location = m.groups()
        return TaskEvent(item_id, "event", title.strip().capitalize(), core, date,
                          normalize_time(time), None, _priority_for(), message_id,
                          location=location.strip())

    if re.match(r"^Let us meet sometime next week\.?$", core):
        unresolved = ["deadline", "time", "location", "person"]
        return TaskEvent(item_id, "event", "Proposed team meeting", core, None, None,
                          None, _priority_for(), message_id, location=None,
                          unresolved_fields=unresolved)

    if re.match(r"^The review could be .+\.?$", core):
        unresolved = ["deadline", "time", "location", "person"]
        return TaskEvent(item_id, "event", "Tentative review", core, None, None,
                          None, _priority_for(), message_id, location=None,
                          unresolved_fields=unresolved)

    # ---- Action-required templates ----

    m = re.match(rf"^Don't forget to (.+?); deadline is ({_DATE})\.?$", core)
    if m:
        action, date = m.groups()
        return TaskEvent(item_id, "task", action.strip().capitalize(), core, date,
                          None, _person_from_text(core), _priority_for(), message_id)

    m = re.match(rf"^Can you (.+?) before ({_DATE})\??$", core)
    if m:
        action, date = m.groups()
        return TaskEvent(item_id, "task", action.strip().capitalize(), core, date,
                          None, _person_from_text(core), _priority_for(), message_id)

    m = re.match(rf"^I need you to (.+?) by ({_DATE})\.?$", core)
    if m:
        action, date = m.groups()
        return TaskEvent(item_id, "task", action.strip().capitalize(), core, date,
                          None, _person_from_text(core), _priority_for(), message_id)

    m = re.match(rf"^Please (?!join\b)(.+?) by ({_DATE})\.?$", core)
    if m:
        action, date = m.groups()
        return TaskEvent(item_id, "task", action.strip().capitalize(), core, date,
                          None, _person_from_text(core), _priority_for(), message_id)

    m = re.match(rf"^(.+?) is due on ({_DATE})\.?$", core)
    if m:
        action, date = m.groups()
        return TaskEvent(item_id, "task", action.strip().capitalize(), core, date,
                          None, _person_from_text(core), _priority_for(), message_id)

    if re.match(r"^Could you send it soon\??$", core):
        unresolved = ["deadline", "time", "person"]
        return TaskEvent(item_id, "task", "Send requested item", core, None, None,
                          None, _priority_for(), message_id, unresolved_fields=unresolved)

    if re.match(r"^If possible, review the file before the meeting\.?$", core):
        unresolved = ["deadline", "time", "person"]
        return TaskEvent(item_id, "task", "Review the file", core, None, None,
                          None, _priority_for(), message_id, unresolved_fields=unresolved)

    m = re.match(r"^Please call ([A-Z][a-z]+) when you are free\.?$", core)
    if m:
        person = m.group(1)
        unresolved = ["deadline", "time"]
        return TaskEvent(item_id, "task", f"Call {person}", core, None, None,
                          person, _priority_for(), message_id, unresolved_fields=unresolved)

    # Unmatched but pre-classified as action_required/meeting_or_event by the
    # fallback keyword classifier: still emit a record (never silently drop a
    # task), but mark every field we can't literally read as unresolved.
    return TaskEvent(
        item_id=item_id,
        type="task",
        title=core[:80],
        description=core,
        deadline=None,
        time=None,
        person=_person_from_text(core),
        priority=_priority_for(),
        source_message_id=message_id,
        unresolved_fields=["deadline", "time", "location"],
    )
