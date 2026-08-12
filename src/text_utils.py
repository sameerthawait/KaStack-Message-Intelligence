"""
Shared text-normalization helpers.

The dataset wraps its ~24 core message templates in filler prefixes
("FYI:", "Quick update:", "Can you help? ", etc.) that carry no
classification signal. Stripping them first means every downstream regex
only has to match the core template once, instead of once per prefix
combination.
"""

from __future__ import annotations

import re

# Order matters only in that each is tried repeatedly until none match
# (a message can carry more than one prefix, e.g. "Can you help? FYI: ...").
_FILLER_PREFIXES = [
    "For today: ",
    "Quick update: ",
    "FYI: ",
    "Just checking—",
    "Just checking-",  # tolerate a plain hyphen in case of encoding differences
    "One more thing: ",
    "Please note: ",
    "Important: ",
    "Can you help? ",
    "Hi, ",
]


def strip_filler_prefixes(message: str) -> str:
    """Iteratively remove known filler prefixes to expose the core sentence."""
    text = message.strip()
    changed = True
    while changed:
        changed = False
        for prefix in _FILLER_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):]
                changed = True
    return text.strip()


def normalize_time(raw_time: str) -> str:
    """Normalize a time string like '9:00', '9 AM', '13:00' to 24h 'HH:MM'."""
    raw_time = raw_time.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", raw_time)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    m = re.match(r"^(\d{1,2})\s*(AM|PM)$", raw_time, re.IGNORECASE)
    if m:
        hour = int(m.group(1))
        meridiem = m.group(2).upper()
        if meridiem == "PM" and hour != 12:
            hour += 12
        if meridiem == "AM" and hour == 12:
            hour = 0
        return f"{hour:02d}:00"
    return raw_time  # fall back to raw value rather than guessing
