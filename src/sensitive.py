"""
Part 3: Sensitive information detection and masking.

Design decision: this runs on EVERY message, independent of the Part 1
category, not just on messages already classified as "Sensitive
Information". A sensitive value could in principle appear inside a
message of any category; scanning everything is the safer default for
a system whose stated purpose is to protect sensitive data.

Detected sensitive values are fully redacted to "******" in-place while
preserving surrounding text for context, ensuring maximum privacy protection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.models import SensitiveFinding


@dataclass(frozen=True)
class SensitiveRule:
    rule_id: str
    sensitivity_type: str
    pattern: re.Pattern
    risk: str
    recommended_action: str
    reason_template: str
    mask_group: int = 1  # which capture group to mask; 0 = whole match


def _rule(rule_id, sensitivity_type, pattern, risk, action, reason, mask_group=1):
    return SensitiveRule(
        rule_id=rule_id,
        sensitivity_type=sensitivity_type,
        pattern=re.compile(pattern, re.IGNORECASE),
        risk=risk,
        recommended_action=action,
        reason_template=reason,
        mask_group=mask_group,
    )


# Ordered: more specific / higher-risk patterns first, so a value that could
# match two rules (e.g. a long digit string) is classified by the most
# specific one, not a generic fallback.
SENSITIVE_RULES: list[SensitiveRule] = [
    _rule(
        "otp",
        "one_time_password",
        r"\bOTP\s+is\s+([\d\-]{4,10})",
        "high",
        "do_not_store",
        "Message contains a one-time password (OTP), which is only valid briefly "
        "but grants immediate account access if intercepted.",
    ),
    _rule(
        "password",
        "password",
        r"\bpassword\s+(\S+)\s+to\s+sign\s+in",
        "high",
        "do_not_store",
        "Message contains a literal account password.",
    ),
    _rule(
        "card_number",
        "payment_card_number",
        r"\bcard number is ([\d\s\-]{12,25})",
        "high",
        "do_not_store",
        "Message contains what appears to be a full payment card number.",
    ),
    _rule(
        "bank_account",
        "bank_account_number",
        r"\bbank account number ([\d\-]{8,20})",
        "high",
        "do_not_store",
        "Message contains a bank account number.",
    ),
    _rule(
        "auth_token",
        "authentication_token",
        r"\baccess token is (\S+)",
        "high",
        "do_not_store",
        "Message contains an authentication/access token that could be used to "
        "impersonate the user in an external service.",
    ),
    _rule(
        "recovery_code",
        "account_recovery_code",
        r"\baccount recovery code is (\S+)",
        "high",
        "do_not_store",
        "Message contains an account recovery code, which can be used to bypass "
        "normal authentication.",
    ),
    _rule(
        "id_number",
        "government_or_org_id",
        r"\bidentification number is (\S+)",
        "medium",
        "ask_for_confirmation",
        "Message contains a personal identification number.",
    ),
    _rule(
        "phone_number",
        "private_contact_number",
        r"\bcontact me on ([\d\s\-]{8,15})",
        "medium",
        "ask_for_confirmation",
        "Message contains a private phone number.",
    ),
    _rule(
        "home_address",
        "private_address",
        r"\bhome address is (.+?)(?:\.\s*$|$)",
        "medium",
        "ask_for_confirmation",
        "Message contains a private home address.",
    ),
]

# Health/medical info is not in the assignment's explicit sensitive-type list,
# but it's a reasonable borderline case worth flagging at low risk rather than
# silently ignoring — documented in the README as a deliberate judgment call.
_HEALTH_RULE = _rule(
    "health_note",
    "health_information",
    r"\btest result says (.+?)(?:\.\s*$|$)",
    "low",
    "ask_for_confirmation",
    "Message contains a personal health detail. Not in the assignment's listed "
    "sensitive-data categories, but treated as low-risk sensitive personal data "
    "out of caution (see README limitations).",
)
SENSITIVE_RULES.append(_HEALTH_RULE)


def _mask_message(message: str, match: re.Match, mask_group: int) -> str:
    """
    Replace only the sensitive span with a fixed-length mask, keeping the
    surrounding sentence for context.

    Full redaction (no partial digits left visible), matching the
    assignment's own example ("Your OTP is ******") — a value like a card
    or account number is still identifying even with only its last couple
    digits shown, so this errs conservative rather than balancing
    readability against exposure.
    """
    span_group = mask_group if mask_group != 0 else 0
    start, end = match.span(span_group)
    masked_value = "******"
    return message[:start] + masked_value + message[end:]


def detect_sensitive_info(message_id: str, message: str) -> Optional[SensitiveFinding]:
    """
    Scan a single message for sensitive content.

    Returns the first (highest-priority) match as a SensitiveFinding, or
    None if nothing matched. Rules are ordered most-specific-first so we
    don't need to score/rank overlapping matches.
    """
    for rule in SENSITIVE_RULES:
        match = rule.pattern.search(message)
        if not match:
            continue
        masked = _mask_message(message, match, rule.mask_group)
        return SensitiveFinding(
            message_id=message_id,
            sensitivity_type=rule.sensitivity_type,
            risk=rule.risk,
            masked_text=masked,
            recommended_action=rule.recommended_action,
            reason=rule.reason_template,
        )
    return None
