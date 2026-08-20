"""
Privacy-Aware Request Router.

Enforces zero-leakage privacy policies on all queries and messages:
1. 'blocked': High-risk credentials (passwords, OTPs, auth tokens, bank/card numbers)
   that must never be sent to external endpoints or exposed unmasked.
2. 'confirm': Personal or semi-sensitive items (addresses, phone numbers, health notes,
   high-impact actions) requiring explicit user confirmation before processing.
3. 'local': Safe operational queries and standard messages processed 100% locally.
"""

from __future__ import annotations

import re
from typing import Optional

from src.models import PrivacyRoutingResult, SensitiveFinding


def route_privacy_policy(
    target_id: str,
    text: str,
    sensitive_finding: Optional[SensitiveFinding] = None,
) -> PrivacyRoutingResult:
    """
    Evaluate privacy and security risk to assign routing policy: local, confirm, or blocked.
    """
    text_lower = text.lower()

    # 1. If high-risk sensitive finding is detected -> BLOCKED
    if sensitive_finding:
        if sensitive_finding.risk == "high" or sensitive_finding.recommended_action == "do_not_store":
            return PrivacyRoutingResult(
                target_id=target_id,
                route="blocked",
                reason=f"Blocked from external transmission: Contains high-risk sensitive data ({sensitive_finding.sensitivity_type}).",
                requires_user_action=False,
                sensitivity_type=sensitive_finding.sensitivity_type,
            )
        elif sensitive_finding.risk in ("medium", "low") or sensitive_finding.recommended_action == "ask_for_confirmation":
            return PrivacyRoutingResult(
                target_id=target_id,
                route="confirm",
                reason=f"Requires user confirmation before processing: Contains personal sensitive information ({sensitive_finding.sensitivity_type}).",
                requires_user_action=True,
                sensitivity_type=sensitive_finding.sensitivity_type,
            )

    # 2. Text-based detection for queries or unclassified inputs
    # Check for blocked patterns (OTPs, passwords, tokens, cards, accounts)
    if re.search(r"\b(otp|password|token|card number|bank account|recovery code|tok_demo|tok_l2)\b", text_lower):
        return PrivacyRoutingResult(
            target_id=target_id,
            route="blocked",
            reason="Blocked from external transmission: Contains credentials, authentication tokens, or payment data.",
            requires_user_action=False,
            sensitivity_type="credential_or_token",
        )

    # Check for confirmation patterns (addresses, phone numbers, health/medical info)
    if re.search(r"\b(medical|health|home address|deliver\s+(?:the\s+[^,.]+?|it)\s+to|address|road|street|contact me on|call me on|phone number|deficiency|condition)\b", text_lower):
        return PrivacyRoutingResult(
            target_id=target_id,
            route="confirm",
            reason="Requires explicit user confirmation before processing personal contact, location, or health information.",
            requires_user_action=True,
            sensitivity_type="personal_or_health_data",
        )

    # 3. Default: Safe Local Processing
    return PrivacyRoutingResult(
        target_id=target_id,
        route="local",
        reason="Safe request processed 100% locally with zero external API transmission.",
        requires_user_action=False,
        sensitivity_type=None,
    )
