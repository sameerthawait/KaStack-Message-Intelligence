"""
Part 1: Message classification.

APPROACH
--------
The dataset (see README) is generated from a fixed, small set of message
templates wrapped in filler prefixes and a sender field. Because the
templates are finite and known (verified by enumerating all 900 messages
before writing a single rule), a deterministic regex/keyword classifier
gives higher accuracy and is fully auditable — every decision traces to
one named rule — versus a statistical/embedding classifier, which would
add opacity and a training/calibration burden for no accuracy benefit on
data this structured.

Trade-off (documented per assignment instructions): this approach is
precise here because the input space is templated. It would NOT
generalize well to arbitrary free-form messages without adding a
statistical fallback (e.g. embedding similarity against category
prototypes) for text that doesn't match any rule. That fallback path
exists below (`_fallback_classification`) but is intentionally simple
(keyword scoring) rather than a heavyweight model, since the assignment
values transparency over raw accuracy on out-of-template text.

Rules are checked in priority order. Order encodes precedence for
messages that could plausibly match more than one category (e.g. a
sensitive value should never be miscategorized as "personal information"
just because it starts with "My ...").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.models import Classification
from src.text_utils import strip_filler_prefixes

CATEGORIES = [
    "action_required",
    "meeting_or_event",
    "personal_information",
    "general_information",
    "promotional",
    "sensitive_information",
]


@dataclass(frozen=True)
class ClassificationRule:
    rule_id: str
    category: str
    confidence: float
    pattern: re.Pattern
    reason: str


def _r(rule_id, category, confidence, pattern, reason) -> ClassificationRule:
    return ClassificationRule(rule_id, category, confidence, re.compile(pattern, re.IGNORECASE), reason)


RULES: list[ClassificationRule] = [
    # --- Sensitive information (checked first: safety takes precedence) ---
    _r("sens_otp", "sensitive_information", 0.97, r"\bOTP\s+is\s+[\d\-]{4,10}",
       "Message discloses a one-time password."),
    _r("sens_password", "sensitive_information", 0.97, r"\bpassword\s+\S+\s+to\s+sign\s+in",
       "Message discloses a literal account password."),
    _r("sens_card", "sensitive_information", 0.97, r"\bcard number is [\d\s\-]{12,25}",
       "Message discloses a payment card number."),
    _r("sens_bank", "sensitive_information", 0.97, r"\bbank account number [\d\-]{8,20}",
       "Message discloses a bank account number."),
    _r("sens_token", "sensitive_information", 0.96, r"\baccess token is \S+",
       "Message discloses an authentication token."),
    _r("sens_recovery", "sensitive_information", 0.96, r"\baccount recovery code is \S+",
       "Message discloses an account recovery code."),
    _r("sens_id", "sensitive_information", 0.9, r"\bidentification number is \S+",
       "Message discloses a personal identification number."),
    _r("sens_phone", "sensitive_information", 0.88, r"\bcontact me on [\d\s\-]{8,15}",
       "Message discloses a private phone number."),
    _r("sens_address", "sensitive_information", 0.9, r"\bhome address is",
       "Message discloses a private home address."),

    # --- Promotional ---
    _r("promo_code", "promotional", 0.95, r"\bUse code SAVE\d+\b",
       "Message contains a promotional discount code."),
    _r("promo_offer", "promotional", 0.85,
       r"\b(limited-time offer|flash sale|festival discount|cashback|reward points|"
       r"premium plan|subscription and save|student plan|food-delivery coupon)\b",
       "Message uses promotional/marketing language (offer, discount, or sale)."),

    # --- Meeting or event (has a date/time/location tied to a named event) ---
    _r("event_calendar_update", "meeting_or_event", 0.95,
       r"^Calendar update: .+, \d{4}-\d{2}-\d{2} at \d{1,2}:\d{2},",
       "Message is a calendar update naming an event, date, time, and location."),
    _r("event_reminder", "meeting_or_event", 0.95,
       r"^Reminder: .+ happens on \d{4}-\d{2}-\d{2} at \d{1,2}:\d{2} in",
       "Message is a reminder for a named event with a specific date, time, and location."),
    _r("event_join", "meeting_or_event", 0.95,
       r"^Please join the .+ on \d{4}-\d{2}-\d{2}, \d{1,2}:\d{2} at",
       "Message invites the recipient to a scheduled event with date, time, and location."),
    _r("event_available", "meeting_or_event", 0.93,
       r"^Are you available for the .+ at \d{1,2}:\d{2} on \d{4}-\d{2}-\d{2}\? Location:",
       "Message asks about availability for a specific scheduled event."),
    _r("event_scheduled", "meeting_or_event", 0.95,
       r"is scheduled for \d{4}-\d{2}-\d{2} at \d{1,2}:\d{2} in",
       "Message states that a named event is scheduled for a specific date/time/location."),
    _r("event_vague_meet", "meeting_or_event", 0.55,
       r"^Let us meet sometime next week\.?$",
       "Message proposes a meeting but gives no confirmed date, time, or location "
       "(low confidence: could also be read as an informal action request)."),
    _r("event_vague_review", "meeting_or_event", 0.5,
       r"^The review could be .+\.?$",
       "Message tentatively references a meeting/review timing without a firm date "
       "(low confidence — ambiguous wording)."),

    # --- Action required (explicit task + deadline, or clear imperative) ---
    _r("action_dont_forget", "action_required", 0.93,
       r"^Don't forget to .+; deadline is \d{4}-\d{2}-\d{2}",
       "Message is an explicit reminder to complete a task with a stated deadline."),
    _r("action_can_you_before", "action_required", 0.93,
       r"^Can you .+ before \d{4}-\d{2}-\d{2}\??",
       "Message directly asks the recipient to complete a task before a deadline."),
    _r("action_need_you_by", "action_required", 0.93,
       r"^I need you to .+ by \d{4}-\d{2}-\d{2}",
       "Message directly assigns a task with a stated deadline."),
    _r("action_please_by", "action_required", 0.93,
       r"^Please (?!join|note|complete the Python exercise\b).+ by \d{4}-\d{2}-\d{2}",
       "Message politely requests a task be completed by a stated deadline."),
    _r("action_please_complete_by", "action_required", 0.93,
       r"^Please complete the .+ by \d{4}-\d{2}-\d{2}",
       "Message requests a task be completed by a stated deadline."),
    _r("action_due_on", "action_required", 0.9,
       r"^.+ is due on \d{4}-\d{2}-\d{2}",
       "Message states a task with an explicit due date."),
    _r("action_vague_soon", "action_required", 0.55,
       r"^Could you send it soon\??$",
       "Message requests an action but the deadline ('soon') is not a resolvable date."),
    _r("action_call_person", "action_required", 0.8,
       r"^Please call [A-Z][a-z]+ when you are free\.?$",
       "Message asks the recipient to contact a named person, with no resolvable "
       "deadline or time given."),
    _r("action_if_possible", "action_required", 0.6,
       r"^If possible, review the file before the meeting\.?$",
       "Message requests a task (review the file) with a relative, unresolved deadline "
       "('before the meeting')."),

    # --- Personal information (preferences/facts about the sender, no task) ---
    _r("personal_profile", "personal_information", 0.9, r"^For my profile,",
       "Message explicitly shares personal profile information."),
    _r("personal_note", "personal_information", 0.9, r"^Personal note:",
       "Message is explicitly labeled as a personal note."),
    _r("personal_remember", "personal_information", 0.85, r"^Remember that\b",
       "Message shares a personal preference/fact for the recipient to remember."),
    _r("personal_just_so_you_know", "personal_information", 0.85, r"^Just so you know,",
       "Message shares a personal preference or fact about the sender."),
    _r("personal_might_prefer", "personal_information", 0.8, r"^I might prefer\b",
       "Message shares a tentative personal preference."),
    _r("personal_health_note", "personal_information", 0.75,
       r"^My recent test result says\b",
       "Message shares a personal health detail about the sender. Categorized as "
       "personal information for Part 1; separately flagged at low risk by the "
       "Part 3 sensitive-data scan since health data isn't in the assignment's "
       "listed sensitive types but is treated cautiously (see README)."),

    # --- General information (explicit factual-statement templates, matched by
    # exact known text so confidence reflects real certainty rather than the
    # 0.6 no-rule-matched default) ---
    _r("general_known_statement", "general_information", 0.85, r"^(?:"
       r"The building entrance has moved temporarily|"
       r"The cafeteria closes at 8 PM|"
       r"The event registration desk opens at 9 AM|"
       r"The laptop battery is fully charged|"
       r"The library has extended weekend hours|"
       r"The new Python version is available|"
       r"The office Wi-Fi will be under maintenance tonight|"
       r"The project folder was reorganized|"
       r"The report may be needed tomorrow|"
       r"The report template has been updated|"
       r"The shuttle leaves every thirty minutes|"
       r"The support team changed its working hours|"
       r"The training material is on the portal|"
       r"The weather forecast says light rain|"
       r"The webinar recording is now available|"
       r"Tomorrow is a public holiday|"
       r"I will send the login details separately"
       r")\.?$",
       "Message matches a known factual/status-update template with no task, "
       "event, sensitive value, or personal-preference content."),
    _r("general_third_party_question", "general_information", 0.55,
       r"^([A-Z][a-z]+) asked whether\b",
       "Message reports that a named person asked a question, but assigns no "
       "task or deadline to the recipient (low confidence: could alternatively "
       "be read as an implicit action request — see README)."),

    # --- Promotional sender override (applied in code as a fallback, see classify()) ---
]

_PROMO_SENDER = "promotions"


def _fallback_classification(sender: str, core_text: str) -> Classification | None:
    """
    Lightweight keyword-scoring fallback for text that matched no rule above.
    Kept intentionally simple/transparent rather than a black-box model — see
    module docstring for rationale.
    """
    text_lower = core_text.lower()

    if sender.lower() == _PROMO_SENDER:
        return ("promotional", 0.7, "Sender is the 'Promotions' account, even though no "
                "discount-code pattern matched.", "fallback_promo_sender")

    task_words = ["submit", "review", "send", "complete", "confirm", "reply", "upload",
                  "verify", "back up", "renew", "call", "pay", "email the"]
    if any(w in text_lower for w in task_words):
        return ("action_required", 0.5,
                "Message contains task-like language but did not match a known template; "
                "classified with reduced confidence.", "fallback_keyword_task")

    event_words = ["meeting", "scheduled", "calendar", "appointment", "workshop", "seminar"]
    if any(w in text_lower for w in event_words):
        return ("meeting_or_event", 0.5,
                "Message references a meeting/event term but did not match a known "
                "scheduling template; classified with reduced confidence.",
                "fallback_keyword_event")

    return None


def classify_message(message_id: str, sender: str, raw_message: str) -> Classification:
    core = strip_filler_prefixes(raw_message)

    for rule in RULES:
        if rule.pattern.search(core):
            return Classification(
                message_id=message_id,
                category=rule.category,
                confidence=rule.confidence,
                reason=rule.reason,
                rule_id=rule.rule_id,
            )

    fallback = _fallback_classification(sender, core)
    if fallback:
        category, confidence, reason, rule_id = fallback
        return Classification(message_id, category, confidence, reason, rule_id)

    # Final default: no task, no event, no sensitive value, no promo signal,
    # no personal-preference phrasing -> treated as a general informational
    # statement. Confidence is moderate because "no rule matched" is a real
    # possibility of a gap in rule coverage, not certainty about the category.
    return Classification(
        message_id=message_id,
        category="general_information",
        confidence=0.6,
        reason="Message is a factual statement with no task, event, sensitive value, "
               "personal preference, or promotional signal detected; defaulted to "
               "general information.",
        rule_id="default_general",
    )
