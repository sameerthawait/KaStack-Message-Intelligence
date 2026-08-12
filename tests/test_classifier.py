import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import classify_message


def test_action_required_deadline():
    c = classify_message("MSG_X", "Ishaan", "Can you review the privacy checklist before 2026-09-09?")
    assert c.category == "action_required"
    assert c.confidence >= 0.9


def test_meeting_calendar_update():
    c = classify_message("MSG_X", "Meera", "For today: Calendar update: family dinner, 2026-09-19 at 10:00, the library.")
    assert c.category == "meeting_or_event"


def test_personal_information():
    c = classify_message("MSG_X", "Meera", "For my profile, my emergency contact is my brother.")
    assert c.category == "personal_information"


def test_general_information():
    c = classify_message("MSG_X", "Meera", "Important: The laptop battery is fully charged.")
    assert c.category == "general_information"
    assert c.confidence >= 0.8  # should hit the explicit template rule, not the 0.6 default


def test_promotional_code():
    c = classify_message("MSG_X", "Promotions", "Can you help? Special festival discount on clothing. Use code SAVE17.")
    assert c.category == "promotional"


def test_sensitive_card_number():
    c = classify_message("MSG_X", "Meera", "One more thing: My card number is 4111 1111 1111 1111-92.")
    assert c.category == "sensitive_information"
    assert c.confidence >= 0.9


def test_sensitive_otp():
    c = classify_message("MSG_X", "Ananya", "FYI: Your OTP is 482193-70. It expires in 10 minutes.")
    assert c.category == "sensitive_information"


def test_ambiguous_case_low_confidence():
    """MSG_0037 in the real dataset — deliberately ambiguous, should be flagged
    with reduced confidence rather than a confident wrong answer."""
    c = classify_message("MSG_0037", "Kabir", "One more thing: The review could be Friday afternoon.")
    assert c.confidence < 0.7


def test_no_prefix_confusion_please_join_is_event_not_action():
    """'Please join the X on DATE...' must not be caught by the generic
    'Please ... by DATE' action rule."""
    c = classify_message(
        "MSG_X", "Ishaan",
        "Just checking—Please join the internship orientation on 2026-09-18, 13:00 at Conference Room 2.",
    )
    assert c.category == "meeting_or_event"
