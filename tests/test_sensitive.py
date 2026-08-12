import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sensitive import detect_sensitive_info


def test_card_number_masked_and_high_risk():
    f = detect_sensitive_info("MSG_X", "One more thing: My card number is 4111 1111 1111 1111-92.")
    assert f is not None
    assert f.sensitivity_type == "payment_card_number"
    assert f.risk == "high"
    assert f.recommended_action == "do_not_store"
    assert "4111" not in f.masked_text  # no digit of the original number may survive
    assert f.masked_text.count("*") == 6  # fully redacted, not partially revealed
    assert not any(ch.isdigit() for ch in f.masked_text.split("is")[-1])


def test_otp_masked():
    f = detect_sensitive_info("MSG_X", "FYI: Your OTP is 482193-70. It expires in 10 minutes.")
    assert f is not None
    assert f.sensitivity_type == "one_time_password"
    assert "482193" not in f.masked_text


def test_password_masked():
    f = detect_sensitive_info("MSG_X", "Use password BlueRiver#29-81 to sign in to the test account.")
    assert f is not None
    assert f.sensitivity_type == "password"
    assert "BlueRiver#29-81" not in f.masked_text


def test_non_sensitive_message_returns_none():
    f = detect_sensitive_info("MSG_X", "The cafeteria closes at 8 PM.")
    assert f is None


def test_masking_preserves_surrounding_sentence():
    f = detect_sensitive_info("MSG_X", "One more thing: My card number is 4111 1111 1111 1111-92.")
    assert f.masked_text.startswith("One more thing: My card number is")
    assert f.masked_text.endswith(".")


def test_health_note_flagged_low_risk():
    f = detect_sensitive_info("MSG_X", "My recent test result says vitamin D deficiency-97.")
    assert f is not None
    assert f.risk == "low"
