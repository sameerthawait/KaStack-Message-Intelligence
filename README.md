# 📖 KaStack Message Intelligence Platform

An enterprise-grade, rule-based message intelligence pipeline and interactive Glassmorphism dashboard for message classification, task/event extraction, and sensitive information protection over large-scale message streams.

---

## 🤖 AI-tool usage declaration

Code was written with assistance from AI pair-programming tools, used as a coding assistant for scaffolding, UI styling, and boilerplate. All classification/extraction logic, regex patterns, and category decisions were derived directly from manually analyzing the actual dataset (see "How this works" below) — not generated blind or copied from an external source. No message content from `messages.csv` was ever sent to an external AI/LLM API; all processing is local, deterministic, rule-based Python with zero model inference calls at runtime.

---

## 🎯 How message classification works (Part 1)

**Approach: Deterministic rule-based classification with 100% template precision, not a black-box model or embedding similarity search.**

Before writing any rules, I enumerated all 900 messages and stripped their filler prefixes (`FYI:`, `Quick update:`, `Can you help?`, etc. — see `src/text_utils.py`). This reduced the dataset to **530 unique core templates**, built from roughly 24 distinct sentence patterns (dates, times, names, and codes vary; sentence structure does not). Given that, a small ordered set of regex rules (`src/classifier.py`) matches the data with high precision — every one of the 900 messages hits a *named* rule (verified — zero fall through to the unmatched-case default), so every classification is traceable to one explicit, human-readable reason.

Rules are checked in strict priority order:
1. **Sensitive information** first (safety-critical — a message must never be miscategorized away from this label)
2. **Promotional** (sender = `Promotions`, or a `Use code SAVE##` pattern)
3. **Meeting or Event** (explicit date+time+location templates)
4. **Action Required** (explicit task+deadline templates)
5. **Personal Information** (`For my profile,`, `Personal note:`, `Remember that`, etc.)
6. **General Information** (matched against the known set of factual status-update templates found in the data; anything genuinely unmatched falls back to this category at reduced confidence, `general_information` @ 0.6, rather than guessing another label)

A lightweight keyword-scoring fallback (`_fallback_classification`) exists for text that matches no rule at all — kept intentionally simple rather than a black-box model, per the emphasis on being able to explain every decision.

---

## 📅 How tasks and events are extracted (Part 2)

`src/extractor.py` runs only on messages already classified `action_required` or `meeting_or_event`. Each extraction pattern mirrors one classifier template, so a field is only ever filled if it is literally present in the text — **the hard rule "do not guess missing information" is enforced structurally**, not just by prompt instruction: fields that cannot be read from the text are set to `null` and their names are also added to an `unresolved_fields` list for quick auditing.

**Example of the "do not guess" rule in practice:** `"The review could be Friday afternoon."` has no ISO date anywhere in the message. The extractor does **not** infer a date (e.g. it will not compute "next Friday" from the message timestamp) — `deadline` stays `null`, and `unresolved_fields` includes `"deadline"`.

Priority is not present as a signal anywhere in the 900 messages (no "urgent"/"low priority" wording), so every extracted item defaults to `"medium"` — a documented assumption, not a per-message guess.

---

## 🔒 How sensitive information is detected and masked (Part 3)

`src/sensitive.py` scans **every** message, independent of its Part 1 category — a sensitive value could in principle appear inside a message of any category, so restricting the scan to messages already labeled `sensitive_information` would be a weaker guarantee.

**Detected types:**
* One-time passwords (OTP)
* Account Passwords
* Payment card numbers
* Bank account numbers
* Authentication/access tokens
* Account recovery codes
* Government/org ID numbers
* Private phone numbers
* Private home addresses

Each match is masked in place (`******` redaction for sensitive patterns) while preserving the surrounding text context, tagged with a risk level (`high`/`medium`/`low`), and paired with a recommended action (`do_not_store` / `ask_for_confirmation`).

---

## 💡 Assumptions and limitations

- **Dataset structure is trusted as-is**: Chronological order is verified by sorting on `timestamp` defensively (not just assumed), but message content is trusted to follow the observed templates.
- **Rule-based approach is dataset-specific**: This is a deliberate choice given the structured nature of the data, ensuring 100% explainability and fast runtime execution.
- **No priority signal exists in the data**: All extracted items default to `medium` priority by convention.
- **Health-information flag is a judgment call**: `"My recent test result says vitamin D deficiency..."` is flagged at `risk: low` to prioritize user privacy safety.
- **The `Maya asked whether...` template**: Genuinely ambiguous between `general_information` and an implicit action request; classified as the former at reduced confidence (0.55).
- **No PII beyond synthetic test values**: Card numbers in this dataset are fictional test values (`4111 1111 1111 1111...`).

---

## 📁 Project Structure

```
src/
  models.py            # Typed Pydantic result records (Classification, TaskEvent, SensitiveFinding)
  text_utils.py        # Filler-prefix stripping, time normalization
  classifier.py        # Part 1: Rule-based classification engine
  extractor.py         # Part 2: Task & event extraction engine
  sensitive.py         # Part 3: Sensitive data scanner & masking engine
  pipeline.py          # Full batch processing pipeline orchestrator
tests/                 # Automated pytest suite for all three parts
api/main.py            # FastAPI backend + Glassmorphism Web Dashboard
print_mandatory_demo.py# Terminal demonstration script
render.yaml            # Render Blueprint deployment configuration
Dockerfile             # Docker container configuration
requirements.txt       # Project dependencies
```

---

## ⚡ Quickstart & Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline over the dataset
python run_pipeline.py data/messages.csv output/

# 3. Run automated tests
pytest tests/ -v

# 4. Launch the Glassmorphism Web App locally
uvicorn api.main:app --reload
# Open http://localhost:8000 in your browser
```

---

## ☁️ Cloud Deployment (Render.com)

This repository includes a `render.yaml` blueprint. To deploy live to Render:
1. Connect your GitHub repository `sameerthawait/KaStack-Message-Intelligence` to **[Render.com](https://render.com)**.
2. Select **Web Service** — Render will automatically build using `requirements.txt` and launch with `uvicorn api.main:app`.

