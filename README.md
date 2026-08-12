# KaStack Labs — AI/ML Engineer Intern Assignment

Message classification, task/event extraction, and sensitive-information
detection over a 900-message dataset.

## AI-tool usage declaration

Code was written with assistance from Claude (Anthropic), used as a coding
assistant for scaffolding and boilerplate. All classification/extraction
logic, regex patterns, and category decisions were derived directly from
manually analyzing the actual dataset (see "How this works" below) — not
generated blind or copied from an external source. No message content from
`messages.csv` was ever sent to an external AI/LLM API; all processing is
local, rule-based Python with no model inference calls at runtime.

## How message classification works (Part 1)

**Approach: deterministic rule-based classification, not a trained model
or embedding similarity search.**

Before writing any rules, I enumerated all 900 messages and stripped their
filler prefixes (`FYI:`, `Quick update:`, `Can you help?`, etc. — see
`src/text_utils.py`). This reduced the dataset to **530 unique core
templates**, built from roughly 24 distinct sentence patterns (dates,
times, names, and codes vary; sentence structure does not). Given that,
a small ordered set of regex rules (`src/classifier.py`) can match the
data with high precision — every one of the 900 messages hits a *named*
rule (verified — zero fall through to the unmatched-case default), so
every classification is traceable to one explicit, human-readable reason.

Rules are checked in priority order:
1. **Sensitive information** first (safety-critical — a message must never
   be miscategorized *away* from this label)
2. **Promotional** (sender = `Promotions`, or a `Use code SAVE##` pattern)
3. **Meeting or Event** (explicit date+time+location templates)
4. **Action Required** (explicit task+deadline templates)
5. **Personal Information** (`For my profile,`, `Personal note:`, `Remember
   that`, etc.)
6. **General Information** (matched against the known set of factual
   status-update templates found in the data; anything genuinely unmatched
   falls back to this category at reduced confidence, `general_information`
   @ 0.6, rather than guessing another label)

A lightweight keyword-scoring fallback (`_fallback_classification`) exists
for text that matches no rule at all — kept intentionally simple rather
than a black-box model, per the assignment's emphasis on being able to
explain every decision.

**Trade-off, stated plainly:** this is precise *because the dataset is
templated*. It would not generalize to arbitrary free-form messages
without adding a statistical layer (e.g., local sentence-transformer
embeddings scored against category prototype sentences) for text that
matches no rule. That's a reasonable next iteration, not something this
submission needed given the actual data.

Two categories were added implicitly as documented judgment calls (not
new top-level categories — folded into the existing six with a lower
confidence score and explanatory reason):
- Vague/ambiguous scheduling language ("Let us meet sometime next week.",
  "The review could be Friday afternoon.") — classified as
  `meeting_or_event` at confidence 0.5–0.55, explicitly reasoned as
  low-confidence in the output.
- "`<Name>` asked whether..." messages — classified as
  `general_information` at confidence 0.55, flagged as arguably also
  readable as an implicit action request.

## How tasks and events are extracted (Part 2)

`src/extractor.py` runs only on messages already classified
`action_required` or `meeting_or_event`. Each extraction pattern mirrors
one classifier template, so a field is only ever filled if it's literally
present in the text — **the hard rule "do not guess missing information"
is enforced structurally**, not just by prompt instruction: fields that
can't be read from the text are set to `null` and their names are also
added to an `unresolved_fields` list for quick auditing.

Example of the "do not guess" rule in practice: `"The review could be
Friday afternoon."` has no ISO date anywhere in the message. The
extractor does **not** infer a date (e.g. it will not compute "next
Friday" from the message timestamp) — `deadline` stays `null`,
`unresolved_fields` includes `"deadline"`.

Priority is not present as a signal anywhere in the 900 messages (no
"urgent"/"low priority" wording), so every extracted item defaults to
`"medium"` — a documented assumption, not a per-message guess.

## How sensitive information is detected and masked (Part 3)

`src/sensitive.py` scans **every** message, independent of its Part 1
category — a sensitive value could in principle appear inside a message
of any category, so restricting the scan to messages already labeled
`sensitive_information` would be a weaker guarantee.

Detected types: one-time passwords, passwords, payment card numbers, bank
account numbers, authentication/access tokens, account recovery codes,
government/org ID numbers, private phone numbers, private home addresses.
Each match is masked in place (digits redacted except the last 2 for
some context, full redaction for opaque tokens/passwords) while the rest
of the sentence is preserved, and tagged with a risk level
(`high`/`medium`/`low`) and a recommended action
(`do_not_store` / `ask_for_confirmation`).

**Judgment call, documented:** `"My recent test result says vitamin D
deficiency..."` is flagged at `risk: low` even though health data isn't
in the assignment's explicit list of sensitive types (passwords,
OTPs/PINs, bank/payment details, tokens, ID details, addresses/contact
details). I chose to still flag it — silently ignoring plausible personal
health data seemed like the wrong default for a "protect sensitive data"
system — but I ranked it well below the explicitly-listed types.

## Assumptions and limitations

- **Dataset structure is trusted as-is**: chronological order is verified
  by sorting on `timestamp` defensively (not just assumed), but message
  content is trusted to follow the observed templates.
- **Rule-based approach is dataset-specific.** See the classification
  trade-off note above — this is a deliberate choice given the data,
  not a claim that regex is generally superior to ML for this problem
  class.
- **No priority signal exists in the data**, so all extracted items are
  `medium` priority by convention, not inference.
- **The health-information flag is a judgment call**, not a literal
  requirement of the brief — documented above.
- **The `Maya asked whether...` template** is genuinely ambiguous between
  `general_information` and an implicit action request; classified as the
  former at reduced confidence (0.55) and called out here rather than
  silently picked.
- **No PII beyond what's synthetic in this dataset was validated** (e.g.
  no real Luhn checksum validation on the card-number pattern) since the
  card numbers in this dataset are fictional test values (`4111 1111
  1111 1111...`, the standard Visa test-card prefix).

## Project structure

```
src/
  models.py        # typed result records (Classification, TaskEvent, SensitiveFinding)
  text_utils.py     # filler-prefix stripping, time normalization
  classifier.py     # Part 1
  extractor.py       # Part 2
  sensitive.py       # Part 3
  pipeline.py         # orchestrates all three parts over the CSV
tests/                # pytest unit tests for all three parts
api/main.py           # FastAPI wrapper + minimal HTML dashboard (cloud demo)
run_pipeline.py        # CLI entrypoint
Dockerfile
```

## Running it

```bash
pip install -r requirements.txt

# Run the full pipeline over the dataset (place messages.csv in data/, or pass a path)
python run_pipeline.py path/to/messages.csv output/

# Run tests
pytest tests/ -v

# Run the API locally
uvicorn api.main:app --reload
# then open http://localhost:8000
```

Output files (`output/classifications.json`, `output/tasks_events.json`,
`output/sensitive_findings.json`, `output/summary.json`) are gitignored —
regenerate them locally; they're derived from the dataset which must not
be published per the assignment rules.

## Deployment

The `Dockerfile` builds a container running the FastAPI app on `$PORT`
(compatible with Render/Railway/Fly.io's standard convention). No dataset
is baked into the image — the cloud demo works via the "upload CSV" form
on the dashboard, so the supplied dataset never has to leave your machine
except at demo time, and never gets committed to the container image or
git history.
