"""
Terminal helper to display the 15 Mandatory Demo Message IDs and their pipeline results.
Usage: python print_mandatory_demo.py [output_dir]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_CSV = BASE_DIR / "data" / "messages.csv"
MANDATORY_CSV = BASE_DIR / "mandatory_demo_ids.csv"
DEFAULT_OUTPUT = BASE_DIR / "output"


def _load_json(file_path: Path) -> list:
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    if not (output_dir / "classifications.json").exists():
        print(f"Error: Output directory '{output_dir}' does not contain output JSON files.")
        print("Please run 'python run_pipeline.py data/messages.csv output/' first.")
        sys.exit(1)

    mandatory_ids = []
    if MANDATORY_CSV.exists():
        with MANDATORY_CSV.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            mandatory_ids = [row["message_id"].strip() for row in reader if row.get("message_id")]
    if not mandatory_ids:
        mandatory_ids = [
            "MSG_0001", "MSG_0002", "MSG_0003", "MSG_0004", "MSG_0005",
            "MSG_0006", "MSG_0007", "MSG_0009", "MSG_0012", "MSG_0013",
            "MSG_0014", "MSG_0015", "MSG_0016", "MSG_0024", "MSG_0037"
        ]

    mandatory_set = set(mandatory_ids)

    classifications = {x["message_id"]: x for x in _load_json(output_dir / "classifications.json") if x.get("message_id") in mandatory_set}
    tasks_events = {x["source_message_id"]: x for x in _load_json(output_dir / "tasks_events.json") if x.get("source_message_id") in mandatory_set}
    sensitive_findings = {x["message_id"]: x for x in _load_json(output_dir / "sensitive_findings.json") if x.get("message_id") in mandatory_set}

    raw_messages = {}
    if DATA_CSV.exists():
        with DATA_CSV.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = row.get("message_id", "").strip()
                if mid in mandatory_set:
                    raw_messages[mid] = row

    print("=" * 90)
    print(f"  KEY BENCHMARK MESSAGE EVALUATION ({len(mandatory_ids)} Test Cases)")
    print("=" * 90)

    for i, mid in enumerate(mandatory_ids, start=1):
        raw = raw_messages.get(mid, {})
        c = classifications.get(mid, {})
        te = tasks_events.get(mid)
        sf = sensitive_findings.get(mid)

        print(f"\n[{i:02d}] Message ID: {mid} | Sender: {raw.get('sender', 'Unknown')}")
        print(f"     Raw Text : \"{raw.get('message', 'N/A')}\"")
        print(f"     [Part 1] Category  : {c.get('category', 'N/A')} (Confidence: {c.get('confidence', 'N/A')})")
        print(f"              Matched Rule: {c.get('matched_rule', 'N/A')}")
        print(f"              Reason      : {c.get('reason', 'N/A')}")

        if te:
            print(f"     [Part 2] Extracted {te.get('type', 'item').upper()}: \"{te.get('title')}\"")
            print(f"              Deadline: {te.get('deadline') or 'None'} | Time: {te.get('time') or 'None'} | Person: {te.get('person') or 'None'}")
            if te.get("unresolved_fields"):
                print(f"              Unresolved Fields: {te.get('unresolved_fields')}")
        else:
            print("     [Part 2] Task/Event : N/A")

        if sf:
            print(f"     [Part 3] Sensitive  : RISK={sf.get('risk', 'N/A').upper()} | Type={sf.get('sensitivity_type')}")
            print(f"              Masked Text: \"{sf.get('masked_text')}\"")
            print(f"              Action     : {sf.get('recommended_action')}")
        else:
            print("     [Part 3] Sensitive  : Clean (No sensitive data detected)")

        print("-" * 90)


if __name__ == "__main__":
    main()
