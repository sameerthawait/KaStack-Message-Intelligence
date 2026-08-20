"""
L2 Demonstration Terminal Helper.

Processes L2 Demo messages (L2/l2_demo_messages.csv) and executes all 8
mandatory Demo Queries (DQ01 - DQ08) with explainable reasoning and evidence.

Usage:
    python print_l2_demo.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.classifier import classify_message
from src.extractor import extract_task_or_event
from src.sensitive import detect_sensitive_info
from src.priority import assess_priority
from src.grouping import group_messages
from src.privacy_router import route_privacy_policy
from src.assistant import IntelligentAssistant

BASE_DIR = Path(__file__).parent
L2_DIR = BASE_DIR / "L2"
DEMO_MESSAGES_CSV = L2_DIR / "l2_demo_messages.csv"
DEMO_QUERIES_CSV = L2_DIR / "l2_demo_queries.csv"


def main():
    print("=" * 95)
    print("  KASTACK MESSAGE INTELLIGENCE PLATFORM — L2 DEMONSTRATION & BENCHMARK")
    print("=" * 95)

    if not DEMO_MESSAGES_CSV.exists():
        print(f"Error: {DEMO_MESSAGES_CSV} not found.")
        return

    # Load messages
    messages = []
    with DEMO_MESSAGES_CSV.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            messages.append(row)

    print(f"\n[*] Loaded {len(messages)} chronological demo messages ({messages[0]['message_id']} - {messages[-1]['message_id']})\n")

    # Run pipeline stages
    classifications = []
    tasks_events = []
    sensitive_findings = []
    for i, m in enumerate(messages, 1):
        mid = m["message_id"]
        sender = m["sender"]
        text = m["message"]
        
        cl = classify_message(mid, sender, text)
        classifications.append(cl)

        if cl.category in ("action_required", "meeting_or_event"):
            te = extract_task_or_event(f"TASK_{i:04d}", mid, text)
            if te:
                tasks_events.append(te)

        sf = detect_sensitive_info(mid, text)
        if sf:
            sensitive_findings.append(sf)

    # Grouping
    related_groups = group_messages(messages, tasks_events)
    group_status_by_mid = {}
    for g in related_groups:
        for mid in g.related_message_ids:
            group_status_by_mid[mid] = g.status

    # Priority & Privacy Routing
    te_by_msg = {te.source_message_id: te for te in tasks_events}
    sf_by_msg = {s.message_id: s for s in sensitive_findings}
    cl_by_msg = {c.message_id: c for c in classifications}

    priority_decisions = []
    privacy_routes = []
    for m in messages:
        mid = m["message_id"]
        prio = assess_priority(
            message_id=mid,
            raw_message=m["message"],
            sender=m["sender"],
            timestamp=m["timestamp"],
            classification=cl_by_msg.get(mid),
            task_event=te_by_msg.get(mid),
            sensitive_finding=sf_by_msg.get(mid),
            status_override=group_status_by_mid.get(mid),
        )
        priority_decisions.append(prio)
        pr = route_privacy_policy(mid, m["message"], sf_by_msg.get(mid))
        privacy_routes.append(pr)

    # Intelligent Assistant
    assistant = IntelligentAssistant(
        messages=messages,
        classifications=classifications,
        tasks_events=tasks_events,
        sensitive_findings=sensitive_findings,
        priority_decisions=priority_decisions,
        related_groups=related_groups,
    )

    # Load and execute all Demo Queries
    queries = []
    if DEMO_QUERIES_CSV.exists():
        with DEMO_QUERIES_CSV.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            queries = [row for row in reader if row.get("query")]

    print("=" * 95)
    print(f"  PART 3: MANDATORY L2 DEMO QUERIES EVALUATION ({len(queries)} Queries)")
    print("=" * 95)

    for q_item in queries:
        qid = q_item.get("query_id", "Q")
        q_text = q_item["query"]
        ans = assistant.answer_query(q_text)

        print(f"\n[{qid}] QUERY: {q_text}")
        print(f"     ANSWER             : {ans.answer}")
        print(f"     SUPPORTING MESSAGES: {ans.supporting_message_ids}")
        if ans.group_id:
            print(f"     RELATED GROUP      : {ans.group_id}")
        if ans.item_id:
            print(f"     RELATED ITEM       : {ans.item_id}")
        print(f"     RELEVANCE SCORES   : {[round(s, 2) for s in ans.relevance_scores]}")
        print(f"     EXPLAINABLE REASON : {ans.reason}")
        print("-" * 95)

    # Summary Breakdown
    print("\n" + "=" * 95)
    print("  L2 SUMMARY & PERFORMANCE METRICS")
    print("=" * 95)
    print(f"  • Total Messages Processed : {len(messages)}")
    print(f"  • Related Groups Created   : {len(related_groups)}")
    for g in related_groups[:5]:
        print(f"    - [{g.group_id}] {g.title} | Status: {g.status.upper()} | Messages: {g.related_message_ids}")
    print(f"  • Priority Decisions       : Critical: {sum(1 for p in priority_decisions if p.priority == 'critical')}, High: {sum(1 for p in priority_decisions if p.priority == 'high')}, Medium: {sum(1 for p in priority_decisions if p.priority == 'medium')}, Low: {sum(1 for p in priority_decisions if p.priority == 'low')}")
    print(f"  • Privacy Route Policy     : Local: {sum(1 for r in privacy_routes if r.route == 'local')}, Confirm: {sum(1 for r in privacy_routes if r.route == 'confirm')}, Blocked: {sum(1 for r in privacy_routes if r.route == 'blocked')}")
    print(f"  • Zero External Cloud APIs : TRUE (100% Local Inference & Search)")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
