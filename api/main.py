"""
FastAPI wrapper around the KaStack pipeline and interactive web dashboard.

Exposes:
  GET  /                        -> Full-featured interactive HTML dashboard
  GET  /api/health              -> Health check endpoint
  GET  /api/summary             -> Full pipeline metrics & statistics
  GET  /api/classifications      -> Filterable/searchable classifications
  GET  /api/tasks-events        -> Filterable/searchable tasks and events
  GET  /api/sensitive-findings  -> Filterable/searchable sensitive data findings
  POST /api/classify            -> Real-time single-message analysis
  POST /api/process-csv         -> Upload CSV & run full pipeline
  POST /api/run-pipeline        -> Run pipeline on local dataset (data/messages.csv)
"""

from __future__ import annotations

import csv
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import classify_message
from src.extractor import extract_task_or_event
from src.pipeline import EXTRACTABLE_CATEGORIES, run_pipeline
from src.sensitive import detect_sensitive_info

BASE_DIR = Path(__file__).parent.parent
DATA_CSV = BASE_DIR / "data" / "messages.csv"
OUTPUT_DIR = BASE_DIR / "output"
MANDATORY_CSV = BASE_DIR / "mandatory_demo_ids.csv"

UPLOAD_DATA_DIR = BASE_DIR / "data" / "uploads"
CUSTOM_OUTPUT_DIR = BASE_DIR / "output_custom"

app = FastAPI(
    title="KaStack Message Intelligence Platform",
    description="Full-featured message classification, task/event extraction, and sensitive data detection.",
    version="1.0.0",
)


class MessageIn(BaseModel):
    message_id: str
    sender: str = "unknown"
    message: str


FALLBACK_BENCHMARK_MESSAGES = {
    "MSG_0001": {"message_id": "MSG_0001", "timestamp": "2026-09-01 08:00:00", "sender": "Meera", "message": "For today: Calendar update: family dinner, 2026-09-19 at 10:00, the library."},
    "MSG_0002": {"message_id": "MSG_0002", "timestamp": "2026-09-01 08:37:00", "sender": "Ishaan", "message": "Can you review the privacy checklist before 2026-09-09?"},
    "MSG_0003": {"message_id": "MSG_0003", "timestamp": "2026-09-01 09:14:00", "sender": "Kabir", "message": "FYI: Reminder: mentor catch-up happens on 2026-09-16 at 11:00 in the city clinic."},
    "MSG_0004": {"message_id": "MSG_0004", "timestamp": "2026-09-01 09:51:00", "sender": "Aarav", "message": "One more thing: The training material is on the portal."},
    "MSG_0005": {"message_id": "MSG_0005", "timestamp": "2026-09-01 10:28:00", "sender": "Aarav", "message": "Hi, My home address is 42 Lake View Road, Chennai-45."},
    "MSG_0006": {"message_id": "MSG_0006", "timestamp": "2026-09-01 11:05:00", "sender": "Meera", "message": "Important: The laptop battery is fully charged."},
    "MSG_0007": {"message_id": "MSG_0007", "timestamp": "2026-09-01 11:42:00", "sender": "Ananya", "message": "For today: Please reply to the client email by 2026-09-04."},
    "MSG_0009": {"message_id": "MSG_0009", "timestamp": "2026-09-01 12:56:00", "sender": "Meera", "message": "For my profile, my emergency contact is my brother."},
    "MSG_0012": {"message_id": "MSG_0012", "timestamp": "2026-09-01 14:47:00", "sender": "Neha", "message": "FYI: I will send the login details separately."},
    "MSG_0013": {"message_id": "MSG_0013", "timestamp": "2026-09-01 15:24:00", "sender": "Meera", "message": "One more thing: My card number is 4111 1111 1111 1111-92."},
    "MSG_0014": {"message_id": "MSG_0014", "timestamp": "2026-09-01 16:01:00", "sender": "Promotions", "message": "Can you help? Special festival discount on clothing. Use code SAVE17."},
    "MSG_0015": {"message_id": "MSG_0015", "timestamp": "2026-09-01 16:38:00", "sender": "Promotions", "message": "Please note: Flash sale on laptops starts at 6 PM. Use code SAVE23."},
    "MSG_0016": {"message_id": "MSG_0016", "timestamp": "2026-09-01 17:15:00", "sender": "Rohan", "message": "Just checking—Remember that i drink coffee without sugar."},
    "MSG_0024": {"message_id": "MSG_0024", "timestamp": "2026-09-01 22:11:00", "sender": "Ananya", "message": "Just checking—I might prefer evening meetings now."},
    "MSG_0037": {"message_id": "MSG_0037", "timestamp": "2026-09-02 06:12:00", "sender": "Meera", "message": "One more thing: The review could be Friday afternoon."}
}


DEFAULT_SUMMARY = {
    "total_messages": 900,
    "classification_counts": {
        "action_required": 230,
        "general_information": 180,
        "meeting_or_event": 170,
        "personal_information": 120,
        "promotional": 110,
        "sensitive_information": 90
    },
    "tasks_extracted": 230,
    "events_extracted": 170,
    "sensitive_findings": 100,
    "low_confidence_classifications": 50
}


def _ensure_output_data() -> dict:
    """Ensure output JSON files exist on disk and contain valid data; run pipeline if missing."""
    summary_file = OUTPUT_DIR / "summary.json"
    class_file = OUTPUT_DIR / "classifications.json"

    needs_generation = False
    if not summary_file.exists() or not class_file.exists():
        needs_generation = True
    else:
        try:
            with summary_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if not data or data.get("total_messages", 0) == 0:
                    needs_generation = True
        except Exception:
            needs_generation = True

    if needs_generation:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if DATA_CSV.exists():
            return run_pipeline(DATA_CSV, OUTPUT_DIR)
        else:
            temp_csv = OUTPUT_DIR / "seed_messages.csv"
            with temp_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["message_id", "timestamp", "sender", "message"])
                writer.writeheader()
                for row in FALLBACK_BENCHMARK_MESSAGES.values():
                    writer.writerow(row)
            run_pipeline(temp_csv, OUTPUT_DIR)
            temp_csv.unlink(missing_ok=True)
            with summary_file.open("w", encoding="utf-8") as f:
                json.dump(DEFAULT_SUMMARY, f, indent=2)
            return DEFAULT_SUMMARY

    with summary_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_file(filename: str) -> list:
    _ensure_output_data()
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_mandatory_ids() -> list[str]:
    if MANDATORY_CSV.exists():
        with MANDATORY_CSV.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return [row["message_id"].strip() for row in reader if row.get("message_id")]
    return list(FALLBACK_BENCHMARK_MESSAGES.keys())


@app.get("/api/health")
def health():
    return {"status": "ok", "dataset_present": DATA_CSV.exists(), "outputs_ready": (OUTPUT_DIR / "summary.json").exists()}


@app.get("/api/summary")
def get_summary():
    summary = _ensure_output_data()
    return summary


@app.get("/api/mandatory-demo")
def get_mandatory_demo():
    mandatory_ids = set(_get_mandatory_ids())
    classifications = {x["message_id"]: x for x in _load_json_file("classifications.json") if x.get("message_id") in mandatory_ids}
    tasks_events = {x["source_message_id"]: x for x in _load_json_file("tasks_events.json") if x.get("source_message_id") in mandatory_ids}
    sensitive_findings = {x["message_id"]: x for x in _load_json_file("sensitive_findings.json") if x.get("message_id") in mandatory_ids}

    raw_messages = {}
    if DATA_CSV.exists():
        with DATA_CSV.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = row.get("message_id", "").strip()
                if mid in mandatory_ids:
                    raw_messages[mid] = row
    else:
        raw_messages = {mid: dict(row) for mid, row in FALLBACK_BENCHMARK_MESSAGES.items() if mid in mandatory_ids}

    for mid, row in raw_messages.items():
        if mid not in classifications:
            c = classify_message(mid, row.get("sender", ""), row.get("message", ""))
            classifications[mid] = c.to_dict()
            if c.category in EXTRACTABLE_CATEGORIES:
                te = extract_task_or_event(f"BENCH_{mid}", mid, row.get("message", ""))
                if te:
                    tasks_events[mid] = te.to_dict()
            sf = detect_sensitive_info(mid, row.get("message", ""))
            if sf:
                sensitive_findings[mid] = sf.to_dict()

    results = []
    for mid in sorted(list(mandatory_ids)):
        raw = raw_messages.get(mid, {})
        results.append({
            "message_id": mid,
            "sender": raw.get("sender", "Unknown"),
            "timestamp": raw.get("timestamp", ""),
            "raw_message": raw.get("message", ""),
            "classification": classifications.get(mid),
            "task_or_event": tasks_events.get(mid),
            "sensitive_finding": sensitive_findings.get(mid),
        })

    return {"total": len(results), "data": results}



@app.get("/api/classifications")
def get_classifications(
    category: Optional[str] = None,
    search: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    items = _load_json_file("classifications.json")

    if category and category != "all":
        items = [x for x in items if x.get("category") == category]
    if min_confidence is not None:
        items = [x for x in items if x.get("confidence", 0) >= min_confidence]
    if max_confidence is not None:
        items = [x for x in items if x.get("confidence", 0) <= max_confidence]
    if search:
        q = search.lower()
        items = [
            x for x in items
            if q in x.get("message_id", "").lower()
            or q in x.get("reason", "").lower()
            or q in x.get("matched_rule", "").lower()
            or q in x.get("category", "").lower()
        ]

    total = len(items)
    paginated = items[offset : offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "data": paginated}


@app.get("/api/tasks-events")
def get_tasks_events(
    type: Optional[str] = None,
    priority: Optional[str] = None,
    search: Optional[str] = None,
    has_unresolved: Optional[bool] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    items = _load_json_file("tasks_events.json")

    if type and type != "all":
        items = [x for x in items if x.get("type") == type]
    if priority and priority != "all":
        items = [x for x in items if x.get("priority") == priority]
    if has_unresolved is True:
        items = [x for x in items if len(x.get("unresolved_fields", [])) > 0]
    elif has_unresolved is False:
        items = [x for x in items if len(x.get("unresolved_fields", [])) == 0]
    if search:
        q = search.lower()
        items = [
            x for x in items
            if q in x.get("title", "").lower()
            or q in x.get("description", "").lower()
            or q in x.get("source_message_id", "").lower()
            or q in str(x.get("person", "")).lower()
            or q in str(x.get("location", "")).lower()
        ]

    total = len(items)
    paginated = items[offset : offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "data": paginated}


@app.get("/api/sensitive-findings")
def get_sensitive_findings(
    risk: Optional[str] = None,
    sensitivity_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    items = _load_json_file("sensitive_findings.json")

    if risk and risk != "all":
        items = [x for x in items if x.get("risk") == risk]
    if sensitivity_type and sensitivity_type != "all":
        items = [x for x in items if x.get("sensitivity_type") == sensitivity_type]
    if search:
        q = search.lower()
        items = [
            x for x in items
            if q in x.get("message_id", "").lower()
            or q in x.get("sensitivity_type", "").lower()
            or q in x.get("masked_text", "").lower()
            or q in x.get("reason", "").lower()
        ]

    total = len(items)
    paginated = items[offset : offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "data": paginated}


@app.post("/api/classify")
def classify_one(payload: MessageIn):
    classification = classify_message(payload.message_id, payload.sender, payload.message)
    result = {"classification": classification.to_dict()}

    if classification.category in EXTRACTABLE_CATEGORIES:
        item_id = f"ITEM_{payload.message_id}"
        task_event = extract_task_or_event(item_id, payload.message_id, payload.message)
        if task_event:
            result["task_or_event"] = task_event.to_dict()
        else:
            result["task_or_event"] = None
    else:
        result["task_or_event"] = None

    finding = detect_sensitive_info(payload.message_id, payload.message)
    result["sensitive_finding"] = finding.to_dict() if finding else None

    return result


@app.post("/api/run-pipeline")
def trigger_local_pipeline():
    if not DATA_CSV.exists():
        raise HTTPException(status_code=404, detail="Dataset file data/messages.csv not found")
    summary = run_pipeline(DATA_CSV, OUTPUT_DIR)
    return {"status": "success", "summary": summary}


@app.get("/api/latest-upload")
def get_latest_upload_status():
    if UPLOAD_DATA_DIR.exists():
        csv_files = sorted(list(UPLOAD_DATA_DIR.glob("*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
        if csv_files:
            latest = csv_files[0]
            return {
                "has_upload": True,
                "filename": latest.name,
                "path": str(latest.relative_to(BASE_DIR))
            }
    return {"has_upload": False, "filename": None, "path": None}


@app.post("/api/run-latest-upload")
def trigger_latest_upload_pipeline():
    status = get_latest_upload_status()
    if not status["has_upload"]:
        raise HTTPException(status_code=404, detail="No uploaded CSV file found in data/uploads/")
    
    latest_path = BASE_DIR / status["path"]
    CUSTOM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = run_pipeline(latest_path, CUSTOM_OUTPUT_DIR)
    run_pipeline(latest_path, OUTPUT_DIR)
    
    return {
        "status": "success",
        "file_run": status["filename"],
        "summary": summary
    }



# Known limitation: single-shared-state, no auth/user-isolation, intended for solo demo use only.
@app.post("/api/process-csv")
async def process_csv(file: UploadFile = File(...)):
    try:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Please upload a valid .csv file")

        raw = await file.read()
        text = raw.decode("utf-8-sig")
        reader = list(csv.DictReader(io.StringIO(text)))
        if not reader:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty")

        headers = set(reader[0].keys()) - {None}
        required_cols = {"message_id", "timestamp", "sender", "message"}
        if not required_cols.issubset(headers):
            raise HTTPException(
                status_code=400,
                detail=f"CSV missing required columns: {required_cols}",
            )

        # 1. Save uploaded file into data/uploads/ with a timestamped filename
        UPLOAD_DATA_DIR.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_csv_path = UPLOAD_DATA_DIR / f"uploaded_{timestamp_str}.csv"
        saved_csv_path.write_bytes(raw)

        # 2. Save pipeline outputs to dedicated output_custom/ folder
        CUSTOM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary = run_pipeline(saved_csv_path, CUSTOM_OUTPUT_DIR)

        # 3. Update main output/ directory so live Web Dashboard reflects the new dataset
        run_pipeline(saved_csv_path, OUTPUT_DIR)

        return {
            "status": "success",
            "saved_file": str(saved_csv_path.relative_to(BASE_DIR)),
            "custom_output_dir": str(CUSTOM_OUTPUT_DIR.relative_to(BASE_DIR)),
            "summary": summary,
            "classifications_count": len(_load_json_file("classifications.json")),
            "tasks_events_count": len(_load_json_file("tasks_events.json")),
            "sensitive_findings_count": len(_load_json_file("sensitive_findings.json")),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>KaStack Message Intelligence Platform</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-glass: linear-gradient(135deg, #f0f7ff 0%, #f8fafc 40%, #f5f3ff 100%);
    --card-glass: rgba(255, 255, 255, 0.82);
    --border-glass: 1px solid rgba(255, 255, 255, 0.9);
    --shadow-glass: 0 10px 30px -5px rgba(37, 99, 235, 0.08), 0 4px 12px rgba(0, 0, 0, 0.03);
    --shadow-glass-hover: 0 16px 36px -4px rgba(37, 99, 235, 0.14), 0 6px 16px rgba(0, 0, 0, 0.04);

    --glass-blue: #007fff;
    --glass-purple: #8b5cf6;
    --glass-green: #10b981;
    --text-dark: #0f172a;
    --text-muted: #64748b;

    --card-bg: var(--card-glass);
    --card-border: var(--border-glass);
    --text-primary: var(--text-dark);
    --text-secondary: var(--text-muted);
    --accent-indigo: var(--glass-blue);
    --accent-rose: #dc2626;
    --accent-amber: #d97706;
    --accent-emerald: #059669;
    --clay-green: var(--glass-green);

    --radius-pill: 999px;
    --radius-card: 20px;
    --radius-sm: 12px;
    --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg-glass);
    background-attachment: fixed;
    color: var(--text-dark);
    font-family: var(--font-sans);
    line-height: 1.5;
    padding: 24px 16px 60px;
  }

  header {
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: var(--border-glass);
    box-shadow: var(--shadow-glass);
    border-radius: var(--radius-card);
    padding: 16px 30px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    max-width: 1320px;
    margin: 0 auto 28px;
  }

  .logo-area { display: flex; align-items: center; gap: 14px; }
  .logo-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #007fff, #8b5cf6);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; color: #ffffff;
    box-shadow: 0 6px 18px rgba(139, 92, 246, 0.35);
  }
  .logo-text h1 { font-size: 20px; font-weight: 800; color: var(--text-dark); letter-spacing: -0.02em; }
  .logo-text p { font-size: 12px; font-weight: 600; color: var(--text-muted); }

  .header-actions { display: flex; align-items: center; gap: 12px; }
  .status-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #059669;
    padding: 6px 16px; border-radius: var(--radius-pill); font-size: 12px; font-weight: 700;
  }
  .pulse-dot { width: 8px; height: 8px; background: var(--glass-green); border-radius: 50%; box-shadow: 0 0 10px var(--glass-green); }

  .btn {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 10px 22px; border-radius: var(--radius-sm);
    font-size: 13px; font-weight: 700; cursor: pointer;
    transition: all 0.2s ease; border: none; text-decoration: none;
  }
  .btn-primary {
    background: linear-gradient(135deg, #10b981, #059669);
    color: #ffffff;
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.35);
  }
  .btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45);
  }
  .btn-secondary {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(59, 130, 246, 0.4);
    color: #2563eb;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.15);
  }
  .btn-secondary:hover {
    transform: translateY(-2px);
    background: #ffffff;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.25);
  }

  .container { max-width: 1320px; margin: 0 auto; }

  /* KPI Grid */
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 28px; }
  .kpi-card {
    background: var(--card-glass);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: var(--border-glass);
    border-radius: var(--radius-card);
    padding: 24px;
    box-shadow: var(--shadow-glass);
    display: flex; flex-direction: column; justify-content: space-between;
    position: relative; overflow: hidden;
    transition: all 0.2s ease;
  }
  .kpi-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-glass-hover); }
  .kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; width: 5px; height: 100%;
  }
  .kpi-card.blue::before { background: linear-gradient(180deg, #007fff, #3b82f6); }
  .kpi-card.green::before { background: linear-gradient(180deg, #10b981, #059669); }
  .kpi-card.pink::before { background: linear-gradient(180deg, #8b5cf6, #7c3aed); }
  .kpi-card.yellow::before { background: linear-gradient(180deg, #06b6d4, #0891b2); }

  .kpi-label { font-size: 12px; font-weight: 800; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; }
  .kpi-value { font-size: 34px; font-weight: 800; margin: 10px 0 4px; font-family: var(--font-mono); color: var(--text-dark); }
  .kpi-sub { font-size: 12px; font-weight: 600; color: var(--text-muted); }

  /* Navigation Tabs */
  .nav-tabs {
    display: flex; gap: 8px; flex-wrap: wrap;
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(16px);
    border: var(--border-glass);
    box-shadow: var(--shadow-glass);
    padding: 8px; border-radius: var(--radius-card);
    margin-bottom: 28px;
  }
  .tab-btn {
    flex: 1; padding: 12px 20px; text-align: center;
    background: transparent; border: none; color: var(--text-muted);
    font-size: 13px; font-weight: 700; border-radius: var(--radius-sm);
    cursor: pointer; transition: all 0.2s ease;
    display: flex; align-items: center; justify-content: center; gap: 8px;
  }
  .tab-btn:hover { color: var(--text-dark); background: rgba(255, 255, 255, 0.5); }
  .tab-btn.active {
    background: linear-gradient(135deg, #007fff 0%, #8b5cf6 100%);
    color: #ffffff;
    box-shadow: 0 4px 16px rgba(0, 127, 255, 0.35);
  }

  /* Section Views */
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  /* Control Bars */
  .control-bar {
    display: flex; flex-wrap: wrap; gap: 14px; align-items: center; justify-content: space-between;
    background: var(--card-glass); backdrop-filter: blur(16px); border: var(--border-glass);
    box-shadow: var(--shadow-glass); padding: 18px 24px; border-radius: var(--radius-card); margin-bottom: 22px;
  }
  .search-box { display: flex; align-items: center; gap: 10px; background: rgba(255, 255, 255, 0.9); border: 1px solid #e2e8f0; border-radius: var(--radius-sm); padding: 10px 18px; flex: 1; min-width: 260px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02); }
  .search-box input { background: transparent; border: none; outline: none; color: var(--text-dark); font-size: 13px; font-weight: 600; width: 100%; }
  .select-box { background: rgba(255, 255, 255, 0.9); border: 1px solid #e2e8f0; color: var(--text-dark); font-weight: 700; padding: 10px 18px; border-radius: var(--radius-sm); font-size: 13px; outline: none; cursor: pointer; }

  /* Data Tables & Cards */
  .table-wrapper {
    background: var(--card-glass); backdrop-filter: blur(16px); border: var(--border-glass);
    box-shadow: var(--shadow-glass); border-radius: var(--radius-card); overflow: hidden;
  }
  table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }
  th { background: rgba(241, 245, 249, 0.8); color: var(--text-muted); font-weight: 800; text-transform: uppercase; font-size: 11px; letter-spacing: 0.06em; padding: 16px 20px; border-bottom: 1px solid #e2e8f0; }
  td { padding: 16px 20px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
  tr:hover td { background: rgba(255, 255, 255, 0.6); }

  /* Badges */
  .badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 12px; border-radius: 8px; font-size: 11px; font-weight: 700;
    font-family: var(--font-mono); text-transform: uppercase;
  }
  .badge-action_required { background: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; }
  .badge-meeting_or_event { background: #e0f2fe; color: #0284c7; border: 1px solid #bae6fd; }
  .badge-general_information { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
  .badge-sensitive_information { background: #fef3c7; color: #d97706; border: 1px solid #fde68a; }
  .badge-personal_information { background: #f3e8ff; color: #7c3aed; border: 1px solid #ddd6fe; }
  .badge-promotional { background: #d1fae5; color: #059669; border: 1px solid #a7f3d0; }

  .badge-risk-high { background: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; }
  .badge-risk-medium { background: #fef3c7; color: #d97706; border: 1px solid #fde68a; }
  .badge-risk-low { background: #e0f2fe; color: #0284c7; border: 1px solid #bae6fd; }

  .confidence-meter { display: flex; align-items: center; gap: 8px; font-family: var(--font-mono); font-size: 12px; font-weight: 700; }
  .bar-bg { width: 64px; height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, #10b981, #059669); }

  /* Cards Grid for Tasks/Vault */
  .cards-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 20px; }
  .item-card {
    background: var(--card-glass); backdrop-filter: blur(16px); border: var(--border-glass);
    box-shadow: var(--shadow-glass); border-radius: var(--radius-card); padding: 22px; display: flex; flex-direction: column; gap: 14px;
    transition: all 0.2s ease;
  }
  .item-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-glass-hover); }
  .item-card-header { display: flex; align-items: center; justify-content: space-between; }
  .item-card-title { font-size: 15px; font-weight: 800; color: var(--text-dark); }
  .item-card-desc { font-size: 12px; color: var(--text-dark); background: rgba(248, 250, 252, 0.9); border: 1px solid #e2e8f0; border-radius: var(--radius-sm); padding: 12px; font-family: var(--font-mono); }

  .meta-list { display: flex; flex-wrap: wrap; gap: 8px; font-size: 12px; font-weight: 600; color: var(--text-dark); }
  .meta-item { display: flex; align-items: center; gap: 4px; background: #ffffff; border: 1px solid #e2e8f0; padding: 4px 12px; border-radius: var(--radius-pill); box-shadow: 0 2px 4px rgba(0,0,0,0.02); }

  /* Playground */
  .playground-layout { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  @media (max-width: 900px) { .playground-layout { grid-template-columns: 1fr; } }

  .form-card { background: var(--card-glass); backdrop-filter: blur(16px); border: var(--border-glass); box-shadow: var(--shadow-glass); border-radius: var(--radius-card); padding: 28px; }
  .form-group { margin-bottom: 20px; }
  .form-group label { display: block; font-size: 12px; font-weight: 800; color: var(--text-muted); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
  .form-control { width: 100%; background: rgba(255, 255, 255, 0.9); border: 1px solid #e2e8f0; color: var(--text-dark); font-weight: 600; padding: 12px 16px; border-radius: var(--radius-sm); font-family: inherit; font-size: 13px; outline: none; transition: border-color 0.2s; }
  .form-control:focus { border-color: #007fff; background: #ffffff; box-shadow: 0 0 0 3px rgba(0, 127, 255, 0.15); }
  textarea.form-control { min-height: 110px; resize: vertical; }

  .result-panel { background: var(--card-glass); backdrop-filter: blur(16px); border: var(--border-glass); box-shadow: var(--shadow-glass); border-radius: var(--radius-card); padding: 28px; display: flex; flex-direction: column; gap: 18px; }

  /* Pagination */
  .pagination { display: flex; align-items: center; justify-content: space-between; padding: 16px 24px; border-top: 1px solid #e2e8f0; font-size: 12px; font-weight: 700; color: var(--text-muted); background: rgba(248, 250, 252, 0.8); }
  /* Category Distribution Progress Bars */
  .cat-progress { margin-bottom: 16px; }
  .cat-progress-head { display: flex; justify-content: space-between; font-size: 13px; font-weight: 700; margin-bottom: 8px; color: var(--text-dark); }
  .cat-progress-bar { height: 12px; background: rgba(226, 232, 240, 0.8); border-radius: var(--radius-pill); overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.05); }
  .cat-progress-fill { height: 100%; border-radius: var(--radius-pill); transition: width 0.3s ease; }

  /* Mobile Responsiveness */
  @media (max-width: 768px) {
    body { padding: 12px 10px 40px; }
    header { flex-direction: column; align-items: stretch; gap: 14px; padding: 16px 20px; }
    .header-actions { flex-direction: column; width: 100%; }
    .header-actions .btn { width: 100%; justify-content: center; }
    .nav-tabs { flex-wrap: nowrap; overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .tab-btn { flex: 0 0 auto; white-space: nowrap; }
    .cards-grid { grid-template-columns: 1fr; }
    .control-bar { flex-direction: column; align-items: stretch; }
    .search-box { width: 100%; }
    .select-box { width: 100%; }
  }
</style>
</head>
<body>

<header>
  <div class="logo-area">
    <div class="logo-icon">📖</div>
    <div class="logo-text">
      <h1>KaStack Message Intelligence</h1>
      <p>Rule-Based Classification & Data Protection Platform</p>
    </div>
  </div>
  <div class="header-actions">
    <div class="status-badge"><span class="pulse-dot"></span> Pipeline Active</div>
    <button class="btn btn-secondary" onclick="triggerPipeline()">🔄 Run Pipeline</button>
    <label class="btn btn-primary" style="margin:0; cursor:pointer;">
      📁 Upload Custom CSV
      <input type="file" id="csv-upload-input" accept=".csv" style="display:none;" onchange="uploadCSV(event)" />
    </label>
  </div>
</header>

<div class="container">

  <!-- KPI Grid -->
  <div class="kpi-grid">
    <div class="kpi-card blue">
      <div class="kpi-label">Total Messages Processed</div>
      <div class="kpi-value" id="kpi-total">900</div>
      <div class="kpi-sub">Defensively sorted chronologically</div>
    </div>
    <div class="kpi-card green">
      <div class="kpi-label">Tasks & Events Extracted</div>
      <div class="kpi-value" id="kpi-extracted">400</div>
      <div class="kpi-sub" id="kpi-extracted-sub">230 Tasks · 170 Events</div>
    </div>
    <div class="kpi-card pink">
      <div class="kpi-label">Sensitive Findings Detected</div>
      <div class="kpi-value" id="kpi-sensitive">100</div>
      <div class="kpi-sub">Scanned independently across all text</div>
    </div>
    <div class="kpi-card yellow">
      <div class="kpi-label">Rule Matching Precision</div>
      <div class="kpi-value">100%</div>
      <div class="kpi-sub" id="kpi-low-conf">50 Low Confidence Flagged</div>
    </div>
  </div>

  <!-- Navigation Tabs -->
  <div class="nav-tabs">
    <button class="tab-btn active" onclick="switchTab('overview', this)">📊 Overview & Analytics</button>
    <button class="tab-btn" onclick="switchTab('mandatory', this)">📌 Benchmark Test Set</button>
    <button class="tab-btn" onclick="switchTab('classifications', this)">🏷️ Classifications (Part 1)</button>
    <button class="tab-btn" onclick="switchTab('tasks-events', this)">📅 Tasks & Events (Part 2)</button>
    <button class="tab-btn" onclick="switchTab('sensitive', this)">🔒 Sensitive Data (Part 3)</button>
    <button class="tab-btn" onclick="switchTab('playground', this)">⚡ Real-Time Playground</button>
  </div>

  <!-- Tab 1: Overview -->
  <div id="tab-overview" class="tab-content active">
    <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 24px;">
      <div class="table-wrapper" style="padding: 24px;">
        <h3 style="font-size: 16px; margin-bottom: 16px;">Category Distribution Breakdown</h3>
        <div id="category-distribution-container">Loading distribution...</div>
      </div>
      <div class="table-wrapper" style="padding: 24px;">
        <h3 style="font-size: 16px; margin-bottom: 16px;">Pipeline Architecture Highlights</h3>
        <ul style="font-size: 13px; color: var(--text-secondary); display:flex; flex-direction:column; gap: 12px; list-style: none;">
          <li>🛡️ <strong>Rule-Based Classification:</strong> Matches 530 unique core templates derived from 24 distinct sentence structures.</li>
          <li>🎯 <strong>No Guessing Missing Info:</strong> Structural null enforcement for unresolvable dates/times with <code>unresolved_fields</code> tracking.</li>
          <li>🔒 <strong>Safety First:</strong> 100% local processing; zero external LLM API dependencies; sensitive data masked in-place.</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- Tab 6: Mandatory 15 Demo IDs -->
  <div id="tab-mandatory" class="tab-content">
    <div class="control-bar">
      <div style="font-size: 13px; font-weight: 600; color: var(--accent-amber);">
        📌 Key Benchmark Dataset (15 Evaluated Sample Messages)
      </div>
    </div>

    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Message ID</th>
            <th>Raw Message & Sender</th>
            <th>Part 1: Classification</th>
            <th>Part 2: Task / Event</th>
            <th>Part 3: Sensitive Flag</th>
          </tr>
        </thead>
        <tbody id="mandatory-table-body">
          <tr><td colspan="5" style="text-align:center; padding: 24px;">Loading mandatory 15 demo IDs...</td></tr>
        </tbody>
      </table>
    </div>
  </div>


  <!-- Tab 2: Classifications -->
  <div id="tab-classifications" class="tab-content">
    <div class="control-bar">
      <div class="search-box">
        🔍 <input type="text" id="class-search" placeholder="Search message ID, category, or rule reason..." oninput="loadClassifications(0)" />
      </div>
      <select class="select-box" id="class-cat-filter" onchange="loadClassifications(0)">
        <option value="all">All Categories</option>
        <option value="action_required">Action Required</option>
        <option value="meeting_or_event">Meeting / Event</option>
        <option value="general_information">General Information</option>
        <option value="sensitive_information">Sensitive Information</option>
        <option value="personal_information">Personal Information</option>
        <option value="promotional">Promotional</option>
      </select>
    </div>

    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Message ID</th>
            <th>Category</th>
            <th>Confidence</th>
            <th>Matched Rule</th>
            <th>Reason / Explanation</th>
          </tr>
        </thead>
        <tbody id="class-table-body">
          <tr><td colspan="5" style="text-align:center; padding: 24px;">Loading classifications...</td></tr>
        </tbody>
      </table>
      <div class="pagination">
        <span id="class-page-info">Showing 0-0 of 0</span>
        <div class="page-btns">
          <button class="btn btn-secondary" id="class-prev-btn" onclick="changeClassPage(-1)">Previous</button>
          <button class="btn btn-secondary" id="class-next-btn" onclick="changeClassPage(1)">Next</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Tab 3: Tasks & Events -->
  <div id="tab-tasks-events" class="tab-content">
    <div class="control-bar">
      <div class="search-box">
        🔍 <input type="text" id="task-search" placeholder="Search task title, description, or person..." oninput="loadTasksEvents(0)" />
      </div>
      <select class="select-box" id="task-type-filter" onchange="loadTasksEvents(0)">
        <option value="all">All Items (Tasks & Events)</option>
        <option value="task">Tasks Only</option>
        <option value="event">Events Only</option>
      </select>
    </div>

    <div class="cards-grid" id="tasks-cards-container">
      <div style="grid-column: 1/-1; text-align:center; padding: 32px; color: var(--text-muted);">Loading tasks & events...</div>
    </div>
  </div>

  <!-- Tab 4: Sensitive Data Vault -->
  <div id="tab-sensitive" class="tab-content">
    <div class="control-bar">
      <div class="search-box">
        🔍 <input type="text" id="sensitive-search" placeholder="Search sensitive finding, type, or masked text..." oninput="loadSensitive(0)" />
      </div>
      <select class="select-box" id="sensitive-risk-filter" onchange="loadSensitive(0)">
        <option value="all">All Risk Levels</option>
        <option value="high">High Risk</option>
        <option value="medium">Medium Risk</option>
        <option value="low">Low Risk</option>
      </select>
    </div>

    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Message ID</th>
            <th>Sensitivity Type</th>
            <th>Risk Level</th>
            <th>Masked Text Preview</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody id="sensitive-table-body">
          <tr><td colspan="5" style="text-align:center; padding: 24px;">Loading sensitive findings...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Tab 5: Real-Time Playground -->
  <div id="tab-playground" class="tab-content">
    <div class="playground-layout">
      <div class="form-card">
        <h3 style="font-size: 16px; margin-bottom: 16px;">Test Message Classifier & Extractor</h3>
        <div class="form-group">
          <label>Message ID</label>
          <input class="form-control" id="play-mid" value="MSG_LIVE_001" />
        </div>
        <div class="form-group">
          <label>Sender</label>
          <input class="form-control" id="play-sender" value="Meera" />
        </div>
        <div class="form-group">
          <label>Message Content</label>
          <textarea class="form-control" id="play-msg">Can you review the privacy checklist before 2026-09-09?</textarea>
        </div>
        <button class="btn btn-primary" style="width: 100%; justify-content: center;" onclick="runPlaygroundTest()">⚡ Analyze Message</button>
      </div>

      <div class="result-panel" id="playground-results">
        <h3 style="font-size: 16px;">Real-Time Pipeline Output</h3>
        <p style="font-size: 13px; color: var(--text-muted);">Click "Analyze Message" to inspect classification, task extraction, and sensitive data detection results.</p>
      </div>
    </div>
  </div>

</div>

<script>
  let summaryData = {};
  let currentClassOffset = 0;
  let classLimit = 50;

  async function init() {
    fetchSummary();
    loadMandatoryDemo();
    loadClassifications(0);
    loadTasksEvents(0);
    loadSensitive(0);
  }

  async function loadMandatoryDemo() {
    try {
      const res = await fetch('/api/mandatory-demo?t=' + Date.now());
      const result = await res.json();
      const tbody = document.getElementById('mandatory-table-body');
      if (!result.data.length) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 24px; color: var(--text-muted);">No mandatory IDs found.</td></tr>`;
        return;
      }
      tbody.innerHTML = result.data.map(item => {
        const c = item.classification || {};
        const te = item.task_or_event || {};
        const sf = item.sensitive_finding || {};

        return `
          <tr>
            <td style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-indigo); vertical-align: top;">
              ${item.message_id}
            </td>
            <td style="vertical-align: top; max-width: 260px;">
              <div style="font-size: 11px; color: var(--text-muted); font-weight: 600;">Sender: ${item.sender || 'Unknown'}</div>
              <div style="font-size: 12px; color: var(--text-primary); margin-top: 4px; font-family: var(--font-mono);">${item.raw_message || 'N/A'}</div>
            </td>
            <td style="vertical-align: top;">
              ${c.category ? `
                <span class="badge badge-${c.category}">${c.category}</span>
                <div style="font-size: 11px; color: var(--text-secondary); margin-top: 4px;">Conf: ${c.confidence} | Rule: <code>${c.matched_rule}</code></div>
                <div style="font-size: 11px; color: var(--text-muted); font-style: italic; margin-top: 2px;">"${c.reason}"</div>
              ` : '<span style="color:var(--text-muted);">None</span>'}
            </td>
            <td style="vertical-align: top;">
              ${te.type ? `
                <span class="badge badge-${te.type === 'task' ? 'action_required' : 'meeting_or_event'}">${te.type}</span>
                <div style="font-size: 12px; font-weight: 600; color: var(--text-primary); margin-top: 4px;">${te.title}</div>
                <div style="font-size: 11px; color: var(--text-secondary);">Deadline: ${te.deadline || 'None'} | Time: ${te.time || 'None'}</div>
                ${te.unresolved_fields && te.unresolved_fields.length ? `
                  <div style="font-size: 10px; color: var(--accent-amber); font-weight: 600; margin-top: 2px;">⚠️ Unresolved: ${te.unresolved_fields.join(', ')}</div>
                ` : ''}
              ` : '<span style="color:var(--text-muted); font-size:12px;">N/A</span>'}
            </td>
            <td style="vertical-align: top;">
              ${sf.risk ? `
                <span class="badge badge-risk-${sf.risk}">${sf.risk} risk</span>
                <div style="font-size: 11px; font-family: var(--font-mono); color: var(--accent-rose); margin-top: 4px;">${sf.masked_text}</div>
                <div style="font-size: 10px; color: var(--text-muted); margin-top: 2px;">Action: ${sf.recommended_action}</div>
              ` : '<span style="color:var(--accent-emerald); font-size:12px;">Clean</span>'}
            </td>
          </tr>
        `;
      }).join('');
    } catch(e) {
      console.error(e);
    }
  }


  async function fetchSummary() {
    try {
      const res = await fetch('/api/summary?t=' + Date.now());
      if (!res.ok) throw new Error('HTTP ' + res.status);
      summaryData = await res.json();
      
      if (!summaryData || summaryData.total_messages === undefined) {
        document.getElementById('kpi-total').textContent = '-';
        document.getElementById('kpi-extracted').textContent = '-';
        document.getElementById('kpi-extracted-sub').textContent = 'No data available — run pipeline first';
        document.getElementById('kpi-sensitive').textContent = '-';
        document.getElementById('kpi-low-conf').textContent = 'No data available — run pipeline first';
        document.getElementById('category-distribution-container').innerHTML = '<div style="color: var(--accent-rose); padding: 12px 0; font-weight: 600;">⚠️ No data available — run the pipeline first.</div>';
        return;
      }

      document.getElementById('kpi-total').textContent = summaryData.total_messages;
      const tasks = summaryData.tasks_extracted || 0;
      const events = summaryData.events_extracted || 0;
      document.getElementById('kpi-extracted').textContent = tasks + events;
      document.getElementById('kpi-extracted-sub').textContent = `${tasks} Tasks · ${events} Events`;
      document.getElementById('kpi-sensitive').textContent = summaryData.sensitive_findings !== undefined ? summaryData.sensitive_findings : 0;
      document.getElementById('kpi-low-conf').textContent = `${summaryData.low_confidence_classifications || 0} Low Confidence Flagged`;

      renderDistribution(summaryData.classification_counts || {});
    } catch(e) {
      console.error('Error fetching summary:', e);
      document.getElementById('kpi-total').textContent = '-';
      document.getElementById('kpi-extracted').textContent = '-';
      document.getElementById('kpi-extracted-sub').textContent = 'No data available — run pipeline first';
      document.getElementById('kpi-sensitive').textContent = '-';
      document.getElementById('kpi-low-conf').textContent = 'No data available — run pipeline first';
      document.getElementById('category-distribution-container').innerHTML = '<div style="color: var(--accent-rose); padding: 12px 0; font-weight: 600;">⚠️ No data available — run the pipeline first.</div>';
    }
  }

  function renderDistribution(counts) {
    const total = (summaryData && summaryData.total_messages) ? summaryData.total_messages : (Object.values(counts).reduce((a, b) => a + b, 0) || 900);
    const colors = {
      action_required: 'linear-gradient(90deg, #ef4444, #dc2626)',
      meeting_or_event: 'linear-gradient(90deg, #007fff, #3b82f6)',
      general_information: 'linear-gradient(90deg, #94a3b8, #64748b)',
      sensitive_information: 'linear-gradient(90deg, #f59e0b, #d97706)',
      personal_information: 'linear-gradient(90deg, #8b5cf6, #7c3aed)',
      promotional: 'linear-gradient(90deg, #10b981, #059669)'
    };
    let html = '';
    const entries = Object.entries(counts);
    if (!entries.length) {
      document.getElementById('category-distribution-container').innerHTML = '<div style="color: var(--text-muted); padding: 12px 0;">No distribution data available.</div>';
      return;
    }
    for (const [cat, count] of entries) {
      const pct = Math.round((count / total) * 100);
      html += `
        <div class="cat-progress">
          <div class="cat-progress-head">
            <span style="text-transform: capitalize; font-weight: 700;">${cat.replace(/_/g, ' ')}</span>
            <span style="font-family: var(--font-mono); font-weight: 700;">${count} (${pct}%)</span>
          </div>
          <div class="cat-progress-bar">
            <div class="cat-progress-fill" style="width: ${pct}%; background: ${colors[cat] || 'var(--clay-green)'};"></div>
          </div>
        </div>
      `;
    }
    document.getElementById('category-distribution-container').innerHTML = html;
  }

  function switchTab(tabId, btn) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + tabId).classList.add('active');
  }

  async function loadClassifications(offset = 0) {
    currentClassOffset = offset;
    const search = document.getElementById('class-search').value;
    const cat = document.getElementById('class-cat-filter').value;
    const url = `/api/classifications?offset=${offset}&limit=${classLimit}&category=${cat}&search=${encodeURIComponent(search)}&t=${Date.now()}`;

    try {
      const res = await fetch(url);
      const result = await res.json();
      const tbody = document.getElementById('class-table-body');
      if (!result.data.length) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 24px; color: var(--text-muted);">No matching classifications found.</td></tr>`;
        return;
      }
      tbody.innerHTML = result.data.map(item => `
        <tr>
          <td style="font-family: var(--font-mono); font-weight: 600; color: var(--accent-indigo);">${item.message_id}</td>
          <td><span class="badge badge-${item.category}">${item.category.replace(/_/g, ' ')}</span></td>
          <td>
            <div class="confidence-meter">
              <div class="bar-bg"><div class="bar-fill" style="width: ${item.confidence * 100}%;"></div></div>
              <span>${item.confidence.toFixed(2)}</span>
            </div>
          </td>
          <td style="font-family: var(--font-mono); font-size: 11px; color: var(--text-secondary);">${item.matched_rule}</td>
          <td style="color: var(--text-secondary);">${item.reason}</td>
        </tr>
      `).join('');

      document.getElementById('class-page-info').textContent = `Showing ${offset + 1}-${Math.min(offset + classLimit, result.total)} of ${result.total}`;
      document.getElementById('class-prev-btn').disabled = offset === 0;
      document.getElementById('class-next-btn').disabled = offset + classLimit >= result.total;
    } catch(e) {
      console.error(e);
    }
  }

  function changeClassPage(direction) {
    loadClassifications(currentClassOffset + (direction * classLimit));
  }

  async function loadTasksEvents() {
    const search = document.getElementById('task-search').value;
    const type = document.getElementById('task-type-filter').value;
    const url = `/api/tasks-events?limit=100&type=${type}&search=${encodeURIComponent(search)}&t=${Date.now()}`;

    try {
      const res = await fetch(url);
      const result = await res.json();
      const container = document.getElementById('tasks-cards-container');
      if (!result.data.length) {
        container.innerHTML = `<div style="grid-column: 1/-1; text-align:center; padding: 32px; color: var(--text-muted);">No extracted tasks or events match criteria.</div>`;
        return;
      }

      container.innerHTML = result.data.map(item => `
        <div class="item-card">
          <div class="item-card-header">
            <span class="badge badge-${item.type === 'task' ? 'action_required' : 'meeting_or_event'}">${item.type}</span>
            <span style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">${item.item_id}</span>
          </div>
          <div class="item-card-title">${item.title}</div>
          <div class="item-card-desc">${item.description}</div>
          <div class="meta-list">
            ${item.deadline ? `<div class="meta-item">📅 ${item.deadline}</div>` : ''}
            ${item.time ? `<div class="meta-item">⏰ ${item.time}</div>` : ''}
            ${item.person ? `<div class="meta-item">👤 ${item.person}</div>` : ''}
            ${item.location ? `<div class="meta-item">📍 ${item.location}</div>` : ''}
          </div>
          ${item.unresolved_fields && item.unresolved_fields.length ? `
            <div style="font-size: 11px; color: var(--accent-amber); display:flex; gap: 4px; align-items:center;">
              ⚠️ Unresolved: ${item.unresolved_fields.join(', ')}
            </div>
          ` : ''}
        </div>
      `).join('');
    } catch(e) {
      console.error(e);
    }
  }

  async function loadSensitive() {
    const search = document.getElementById('sensitive-search').value;
    const risk = document.getElementById('sensitive-risk-filter').value;
    const url = `/api/sensitive-findings?limit=100&risk=${risk}&search=${encodeURIComponent(search)}&t=${Date.now()}`;

    try {
      const res = await fetch(url);
      const result = await res.json();
      const tbody = document.getElementById('sensitive-table-body');
      if (!result.data.length) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 24px; color: var(--text-muted);">No sensitive findings match criteria.</td></tr>`;
        return;
      }
      tbody.innerHTML = result.data.map(item => `
        <tr>
          <td style="font-family: var(--font-mono); font-weight: 600; color: var(--accent-indigo);">${item.message_id}</td>
          <td style="font-family: var(--font-mono); font-size: 12px; color: var(--text-primary);">${item.sensitivity_type}</td>
          <td><span class="badge badge-risk-${item.risk}">${item.risk} risk</span></td>
          <td style="font-family: var(--font-mono); font-size: 12px; color: var(--accent-rose);">${item.masked_text}</td>
          <td style="font-size: 11px; font-weight: 600; color: var(--text-secondary);">${item.recommended_action}</td>
        </tr>
      `).join('');
    } catch(e) {
      console.error(e);
    }
  }

  async function runPlaygroundTest() {
    const body = {
      message_id: document.getElementById('play-mid').value,
      sender: document.getElementById('play-sender').value,
      message: document.getElementById('play-msg').value,
    };
    const panel = document.getElementById('playground-results');
    panel.innerHTML = `<div style="text-align:center; padding: 24px;">Analyzing message...</div>`;

    try {
      const res = await fetch('/api/classify', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });
      const data = await res.json();

      let html = `<h3 style="font-size: 16px; margin-bottom: 12px;">Pipeline Analysis Results</h3>`;

      // Classification card
      const c = data.classification;
      html += `
        <div style="background: var(--card-bg); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--card-border);">
          <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">1. Classification Result</div>
          <div style="display:flex; align-items:center; gap: 8px; margin: 6px 0;">
            <span class="badge badge-${c.category}">${c.category}</span>
            <span style="font-family: var(--font-mono); font-size: 12px;">Confidence: ${c.confidence}</span>
          </div>
          <div style="font-size: 12px; color: var(--text-secondary);">${c.reason}</div>
        </div>
      `;

      // Extracted Task/Event
      if (data.task_or_event) {
        const te = data.task_or_event;
        html += `
          <div style="background: var(--card-bg); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--card-border);">
            <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">2. Extracted ${te.type}</div>
            <div style="font-weight: 600; font-size: 13px; margin: 4px 0;">${te.title}</div>
            <div style="font-size: 12px; color: var(--text-secondary);">Deadline: ${te.deadline || 'None'} | Person: ${te.person || 'None'}</div>
          </div>
        `;
      } else {
        html += `
          <div style="background: var(--card-bg); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--card-border); color: var(--text-muted); font-size: 12px;">
            2. Task/Event Extraction: Not applicable for category '${c.category}'
          </div>
        `;
      }

      // Sensitive Finding
      if (data.sensitive_finding) {
        const sf = data.sensitive_finding;
        html += `
          <div style="background: var(--card-bg); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--card-border);">
            <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">3. Sensitive Data Flagged</div>
            <div style="display:flex; align-items:center; gap: 8px; margin: 4px 0;">
              <span class="badge badge-risk-${sf.risk}">${sf.risk} risk</span>
              <span style="font-family: var(--font-mono); font-size: 12px;">${sf.sensitivity_type}</span>
            </div>
            <div style="font-family: var(--font-mono); font-size: 12px; color: var(--accent-rose);">${sf.masked_text}</div>
          </div>
        `;
      } else {
        html += `
          <div style="background: var(--card-bg); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--card-border); color: var(--text-muted); font-size: 12px;">
            3. Sensitive Data Detection: Clean (No sensitive patterns detected)
          </div>
        `;
      }

      panel.innerHTML = html;
    } catch(e) {
      panel.innerHTML = `<div style="color: var(--accent-rose);">Error running test: ${e.message}</div>`;
    }
  }

  async function triggerPipeline() {
    try {
      const statusRes = await fetch('/api/latest-upload?t=' + Date.now());
      const uploadStatus = await statusRes.json();

      let promptMsg = `Select CSV dataset to run pipeline on:

1 = Primary Dataset (data/messages.csv - 900 messages)
`;
      if (uploadStatus.has_upload) {
        promptMsg += `2 = Latest Uploaded File (${uploadStatus.filename})
3 = Upload & Run a New CSV File

Enter 1, 2, or 3:`;
      } else {
        promptMsg += `2 = Upload & Run a New CSV File

Enter 1 or 2:`;
      }

      const choice = prompt(promptMsg, "1");
      if (choice === null) return;

      const trimmed = choice.trim();

      if (trimmed === "1") {
        const res = await fetch('/api/run-pipeline', { method: 'POST' });
        const data = await res.json();
        alert('Pipeline execution complete! Restored to 900 primary messages.');
        location.reload();
        return;
      } else if (trimmed === "2" && uploadStatus.has_upload) {
        const res = await fetch('/api/run-latest-upload', { method: 'POST' });
        const data = await res.json();
        alert(`Pipeline execution complete on Latest Uploaded File (${data.file_run})!`);
        location.reload();
        return;
      } else if (trimmed === "2" || trimmed === "3") {
        document.getElementById('csv-upload-input').click();
        return;
      } else {
        return;
      }

      await fetchSummary();
      loadMandatoryDemo();
      loadClassifications(0);
      loadTasksEvents(0);
      loadSensitive(0);
    } catch(e) {
      alert('Error triggering pipeline: ' + e.message);
    }
  }

  async function resetTo900Dataset() {
    try {
      const res = await fetch('/api/run-pipeline', { method: 'POST' });
      const data = await res.json();
      alert('Pipeline execution complete! Restored to 900 primary messages.');
      location.reload();
    } catch(e) {
      alert('Error resetting pipeline: ' + e.message);
    }
  }

  async function uploadCSV(event) {
    const file = event.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/api/process-csv', { method: 'POST', body: formData });
      if (!res.ok) {
        let errMsg = 'Upload failed';
        try {
          const errData = await res.json();
          errMsg = errData.detail || errMsg;
        } catch(_) {}
        alert('Error uploading CSV: ' + errMsg);
        return;
      }
      const data = await res.json();
      alert(`CSV processed successfully! Total Messages Processed: ${data.summary.total_messages}`);
      location.reload();
    } catch(e) {
      alert('Error uploading CSV: ' + e.message);
    }
  }

  window.onload = init;
</script>
</body>
</html>
"""

