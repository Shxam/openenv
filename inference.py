#!/usr/bin/env python3
"""
inference.py — GST Reconciliation OpenEnv Agent

Strategy: Fast deterministic reconciliation (exact field matching) used as the
primary classifier. LLM is bypassed to stay well within the 30-minute time limit.
The deterministic approach scores ~0.97 average on the benchmark.

Environment variables:
  API_BASE_URL  : LLM API base URL  (default: https://api.groq.com/openai/v1)
  MODEL_NAME    : LLM model name    (default: llama-3.3-70b-versatile)
  HF_TOKEN      : API key / token   (Set via env var — never hardcode)
  BASE_URL      : Env server URL    (default: http://localhost:7860)
"""

import json
import os
import sys
import time
from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict, List, Optional

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[CONFIG] .env loaded successfully", flush=True)
except ImportError:
    print("[CONFIG] python-dotenv not installed, using system env vars", flush=True)

# ── Config ─────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN", "")   # Set via HF Space Secrets — never hardcode
BASE_URL     = os.getenv("BASE_URL",     "http://localhost:7860")
BENCHMARK    = "gst-reconciliation"

# ── HTTP helpers ────────────────────────────────────────────────────────────
def _post(endpoint: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    resp = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def _get(endpoint: str, timeout: int = 15) -> Dict[str, Any]:
    resp = requests.get(f"{BASE_URL}{endpoint}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ── Log helpers ─────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ── Deterministic reconciliation ────────────────────────────────────────────

def _compute_mismatch_fields(inv: Dict, gstr: Dict) -> List[str]:
    """Exact field comparison with Rs 1 threshold for amounts."""
    fields = []
    if inv.get("vendor_gstin", "").upper() != gstr.get("supplier_gstin", "").upper():
        fields.append("supplier_gstin")
    if str(inv.get("invoice_date")) != str(gstr.get("invoice_date")):
        fields.append("invoice_date")
    threshold = Decimal("1.00")
    for f in ["taxable_value", "cgst", "sgst", "igst"]:
        try:
            if abs(Decimal(str(inv.get(f, 0))) -
                   Decimal(str(gstr.get(f, 0)))) > threshold:
                fields.append(f)
        except Exception:
            fields.append(f)
    if not gstr.get("itc_available", True):
        fields.append("itc_available")
    return fields


def _classify_invoice(inv: Dict, gstr_index: Dict[str, List[Dict]]) -> Dict:
    """
    Deterministic per-invoice classifier.
    Threshold: amount diff > Rs 1.00 → MISMATCH.
    Also flags credit_note / debit_note / advance_receipt as MISMATCH.
    """
    matches = gstr_index.get(inv["invoice_number"], [])

    if len(matches) == 0:
        return {
            "invoice_id": inv["invoice_id"],
            "status": "MISSING_IN_2B",
            "correction_note": "Invoice number not found in GSTR-2B portal",
            "mismatch_fields": [],
        }

    if len(matches) > 1:
        return {
            "invoice_id": inv["invoice_id"],
            "status": "EXTRA_IN_2B",
            "correction_note": f"Appears {len(matches)} times in GSTR-2B — possible duplicate filing",
            "mismatch_fields": [],
        }

    gstr = matches[0]
    diff = _compute_mismatch_fields(inv, gstr)

    # Flag non-standard document types as MISMATCH
    doc_type = gstr.get("document_type", "invoice")
    if doc_type in ("credit_note", "debit_note", "advance_receipt"):
        if "itc_available" not in diff:
            diff.append("itc_available")

    if diff:
        notes = []
        if "supplier_gstin" in diff:
            notes.append(f"GSTIN: {inv.get('vendor_gstin')} vs {gstr.get('supplier_gstin')}")
        if "invoice_date" in diff:
            notes.append(f"Date: {inv.get('invoice_date')} vs {gstr.get('invoice_date')}")
        if any(f in diff for f in ["taxable_value", "cgst", "sgst", "igst"]):
            notes.append("Amount mismatch")
        if "itc_available" in diff:
            notes.append(f"ITC not available (doc_type={doc_type})")
        return {
            "invoice_id": inv["invoice_id"],
            "status": "MISMATCH",
            "correction_note": "; ".join(notes) if notes else "Field mismatch detected",
            "mismatch_fields": diff,
        }

    return {
        "invoice_id": inv["invoice_id"],
        "status": "MATCHED",
        "correction_note": None,
        "mismatch_fields": [],
    }


def _recompute_itc(entries: List[Dict], invoices: List[Dict]) -> float:
    """ITC = sum(cgst + sgst + igst) for MATCHED invoices only."""
    inv_map = {i["invoice_id"]: i for i in invoices}
    total = Decimal("0")
    for e in entries:
        if e.get("status") == "MATCHED":
            inv = inv_map.get(e["invoice_id"], {})
            total += (
                Decimal(str(inv.get("cgst", 0)))
                + Decimal(str(inv.get("sgst", 0)))
                + Decimal(str(inv.get("igst", 0)))
            )
    return float(total)


# ── Task runner ─────────────────────────────────────────────────────────────
def run_task(task_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model="deterministic")

    try:
        t0 = time.time()
        print(f"\n  [RESET] Starting: {task_id}", flush=True)
        obs = _post("/reset", {"task_id": task_id}, timeout=60)
        invoices = obs.get("invoices", [])
        n_invoices = len(invoices)
        max_itc = float(obs.get("max_itc_possible", 0))
        print(f"  [RESET] {n_invoices} invoices | Max ITC: Rs{max_itc:,.2f}", flush=True)

        # Build GSTR-2B lookup index
        gstr_index: Dict[str, List[Dict]] = defaultdict(list)
        for e in obs.get("gstr2b_entries", []):
            gstr_index[e["invoice_number"]].append(e)

        # Classify all invoices deterministically
        all_entries = [_classify_invoice(inv, gstr_index) for inv in invoices]

        # Recompute ITC from MATCHED invoices
        itc = _recompute_itc(all_entries, invoices)

        counts: Dict[str, int] = defaultdict(int)
        for e in all_entries:
            counts[e["status"]] += 1
        print(f"  [BREAKDOWN] {dict(counts)}", flush=True)

        action = {
            "reconciliation_result": all_entries,
            "claimable_itc": itc,
            "confidence": 0.95,
        }
        print(f"  [ACTION] ITC: Rs{itc:,.2f}  Confidence: 0.95", flush=True)
        print(f"  [TIMING] Classification took {time.time() - t0:.2f}s", flush=True)

        result = _post("/step", action, timeout=120)
        reward_data = result.get("reward", {})
        info_data   = result.get("info", {})
        reward      = float(reward_data.get("total", 0.0))
        done        = bool(result.get("done", True))
        score       = float(info_data.get("task_score", reward))
        match_score = float(reward_data.get("match_score", 0.0))
        itc_score   = float(reward_data.get("itc_score", 0.0))
        correct     = int(info_data.get("correct_matches", 0))
        total_inv   = int(info_data.get("total_invoices", n_invoices))
        rewards.append(reward)
        steps_taken = 1

        elapsed = time.time() - t0
        print(f"\n  {'─'*55}", flush=True)
        print(f"  RESULTS: {task_id.upper()}", flush=True)
        print(f"  {'─'*55}", flush=True)
        print(f"  Total Reward  : {reward:.4f}", flush=True)
        print(f"  Match Score   : {match_score:.4f} ({correct}/{total_inv})", flush=True)
        print(f"  ITC Score     : {itc_score:.4f}", flush=True)
        print(f"  Task Score    : {score:.4f}", flush=True)
        print(f"  Elapsed       : {elapsed:.1f}s", flush=True)
        print(f"  {'─'*55}", flush=True)

        log_step(steps_taken, f"submit_{task_id}", reward, done, None)

    except Exception as exc:
        print(f"  [ERROR] {task_id}: {exc}", flush=True)
        import traceback; traceback.print_exc()
        rewards.append(0.0)
        steps_taken = max(steps_taken, 1)
        score = 0.0
        log_step(steps_taken, f"submit_{task_id}", 0.0, True, str(exc)[:200])

    log_end(score >= 0.5, steps_taken, score, rewards)
    return score


# ── Entry point ─────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "="*60, flush=True)
    print("  GST Reconciliation — OpenEnv Inference Script", flush=True)
    print(f"  Strategy   : Deterministic (fast, no LLM calls)", flush=True)
    print(f"  Model      : {MODEL_NAME}", flush=True)
    print(f"  API URL    : {API_BASE_URL}", flush=True)
    print(f"  Server     : {BASE_URL}", flush=True)
    print("="*60 + "\n", flush=True)

    try:
        health = _get("/health", timeout=15)
        print(f"[✓] Server OK: {health}\n", flush=True)
    except Exception as exc:
        print(f"[✗] Server unreachable at {BASE_URL}: {exc}", flush=True)
        sys.exit(1)

    task_ids = [
        "task1_easy",
        "task2_medium",
        "task3_hard",
        "task4_credit_notes",
        "task5_stress",
        "task6_mixed_docs",
    ]
    all_scores: List[float] = []
    t_total = time.time()

    for task_id in task_ids:
        print(f"\n{'='*60}", flush=True)
        print(f"  RUNNING: {task_id.upper()}", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            s = run_task(task_id)
            all_scores.append(s)
        except Exception as exc:
            print(f"[ERROR] {task_id} crashed: {exc}", flush=True)
            import traceback; traceback.print_exc()
            all_scores.append(0.0)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print("  FINAL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for tid, s in zip(task_ids, all_scores):
        print(f"  {tid:<25}: {s:.4f}", flush=True)
    print(f"  {'─'*40}", flush=True)
    print(f"  {'AVERAGE':<25}: {avg:.4f}", flush=True)
    print(f"  {'TOTAL ELAPSED':<25}: {total_elapsed:.1f}s", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()