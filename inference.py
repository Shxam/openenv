#!/usr/bin/env python3
"""
inference.py — GST Reconciliation OpenEnv Agent

Environment variables:
  API_BASE_URL  : LLM API base URL  (default: https://api.groq.com/openai/v1)
  MODEL_NAME    : LLM model name    (default: llama-3.3-70b-versatile)
  HF_TOKEN      : API key / token   (REQUIRED — no default for security)
  BASE_URL      : Env server URL    (default: http://localhost:7860)
"""

import json
import os
import re
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
HF_TOKEN     = os.getenv("HF_TOKEN", "")          # Set via env var — never hardcode
BASE_URL     = os.getenv("BASE_URL",     "http://localhost:7860")
BENCHMARK    = "gst-reconciliation"

# ── OpenAI client ──────────────────────────────────────────────────────────
client = None
try:
    from openai import OpenAI
    if HF_TOKEN:
        client = OpenAI(
            api_key=HF_TOKEN,
            base_url=API_BASE_URL,
            timeout=120.0,
            max_retries=2,
        )
        print(f"[CONFIG] OpenAI client initialised — model={MODEL_NAME}", flush=True)
    else:
        print("[WARN] HF_TOKEN not set — client not initialised.", flush=True)
except Exception as e:
    print(f"[ERROR] OpenAI client init failed: {e}", flush=True)

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Indian GST reconciliation engine.
You MUST respond with ONLY a valid JSON object — no markdown, no explanation.

CLASSIFICATION RULES (apply in order):
1. MISSING_IN_2B  : invoice_number NOT found in GSTR-2B at all
2. EXTRA_IN_2B   : invoice_number appears 2+ times in GSTR-2B (duplicate filing)
3. MISMATCH      : exactly 1 GSTR-2B match, BUT any of these differ:
   - supplier_gstin ≠ vendor_gstin (case-insensitive, watch OCR errors: 0/O, 1/I, 8/B, 5/S)
   - invoice_date differs by even 1 day (including month-boundary shifts e.g. Mar31→Apr1)
   - |taxable_value diff| > Rs 1.00 (IMPORTANT: Rs 1.01 difference = MISMATCH)
   - |cgst diff| > Rs 1.00 OR |sgst diff| > Rs 1.00 OR |igst diff| > Rs 1.00
   - itc_available = False in GSTR-2B
   - document_type is "credit_note", "debit_note", or "advance_receipt" in GSTR-2B
4. MATCHED       : exactly 1 match, vendor_gstin == supplier_gstin (case-insensitive),
                   dates exactly equal, all amounts within Rs 1.00, itc_available=True,
                   document_type="invoice"

CRITICAL WARNINGS:
- Near-miss amounts: Rs 1.01 difference is a MISMATCH. Rs 1.00 or less = MATCHED.
- OCR errors: "29AABCT1332L1ZB" vs "29AABCT1332L1Z8" differ in last char (B vs 8) = MISMATCH
- credit_note / debit_note / advance_receipt in GSTR-2B → always MISMATCH
- Do NOT claim ITC on MISSING_IN_2B, MISMATCH, or itc_available=False invoices

Valid mismatch_fields: taxable_value invoice_date supplier_gstin cgst sgst igst itc_available
Report ALL fields that differ for MISMATCH entries (there may be multiple).
ITC is recomputed server-side — always output claimable_itc as exactly 0.0

REQUIRED OUTPUT FORMAT (JSON only, no other text):
{
  "reconciliation_result": [
    {"invoice_id":"","status":"","correction_note":"brief reason","mismatch_fields":[]}
  ],
  "claimable_itc": 0.0,
  "confidence": 0.95
}

Every invoice_id in the input MUST appear in reconciliation_result.
mismatch_fields must be [] unless status is MISMATCH.
correction_note should be a brief human-readable explanation for MISMATCH entries.
claimable_itc MUST always be the literal number 0.0 — never an expression."""


# ── Log format ─────────────────────────────────────────────────────────────
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


# ── HTTP helpers ────────────────────────────────────────────────────────────
def _post(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()

def _get(endpoint: str) -> Dict[str, Any]:
    resp = requests.get(f"{BASE_URL}{endpoint}", timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Deterministic helpers ───────────────────────────────────────────────────
def _compute_mismatch_fields(inv: Dict, gstr: Dict) -> List[str]:
    fields = []
    if inv.get("vendor_gstin", "").upper() != gstr.get("supplier_gstin", "").upper():
        fields.append("supplier_gstin")
    if str(inv.get("invoice_date")) != str(gstr.get("invoice_date")):
        fields.append("invoice_date")
    for f in ["taxable_value", "cgst", "sgst", "igst"]:
        try:
            if abs(Decimal(str(inv.get(f, 0))) -
                   Decimal(str(gstr.get(f, 0)))) > Decimal("1"):
                fields.append(f)
        except Exception:
            fields.append(f)
    if not gstr.get("itc_available", True):
        fields.append("itc_available")
    return fields


def _recompute_itc(entries: List[Dict], obs: Dict) -> float:
    """Always recompute ITC from raw invoice data — never trust LLM value."""
    inv_map = {i["invoice_id"]: i for i in obs.get("invoices", [])}
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


def _enrich_mismatch_fields(entries: List[Dict], obs: Dict) -> None:
    """Fill empty mismatch_fields on MISMATCH entries using Python differ."""
    inv_map = {i["invoice_id"]: i for i in obs.get("invoices", [])}
    gstr_index: Dict[str, List[Dict]] = defaultdict(list)
    for e in obs.get("gstr2b_entries", []):
        gstr_index[e["invoice_number"]].append(e)
    for entry in entries:
        if entry.get("status") == "MISMATCH" and not entry.get("mismatch_fields"):
            inv = inv_map.get(entry["invoice_id"])
            if inv:
                matches = gstr_index.get(inv["invoice_number"], [])
                if matches:
                    entry["mismatch_fields"] = _compute_mismatch_fields(
                        inv, matches[0])


def _deterministic_classify(inv: Dict, gstr_index: Dict[str, List[Dict]]) -> Dict:
    """Pure Python fallback — used when LLM fails after all retries."""
    matches = gstr_index.get(inv["invoice_number"], [])
    if len(matches) == 0:
        return {
            "invoice_id": inv["invoice_id"],
            "status": "MISSING_IN_2B",
            "correction_note": "Not found in GSTR-2B",
            "mismatch_fields": [],
        }
    if len(matches) > 1:
        return {
            "invoice_id": inv["invoice_id"],
            "status": "EXTRA_IN_2B",
            "correction_note": f"Appears {len(matches)}x in GSTR-2B",
            "mismatch_fields": [],
        }
    diff = _compute_mismatch_fields(inv, matches[0])
    if diff:
        return {
            "invoice_id": inv["invoice_id"],
            "status": "MISMATCH",
            "correction_note": f"Differs: {','.join(diff)}",
            "mismatch_fields": diff,
        }
    return {
        "invoice_id": inv["invoice_id"],
        "status": "MATCHED",
        "correction_note": None,
        "mismatch_fields": [],
    }


# ── Prompt builder ──────────────────────────────────────────────────────────
def _build_prompt(batch: List[Dict], obs: Dict) -> str:
    gstr_entries  = obs.get("gstr2b_entries", [])
    batch_inv_nos = {inv["invoice_number"] for inv in batch}
    batch_gstr    = [e for e in gstr_entries if e["invoice_number"] in batch_inv_nos]

    gstr_count: Dict[str, int] = defaultdict(int)
    for e in batch_gstr:
        gstr_count[e["invoice_number"]] += 1

    inv_lines = []
    for inv in batch:
        c = gstr_count.get(inv["invoice_number"], 0)
        tag = "" if c == 1 else ("[M]" if c == 0 else f"[D{c}]")
        inv_lines.append(
            f"{inv['invoice_id']}|{inv['invoice_number']}{tag}"
            f"|{inv['vendor_gstin']}"
            f"|{str(inv['invoice_date'])}"
            f"|{int(float(str(inv['taxable_value'])))}"
            f"|{int(float(str(inv['cgst'])))}"
            f"|{int(float(str(inv['sgst'])))}"
            f"|{int(float(str(inv['igst'])))}"
        )

    gstr_lines = []
    for e in batch_gstr:
        gstr_lines.append(
            f"{e['invoice_number']}"
            f"|{e['supplier_gstin']}"
            f"|{str(e['invoice_date'])}"
            f"|{int(float(str(e['taxable_value'])))}"
            f"|{int(float(str(e.get('cgst', 0))))}"
            f"|{int(float(str(e.get('sgst', 0))))}"
            f"|{int(float(str(e.get('igst', 0))))}"
            f"|itc={e['itc_available']}"
        )

    invoice_ids = [inv["invoice_id"] for inv in batch]

    return (
        f"Reconcile these {len(batch)} invoices against GSTR-2B.\n"
        f"[M]=MISSING_IN_2B [Dn]=EXTRA_IN_2B (appears n times)\n"
        f"Expected invoice_ids in output: {invoice_ids}\n\n"
        f"INVOICES (id|number[tag]|gstin|date|taxable|cgst|sgst|igst):\n"
        + "\n".join(inv_lines)
        + f"\n\nGSTR2B (number|gstin|date|taxable|cgst|sgst|igst|itc):\n"
        + ("\n".join(gstr_lines) if gstr_lines else "(none — all invoices missing)")
        + "\n\nIMPORTANT: claimable_itc must be exactly 0.0 (a number, not an expression). "
          "Return the JSON object only."
    )


# ── LLM call ────────────────────────────────────────────────────────────────
def _call_llm(prompt: str, attempt: int = 0, force_json: bool = True) -> str:
    if client is None:
        raise RuntimeError("OpenAI client not initialised — check HF_TOKEN env var")

    max_retries = 3
    print(
        f"    [LLM] Calling {MODEL_NAME} (attempt {attempt+1}/{max_retries+1}) ...",
        flush=True,
    )
    try:
        kwargs: Dict[str, Any] = dict(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        completion = client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content or ""
        if hasattr(completion, "usage") and completion.usage:
            u = completion.usage
            print(
                f"    [LLM] Tokens prompt:{u.prompt_tokens} "
                f"completion:{u.completion_tokens} total:{u.total_tokens}",
                flush=True,
            )
        return content

    except Exception as exc:
        err_str = str(exc)
        if "rate_limit" in err_str or "429" in err_str:
            wait = 60
            m = re.search(r"Please try again in ([\d.]+)s", err_str)
            if m:
                wait = int(float(m.group(1))) + 2
            print(f"    [LLM] Rate limited — sleeping {wait}s ...", flush=True)
            time.sleep(wait)
            if attempt < max_retries:
                return _call_llm(prompt, attempt + 1, force_json)
        elif "json_object" in err_str or "response_format" in err_str:
            print("    [LLM] response_format not supported — retrying without it.",
                  flush=True)
            return _call_llm(prompt, attempt, force_json=False)
        elif "json_validate_failed" in err_str:
            print("    [LLM] Groq json_validate_failed — falling back to deterministic.",
                  flush=True)
            raise
        elif attempt < max_retries:
            wait = 4.0 * (attempt + 1)
            print(f"    [LLM] Error: {exc}. Retrying in {wait:.0f}s ...", flush=True)
            time.sleep(wait)
            return _call_llm(prompt, attempt + 1, force_json)
        raise


def _parse_response(text: str) -> Optional[Dict]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":   depth += 1
            elif ch == "}": depth -= 1
            if depth == 0:
                try:    return json.loads(text[start:i + 1])
                except: break
    return None


def _call_llm_with_retry_on_parse_fail(
    prompt: str, batch: List[Dict]
) -> Optional[Dict]:
    invoice_ids = [inv["invoice_id"] for inv in batch]

    try:
        raw = _call_llm(prompt)
    except Exception as exc:
        print(f"    [LLM] _call_llm raised: {exc} — skipping to deterministic.",
              flush=True)
        return None

    parsed = _parse_response(raw)
    if parsed and "reconciliation_result" in parsed:
        return parsed

    print("    [LLM] Parse failed — sending corrective prompt ...", flush=True)
    corrective = (
        f"Your previous response was not valid JSON or missing reconciliation_result.\n"
        f"Required invoice_ids: {invoice_ids}\n"
        f"Respond ONLY with a JSON object matching the required format.\n"
        f"Each invoice_id must have: status, correction_note, mismatch_fields.\n"
        f"claimable_itc must be exactly 0.0 — never an arithmetic expression.\n"
        f"Original data:\n{prompt}"
    )
    try:
        raw2 = _call_llm(corrective, attempt=0)
        parsed2 = _parse_response(raw2)
        if parsed2 and "reconciliation_result" in parsed2:
            print("    [LLM] Corrective prompt succeeded.", flush=True)
            return parsed2
    except Exception as e:
        print(f"    [LLM] Corrective prompt failed: {e}", flush=True)

    return None


def _run_batch(
    batch: List[Dict], obs: Dict, gstr_index: Dict[str, List[Dict]]
) -> List[Dict]:
    valid = {"MATCHED", "MISMATCH", "MISSING_IN_2B", "EXTRA_IN_2B"}
    prompt = _build_prompt(batch, obs)

    try:
        parsed = _call_llm_with_retry_on_parse_fail(prompt, batch)

        if parsed and "reconciliation_result" in parsed:
            llm_map = {
                e["invoice_id"]: e
                for e in parsed["reconciliation_result"]
                if isinstance(e, dict) and "invoice_id" in e
            }
            results = []
            n_llm = 0
            n_det = 0
            for inv in batch:
                iid = inv["invoice_id"]
                if iid in llm_map and llm_map[iid].get("status") in valid:
                    e = llm_map[iid]
                    if not isinstance(e.get("mismatch_fields"), list):
                        e["mismatch_fields"] = []
                    results.append(e)
                    n_llm += 1
                else:
                    reason = (
                        f"bad status={llm_map[iid].get('status')}"
                        if iid in llm_map else "omitted by LLM"
                    )
                    print(f"    [WARN] {iid} {reason} — per-invoice deterministic",
                          flush=True)
                    results.append(_deterministic_classify(inv, gstr_index))
                    n_det += 1
            if n_det:
                print(f"    [INFO] LLM:{n_llm} Det:{n_det} in this batch", flush=True)
            return results

        print("    [WARN] Both LLM attempts failed — full batch deterministic.",
              flush=True)

    except Exception as exc:
        print(f"    [ERROR] LLM failed: {exc} — full batch deterministic.", flush=True)

    return [_deterministic_classify(inv, gstr_index) for inv in batch]


# ── Task runner ─────────────────────────────────────────────────────────────
def run_task(task_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        print(f"\n  [RESET] Starting: {task_id}", flush=True)
        obs        = _post("/reset", {"task_id": task_id})
        invoices   = obs.get("invoices", [])
        n_invoices = len(invoices)
        max_itc    = float(obs.get("max_itc_possible", 0))
        print(f"  [RESET] {n_invoices} invoices | Max ITC: Rs{max_itc:,.2f}",
              flush=True)

        gstr_index: Dict[str, List[Dict]] = defaultdict(list)
        for e in obs.get("gstr2b_entries", []):
            gstr_index[e["invoice_number"]].append(e)

        batch_size = 8
        sleep_secs = 5
        batches    = [invoices[i:i + batch_size]
                      for i in range(0, n_invoices, batch_size)]
        n_batches  = len(batches)
        all_entries: List[Dict] = []

        print(
            f"  [LLM] Sending {n_invoices} invoices in "
            f"{n_batches} batch(es) of {batch_size} ...",
            flush=True,
        )

        for idx, batch in enumerate(batches):
            print(f"  [LLM] Batch {idx+1}/{n_batches} ({len(batch)} invoices) ...",
                  flush=True)
            entries = _run_batch(batch, obs, gstr_index)
            all_entries.extend(entries)
            print(f"    [LLM] Batch {idx+1} done — {len(entries)} entries.",
                  flush=True)
            if idx < n_batches - 1:
                time.sleep(sleep_secs)

        # Safety net: fill any invoice the LLM silently dropped
        seen = {e["invoice_id"] for e in all_entries}
        for inv in invoices:
            if inv["invoice_id"] not in seen:
                all_entries.append(_deterministic_classify(inv, gstr_index))
                print(f"  [WARN] Safety-net filled {inv['invoice_id']}", flush=True)

        _enrich_mismatch_fields(all_entries, obs)
        itc = _recompute_itc(all_entries, obs)

        counts: Dict[str, int] = defaultdict(int)
        for e in all_entries:
            counts[e["status"]] += 1
        print(f"  [BREAKDOWN] {dict(counts)}", flush=True)

        confidence = round(min(0.99, 0.99 - (n_batches - 1) * 0.001), 2)
        action = {
            "reconciliation_result": all_entries,
            "claimable_itc": itc,
            "confidence": confidence,
        }
        print(f"  [ACTION] ITC: Rs{itc:,.2f}  Confidence: {confidence}", flush=True)

        result      = _post("/step", action)
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

        print(f"\n  {'─'*55}", flush=True)
        print(f"  RESULTS: {task_id.upper()}", flush=True)
        print(f"  {'─'*55}", flush=True)
        print(f"  Total Reward  : {reward:.4f}", flush=True)
        print(f"  Match Score   : {match_score:.4f} ({correct}/{total_inv})",
              flush=True)
        print(f"  ITC Score     : {itc_score:.4f}", flush=True)
        print(f"  Task Score    : {score:.4f}", flush=True)
        print(f"  Batches       : {n_batches} ({n_invoices} invoices)", flush=True)
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
    print(f"  Model      : {MODEL_NAME}", flush=True)
    print(f"  API URL    : {API_BASE_URL}", flush=True)
    print(f"  Server     : {BASE_URL}", flush=True)
    print(f"  HF Token   : {'SET ✓' if HF_TOKEN else 'NOT SET ✗'}", flush=True)
    print("="*60 + "\n", flush=True)

    if not HF_TOKEN:
        print("[FATAL] HF_TOKEN env var not set. Export it before running:", flush=True)
        print("  export HF_TOKEN=gsk_your_key_here", flush=True)
        sys.exit(1)
    if client is None:
        print("[FATAL] OpenAI client failed to initialise.", flush=True)
        sys.exit(1)

    try:
        health = _get("/health")
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
    print(f"\n{'='*60}", flush=True)
    print("  FINAL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for tid, s in zip(task_ids, all_scores):
        print(f"  {tid:<25}: {s:.4f}", flush=True)
    print(f"  {'─'*40}", flush=True)
    print(f"  {'AVERAGE':<25}: {avg:.4f}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()