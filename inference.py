#!/usr/bin/env python3
"""
inference.py — GST Reconciliation OpenEnv — Smart Triage + Batched LLM Agent

Architecture (60-40 Deterministic-First Hybrid):
  Phase 1 — Deterministic pre-filter on ALL invoices (~1-2s per task)
             Checks: tax formula, GSTIN regex, date validity, ITC rules, amount threshold
             → high_confidence (~60-70%): final labels locked in
             → low_confidence (~30-40%): sent to LLM for review

  Phase 2 — ONE Groq LLM call per task for low-confidence invoices only
             Hard timeout: 25 seconds. Task budget: 240s (4 min × 6 tasks = 24 min total).
             Smart prompt includes deterministic_label so LLM only corrects, not re-classifies.

  Phase 3 — Merge: LLM results override deterministic for low-confidence;
             deterministic fallback if LLM times out or fails.

Performance target (per task):
  Deterministic pre-filter : ~1-2s
  Single LLM batch call    : ~15-20s
  Merge + ITC recompute    : <1s
  Total per task           : ~22-25s
  6 tasks total            : ~150s (2.5 min) — well under 30-min limit

Environment variables:
  API_BASE_URL  : LLM API base URL  (default: https://api.groq.com/openai/v1)
  MODEL_NAME    : LLM model name    (default: llama-3.3-70b-versatile)
  HF_TOKEN      : API key / token   (Set via HF Space Secrets — never hardcode)
  BASE_URL      : Env server URL    (default: http://localhost:7860)
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import date as date_type
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

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

# Timing budgets (all in seconds)
LLM_CALL_TIMEOUT    = 25    # hard cap per LLM call (asyncio.wait_for style via openai timeout)
TASK_BUDGET_SECONDS = 240   # 4 min per task; 6 tasks = 24 min total (under 30-min limit)
TASK_SKIP_BUFFER    = 40    # if elapsed > TASK_BUDGET - SKIP_BUFFER, skip LLM call
HTTP_STEP_TIMEOUT   = 120   # /step can be slow for 500-invoice payloads

# Gap between tasks to avoid Groq rate-limit windows
INTER_TASK_SLEEP = 1

# ── GSTIN validation regex ──────────────────────────────────────────────────
import re as _re
_GSTIN_REGEX = _re.compile(
    r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}$"
)

# OCR confusion pairs
_OCR_PAIRS: Dict[str, str] = {
    "0": "O", "O": "0",
    "1": "I", "I": "1",
    "8": "B", "B": "8",
    "5": "S", "S": "5",
    "2": "Z", "Z": "2",
}

# Tax rates by approximate check (GST is 0%, 5%, 12%, 18%, 28%)
_GST_RATES = [Decimal("0.00"), Decimal("0.05"), Decimal("0.12"),
              Decimal("0.18"), Decimal("0.28")]
_AMOUNT_THRESHOLD = Decimal("1.00")
_NEAR_MISS_WINDOW  = Decimal("10.00")   # Rs 1–10: flagged borderline for LLM

# ── OpenAI client ───────────────────────────────────────────────────────────
client = None
try:
    from openai import OpenAI
    if HF_TOKEN:
        client = OpenAI(
            api_key=HF_TOKEN,
            base_url=API_BASE_URL,
            timeout=LLM_CALL_TIMEOUT + 5,  # slightly wider than our manual cap
            max_retries=0,                  # zero retries — time is critical
        )
        print(f"[CONFIG] LLM client ready — model={MODEL_NAME}", flush=True)
    else:
        print("[WARN] HF_TOKEN not set — LLM disabled, running deterministic-only.", flush=True)
except Exception as e:
    print(f"[WARN] OpenAI client init failed: {e} — running deterministic-only.", flush=True)


# ── Groq LLM prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a GST compliance checker for Indian invoices.
Deterministic logic has already produced a label for each invoice below.
Your role: ONLY change the label when confident the deterministic result is wrong.

Override when:
- Amount is near-miss (Rs 1–10 from threshold) and you are confident it's still MISMATCH or MATCHED
- GSTIN looks like OCR-corruption (0↔O, 1↔I, 8↔B, 5↔S, 2↔Z) — if swap makes it match, it's still MISMATCH (changed GSTIN = MISMATCH)
- Fields are ambiguous or itc_available=False was missed
- document_type is credit_note/debit_note/advance_receipt → MISMATCH

Valid statuses: MATCHED MISMATCH MISSING_IN_2B EXTRA_IN_2B
Valid mismatch_fields: taxable_value invoice_date supplier_gstin cgst sgst igst itc_available

Respond ONLY with this JSON (no markdown, no explanation):
{
  "results": {
    "<invoice_id>": {
      "status": "MATCHED|MISMATCH|MISSING_IN_2B|EXTRA_IN_2B",
      "correction_note": "brief plain-text reason",
      "mismatch_fields": ["field1", "field2"]
    }
  }
}

Every invoice_id given to you MUST appear in results.
mismatch_fields must be [] unless status is MISMATCH."""


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


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Deterministic pre-filter
# ════════════════════════════════════════════════════════════════════════════

def _validate_gstin(gstin: str) -> bool:
    """True if GSTIN passes format check."""
    return bool(_GSTIN_REGEX.match(gstin.upper()))


def _is_ocr_gstin_error(a: str, b: str) -> bool:
    """True if a and b differ by exactly 1 character that is an OCR confusion pair."""
    a, b = a.upper(), b.upper()
    if len(a) != 15 or len(b) != 15:
        return False
    diffs = [(ca, cb) for ca, cb in zip(a, b) if ca != cb]
    if len(diffs) != 1:
        return False
    ca, cb = diffs[0]
    return _OCR_PAIRS.get(ca) == cb


def _check_tax_formula(taxable: Decimal, cgst: Decimal, sgst: Decimal, igst: Decimal) -> bool:
    """
    Returns True if the tax amounts are consistent with any standard GST rate.
    Intra-state: cgst == sgst == taxable * rate/2; inter-state: igst == taxable * rate
    """
    tolerance = Decimal("2.00")
    for rate in _GST_RATES:
        half = (taxable * rate / 2).quantize(Decimal("0.01"))
        full = (taxable * rate).quantize(Decimal("0.01"))
        # Intra-state
        if (abs(cgst - half) <= tolerance and
                abs(sgst - half) <= tolerance and
                igst == Decimal("0")):
            return True
        # Inter-state
        if (cgst == Decimal("0") and
                sgst == Decimal("0") and
                abs(igst - full) <= tolerance):
            return True
    return False


class _ClassifiedInvoice:
    """Result of deterministic classification for one invoice."""
    __slots__ = (
        "invoice_id", "status", "correction_note", "mismatch_fields",
        "high_confidence", "borderline_reason",
    )

    def __init__(
        self,
        invoice_id: str,
        status: str,
        correction_note: Optional[str],
        mismatch_fields: List[str],
        high_confidence: bool,
        borderline_reason: str = "",
    ):
        self.invoice_id = invoice_id
        self.status = status
        self.correction_note = correction_note
        self.mismatch_fields = mismatch_fields
        self.high_confidence = high_confidence
        self.borderline_reason = borderline_reason

    def to_dict(self) -> Dict:
        return {
            "invoice_id": self.invoice_id,
            "status": self.status,
            "correction_note": self.correction_note,
            "mismatch_fields": self.mismatch_fields,
        }


def _deterministic_classify(
    inv: Dict, gstr_index: Dict[str, List[Dict]]
) -> _ClassifiedInvoice:
    """
    Phase 1 deterministic classifier.

    Confidence rules:
    - high_confidence=True  → label is final, LLM skips this invoice
    - high_confidence=False → borderline, send to LLM for review

    Borderline triggers:
    - Near-miss amount (Rs 1.01–10.00 difference)
    - Single OCR-style GSTIN character error
    - Multi-field mismatch (amount + date both wrong)
    - invoice has suspicious tax formula
    """
    iid = inv["invoice_id"]
    matches = gstr_index.get(inv["invoice_number"], [])

    # ── MISSING_IN_2B (always high-confidence) ──────────────────────────────
    if len(matches) == 0:
        return _ClassifiedInvoice(
            invoice_id=iid,
            status="MISSING_IN_2B",
            correction_note="Invoice number not found in GSTR-2B portal",
            mismatch_fields=[],
            high_confidence=True,
        )

    # ── EXTRA_IN_2B (always high-confidence) ────────────────────────────────
    if len(matches) > 1:
        return _ClassifiedInvoice(
            invoice_id=iid,
            status="EXTRA_IN_2B",
            correction_note=f"Appears {len(matches)} times in GSTR-2B — possible duplicate filing",
            mismatch_fields=[],
            high_confidence=True,
        )

    gstr = matches[0]
    diff_fields: List[str] = []
    notes: List[str] = []
    borderline = False
    borderline_reasons: List[str] = []

    # ── GSTIN check ─────────────────────────────────────────────────────────
    inv_gstin  = inv.get("vendor_gstin", "").upper()
    gstr_gstin = gstr.get("supplier_gstin", "").upper()
    if inv_gstin != gstr_gstin:
        diff_fields.append("supplier_gstin")
        notes.append(f"GSTIN: {inv_gstin} vs {gstr_gstin}")
        if _is_ocr_gstin_error(inv_gstin, gstr_gstin):
            borderline = True
            borderline_reasons.append("OCR GSTIN error (1-char OCR confusion)")

    # ── Date check ───────────────────────────────────────────────────────────
    inv_date  = str(inv.get("invoice_date", ""))
    gstr_date = str(gstr.get("invoice_date", ""))
    date_differs = inv_date != gstr_date
    if date_differs:
        diff_fields.append("invoice_date")
        notes.append(f"Date: {inv_date} vs {gstr_date}")

    # ── Amount checks ────────────────────────────────────────────────────────
    amount_fields: List[str] = []
    for f in ["taxable_value", "cgst", "sgst", "igst"]:
        try:
            inv_val  = Decimal(str(inv.get(f, 0)))
            gstr_val = Decimal(str(gstr.get(f, 0)))
            diff_amt = abs(inv_val - gstr_val)
            if diff_amt > _AMOUNT_THRESHOLD:
                amount_fields.append(f)
                diff_fields.append(f)
                # Near-miss: Rs 1.01–10.00 — still MISMATCH but borderline
                if diff_amt <= _NEAR_MISS_WINDOW:
                    borderline = True
                    borderline_reasons.append(f"Near-miss amount in {f} (diff Rs {diff_amt:.2f})")
        except Exception:
            diff_fields.append(f)

    if amount_fields:
        notes.append(f"Amount mismatch: {', '.join(amount_fields)}")

    # Multi-field: both amount and date differ → borderline (LLM writes better note)
    if amount_fields and date_differs:
        borderline = True
        borderline_reasons.append("Multi-field: amount + date both differ")

    # ── ITC / document type check ────────────────────────────────────────────
    itc_available = gstr.get("itc_available", True)
    doc_type = gstr.get("document_type", "invoice")
    if not itc_available or doc_type in ("credit_note", "debit_note", "advance_receipt"):
        if "itc_available" not in diff_fields:
            diff_fields.append("itc_available")
        notes.append(f"ITC not available (doc_type={doc_type})")

    # ── Tax formula check (flags borderline if suspicious) ───────────────────
    try:
        taxable = Decimal(str(inv.get("taxable_value", 0)))
        cgst    = Decimal(str(inv.get("cgst", 0)))
        sgst    = Decimal(str(inv.get("sgst", 0)))
        igst    = Decimal(str(inv.get("igst", 0)))
        if not _check_tax_formula(taxable, cgst, sgst, igst):
            borderline = True
            borderline_reasons.append("Suspicious tax formula (rate mismatch)")
    except Exception:
        pass

    # ── Final classification ──────────────────────────────────────────────────
    if diff_fields:
        correction_note = "; ".join(notes) if notes else "Field mismatch detected"
        # High confidence only when diff is large (>Rs 10) and not OCR GSTIN
        high_conf = (not borderline)
        return _ClassifiedInvoice(
            invoice_id=iid,
            status="MISMATCH",
            correction_note=correction_note,
            mismatch_fields=diff_fields,
            high_confidence=high_conf,
            borderline_reason="; ".join(borderline_reasons),
        )

    return _ClassifiedInvoice(
        invoice_id=iid,
        status="MATCHED",
        correction_note=None,
        mismatch_fields=[],
        high_confidence=True,
    )


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


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Single LLM batch call per task
# ════════════════════════════════════════════════════════════════════════════

def _build_llm_prompt(
    low_conf: List[_ClassifiedInvoice],
    invoices: List[Dict],
    gstr_index: Dict[str, List[Dict]],
) -> str:
    """Build a compact prompt for only the borderline invoices."""
    inv_map = {i["invoice_id"]: i for i in invoices}
    seen_gstr: set = set()
    inv_lines  = []
    gstr_lines = []

    for clf in low_conf:
        inv = inv_map.get(clf.invoice_id, {})
        inv_lines.append(
            f'{clf.invoice_id}|{inv.get("invoice_number","")}'
            f'|{inv.get("vendor_gstin","")}'
            f'|{inv.get("invoice_date","")}'
            f'|taxable={float(str(inv.get("taxable_value",0))):.2f}'
            f'|cgst={float(str(inv.get("cgst",0))):.2f}'
            f'|sgst={float(str(inv.get("sgst",0))):.2f}'
            f'|igst={float(str(inv.get("igst",0))):.2f}'
            f'|det={clf.status}'
            f'|why={clf.borderline_reason}'
        )
        inv_num = inv.get("invoice_number", "")
        for g in gstr_index.get(inv_num, []):
            key = g["invoice_number"]
            if key not in seen_gstr:
                seen_gstr.add(key)
                gstr_lines.append(
                    f'{g["invoice_number"]}'
                    f'|{g.get("supplier_gstin","")}'
                    f'|{g.get("invoice_date","")}'
                    f'|taxable={float(str(g.get("taxable_value",0))):.2f}'
                    f'|cgst={float(str(g.get("cgst",0))):.2f}'
                    f'|sgst={float(str(g.get("sgst",0))):.2f}'
                    f'|igst={float(str(g.get("igst",0))):.2f}'
                    f'|itc={g.get("itc_available",True)}'
                    f'|doc={g.get("document_type","invoice")}'
                )

    ids = [c.invoice_id for c in low_conf]
    return (
        f"Review {len(low_conf)} borderline invoices (deterministic label included).\n"
        f"Required invoice_ids (ALL must appear in results): {ids}\n\n"
        f"INVOICES (id|number|gstin|date|taxable|cgst|sgst|igst|det_label|reason):\n"
        + "\n".join(inv_lines)
        + "\n\nGSTR2B (number|gstin|date|taxable|cgst|sgst|igst|itc|doc):\n"
        + ("\n".join(gstr_lines) if gstr_lines else "(none)")
        + "\n\nRespond with the JSON object only."
    )


def _call_llm_once(
    prompt: str,
    task_start: float,
) -> Optional[str]:
    """
    Single LLM call with hard LLM_CALL_TIMEOUT.
    Checks task budget before calling — skips if time is running out.
    Returns raw string or None on any failure/timeout.
    """
    if client is None:
        return None

    elapsed = time.time() - task_start
    remaining = TASK_BUDGET_SECONDS - elapsed
    if elapsed > (TASK_BUDGET_SECONDS - TASK_SKIP_BUFFER):
        print(
            f"    [LLM] Skipping — only {remaining:.0f}s left in task budget.",
            flush=True,
        )
        return None

    print(
        f"    [LLM] Calling {MODEL_NAME} (timeout={LLM_CALL_TIMEOUT}s, "
        f"elapsed={elapsed:.1f}s/{TASK_BUDGET_SECONDS}s) ...",
        flush=True,
    )
    t_llm = time.time()
    try:
        kwargs: Dict[str, Any] = dict(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        try:
            kwargs["response_format"] = {"type": "json_object"}
        except Exception:
            pass

        completion = client.chat.completions.create(**kwargs)
        llm_elapsed = time.time() - t_llm
        content = completion.choices[0].message.content or ""

        if hasattr(completion, "usage") and completion.usage:
            u = completion.usage
            print(
                f"    [LLM] Done in {llm_elapsed:.1f}s | "
                f"tokens: prompt={u.prompt_tokens} completion={u.completion_tokens}",
                flush=True,
            )
        return content

    except Exception as exc:
        llm_elapsed = time.time() - t_llm
        err = str(exc)
        if "rate_limit" in err or "429" in err:
            print(f"    [LLM] Rate limited after {llm_elapsed:.1f}s — no retry (time critical).", flush=True)
        elif "timeout" in err.lower() or "timed out" in err.lower():
            print(f"    [LLM] Timed out after {llm_elapsed:.1f}s → falling back to deterministic.", flush=True)
        else:
            print(f"    [LLM] Error after {llm_elapsed:.1f}s: {exc} → deterministic.", flush=True)
        return None


def _parse_llm_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract first JSON object
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


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Merge results
# ════════════════════════════════════════════════════════════════════════════

def _merge_results(
    all_classified: List[_ClassifiedInvoice],
    llm_json: Optional[Dict],
) -> List[Dict]:
    """
    Merge deterministic + LLM results.

    - high_confidence invoices always keep deterministic label
    - low_confidence invoices: use LLM result if valid, else deterministic
    """
    valid_statuses = {"MATCHED", "MISMATCH", "MISSING_IN_2B", "EXTRA_IN_2B"}
    llm_results: Dict[str, Dict] = {}

    if llm_json:
        # Support {"results": {id: {...}}} format
        raw = llm_json.get("results") or {}
        if isinstance(raw, dict):
            llm_results = raw
        # Also support list format
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and "invoice_id" in item:
                    llm_results[item["invoice_id"]] = item

    final: List[Dict] = []
    n_llm = 0
    n_det = 0

    for clf in all_classified:
        iid = clf.invoice_id
        if clf.high_confidence:
            final.append(clf.to_dict())
            n_det += 1
        else:
            # Try LLM result
            llm_item = llm_results.get(iid)
            if isinstance(llm_item, dict):
                status = llm_item.get("status")
                if status in valid_statuses:
                    mf = llm_item.get("mismatch_fields", [])
                    if not isinstance(mf, list):
                        mf = []
                    final.append({
                        "invoice_id": iid,
                        "status": status,
                        "correction_note": llm_item.get("correction_note") or clf.correction_note,
                        "mismatch_fields": mf,
                    })
                    n_llm += 1
                    continue

            # Fallback to deterministic
            final.append(clf.to_dict())
            n_det += 1

    print(
        f"  [MERGE] LLM-used={n_llm} | Deterministic-used={n_det} | Total={len(final)}",
        flush=True,
    )
    return final


# ── Task runner ─────────────────────────────────────────────────────────────
def run_task(task_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME if client else "deterministic-only")

    try:
        task_start = time.time()
        print(f"\n  [RESET] Starting: {task_id}", flush=True)
        obs = _post("/reset", {"task_id": task_id}, timeout=60)
        invoices   = obs.get("invoices", [])
        n_invoices = len(invoices)
        max_itc    = float(obs.get("max_itc_possible", 0))
        print(f"  [RESET] {n_invoices} invoices | Max ITC: Rs{max_itc:,.2f}", flush=True)

        # Build GSTR-2B lookup index
        gstr_index: Dict[str, List[Dict]] = defaultdict(list)
        for e in obs.get("gstr2b_entries", []):
            gstr_index[e["invoice_number"]].append(e)

        # ────────────────────────────────────────────────────────────────────
        # PHASE 1 — Deterministic pre-filter (all invoices)
        # ────────────────────────────────────────────────────────────────────
        t1 = time.time()
        all_classified: List[_ClassifiedInvoice] = [
            _deterministic_classify(inv, gstr_index) for inv in invoices
        ]
        low_conf  = [c for c in all_classified if not c.high_confidence]
        high_conf = [c for c in all_classified if c.high_confidence]

        counts_det: Dict[str, int] = defaultdict(int)
        for c in all_classified:
            counts_det[c.status] += 1
        print(
            f"  [P1] {n_invoices} invoices in {time.time()-t1:.2f}s | "
            f"{dict(counts_det)} | "
            f"high_conf={len(high_conf)} low_conf={len(low_conf)} "
            f"({100*len(low_conf)/max(n_invoices,1):.0f}% to LLM)",
            flush=True,
        )

        # ────────────────────────────────────────────────────────────────────
        # PHASE 2 — Single LLM call for low-confidence invoices
        # ────────────────────────────────────────────────────────────────────
        llm_json: Optional[Dict] = None
        if low_conf and client is not None:
            prompt = _build_llm_prompt(low_conf, invoices, gstr_index)
            raw = _call_llm_once(prompt, task_start)
            if raw:
                llm_json = _parse_llm_json(raw)
                if not llm_json:
                    print("    [LLM] Parse failed — all borderline keep deterministic label.", flush=True)
                else:
                    print(f"    [LLM] Parsed successfully.", flush=True)
        elif not low_conf:
            print(f"  [P2] No borderline cases — LLM call skipped.", flush=True)
        else:
            print(f"  [P2] LLM not available — deterministic-only.", flush=True)

        # ────────────────────────────────────────────────────────────────────
        # PHASE 3 — Merge and submit
        # ────────────────────────────────────────────────────────────────────
        final_entries = _merge_results(all_classified, llm_json)

        # Recompute ITC from actual MATCHED invoices
        itc = _recompute_itc(final_entries, invoices)

        counts_final: Dict[str, int] = defaultdict(int)
        for e in final_entries:
            counts_final[e["status"]] += 1
        print(f"  [P3] Final breakdown: {dict(counts_final)}", flush=True)

        classification_elapsed = time.time() - task_start
        print(
            f"  [ACTION] ITC: Rs{itc:,.2f} | "
            f"Classification: {classification_elapsed:.1f}s/{TASK_BUDGET_SECONDS}s",
            flush=True,
        )

        action = {
            "reconciliation_result": final_entries,
            "claimable_itc": itc,
            "confidence": 0.92 if llm_json else 0.85,
        }

        result      = _post("/step", action, timeout=HTTP_STEP_TIMEOUT)
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

        task_elapsed = time.time() - task_start
        print(f"\n  {'─'*55}", flush=True)
        print(f"  RESULTS: {task_id.upper()}", flush=True)
        print(f"  {'─'*55}", flush=True)
        print(f"  Total Reward  : {reward:.4f}", flush=True)
        print(f"  Match Score   : {match_score:.4f} ({correct}/{total_inv})", flush=True)
        print(f"  ITC Score     : {itc_score:.4f}", flush=True)
        print(f"  Task Score    : {score:.4f}", flush=True)
        print(f"  Task Elapsed  : {task_elapsed:.1f}s / {TASK_BUDGET_SECONDS}s budget", flush=True)
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
    print("  GST Reconciliation — Smart Triage + Batched LLM Agent", flush=True)
    print(f"  Strategy   : Deterministic (60-70%) + LLM (30-40% borderline)", flush=True)
    print(f"  Model      : {MODEL_NAME}", flush=True)
    print(f"  LLM Timeout: {LLM_CALL_TIMEOUT}s per call", flush=True)
    print(f"  Task Budget: {TASK_BUDGET_SECONDS}s × 6 tasks = {TASK_BUDGET_SECONDS*6}s max", flush=True)
    print(f"  API URL    : {API_BASE_URL}", flush=True)
    print(f"  Server     : {BASE_URL}", flush=True)
    print(f"  HF Token   : {'SET ✓' if HF_TOKEN else 'NOT SET — running deterministic-only'}", flush=True)
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

    for i, task_id in enumerate(task_ids):
        print(f"\n{'='*60}", flush=True)
        print(f"  RUNNING: {task_id.upper()} ({i+1}/{len(task_ids)})", flush=True)
        print(f"  Total elapsed so far: {time.time()-t_total:.1f}s", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            s = run_task(task_id)
            all_scores.append(s)
        except Exception as exc:
            print(f"[ERROR] {task_id} crashed: {exc}", flush=True)
            import traceback; traceback.print_exc()
            all_scores.append(0.0)

        # Small sleep between tasks to avoid Groq rate-limit windows
        if i < len(task_ids) - 1:
            time.sleep(INTER_TASK_SLEEP)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print("  FINAL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for tid, s in zip(task_ids, all_scores):
        print(f"  {tid:<25}: {s:.4f}", flush=True)
    print(f"  {'─'*40}", flush=True)
    print(f"  {'AVERAGE':<25}: {avg:.4f}", flush=True)
    print(f"  {'─'*40}", flush=True)
    print(f"  {'TOTAL TIME':<25}: {total_elapsed:.1f}s", flush=True)
    print(f"  {'TIME LIMIT':<25}: 1800s", flush=True)
    print(f"  {'UTILISATION':<25}: {total_elapsed/1800*100:.1f}%", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()