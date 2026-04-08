from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Dict, List, Any

from .env import GSTReconciliationEnv
from .models import Action, ReconciliationEntry, Invoice, GSTR2BEntry
from .graders import (
    grade_task1,
    grade_task2,
    grade_task3,
    grade_task4,
    grade_task5,
    grade_task6,
)


# ── Deterministic reconciliation logic ──────────────────────────────────────

def _build_gstr_index(gstr_entries: List[GSTR2BEntry]) -> Dict[str, List[GSTR2BEntry]]:
    """Index GSTR-2B entries by invoice_number for O(1) lookup."""
    index: Dict[str, List[GSTR2BEntry]] = defaultdict(list)
    for entry in gstr_entries:
        index[entry.invoice_number].append(entry)
    return index


def _classify_invoice(
    inv: Invoice,
    gstr_index: Dict[str, List[GSTR2BEntry]],
) -> ReconciliationEntry:
    """
    Deterministic classification using exact field comparison.
    Threshold: amount differences > Rs 1.00 are mismatches.
    """
    matches = gstr_index.get(inv.invoice_number, [])

    if len(matches) == 0:
        return ReconciliationEntry(
            invoice_id=inv.invoice_id,
            status="MISSING_IN_2B",
            correction_note="Invoice number not found in GSTR-2B portal",
            mismatch_fields=[],
        )

    if len(matches) > 1:
        return ReconciliationEntry(
            invoice_id=inv.invoice_id,
            status="EXTRA_IN_2B",
            correction_note=f"Invoice appears {len(matches)} times in GSTR-2B — possible duplicate filing",
            mismatch_fields=[],
        )

    gstr = matches[0]
    mismatch_fields: List[str] = []
    notes: List[str] = []

    # GSTIN check (case-insensitive)
    if inv.vendor_gstin.upper() != gstr.supplier_gstin.upper():
        mismatch_fields.append("supplier_gstin")
        notes.append(f"GSTIN: {inv.vendor_gstin} vs {gstr.supplier_gstin}")

    # Date check (exact)
    if inv.invoice_date != gstr.invoice_date:
        mismatch_fields.append("invoice_date")
        notes.append(f"Date: {inv.invoice_date} vs {gstr.invoice_date}")

    # Amount checks (threshold Rs 1)
    threshold = Decimal("1.00")
    for field in ("taxable_value", "cgst", "sgst", "igst"):
        inv_val = getattr(inv, field)
        gstr_val = getattr(gstr, field)
        if abs(Decimal(str(inv_val)) - Decimal(str(gstr_val))) > threshold:
            mismatch_fields.append(field)
            notes.append(f"{field}: {inv_val} vs {gstr_val}")

    # ITC availability check
    if not gstr.itc_available:
        mismatch_fields.append("itc_available")
        notes.append(f"ITC not available (document_type={gstr.document_type})")

    if mismatch_fields:
        return ReconciliationEntry(
            invoice_id=inv.invoice_id,
            status="MISMATCH",
            correction_note="; ".join(notes),
            mismatch_fields=mismatch_fields,
        )

    return ReconciliationEntry(
        invoice_id=inv.invoice_id,
        status="MATCHED",
        correction_note=None,
        mismatch_fields=[],
    )


def _run_deterministic(env: GSTReconciliationEnv, task_id: str) -> tuple[Action, float]:
    """Reset env, run deterministic reconciliation, return action and ITC."""
    obs = env.reset(task_id)
    gstr_index = _build_gstr_index(obs.gstr2b_entries)

    entries = [_classify_invoice(inv, gstr_index) for inv in obs.invoices]

    # ITC = sum(cgst+sgst+igst) for MATCHED invoices only
    inv_map = {inv.invoice_id: inv for inv in obs.invoices}
    itc = Decimal("0")
    for e in entries:
        if e.status == "MATCHED":
            inv = inv_map[e.invoice_id]
            itc += Decimal(str(inv.cgst)) + Decimal(str(inv.sgst)) + Decimal(str(inv.igst))

    action = Action(
        reconciliation_result=entries,
        claimable_itc=itc,
        confidence=0.95,
    )
    return action, float(itc)


# ── Main baseline runner ─────────────────────────────────────────────────────

def run_baseline() -> Dict[str, Any]:
    """
    Run the deterministic baseline agent against all 6 tasks.

    This baseline uses exact field matching (no LLM) and serves as a reproducible
    lower bound. A good LLM agent should consistently beat this baseline.
    """
    base_env = GSTReconciliationEnv()
    scores: Dict[str, float] = {}
    graders = {
        "task1_easy": grade_task1,
        "task2_medium": grade_task2,
        "task3_hard": grade_task3,
        "task4_credit_notes": grade_task4,
        "task5_stress": grade_task5,
        "task6_mixed_docs": grade_task6,
    }

    for task_id, grader_fn in graders.items():
        action, _ = _run_deterministic(base_env, task_id)
        score = grader_fn(action, base_env.ground_truth)
        scores[task_id] = round(score, 4)

    avg = round(sum(scores.values()) / len(scores), 4)
    scores["average"] = avg
    return scores


if __name__ == "__main__":
    results = run_baseline()
    print("\nBaseline Results (Deterministic Agent):")
    print("=" * 45)
    for k, v in results.items():
        marker = " ← avg" if k == "average" else ""
        print(f"  {k:<25}: {v:.4f}{marker}")
    print()
    print("Note: Deterministic baseline uses exact field matching.")
    print("A good LLM agent should score significantly higher on hard tasks.")