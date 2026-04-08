from __future__ import annotations

from typing import Dict, Any, Set

from gst_env.models import Action

_WEIGHTS: Dict[str, float] = {
    "MATCHED": 0.20,
    "MISMATCH": 0.30,
    "MISSING_IN_2B": 0.30,
    "EXTRA_IN_2B": 0.20,
}

_VALID_MISMATCH_FIELDS: Set[str] = {
    "taxable_value", "invoice_date", "supplier_gstin",
    "cgst", "sgst", "igst", "itc_available",
}


def grade(action: Action, ground_truth: Dict[str, Any]) -> float:
    """
    Task 5 grader — 500-invoice high-volume stress test.

    Score = 0.30 * weighted_category_accuracy
          + 0.30 * itc_score
          + 0.20 * field_precision_score
          + 0.10 * coverage_score      (misses penalised at 0.001 each, cap -0.10)
          + 0.10 * filing_timeliness
          − fraud_penalty              (0.05 per fraudulent claim, capped at 0.25)

    At 500 invoices, coverage matters: missing invoices in agent output are common.
    """
    entries = action.reconciliation_result
    if not entries:
        return 0.0

    invoice_ids_in_gt = {k for k in ground_truth if k not in ("max_itc", "penalty_days")}
    invoice_ids_in_action = {e.invoice_id for e in entries}

    # Coverage: penalise each missing invoice
    missing_count = len(invoice_ids_in_gt - invoice_ids_in_action)
    coverage_score = max(0.0, 1.0 - missing_count * 0.002)

    status_counts: Dict[str, int] = {k: 0 for k in _WEIGHTS}
    correct_counts: Dict[str, int] = {k: 0 for k in _WEIGHTS}
    field_precision_total = 0.0
    field_entry_count = 0
    fraud_count = 0

    for e in entries:
        expected = ground_truth.get(e.invoice_id)
        if expected is None or expected not in status_counts:
            continue

        status_counts[expected] += 1

        if e.status == expected:
            correct_counts[expected] += 1

            if expected == "MISMATCH" and e.mismatch_fields:
                valid = sum(1 for f in e.mismatch_fields if f in _VALID_MISMATCH_FIELDS)
                precision = valid / len(e.mismatch_fields)
                field_precision_total += precision
                field_entry_count += 1

        if e.status == "MATCHED" and expected == "MISSING_IN_2B":
            fraud_count += 1

    weighted_acc = 0.0
    for k, w in _WEIGHTS.items():
        if status_counts[k] > 0:
            weighted_acc += w * (correct_counts[k] / status_counts[k])
        else:
            weighted_acc += w

    true_itc = float(ground_truth.get("max_itc", 0.0))
    pred_itc = float(action.claimable_itc)
    itc_error = abs(pred_itc - true_itc) / (true_itc + 1e-9)
    itc_score = max(0.0, 1.0 - itc_error)

    field_precision = field_precision_total / field_entry_count if field_entry_count > 0 else 0.0
    penalty_days = int(ground_truth.get("penalty_days", 0))
    filing_timeliness = max(0.0, 1.0 - penalty_days / 30.0)
    fraud_penalty = min(0.25, fraud_count * 0.05)

    raw = (
        0.30 * weighted_acc
        + 0.30 * itc_score
        + 0.20 * field_precision
        + 0.10 * coverage_score
        + 0.10 * filing_timeliness
        - fraud_penalty
    )
    return round(max(0.0, min(1.0, raw)), 4)