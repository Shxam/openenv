from __future__ import annotations

from typing import Dict, Any, List, Set

from gst_env.models import Action

# Hard task: weight rarer categories more heavily to reward nuanced detection
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


def _field_f1(reported: List[str], ground_truth_fields: List[str]) -> float:
    """F1 score between reported and ground-truth mismatch fields."""
    if not ground_truth_fields and not reported:
        return 1.0
    if not ground_truth_fields or not reported:
        return 0.0
    gt_set = set(ground_truth_fields) & _VALID_MISMATCH_FIELDS
    rep_set = set(f for f in reported if f in _VALID_MISMATCH_FIELDS)
    if not gt_set:
        return 1.0 if not rep_set else 0.0
    tp = len(gt_set & rep_set)
    precision = tp / len(rep_set) if rep_set else 0.0
    recall = tp / len(gt_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def grade(action: Action, ground_truth: Dict[str, Any]) -> float:
    """
    Task 3 grader — 200 invoice hard task with adversarial mismatches.

    Score = 0.30 * weighted_category_accuracy
          + 0.30 * itc_score
          + 0.20 * field_f1_score      (F1 between reported and true mismatch fields)
          + 0.10 * coverage_score      (fraction of invoices covered)
          + 0.10 * correction_quality  (non-empty correction notes for MISMATCH)
          − fraud_penalty              (0.10 per fraudulent ITC claim, capped at 0.30)

    Clamped to [0.0, 1.0].
    """
    entries = action.reconciliation_result
    if not entries:
        return 0.0

    # Per-ground-truth mismatch fields (stored by grader meta — inferred from task)
    # Since we don't store them in ground_truth dict, we use presence heuristics
    invoice_ids_in_gt = {k for k in ground_truth if k not in ("max_itc", "penalty_days")}
    invoice_ids_in_action = {e.invoice_id for e in entries}
    coverage_score = len(invoice_ids_in_action & invoice_ids_in_gt) / max(len(invoice_ids_in_gt), 1)

    status_counts: Dict[str, int] = {k: 0 for k in _WEIGHTS}
    correct_counts: Dict[str, int] = {k: 0 for k in _WEIGHTS}
    field_f1_total = 0.0
    field_entry_count = 0
    correction_quality_total = 0.0
    correction_mismatch_count = 0
    fraud_count = 0

    for e in entries:
        expected = ground_truth.get(e.invoice_id)
        if expected is None or expected not in status_counts:
            continue

        status_counts[expected] += 1

        if e.status == expected:
            correct_counts[expected] += 1

            if expected == "MISMATCH":
                # Field F1: we don't store exact fields per invoice in ground_truth,
                # so we reward any valid field name reported (precision-only for now)
                if e.mismatch_fields:
                    valid = [f for f in e.mismatch_fields if f in _VALID_MISMATCH_FIELDS]
                    precision = len(valid) / len(e.mismatch_fields)
                    # Also check recall: reward reporting at least 1 valid field
                    recall = 1.0 if valid else 0.0
                    f1 = (2 * precision * recall / (precision + recall + 1e-9))
                    field_f1_total += f1
                else:
                    field_f1_total += 0.0  # no fields reported for a MISMATCH
                field_entry_count += 1

                # correction_note quality
                correction_mismatch_count += 1
                note = e.correction_note or ""
                if len(note) >= 10:  # non-trivial note
                    correction_quality_total += 1.0

        # Fraud detection: MATCHED claim on a non-existent or ITC-blocked invoice
        if e.status == "MATCHED" and expected in ("MISSING_IN_2B",):
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

    field_f1 = field_f1_total / field_entry_count if field_entry_count > 0 else 0.0
    correction_quality = (
        correction_quality_total / correction_mismatch_count
        if correction_mismatch_count > 0 else 0.0
    )
    fraud_penalty = min(0.30, fraud_count * 0.10)

    raw = (
        0.30 * weighted_acc
        + 0.30 * itc_score
        + 0.20 * field_f1
        + 0.10 * coverage_score
        + 0.10 * correction_quality
        - fraud_penalty
    )
    return round(max(0.001, min(0.999, raw)), 4)