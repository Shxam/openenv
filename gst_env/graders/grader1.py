from __future__ import annotations

from decimal import Decimal
from typing import Dict, Any

from gst_env.models import Action


def grade(action: Action, ground_truth: Dict[str, Any]) -> float:
    """
    Task 1 grader — 10 perfectly matched invoices.
    Score = 0.70 * accuracy + 0.30 * itc_score − fraud_penalty

    fraud_penalty: 0.05 per invoice claimed MATCHED when ground truth is MISSING_IN_2B
    (capped at 0.15). Simulates the real-world cost of fraudulent ITC claims.
    """
    entries = action.reconciliation_result
    if not entries:
        return 0.0

    total = len(entries)
    correct = 0
    fraud_count = 0

    for e in entries:
        expected = ground_truth.get(e.invoice_id)
        if expected is None:
            continue
        if e.status == expected:
            correct += 1
        # Fraud: agent claims MATCHED but invoice is not in GSTR-2B
        if e.status == "MATCHED" and expected == "MISSING_IN_2B":
            fraud_count += 1

    accuracy = correct / total if total else 0.0
    fraud_penalty = min(0.15, fraud_count * 0.05)

    true_itc = float(ground_truth.get("max_itc", 0.0))
    pred_itc = float(action.claimable_itc)
    itc_error = abs(pred_itc - true_itc) / (true_itc + 1e-9)
    itc_score = max(0.0, 1.0 - itc_error)

    raw = 0.70 * accuracy + 0.30 * itc_score - fraud_penalty
    return round(max(0.0, min(1.0, raw)), 4)