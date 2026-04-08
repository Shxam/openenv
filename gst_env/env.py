from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from .data_generator import (
    generate_task1_data,
    generate_task2_data,
    generate_task3_data,
    generate_task4_data,
    generate_task5_data,
    generate_task6_data,
)
from .models import (
    Action,
    GSTR2BEntry,
    Invoice,
    Observation,
    Reward,
    RewardInfo,
    StateResponse,
)
from .graders import grade as grade_action

TASK_GENERATORS = {
    "task1_easy": generate_task1_data,
    "task2_medium": generate_task2_data,
    "task3_hard": generate_task3_data,
    "task4_credit_notes": generate_task4_data,
    "task5_stress": generate_task5_data,
    "task6_mixed_docs": generate_task6_data,
}

# Task-specific instructions to help agents understand the scenario
TASK_INSTRUCTIONS: Dict[str, str] = {
    "task1_easy": (
        "Reconcile 10 purchase invoices against GSTR-2B for FY 2024-25. "
        "All invoices are expected to be perfectly matched. "
        "Classify each as MATCHED, MISMATCH, MISSING_IN_2B, or EXTRA_IN_2B. "
        "Compute claimable ITC = sum(cgst + sgst + igst) for MATCHED invoices only. "
        "Threshold for amount mismatch: > Rs 1 difference."
    ),
    "task2_medium": (
        "Reconcile 50 purchase invoices against GSTR-2B. "
        "~8 invoices have mismatches: amount differences >15%, date shifts >5 days, "
        "GSTIN errors, or missing entries. "
        "Classify each as MATCHED, MISMATCH, MISSING_IN_2B, or EXTRA_IN_2B. "
        "Report mismatch_fields for every MISMATCH entry. "
        "Threshold for amount mismatch: > Rs 1 difference. "
        "Compute claimable ITC = sum(cgst + sgst + igst) for MATCHED invoices only."
    ),
    "task3_hard": (
        "Reconcile 200 purchase invoices against GSTR-2B. WARNING: Adversarial mismatches included. "
        "Watch for: (1) NEAR-MISS amounts differing by only Rs 1.01–5.00 — these are MISMATCH. "
        "(2) OCR-style GSTIN errors where 0/O, 1/I, 8/B, 5/S characters are swapped — MISMATCH. "
        "(3) Month-boundary date shifts (e.g. 31-Mar → 1-Apr) — MISMATCH. "
        "(4) Multi-field mismatches where BOTH amount AND date differ. "
        "(5) Duplicate GSTR-2B entries (EXTRA_IN_2B). "
        "(6) Invoices with itc_available=False — MISMATCH. "
        "Threshold for amount mismatch: > Rs 1 difference. "
        "claimable_itc = sum(cgst + sgst + igst) for MATCHED invoices only. "
        "Do NOT claim ITC on MISSING_IN_2B or MISMATCH invoices — this is a GST compliance violation."
    ),
    "task4_credit_notes": (
        "Reconcile 75 purchase invoices against GSTR-2B, focusing on ITC eligibility. "
        "Special scenarios: credit notes (itc_available=False, document_type=credit_note), "
        "advance receipts, and blocked ITC under Section 17(5). "
        "Credit notes in GSTR-2B mean ITC has been reversed — classify as MISMATCH. "
        "Advance receipts with itc_available=False — classify as MISMATCH. "
        "Report itc_available in mismatch_fields when applicable. "
        "Threshold for amount mismatch: > Rs 1 difference. "
        "claimable_itc = sum(cgst + sgst + igst) for MATCHED invoices only."
    ),
    "task5_stress": (
        "High-volume reconciliation: 500 purchase invoices against GSTR-2B. "
        "All mismatch types present including adversarial ones. "
        "Watch for: near-miss amounts (Rs 1.01–4.99 differences), OCR GSTIN errors, "
        "month-boundary date shifts, multi-field mismatches. "
        "You MUST include ALL 500 invoice_ids in your reconciliation_result. "
        "Missing invoices in your output will reduce your coverage score. "
        "Threshold for amount mismatch: > Rs 1 difference. "
        "claimable_itc = sum(cgst + sgst + igst) for MATCHED invoices only."
    ),
    "task6_mixed_docs": (
        "Reconcile 150 documents of MIXED types against GSTR-2B. "
        "Document types present: invoice, credit_note, debit_note, advance_receipt. "
        "Rules: "
        "(1) credit_note in GSTR-2B → MISMATCH (ITC reversal). "
        "(2) debit_note with inflated amount → MISMATCH. "
        "(3) advance_receipt with itc_available=False → MISMATCH. "
        "(4) Near-miss amounts (Rs 1.01–4.99 diff) → MISMATCH. "
        "(5) OCR-style GSTIN errors → MISMATCH. "
        "Check document_type field in GSTR-2B entries carefully. "
        "Threshold for amount mismatch: > Rs 1 difference. "
        "claimable_itc = sum(cgst + sgst + igst) for MATCHED invoices ONLY."
    ),
}


class GSTReconciliationEnv:
    """
    OpenEnv-compliant environment for Indian GST invoice reconciliation.

    The agent receives an Observation containing purchase invoices and GSTR-2B portal
    entries, then submits an Action classifying each invoice and computing claimable ITC.
    """

    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._episode_id: Optional[str] = None
        self._step_number: int = 0
        self._done: bool = False
        self._obs: Optional[Observation] = None
        self._ground_truth: Dict[str, Any] = {}
        self._raw_data: Dict[str, Any] = {}

    @property
    def current_task_id(self) -> Optional[str]:
        return self._task_id

    @property
    def ground_truth(self) -> Dict[str, Any]:
        return self._ground_truth

    def reset(self, task_id: str) -> Observation:
        if task_id not in TASK_GENERATORS:
            raise ValueError(
                f"Unknown task_id {task_id!r}. Valid: {list(TASK_GENERATORS.keys())}"
            )

        self._task_id = task_id
        self._episode_id = str(uuid.uuid4())
        self._step_number = 0
        self._done = False

        data = TASK_GENERATORS[task_id](seed=42)
        self._raw_data = data
        self._ground_truth = data["ground_truth"].copy()
        self._ground_truth["max_itc"] = data["max_itc"]
        if "penalty_days" in data:
            self._ground_truth["penalty_days"] = data["penalty_days"]

        tax_period = "2024-25"
        instructions = TASK_INSTRUCTIONS.get(task_id, TASK_INSTRUCTIONS["task1_easy"])

        invoices = [Invoice(**inv) for inv in data["invoices"]]
        gstr2b = [GSTR2BEntry(**e) for e in data["gstr2b_entries"]]

        self._obs = Observation(
            task_id=task_id,
            episode_id=self._episode_id,
            invoices=invoices,
            gstr2b_entries=gstr2b,
            tax_period=tax_period,
            filing_month="",
            max_itc_possible=Decimal(str(data["max_itc"])),
            step_number=self._step_number,
            instructions=instructions,
        )
        return self._obs

    def step(
        self, action: Action
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._obs is None or self._task_id is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._step_number += 1
        self._done = True

        score = grade_action(self._task_id, action, self._ground_truth)

        total_invoices = len(self._obs.invoices)
        submitted_ids = {e.invoice_id for e in action.reconciliation_result}
        gt_ids = {k for k in self._ground_truth if k not in ("max_itc", "penalty_days")}

        correct_matches = sum(
            1
            for e in action.reconciliation_result
            if self._ground_truth.get(e.invoice_id) == e.status
        )

        coverage = len(submitted_ids & gt_ids) / max(len(gt_ids), 1)

        # Compute fraud count: MATCHED claimed on MISSING_IN_2B invoices
        fraud_count = sum(
            1 for e in action.reconciliation_result
            if e.status == "MATCHED"
            and self._ground_truth.get(e.invoice_id) == "MISSING_IN_2B"
        )

        true_itc = float(self._ground_truth.get("max_itc", 0.0))
        pred_itc = float(action.claimable_itc)
        itc_error = abs(pred_itc - true_itc) / (true_itc + 1e-9)
        itc_score = max(0.0, 1.0 - itc_error)
        match_score = correct_matches / total_invoices if total_invoices else 0.0

        penalty_days = int(self._ground_truth.get("penalty_days", 0))
        penalty_day_penalty = max(0.0, 1.0 - penalty_days / 30.0)

        # false_positive_penalty: 0.05 per fraudulent ITC claim, capped at 0.30
        false_positive_penalty = min(0.30, fraud_count * 0.05)

        reward_info = RewardInfo(
            correct_matches=correct_matches,
            total_invoices=total_invoices,
            coverage=round(coverage, 4),
            episode_id=self._episode_id or "",
            itc_error=round(itc_error, 4),
            fraud_count=fraud_count,
            task_score=round(score, 4),
        )

        reward = Reward(
            total=round(score, 4),
            match_score=round(match_score, 4),
            itc_score=round(itc_score, 4),
            false_positive_penalty=round(false_positive_penalty, 4),
            coverage_score=round(coverage, 4),
            penalty_day_penalty=round(penalty_day_penalty, 4),
            done=self._done,
            info=reward_info,
        )

        info_dict = reward_info.model_dump()
        return self._obs, reward, self._done, info_dict

    def state(self) -> StateResponse:
        return StateResponse(
            task_id=self._task_id or "",
            episode_id=self._episode_id or "",
            step_number=self._step_number,
            done=self._done,
            has_active_episode=self._obs is not None and not self._done,
        )