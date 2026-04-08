from __future__ import annotations
from datetime import date
from decimal import Decimal
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, field_validator, field_serializer


class Invoice(BaseModel):
    model_config = {"extra": "forbid"}

    invoice_id: str = Field(..., description="Unique identifier for this invoice in the episode")
    vendor_gstin: str = Field(..., description="15-character GSTIN of the vendor/supplier")
    invoice_number: str = Field(..., description="Invoice number as printed on the document")
    invoice_date: date = Field(..., description="Date of the invoice (ISO 8601)")
    taxable_value: Decimal = Field(..., description="Taxable value before GST")
    cgst: Decimal = Field(..., description="Central GST amount (0 for inter-state)")
    sgst: Decimal = Field(..., description="State GST amount (0 for inter-state)")
    igst: Decimal = Field(..., description="Integrated GST amount (0 for intra-state)")
    hsn_code: str = Field(..., description="HSN/SAC code for the goods/services")
    vendor_name: str = Field(..., description="Name of the vendor/supplier")
    document_type: str = Field(default="invoice", description="Document type: invoice, credit_note, debit_note, advance_receipt")

    @field_validator("vendor_gstin")
    @classmethod
    def validate_gstin(cls, v: str) -> str:
        if len(v) != 15:
            raise ValueError(f"GSTIN must be 15 characters, got {len(v)}: {v}")
        return v.upper()

    @field_serializer('taxable_value', 'cgst', 'sgst', 'igst')
    def serialize_decimal(self, v: Decimal) -> float:
        return float(v)


class GSTR2BEntry(BaseModel):
    model_config = {"extra": "forbid"}

    supplier_gstin: str = Field(..., description="15-character GSTIN of the supplier")
    invoice_number: str = Field(..., description="Invoice/document number as in GSTR-2B portal")
    invoice_date: date = Field(..., description="Date of the invoice (ISO 8601)")
    taxable_value: Decimal = Field(..., description="Taxable value as reported in GSTR-2B")
    cgst: Decimal = Field(..., description="CGST amount in GSTR-2B (0 for inter-state)")
    sgst: Decimal = Field(..., description="SGST amount in GSTR-2B (0 for inter-state)")
    igst: Decimal = Field(..., description="IGST amount in GSTR-2B (0 for intra-state)")
    itc_available: bool = Field(..., description="Whether ITC is eligible per GSTR-2B")
    document_type: str = Field(default="invoice", description="Document type reported in GSTR-2B")

    @field_validator("supplier_gstin")
    @classmethod
    def validate_gstin(cls, v: str) -> str:
        if len(v) != 15:
            raise ValueError(f"GSTIN must be 15 characters, got {len(v)}: {v}")
        return v.upper()

    @field_serializer('taxable_value', 'cgst', 'sgst', 'igst')
    def serialize_decimal(self, v: Decimal) -> float:
        return float(v)


class ReconciliationEntry(BaseModel):
    model_config = {"extra": "forbid"}
    invoice_id: str = Field(..., description="Must match an invoice_id from the observation")
    status: Literal["MATCHED", "MISMATCH", "MISSING_IN_2B", "EXTRA_IN_2B"] = Field(
        ..., description="Reconciliation status for this invoice"
    )
    correction_note: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of the mismatch (required for MISMATCH entries)"
    )
    mismatch_fields: List[str] = Field(
        default_factory=list,
        description="List of field names that differ (only for MISMATCH status)"
    )


class Action(BaseModel):
    model_config = {"extra": "forbid"}
    reconciliation_result: List[ReconciliationEntry] = Field(
        ..., description="One entry per invoice in the observation"
    )
    claimable_itc: Decimal = Field(
        ..., description="Total ITC claimable = sum(cgst+sgst+igst) for MATCHED invoices only"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Agent's confidence in its reconciliation (0.0–1.0)"
    )

    @field_serializer('claimable_itc')
    def serialize_claimable_itc(self, v: Decimal) -> float:
        return float(v)


class Observation(BaseModel):
    model_config = {"extra": "forbid"}
    task_id: str = Field(..., description="Task identifier for this episode")
    episode_id: str = Field(..., description="Unique UUID for this episode")
    invoices: List[Invoice] = Field(..., description="Purchase invoices to reconcile")
    gstr2b_entries: List[GSTR2BEntry] = Field(..., description="GSTR-2B portal entries for the period")
    tax_period: str = Field(..., description="Financial year (e.g. '2024-25')")
    filing_month: str = Field(default="", description="Filing month if task involves monthly period (YYYY-MM)")
    max_itc_possible: Decimal = Field(..., description="Maximum possible ITC if all eligible invoices are MATCHED")
    step_number: int = Field(..., description="Current step number (0 = fresh reset)")
    instructions: str = Field(..., description="Task-specific instructions for the agent")

    @field_serializer('max_itc_possible')
    def serialize_max_itc(self, v: Decimal) -> float:
        return float(v)


class RewardInfo(BaseModel):
    model_config = {"extra": "forbid"}
    correct_matches: int = Field(..., description="Number of invoices correctly classified")
    total_invoices: int = Field(..., description="Total invoices in the episode")
    coverage: float = Field(default=1.0, description="Fraction of invoices included in agent output")
    episode_id: str = Field(..., description="Episode UUID")
    itc_error: float = Field(default=0.0, description="Absolute relative error in ITC computation")
    fraud_count: int = Field(default=0, description="Number of fraudulent ITC claims detected")
    task_score: float = Field(default=0.0, description="Final task score [0.0, 1.0]")


class Reward(BaseModel):
    model_config = {"extra": "forbid"}
    total: float = Field(..., description="Overall task score [0.0, 1.0]")
    match_score: float = Field(..., description="Fraction of invoices correctly classified")
    itc_score: float = Field(..., description="ITC computation accuracy [0.0, 1.0]")
    false_positive_penalty: float = Field(default=0.0, description="Penalty for fraudulent ITC claims")
    coverage_score: float = Field(default=1.0, description="Fraction of invoices included in agent output")
    penalty_day_penalty: float = Field(default=0.0, description="Filing timeliness score")
    done: bool = Field(..., description="Whether the episode is finished")
    info: RewardInfo = Field(..., description="Detailed breakdown of the score")


class TaskInfo(BaseModel):
    model_config = {"extra": "forbid"}
    task_id: str
    description: str
    difficulty: str
    num_invoices: int
    invoice_range: str


class StateResponse(BaseModel):
    model_config = {"extra": "forbid"}
    task_id: str
    episode_id: str
    step_number: int
    done: bool
    has_active_episode: bool