from __future__ import annotations
from typing import Any, Dict, List
import json
from decimal import Decimal
from datetime import date, datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .env import GSTReconciliationEnv
from .models import Action, Observation, StateResponse, TaskInfo
from .graders import grade as grade_action

app = FastAPI(
    title="GST Reconciliation OpenEnv",
    version="1.0.0",
    description="GST Reconciliation environment for OpenEnv",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = GSTReconciliationEnv()


class ResetRequest(BaseModel):
    task_id: str


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)


TASKS: List[TaskInfo] = [
    TaskInfo(
        task_id="task1_easy",
        description="10 perfectly matched invoices — all MATCHED, full ITC claimable.",
        difficulty="easy",
        num_invoices=10,
        invoice_range="1–10",
    ),
    TaskInfo(
        task_id="task2_medium",
        description="50 invoices with 8 mismatches (amount, date, GSTIN, or missing).",
        difficulty="medium",
        num_invoices=50,
        invoice_range="1–50",
    ),
    TaskInfo(
        task_id="task3_hard",
        description="200 invoices with all mismatch types plus a random filing penalty.",
        difficulty="hard",
        num_invoices=200,
        invoice_range="1–200",
    ),
    TaskInfo(
        task_id="task4_credit_notes",
        description="75 invoices with ITC-unavailable mismatches and credit-note scenarios.",
        difficulty="medium",
        num_invoices=75,
        invoice_range="1–75",
    ),
    TaskInfo(
        task_id="task5_stress",
        description="High-volume stress test: 500 invoices covering all mismatch types.",
        difficulty="hard",
        num_invoices=500,
        invoice_range="1–500",
    ),
    TaskInfo(
        task_id="task6_mixed_docs",
        description="150 invoices with mixed document types and all mismatch categories.",
        difficulty="hard",
        num_invoices=150,
        invoice_range="1–150",
    ),
]


@app.get("/health", tags=["System"])
async def health() -> Dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks", tags=["OpenEnv"])
async def list_tasks() -> List[Dict[str, Any]]:
    return [t.model_dump() for t in TASKS]


@app.get("/state", tags=["OpenEnv"])
async def get_state() -> Dict[str, Any]:
    return env.state().model_dump()


@app.post("/reset", tags=["OpenEnv"])
async def reset(request: ResetRequest) -> Dict[str, Any]:
    try:
        obs: Observation = env.reset(request.task_id)
        data = obs.model_dump(mode="python")
        json_str = json.dumps(data, cls=DecimalEncoder)
        return json.loads(json_str)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step", tags=["OpenEnv"])
async def step(action: Action) -> Dict[str, Any]:
    try:
        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump(mode="python")
        reward_dict = reward.model_dump(mode="python")
        return {
            "observation": json.loads(json.dumps(obs_dict, cls=DecimalEncoder)),
            "reward": json.loads(json.dumps(reward_dict, cls=DecimalEncoder)),
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.post("/grader", tags=["Evaluation"])
async def grader_endpoint(action: Action) -> Dict[str, Any]:
    if not env.current_task_id:
        raise HTTPException(
            status_code=400, detail="No active episode. Call /reset first."
        )
    try:
        score = grade_action(env.current_task_id, action, env.ground_truth)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grading error: {e}")

    true_itc = float(env.ground_truth.get("max_itc", 0.0))
    pred_itc = float(action.claimable_itc)
    itc_error = abs(pred_itc - true_itc) / (true_itc + 1e-9)
    total = len(action.reconciliation_result)
    correct = sum(
        1 for e in action.reconciliation_result
        if env.ground_truth.get(e.invoice_id) == e.status
    )

    return {
        "task_id": env.current_task_id,
        "score": score,
        "breakdown": {
            "accuracy": round(correct / total, 4) if total else 0.0,
            "itc_error": round(itc_error, 4),
            "itc_score": round(max(0.0, 1.0 - itc_error), 4),
            "confidence": float(getattr(action, "confidence", 0.5)),
            "correct_matches": correct,
            "total_submitted": total,
        },
    }


@app.get("/baseline", tags=["Evaluation"])
async def baseline() -> Dict[str, Any]:
    from .baseline import run_baseline
    return run_baseline()