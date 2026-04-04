"""
Email Triage Environment - FastAPI Server
Exposes step(), reset(), state() as HTTP endpoints per OpenEnv spec.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import (
    ActionType,
    Category,
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
    Priority,
)
from server.environment import EmailTriageEnvironment


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Email Triage Environment",
    description=(
        "OpenEnv-compatible email triage RL environment. "
        "Agents must prioritise, categorise and reply to realistic inbox emails."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-session; for multi-session use session IDs)
_env: Optional[EmailTriageEnvironment] = None


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: str = "task1_easy_labelling"
    seed: int = 42


class StepRequest(BaseModel):
    action_type: str
    priority: Optional[str] = None
    category: Optional[str] = None
    reply_text: Optional[str] = None
    reasoning: Optional[str] = None


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "env": "email_triage"}


@app.post("/reset", response_model=Dict[str, Any])
async def reset(request: ResetRequest):
    global _env
    try:
        _env = EmailTriageEnvironment(task_id=request.task_id, seed=request.seed)
        obs = _env.reset()
        return {"observation": obs.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    # Build action
    try:
        action = EmailTriageAction(
            action_type=ActionType(request.action_type),
            priority=Priority(request.priority) if request.priority else None,
            category=Category(request.category) if request.category else None,
            reply_text=request.reply_text,
            reasoning=request.reasoning,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    obs, reward, done, info = _env.step(action)
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=Dict[str, Any])
async def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    s = _env.state()
    return s.model_dump()


@app.get("/tasks")
async def list_tasks():
    """List available tasks."""
    from emails import ALL_TASKS
    return {
        task_id: {
            "description": cfg["description"],
            "difficulty": cfg["difficulty"],
            "max_steps": cfg["max_steps"],
            "num_emails": len(cfg["emails"]),
        }
        for task_id, cfg in ALL_TASKS.items()
    }


@app.get("/score")
async def get_score():
    """Return the current episode's final normalised score."""
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return {
        "final_score": _env.final_score(),
        "cumulative_reward": _env._cumulative_reward,
        "done": _env._done,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
