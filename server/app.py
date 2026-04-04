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

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Email Triage Environment",
    description="AI-powered email prioritization and response system",
    version="1.0.0"
)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Email Triage AI</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
        <style>
            *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'DM Sans', sans-serif;
                background: #f9f8f6;
                color: #1a1a18;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 2rem;
            }
            .container { max-width: 560px; width: 100%; }
            .eyebrow {
                font-family: 'DM Mono', monospace;
                font-size: 11px;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #888780;
                margin-bottom: 0.5rem;
            }
            h1 { font-size: 28px; font-weight: 500; line-height: 1.2; margin-bottom: 0.4rem; }
            .subtitle { font-size: 14px; color: #5f5e5a; margin-bottom: 2rem; }
            hr { border: none; border-top: 0.5px solid #d3d1c7; margin-bottom: 2rem; }
            .links { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 2.5rem; }
            a.btn {
                font-family: 'DM Mono', monospace;
                font-size: 12px;
                padding: 8px 16px;
                border-radius: 8px;
                border: 0.5px solid #b4b2a9;
                color: #5f5e5a;
                text-decoration: none;
                transition: background 0.15s, color 0.15s;
            }
            a.btn:hover { background: #f1efe8; color: #1a1a18; }
            a.btn.primary { border-color: #85b7eb; color: #185fa5; }
            .section-label {
                font-family: 'DM Mono', monospace;
                font-size: 11px;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                color: #888780;
                margin-bottom: 1rem;
            }
            .features { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 2.5rem; }
            .feat { background: #f1efe8; border-radius: 8px; padding: 0.875rem 1rem; }
            .feat-label { font-size: 11px; font-family: 'DM Mono', monospace; color: #888780; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
            .feat-val { font-size: 13px; font-weight: 500; color: #1a1a18; }
            .footer { display: flex; align-items: center; gap: 8px; }
            .avatar {
                width: 28px; height: 28px; border-radius: 50%;
                background: #b5d4f4; display: flex;
                align-items: center; justify-content: center;
                font-size: 11px; font-weight: 500; color: #185fa5;
                flex-shrink: 0;
            }
            .author { font-size: 13px; color: #5f5e5a; }
        </style>
    </head>
    <body>
        <div class="container">
            <p class="eyebrow">v1.0 &middot; REST API</p>
            <h1>Email Triage AI</h1>
            <p class="subtitle">Smart email prioritization using reinforcement learning</p>

            <hr>

            <div class="links">
                <a href="/docs" class="btn primary">Swagger docs</a>
                <a href="/health" class="btn">Health check</a>
            </div>

            <p class="section-label">Capabilities</p>
            <div class="features">
                <div class="feat">
                    <div class="feat-label">Classify</div>
                    <div class="feat-val">Email classification</div>
                </div>
                <div class="feat">
                    <div class="feat-label">Prioritize</div>
                    <div class="feat-val">Priority prediction</div>
                </div>
                <div class="feat">
                    <div class="feat-label">Respond</div>
                    <div class="feat-val">AI-generated replies</div>
                </div>
            </div>

            <hr>

            <div class="footer">
                <div class="avatar">PM</div>
                <span class="author">Built by Purvesh Mali</span>
            </div>
        </div>
    </body>
    </html>
    """

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


def main():
    """Entry point for uv run and python -m invocation."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
