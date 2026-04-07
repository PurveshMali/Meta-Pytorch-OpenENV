"""
Email Triage Environment - FastAPI Server
Exposes step(), reset(), state() as HTTP endpoints per OpenEnv spec.

KEY FIX: /reset accepts POST with empty body, no body, or partial body.
The hackathon checker calls POST /reset with null/empty body.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
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

_env: Optional[EmailTriageEnvironment] = None

DEFAULT_TASK_ID = "task1_easy_labelling"
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root(request: Request):
    if "text/html" in request.headers.get("accept", ""):
        return {"message": "Welcome to Email Triage Environment. Use /docs for API documentation."}
    return {"status": "ok", "env": "email_triage", "endpoints": ["/reset", "/step", "/state", "/health", "/tasks"]}

from fastapi.responses import HTMLResponse

@app.get("/web", response_class=HTMLResponse)
@app.get("/web/", response_class=HTMLResponse)
async def web_interface():
    return """
    <html>
        <head>
            <title>Email Triage Environment</title>
            <style>
                body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; background: #f0f2f5; color: #1c1e21; }
                .card { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; }
                h1 { color: #1a73e8; }
                .status { margin-top: 1rem; padding: 0.5rem 1rem; border-radius: 20px; background: #e6f4ea; color: #1e7e34; font-weight: bold; }
                ul { text-align: left; margin-top: 1rem; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>📧 Email Triage Env</h1>
                <p>Status: <span class="status">ONLINE</span></p>
                <p>The environment is running and ready for agents.</p>
                <ul>
                    <li><b>API Endpoints:</b></li>
                    <li>POST <code>/reset</code></li>
                    <li>POST <code>/step</code></li>
                    <li>GET <code>/state</code></li>
                </ul>
                <p><small>Powered by OpenEnv</small></p>
            </div>
        </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "ok", "env": "email_triage"}


@app.post("/reset", response_model=Dict[str, Any])
async def reset(request: Request):
    """
    Reset the environment.
    Accepts: empty body, null body, or {"task_id": "...", "seed": 42}
    All fields optional - defaults to task1_easy_labelling, seed=42.
    """
    global _env

    task_id = DEFAULT_TASK_ID
    seed = DEFAULT_SEED

    try:
        body_bytes = await request.body()
        if body_bytes and body_bytes.strip() not in (b"", b"null", b"{}"):
            body = await request.json()
            if isinstance(body, dict):
                task_id = body.get("task_id") or DEFAULT_TASK_ID
                seed_val = body.get("seed")
                seed = seed_val if seed_val is not None else DEFAULT_SEED
    except Exception:
        pass

    try:
        _env = EmailTriageEnvironment(task_id=task_id, seed=seed)
        obs = _env.reset()
        return {"observation": obs.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(request: Request):
    """
    Execute one action. Auto-resets if env not initialised.
    """
    global _env

    if _env is None:
        _env = EmailTriageEnvironment(task_id=DEFAULT_TASK_ID, seed=DEFAULT_SEED)
        _env.reset()

    action_type_str = "next"
    priority_str = None
    category_str = None
    reply_text = None
    reasoning = None

    try:
        body_bytes = await request.body()
        if body_bytes and body_bytes.strip() not in (b"", b"null"):
            body = await request.json()
            if isinstance(body, dict):
                action_type_str = body.get("action_type") or "next"
                priority_str = body.get("priority")
                category_str = body.get("category")
                reply_text = body.get("reply_text")
                reasoning = body.get("reasoning")
    except Exception:
        pass

    try:
        action = EmailTriageAction(
            action_type=ActionType(action_type_str),
            priority=Priority(priority_str) if priority_str else None,
            category=Category(category_str) if category_str else None,
            reply_text=reply_text,
            reasoning=reasoning,
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
        return {
            "episode_id": "",
            "task_id": DEFAULT_TASK_ID,
            "step_count": 0,
            "max_steps": 20,
            "inbox": [],
            "processed": [],
            "cumulative_reward": 0.0,
            "done": False,
            "seed": DEFAULT_SEED,
        }
    return _env.state().model_dump()


@app.get("/tasks")
async def list_tasks():
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
    global _env
    if _env is None:
        return {"final_score": 0.0, "cumulative_reward": 0.0, "done": False}
    return {
        "final_score": _env.final_score(),
        "cumulative_reward": _env._cumulative_reward,
        "done": _env._done,
    }


def main():
    """Entry point for uv run and python -m invocation."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()