"""Bug Triage Environment - FastAPI Server (OpenEnv spec compliant)"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import ActionType, Component, BugTriageAction, Severity
from server.environment import BugTriageEnvironment

app = FastAPI(title="Bug Triage & Patch Validation Environment", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_env: Optional[BugTriageEnvironment] = None
DEFAULT_TASK = "task1_easy_severity_routing"
DEFAULT_SEED = 42


@app.get("/")
async def root():
    return {"status": "ok", "env": "bug_triage", "endpoints": ["/reset", "/step", "/state", "/health", "/tasks"]}


@app.get("/health")
async def health():
    return {"status": "ok", "env": "bug_triage"}


@app.post("/reset")
async def reset(request: Request):
    """Reset - accepts empty body, null, or {task_id, seed}."""
    global _env
    task_id, seed = DEFAULT_TASK, DEFAULT_SEED
    try:
        body = await request.body()
        if body and body.strip() not in (b"", b"null", b"{}"):
            data = await request.json()
            if isinstance(data, dict):
                task_id = data.get("task_id") or DEFAULT_TASK
                seed = data.get("seed") if data.get("seed") is not None else DEFAULT_SEED
    except Exception:
        pass
    try:
        _env = BugTriageEnvironment(task_id=task_id, seed=seed)
        obs = _env.reset()
        return {"observation": obs.model_dump()}
    except ValueError as e:
        raise HTTPException(400, detail=str(e))


@app.post("/step")
async def step(request: Request):
    """Execute one action."""
    global _env
    if _env is None:
        _env = BugTriageEnvironment(DEFAULT_TASK, DEFAULT_SEED)
        _env.reset()

    fields = {"action_type": "diagnose", "severity": None, "component": None,
              "test_case": None, "root_cause": None, "patch_code": None,
              "patch_explanation": None, "test_results": None,
              "resolution": None, "info_request": None, "reasoning": None}
    try:
        body = await request.body()
        if body and body.strip() not in (b"", b"null"):
            data = await request.json()
            if isinstance(data, dict):
                for k in fields:
                    if k in data:
                        fields[k] = data[k]
    except Exception:
        pass

    try:
        action = BugTriageAction(
            action_type=ActionType(fields["action_type"]),
            severity=Severity(fields["severity"]) if fields["severity"] else None,
            component=Component(fields["component"]) if fields["component"] else None,
            test_case=fields["test_case"],
            root_cause=fields["root_cause"],
            patch_code=fields["patch_code"],
            patch_explanation=fields["patch_explanation"],
            test_results=fields["test_results"],
            resolution=fields["resolution"],
            info_request=fields["info_request"],
            reasoning=fields["reasoning"],
        )
    except ValueError as e:
        raise HTTPException(422, detail=f"Invalid action: {e}")

    obs, reward, done, info = _env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}


@app.get("/state")
async def state():
    if _env is None:
        return {"episode_id": "", "task_id": DEFAULT_TASK, "step_count": 0,
                "max_steps": 45, "backlog": [], "resolved": [],
                "current_bug_state": "new", "cumulative_reward": 0.0, "done": False, "seed": DEFAULT_SEED}
    return _env.state().model_dump()


@app.get("/tasks")
async def list_tasks():
    from bugs import ALL_TASKS
    return {tid: {"description": cfg["description"], "difficulty": cfg["difficulty"],
                  "max_steps": cfg["max_steps"], "num_bugs": len(cfg["bugs"])}
            for tid, cfg in ALL_TASKS.items()}


@app.get("/score")
async def score():
    if _env is None:
        return {"final_score": 0.0, "cumulative_reward": 0.0, "done": False}
    return {"final_score": _env.final_score(),
            "cumulative_reward": _env._cumulative_reward, "done": _env._done}


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
