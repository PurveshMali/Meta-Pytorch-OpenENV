"""
Bug Triage & Patch Validation - Baseline Inference Script
Runs an LLM agent against all 3 tasks using the OpenAI client.
Emits [START], [STEP], [END] structured logs per hackathon spec.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TEMPERATURE: float = 0.2
MAX_TOKENS: int = 1024
MAX_STEPS: int = 45
SUCCESS_SCORE_THRESHOLD: float = 0.6

TASKS = [
    "task1_easy_severity_routing",
    "task2_medium_full_debug_cycle",
    "task3_hard_mixed_backlog",
]

# ---------------------------------------------------------------------------
# Structured log helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} reward={reward:.4f} done={done} action={action}", flush=True)

def log_end(task: str, success: bool, steps: int, score: float) -> None:
    print(f"[END] task={task} score={score:.4f} steps={steps} success={success}", flush=True)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior software engineer responsible for bug triage and patch validation.
You must work through a backlog of bugs using a specific workflow.

Workflow:
1. REPRODUCE: Write a test case to confirm the bug.
2. DIAGNOSE: Assess severity, component, and root cause.
3. PATCH: Write the actual code fix.
4. VALIDATE: Run tests to verify your fix works.
5. CLOSE: Once fixed, close the bug to move to the next.

Other actions:
- ESCALATE: For critical security bugs or GDPR issues.
- REQUEST_INFO: If a bug report is missing reproduction steps.

IMPORTANT:
- For 'diagnose', you MUST provide 'severity' (critical/high/medium/low) and 'component'.
- Some bugs are DUPLICATES. If a bug seems identical to a previous one, mention this in 'reasoning' and use 'close' or 'diagnose' with duplicate details.
- Respond with ONLY valid JSON.

JSON Schema:
{
  "action_type": "reproduce" | "diagnose" | "patch" | "validate" | "escalate" | "close" | "request_info",
  "severity": "critical"|"high"|"medium"|"low"|"wontfix",
  "component": "auth"|"database"|"api"|"frontend"|"payments"|"security"|"infra"|"other",
  "test_case": "string",
  "root_cause": "string",
  "patch_code": "string",
  "patch_explanation": "string",
  "test_results": "string",
  "resolution": "string",
  "reasoning": "string"
}
"""

def build_user_prompt(step: int, obs: Dict[str, Any], history: List[str]) -> str:
    bug = obs.get("current_bug")
    state = obs.get("bug_state", "new")
    backlog_size = obs.get("backlog_size", 0)
    cumulative = obs.get("cumulative_reward", 0.0)
    last_feedback = obs.get("last_action_feedback", "")
    
    history_str = "\n".join(history[-5:]) if history else "None"

    if bug is None:
        return f"Step {step}: Backlog is empty. Cumulative score: {cumulative:.2f}. Use 'close' or 'next'."

    return f"""Step {step} | Bug State: {state} | Backlog: {backlog_size} | Score: {cumulative:.3f}

--- CURRENT BUG: {bug.get('bug_id')} ---
Title: {bug.get('title')}
Reported by: {bug.get('reported_by')} ({bug.get('reporter_type')})
Environment: {bug.get('environment')}
Affected Users: {bug.get('num_affected_users')}

Description:
{bug.get('description')}

Stack Trace:
{bug.get('stack_trace') or 'N/A'}

Steps to Reproduce:
{bug.get('steps_to_reproduce') or 'N/A'}

--- OBSERVED FEEDBACK ---
Reproduction: {obs.get('reproduce_result') or 'N/A'}
Diagnosis: {obs.get('diagnosis_feedback') or 'N/A'}
Patch: {obs.get('patch_feedback') or 'N/A'}
Validation: {obs.get('validation_feedback') or 'N/A'}
Last Action Result: {last_feedback}

--- RECENT HISTORY ---
{history_str}

Respond with ONLY the JSON action object."""

# ---------------------------------------------------------------------------
# Agent Logic
# ---------------------------------------------------------------------------

def get_model_action(client: OpenAI, step: int, obs: Dict[str, Any], history: List[str]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(step, obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"): text = text[4:]
            text = text.strip()
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] Model error: {exc}", flush=True)
        return {"action_type": "request_info", "reasoning": "api_error_fallback"}

async def run_task(client: OpenAI, http: httpx.AsyncClient, task_id: str) -> Dict[str, Any]:
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    
    log_start(task=task_id, env="bug_triage_env", model=MODEL_NAME)

    try:
        resp = await http.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": 42})
        obs = resp.json()["observation"]

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False): break

            action_payload = get_model_action(client, step, obs, history)
            
            resp = await http.post(f"{ENV_BASE_URL}/step", json=action_payload)
            result = resp.json()
            
            step_obs = result["observation"]
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            
            steps_taken = step
            log_step(step=step, action=action_payload.get("action_type", "unknown"), reward=reward, done=done, error=None)

            history.append(f"Step {step}: {action_payload.get('action_type')} -> {reward:+.2f}")
            obs = step_obs
            if done: break

        resp = await http.get(f"{ENV_BASE_URL}/score")
        score = resp.json()["final_score"]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True)
        score, success = 0.0, False

    log_end(task=task_id, success=success, steps=steps_taken, score=score)
    return {"task_id": task_id, "score": score, "success": success}

async def main() -> None:
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN not set.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    async with httpx.AsyncClient(timeout=60.0) as http:
        for _ in range(30):
            try:
                if (await http.get(f"{ENV_BASE_URL}/health")).status_code == 200: break
            except: pass
            await asyncio.sleep(2)

        for task_id in TASKS:
            await run_task(client, http, task_id)

if __name__ == "__main__":
    asyncio.run(main())
