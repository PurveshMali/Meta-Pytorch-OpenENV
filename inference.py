"""
Email Triage Environment - Baseline Inference Script
Runs an LLM agent against all 3 tasks using the OpenAI client.
Emits [START], [STEP], [END] structured logs per hackathon spec.

Required environment variables:
    API_BASE_URL  - LLM API endpoint (OpenAI-compatible)
    MODEL_NAME    - Model identifier
    HF_TOKEN      - Hugging Face / API key
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
# Config — read from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY: str = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TEMPERATURE: float = 0.2
MAX_TOKENS: int = 512
MAX_STEPS: int = 25
SUCCESS_SCORE_THRESHOLD: float = 0.5

TASKS = [
    "task1_easy_labelling",
    "task2_medium_triage_reply",
    "task3_hard_full_inbox",
]

# ---------------------------------------------------------------------------
# Structured log helpers (exact hackathon format)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(
        json.dumps({
            "type": "START",
            "task": task,
            "env": env,
            "model": model,
        }),
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    print(
        json.dumps({
            "type": "STEP",
            "step": step,
            "action": action,
            "reward": round(reward, 4),
            "done": done,
            "error": error,
        }),
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    print(
        json.dumps({
            "type": "END",
            "success": success,
            "steps": steps,
            "score": round(score, 4),
            "rewards": [round(r, 4) for r in rewards],
        }),
        flush=True,
    )


# ---------------------------------------------------------------------------
# System and user prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert email triage assistant.
You are given one email at a time and must decide how to handle it.

You MUST respond with ONLY a valid JSON object — no prose, no markdown fences.

The JSON must have:
  "action_type": one of ["label", "reply", "archive", "delete", "escalate", "next"]
  "priority": one of ["urgent", "high", "normal", "low", "spam"]  (required for label/reply)
  "category": one of ["customer_support", "sales_inquiry", "billing", "technical",
                       "internal", "newsletter", "spam", "other"]  (required for label/reply)
  "reply_text": string (required for reply action, max 500 chars)
  "reasoning": string (brief explanation of your decision)

Guidelines:
- Use "label" to classify an email without replying
- Use "reply" to classify AND send a response (include both priority/category AND reply_text)
- Use "delete" ONLY for clear spam/phishing
- Use "archive" for low-value emails that don't need a response
- Use "escalate" for emails requiring urgent human attention you cannot handle
- Use "next" to skip (avoid this — you lose points)

Urgency signals: "URGENT", "ASAP", "NOW", system outages, legal deadlines, C-suite senders.
Spam signals: suspicious domains, prize claims, unsolicited offers, grammar errors.
"""


def build_user_prompt(
    step: int,
    obs: Dict[str, Any],
    history: List[str],
) -> str:
    email = obs.get("current_email")
    inbox_size = obs.get("inbox_size", 0)
    processed = obs.get("processed_count", 0)
    last_feedback = obs.get("last_action_feedback", "")
    cumulative = obs.get("cumulative_reward", 0.0)

    history_str = "\n".join(history[-5:]) if history else "None"

    if email is None:
        return (
            f"Step {step}: Inbox is now empty. "
            f"Total processed: {processed}. Score so far: {cumulative:.2f}.\n"
            'Respond with: {"action_type": "next", "reasoning": "Inbox empty"}'
        )

    return f"""Step {step} | Emails remaining: {inbox_size} | Processed: {processed} | Score: {cumulative:.3f}

--- CURRENT EMAIL ---
From: {email.get('sender', '?')}
Domain: {email.get('sender_domain', '?')}
Subject: {email.get('subject', '?')}
Time: {email.get('timestamp', '?')}
Attachment: {email.get('has_attachment', False)}
Thread length: {email.get('thread_length', 1)}

Body:
{email.get('body', '')}

--- LAST FEEDBACK ---
{last_feedback or 'N/A'}

--- RECENT HISTORY ---
{history_str}

Respond with ONLY a JSON object. Think carefully about urgency, sender authority, and context."""


# ---------------------------------------------------------------------------
# Agent — calls the LLM and parses its response
# ---------------------------------------------------------------------------


def get_model_action(
    client: OpenAI,
    step: int,
    obs: Dict[str, Any],
    history: List[str],
) -> Dict[str, Any]:
    """Call the LLM and parse its JSON action."""
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
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"[DEBUG] JSON parse failed, using fallback action", flush=True)
        return {"action_type": "label", "priority": "normal", "category": "other", "reasoning": "parse_error"}
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "next", "reasoning": "api_error"}


# ---------------------------------------------------------------------------
# Environment interaction via HTTP
# ---------------------------------------------------------------------------


async def call_reset(http: httpx.AsyncClient, task_id: str, seed: int = 42) -> Dict[str, Any]:
    resp = await http.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": seed})
    resp.raise_for_status()
    return resp.json()["observation"]


async def call_step(http: httpx.AsyncClient, action: Dict[str, Any]) -> Dict[str, Any]:
    resp = await http.post(f"{ENV_BASE_URL}/step", json=action)
    resp.raise_for_status()
    return resp.json()


async def call_score(http: httpx.AsyncClient) -> float:
    resp = await http.get(f"{ENV_BASE_URL}/score")
    resp.raise_for_status()
    return resp.json()["final_score"]


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------


async def run_task(
    client: OpenAI,
    http: httpx.AsyncClient,
    task_id: str,
) -> Dict[str, Any]:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env="email_triage_env", model=MODEL_NAME)

    try:
        obs = await call_reset(http, task_id=task_id, seed=42)

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            # Agent decides action
            action_dict = get_model_action(client, step, obs, history)

            # Clean up: ensure required fields exist
            action_payload = {
                "action_type": action_dict.get("action_type", "next"),
                "priority": action_dict.get("priority"),
                "category": action_dict.get("category"),
                "reply_text": action_dict.get("reply_text"),
                "reasoning": action_dict.get("reasoning", ""),
            }

            # Execute step
            result = await call_step(http, action_payload)
            step_obs = result["observation"]
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            error = None

            rewards.append(reward)
            steps_taken = step

            # Log
            action_str = json.dumps(action_payload)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_payload['action_type']} → reward {reward:+.3f} | "
                f"{step_obs.get('last_action_feedback', '')[:80]}"
            )

            obs = step_obs

            if done:
                break

        # Get final normalised score
        score = await call_score(http)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN or OPENAI_API_KEY environment variable not set.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Wait for env server to be ready
    async with httpx.AsyncClient(timeout=60.0) as http:
        for attempt in range(30):
            try:
                resp = await http.get(f"{ENV_BASE_URL}/health")
                if resp.status_code == 200:
                    print("[DEBUG] Environment server is ready.", flush=True)
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            print("[ERROR] Environment server did not start in time.", flush=True)
            sys.exit(1)

        # Run all tasks sequentially
        all_results = []
        for task_id in TASKS:
            result = await run_task(client, http, task_id)
            all_results.append(result)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("BASELINE RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in all_results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        print(
            f"{status} | {r['task_id']:40s} | score={r['score']:.4f} | steps={r['steps']}",
            flush=True,
        )
    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\nAverage score across all tasks: {avg:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
