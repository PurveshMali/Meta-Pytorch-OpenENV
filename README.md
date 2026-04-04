---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# 📧 Email Triage Environment

> An OpenEnv-compatible RL environment where an AI agent learns to manage a realistic business inbox — prioritising, categorising, and replying to emails with accuracy and nuance.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://meta-pytorch.org/OpenEnv/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Environment Description

Email triage is a universal, high-value task performed by millions of knowledge workers daily. An inbox is a stream of signals with wildly varying urgency — from a spam prize notification to a regulatory audit notice — and routing them correctly has real business impact.

This environment puts an AI agent in the role of an executive assistant. It receives one email at a time and must:

1. **Classify** the email's priority (urgent → spam) and category
2. **Act** — label it, draft a reply, archive, delete, escalate, or skip
3. **Handle ambiguity** — casual-toned C-suite requests, auto-resolved alerts that look critical, and compliance notices that look routine

The reward function provides dense, partial-credit signals throughout the episode, not just at the end.

---

## 🗂️ Project Structure

```
email_triage_env/
├── models.py              # Pydantic models: Action, Observation, State
├── emails.py              # Email dataset with ground-truth labels for all 3 tasks
├── client.py              # Async HTTP client for the environment
├── inference.py           # ← Baseline inference script (hackathon entrypoint)
├── openenv.yaml           # OpenEnv manifest
├── README.md
├── server/
│   ├── environment.py     # Core RL logic: reset(), step(), state(), graders
│   ├── app.py             # FastAPI server
│   ├── Dockerfile
│   └── requirements.txt
└── tests/
    └── test_environment.py
```

---

## 🎮 Action Space

| Field | Type | Values |
|-------|------|--------|
| `action_type` | enum | `label`, `reply`, `archive`, `delete`, `escalate`, `next` |
| `priority` | enum (optional) | `urgent`, `high`, `normal`, `low`, `spam` |
| `category` | enum (optional) | `customer_support`, `sales_inquiry`, `billing`, `technical`, `internal`, `newsletter`, `spam`, `other` |
| `reply_text` | string (optional) | Up to 500 characters |
| `reasoning` | string (optional) | Agent's explanation |

**Action semantics:**
- `label` — classify only (requires priority + category)
- `reply` — classify AND send a reply (requires priority + category + reply_text)
- `archive` — store without replying (good for low/normal non-urgent)
- `delete` — discard (best for spam; penalised if used on important emails)
- `escalate` — hand off to a human (rewarded for urgent/high, penalised otherwise)
- `next` — skip current email (small penalty to discourage lazy agents)

---

## 👁️ Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `current_email` | object | The email to act on (sender, subject, body, etc.) |
| `inbox_size` | int | Emails remaining |
| `processed_count` | int | Emails already handled |
| `step_count` | int | Current step number |
| `last_action_feedback` | string | Human-readable explanation of last reward |
| `last_action_reward` | float | Reward from last action |
| `cumulative_reward` | float | Total reward so far |
| `done` | bool | Episode finished |
| `task_id` | string | Which task is active |

---

## 📋 Tasks

### Task 1 — Easy: Priority Labelling
- **8 emails** with clear, unambiguous signals
- Goal: assign correct `priority` + `category` to each email
- Spam emails have obvious giveaways (`.xyz` domain, prize language)
- Urgent emails clearly signal urgency in subject line
- **Max steps:** 20
- **Expected score (baseline LLM):** ~0.70–0.85

### Task 2 — Medium: Triage + Reply Drafting
- **6 emails** requiring both classification AND reply drafting
- Some emails should NOT be replied to (penalised if you do)
- Reply quality scored on keyword relevance, length, and tone
- **Max steps:** 18
- **Expected score (baseline LLM):** ~0.45–0.65

### Task 3 — Hard: Ambiguous Full Inbox
- **10 emails** with deliberate difficulty:
  - Casual-tone email from CFO (still `high` priority)
  - Auto-resolved `CRITICAL` alert (actually `normal`)
  - Legal/compliance notice from government regulator (`urgent`)
  - Feature request from churning paying customer (`high`)
  - Unsolicited competitor BD outreach (`low`)
- Agent must avoid surface-level heuristics
- **Max steps:** 30
- **Expected score (baseline LLM):** ~0.30–0.50

---

## 🏆 Reward Function

### Priority scoring (partial credit)
Priority is scored by distance on the urgency scale: `urgent → high → normal → low → spam`

| Distance | Score |
|----------|-------|
| 0 (exact) | 1.0 |
| 1 (adjacent) | 0.5 |
| 2 | 0.1 |
| 3+ | 0.0 |

### Reply quality scoring
- **Keyword presence:** Does the reply address the email's core concern?
- **Length check:** Rewards appropriately concise replies (30–500 chars)
- **Avoidance:** Penalises corporate-speak phrases that damage relationships

### Action rewards
| Action | Condition | Reward |
|--------|-----------|--------|
| `delete` | True spam | +0.8 |
| `delete` | Normal/high priority | -0.5 |
| `delete` | Urgent email | -0.8 |
| `archive` | Low/normal email | +0.4 |
| `archive` | High email | -0.2 |
| `archive` | Urgent email | -0.6 |
| `escalate` | Urgent/high email | +(escalation_weight) |
| `escalate` | Low/spam | -0.1 |
| `next` | Any | -0.05 |

---

## ⚙️ Setup & Usage

### Local (no Docker)

```bash
cd email_triage_env
pip install fastapi uvicorn pydantic httpx openai

# Start the server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal, run tests
pip install pytest
pytest tests/ -v

# Run inference
export HF_TOKEN=your_api_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
python inference.py
```

### Docker

```bash
cd email_triage_env
docker build -f server/Dockerfile -t email-triage-env .
docker run -p 7860:7860 email-triage-env

# Run inference against the container
export HF_TOKEN=your_api_key
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

### API Usage

```python
import httpx

# Reset
resp = httpx.post("http://localhost:7860/reset", json={"task_id": "task1_easy_labelling"})
obs = resp.json()["observation"]

# Step
resp = httpx.post("http://localhost:7860/step", json={
    "action_type": "label",
    "priority": "urgent",
    "category": "customer_support",
    "reasoning": "Production outage from paying customer"
})
result = resp.json()
print(result["reward"])  # e.g. 0.96
```

---

## 📊 Baseline Scores

Scores obtained with `gpt-4o-mini` at temperature 0.2:

| Task | Difficulty | Score |
|------|-----------|-------|
| task1_easy_labelling | Easy | ~0.78 |
| task2_medium_triage_reply | Medium | ~0.54 |
| task3_hard_full_inbox | Hard | ~0.41 |
| **Average** | | **~0.58** |

---

## 🌐 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Execute action |
| `GET` | `/state` | Get internal state |
| `GET` | `/score` | Get current normalised score |
| `GET` | `/tasks` | List available tasks |
| `GET` | `/health` | Health check |

---

## 🧪 Running Tests

```bash
pytest tests/test_environment.py -v
```

Tests cover: priority/category/reply graders, episode lifecycle, edge cases (spam deletion, urgent email deletion penalty, task 3 ambiguity), state endpoint, and max-steps termination.

---

## 📄 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | Your HuggingFace / OpenAI-compatible API key | Yes |
| `API_BASE_URL` | LLM API endpoint | Yes |
| `MODEL_NAME` | Model identifier (e.g. `gpt-4o-mini`) | Yes |
| `ENV_BASE_URL` | Email triage server URL (default: `http://localhost:7860`) | No |

---
