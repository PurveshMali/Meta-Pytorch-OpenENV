---
title: Bug Triage Env
emoji: 🤖
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
tags:
  - openenv
---

# Bug Triage & Patch Validation (Meta-Pytorch-OpenENV)

A high-fidelity Reinforcement Learning environment designed for evaluating AI agents on complete engineering workflows. Unlike simple classification tasks, this environment requires agents to navigate a multi-step state machine for each bug report.

## The Challenge

Modern software engineering isn't just about writing code; it's about diagnosing issues, reproducing them reliably, and verifying fixes. In this environment, an agent acts as a Senior Engineer managing a backlog of complex bugs.

### Core Workflow (The 5-Step State Machine)
Agents must move each bug through the following life-cycle:
1. **Reproduce**: Create a valid test case to confirm the issue.
2. **Diagnose**: Identify the root cause, assess severity (Critical/High/Medium/Low), and route to the correct component (Auth, DB, API, etc.).
3. **Patch**: Author a high-quality code fix and explain the rationale.
4. **Validate**: Run tests to confirm the fix works and doesn't introduce regressions.
5. **Close**: Finalize the bug report and move to the next item.

## Truly Hard Engineering Tasks

This submission includes several "frontier-grade" challenges tailored for the latest LLMs:
- **BUG-302 (Duplicate Detection)**: Agent must recognize it is a duplicate of BUG-301 across the backlog, avoiding redundant work.
- **BUG-304 & BUG-306 (Security & GDPR)**: Requires identifying escalation-worthy vulnerabilities (Auth bypass, data leak) and using the `ESCALATE` action instead of standard patching.
- **BUG-305 (Unicode Collation)**: Requires specific technical knowledge of `unaccent/NFC` normalization for full credit.
- **BUG-107 & BUG-201 (Race Conditions & Leaks)**: Challenges agents to diagnose complex concurrency and infrastructure issues from minimal logs.

##         Technology Stack

- **Backend**: FastAPI (Python 3.10+)
- **Models**: Pydantic v2
- **Validation**: OpenEnv CLI

##      Getting Started

### 1. Start the Environment Server
```powershell
python -m server.app
```
The server will be reachable at `http://localhost:7860`.

### 2. Verify Your Submission (Windows)
We provide a native PowerShell script to audit your submission for compliance with the OpenEnv specifications.
```powershell
powershell -ExecutionPolicy Bypass -File "./validate-submission.ps1"
```

### 3. Run Baseline Inference
```powershell
# Set your environment variables
$env:HF_TOKEN = "your_token"
$env:MODEL_NAME = "gpt-4o-mini"

python inference.py
```

### 4. Push to OpenEnv
```powershell
openenv push --repo-id Purvesh09/Meta-Pytorch-OpenENV
```

##      Scoring Dimensions

- **Real-world utility (30%)**: Evaluates the full engineering workflow.
- **Task & Grader Quality (25%)**: Multi-step state machine with feedback at every stage.
- **Environment Design (20%)**: Episode memory and state preservation.
- **Creativity & Novelty (10%)**: Duplication detection and security escalation pathways.
