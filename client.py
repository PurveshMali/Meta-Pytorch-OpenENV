"""
Email Triage Environment - Python Client
Connects to the environment server and provides a clean Python API.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx

from models import (
    ActionType,
    Category,
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
    Priority,
)


class EmailTriageEnvClient:
    """
    Async HTTP client for the Email Triage environment.

    Usage:
        client = EmailTriageEnvClient(base_url="http://localhost:7860")
        obs = await client.reset(task_id="task1_easy_labelling")
        obs, reward, done, info = await client.step(action)
        state = await client.state()
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def reset(
        self, task_id: str = "task1_easy_labelling", seed: int = 42
    ) -> EmailTriageObservation:
        """Reset the environment and return the initial observation."""
        resp = await self._client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
        )
        resp.raise_for_status()
        data = resp.json()
        return EmailTriageObservation(**data["observation"])

    async def step(
        self, action: EmailTriageAction
    ) -> tuple[EmailTriageObservation, float, bool, Dict[str, Any]]:
        """Execute an action and return (observation, reward, done, info)."""
        payload = {
            "action_type": action.action_type.value,
            "priority": action.priority.value if action.priority else None,
            "category": action.category.value if action.category else None,
            "reply_text": action.reply_text,
            "reasoning": action.reasoning,
        }
        resp = await self._client.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        obs = EmailTriageObservation(**data["observation"])
        return obs, data["reward"], data["done"], data["info"]

    async def state(self) -> EmailTriageState:
        """Return the current environment state."""
        resp = await self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return EmailTriageState(**resp.json())

    async def score(self) -> Dict[str, Any]:
        """Return current episode score."""
        resp = await self._client.get(f"{self.base_url}/score")
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self._client.aclose()

    # Sync convenience wrapper
    @classmethod
    def from_url(cls, url: str) -> "EmailTriageEnvClient":
        return cls(base_url=url)
