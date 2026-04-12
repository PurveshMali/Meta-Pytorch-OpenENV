"""
Bug Triage & Patch Validation Environment - Python Client
Connects to the environment server and provides a clean Python API.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx

from models import (
    ActionType,
    Component,
    BugTriageAction,
    BugTriageObservation,
    BugTriageState,
    Severity,
)


class BugTriageEnvClient:
    """
    Async HTTP client for the Bug Triage environment.

    Usage:
        client = BugTriageEnvClient(base_url="http://localhost:7860")
        obs = await client.reset(task_id="task1_easy_severity_routing")
        obs, reward, done, info = await client.step(action)
        state = await client.state()
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def reset(
        self, task_id: str = "task1_easy_severity_routing", seed: int = 42
    ) -> BugTriageObservation:
        """Reset the environment and return the initial observation."""
        resp = await self._client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
        )
        resp.raise_for_status()
        data = resp.json()
        return BugTriageObservation(**data["observation"])

    async def step(
        self, action: BugTriageAction
    ) -> tuple[BugTriageObservation, float, bool, Dict[str, Any]]:
        """Execute an action and return (observation, reward, done, info)."""
        payload = action.model_dump()
        resp = await self._client.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        obs = BugTriageObservation(**data["observation"])
        return obs, data["reward"], data["done"], data["info"]

    async def state(self) -> BugTriageState:
        """Return the current environment state."""
        resp = await self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return BugTriageState(**resp.json())

    async def score(self) -> Dict[str, Any]:
        """Return current episode score."""
        resp = await self._client.get(f"{self.base_url}/score")
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self._client.aclose()

    @classmethod
    def from_url(cls, url: str) -> "BugTriageEnvClient":
        return cls(base_url=url)
