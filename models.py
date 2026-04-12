"""
Bug Triage & Patch Validation Environment - Models
An AI agent acts as a senior engineer: reproduce bugs, diagnose root causes,
write fixes, validate patches, and prioritise the backlog.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "critical"   # System crash / data loss
    HIGH = "high"           # Major feature broken
    MEDIUM = "medium"       # Feature degraded, workaround exists
    LOW = "low"             # Minor cosmetic/UX issue
    WONTFIX = "wontfix"     # Known limitation, not worth fixing


class ActionType(str, Enum):
    REPRODUCE   = "reproduce"    # Attempt to reproduce the bug with a test case
    DIAGNOSE    = "diagnose"     # Identify root cause from stack trace / code
    PATCH       = "patch"        # Write a code fix
    VALIDATE    = "validate"     # Run tests to verify patch works
    ESCALATE    = "escalate"     # Escalate to senior team / security
    CLOSE       = "close"        # Close bug (fixed / wontfix / duplicate)
    REQUEST_INFO = "request_info" # Ask reporter for more info


class Component(str, Enum):
    AUTH        = "auth"
    DATABASE    = "database"
    API         = "api"
    FRONTEND    = "frontend"
    PAYMENTS    = "payments"
    SECURITY    = "security"
    INFRA       = "infra"
    OTHER       = "other"


class BugReport(BaseModel):
    bug_id: str
    title: str
    description: str
    stack_trace: Optional[str] = None
    steps_to_reproduce: Optional[str] = None
    reported_by: str
    reporter_type: str   # "customer", "internal", "security_researcher"
    created_at: str
    affected_version: str
    environment: str     # "production", "staging", "development"
    num_affected_users: int = 0
    has_workaround: bool = False


class BugTriageAction(BaseModel):
    """Action the agent takes on the current bug."""
    action_type: ActionType = Field(..., description="What action to take")
    severity: Optional[Severity] = Field(None, description="Assessed severity")
    component: Optional[Component] = Field(None, description="Which component owns this bug")
    # For REPRODUCE action
    test_case: Optional[str] = Field(None, description="Test case code to reproduce the bug")
    # For DIAGNOSE action
    root_cause: Optional[str] = Field(None, description="Identified root cause explanation")
    # For PATCH action
    patch_code: Optional[str] = Field(None, description="The fix code snippet")
    patch_explanation: Optional[str] = Field(None, description="Why this patch works")
    # For VALIDATE action
    test_results: Optional[str] = Field(None, description="Test output / validation results")
    # For CLOSE action
    resolution: Optional[str] = Field(None, description="How it was resolved")
    # For REQUEST_INFO
    info_request: Optional[str] = Field(None, description="What info is needed from reporter")
    # Universal
    reasoning: Optional[str] = Field(None, description="Agent's reasoning")


class BugTriageObservation(BaseModel):
    """Observation returned after each step."""
    current_bug: Optional[Dict[str, Any]] = Field(None, description="Current bug to work on")
    # Bug state machine progress
    bug_state: str = Field("new", description="new -> reproduced -> diagnosed -> patched -> validated -> closed")
    reproduce_result: Optional[str] = Field(None, description="Result of reproduce attempt")
    diagnosis_feedback: Optional[str] = Field(None, description="Feedback on diagnosis quality")
    patch_feedback: Optional[str] = Field(None, description="Feedback on patch quality")
    validation_feedback: Optional[str] = Field(None, description="Test results feedback")
    # Queue info
    backlog_size: int = Field(0)
    resolved_count: int = Field(0)
    step_count: int = Field(0)
    # Feedback
    last_action_feedback: str = Field("")
    last_action_reward: float = Field(0.0)
    cumulative_reward: float = Field(0.0)
    task_id: str = Field("")
    done: bool = Field(False)


class BugTriageState(BaseModel):
    """Full internal episode state."""
    episode_id: str
    task_id: str
    step_count: int
    max_steps: int
    backlog: List[Dict[str, Any]]
    resolved: List[Dict[str, Any]]
    current_bug_state: str
    cumulative_reward: float
    done: bool
    seed: int
