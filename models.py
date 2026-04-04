"""
Email Triage Environment - Models
Typed Pydantic models for Action, Observation, and State.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Priority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    SPAM = "spam"


class ActionType(str, Enum):
    LABEL = "label"          # Assign priority + category label to current email
    REPLY = "reply"          # Draft a reply to current email
    ARCHIVE = "archive"      # Archive without reply
    DELETE = "delete"        # Delete (spam / junk)
    ESCALATE = "escalate"    # Escalate to human / manager
    NEXT = "next"            # Move to next email without acting


class Category(str, Enum):
    CUSTOMER_SUPPORT = "customer_support"
    SALES_INQUIRY = "sales_inquiry"
    BILLING = "billing"
    TECHNICAL = "technical"
    INTERNAL = "internal"
    NEWSLETTER = "newsletter"
    SPAM = "spam"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Email data model (read-only for the agent)
# ---------------------------------------------------------------------------


class Email(BaseModel):
    email_id: str
    sender: str
    sender_domain: str
    subject: str
    body: str
    timestamp: str
    has_attachment: bool = False
    thread_length: int = 1
    # Ground-truth labels (hidden from agent, used by grader)
    _true_priority: Optional[str] = None
    _true_category: Optional[str] = None
    _requires_reply: Optional[bool] = None


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class EmailTriageAction(BaseModel):
    """Action the agent takes on the current email."""

    action_type: ActionType = Field(..., description="Type of action to perform")
    priority: Optional[Priority] = Field(
        None,
        description="Priority level (required for LABEL action)",
    )
    category: Optional[Category] = Field(
        None,
        description="Email category (required for LABEL action)",
    )
    reply_text: Optional[str] = Field(
        None,
        description="Reply body (required for REPLY action, max 500 chars)",
    )
    reasoning: Optional[str] = Field(
        None,
        description="Agent's reasoning for this action (used for partial credit)",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class EmailTriageObservation(BaseModel):
    """Observation returned after each step."""

    # Current email being shown (None if inbox is empty)
    current_email: Optional[Dict[str, Any]] = Field(
        None,
        description="The email the agent should act on now",
    )

    # Inbox summary
    inbox_size: int = Field(..., description="Total emails remaining in inbox")
    processed_count: int = Field(..., description="Emails processed so far")
    step_count: int = Field(..., description="Current step number")

    # Feedback from last action
    last_action_feedback: str = Field(
        "", description="Human-readable feedback on last action"
    )
    last_action_reward: float = Field(0.0, description="Reward from last action")

    # Episode metadata
    task_id: str = Field(..., description="Which task is being evaluated")
    done: bool = Field(False, description="Is the episode finished")

    # Running score
    cumulative_reward: float = Field(0.0, description="Total reward accumulated")


# ---------------------------------------------------------------------------
# State (internal, returned by state() endpoint)
# ---------------------------------------------------------------------------


class EmailTriageState(BaseModel):
    """Full internal state of the environment (for debugging / logging)."""

    episode_id: str
    task_id: str
    step_count: int
    max_steps: int
    inbox: List[Dict[str, Any]]
    processed: List[Dict[str, Any]]
    cumulative_reward: float
    done: bool
    seed: int
