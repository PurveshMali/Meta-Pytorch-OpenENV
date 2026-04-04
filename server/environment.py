"""
Email Triage Environment - Core Logic
Implements reset(), step(), state() with graders and reward shaping.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional, Tuple

from models import (
    ActionType,
    Category,
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
    Priority,
)
from emails import ALL_TASKS


# ---------------------------------------------------------------------------
# Priority adjacency for partial credit
# ---------------------------------------------------------------------------

PRIORITY_ORDER = ["urgent", "high", "normal", "low", "spam"]
PRIORITY_DISTANCE_PENALTY = {0: 1.0, 1: 0.5, 2: 0.1, 3: 0.0, 4: 0.0}

CATEGORY_EXACT_SCORE = 1.0
CATEGORY_WRONG_SCORE = 0.0


def _priority_score(predicted: str, true: str) -> float:
    """Partial credit for priority based on proximity in severity scale."""
    if predicted == true:
        return 1.0
    try:
        pd = PRIORITY_ORDER.index(predicted)
        td = PRIORITY_ORDER.index(true)
        dist = abs(pd - td)
        return PRIORITY_DISTANCE_PENALTY.get(dist, 0.0)
    except ValueError:
        return 0.0


def _category_score(predicted: str, true: str) -> float:
    """Exact match for category (no partial credit)."""
    return CATEGORY_EXACT_SCORE if predicted == true else CATEGORY_WRONG_SCORE


def _reply_quality_score(reply_text: str, keywords: List[str], avoid: List[str]) -> float:
    """Score reply quality by keyword presence and avoidance."""
    if not reply_text or len(reply_text.strip()) < 20:
        return 0.0

    text_lower = reply_text.lower()

    # Keyword presence (up to 0.7 of score)
    if keywords:
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        keyword_score = min(hits / max(len(keywords) * 0.5, 1), 1.0) * 0.7
    else:
        keyword_score = 0.7  # No keywords required → full keyword score

    # Avoid bad phrases (up to 0.3 of score — deductions)
    avoid_penalty = sum(0.15 for phrase in avoid if phrase.lower() in text_lower)

    # Length bonus: penalise very short or very long replies
    length = len(reply_text.strip())
    if length < 30:
        length_score = 0.0
    elif length < 80:
        length_score = 0.15
    elif length <= 500:
        length_score = 0.3
    else:
        length_score = 0.1  # Too verbose

    raw = keyword_score + length_score - avoid_penalty
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------


class EmailTriageEnvironment:
    """
    Email Triage RL Environment.

    Real-world task: process an inbox by labelling priority, categorising,
    and drafting appropriate replies. Three tasks of increasing difficulty.
    """

    def __init__(self, task_id: str = "task1_easy_labelling", seed: int = 42):
        if task_id not in ALL_TASKS:
            raise ValueError(f"Unknown task_id: {task_id}. Choose from {list(ALL_TASKS.keys())}")
        self.task_id = task_id
        self.seed = seed
        self._task_config = ALL_TASKS[task_id]

        # Episode state
        self._episode_id: str = ""
        self._inbox: List[Dict[str, Any]] = []
        self._processed: List[Dict[str, Any]] = []
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._current_email_idx: int = 0

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> EmailTriageObservation:
        """Start a fresh episode."""
        self._episode_id = str(uuid.uuid4())[:8]
        self._inbox = copy.deepcopy(self._task_config["emails"])
        self._processed = []
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._current_email_idx = 0

        return self._build_observation(
            last_feedback="Inbox loaded. Start triaging emails.",
            last_reward=0.0,
        )

    def step(self, action: EmailTriageAction) -> Tuple[EmailTriageObservation, float, bool, Dict]:
        """Process one action on the current email."""
        if self._done:
            obs = self._build_observation("Episode already done.", 0.0)
            return obs, 0.0, True, {"error": "episode_done"}

        self._step_count += 1
        reward = 0.0
        feedback = ""
        info: Dict[str, Any] = {}

        # Get current email
        if self._current_email_idx >= len(self._inbox):
            self._done = True
            obs = self._build_observation("All emails processed.", 0.0)
            return obs, 0.0, True, {"info": "inbox_empty"}

        current_email = self._inbox[self._current_email_idx]

        # --- Process action ---
        action_type = action.action_type

        if action_type == ActionType.NEXT:
            # Skip without acting — small penalty for ignoring emails
            reward = -0.05
            feedback = f"Skipped email '{current_email['subject'][:40]}'. (-0.05)"
            self._advance_email(current_email, action_type.value, reward)

        elif action_type == ActionType.LABEL:
            reward, feedback = self._grade_label(action, current_email)
            self._advance_email(current_email, action_type.value, reward)

        elif action_type == ActionType.REPLY:
            reward, feedback = self._grade_reply(action, current_email)
            self._advance_email(current_email, action_type.value, reward)

        elif action_type == ActionType.DELETE:
            reward, feedback = self._grade_delete(action, current_email)
            self._advance_email(current_email, action_type.value, reward)

        elif action_type == ActionType.ARCHIVE:
            reward, feedback = self._grade_archive(action, current_email)
            self._advance_email(current_email, action_type.value, reward)

        elif action_type == ActionType.ESCALATE:
            reward, feedback = self._grade_escalate(action, current_email)
            self._advance_email(current_email, action_type.value, reward)

        else:
            reward = -0.1
            feedback = f"Unknown action type: {action_type}"

        # Clamp reward
        reward = max(-1.0, min(1.0, reward))
        self._cumulative_reward += reward
        info["email_id"] = current_email["email_id"]
        info["action"] = action_type.value

        # Check episode end
        if self._current_email_idx >= len(self._inbox):
            self._done = True
        elif self._step_count >= self._task_config["max_steps"]:
            self._done = True
            feedback += " [MAX STEPS REACHED]"

        obs = self._build_observation(feedback, reward)
        return obs, reward, self._done, info

    def state(self) -> EmailTriageState:
        """Return full internal state."""
        return EmailTriageState(
            episode_id=self._episode_id,
            task_id=self.task_id,
            step_count=self._step_count,
            max_steps=self._task_config["max_steps"],
            inbox=[self._safe_email(e) for e in self._inbox[self._current_email_idx:]],
            processed=self._processed,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    # Graders
    # ------------------------------------------------------------------

    def _grade_label(self, action: EmailTriageAction, email: Dict) -> Tuple[float, str]:
        """Grade a LABEL action (priority + category)."""
        config = self._task_config["scoring"]

        if action.priority is None or action.category is None:
            return -0.1, "LABEL action requires both priority and category. (-0.1)"

        p_score = _priority_score(action.priority.value, email["true_priority"])
        c_score = _category_score(action.category.value, email["true_category"])

        p_weight = config.get("priority_weight", 0.6)
        c_weight = config.get("category_weight", 0.4)

        reward = (p_weight * p_score + c_weight * c_score)
        feedback = (
            f"Label: priority={action.priority.value} (true={email['true_priority']}, "
            f"score={p_score:.2f}), category={action.category.value} "
            f"(true={email['true_category']}, score={c_score:.2f}) → reward={reward:.3f}"
        )
        return reward, feedback

    def _grade_reply(self, action: EmailTriageAction, email: Dict) -> Tuple[float, str]:
        """Grade a REPLY action on quality and appropriateness."""
        config = self._task_config["scoring"]
        reply_weight = config.get("reply_quality_weight", 0.40)

        if not email.get("requires_reply", False):
            # Replying when not needed — small penalty for noise
            return -0.1, f"Unnecessary reply to '{email['subject'][:40]}'. (-0.1)"

        keywords = email.get("reply_keywords", [])
        avoid = email.get("reply_avoid", [])
        r_score = _reply_quality_score(action.reply_text or "", keywords, avoid)
        reward = reply_weight * r_score

        # Also award partial credit for labelling (if priority/category provided)
        if action.priority and action.category:
            p_score = _priority_score(action.priority.value, email["true_priority"])
            c_score = _category_score(action.category.value, email["true_category"])
            p_weight = config.get("priority_weight", 0.35)
            c_weight = config.get("category_weight", 0.25)
            reward += p_weight * p_score + c_weight * c_score

        reward = min(1.0, reward)
        feedback = (
            f"Reply quality score={r_score:.2f} → reward={reward:.3f}. "
            f"Reply preview: '{(action.reply_text or '')[:60]}...'"
        )
        return reward, feedback

    def _grade_delete(self, action: EmailTriageAction, email: Dict) -> Tuple[float, str]:
        """Grade a DELETE action (spam detection)."""
        true_priority = email["true_priority"]
        if true_priority == "spam":
            reward = 0.8
            feedback = f"Correctly deleted spam email. (+0.8)"
        elif true_priority == "low":
            reward = 0.1
            feedback = f"Deleted low-priority email (acceptable but not ideal). (+0.1)"
        elif true_priority in ("normal", "high"):
            reward = -0.5
            feedback = f"MISTAKE: Deleted a {true_priority}-priority email! (-0.5)"
        else:
            reward = -0.8
            feedback = f"CRITICAL MISTAKE: Deleted an urgent email! (-0.8)"
        return reward, feedback

    def _grade_archive(self, action: EmailTriageAction, email: Dict) -> Tuple[float, str]:
        """Grade an ARCHIVE action."""
        true_priority = email["true_priority"]
        if true_priority in ("low", "normal"):
            reward = 0.4
            feedback = f"Archived {true_priority} email appropriately. (+0.4)"
        elif true_priority == "spam":
            reward = 0.3
            feedback = "Archived spam (delete would be better). (+0.3)"
        elif true_priority == "high":
            reward = -0.2
            feedback = f"Archived a HIGH-priority email without acting. (-0.2)"
        else:
            reward = -0.6
            feedback = f"Archived URGENT email without acting. (-0.6)"
        return reward, feedback

    def _grade_escalate(self, action: EmailTriageAction, email: Dict) -> Tuple[float, str]:
        """Grade an ESCALATE action."""
        true_priority = email["true_priority"]
        config = self._task_config["scoring"]
        escalation_weight = config.get("escalation_accuracy_weight", 0.15)

        if true_priority in ("urgent", "high"):
            reward = escalation_weight * 1.0
            feedback = f"Correctly escalated {true_priority} email. (+{reward:.2f})"
        elif true_priority == "normal":
            reward = 0.0
            feedback = "Escalated a normal-priority email (unnecessary). (0.0)"
        else:
            reward = -0.1
            feedback = f"Escalated a {true_priority} email unnecessarily. (-0.1)"
        return reward, feedback

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _advance_email(self, email: Dict, action_taken: str, reward: float):
        """Move current email to processed list and advance pointer."""
        record = {**self._safe_email(email), "action_taken": action_taken, "reward": reward}
        self._processed.append(record)
        self._current_email_idx += 1

    def _safe_email(self, email: Dict) -> Dict:
        """Return email dict without hidden ground-truth fields."""
        return {
            k: v
            for k, v in email.items()
            if not k.startswith("true_") and k not in ("reply_keywords", "reply_avoid", "requires_reply")
        }

    def _build_observation(self, last_feedback: str, last_reward: float) -> EmailTriageObservation:
        """Construct an observation from current state."""
        remaining = len(self._inbox) - self._current_email_idx
        current = None
        if self._current_email_idx < len(self._inbox):
            current = self._safe_email(self._inbox[self._current_email_idx])

        return EmailTriageObservation(
            current_email=current,
            inbox_size=remaining,
            processed_count=len(self._processed),
            step_count=self._step_count,
            last_action_feedback=last_feedback,
            last_action_reward=last_reward,
            task_id=self.task_id,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
        )

    # ------------------------------------------------------------------
    # Episode summary / grader score
    # ------------------------------------------------------------------

    def final_score(self) -> float:
        """
        Compute normalised final score [0, 1] for the episode.
        Based on total reward relative to theoretical maximum.
        """
        n_emails = len(self._inbox)
        if n_emails == 0:
            return 0.0

        # Theoretical max reward per email depends on task config
        config = self._task_config["scoring"]
        p_w = config.get("priority_weight", 0.6)
        c_w = config.get("category_weight", 0.4)
        r_w = config.get("reply_quality_weight", 0.0)
        e_w = config.get("escalation_accuracy_weight", 0.0)
        max_per_email = p_w + c_w + r_w + e_w

        theoretical_max = n_emails * max_per_email
        raw = self._cumulative_reward / theoretical_max if theoretical_max > 0 else 0.0
        return max(0.0, min(1.0, raw))
