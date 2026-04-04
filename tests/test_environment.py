"""
Email Triage Environment - Tests
Tests for graders, reward logic, and episode lifecycle.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from models import (
    ActionType,
    Category,
    EmailTriageAction,
    Priority,
)
from server.environment import (
    EmailTriageEnvironment,
    _priority_score,
    _category_score,
    _reply_quality_score,
)


# ---------------------------------------------------------------------------
# Unit tests: scoring helpers
# ---------------------------------------------------------------------------


class TestPriorityScore:
    def test_exact_match(self):
        assert _priority_score("urgent", "urgent") == 1.0

    def test_adjacent(self):
        score = _priority_score("high", "urgent")
        assert 0.0 < score < 1.0

    def test_far_apart(self):
        score = _priority_score("spam", "urgent")
        assert score == 0.0

    def test_spam_correct(self):
        assert _priority_score("spam", "spam") == 1.0


class TestCategoryScore:
    def test_exact(self):
        assert _category_score("billing", "billing") == 1.0

    def test_wrong(self):
        assert _category_score("spam", "billing") == 0.0


class TestReplyQualityScore:
    def test_empty_reply(self):
        assert _reply_quality_score("", ["help", "resolve"], []) == 0.0

    def test_good_reply(self):
        score = _reply_quality_score(
            "We sincerely apologize for the issue. Our team will resolve this immediately.",
            ["apologize", "resolve"],
            ["unfortunately"],
        )
        assert score > 0.5

    def test_avoid_penalty(self):
        score_bad = _reply_quality_score(
            "unfortunately we are unable to help you with that.",
            ["help"],
            ["unfortunately", "unable"],
        )
        score_good = _reply_quality_score(
            "We are happy to help you with that right away.",
            ["help"],
            ["unfortunately", "unable"],
        )
        assert score_good > score_bad

    def test_too_short(self):
        assert _reply_quality_score("ok", ["help"], []) == 0.0


# ---------------------------------------------------------------------------
# Integration tests: environment lifecycle
# ---------------------------------------------------------------------------


class TestEnvironmentTask1:
    def setup_method(self):
        self.env = EmailTriageEnvironment(task_id="task1_easy_labelling")

    def test_reset_returns_observation(self):
        obs = self.env.reset()
        assert obs.current_email is not None
        assert obs.inbox_size == 8
        assert obs.processed_count == 0
        assert not obs.done

    def test_correct_label_gives_positive_reward(self):
        self.env.reset()
        # First email is urgent/internal
        action = EmailTriageAction(
            action_type=ActionType.LABEL,
            priority=Priority.URGENT,
            category=Category.INTERNAL,
        )
        obs, reward, done, info = self.env.step(action)
        assert reward > 0.5
        assert obs.processed_count == 1

    def test_wrong_priority_gives_lower_reward(self):
        self.env.reset()
        action = EmailTriageAction(
            action_type=ActionType.LABEL,
            priority=Priority.LOW,       # Wrong: it's urgent
            category=Category.INTERNAL,
        )
        obs, reward, done, info = self.env.step(action)
        assert reward < 0.5

    def test_deleting_spam_gives_high_reward(self):
        env = EmailTriageEnvironment(task_id="task1_easy_labelling")
        env.reset()
        # Skip to the spam email (index 4)
        skip = EmailTriageAction(action_type=ActionType.NEXT)
        for _ in range(4):
            env.step(skip)
        action = EmailTriageAction(action_type=ActionType.DELETE)
        _, reward, _, _ = env.step(action)
        assert reward >= 0.7

    def test_deleting_urgent_email_penalises(self):
        env = EmailTriageEnvironment(task_id="task1_easy_labelling")
        env.reset()
        # First email is urgent
        action = EmailTriageAction(action_type=ActionType.DELETE)
        _, reward, _, _ = env.step(action)
        assert reward < 0.0

    def test_full_episode_completes(self):
        env = EmailTriageEnvironment(task_id="task1_easy_labelling")
        env.reset()
        for _ in range(8):
            action = EmailTriageAction(
                action_type=ActionType.LABEL,
                priority=Priority.NORMAL,
                category=Category.OTHER,
            )
            obs, _, done, _ = env.step(action)
        assert done or obs.done

    def test_final_score_in_range(self):
        env = EmailTriageEnvironment(task_id="task1_easy_labelling")
        env.reset()
        for _ in range(8):
            env.step(EmailTriageAction(
                action_type=ActionType.LABEL,
                priority=Priority.URGENT,
                category=Category.CUSTOMER_SUPPORT,
            ))
        score = env.final_score()
        assert 0.0 <= score <= 1.0


class TestEnvironmentTask2:
    def test_unnecessary_reply_penalises(self):
        env = EmailTriageEnvironment(task_id="task2_medium_triage_reply")
        env.reset()
        # Skip to email 4 (newsletter, no reply needed)
        skip = EmailTriageAction(action_type=ActionType.NEXT)
        for _ in range(3):
            env.step(skip)
        action = EmailTriageAction(
            action_type=ActionType.REPLY,
            priority=Priority.LOW,
            category=Category.NEWSLETTER,
            reply_text="Thank you for your email. We will look into this.",
        )
        _, reward, _, _ = env.step(action)
        assert reward < 0.0

    def test_quality_reply_to_angry_customer(self):
        env = EmailTriageEnvironment(task_id="task2_medium_triage_reply")
        env.reset()
        # First email is angry customer
        action = EmailTriageAction(
            action_type=ActionType.REPLY,
            priority=Priority.URGENT,
            category=Category.CUSTOMER_SUPPORT,
            reply_text=(
                "We sincerely apologize for the disruption caused by our update. "
                "Our technical team is treating this as a top priority and will "
                "resolve the export issue immediately. We will escalate this to our "
                "engineers and keep you updated. Thank you for your patience."
            ),
        )
        _, reward, _, _ = env.step(action)
        assert reward > 0.5


class TestEnvironmentTask3:
    def test_casual_cfo_email_gets_high_priority(self):
        env = EmailTriageEnvironment(task_id="task3_hard_full_inbox")
        env.reset()
        # First email in task3 is casual-tone CFO request — true priority is high
        action = EmailTriageAction(
            action_type=ActionType.LABEL,
            priority=Priority.HIGH,
            category=Category.INTERNAL,
        )
        _, reward, _, _ = env.step(action)
        assert reward > 0.4  # Should score well

    def test_regulatory_email_is_urgent(self):
        env = EmailTriageEnvironment(task_id="task3_hard_full_inbox")
        env.reset()
        # Skip to regulatory email (index 6)
        skip = EmailTriageAction(action_type=ActionType.NEXT)
        for _ in range(6):
            env.step(skip)
        action = EmailTriageAction(
            action_type=ActionType.ESCALATE,
            reasoning="This is a regulatory compliance audit — must escalate to legal team.",
        )
        _, reward, _, _ = env.step(action)
        # Escalating an urgent email should yield positive reward
        assert reward >= 0.0

    def test_max_steps_terminates_episode(self):
        env = EmailTriageEnvironment(task_id="task3_hard_full_inbox")
        env.reset()
        done = False
        for _ in range(35):  # More than max_steps=30
            _, _, done, _ = env.step(EmailTriageAction(action_type=ActionType.NEXT))
            if done:
                break
        assert done


# ---------------------------------------------------------------------------
# State endpoint test
# ---------------------------------------------------------------------------


def test_state_reflects_episode():
    env = EmailTriageEnvironment(task_id="task1_easy_labelling")
    env.reset()
    state = env.state()
    assert state.task_id == "task1_easy_labelling"
    assert state.step_count == 0
    assert not state.done


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
