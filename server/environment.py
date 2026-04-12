"""
Bug Triage & Patch Validation Environment - Core Logic
State machine per bug: new -> reproduced -> diagnosed -> patched -> validated -> closed
Rich reward shaping with partial credit at every step.
"""

from __future__ import annotations
import copy, uuid
from typing import Any, Dict, List, Optional, Tuple

from models import (
    ActionType, Component, BugTriageAction,
    BugTriageObservation, BugTriageState, Severity,
)
from bugs import ALL_TASKS

SEVERITY_ORDER = ["critical", "high", "medium", "low", "wontfix"]
SEVERITY_DISTANCE_PENALTY = {0: 1.0, 1: 0.5, 2: 0.1, 3: 0.0, 4: 0.0}

BUG_STATE_MACHINE = ["new", "reproduced", "diagnosed", "patched", "validated", "closed"]


def _severity_score(predicted: str, true: str) -> float:
    try:
        d = abs(SEVERITY_ORDER.index(predicted) - SEVERITY_ORDER.index(true))
        return SEVERITY_DISTANCE_PENALTY.get(d, 0.0)
    except ValueError:
        return 0.0


def _component_score(predicted: str, true: str) -> float:
    return 1.0 if predicted == true else 0.0


def _keyword_score(text: str, keywords: List[str], weight: float = 1.0) -> float:
    if not text or not keywords:
        return weight  # full credit if no keywords required
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return weight * min(hits / max(len(keywords) * 0.4, 1), 1.0)


def _length_ok(text: Optional[str], min_len: int = 30) -> bool:
    return bool(text and len(text.strip()) >= min_len)


class BugTriageEnvironment:
    def __init__(self, task_id: str = "task1_easy_severity_routing", seed: int = 42):
        if task_id not in ALL_TASKS:
            raise ValueError(f"Unknown task_id: {task_id}. Choose from {list(ALL_TASKS.keys())}")
        self.task_id = task_id
        self.seed = seed
        self._task_config = ALL_TASKS[task_id]
        self._episode_id = ""
        self._backlog: List[Dict] = []
        self._resolved: List[Dict] = []
        self._current_idx = 0
        self._bug_state = "new"
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._reproduce_result: Optional[str] = None
        self._diagnosis_feedback: Optional[str] = None
        self._patch_feedback: Optional[str] = None
        self._validation_feedback: Optional[str] = None

    def reset(self) -> BugTriageObservation:
        self._episode_id = str(uuid.uuid4())[:8]
        self._backlog = copy.deepcopy(self._task_config["bugs"])
        self._resolved = []
        self._current_idx = 0
        self._bug_state = "new"
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._clear_step_state()
        return self._build_obs("Backlog loaded. Start with the first bug.", 0.0)

    def step(self, action: BugTriageAction) -> Tuple[BugTriageObservation, float, bool, Dict]:
        if self._done:
            return self._build_obs("Episode done.", 0.0), 0.0, True, {}

        self._step_count += 1
        reward = 0.0
        feedback = ""
        info: Dict[str, Any] = {}

        if self._current_idx >= len(self._backlog):
            self._done = True
            return self._build_obs("All bugs resolved.", 0.0), 0.0, True, {}

        bug = self._backlog[self._current_idx]
        atype = action.action_type

        # ---- REPRODUCE ----
        if atype == ActionType.REPRODUCE:
            reward, feedback = self._grade_reproduce(action, bug)

        # ---- DIAGNOSE ----
        elif atype == ActionType.DIAGNOSE:
            reward, feedback = self._grade_diagnose(action, bug)

        # ---- PATCH ----
        elif atype == ActionType.PATCH:
            reward, feedback = self._grade_patch(action, bug)

        # ---- VALIDATE ----
        elif atype == ActionType.VALIDATE:
            reward, feedback = self._grade_validate(action, bug)

        # ---- ESCALATE ----
        elif atype == ActionType.ESCALATE:
            reward, feedback = self._grade_escalate(action, bug)

        # ---- CLOSE ----
        elif atype == ActionType.CLOSE:
            reward, feedback = self._grade_close(action, bug)
            if reward >= 0:
                self._advance_bug(bug, atype.value, reward)

        # ---- REQUEST_INFO ----
        elif atype == ActionType.REQUEST_INFO:
            if not bug.get("steps_to_reproduce"):
                reward = 0.1
                feedback = "Reasonably requested more info for unclear bug. (+0.1)"
            else:
                reward = -0.05
                feedback = "Requested info unnecessarily - steps already provided. (-0.05)"

        else:
            reward = -0.1
            feedback = f"Unknown action: {atype}"

        reward = max(-1.0, min(1.0, reward))
        self._cumulative_reward += reward
        info["bug_id"] = bug["bug_id"]
        info["action"] = atype.value
        info["bug_state"] = self._bug_state

        if self._current_idx >= len(self._backlog):
            self._done = True
        elif self._step_count >= self._task_config["max_steps"]:
            self._done = True
            feedback += " [MAX STEPS]"

        return self._build_obs(feedback, reward), reward, self._done, info

    def state(self) -> BugTriageState:
        return BugTriageState(
            episode_id=self._episode_id,
            task_id=self.task_id,
            step_count=self._step_count,
            max_steps=self._task_config["max_steps"],
            backlog=[self._safe_bug(b) for b in self._backlog[self._current_idx:]],
            resolved=self._resolved,
            current_bug_state=self._bug_state,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    # Graders
    # ------------------------------------------------------------------

    def _grade_reproduce(self, action: BugTriageAction, bug: Dict) -> Tuple[float, str]:
        w = self._task_config["scoring"].get("reproduce_weight", 0.15)
        if not bug.get("reproduce_keywords"):
            # Bug doesn't need reproduction - penalise wasted step
            return -0.05, "This bug doesn't require reproduction. (-0.05)"

        if not _length_ok(action.test_case, 20):
            return 0.0, "Test case too short or missing. (0.0)"

        score = _keyword_score(action.test_case or "", bug["reproduce_keywords"])
        reward = w * score
        self._bug_state = "reproduced"
        self._reproduce_result = f"Reproduced with score {score:.2f}"
        return reward, f"Reproduce score={score:.2f} -> reward={reward:.3f}"

    def _grade_diagnose(self, action: BugTriageAction, bug: Dict) -> Tuple[float, str]:
        cfg = self._task_config["scoring"]
        w_diag = cfg.get("diagnosis_weight", 0.25)
        w_sev = cfg.get("severity_weight", 0.15)
        w_comp = cfg.get("component_weight", 0.10)

        # Duplicate detection bonus
        if bug.get("is_duplicate_of"):
            dup_text = (action.root_cause or "") + (action.reasoning or "")
            if "duplicate" in dup_text.lower() or bug["is_duplicate_of"].lower() in dup_text.lower():
                self._bug_state = "diagnosed"
                self._diagnosis_feedback = "Correctly identified as duplicate"
                return 0.3, f"Correctly identified as duplicate of {bug['is_duplicate_of']}. (+0.3)"
            else:
                return -0.1, f"Missed that this is a duplicate of {bug['is_duplicate_of']}. (-0.1)"

        if not _length_ok(action.root_cause, 20):
            return 0.0, "Root cause too short or missing. (0.0)"

        diag_score = _keyword_score(action.root_cause or "", bug["diagnosis_keywords"])
        sev_score = _severity_score(action.severity.value if action.severity else "low", bug["true_severity"])
        comp_score = _component_score(action.component.value if action.component else "other", bug["true_component"])

        reward = w_diag * diag_score + w_sev * sev_score + w_comp * comp_score
        self._bug_state = "diagnosed"
        self._diagnosis_feedback = f"diag={diag_score:.2f} sev={sev_score:.2f} comp={comp_score:.2f}"
        return reward, f"Diagnosis: diag_score={diag_score:.2f}, severity={sev_score:.2f}, component={comp_score:.2f} -> reward={reward:.3f}"

    def _grade_patch(self, action: BugTriageAction, bug: Dict) -> Tuple[float, str]:
        w = self._task_config["scoring"].get("patch_weight", 0.25)

        if bug.get("is_duplicate_of"):
            return -0.1, f"No patch needed - this is a duplicate. (-0.1)"

        if self._bug_state not in ("diagnosed", "reproduced"):
            return -0.05, f"Patch before diagnosis - rushing. (-0.05) (current state: {self._bug_state})"

        if not _length_ok(action.patch_code, 20):
            return 0.0, "Patch code too short or missing. (0.0)"

        valid_conditions = bug.get("patch_valid_if", [])
        score = _keyword_score(action.patch_code or "", valid_conditions)
        # Also score explanation
        expl_score = _keyword_score(action.patch_explanation or "", bug.get("patch_keywords", []))
        combined = 0.7 * score + 0.3 * expl_score
        reward = w * combined
        self._bug_state = "patched"
        self._patch_feedback = f"Patch score={combined:.2f}"
        return reward, f"Patch quality={combined:.2f} -> reward={reward:.3f}"

    def _grade_validate(self, action: BugTriageAction, bug: Dict) -> Tuple[float, str]:
        w = self._task_config["scoring"].get("validation_weight", 0.10)

        if self._bug_state != "patched":
            return -0.05, f"Validate before patch! (state: {self._bug_state}) (-0.05)"

        if not _length_ok(action.test_results, 15):
            return 0.0, "Test results too short or missing. (0.0)"

        valid_conditions = bug.get("test_valid_if", [])
        score = _keyword_score(action.test_results or "", valid_conditions)
        reward = w * score
        self._bug_state = "validated"
        self._validation_feedback = f"Validation score={score:.2f}"
        return reward, f"Validation score={score:.2f} -> reward={reward:.3f}"

    def _grade_escalate(self, action: BugTriageAction, bug: Dict) -> Tuple[float, str]:
        w = self._task_config["scoring"].get("escalation_accuracy_weight", 0.05)
        if bug.get("requires_escalation"):
            self._bug_state = "diagnosed"
            return w * 1.0, f"Correctly escalated security/critical bug. (+{w:.2f})"
        elif bug["true_severity"] in ("critical", "high"):
            return 0.0, "Escalated high-severity bug (acceptable, not required). (0.0)"
        else:
            return -0.1, f"Unnecessarily escalated a {bug['true_severity']} bug. (-0.1)"

    def _grade_close(self, action: BugTriageAction, bug: Dict) -> Tuple[float, str]:
        # Closing earns a small bonus for completing the workflow
        if self._bug_state in ("validated", "diagnosed", "patched"):
            bonus = 0.05
            self._bug_state = "closed"
            return bonus, f"Bug closed after {self._bug_state} stage. (+{bonus})"
        elif self._bug_state == "new":
            return -0.1, "Closed bug without any investigation. (-0.1)"
        else:
            self._bug_state = "closed"
            return 0.02, "Bug closed. (+0.02)"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _advance_bug(self, bug: Dict, action: str, reward: float):
        record = {**self._safe_bug(bug), "action_taken": action,
                  "reward": reward, "final_state": self._bug_state}
        self._resolved.append(record)
        self._current_idx += 1
        self._bug_state = "new"
        self._clear_step_state()

    def _clear_step_state(self):
        self._reproduce_result = None
        self._diagnosis_feedback = None
        self._patch_feedback = None
        self._validation_feedback = None

    def _safe_bug(self, bug: Dict) -> Dict:
        hidden = {"true_severity", "true_component", "root_cause", "diagnosis_keywords",
                  "patch_keywords", "patch_valid_if", "test_valid_if", "reproduce_keywords",
                  "requires_escalation", "is_duplicate_of", "is_by_design"}
        return {k: v for k, v in bug.items() if k not in hidden}

    def _build_obs(self, feedback: str, reward: float) -> BugTriageObservation:
        remaining = len(self._backlog) - self._current_idx
        current = None
        if self._current_idx < len(self._backlog):
            current = self._safe_bug(self._backlog[self._current_idx])
        return BugTriageObservation(
            current_bug=current,
            bug_state=self._bug_state,
            reproduce_result=self._reproduce_result,
            diagnosis_feedback=self._diagnosis_feedback,
            patch_feedback=self._patch_feedback,
            validation_feedback=self._validation_feedback,
            backlog_size=remaining,
            resolved_count=len(self._resolved),
            step_count=self._step_count,
            last_action_feedback=feedback,
            last_action_reward=reward,
            cumulative_reward=self._cumulative_reward,
            task_id=self.task_id,
            done=self._done,
        )

    def final_score(self) -> float:
        n = len(self._backlog)
        if n == 0:
            return 0.0
        cfg = self._task_config["scoring"]
        max_per = sum(cfg.values())
        theoretical = n * max_per
        return max(0.0, min(1.0, self._cumulative_reward / theoretical)) if theoretical > 0 else 0.0
