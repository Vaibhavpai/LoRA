"""
reflexion.py — Phase 6, Step 6.4
==================================
Reflexion memory for the LoRA-SafeLoop agent.

Stores a rolling log of agent decisions and their measured outcomes.
The last N records are prepended to every agent observation, enabling
the agent to learn from past mistakes and avoid repeating failed strategies.

Each record captures:
  - Step when decision was made
  - Lambda values the agent set (per-layer)
  - Refusal rate before and after (measured at the NEXT eval checkpoint)
  - Task metric after
  - Agent's stated rationale
  - Outcome label: IMPROVEMENT / STABLE / SLIGHT_DECLINE / DEGRADATION

Records are persisted to a .jsonl file and held in memory for fast retrieval.
On resume, the file is replayed to restore memory state.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ReflexionMemory:
    """
    Rolling-window memory of agent decision records.

    Usage:
        memory = ReflexionMemory(log_path, max_recent=3)

        # After each agent decision, store a PENDING record:
        memory.add_pending(step, lambda_decisions, refusal_before, rationale)

        # At the NEXT eval checkpoint, complete the record with the observed outcome:
        memory.complete_pending(refusal_after, task_metric_after)

        # Format for agent observation:
        memory.format_for_agent()
    """

    def __init__(self, log_path: Path, max_recent: int = 5):
        self.log_path   = log_path
        self.max_recent = max_recent
        self.records: list[dict]     = []
        self._pending: Optional[dict] = None  # record awaiting its outcome

        # Replay existing log if resuming
        if log_path.exists():
            self._load_from_file()
            logger.info(
                f"ReflexionMemory: loaded {len(self.records)} existing records "
                f"from {log_path}."
            )
        else:
            logger.info(f"ReflexionMemory: new log at {log_path}.")

    # ------------------------------------------------------------------
    # Two-phase record creation (decision → pending → outcome → complete)
    # ------------------------------------------------------------------

    def add_pending(
        self,
        step: int,
        lambda_decisions: dict,
        refusal_before: float,
        rationale: str,
    ):
        """
        Store a decision that is awaiting its outcome.
        Called immediately after the agent acts.

        Args:
            step: Training step when the decision was made
            lambda_decisions: Full lambda_state at decision time
            refusal_before: Refusal rate at the time of decision
            rationale: Agent's stated rationale (truncated to 200 chars)
        """
        self._pending = {
            "step":             step,
            "lambda_decisions": lambda_decisions,
            "refusal_before":   round(refusal_before, 4),
            "rationale":        rationale[:200],
        }

    def complete_pending(
        self,
        refusal_after: float,
        task_metric_after: float,
    ):
        """
        Complete a pending record with the observed outcome.
        Called at the NEXT eval checkpoint.

        Args:
            refusal_after: Refusal rate measured at the next checkpoint
            task_metric_after: Task metric at the next checkpoint
        """
        if self._pending is None:
            return  # no pending record — first eval, nothing to complete

        change = refusal_after - self._pending["refusal_before"]

        if change > 0.02:
            label = "IMPROVEMENT"
        elif change >= -0.02:
            label = "STABLE"
        elif change >= -0.05:
            label = "SLIGHT_DECLINE"
        else:
            label = "DEGRADATION"

        record = {
            **self._pending,
            "refusal_after":    round(refusal_after, 4),
            "refusal_change":   round(change, 4),
            "task_metric_after": round(task_metric_after, 4),
            "outcome_label":    label,
        }

        self.records.append(record)
        self._append_to_file(record)
        self._pending = None

        logger.info(
            f"Reflexion record completed: Step {record['step']}, "
            f"refusal {record['refusal_before']:.3f} → {record['refusal_after']:.3f} "
            f"Δ={record['refusal_change']:+.3f} [{label}]"
        )

    # ------------------------------------------------------------------
    # Legacy single-call API (kept for compatibility with run_agent.py)
    # ------------------------------------------------------------------

    def add(
        self,
        step: int,
        lambda_decisions: dict,
        refusal_before: float,
        refusal_after: float,
        task_metric_after: float,
        rationale: str,
    ):
        """
        Add a completed record in one call.
        Used when refusal_before and refusal_after are both known.
        """
        change = refusal_after - refusal_before

        if change > 0.02:
            label = "IMPROVEMENT"
        elif change >= -0.02:
            label = "STABLE"
        elif change >= -0.05:
            label = "SLIGHT_DECLINE"
        else:
            label = "DEGRADATION"

        record = {
            "step":              step,
            "lambda_decisions":  lambda_decisions,
            "refusal_before":    round(refusal_before, 4),
            "refusal_after":     round(refusal_after, 4),
            "refusal_change":    round(change, 4),
            "task_metric_after": round(task_metric_after, 4),
            "outcome_label":     label,
            "rationale":         rationale[:200],
        }

        self.records.append(record)
        self._append_to_file(record)
        logger.info(
            f"Reflexion record added: Step {step}, "
            f"refusal {refusal_before:.3f} → {refusal_after:.3f} [{label}]"
        )

    # ------------------------------------------------------------------
    # Retrieval & formatting
    # ------------------------------------------------------------------

    def get_recent(self, n: Optional[int] = None) -> list:
        """Return the most recent N completed records."""
        n = n or self.max_recent
        return self.records[-n:]

    def format_for_agent(self, n: Optional[int] = None) -> str:
        """
        Format recent records as a human-readable string for the agent prompt.
        Emphasises outcome labels so the agent can detect patterns.
        """
        recent = self.get_recent(n)

        if not recent:
            return "  (No previous decisions recorded yet — this is the first checkpoint.)"

        lines = []
        for r in recent:
            # Aggregate per-projection lambdas to per-layer for compact display
            layer_avg = {}
            for k, v in r["lambda_decisions"].items():
                # Handle both string keys ("24.q_proj") and int keys (24)
                key_str = str(k)
                if "." in key_str:
                    layer_idx = key_str.split(".")[0]
                else:
                    layer_idx = key_str
                if layer_idx not in layer_avg:
                    layer_avg[layer_idx] = []
                layer_avg[layer_idx].append(float(v))
            per_layer = {l: sum(vs)/len(vs) for l, vs in layer_avg.items()}
            top_lambdas = sorted(per_layer.items(), key=lambda x: x[1], reverse=True)[:3]
            lambda_str = ", ".join(f"λ_{l}={v:.2f}" for l, v in top_lambdas)

            lines.append(
                f"  [Step {r['step']}] {lambda_str}\n"
                f"    Refusal: {r['refusal_before']:.3f} → {r['refusal_after']:.3f} "
                f"(Δ={r['refusal_change']:+.3f}) [{r['outcome_label']}]\n"
                f"    Rationale: {r['rationale'][:120]}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _append_to_file(self, record: dict):
        """Append a single JSON record to the log file."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write reflexion record: {e}")

    def _load_from_file(self):
        """Load and replay the log file on resume."""
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.error(f"Failed to load reflexion log from {self.log_path}: {e}")