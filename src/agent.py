"""
agent.py — Phase 6 (v4.1, per-projection)
============================================
LLM-based training controller for LoRA-SafeLoop.

v4.1 changes from v4:
  - Observation shows per-projection alignment (q/v separate) but agent still
    decides per-LAYER (28 decisions). Lambda is split equally to q_proj and
    v_proj within each layer by the parser — keeps the LLM's action space
    manageable while the constraint operates per-projection.
  - Internal state uses string keys ("24.q_proj", "24.v_proj").
  - System prompt updated for per-projection context.

Agent contract:
  INPUT : observation string (metrics + reflexion memory)
  OUTPUT: {
    "layer_constraints": {"24": 0.8, "22": 0.7, ...},  // per-LAYER, split to q/v
    "rationale": "...",
    "predicted_outcome": "..."
  }
"""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Optional

import groq

from src.reflexion import ReflexionMemory

logger = logging.getLogger(__name__)


# ===========================================================================
# System prompt
# ===========================================================================

AGENT_SYSTEM_PROMPT = """You are a safety-preserving training controller for a language model fine-tuning pipeline.

A base LLM (Qwen2.5-1.5B-Instruct) is being fine-tuned on a task dataset. Fine-tuning causes the model to gradually lose safety alignment ("safety drift"). Your DUAL objective is:
  1. MAINTAIN refusal rate above the target floor (baseline - 5%) — safety must not erode
  2. MAXIMIZE task performance (lower loss for Alpaca, higher accuracy for GSM8K)

The ideal λ configuration is the MINIMUM constraint that keeps safety stable.

HOW THE CONSTRAINT WORKS (continuous penalty, per-projection):
- Each transformer layer (0-27) has TWO constraint strengths: one for q_proj, one for v_proj.
- You set lambda PER LAYER — it is applied equally to both q_proj and v_proj in that layer.
- Every training step, a penalty is added: penalty = λ × (how much the weight update points toward the unsafe direction)².
- λ = 0.0: NO constraint. λ = 0.3: light. λ = 0.5: moderate. λ = 0.7: strong. λ = 1.0: maximum.
- λ values AUTO-DECAY by 2% each checkpoint toward 0.

In the observation, you will see per-projection alignment scores (q_align, v_align). These show how much each projection's weight update aligns with the unsafe direction. Use these to decide which layers need the most constraint.

STRATEGY PRINCIPLES:
1. USE SMOOTHED REFUSAL (3-step average) for decisions, NOT raw refusal.
2. TARGET LAYERS SELECTIVELY: Set λ=0 on layers with near-zero alignment. Concentrate on layers with high alignment.
3. ACTIVELY REDUCE λ WHEN SAFE: If smoothed refusal is ABOVE the target floor, lower λ on low-alignment layers.
4. If smoothed refusal is NEAR the target floor (within 5%): raise λ moderately (+0.10 to +0.20) on the top-5 alignment layers.
5. If smoothed refusal is BELOW the target floor: raise λ aggressively on the top-10 alignment layers (to 0.7-0.9).
6. Learn from reflexion memory: if a strategy produced DEGRADATION, avoid repeating it.
7. Make PROPORTIONAL adjustments based on alignment scores.

RESPOND WITH VALID JSON ONLY. No text outside the JSON block. Required format:
{
  "layer_constraints": {
    "LAYER_IDX_AS_STRING": LAMBDA_VALUE_AS_FLOAT,
    ...
  },
  "rationale": "1-2 sentence explanation of your decision",
  "predicted_outcome": "Brief prediction for the next 100 steps"
}"""


# ===========================================================================
# Observation Formatter
# ===========================================================================

def format_observation(
    step: int,
    total_steps: int,
    task: str,
    refusal_rate: float,
    prev_refusal_rate: float,
    baseline_refusal_rate: float,
    task_metric: float,
    metric_name: str,
    alignments: dict,
    lambda_state: dict,
    reflexion_memory: ReflexionMemory,
    smoothed_refusal_rate: float = None,
    top_k_layers: int = 10,
) -> str:
    """
    Builds a structured observation string for the agent.

    v4.1: alignments and lambda_state use string keys ("{layer}.{proj}").
    The observation aggregates per-projection info into per-layer rows for readability.

    Args:
        alignments: dict "{layer}.{proj}" -> alignment score
        lambda_state: dict "{layer}.{proj}" -> current lambda
        (other args unchanged)
    """
    pct_done = 100.0 * step / total_steps
    step_change = refusal_rate - prev_refusal_rate

    target_floor = max(0.0, baseline_refusal_rate - 0.05)

    # Detect declining trend
    recent_records = reflexion_memory.get_recent(3)
    declining_steps = 0
    for r in recent_records:
        if r.get("refusal_change", 0) < -0.02:
            declining_steps += 1

    # Aggregate per-projection data into per-layer
    layer_data = {}
    for key in sorted(alignments.keys()):
        parts = key.split(".")
        layer_idx = int(parts[0])
        proj_name = parts[1]

        if layer_idx not in layer_data:
            layer_data[layer_idx] = {"q_align": 0.0, "v_align": 0.0, "q_lam": 0.0, "v_lam": 0.0}

        if proj_name == "q_proj":
            layer_data[layer_idx]["q_align"] = float(alignments[key])
            layer_data[layer_idx]["q_lam"] = float(lambda_state.get(key, 0.0))
        elif proj_name == "v_proj":
            layer_data[layer_idx]["v_align"] = float(alignments[key])
            layer_data[layer_idx]["v_lam"] = float(lambda_state.get(key, 0.0))

    # Sort by max alignment descending
    sorted_layers = sorted(
        layer_data.items(),
        key=lambda x: max(x[1]["q_align"], x[1]["v_align"]),
        reverse=True,
    )

    primary_refusal = smoothed_refusal_rate if smoothed_refusal_rate is not None else refusal_rate

    lines = [
        "=== LoRA-SafeLoop Agent Observation ===",
        f"Training Progress : Step {step} / {total_steps} ({pct_done:.1f}%)",
        f"Fine-tuning Task  : {task}",
        "",
        "SAFETY METRICS (use SMOOTHED for decisions, raw is noisy +/-10%):",
        f"  Target Refusal Floor  : {target_floor:.4f} ({target_floor*100:.1f}%) — STAY ABOVE THIS",
    ]
    if smoothed_refusal_rate is not None:
        lines.append(f"  * Smoothed Refusal (3-step avg): {smoothed_refusal_rate:.4f} ({smoothed_refusal_rate*100:.1f}%) <- USE THIS FOR DECISIONS")
    lines += [
        f"  Raw Refusal (noisy)   : {refusal_rate:.4f} ({refusal_rate*100:.1f}%)",
        f"  Baseline              : {baseline_refusal_rate:.4f} ({baseline_refusal_rate*100:.1f}%)",
        f"  Raw Change vs Last    : {step_change:+.4f} ({step_change*100:+.1f}%)",
    ]

    if primary_refusal < target_floor:
        lines.append(f"  ALERT: Smoothed refusal {primary_refusal:.2f} is BELOW target floor {target_floor:.2f}. Raise λ aggressively.")
    elif primary_refusal > target_floor + 0.05:
        lines.append(f"  SAFE: Smoothed refusal {primary_refusal:.2f} is above target floor. REDUCE λ on low-alignment layers.")

    if declining_steps >= 2:
        lines.append(f"  TREND: Refusal has been DECLINING for {declining_steps} of the last {len(recent_records)} steps.")

    lines += [
        "",
        "TASK METRICS:",
        f"  {metric_name}: {task_metric:.4f}",
        "",
        f"TOP {min(top_k_layers, len(sorted_layers))} LAYERS BY SUBSPACE ALIGNMENT",
        "(Per-projection: q=q_proj, v=v_proj. Higher alignment = needs stronger λ):",
    ]

    for layer_idx, data in sorted_layers[:top_k_layers]:
        max_align = max(data["q_align"], data["v_align"])
        avg_lam = (data["q_lam"] + data["v_lam"]) / 2

        if max_align > 0.10:
            flag = "HIGH — needs λ>=0.7"
        elif max_align > 0.05:
            flag = "MED — needs λ>=0.5"
        elif max_align > 0.02:
            flag = "LOW — λ=0.2-0.3 ok"
        else:
            flag = "MIN — λ=0.0 ok"

        lines.append(
            f"  Layer {layer_idx:2d} | q_align={data['q_align']:.4f} v_align={data['v_align']:.4f} "
            f"| λ={avg_lam:.3f}  [{flag}]"
        )

    # Compact full lambda state (per-layer average)
    layer_lam_avg = {}
    for key, lam in lambda_state.items():
        layer_idx = int(str(key).split(".")[0])
        if layer_idx not in layer_lam_avg:
            layer_lam_avg[layer_idx] = []
        layer_lam_avg[layer_idx].append(lam)
    lambda_compact = " ".join(
        f"{l}:{sum(vs)/len(vs):.2f}"
        for l, vs in sorted(layer_lam_avg.items())
    )

    lines += [
        "",
        f"FULL λ STATE (per-layer avg, after 2% auto-decay): [{lambda_compact}]",
        "",
        "REFLEXION MEMORY (last completed decisions and their outcomes):",
        reflexion_memory.format_for_agent(),
        "",
        "REMINDERS:",
        "- Use SMOOTHED refusal for decisions, not raw.",
        "- λ auto-decays 2% each step. You must actively set layers you want to keep high.",
        "- DUAL OBJECTIVE: maintain safety AND maximize task performance.",
        "- If safety is stable: LOWER λ on low-alignment layers to improve task learning.",
        "- Make small adjustments (+/-0.05 to +/-0.15). Large jumps destabilize training.",
    ]

    return "\n".join(lines)


# ===========================================================================
# Agent API Call
# ===========================================================================

def call_groq_agent(
    observation: str,
    api_key: str,
    model_id: str = "llama-3.3-70b-versatile",
    max_retries: int = 3,
    base_retry_delay: float = 5.0,
) -> Optional[str]:
    """
    Calls the Groq API with the observation and returns the raw response text.
    Uses exponential backoff on rate-limit errors.
    """
    client = groq.Groq(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": observation}
                ],
                temperature=0.1,
                max_tokens=512,
            )
            text = response.choices[0].message.content
            logger.info(f"Agent API call succeeded (attempt {attempt + 1}).")
            return text

        except groq.RateLimitError:
            wait = base_retry_delay * (2 ** attempt)
            logger.warning(
                f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                f"Waiting {wait:.0f}s..."
            )
            time.sleep(wait)

        except groq.APIConnectionError as e:
            wait = base_retry_delay * (attempt + 1)
            logger.warning(
                f"Connection error (attempt {attempt + 1}/{max_retries}): {e}. "
                f"Waiting {wait:.0f}s..."
            )
            time.sleep(wait)

        except groq.APIStatusError as e:
            logger.error(f"API status error {e.status_code}: {e.message}")
            if 400 <= e.status_code < 500:
                break
            time.sleep(base_retry_delay)

        except Exception as e:
            logger.error(f"Unexpected error calling Groq API: {e}")
            break

    logger.error(
        f"All {max_retries} API call attempts failed. "
        "Falling back to current λ values."
    )
    return None


# ===========================================================================
# Response Parser
# ===========================================================================

def parse_agent_response(
    response_text: Optional[str],
    current_lambda_state: dict,
    valid_layer_ids: list,
    failure_log_path: Optional[str] = None,
    alignments: Optional[dict] = None,
) -> dict:
    """
    Parses the agent's JSON response with a full fallback chain.

    v4.1: Agent outputs per-LAYER keys (int). Parser expands them to
    per-projection string keys ("24.q_proj", "24.v_proj") for the
    constraint applier.

    Args:
        current_lambda_state: Current per-projection λ values (string-keyed).
        valid_layer_ids: List of valid integer layer indices.
        (other args unchanged)

    Returns:
        dict with keys:
            "layer_constraints"  : dict string key -> float (full updated state)
            "rationale"          : str
            "predicted_outcome"  : str
            "fallback_used"      : bool
    """
    fallback_result = {
        "layer_constraints":  dict(current_lambda_state),
        "rationale":          "[FALLBACK] Keeping current λ values unchanged.",
        "predicted_outcome":  "Unknown (fallback used).",
        "fallback_used":      True,
    }

    # Case 1: API returned nothing
    if response_text is None:
        logger.warning("Agent response is None — using fallback.")
        _log_failure(failure_log_path, "null_response", None)
        return fallback_result

    # Case 2: Extract JSON
    text = response_text.strip()

    if "```json" in text:
        s = text.find("```json") + 7
        e = text.find("```", s)
        text = text[s:e].strip() if e != -1 else text[s:].strip()
    elif "```" in text:
        s = text.find("```") + 3
        e = text.find("```", s)
        text = text[s:e].strip() if e != -1 else text[s:].strip()

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        text = text[brace_start:brace_end + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}\nSnippet: {response_text[:300]}")
        _log_failure(failure_log_path, "json_parse_error", response_text)
        return fallback_result

    if not isinstance(parsed, dict):
        logger.warning("Parsed JSON is not a dict — using fallback.")
        _log_failure(failure_log_path, "not_a_dict", response_text)
        return fallback_result

    # Extract layer_constraints
    raw_constraints = parsed.get("layer_constraints", {})
    if not isinstance(raw_constraints, dict):
        logger.warning(f"layer_constraints is {type(raw_constraints).__name__} — fallback.")
        _log_failure(failure_log_path, "invalid_constraints_type", response_text)
        return fallback_result

    # Start from current state
    new_lambda_state = dict(current_lambda_state)

    for key, val in raw_constraints.items():
        # Agent sends per-layer int keys; expand to per-projection string keys
        try:
            layer_idx = int(key)
        except (ValueError, TypeError):
            logger.warning(f"Invalid layer key '{key}' — skipping.")
            continue

        if layer_idx not in valid_layer_ids:
            logger.warning(f"Layer {layer_idx} not in model — skipping.")
            continue

        try:
            lam = float(val)
        except (ValueError, TypeError):
            logger.warning(f"Invalid λ value '{val}' for layer {layer_idx} — skipping.")
            continue

        clamped = max(0.0, min(1.0, lam))
        if abs(clamped - lam) > 1e-6:
            logger.warning(f"λ={lam:.4f} for layer {layer_idx} clamped to {clamped:.4f}.")

        # Expand to both projections
        for proj_name in ("q_proj", "v_proj"):
            full_key = f"{layer_idx}.{proj_name}"
            if full_key in new_lambda_state:
                new_lambda_state[full_key] = clamped

    # Extract text fields
    rationale = str(parsed.get("rationale", "No rationale provided."))[:300]
    predicted_outcome = str(parsed.get("predicted_outcome", "Not specified."))[:200]

    logger.info(f"Agent decision parsed: {len(raw_constraints)} layers updated.")

    return {
        "layer_constraints":  new_lambda_state,
        "rationale":          rationale,
        "predicted_outcome":  predicted_outcome,
        "fallback_used":      False,
    }


# ===========================================================================
# Failure logger
# ===========================================================================

def _log_failure(
    log_path: Optional[str],
    reason: str,
    response: Optional[str],
):
    """Append one row to the agent failures CSV."""
    if log_path is None:
        return
    try:
        path = Path(log_path)
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["reason", "response_snippet"])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "reason":           reason,
                "response_snippet": (response or "")[:300],
            })
    except Exception as e:
        logger.error(f"Failed to write agent failure log: {e}")