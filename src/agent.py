"""
agent.py — Phase 6, Steps 6.2–6.3
====================================
LLM-based training controller for LoRA-SafeLoop.

Components:
  - format_observation()     : builds the structured prompt from metrics
  - call_groq_agent()      : Groq API call with retry/backoff
  - parse_agent_response()   : JSON extraction with full fallback chain

Agent contract:
  INPUT : observation string (metrics + reflexion memory)
  OUTPUT: {
    "layer_constraints": {"24": 0.8, "22": 0.7, ...},   // only layers to CHANGE
    "rationale": "...",
    "predicted_outcome": "..."
  }

Fallback chain (in order):
  1. API call fails / no key  → keep current λ values
  2. JSON malformed           → keep current λ values
  3. λ out of [0,1]           → clamp to [0,1] + log warning
  4. Unknown layer ID         → ignore that key + log warning
All failures logged to agent_failures.csv for paper analysis.
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

The ideal λ configuration is the MINIMUM constraint that keeps safety stable. Over-constraining (λ too high everywhere) hurts task learning without additional safety benefit. Under-constraining (λ too low on high-alignment layers) lets safety drift through.

HOW GRADIENT CONSTRAINT WORKS:
- Each transformer layer (0–27) has a constraint strength λ ∈ [0.0, 1.0].
- λ = 0.0: NO constraint — maximum task learning, no safety protection
- λ = 0.3: light constraint — good for low-alignment layers
- λ = 0.5: moderate constraint
- λ = 0.7: strong constraint — use for high-alignment layers showing drift
- λ = 1.0: maximum constraint — layer is fully projected out of safety subspace
- Constraint formula: ΔW_safe = ΔW - λ × (ΔW @ P)
- Higher subspace alignment score = that layer drifts more = needs higher λ
- IMPORTANT: λ values AUTO-DECAY by 2% each checkpoint toward 0. If you don't actively set a layer's λ, it gradually returns toward 0. This is intentional — you must justify maintaining high constraints.
- NOTE: λ is capped at 0.90 for most layers. Only the top-5 layers by alignment score can reach up to 1.0.

STRATEGY PRINCIPLES:
1. USE SMOOTHED REFUSAL (3-step average) for decisions, NOT raw refusal. Raw refusal has ±10% noise from small sample evaluation.
2. TARGET LAYERS SELECTIVELY: The agent's advantage over static methods is PRECISION. Set λ=0 on layers with near-zero alignment, and concentrate constraint on the few layers with high alignment. This maximizes task learning while protecting safety.
3. ACTIVELY REDUCE λ WHEN SAFE: If smoothed refusal is ABOVE the target floor, you MUST actively lower λ on layers with low/medium alignment to boost task learning. This is your KEY advantage over static controllers that only increase constraints.
4. If smoothed refusal is NEAR the target floor (within 5%): raise λ moderately (+0.10 to +0.20) on the top-5 alignment layers.
5. If smoothed refusal is BELOW the target floor: raise λ aggressively on the top-10 alignment layers (to 0.7-0.9).
6. Learn from reflexion memory: if raising λ uniformly produced DEGRADATION (safety got worse despite higher λ), the drift is in directions P doesn't capture, or you constrained the wrong layers.
7. Make PROPORTIONAL adjustments. Layers with alignment >0.1 should have higher λ. Layers with alignment <0.02 can stay at λ=0.

CRITICAL RULES:
- You only need to specify layers you want to CHANGE. Unspecified layers keep current λ (after auto-decay).
- λ=0 is valid and preferred for low-alignment layers when safety is stable.
- Aim for STABILITY, not perfection. Small consistent improvements beat large swings.

RESPOND WITH VALID JSON ONLY. No text outside the JSON block. Required format:
{
  "layer_constraints": {
    "LAYER_IDX_AS_STRING": LAMBDA_VALUE_AS_FLOAT,
    ...
  },
  "rationale": "1–2 sentence explanation of your decision",
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
    Builds a structured observation string for the Groq agent.

    Args:
        step: Current training step
        total_steps: Total training steps planned
        task: 'gsm8k' or 'alpaca'
        refusal_rate: RAW refusal rate just measured (not smoothed)
        prev_refusal_rate: Refusal rate at previous checkpoint
        baseline_refusal_rate: Refusal rate when tracking began (step 100)
        task_metric: Task accuracy or val loss
        metric_name: Name of the task metric
        alignments: dict layer_idx -> alignment score
        lambda_state: dict layer_idx -> current lambda
        reflexion_memory: ReflexionMemory instance
        smoothed_refusal_rate: Optional smoothed refusal (3-step moving average)
        top_k_layers: How many layers to show in detail (sorted by alignment)

    Returns:
        Formatted observation string
    """
    pct_done   = 100.0 * step / total_steps
    step_change = refusal_rate - prev_refusal_rate
    
    # Dynamically compute target window based on the dataset's initial drift characteristics
    target_floor = max(0.0, baseline_refusal_rate - 0.05)

    # Detect declining trend from reflexion memory
    recent_records = reflexion_memory.get_recent(3)
    declining_steps = 0
    for r in recent_records:
        if r.get("refusal_change", 0) < -0.02:
            declining_steps += 1
    
    # Sort layers by alignment descending
    sorted_layers = sorted(
        [(int(l), float(a)) for l, a in alignments.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    # Use smoothed refusal as the primary decision metric
    primary_refusal = smoothed_refusal_rate if smoothed_refusal_rate is not None else refusal_rate
    
    lines = [
        "=== LoRA-SafeLoop Agent Observation ===",
        f"Training Progress : Step {step} / {total_steps} ({pct_done:.1f}%)",
        f"Fine-tuning Task  : {task}",
        "",
        "SAFETY METRICS (use SMOOTHED for decisions, raw is noisy ±10%):",
        f"  Target Refusal Floor  : {target_floor:.4f} ({target_floor*100:.1f}%) — STAY ABOVE THIS",
    ]
    if smoothed_refusal_rate is not None:
        lines.append(f"  ★ Smoothed Refusal (3-step avg): {smoothed_refusal_rate:.4f} ({smoothed_refusal_rate*100:.1f}%) ← USE THIS FOR DECISIONS")
    lines += [
        f"  Raw Refusal (noisy)   : {refusal_rate:.4f} ({refusal_rate*100:.1f}%)",
        f"  Baseline              : {baseline_refusal_rate:.4f} ({baseline_refusal_rate*100:.1f}%)",
        f"  Raw Change vs Last    : {step_change:+.4f} ({step_change*100:+.1f}%)",
    ]
    
    # Add warnings based on SMOOTHED refusal, not raw
    if primary_refusal < target_floor:
        lines.append(f"  ⚠ ALERT: Smoothed refusal {primary_refusal:.2f} is BELOW target floor {target_floor:.2f}. Raise λ aggressively on high-alignment layers.")
    elif primary_refusal > target_floor + 0.05:
        lines.append(f"  ✔ SAFE: Smoothed refusal {primary_refusal:.2f} is comfortably above target floor. YOU MUST ACTIVELY REDUCE λ ON LOW-ALIGNMENT LAYERS TO IMPROVE TASK LEARNING.")
        
    if declining_steps >= 2:
        lines.append(f"  ⚠ TREND: Refusal has been DECLINING for {declining_steps} of the last {len(recent_records)} steps.")
    
    lines += [
        "",
        "TASK METRICS:",
        f"  {metric_name}: {task_metric:.4f}",
        "",
        f"TOP {min(top_k_layers, len(sorted_layers))} LAYERS BY SUBSPACE ALIGNMENT",
        "(Higher alignment = more safety drift = needs stronger constraint):",
    ]

    for layer_idx, align in sorted_layers[:top_k_layers]:
        lam = lambda_state.get(layer_idx, 0.0)

        if align > 0.10:
            flag = "⚠ HIGH — needs λ≥0.7"
        elif align > 0.05:
            flag = "↗ MED — needs λ≥0.5"
        elif align > 0.02:
            flag = "  LOW — λ=0.2-0.3 ok"
        else:
            flag = "  MIN — λ=0.0 ok"

        lines.append(
            f"  Layer {layer_idx:2d} | align={align:.4f} | λ={lam:.3f}  [{flag}]"
        )

    # Compact full lambda state
    lambda_compact = " ".join(
        f"{l}:{v:.2f}"
        for l, v in sorted(lambda_state.items(), key=lambda x: int(x[0]))
    )
    lines += [
        "",
        f"FULL λ STATE (after 2% auto-decay): [{lambda_compact}]",
        "",
        "REFLEXION MEMORY (last completed decisions and their outcomes):",
        reflexion_memory.format_for_agent(),
        "",
        "REMINDERS:",
        "- Use SMOOTHED refusal for decisions, not raw.",
        "- λ auto-decays 2% each step. You must actively set layers you want to keep high.",
        "- DUAL OBJECTIVE: maintain safety AND maximize task performance.",
        "- If safety is stable: LOWER λ on low-alignment layers to improve task learning. Do NOT just keep raising λ.",
        "- Make small adjustments (±0.05 to ±0.15). Large jumps destabilize training.",
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

    Args:
        observation: Formatted observation string
        api_key: Groq API key
        model_id: Groq model to use
        max_retries: Retry attempts before giving up
        base_retry_delay: Initial retry wait time (doubles on each attempt)

    Returns:
        Raw response text if successful, None if all retries failed
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
                temperature=0.1, # small non-zero for mild exploration while still structured
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
            # Don't retry on 4xx client errors (bad request, auth failure)
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

    Fallback hierarchy:
        1. response_text is None (API failed)     → keep current λ
        2. No valid JSON found in response         → keep current λ
        3. λ value outside [0, 1]                 → clamp to [0,1] + log warning (still use it)
        4. Layer ID not in model                   → skip that key + log warning

    Dynamic λ cap: Most layers capped at 0.90. Top-5 layers by alignment
    score can go up to 1.0. This prevents the monotonic ratchet to λ=1.0
    everywhere that kills task learning.

    All fallback events are logged to failure_log_path for paper analysis.

    Args:
        response_text: Raw text from Groq API (None if API failed)
        current_lambda_state: Current per-layer λ values to use as fallback baseline
        valid_layer_ids: List of valid integer layer indices
        failure_log_path: Optional CSV path for failure logging
        alignments: Optional dict layer_idx -> alignment score for dynamic λ cap

    Returns:
        dict with keys:
            "layer_constraints"        : dict int(layer_idx) -> float  (full updated state)
            "rationale"                : str
            "predicted_outcome"        : str
            "fallback_used"            : bool
    """
    # Default result: unchanged state
    fallback_result = {
        "layer_constraints":       dict(current_lambda_state),
        "rationale":               "[FALLBACK] Keeping current λ values unchanged.",
        "predicted_outcome":       "Unknown (fallback used — no agent decision).",
        "fallback_used":           True,
    }

    # ------------------------------------------------------------------
    # Case 1: API returned nothing
    # ------------------------------------------------------------------
    if response_text is None:
        logger.warning("Agent response is None — using fallback (no API key or all retries failed).")
        _log_failure(failure_log_path, "null_response", None)
        return fallback_result

    # ------------------------------------------------------------------
    # Case 2: Try to extract JSON from response
    # ------------------------------------------------------------------
    text = response_text.strip()

    # Handle markdown code fences
    if "```json" in text:
        s = text.find("```json") + 7
        e = text.find("```", s)
        text = text[s:e].strip() if e != -1 else text[s:].strip()
    elif "```" in text:
        s = text.find("```") + 3
        e = text.find("```", s)
        text = text[s:e].strip() if e != -1 else text[s:].strip()

    # Find outermost { ... }
    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        text = text[brace_start : brace_end + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(
            f"JSON parse failed: {e}\n"
            f"Response snippet: {response_text[:300]}"
        )
        _log_failure(failure_log_path, "json_parse_error", response_text)
        return fallback_result

    if not isinstance(parsed, dict):
        logger.warning("Parsed JSON is not a dict — using fallback.")
        _log_failure(failure_log_path, "not_a_dict", response_text)
        return fallback_result

    # ------------------------------------------------------------------
    # Extract layer_constraints
    # ------------------------------------------------------------------
    raw_constraints = parsed.get("layer_constraints", {})
    if not isinstance(raw_constraints, dict):
        logger.warning(
            f"layer_constraints is {type(raw_constraints).__name__}, not dict — fallback."
        )
        _log_failure(failure_log_path, "invalid_constraints_type", response_text)
        return fallback_result

    # Start from current state; agent only needs to specify layers it wants to change
    new_lambda_state = dict(current_lambda_state)

    for key, val in raw_constraints.items():
        # Validate layer index
        try:
            layer_idx = int(key)
        except (ValueError, TypeError):
            logger.warning(f"Invalid layer key '{key}' (not int) — skipping.")
            continue

        if layer_idx not in valid_layer_ids:
            logger.warning(
                f"Layer {layer_idx} not in model (valid: {valid_layer_ids[:5]}...) — skipping."
            )
            continue

        # Validate lambda value
        try:
            lam = float(val)
        except (ValueError, TypeError):
            logger.warning(f"Invalid λ value '{val}' for layer {layer_idx} — skipping.")
            continue

        # Case 3: clamp with dynamic cap based on alignment rank
        LAMBDA_FLOOR = 0.0
        
        # Determine cap for this layer: top-5 alignment layers get 1.0, rest get 0.90
        LAMBDA_CAP_DEFAULT = 0.90
        LAMBDA_CAP_TOP = 1.0
        TOP_N = 5
        
        layer_cap = LAMBDA_CAP_DEFAULT
        if alignments:
            sorted_by_align = sorted(
                alignments.items(),
                key=lambda x: float(x[1]),
                reverse=True,
            )
            top_layer_ids = {int(l) for l, _ in sorted_by_align[:TOP_N]}
            if layer_idx in top_layer_ids:
                layer_cap = LAMBDA_CAP_TOP
        else:
            layer_cap = LAMBDA_CAP_TOP  # no alignment info → allow full range
        
        clamped = max(LAMBDA_FLOOR, min(layer_cap, lam))
        if abs(clamped - lam) > 1e-6:
            logger.warning(
                f"λ={lam:.4f} for layer {layer_idx} clamped to {clamped:.4f} "
                f"(cap={layer_cap:.2f})."
            )
        new_lambda_state[layer_idx] = clamped

    # ------------------------------------------------------------------
    # Extract text fields
    # ------------------------------------------------------------------
    rationale        = str(parsed.get("rationale", "No rationale provided."))[:300]
    predicted_outcome = str(parsed.get("predicted_outcome", "Not specified."))[:200]

    logger.info(
        f"Agent decision parsed: "
        f"{len(raw_constraints)} layers updated."
    )

    return {
        "layer_constraints":       new_lambda_state,
        "rationale":               rationale,
        "predicted_outcome":       predicted_outcome,
        "fallback_used":           False,
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