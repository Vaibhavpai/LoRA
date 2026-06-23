"""
subspace_extraction.py — Phase 3, Step 3.1
===========================================
Extracts the safety subspace directions from the BASE (unmodified) Qwen2.5-1.5B-Instruct
using contrastive activation analysis (following Arditi et al. 2024).

How it works:
  1. Run 520 harmful AdvBench prompts through the base model → collect last-token
     hidden states at every transformer layer.
  2. Run 520 safe prompts through the same model.
  3. Subtract: difference_vector = harmful_activation - safe_activation per layer.
  4. Stack all 520 difference vectors into a matrix D_l of shape [520, d_model].
  5. SVD on D_l → take the top-k RIGHT singular vectors (columns of V).
     These vectors live in activation space (R^d_model) and capture the
     "safety vs. unsafe" direction at each layer.
  6. Save to models/safety_directions.pt as: dict[layer_idx -> tensor[d_model, k]]

Audit changelog (Phase 3 audit):
  - logger.debug → logger.info for variance explained logging (was silently suppressed).
  - verify_directions replaced with verify_directions_full: now covers ALL layers,
    reports ALL k directions, and exports a separation statistics CSV.
  - verify_directions kept as a thin wrapper calling verify_directions_full for
    backward compatibility with any external callers.
  - All extraction math UNCHANGED.

⚠️  KNOWN COMPLIANCE ISSUE (flagged, not auto-fixed):
    Safe prompts are loaded from the AdvBench-Safe dataset
    using semantically matched harmful/safe pairs. The plan requires:
      "Load or construct a set of 520 safe paraphrases of AdvBench prompts"
    See PHASE3_AUDIT_REPORT.md § C for fix options.

Usage:
  python src/subspace_extraction.py
  python src/subspace_extraction.py --k 5 --batch_size 8 --device cpu
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import load_advbench, load_safe_prompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# Step 1: Collect last-token hidden states for a list of prompts
# ===========================================================================

def collect_hidden_states(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 8,
    device: str = "cpu",
) -> dict[int, torch.Tensor]:
    """
    Runs all prompts through the model and collects the LAST-TOKEN hidden state
    at every transformer layer.

    Why last-token only?
    - Prompts have different lengths, so full sequence tensors can't be stacked.
    - The last token's hidden state is the model's compressed representation of
      the entire input — this is exactly what Arditi et al. 2024 use.
    - We use LEFT padding so that position -1 is always the last real content
      token (not a pad token).

    Args:
        model      : Loaded HuggingFace model (base, no LoRA).
        tokenizer  : Matching tokenizer.
        prompts    : List of instruction strings.
        batch_size : How many prompts to process at once. Keep low on CPU.
        device     : 'cpu' or 'cuda'.

    Returns:
        dict mapping layer_idx -> tensor of shape [N, d_model]
        where N = len(prompts), d_model = hidden size of the model.
    """
    model.eval()
    num_layers = model.config.num_hidden_layers  # 28 for Qwen2.5-1.5B

    layer_states: dict[int, list[torch.Tensor]] = {l: [] for l in range(num_layers)}

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Collecting activations"):
            batch_prompts = prompts[i: i + batch_size]

            formatted = []
            for p in batch_prompts:
                msg = [{"role": "user", "content": p}]
                formatted.append(
                    tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                )

            # Left-padding: last real token is always at position index -1
            tokenizer.padding_side = "left"
            inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(device)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            # outputs.hidden_states: tuple of (num_layers + 1) tensors
            # Index 0 = embedding layer; indices 1..num_layers = transformer layers
            hidden_states = outputs.hidden_states

            for layer_idx in range(num_layers):
                layer_out = hidden_states[layer_idx + 1]  # [batch, seq_len, d_model]
                last_token = layer_out[:, -1, :].cpu()    # [batch, d_model]
                layer_states[layer_idx].append(last_token)

    result = {l: torch.cat(tensors, dim=0) for l, tensors in layer_states.items()}
    logger.info(f"Collected hidden states for {len(prompts)} prompts across {num_layers} layers.")
    logger.info(f"Shape per layer: {result[0].shape}")
    return result


# ===========================================================================
# Step 2: Compute difference matrix and extract safety directions via SVD
# ===========================================================================

def extract_safety_directions(
    harmful_states: dict[int, torch.Tensor],
    safe_states: dict[int, torch.Tensor],
    k: int = 5,
) -> dict[int, torch.Tensor]:
    """
    For each transformer layer, computes the top-k safety directions using SVD
    on the contrastive difference matrix.

    Math (UNCHANGED):
        D_l = harmful_states_l - safe_states_l   shape: [N, d_model]
        D_l = U @ S @ Vh                          SVD decomposition
        safety_directions_l = Vh[:k].T            shape: [d_model, k]

    The ROWS of Vh are the right singular vectors in R^d_model.
    We take the top-k rows of Vh (first k rows), then transpose to get [d_model, k].
    These k columns are the "safety directions" at layer l.

    Args:
        harmful_states : Layer -> [N, d_model] hidden states for harmful prompts.
        safe_states    : Layer -> [N, d_model] hidden states for safe prompts.
        k              : Number of top directions to keep (committed to 5).

    Returns:
        dict mapping layer_idx -> tensor of shape [d_model, k].
    """
    assert harmful_states.keys() == safe_states.keys(), "Layer mismatch between prompt sets"
    safety_directions = {}

    for layer_idx in tqdm(harmful_states.keys(), desc="Computing SVD per layer"):
        H = harmful_states[layer_idx].to(torch.float32)  # [N, d_model]
        S = safe_states[layer_idx].to(torch.float32)     # [N, d_model]

        # Difference matrix: D_l = H - S,  shape [N, d_model]
        D = H - S

        # Economy SVD: Vh shape [min(N, d_model), d_model]
        _, singular_values, Vh = torch.linalg.svd(D, full_matrices=False)

        # Top-k right singular vectors: rows of Vh, transposed → [d_model, k]
        top_k_directions = Vh[:k].T  # shape: [d_model, k]
        safety_directions[layer_idx] = top_k_directions

        # AUDIT FIX: changed logger.debug → logger.info so variance info is visible
        # by default (was silently suppressed at DEBUG level in the original code).
        total_var = (singular_values ** 2).sum().item()
        top_k_var = (singular_values[:k] ** 2).sum().item()
        explained = top_k_var / total_var * 100 if total_var > 0 else 0.0
        logger.info(
            f"Layer {layer_idx:2d} | top-{k} singular values explain {explained:.1f}% of variance "
            f"| top singular value: {singular_values[0].item():.4f}"
        )

    return safety_directions


# ===========================================================================
# Step 3: Verify directions — FULL VERSION (all layers, all k directions)
# ===========================================================================

def verify_directions_full(
    safety_directions: dict[int, torch.Tensor],
    harmful_states: dict[int, torch.Tensor],
    safe_states: dict[int, torch.Tensor],
    output_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Full verification across ALL layers and ALL k directions.

    For each layer l and each safety direction j:
        - Computes mean |projection| of harmful activations onto direction j.
        - Computes mean |projection| of safe activations onto direction j.
        - Reports separation = harmful_proj - safe_proj.

    A positive separation for direction 1 (the top singular vector) confirms
    the directions meaningfully separate harmful from safe content.

    Args:
        safety_directions : Layer -> [d_model, k] extracted directions.
        harmful_states    : Layer -> [N, d_model] harmful activations.
        safe_states       : Layer -> [N, d_model] safe activations.
        output_csv_path   : If provided, exports full statistics to this CSV path.

    Returns:
        pd.DataFrame with columns:
            layer, dir_idx, harm_proj_mean, safe_proj_mean, separation,
            harm_proj_std, safe_proj_std
    """
    logger.info("--- Full verification of safety directions (all layers, all k directions) ---")

    records = []

    for l in sorted(harmful_states.keys()):
        directions = safety_directions[l]      # [d_model, k]
        k = directions.shape[1]

        H = harmful_states[l].to(torch.float32)  # [N, d_model]
        S = safe_states[l].to(torch.float32)     # [N, d_model]

        for j in range(k):
            direction = directions[:, j]  # [d_model]

            # Project each example onto this direction
            proj_harm = H @ direction   # [N]  (signed projections)
            proj_safe = S @ direction   # [N]

            harm_mean  = proj_harm.abs().mean().item()
            harm_std   = proj_harm.abs().std().item()
            safe_mean  = proj_safe.abs().mean().item()
            safe_std   = proj_safe.abs().std().item()
            separation = harm_mean - safe_mean

            records.append({
                "layer": l,
                "dir_idx": j + 1,   # 1-indexed for readability
                "harm_proj_mean": harm_mean,
                "harm_proj_std": harm_std,
                "safe_proj_mean": safe_mean,
                "safe_proj_std": safe_std,
                "separation": separation,
                "separation_positive": separation > 0.0,
            })

    df = pd.DataFrame(records)

    if output_csv_path is not None:
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Separation statistics saved to: {output_csv_path}")

    # Print a summary for direction 1 (the top safety direction — used in the paper metric)
    dir1 = df[df["dir_idx"] == 1].copy()
    n_positive = dir1["separation_positive"].sum()
    logger.info(
        f"\nDirection 1 (paper metric) summary across {len(dir1)} layers:\n"
        f"  Positive separation (harm > safe): {n_positive}/{len(dir1)} layers\n"
        f"  Mean separation: {dir1['separation'].mean():.4f}\n"
        f"  Max  separation: {dir1['separation'].max():.4f}  (layer {dir1.loc[dir1['separation'].idxmax(), 'layer']})\n"
        f"  Min  separation: {dir1['separation'].min():.4f}  (layer {dir1.loc[dir1['separation'].idxmin(), 'layer']})"
    )

    return df


def verify_directions(
    safety_directions: dict[int, torch.Tensor],
    harmful_states: dict[int, torch.Tensor],
    safe_states: dict[int, torch.Tensor],
    num_layers_to_check: int = 5,
) -> None:
    """
    Original verify_directions — kept for backward compatibility.
    Now delegates to verify_directions_full and prints a subset of layers.

    AUDIT NOTE: The original implementation only checked num_layers_to_check
    (default 5) sampled layers, hiding failures in unchecked layers. This
    wrapper still accepts the same signature but internally runs the full check.
    """
    df = verify_directions_full(safety_directions, harmful_states, safe_states)

    logger.info("--- Sampled layer verification (Direction 1 only) ---")

    total_layers = len(safety_directions)
    step = max(1, total_layers // num_layers_to_check)
    check_layers = sorted(safety_directions.keys())[::step][:num_layers_to_check]

    for l in check_layers:
        row = df[(df["layer"] == l) & (df["dir_idx"] == 1)]
        if row.empty:
            continue
        harm = row["harm_proj_mean"].iloc[0]
        safe = row["safe_proj_mean"].iloc[0]
        logger.info(
            f"Layer {l:2d} | mean |projection| — harmful: {harm:.4f}, safe: {safe:.4f}, "
            f"separation: {harm - safe:+.4f}"
        )

    logger.info(
        "Verification complete. Positive separation means directions capture "
        "the harmful/safe distinction."
    )


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Safety Subspace Extraction")
    parser.add_argument("--k", type=int, default=5, help="Number of top safety directions to keep per layer")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for activation collection")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--num_prompts", type=int, default=520,
                        help="Number of harmful/safe pairs to use (max 520)")
    parser.add_argument("--export_separation_csv", action="store_true",
                        help="Export full per-layer/per-direction separation stats to CSV")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    output_path = models_dir / "safety_directions.pt"

    logger.info("=" * 60)
    logger.info("Phase 3: Safety Subspace Extraction")
    logger.info("=" * 60)
    logger.info(f"  k (directions per layer) : {args.k}")
    logger.info(f"  batch_size               : {args.batch_size}")
    logger.info(f"  device                   : {args.device}")
    logger.info(f"  num_prompts              : {args.num_prompts}")

    # -----------------------------------------------------------------------
    # COMPLIANCE WARNING: safe prompts are not semantic paraphrases
    # -----------------------------------------------------------------------
    

    # -----------------------------------------------------------------------
    # 1. Load datasets
    # -----------------------------------------------------------------------
    logger.info("\nLoading datasets...")
    harmful_prompts = load_advbench()[: args.num_prompts]
    safe_prompts = load_safe_prompts(num_examples=args.num_prompts)

    assert len(harmful_prompts) == len(safe_prompts), (
        f"Prompt count mismatch: {len(harmful_prompts)} harmful vs {len(safe_prompts)} safe."
    )
    logger.info(f"Loaded {len(harmful_prompts)} harmful and {len(safe_prompts)} safe prompts.")

    # -----------------------------------------------------------------------
    # 2. Load BASE model (no LoRA, no fine-tuning)
    # -----------------------------------------------------------------------
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    logger.info(f"\nLoading base model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()

    logger.info(f"Model loaded. Layers: {model.config.num_hidden_layers}, d_model: {model.config.hidden_size}")

    # -----------------------------------------------------------------------
    # 3. Collect hidden states
    # -----------------------------------------------------------------------
    logger.info("\nCollecting hidden states for HARMFUL prompts...")
    harmful_states = collect_hidden_states(
        model, tokenizer, harmful_prompts,
        batch_size=args.batch_size, device=args.device
    )

    logger.info("\nCollecting hidden states for SAFE prompts...")
    safe_states = collect_hidden_states(
        model, tokenizer, safe_prompts,
        batch_size=args.batch_size, device=args.device
    )

    # -----------------------------------------------------------------------
    # 4. Extract safety directions via SVD
    # -----------------------------------------------------------------------
    logger.info(f"\nComputing top-{args.k} safety directions per layer via SVD...")
    safety_directions = extract_safety_directions(harmful_states, safe_states, k=args.k)

    # -----------------------------------------------------------------------
    # 5. Full verification of all layers and all k directions
    # -----------------------------------------------------------------------
    sep_csv_path = results_dir / "separation_statistics.csv" if args.export_separation_csv else None
    verify_directions_full(safety_directions, harmful_states, safe_states, output_csv_path=sep_csv_path)

    # -----------------------------------------------------------------------
    # 6. Save to disk
    # -----------------------------------------------------------------------
    save_payload = {
        "directions": safety_directions,
        "metadata": {
            "model_id": model_id,
            "k": args.k,
            "num_prompts": len(harmful_prompts),
            "num_layers": model.config.num_hidden_layers,
            "d_model": model.config.hidden_size,
            "safe_prompt_type": "advbench_safe",# flagged: not true paraphrases
        }
    }
    torch.save(save_payload, output_path)
    logger.info(f"\nSaved safety directions to: {output_path}")

    # Shape assertion
    for l, directions in safety_directions.items():
        assert directions.shape == (model.config.hidden_size, args.k), (
            f"Layer {l}: expected shape ({model.config.hidden_size}, {args.k}), got {directions.shape}"
        )
    logger.info(f"Shape check passed: all layers have directions of shape [{model.config.hidden_size}, {args.k}]")

    logger.info("\n" + "=" * 60)
    logger.info("Phase 3 complete.")
    logger.info("Next: run Phase 3 Step 3.2 — python src/verify_safety_directions.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()