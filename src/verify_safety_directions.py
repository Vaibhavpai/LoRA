"""
verify_safety_directions.py — Phase 3, Step 3.2
================================================
Verifies that the safety directions extracted in Step 3.1 are meaningful by
comparing subspace alignment scores for:

  A. The BASE model (before any fine-tuning)
  B. The VANILLA LORA model (after fine-tuning — from Phase 2)

Expected result:
  - Base model:          alignment near 0 (no drift, no LoRA → AttributeError handled)
  - Fine-tuned model:    alignment clearly above 0 (weight drift detected)

Audit changelog (Phase 3 audit):
  - Added CSV export of all per-layer alignments (base + fine-tuned).
  - Added summary statistics: mean, std, max alignment across layers.
  - Added top-10 layers ranked by alignment delta (fine-tuned − base).
  - Added multi-direction reporting using compute_subspace_alignment_full.
  - Official paper metric (compute_subspace_alignment → |v1·u1|) UNCHANGED.
  - All diagnostic additions are clearly labelled; they do not affect the
    numbers reported in the paper.

Usage:
  python src/verify_safety_directions.py
  python src/verify_safety_directions.py --task gsm8k --seed 42
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.metrics import compute_subspace_alignment, compute_subspace_alignment_full

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================

def load_safety_directions(models_dir: Path) -> tuple[dict, dict]:
    """Load the saved safety directions and metadata."""
    directions_path = models_dir / "safety_directions.pt"
    if not directions_path.exists():
        raise FileNotFoundError(
            f"safety_directions.pt not found at {directions_path}.\n"
            f"Run 'python src/subspace_extraction.py' first."
        )
    payload = torch.load(directions_path, map_location="cpu")
    return payload["directions"], payload["metadata"]


def compute_mean_alignment(alignments: dict[int, float]) -> float:
    return sum(alignments.values()) / len(alignments) if alignments else 0.0


def compute_std_alignment(alignments: dict[int, float]) -> float:
    """Population standard deviation of alignment scores across layers."""
    if not alignments:
        return 0.0
    mean = compute_mean_alignment(alignments)
    variance = sum((v - mean) ** 2 for v in alignments.values()) / len(alignments)
    return variance ** 0.5


def _alignments_to_df(alignments: dict[int, float], label: str) -> pd.DataFrame:
    rows = [{"layer": l, f"alignment_{label}": v} for l, v in sorted(alignments.items())]
    return pd.DataFrame(rows)


# ===========================================================================
# Per-direction diagnostic table
# ===========================================================================

def print_multi_direction_table(
    full_results: dict[int, dict],
    label: str,
    num_sample_layers: int = 5,
) -> None:
    """
    Prints a diagnostic table showing alignment with each of the k directions
    for a sample of layers. The 'official' column matches compute_subspace_alignment.

    This is a diagnostic output only — it does not change the paper metric.
    """
    if not full_results:
        return

    # Pick k from first available layer that has LoRA computed
    k = 0
    for r in full_results.values():
        if r["delta_W_computed"]:
            k = len(r["per_dir"])
            break

    if k == 0:
        logger.info(f"  [{label}] No LoRA weight deltas found — base model (expected all 0.0).")
        return

    print(f"\n--- Multi-direction alignment table ({label}, k={k} directions) ---")
    header = f"{'Layer':<8} {'Official':>10}"
    for j in range(k):
        header += f"  {'Dir'+str(j+1):>8}"
    header += f"  {'Mean_k':>8}  {'Max_k':>8}"
    print(header)
    print("-" * len(header))

    layer_ids = sorted(full_results.keys())
    step = max(1, len(layer_ids) // num_sample_layers)
    sample_layers = layer_ids[::step][:num_sample_layers]

    for l in sample_layers:
        r = full_results[l]
        row = f"Layer {l:<3} {r['official']:>10.4f}"
        for val in r["per_dir"]:
            row += f"  {val:>8.4f}"
        row += f"  {r['mean_k']:>8.4f}  {r['max_k']:>8.4f}"
        print(row)

    # Summary across all layers
    officials = [r["official"] for r in full_results.values() if r["delta_W_computed"]]
    means_k   = [r["mean_k"]   for r in full_results.values() if r["delta_W_computed"]]
    maxes_k   = [r["max_k"]    for r in full_results.values() if r["delta_W_computed"]]

    if officials:
        print("-" * len(header))
        print(
            f"{'All-layer mean:':<20} official={sum(officials)/len(officials):.4f}  "
            f"mean_k={sum(means_k)/len(means_k):.4f}  "
            f"max_k={max(maxes_k):.4f}"
        )


# ===========================================================================
# Top-N layers by alignment delta
# ===========================================================================

def print_top_n_layers(
    base_alignments: dict[int, float],
    ft_alignments: dict[int, float],
    n: int = 10,
) -> None:
    """Prints top-N layers ranked by alignment increase (fine-tuned − base)."""
    deltas = {
        l: ft_alignments.get(l, 0.0) - base_alignments.get(l, 0.0)
        for l in ft_alignments
    }
    ranked = sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:n]

    print(f"\n--- Top {n} layers by alignment increase (fine-tuned − base) ---")
    print(f"{'Rank':<6} {'Layer':<8} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print("-" * 50)
    for rank, (l, delta) in enumerate(ranked, 1):
        b = base_alignments.get(l, 0.0)
        f = ft_alignments.get(l, 0.0)
        flag = "  ← HIGH DRIFT" if delta > 0.1 else ""
        print(f"{rank:<6} Layer {l:<3} {b:>10.4f} {f:>12.4f} {delta:>+10.4f}{flag}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Verify safety directions: base vs fine-tuned")
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "alpaca"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--layers", type=int, default=5,
                        help="How many layers to print in sampled tables")
    parser.add_argument("--top_n", type=int, default=10,
                        help="How many top-drift layers to rank in the diagnostic table")
    parser.add_argument("--export_csv", action="store_true",
                        help="Export per-layer alignment scores to results/alignment_scores.csv")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 3 Step 3.2: Safety Direction Verification")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Load safety directions
    # -----------------------------------------------------------------------
    logger.info("\nLoading safety directions...")
    safety_directions, meta = load_safety_directions(models_dir)
    num_layers = meta["num_layers"]
    d_model = meta["d_model"]
    k = meta["k"]
    safe_prompt_type = meta.get("safe_prompt_type", "unknown")
    logger.info(f"  Directions loaded: {num_layers} layers, d_model={d_model}, k={k}")
    logger.info(f"  Safe prompt type used during extraction: {safe_prompt_type}")

    if safe_prompt_type == "alpaca_random_seed99":
        logger.warning(
            "⚠️  Safe prompts used for extraction were random Alpaca instructions, "
            "NOT semantic paraphrases of AdvBench. Alignment scores may be degraded. "
            "See PHASE3_AUDIT_REPORT.md for the fix."
        )

    model_id = meta["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------------
    # A. Base model alignment (expected ~0.0 — no LoRA exists yet)
    # -----------------------------------------------------------------------
    logger.info(f"\nLoading BASE model: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    logger.info("Computing OFFICIAL alignment for BASE model (no LoRA, expected 0.0 all layers)...")
    base_alignments = compute_subspace_alignment(base_model, safety_directions)
    mean_base = compute_mean_alignment(base_alignments)
    std_base  = compute_std_alignment(base_alignments)
    max_base  = max(base_alignments.values())
    max_base_layer = max(base_alignments, key=lambda l: base_alignments[l])
    logger.info(f"  Base — mean: {mean_base:.4f}  std: {std_base:.4f}  max: {max_base:.4f} (layer {max_base_layer})")

    # Diagnostic: multi-direction (base should all be 0 since no LoRA)
    base_full = compute_subspace_alignment_full(base_model, safety_directions)
    del base_model

    # -----------------------------------------------------------------------
    # B. Fine-tuned model alignment (expected > 0)
    # -----------------------------------------------------------------------
    adapter_path = models_dir / f"vanilla_{args.task}_seed{args.seed}"
    if not adapter_path.exists():
        logger.error(
            f"Fine-tuned adapter not found at {adapter_path}.\n"
            f"Run: python experiments/train_vanilla.py --task {args.task} --seed {args.seed}"
        )
        return

    logger.info(f"\nLoading FINE-TUNED model from: {adapter_path}")
    base_for_peft = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    finetuned_model = PeftModel.from_pretrained(base_for_peft, adapter_path)
    finetuned_model.to(args.device)

    logger.info("Computing OFFICIAL alignment for FINE-TUNED model...")
    finetuned_alignments = compute_subspace_alignment(finetuned_model, safety_directions)
    mean_ft = compute_mean_alignment(finetuned_alignments)
    std_ft  = compute_std_alignment(finetuned_alignments)
    max_ft  = max(finetuned_alignments.values())
    max_ft_layer = max(finetuned_alignments, key=lambda l: finetuned_alignments[l])
    logger.info(f"  Fine-tuned — mean: {mean_ft:.4f}  std: {std_ft:.4f}  max: {max_ft:.4f} (layer {max_ft_layer})")

    # Diagnostic: multi-direction for fine-tuned model
    finetuned_full = compute_subspace_alignment_full(finetuned_model, safety_directions)

    # -----------------------------------------------------------------------
    # Official alignment table (paper-reportable)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"OFFICIAL ALIGNMENT SUMMARY (task={args.task}, seed={args.seed})")
    print(f"  Metric: Alignment = |v1 · u1|   (top right singular vectors)")
    print("=" * 65)
    print(f"{'Layer':<10} {'Base':>12} {'Fine-tuned':>14} {'Difference':>14}")
    print("-" * 54)

    layer_ids = sorted(base_alignments.keys())
    step = max(1, len(layer_ids) // args.layers)
    sample_layers = layer_ids[::step][: args.layers]

    for l in sample_layers:
        b = base_alignments.get(l, 0.0)
        f = finetuned_alignments.get(l, 0.0)
        diff = f - b
        flag = "  ← DRIFT" if diff > 0.05 else ""
        print(f"Layer {l:<5} {b:>12.4f} {f:>14.4f} {diff:>+14.4f}{flag}")

    print("-" * 54)
    print(f"{'MEAN':<10} {mean_base:>12.4f} {mean_ft:>14.4f} {mean_ft - mean_base:>+14.4f}")
    print(f"{'STD':<10} {std_base:>12.4f} {std_ft:>14.4f} {'':>14}")
    print(f"{'MAX':<10} {max_base:>12.4f} {max_ft:>14.4f} {'':>14}")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # Diagnostic: top-N layers by drift
    # -----------------------------------------------------------------------
    print_top_n_layers(base_alignments, finetuned_alignments, n=args.top_n)

    # -----------------------------------------------------------------------
    # Diagnostic: multi-direction tables (fine-tuned only — base is all 0)
    # -----------------------------------------------------------------------
    print_multi_direction_table(finetuned_full, label=f"fine-tuned ({args.task})", num_sample_layers=args.layers)

    # -----------------------------------------------------------------------
    # CSV export
    # -----------------------------------------------------------------------
    if args.export_csv:
        df_base = _alignments_to_df(base_alignments, "base")
        df_ft   = _alignments_to_df(finetuned_alignments, "finetuned")
        df_merged = df_base.merge(df_ft, on="layer")
        df_merged["delta"] = df_merged["alignment_finetuned"] - df_merged["alignment_base"]

        # Append per-direction diagnostics for fine-tuned model
        for j in range(k):
            df_merged[f"finetuned_dir{j+1}"] = df_merged["layer"].map(
                lambda l: finetuned_full[l]["per_dir"][j] if l in finetuned_full else 0.0
            )
        df_merged["finetuned_mean_k"] = df_merged["layer"].map(
            lambda l: finetuned_full[l]["mean_k"] if l in finetuned_full else 0.0
        )
        df_merged["finetuned_max_k"] = df_merged["layer"].map(
            lambda l: finetuned_full[l]["max_k"] if l in finetuned_full else 0.0
        )

        csv_path = results_dir / f"alignment_scores_{args.task}_seed{args.seed}.csv"
        df_merged.to_csv(csv_path, index=False)
        logger.info(f"\nAlignment scores exported to: {csv_path}")

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    print("\nINTERPRETATION:")
    delta_mean = mean_ft - mean_base
    if delta_mean > 0.05:
        print("✅ GOOD: Fine-tuned model shows higher alignment than base.")
        print("   Weight drift IS happening in the safety subspace.")
        print("   Safety directions successfully capture what changed during fine-tuning.")
        print("   You're ready to move to Phase 4 (static baselines).")
    elif delta_mean > 0.0:
        print("⚠️  WEAK: Fine-tuned model shows slightly higher alignment, but the gap is small.")
        print("   Check per-layer values in the top-N table above.")
        print("   This may be caused by the safe prompt compliance issue (random Alpaca ≠ paraphrases).")
        print("   You can proceed to Phase 4 but note this in your paper.")
    else:
        print("❌ UNEXPECTED: Fine-tuned model shows similar or lower alignment than base.")
        print("   Most likely cause: safe prompts are random Alpaca instructions, not paraphrases.")
        print("   See PHASE3_AUDIT_REPORT.md § D for a full diagnosis and § E for the fix.")


if __name__ == "__main__":
    main()