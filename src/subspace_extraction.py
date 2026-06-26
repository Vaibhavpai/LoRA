"""
subspace_extraction.py — Phase 3 (v2, per-projection)
======================================================
Extracts safety subspace directions from the BASE Qwen2.5-1.5B-Instruct
using contrastive activation analysis (following Arditi et al. 2024).

v2 changes from original:
  - Per-projection subspaces: separate directions for q_proj and v_proj
    at each layer (56 keys total instead of 28).
  - Forward hooks capture the actual INPUT to each projection module
    (post-layernorm), not the residual stream output.
  - Mean-centering of the difference matrix before SVD.
  - L2-normalization of extracted direction vectors.
  - String key format: "{layer_idx}.{proj_name}" (e.g., "0.q_proj").

⚠️  KNOWN COMPLIANCE ISSUE (unchanged from v1):
    Safe prompts are loaded from the AdvBench-Safe dataset using generic
    safe instructions, NOT true semantic paraphrases of each harmful prompt.
    See PHASE3_AUDIT_REPORT.md for fix options.

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

# Target projections — must match LoRA target_modules
TARGET_PROJECTIONS = ("q_proj", "v_proj")


# ===========================================================================
# Step 1: Collect per-projection input activations via forward hooks
# ===========================================================================

class _InputCapture:
    """Captures the input tensor to a specific projection module."""

    def __init__(self):
        self.activations: list[torch.Tensor] = []
        self._hook = None

    def register(self, module: torch.nn.Module):
        self._hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input_args, output):
        # input_args[0] shape: [batch, seq_len, d_in]
        # Take last-token activation (left-padded, so index -1 is always real content)
        x = input_args[0].detach().float().cpu()
        last_token = x[:, -1, :]  # [batch, d_in]
        self.activations.append(last_token)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def get_stacked(self) -> Optional[torch.Tensor]:
        """Returns [N, d_in] tensor of all captured last-token activations."""
        if not self.activations:
            return None
        return torch.cat(self.activations, dim=0)

    def clear(self):
        self.activations.clear()


def collect_projection_inputs(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 8,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Runs all prompts through the model and collects the INPUT activation
    to each target projection (q_proj, v_proj) at every transformer layer.

    Uses forward hooks on the projection modules themselves, so we capture
    the post-layernorm activation that the projection actually sees.

    Args:
        model      : Loaded HuggingFace model (base, no LoRA).
        tokenizer  : Matching tokenizer.
        prompts    : List of instruction strings.
        batch_size : How many prompts to process at once.
        device     : 'cpu' or 'cuda'.

    Returns:
        dict mapping "{layer_idx}.{proj_name}" -> tensor of shape [N, d_in]
        where N = len(prompts), d_in = hidden size of the model.
    """
    model.eval()
    num_layers = model.config.num_hidden_layers

    # Register hooks on all target projections
    captures: dict[str, _InputCapture] = {}
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for proj_name in TARGET_PROJECTIONS:
            proj_module = getattr(layer.self_attn, proj_name, None)
            if proj_module is not None:
                key = f"{layer_idx}.{proj_name}"
                cap = _InputCapture()
                cap.register(proj_module)
                captures[key] = cap

    logger.info(f"Registered {len(captures)} input captures across {num_layers} layers")

    # Left-pad so last token is always at index -1
    tokenizer.padding_side = "left"

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Collecting activations"):
            batch_prompts = prompts[i:i + batch_size]

            formatted = []
            for p in batch_prompts:
                msg = [{"role": "user", "content": p}]
                formatted.append(
                    tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                )

            inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(device)
            model(**inputs, output_hidden_states=False, return_dict=True)

    # Collect results and remove hooks
    result = {}
    for key, cap in captures.items():
        stacked = cap.get_stacked()
        if stacked is not None:
            result[key] = stacked
        cap.remove()

    logger.info(f"Collected activations for {len(prompts)} prompts across {len(result)} projection points.")
    if result:
        sample_key = next(iter(result))
        logger.info(f"Shape per projection: {result[sample_key].shape}")

    return result


# ===========================================================================
# Step 2: Compute difference matrix and extract safety directions via SVD
# ===========================================================================

def extract_safety_directions(
    harmful_states: dict[str, torch.Tensor],
    safe_states: dict[str, torch.Tensor],
    k: int = 5,
) -> dict[str, torch.Tensor]:
    """
    For each projection point, computes the top-k safety directions using SVD
    on the mean-centered contrastive difference matrix.

    Math (v2 — mean-centered, L2-normalized):
        D = harmful_states - safe_states           shape: [N, d_in]
        D = D - D.mean(dim=0, keepdim=True)        mean-center
        D = U @ S @ Vh                             SVD
        directions = Vh[:k].T                      shape: [d_in, k]
        directions = directions / ||directions||   L2-normalize each column

    Args:
        harmful_states : "{layer}.{proj}" -> [N, d_in] hidden states for harmful prompts.
        safe_states    : "{layer}.{proj}" -> [N, d_in] hidden states for safe prompts.
        k              : Number of top directions to keep.

    Returns:
        dict mapping "{layer}.{proj}" -> tensor of shape [d_in, k], L2-normalized.
    """
    assert harmful_states.keys() == safe_states.keys(), (
        f"Key mismatch: {set(harmful_states.keys()) - set(safe_states.keys())}"
    )

    safety_directions = {}

    for key in tqdm(sorted(harmful_states.keys()), desc="Computing SVD per projection"):
        H = harmful_states[key].to(torch.float32)  # [N, d_in]
        S = safe_states[key].to(torch.float32)      # [N, d_in]

        # Difference matrix
        D = H - S  # [N, d_in]

        # Mean-center (v2 improvement)
        D = D - D.mean(dim=0, keepdim=True)

        # Economy SVD
        _, singular_values, Vh = torch.linalg.svd(D, full_matrices=False)

        # Top-k right singular vectors, transposed -> [d_in, k]
        top_k_directions = Vh[:k].T

        # L2-normalize each direction column (v2 improvement)
        norms = top_k_directions.norm(dim=0, keepdim=True).clamp(min=1e-8)
        top_k_directions = top_k_directions / norms

        safety_directions[key] = top_k_directions

        # Log variance explained
        total_var = (singular_values ** 2).sum().item()
        top_k_var = (singular_values[:k] ** 2).sum().item()
        explained = top_k_var / total_var * 100 if total_var > 0 else 0.0
        logger.info(
            f"{key:15s} | top-{k} explain {explained:.1f}% variance "
            f"| σ₁={singular_values[0].item():.4f}"
        )

    return safety_directions


# ===========================================================================
# Step 3: Verify directions — full version
# ===========================================================================

def verify_directions_full(
    safety_directions: dict[str, torch.Tensor],
    harmful_states: dict[str, torch.Tensor],
    safe_states: dict[str, torch.Tensor],
    output_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Full verification across ALL projection points and ALL k directions.

    For each key and each safety direction j:
        - Computes mean |projection| of harmful activations onto direction j.
        - Computes mean |projection| of safe activations onto direction j.
        - Reports separation = harmful_proj - safe_proj.

    Args:
        safety_directions : "{layer}.{proj}" -> [d_in, k] extracted directions.
        harmful_states    : "{layer}.{proj}" -> [N, d_in] harmful activations.
        safe_states       : "{layer}.{proj}" -> [N, d_in] safe activations.
        output_csv_path   : If provided, exports full statistics to this CSV path.

    Returns:
        pd.DataFrame with columns:
            key, dir_idx, harm_proj_mean, safe_proj_mean, separation, etc.
    """
    logger.info("--- Full verification of safety directions (all projections, all k directions) ---")

    records = []

    for key in sorted(safety_directions.keys()):
        directions = safety_directions[key]  # [d_in, k]
        k = directions.shape[1]

        H = harmful_states[key].to(torch.float32)
        S = safe_states[key].to(torch.float32)

        for j in range(k):
            direction = directions[:, j]  # [d_in]

            proj_harm = H @ direction  # [N]
            proj_safe = S @ direction  # [N]

            harm_mean = proj_harm.abs().mean().item()
            harm_std = proj_harm.abs().std().item()
            safe_mean = proj_safe.abs().mean().item()
            safe_std = proj_safe.abs().std().item()
            separation = harm_mean - safe_mean

            records.append({
                "key": key,
                "dir_idx": j + 1,
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

    # Summary for direction 1
    dir1 = df[df["dir_idx"] == 1].copy()
    n_positive = dir1["separation_positive"].sum()
    logger.info(
        f"\nDirection 1 (paper metric) summary across {len(dir1)} projection points:\n"
        f"  Positive separation (harm > safe): {n_positive}/{len(dir1)}\n"
        f"  Mean separation: {dir1['separation'].mean():.4f}\n"
        f"  Max  separation: {dir1['separation'].max():.4f}  ({dir1.loc[dir1['separation'].idxmax(), 'key']})\n"
        f"  Min  separation: {dir1['separation'].min():.4f}  ({dir1.loc[dir1['separation'].idxmin(), 'key']})"
    )

    return df


# ===========================================================================
# Step 4: Backward-compatible verification wrapper
# ===========================================================================

def verify_directions(
    safety_directions: dict[str, torch.Tensor],
    harmful_states: dict[str, torch.Tensor],
    safe_states: dict[str, torch.Tensor],
    num_layers_to_check: int = 5,
) -> None:
    """
    Backward-compatible wrapper — delegates to verify_directions_full
    and prints a subset of projection points.
    """
    df = verify_directions_full(safety_directions, harmful_states, safe_states)

    logger.info("--- Sampled projection verification (Direction 1 only) ---")

    all_keys = sorted(safety_directions.keys())
    step = max(1, len(all_keys) // (num_layers_to_check * 2))
    check_keys = all_keys[::step][:num_layers_to_check * 2]

    for key in check_keys:
        row = df[(df["key"] == key) & (df["dir_idx"] == 1)]
        if row.empty:
            continue
        harm = row["harm_proj_mean"].iloc[0]
        safe = row["safe_proj_mean"].iloc[0]
        logger.info(
            f"{key:15s} | mean |projection| — harmful: {harm:.4f}, safe: {safe:.4f}, "
            f"separation: {harm - safe:+.4f}"
        )

    logger.info(
        "Verification complete. Positive separation means directions capture "
        "the harmful/safe distinction."
    )


# ===========================================================================
# Utility: parse key format
# ===========================================================================

def parse_direction_key(key: str) -> tuple[int, str]:
    """Parse '{layer_idx}.{proj_name}' -> (layer_idx, proj_name)."""
    parts = key.split(".")
    return int(parts[0]), parts[1]


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Safety Subspace Extraction (v2, per-projection)")
    parser.add_argument("--k", type=int, default=5, help="Number of top safety directions to keep per projection")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for activation collection")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--num_prompts", type=int, default=520,
                        help="Number of harmful/safe pairs to use (max 520)")
    parser.add_argument("--export_separation_csv", action="store_true",
                        help="Export full per-projection/per-direction separation stats to CSV")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    output_path = models_dir / "safety_directions.pt"

    logger.info("=" * 60)
    logger.info("Phase 3: Safety Subspace Extraction (v2, per-projection)")
    logger.info("=" * 60)
    logger.info(f"  k (directions per projection) : {args.k}")
    logger.info(f"  batch_size                    : {args.batch_size}")
    logger.info(f"  device                        : {args.device}")
    logger.info(f"  num_prompts                   : {args.num_prompts}")

    # -----------------------------------------------------------------------
    # 1. Load datasets
    # -----------------------------------------------------------------------
    logger.info("\nLoading datasets...")
    harmful_prompts = load_advbench()[:args.num_prompts]
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
    # 3. Collect per-projection input activations
    # -----------------------------------------------------------------------
    logger.info("\nCollecting per-projection inputs for HARMFUL prompts...")
    harmful_states = collect_projection_inputs(
        model, tokenizer, harmful_prompts,
        batch_size=args.batch_size, device=args.device
    )

    logger.info("\nCollecting per-projection inputs for SAFE prompts...")
    safe_states = collect_projection_inputs(
        model, tokenizer, safe_prompts,
        batch_size=args.batch_size, device=args.device
    )

    # -----------------------------------------------------------------------
    # 4. Extract safety directions via SVD (mean-centered, L2-normalized)
    # -----------------------------------------------------------------------
    logger.info(f"\nComputing top-{args.k} safety directions per projection via SVD...")
    safety_directions = extract_safety_directions(harmful_states, safe_states, k=args.k)

    # -----------------------------------------------------------------------
    # 5. Full verification
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
            "safe_prompt_type": "advbench_safe",  # flagged: not true paraphrases
            "key_format": "per_projection",       # v2 marker
            "target_projections": list(TARGET_PROJECTIONS),
            "mean_centered": True,
            "l2_normalized": True,
        }
    }
    torch.save(save_payload, output_path)
    logger.info(f"\nSaved safety directions to: {output_path}")

    # Shape assertion
    for key, directions in safety_directions.items():
        assert directions.shape == (model.config.hidden_size, args.k), (
            f"{key}: expected shape ({model.config.hidden_size}, {args.k}), got {directions.shape}"
        )
    logger.info(f"Shape check passed: all {len(safety_directions)} projections have directions "
                f"of shape [{model.config.hidden_size}, {args.k}]")
    logger.info(f"Key format: {list(safety_directions.keys())[:4]}...")

    logger.info("\n" + "=" * 60)
    logger.info("Phase 3 complete (v2, per-projection).")
    logger.info("Next: run Phase 4 baselines with the new directions.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()