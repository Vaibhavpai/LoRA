"""
run_baselines.py — Phase 4
===========================
Runs SafeLoRA static baselines against the vanilla LoRA from Phase 2.

  --variant a  : SafeLoRA Post-Hoc (B2A)
                 Loads the finished vanilla model → projects weights → evaluates.
                 No training curve; produces a single final evaluation point.

  --variant b  : SafeLoRA In-Training (B2B)
                 Trains from scratch with weight projection every 100 steps.
                 Produces a full training curve CSV matching vanilla format.

Both variants save results to results/ in the same format as vanilla runs,
so plot_drift_curves.py can compare them directly.

Usage:
  python experiments/run_baselines.py --variant a --task gsm8k --seed 42
  python experiments/run_baselines.py --variant b --task alpaca --seed 42
  python experiments/run_baselines.py --variant b --task gsm8k --seed 42
"""

import os
import sys
import argparse
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import (
    load_gsm8k_train, load_gsm8k_test,
    load_alpaca_train, load_alpaca_val,
    load_advbench,
)
from src.metrics import (
    evaluate_task_gsm8k, evaluate_task_alpaca,
    evaluate_safety, compute_subspace_alignment,
)
from src.baselines import (
    load_safety_directions,
    build_projection_matrices,
    project_all_lora_layers,
    verify_projection,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# Shared: dataset wrapper (identical to train_vanilla.py)
# ===========================================================================

class MaskedTrainingDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.tokenized_data = []

        for item in examples:
            prompt = item.get("question") or item.get("prompt")
            target = item.get("answer")  or item.get("output")

            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True
            )
            prompt_len = len(tokenizer(formatted_prompt, add_special_tokens=False).input_ids)

            full_text   = formatted_prompt + target + tokenizer.eos_token
            full_inputs = tokenizer(
                full_text, max_length=self.max_length,
                truncation=True, add_special_tokens=False
            )
            input_ids      = full_inputs.input_ids
            attention_mask = full_inputs.attention_mask
            labels         = [-100] * len(input_ids)
            for j in range(prompt_len, len(input_ids)):
                labels[j] = input_ids[j]

            if len(input_ids) > prompt_len:
                self.tokenized_data.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                })

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]


def training_collate_fn(batch, tokenizer):
    max_len = max(len(x["input_ids"]) for x in batch)
    ib, ab, lb = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        ib.append(x["input_ids"]      + [tokenizer.pad_token_id] * pad)
        ab.append(x["attention_mask"] + [0]                      * pad)
        lb.append(x["labels"]         + [-100]                   * pad)
    return {
        "input_ids":      torch.tensor(ib),
        "attention_mask": torch.tensor(ab),
        "labels":         torch.tensor(lb),
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed}")


# ===========================================================================
# Shared: build LoRA model
# ===========================================================================

def build_lora_model(model_id: str, device: str):
    """Loads base model and applies LoRA config (identical hyperparams as Phase 2)."""
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)
    return model


# ===========================================================================
# Variant A — SafeLoRA Post-Hoc
# ===========================================================================

def run_variant_a(args, project_root: Path):
    """
    Loads the finished vanilla model from Phase 2, applies one-time
    SafeLoRA projection to its weights, then evaluates.

    Saves a single-row CSV (no training curve).
    """
    logger.info("=" * 60)
    logger.info("SafeLoRA Variant A — Post-Hoc Projection")
    logger.info("=" * 60)

    models_dir  = project_root / "models"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    # 1. Load safety directions and build projection matrices
    safety_directions, _ = load_safety_directions(models_dir)
    projection_matrices  = build_projection_matrices(safety_directions, device)

    # 2. Load vanilla fine-tuned model (the one we're projecting)
    adapter_path = models_dir / f"vanilla_{args.task}_seed{args.seed}"
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Vanilla adapter not found: {adapter_path}\n"
            f"Run Phase 2 first: python experiments/train_vanilla.py "
            f"--task {args.task} --seed {args.seed}"
        )
    logger.info(f"Loading vanilla adapter from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.to(device)

    # 3. Apply SafeLoRA post-hoc projection (ONCE, to final weights)
    logger.info("Applying SafeLoRA post-hoc projection...")
    n = project_all_lora_layers(model, projection_matrices)
    logger.info(f"Projected {n} (layer, proj) pairs.")

    # 4. Sanity check — safety-subspace energy should be near zero
    logger.info("Verifying projection correctness...")
    ok = verify_projection(model, projection_matrices)
    if not ok:
        logger.warning("Projection verification FAILED for some layers. Check tolerance.")

    # 5. Evaluate
    logger.info("Evaluating post-hoc projected model...")
    advbench_prompts = load_advbench()[:100]
    refusal_rate = evaluate_safety(model, tokenizer, advbench_prompts, batch_size=4, device=device)

    if args.task == "gsm8k":
        eval_data   = load_gsm8k_test(num_examples=200, seed=args.seed)
        task_metric = evaluate_task_gsm8k(model, tokenizer, eval_data, batch_size=4, device=device)
        metric_name = "gsm8k_accuracy"
    else:
        eval_data   = load_alpaca_val(num_examples=500, seed=args.seed)
        task_metric = evaluate_task_alpaca(model, tokenizer, eval_data, batch_size=4, device=device)
        metric_name = "alpaca_val_loss"

    # Subspace alignment after projection (should drop vs. vanilla)
    alignments = compute_subspace_alignment(model, safety_directions)
    mean_align = sum(alignments.values()) / len(alignments)

    logger.info(
        f"\n{'='*50}\n"
        f"SafeLoRA A Results ({args.task}, seed={args.seed})\n"
        f"  Refusal Rate : {refusal_rate:.4f}\n"
        f"  {metric_name}: {task_metric:.4f}\n"
        f"  Mean Alignment (post-proj): {mean_align:.4f}  (should be lower than vanilla)\n"
        f"{'='*50}"
    )

    # 6. Save result
    record = {
        "step":       2000,          # equivalent to end-of-training
        "variant":    "safelora_a",
        "refusal_rate": refusal_rate,
        metric_name:    task_metric,
        "mean_alignment": mean_align,
    }
    csv_path = results_dir / f"safelora_a_{args.task}_seed{args.seed}.csv"
    pd.DataFrame([record]).to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")


# ===========================================================================
# Variant B — SafeLoRA In-Training
# ===========================================================================

def run_variant_b(args, project_root: Path):
    """
    Trains from scratch (same hyperparameters as vanilla LoRA Phase 2),
    but applies SafeLoRA weight projection every eval_every=100 steps.

    Saves a full training curve CSV (same format as vanilla_gsm8k_seed42.csv).
    """
    logger.info("=" * 60)
    logger.info("SafeLoRA Variant B — In-Training Projection")
    logger.info("=" * 60)

    set_seed(args.seed)

    models_dir  = project_root / "models"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path        = results_dir / f"safelora_b_{args.task}_seed{args.seed}.csv"
    adapter_save_dir = models_dir / f"safelora_b_{args.task}_seed{args.seed}"

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    logger.info(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load safety directions
    # -----------------------------------------------------------------------
    safety_directions, _ = load_safety_directions(models_dir)
    projection_matrices  = build_projection_matrices(safety_directions, device)

    # -----------------------------------------------------------------------
    # Load datasets (identical to train_vanilla.py)
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.task == "gsm8k":
        raw_train   = load_gsm8k_train()
        eval_data   = load_gsm8k_test(num_examples=200, seed=args.seed)
        metric_name = "gsm8k_accuracy"
    else:
        raw_train   = load_alpaca_train(num_examples=5000, seed=args.seed)
        eval_data   = load_alpaca_val(num_examples=500, seed=args.seed)
        metric_name = "alpaca_val_loss"

    advbench_prompts = load_advbench()[:100]

    logger.info("Tokenizing training data...")
    train_dataset = MaskedTrainingDataset(raw_train, tokenizer)
    generator     = torch.Generator().manual_seed(args.seed)
    train_loader  = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        collate_fn=lambda b: training_collate_fn(b, tokenizer),
        generator=generator,
    )
    logger.info(f"Training examples: {len(train_dataset)}")

    # -----------------------------------------------------------------------
    # Build model with LoRA
    # -----------------------------------------------------------------------
    model     = build_lora_model(model_id, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # -----------------------------------------------------------------------
    # Training loop (identical structure to train_vanilla.py + projection step)
    # -----------------------------------------------------------------------
    grad_accumulation_steps = 4
    total_steps  = args.steps
    eval_every   = args.eval_every

    history     = []
    step        = 0
    epoch       = 0
    running_loss     = 0.0
    accumulated_loss = 0.0
    batch_idx   = 0
    data_iter   = iter(train_loader)

    model.zero_grad()

    logger.info(f"Starting SafeLoRA B training ({args.task}, seed={args.seed})...")

    while step < total_steps:
        model.train()

        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            logger.info(f"Epoch {epoch}")
            train_loader.generator.manual_seed(args.seed + epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss / grad_accumulation_steps
        loss.backward()

        accumulated_loss += loss.item() * grad_accumulation_steps
        running_loss     += loss.item() * grad_accumulation_steps
        batch_idx        += 1

        if batch_idx % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            current_step = step + 1

            if current_step % 10 == 0:
                logger.info(
                    f"Step {current_step}/{total_steps} | "
                    f"Loss: {running_loss / 10:.4f}"
                )
                running_loss = 0.0

            if current_step % eval_every == 0:
                logger.info(f"--- Eval + SafeLoRA B Projection at Step {current_step} ---")

                # KEY DIFFERENCE from vanilla: project weights before eval
                project_all_lora_layers(model, projection_matrices)

                # Evaluate (same as vanilla)
                refusal_rate = evaluate_safety(
                    model, tokenizer, advbench_prompts, batch_size=4, device=device
                )
                if args.task == "gsm8k":
                    task_metric = evaluate_task_gsm8k(
                        model, tokenizer, eval_data, batch_size=4, device=device
                    )
                else:
                    task_metric = evaluate_task_alpaca(
                        model, tokenizer, eval_data, batch_size=4, device=device
                    )

                # Subspace alignment (should stay lower than vanilla B due to projection)
                alignments = compute_subspace_alignment(model, safety_directions)
                mean_align = sum(alignments.values()) / len(alignments)

                record = {
                    "step":        current_step,
                    "train_loss":  accumulated_loss / eval_every,
                    "refusal_rate": refusal_rate,
                    metric_name:   task_metric,
                    "mean_alignment": mean_align,
                }
                history.append(record)

                logger.info(
                    f"Step {current_step} | Refusal: {refusal_rate:.3f} | "
                    f"{metric_name}: {task_metric:.4f} | Alignment: {mean_align:.4f}"
                )

                pd.DataFrame(history).to_csv(csv_path, index=False)

                # Save checkpoint
                ckpt_dir = adapter_save_dir / f"checkpoint-{current_step}"
                model.save_pretrained(ckpt_dir)

                accumulated_loss = 0.0

            step += 1

    # Save final adapter
    adapter_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_save_dir)
    logger.info(f"SafeLoRA B training complete. Results: {csv_path}")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 4: SafeLoRA Baselines")
    parser.add_argument("--variant", type=str, required=True, choices=["a", "b"],
                        help="'a' = post-hoc, 'b' = in-training")
    parser.add_argument("--task",    type=str, required=True, choices=["gsm8k", "alpaca"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--steps",   type=int, default=2000,
                        help="Training steps (variant b only)")
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Eval + projection frequency (variant b only)")
    parser.add_argument("--lr",      type=float, default=2e-4,
                        help="Learning rate (variant b only)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.variant == "a":
        run_variant_a(args, project_root)
    else:
        run_variant_b(args, project_root)


if __name__ == "__main__":
    main()