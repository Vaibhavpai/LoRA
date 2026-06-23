"""
run_salora.py — Phase 4, Step 4.2: SaLoRA (Baseline 3)
========================================================
Trains LoRA with the SaLoRA safety projection module active during
every forward pass.

SaLoRA (ICLR 2025) preserves safety alignment by projecting the
LoRA adapter output through C_SaLoRA = I - U_C @ U_C^T on every
forward pass, ensuring the adapter's contribution is orthogonal to
the safety-critical subspace.

Unlike SafeLoRA (which modifies weights periodically), SaLoRA works
in the activation space during every forward computation.

Usage:
    python experiments/run_salora.py --task gsm8k --seed 42
    python experiments/run_salora.py --task alpaca --seed 42
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
from peft import LoraConfig, get_peft_model

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import (
    load_gsm8k_train, load_gsm8k_test,
    load_alpaca_train, load_alpaca_val,
    load_advbench, load_safe_prompts,
)
from src.metrics import (
    evaluate_task_gsm8k, evaluate_task_alpaca,
    evaluate_safety, compute_subspace_alignment,
)
from src.baselines import load_safety_directions
from src.salora import (
    extract_per_layer_directions, setup_salora, verify_salora_active,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Dataset wrapper (identical to train_vanilla.py and run_baselines.py)
# ===========================================================================

class MaskedTrainingDataset(Dataset):
    """
    Formats prompt/target pairs into Qwen chat template with prompt masking.
    Loss is only computed on the response tokens.
    """
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_data = []

        for item in examples:
            prompt = item.get("question") or item.get("prompt")
            target = item.get("answer") or item.get("output")

            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True
            )
            prompt_len = len(tokenizer(
                formatted_prompt, add_special_tokens=False
            ).input_ids)

            full_text = formatted_prompt + target + tokenizer.eos_token
            full_inputs = tokenizer(
                full_text, max_length=self.max_length,
                truncation=True, add_special_tokens=False
            )
            input_ids = full_inputs.input_ids
            attention_mask = full_inputs.attention_mask
            labels = [-100] * len(input_ids)
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
    """Pads a batch of masked sequence dicts to the max length in the batch."""
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
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed}")


# ===========================================================================
# Build LoRA model (same hyperparameters as Phase 2)
# ===========================================================================

def build_lora_model(model_id: str, device: str):
    """
    Loads base Qwen2.5-1.5B-Instruct and applies LoRA config
    with the committed hyperparameters from the plan.
    """
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if capability >= 8 else torch.float16
    else:
        dtype = torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)
    return model


# ===========================================================================
# Main training loop with SaLoRA
# ===========================================================================

def run_salora(args, project_root: Path):
    """
    Trains LoRA from scratch with SaLoRA safety projection module active
    during every forward pass.

    This is identical to the vanilla LoRA training loop (Phase 2) except:
      - Before training, we build C_SaLoRA matrices from Phase 3 directions
      - We register forward hooks that apply the projection on every forward pass
      - We also log subspace alignment to track safety drift

    Produces a full training curve CSV (same format as other baselines).
    """
    logger.info("=" * 60)
    logger.info("SaLoRA — Safety-Alignment Preserved LoRA (Baseline 3)")
    logger.info("=" * 60)

    set_seed(args.seed)

    models_dir  = project_root / "models"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path         = results_dir / f"salora_{args.task}_seed{args.seed}.csv"
    adapter_save_dir = models_dir / f"salora_{args.task}_seed{args.seed}"

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    logger.info(f"Device: {device}")

    # -------------------------------------------------------------------
    # 1. Load datasets (identical to vanilla training)
    # -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
    # 2. Extract per-layer safety directions from BASE model
    #    (Paper: U_C = top-r_s left singular vectors of W·X_h per layer)
    # -------------------------------------------------------------------
    logger.info("Loading base model for per-layer direction extraction...")
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if capability >= 8 else torch.float16
    else:
        dtype = torch.float32

    base_model_for_extraction = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)

    # Load harmful prompts for direction extraction (70% of 520 = 364)
    harmful_for_extraction = load_advbench()[:364]

    per_layer_directions = extract_per_layer_directions(
        model=base_model_for_extraction,
        tokenizer=tokenizer,
        harmful_prompts=harmful_for_extraction,
        target_modules=("q_proj", "v_proj"),
        r_s=32,
        device=device,
        max_prompts=364,
    )
    logger.info(f"Extracted {len(per_layer_directions)} per-layer direction sets")

    # Free the extraction model to make room for the PEFT model
    del base_model_for_extraction
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Also load Phase 3 safety directions for the alignment metric
    safety_directions, metadata = load_safety_directions(models_dir)

    logger.info("Tokenizing training data...")
    train_dataset = MaskedTrainingDataset(raw_train, tokenizer)
    generator     = torch.Generator().manual_seed(args.seed)
    train_loader  = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        collate_fn=lambda b: training_collate_fn(b, tokenizer),
        generator=generator,
    )
    # Free raw_train to save memory
    del raw_train
    gc.collect()
    logger.info(f"Training examples: {len(train_dataset)}")

    # -------------------------------------------------------------------
    # 3. Build model with LoRA
    # -------------------------------------------------------------------
    model = build_lora_model(model_id, device)

    # -------------------------------------------------------------------
    # 4. Set up paper-accurate SaLoRA (3 components)
    #    a) Build C_SaLoRA matrices from per-layer U_C
    #    b) Task-specific initialization of A, B adapters
    #    c) Weight re-parameterization: W' = W - C·B_0·A_0
    #    d) Register forward hooks for safety projection
    # -------------------------------------------------------------------
    logger.info("Setting up paper-accurate SaLoRA...")
    hook_manager = setup_salora(
        model, per_layer_directions, device
    )

    # Verify hooks are working
    hooks_ok = verify_salora_active(model, tokenizer, hook_manager, device)
    if not hooks_ok:
        logger.warning(
            "SaLoRA hooks verification failed! Hooks may not be modifying "
            "the output. Training will continue but results may be invalid."
        )

    # -------------------------------------------------------------------
    # 5. Training loop (identical structure to vanilla + SaLoRA hooks)
    # -------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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

    logger.info(
        f"Starting SaLoRA training ({args.task}, seed={args.seed})...\n"
        f"  Total steps: {total_steps}\n"
        f"  Eval every:  {eval_every}\n"
        f"  Batch size:  1 × {grad_accumulation_steps} = {grad_accumulation_steps}\n"
        f"  Learning rate: {args.lr}"
    )

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

        # Forward pass — SaLoRA hooks automatically apply C_SaLoRA
        # projection to the LoRA adapter output during this forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss / grad_accumulation_steps
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
                logger.info(
                    f"--- Eval at Step {current_step} ---"
                )

                # Evaluate safety (refusal rate)
                refusal_rate = evaluate_safety(
                    model, tokenizer, advbench_prompts,
                    batch_size=4, device=device
                )

                # Evaluate task capability
                if args.task == "gsm8k":
                    task_metric = evaluate_task_gsm8k(
                        model, tokenizer, eval_data,
                        batch_size=4, device=device
                    )
                else:
                    task_metric = evaluate_task_alpaca(
                        model, tokenizer, eval_data,
                        batch_size=4, device=device
                    )

                # Subspace alignment
                alignments = compute_subspace_alignment(
                    model, safety_directions
                )
                mean_align = sum(alignments.values()) / len(alignments)

                record = {
                    "step":         current_step,
                    "train_loss":   accumulated_loss / eval_every,
                    "refusal_rate": refusal_rate,
                    metric_name:    task_metric,
                    "mean_alignment": mean_align,
                }
                history.append(record)

                logger.info(
                    f"Step {current_step} | "
                    f"Refusal: {refusal_rate:.3f} | "
                    f"{metric_name}: {task_metric:.4f} | "
                    f"Alignment: {mean_align:.6f}"
                )

                # Save progress to CSV immediately
                pd.DataFrame(history).to_csv(csv_path, index=False)

                # Save checkpoint
                ckpt_dir = adapter_save_dir / f"checkpoint-{current_step}"
                model.save_pretrained(ckpt_dir)

                accumulated_loss = 0.0

            step += 1

    # -------------------------------------------------------------------
    # 6. Save final adapter and clean up
    # -------------------------------------------------------------------
    adapter_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_save_dir)

    # Remove hooks (cleanup)
    hook_manager.remove_hooks()

    logger.info(f"SaLoRA training complete. Results: {csv_path}")
    logger.info(f"Final adapter saved to: {adapter_save_dir}")

    # Print final summary
    if history:
        final = history[-1]
        logger.info(
            f"\n{'='*50}\n"
            f"SaLoRA Final Results ({args.task}, seed={args.seed})\n"
            f"  Refusal Rate : {final['refusal_rate']:.4f}\n"
            f"  {metric_name}: {final.get(metric_name, 'N/A')}\n"
            f"  Mean Alignment: {final['mean_alignment']:.6f}\n"
            f"{'='*50}"
        )


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4, Step 4.2: SaLoRA (Baseline 3)"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["gsm8k", "alpaca"],
        help="Task to train on: 'gsm8k' or 'alpaca'"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--steps", type=int, default=2000,
        help="Total training steps"
    )
    parser.add_argument(
        "--eval_every", type=int, default=100,
        help="Evaluation frequency (in steps)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    run_salora(args, project_root)


if __name__ == "__main__":
    main()
