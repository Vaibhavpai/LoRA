"""
run_simple_ctrl.py — Phase 5
=============================
Runs Baseline 4: Simple Adaptive Controller (Rule-Based).
Dynamically adjusts the penalty coefficient (lambda) every 100 steps
based on the refusal rate using a PID-like logic.

Usage:
  python experiments/run_simple_ctrl.py --task gsm8k --seed 42
  python experiments/run_simple_ctrl.py --task alpaca --seed 42
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from src.baselines import load_safety_directions
from experiments.run_baselines import MaskedTrainingDataset, training_collate_fn, set_seed, build_lora_model
from src.simple_controller import SimpleAdaptiveController, register_gradient_hooks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Simple Adaptive Controller")
    parser.add_argument("--task",    type=str, required=True, choices=["gsm8k", "alpaca"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--steps",   type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--lr",      type=float, default=2e-4)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume from (e.g. models/simple_ctrl_gsm8k_seed42/checkpoint-1200)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    logger.info("=" * 60)
    logger.info(f"Phase 5: Simple Adaptive Controller ({args.task}, seed={args.seed})")
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    logger.info("=" * 60)

    set_seed(args.seed)

    models_dir  = project_root / "models"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path         = results_dir / f"simple_ctrl_{args.task}_seed{args.seed}.csv"
    adapter_save_dir = models_dir / f"simple_ctrl_{args.task}_seed{args.seed}"

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    # 1. Load safety directions
    safety_directions, _ = load_safety_directions(models_dir)

    # 2. Load datasets
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

    # 3. Build model and initialize controller
    # If resuming, load the PEFT model from the checkpoint path
    if args.resume_from_checkpoint:
        logger.info(f"Loading checkpoint weights from {args.resume_from_checkpoint}...")
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.resume_from_checkpoint, is_trainable=True)
        model.to(device)
    else:
        model = build_lora_model(model_id, device)
    
    # Initialize controller
    initial_lambda = 0.3
    start_step = 0
    history = []
    
    # If resuming, restore controller state from the existing CSV
    if args.resume_from_checkpoint:
        if csv_path.exists():
            try:
                history_df = pd.read_csv(csv_path)
                history = history_df.to_dict(orient="records")
                # Parse step number from checkpoint path: .../checkpoint-X -> X
                checkpoint_step = int(Path(args.resume_from_checkpoint).name.split("-")[-1])
                # Filter history to match the step we are resuming from
                history = [h for h in history if h["step"] <= checkpoint_step]
                
                if len(history) > 0:
                    start_step = history[-1]["step"]
                    initial_lambda = history[-1]["lambda"]
                    logger.info(f"Successfully restored state from CSV: step={start_step}, lambda={initial_lambda}")
                else:
                    logger.warning("CSV file found but has no entries matching or preceding the checkpoint step.")
            except Exception as e:
                logger.error(f"Failed to restore history/controller state from CSV: {e}")
        else:
            logger.warning(f"CSV file not found at {csv_path}. Starting with default lambda state.")

    controller = SimpleAdaptiveController(initial_lambda=initial_lambda, delta=0.05)
    
    # Initialize a shared mutable dictionary for the hooks
    lambda_state = {int(layer_idx): controller.current_lambda for layer_idx in safety_directions.keys()}
    
    # Register the backward hooks on lora_A
    hook_handles = register_gradient_hooks(model, safety_directions, lambda_state, device)
    logger.info(f"Registered {len(hook_handles)} gradient hooks.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 4. Training loop
    grad_accumulation_steps = 4
    if not args.resume_from_checkpoint:
        history = []
        step = 0
    else:
        step = start_step
    
    epoch = 0
    running_loss = 0.0
    accumulated_loss = 0.0
    batch_idx = step * grad_accumulation_steps
    data_iter = iter(train_loader)

    # Fast forward data_iter to approximate the resume point
    if args.resume_from_checkpoint and batch_idx > 0:
        logger.info(f"Fast-forwarding data loader to step {step}...")
        for _ in range(batch_idx):
            try:
                next(data_iter)
            except StopIteration:
                epoch += 1
                train_loader.generator.manual_seed(args.seed + epoch)
                data_iter = iter(train_loader)
                next(data_iter)

    model.zero_grad()

    while step < args.steps:
        model.train()

        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            train_loader.generator.manual_seed(args.seed + epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss / grad_accumulation_steps
        loss.backward()  # <--- Hooks apply gradient projection here automatically

        accumulated_loss += loss.item() * grad_accumulation_steps
        running_loss     += loss.item() * grad_accumulation_steps
        batch_idx        += 1

        if batch_idx % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            current_step = step + 1

            if current_step % 10 == 0:
                logger.info(f"Step {current_step}/{args.steps} | Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

            if current_step % args.eval_every == 0:
                logger.info(f"--- Eval + Controller Update at Step {current_step} ---")

                # Evaluate
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

                # Subspace alignment
                alignments = compute_subspace_alignment(model, safety_directions)
                mean_align = sum(alignments.values()) / len(alignments)

                # UPDATE CONTROLLER
                new_lambda = controller.update(refusal_rate)
                
                # Update shared lambda state for the hooks
                for layer_idx in lambda_state.keys():
                    lambda_state[layer_idx] = new_lambda

                record = {
                    "step":        current_step,
                    "train_loss":  accumulated_loss / args.eval_every,
                    "refusal_rate": refusal_rate,
                    metric_name:   task_metric,
                    "mean_alignment": mean_align,
                    "lambda":      new_lambda,
                }
                history.append(record)

                pd.DataFrame(history).to_csv(csv_path, index=False)

                ckpt_dir = adapter_save_dir / f"checkpoint-{current_step}"
                model.save_pretrained(ckpt_dir)

                accumulated_loss = 0.0

            step += 1

    # Cleanup and Save
    for handle in hook_handles:
        handle.remove()
        
    adapter_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_save_dir)
    logger.info(f"Phase 5 Training Complete. Results: {csv_path}")

if __name__ == "__main__":
    main()
