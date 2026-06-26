"""
run_agent.py — Phase 6 (v4.1, per-projection)
================================================
Runs Baseline 5: LoRA-SafeLoop Agentic Framework.
Dynamically sets per-layer lambda constraints using a Groq LLM agent.

v4.1 update: per-projection string-keyed directions (56 keys).
Agent still decides per-layer (28 decisions); constraint_applier expands
to per-projection. CSV logs per-projection lambdas.

Usage:
  export GROQ_API_KEY="your_api_key_here"
  python experiments/run_agent.py --task gsm8k --seed 42
  python experiments/run_agent.py --task alpaca --seed 42 --penalty_scale 0.02
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

# Phase 6 imports
from src.constraint_applier import ConstraintApplier
from src.agent import format_observation, call_groq_agent, parse_agent_response
from src.reflexion import ReflexionMemory


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _extract_layer_ids(lambda_state: dict) -> list:
    """Extract unique integer layer IDs from string-keyed lambda state."""
    layer_ids = set()
    for key in lambda_state.keys():
        layer_idx = int(str(key).split(".")[0])
        layer_ids.add(layer_idx)
    return sorted(layer_ids)


def main():
    parser = argparse.ArgumentParser(description="Phase 6: LoRA-SafeLoop Agent (v4.1, per-projection)")
    parser.add_argument("--task",    type=str, required=True, choices=["gsm8k", "alpaca"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--steps",   type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--lr",      type=float, default=2e-4)
    parser.add_argument("--penalty_scale", type=float, default=0.01,
                        help="Global scale on the safety penalty term.")
    parser.add_argument("--lambda_decay_rate", type=float, default=0.98,
                        help="Per-checkpoint multiplicative decay for lambda.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    logger.info("=" * 60)
    logger.info(f"Phase 6 (v4.1, per-projection): Agentic Controller ({args.task}, seed={args.seed})")
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    logger.info("=" * 60)

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        logger.warning("GROQ_API_KEY not set. Agent calls will fail and fallback will be used.")

    set_seed(args.seed)

    models_dir  = project_root / "models"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path          = results_dir / f"agent_{args.task}_seed{args.seed}.csv"
    failures_log_path = results_dir / f"agent_{args.task}_seed{args.seed}_failures.csv"
    reflexion_path    = results_dir / f"agent_{args.task}_seed{args.seed}_reflexion.jsonl"
    adapter_save_dir  = models_dir / f"agent_{args.task}_seed{args.seed}"

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    # 1. Load safety directions (v2: string-keyed, 56 keys)
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

    # 3. Build model
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

    # Initialize state
    start_step = 0
    history = []

    # Initialize Reflexion memory
    reflexion_memory = ReflexionMemory(reflexion_path)

    # Constraint applier (v4.1: per-projection, string-keyed)
    applier = ConstraintApplier(model, safety_directions, device, initial_lambda=0.0)
    lambda_state = applier.get_lambdas()

    # Extract unique layer IDs for the agent's valid_layer_ids
    valid_layer_ids = _extract_layer_ids(lambda_state)

    # Measure pre-training baseline refusal rate
    logger.info("Measuring pre-training baseline refusal rate...")
    baseline_refusal_rate = evaluate_safety(
        model, tokenizer, advbench_prompts, batch_size=4, device=device
    )
    logger.info(f"Pre-training baseline refusal rate: {baseline_refusal_rate:.3f}")
    prev_refusal_rate = baseline_refusal_rate
    refusal_history = [baseline_refusal_rate]

    if args.resume_from_checkpoint:
        if csv_path.exists():
            try:
                history_df = pd.read_csv(csv_path)
                history = history_df.to_dict(orient="records")
                checkpoint_step = int(Path(args.resume_from_checkpoint).name.split("-")[-1])
                history = [h for h in history if h["step"] <= checkpoint_step]

                if len(history) > 0:
                    start_step = history[-1]["step"]

                    # Restore lambda state from CSV columns
                    for key in lambda_state.keys():
                        col_name = f"lambda_{key}"
                        if col_name in history[-1]:
                            lambda_state[key] = float(history[-1][col_name])

                    baseline_refusal_rate = history[0]["refusal_rate"]
                    prev_refusal_rate = history[-1]["refusal_rate"]
                    refusal_history = [h["refusal_rate"] for h in history]

                    logger.info(f"Restored state from CSV: step={start_step}")
                else:
                    logger.warning("CSV found but has no matching entries.")
            except Exception as e:
                logger.error(f"Failed to restore history from CSV: {e}")

    applier.set_all_lambdas(lambda_state)

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
    accumulated_penalty = 0.0
    batch_idx = step * grad_accumulation_steps
    data_iter = iter(train_loader)

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
        task_loss = outputs.loss / grad_accumulation_steps

        # Continuous safety penalty
        penalty = applier.compute_penalty() / grad_accumulation_steps
        loss = task_loss + args.penalty_scale * penalty
        loss.backward()

        accumulated_loss    += task_loss.item() * grad_accumulation_steps
        accumulated_penalty += penalty.item() * grad_accumulation_steps
        running_loss        += task_loss.item() * grad_accumulation_steps
        batch_idx            += 1

        if batch_idx % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            current_step = step + 1

            if current_step % 10 == 0:
                logger.info(f"Step {current_step}/{args.steps} | Task Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

            if current_step % args.eval_every == 0:
                is_final_step = (current_step >= args.steps)
                logger.info(f"--- Eval{' (FINAL)' if is_final_step else ' + Agent Update'} at Step {current_step} ---")

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

                # Per-projection subspace alignment
                alignments = compute_subspace_alignment(model, safety_directions)
                mean_align = sum(alignments.values()) / len(alignments)

                refusal_history.append(refusal_rate)
                smoothed_refusal = sum(refusal_history[-3:]) / len(refusal_history[-3:])

                # Complete pending reflexion record
                reflexion_memory.complete_pending(refusal_rate, task_metric)

                # Apply λ decay
                applier.decay_lambdas(rate=args.lambda_decay_rate)
                lambda_state = applier.get_lambdas()

                # Build observation
                observation = format_observation(
                    step=current_step,
                    total_steps=args.steps,
                    task=args.task,
                    refusal_rate=refusal_rate,
                    prev_refusal_rate=prev_refusal_rate,
                    baseline_refusal_rate=baseline_refusal_rate,
                    task_metric=task_metric,
                    metric_name=metric_name,
                    alignments=alignments,
                    lambda_state=lambda_state,
                    reflexion_memory=reflexion_memory,
                    smoothed_refusal_rate=smoothed_refusal,
                )

                if is_final_step:
                    logger.info("Final step — skipping agent call.")
                    parsed = {
                        "layer_constraints": dict(lambda_state),
                        "rationale": "[FINAL] Training complete.",
                        "predicted_outcome": "Training complete.",
                        "fallback_used": True,
                    }
                else:
                    logger.info("Calling Agent...")
                    agent_response = None
                    if groq_api_key:
                        agent_response = call_groq_agent(observation, api_key=groq_api_key)

                    parsed = parse_agent_response(
                        agent_response,
                        current_lambda_state=lambda_state,
                        valid_layer_ids=valid_layer_ids,
                        failure_log_path=str(failures_log_path),
                        alignments=alignments,
                    )

                # Update lambdas
                applier.set_all_lambdas(parsed["layer_constraints"])

                reflexion_memory.add_pending(
                    step=current_step,
                    lambda_decisions=parsed["layer_constraints"],
                    refusal_before=refusal_rate,
                    rationale=parsed["rationale"],
                )

                lambda_state = applier.get_lambdas()
                prev_refusal_rate = refusal_rate

                record = {
                    "step":        current_step,
                    "train_loss":  accumulated_loss / args.eval_every,
                    "penalty_term": accumulated_penalty / args.eval_every,
                    "refusal_rate": refusal_rate,
                    "refusal_rate_smoothed": smoothed_refusal,
                    metric_name:   task_metric,
                    "mean_alignment": mean_align,
                }

                # Per-projection lambda columns
                for key in sorted(lambda_state.keys()):
                    record[f"lambda_{key}"] = lambda_state[key]

                history.append(record)

                pd.DataFrame(history).to_csv(csv_path, index=False)

                ckpt_dir = adapter_save_dir / f"checkpoint-{current_step}"
                model.save_pretrained(ckpt_dir)

                accumulated_loss = 0.0
                accumulated_penalty = 0.0

            step += 1

    # Save final adapter
    adapter_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_save_dir)
    logger.info(f"Phase 6 Training Complete. Results: {csv_path}")

if __name__ == "__main__":
    main()
