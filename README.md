# LoRA-SafeLoop
# Adaptive Agent-Based Safety Preservation for LoRA Fine-Tuning

## Project Structure

```
LoRA/
├── src/                        # Core library code
│   ├── dataset_loader.py       # Phase 1 — GSM8K, Alpaca, AdvBench loaders
│   ├── refusal_judge.py        # Phase 1 — LLM-as-Judge refusal classifier
│   ├── metrics.py              # Phase 1 — Evaluation: safety, accuracy, alignment
│   ├── subspace_extraction.py  # Phase 3 — Contrastive activation SVD
│   ├── baselines.py            # Phase 4 — SafeLoRA (A/B), SaLoRA
│   ├── simple_controller.py    # Phase 5 — Rule-based adaptive controller
│   ├── constraint_applier.py   # Phase 6 — Per-layer gradient projection hooks
│   ├── agent.py                # Phase 6 — Claude API agent + fallback
│   └── reflexion.py            # Phase 6 — Reflexion memory module
├── experiments/
│   ├── train_vanilla.py        # Phase 2 — Vanilla LoRA training loop
│   ├── run_baselines.py        # Phase 4 — Run SafeLoRA, SaLoRA
│   ├── run_simple_ctrl.py      # Phase 5 — Run rule-based controller
│   └── run_agent.py            # Phase 6 — Run LoRA-SafeLoop agent
├── models/
│   └── safety_directions.pt   # Phase 3 output — saved safety subspace vectors
├── results/                   # All experiment logs (CSV files per run)
├── requirements.txt
└── README.md
```

## Setup

### Local (CPU) — for Phases 1–3
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Kaggle (GPU) — for Phases 4–6
```bash
pip install -r requirements.txt
```

## Committed Hyperparameters

| Hyperparameter | Value |
|---|---|
| Base Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| LoRA Rank `r` | 16 |
| LoRA Alpha | 32 |
| Target Modules | `q_proj`, `v_proj` |
| Learning Rate | 2e-4 |
| Effective Batch Size | 4 |
| Per-Device Batch Size | 1 |
| Gradient Accumulation | 4 steps |
| Training Steps | 2000 per task |
| Eval Frequency | Every 100 steps |
| Safety Directions `k` | Top-5 right singular vectors |
| Random Seeds | 42, 123, 456 |

## Research Questions
- **RQ1**: Does LoRA fine-tuning cause measurable safety drift across different task types?
- **RQ2**: How do existing static methods (SafeLoRA, SaLoRA) compare on a unified benchmark?
- **RQ3**: Does LoRA-SafeLoop's dynamic agent outperform the best static method?
- **RQ4**: Which component of LoRA-SafeLoop contributes most (ablation)?
