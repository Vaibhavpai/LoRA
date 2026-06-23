"""
test_verification_gate.py — Verification Gate
=============================================
Loads the base model Qwen2.5-1.5B-Instruct and runs a quick diagnostic pass
using the dataset loaders and metrics suite.

This script ensures:
  1. The base model and tokenizer load correctly.
  2. GSM8K evaluation runs on a 5-prompt sample.
  3. Alpaca validation loss runs on a 5-prompt sample.
  4. The safety refusal judge runs if GEMINI_API_KEY is configured.
"""

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import load_gsm8k_test, load_alpaca_val, load_advbench
from src.metrics import evaluate_task_gsm8k, evaluate_task_alpaca, evaluate_safety

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_diagnostics():
    print("=" * 60)
    print("LoRA-SafeLoop: Verification Gate / Diagnostics Run")
    print("=" * 60)
    print()

    device = "cpu"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    # 1. Load Tokenizer & Model
    print(f"Loading tokenizer and model '{model_id}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load in float32 for CPU compatibility, using low_cpu_mem_usage to prevent spikes
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    print("Model loaded successfully.")
    print()

    # 2. Test GSM8K capability evaluation (5 examples)
    print("Loading 5 GSM8K test examples...")
    gsm8k_samples = load_gsm8k_test(num_examples=5, seed=42)
    print("Evaluating GSM8K math accuracy (5 examples)...")
    gsm8k_acc = evaluate_task_gsm8k(model, tokenizer, gsm8k_samples, batch_size=2, device=device)
    print(f"-> GSM8K Accuracy: {gsm8k_acc * 100:.1f}%")
    print()

    # 3. Test Alpaca capability evaluation (5 examples)
    print("Loading 5 Alpaca validation examples...")
    alpaca_samples = load_alpaca_val(num_examples=5, seed=42)
    print("Evaluating Alpaca validation loss (5 examples)...")
    alpaca_loss = evaluate_task_alpaca(model, tokenizer, alpaca_samples, batch_size=2, device=device)
    print(f"-> Alpaca Val Loss: {alpaca_loss:.4f}")
    print()

    # 4. Test Safety Refusal rate (5 examples, requires API key)
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[-] Skipping safety refusal evaluation: GEMINI_API_KEY/GOOGLE_API_KEY not set in environment.")
        print("    To test this part, set the environment variable and run again.")
    else:
        print("[+] API Key detected. Loading 5 AdvBench prompts...")
        advbench_prompts = load_advbench()[:5]
        print("Evaluating safety refusal rate (5 examples)...")
        refusal_rate = evaluate_safety(
            model, tokenizer, advbench_prompts, batch_size=2, device=device
        )
        if refusal_rate is not None:
            print(f"-> Refusal Rate: {refusal_rate * 100:.1f}% (Expected ~100.0% for aligned base model)")
        else:
            print("-> Refusal Rate: N/A (API calls failed)")
    
    print()
    print("=" * 60)
    print("Diagnostics completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_diagnostics()
