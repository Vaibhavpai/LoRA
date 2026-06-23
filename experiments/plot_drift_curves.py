"""
plot_drift_curves.py — Phase 2, Step 2.2
=========================================
Generates Figure 1 for the paper: the safety drift curves showing
Refusal Rate and Task Capability metrics over training steps.

Looks for:
  results/vanilla_gsm8k_seed*.csv
  results/vanilla_alpaca_seed*.csv

Averages logs over seeds if multiple are found, and plots mean +/- standard deviation.
Saves the resulting plot as results/drift_curves.png.
"""

import re
import logging
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_and_aggregate(results_dir: Path, task: str) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Loads all logs for a specific task and computes mean and std at each step."""
    pattern = re.compile(rf"vanilla_{task}_seed(\d+)\.csv")
    files = [f for f in results_dir.glob("*.csv") if pattern.match(f.name)]

    if not files:
        logger.warning(f"No log files found for task: {task} in {results_dir}")
        return None, None

    logger.info(f"Found {len(files)} run files for task: {task}")
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Ensure standard columns exist
        if "step" not in df.columns:
            logger.warning(f"File {f.name} missing 'step' column. Skipping.")
            continue
        dfs.append(df.set_index("step"))

    if not dfs:
        return None, None

    # Align on index (step)
    combined = pd.concat(dfs, axis=0, keys=range(len(dfs)))
    
    # Compute mean and standard deviation grouped by step index
    mean_df = combined.groupby(level=1).mean()
    std_df = combined.groupby(level=1).std().fillna(0.0)  # fillna for single-seed run

    return mean_df.reset_index(), std_df.reset_index()


def generate_plots():
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    
    # Setup matplotlib stylesheet style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Harmonious color palette (HSL tailored / sleek)
    primary_color = "#3B82F6"   # Electric Blue (Task performance)
    safety_color = "#EF4444"    # Coral Red (Safety rate)
    
    has_data = False

    # -----------------------------------------------------------------------
    # Plot 1: GSM8K (Math Task)
    # -----------------------------------------------------------------------
    mean_gsm, std_gsm = load_and_aggregate(results_dir, "gsm8k")
    if mean_gsm is not None:
        has_data = True
        steps = mean_gsm["step"]
        
        # Dual Y-axis plot
        # Left Y-axis: Refusal Rate
        color = safety_color
        ax1.set_xlabel("Training Steps", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Refusal Rate (Safety)", color=color, fontsize=11, fontweight='bold')
        line1 = ax1.plot(steps, mean_gsm["refusal_rate"], color=color, linestyle="-", marker="o", label="Refusal Rate")
        ax1.fill_between(
            steps, 
            np.clip(mean_gsm["refusal_rate"] - std_gsm["refusal_rate"], 0.0, 1.0),
            np.clip(mean_gsm["refusal_rate"] + std_gsm["refusal_rate"], 0.0, 1.0),
            color=color, alpha=0.15
        )
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(-0.05, 1.05)
        
        # Right Y-axis: Math Accuracy
        ax1_twin = ax1.twinx()
        color = primary_color
        ax1_twin.set_ylabel("GSM8K Accuracy (Capability)", color=color, fontsize=11, fontweight='bold')
        line2 = ax1_twin.plot(steps, mean_gsm["gsm8k_accuracy"], color=color, linestyle="--", marker="s", label="Math Accuracy")
        ax1_twin.fill_between(
            steps, 
            np.clip(mean_gsm["gsm8k_accuracy"] - std_gsm["gsm8k_accuracy"], 0.0, 1.0),
            np.clip(mean_gsm["gsm8k_accuracy"] + std_gsm["gsm8k_accuracy"], 0.0, 1.0),
            color=color, alpha=0.15
        )
        ax1_twin.tick_params(axis='y', labelcolor=color)
        ax1_twin.set_ylim(-0.05, 1.05)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center left", frameon=True, facecolor="white", edgecolor="none")
        ax1.set_title("Vanilla LoRA on GSM8K (Math reasoning)", fontsize=13, pad=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, "No GSM8K data found.\nRun experiments first.", 
                 ha='center', va='center', fontsize=12, color='gray')
        ax1.set_title("Vanilla LoRA on GSM8K", fontsize=13, pad=12)

    # -----------------------------------------------------------------------
    # Plot 2: Alpaca (General Instructions)
    # -----------------------------------------------------------------------
    mean_alp, std_alp = load_and_aggregate(results_dir, "alpaca")
    if mean_alp is not None:
        has_data = True
        steps = mean_alp["step"]
        
        # Dual Y-axis plot
        # Left Y-axis: Refusal Rate
        color = safety_color
        ax2.set_xlabel("Training Steps", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Refusal Rate (Safety)", color=color, fontsize=11, fontweight='bold')
        line1 = ax2.plot(steps, mean_alp["refusal_rate"], color=color, linestyle="-", marker="o", label="Refusal Rate")
        ax2.fill_between(
            steps, 
            np.clip(mean_alp["refusal_rate"] - std_alp["refusal_rate"], 0.0, 1.0),
            np.clip(mean_alp["refusal_rate"] + std_alp["refusal_rate"], 0.0, 1.0),
            color=color, alpha=0.15
        )
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-0.05, 1.05)
        
        # Right Y-axis: Validation Loss
        ax2_twin = ax2.twinx()
        color = primary_color
        ax2_twin.set_ylabel("Alpaca Validation Loss (Utility)", color=color, fontsize=11, fontweight='bold')
        line2 = ax2_twin.plot(steps, mean_alp["alpaca_val_loss"], color=color, linestyle="--", marker="x", label="Val Loss")
        ax2_twin.fill_between(
            steps, 
            mean_alp["alpaca_val_loss"] - std_alp["alpaca_val_loss"],
            mean_alp["alpaca_val_loss"] + std_alp["alpaca_val_loss"],
            color=color, alpha=0.15
        )
        ax2_twin.tick_params(axis='y', labelcolor=color)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="center left", frameon=True, facecolor="white", edgecolor="none")
        ax2.set_title("Vanilla LoRA on Alpaca-5k (General instruction)", fontsize=13, pad=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "No Alpaca data found.\nRun experiments first.", 
                 ha='center', va='center', fontsize=12, color='gray')
        ax2.set_title("Vanilla LoRA on Alpaca-5k", fontsize=13, pad=12)

    plt.tight_layout()
    
    if has_data:
        plot_path = results_dir / "drift_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Successfully saved drift curves plot (Figure 1) to {plot_path}")
    else:
        logger.warning("No plot generated: no task datasets were available to aggregate.")
        
    plt.close()


if __name__ == "__main__":
    generate_plots()
