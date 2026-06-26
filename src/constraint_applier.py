"""
constraint_applier.py — Phase 6 (v4.1 — per-projection, continuous soft penalty)
==================================================================================
Continuous auxiliary loss term, now keyed per-projection (56 string keys).

v4.1 changes from v4:
  - String-keyed: "{layer_idx}.{proj_name}" instead of int layer indices.
  - Each key maps to one projection module (q_proj or v_proj at a specific layer).
  - Scaling-aware: uses LoRA's scaling factor in ΔW computation.
  - Agent controls lambda per-layer (split equally to q/v within that layer).

Penalty formula (unchanged math, per-projection keys):
    ΔW_l,p     = scaling * B @ A                    (WITH grad)
    P_l,p      = U @ U^T                            (fixed projector)
    penalty    = Σ λ_{l,p} × || ΔW_{l,p} @ P_{l,p} ||_F²
    total_loss = task_loss + penalty_scale × penalty

Usage:
    from src.constraint_applier import ConstraintApplier

    applier = ConstraintApplier(model, safety_directions, device)
    penalty = applier.compute_penalty()
    loss = task_loss + args.penalty_scale * penalty
    loss.backward()
    applier.set_all_lambdas(new_lambdas)
"""

import logging
import torch

logger = logging.getLogger(__name__)


def _resolve_lora_weights(model, key: str):
    """
    Given a key like "24.q_proj", returns (lora_A, lora_B, scaling) or None.
    lora_A and lora_B are the raw weight tensors (with grad).
    """
    parts = key.split(".")
    layer_idx = int(parts[0])
    proj_name = parts[1]

    try:
        layer = model.base_model.model.model.layers[layer_idx]
        proj = getattr(layer.self_attn, proj_name)
        lora_A = proj.lora_A.default.weight  # [r, d_in], requires_grad=True
        lora_B = proj.lora_B.default.weight  # [d_out, r], requires_grad=True

        scaling = 1.0
        if hasattr(proj, 'scaling'):
            if isinstance(proj.scaling, dict):
                scaling = proj.scaling.get("default", 1.0)
            elif isinstance(proj.scaling, (int, float)):
                scaling = float(proj.scaling)

        return lora_A, lora_B, scaling
    except AttributeError:
        return None


class ConstraintApplier:
    """
    Per-projection lambda + fixed projection matrices.
    Computes a differentiable penalty term every step (no weight surgery).
    """

    def __init__(self, model, safety_directions: dict, device: str, initial_lambda: float = 0.0):
        """
        Args:
            model: PEFT model with LoRA adapters.
            safety_directions: dict "{layer}.{proj}" -> [d_model, k] tensor.
            device: Training device.
            initial_lambda: Starting lambda for all projection points.
        """
        self.model = model
        self.device = device

        # Keys are strings like "0.q_proj", "0.v_proj", etc.
        self.keys = sorted(safety_directions.keys())
        self.lambdas = {k: initial_lambda for k in self.keys}

        # Precompute projection matrices P = U @ U^T
        self.proj_matrices = {}
        for key, U_raw in safety_directions.items():
            U = U_raw.to(torch.float32).to(device)
            P = (U @ U.T).detach().requires_grad_(False)
            self.proj_matrices[key] = P

        logger.info(
            f"ConstraintApplier (v4.1, per-projection) initialized "
            f"for {len(self.keys)} projection points."
        )

    # ------------------------------------------------------------------
    # Lambda management
    # ------------------------------------------------------------------

    def set_lambda(self, key: str, value: float):
        """Set lambda for a single projection point."""
        if key in self.lambdas:
            self.lambdas[key] = max(0.0, min(1.0, value))
        else:
            logger.warning(f"set_lambda: key '{key}' not tracked, ignored.")

    def set_all_lambdas(self, lambda_dict: dict):
        """
        Set lambdas from a dict. Accepts:
          - String keys ("24.q_proj"): set directly
          - Int keys (24): set both q_proj and v_proj for that layer
        """
        for key, value in lambda_dict.items():
            key_str = str(key)

            if key_str in self.lambdas:
                # Direct per-projection key
                self.set_lambda(key_str, value)
            else:
                # Try as layer index: apply to both q_proj and v_proj
                try:
                    layer_idx = int(key)
                    for proj_name in ("q_proj", "v_proj"):
                        full_key = f"{layer_idx}.{proj_name}"
                        if full_key in self.lambdas:
                            self.set_lambda(full_key, value)
                except (ValueError, TypeError):
                    logger.warning(f"set_all_lambdas: key '{key}' not recognized, ignored.")

    def decay_lambdas(self, rate: float = 0.98):
        """Call once per eval checkpoint, before the agent observes state."""
        for k in self.lambdas:
            self.lambdas[k] = max(0.0, self.lambdas[k] * rate)

    def get_lambdas(self) -> dict:
        """Return full lambda state (string-keyed)."""
        return dict(self.lambdas)

    def get_lambdas_per_layer(self) -> dict:
        """
        Return lambda state aggregated per layer (int-keyed).
        For the agent's observation: averages q_proj and v_proj lambdas.
        """
        layer_lambdas = {}
        for key, lam in self.lambdas.items():
            layer_idx = int(key.split(".")[0])
            if layer_idx not in layer_lambdas:
                layer_lambdas[layer_idx] = []
            layer_lambdas[layer_idx].append(lam)

        return {l: sum(vals) / len(vals) for l, vals in layer_lambdas.items()}

    # ------------------------------------------------------------------
    # Penalty computation
    # ------------------------------------------------------------------

    def compute_penalty(self) -> torch.Tensor:
        """
        Differentiable penalty. Call every training step, BEFORE loss.backward().
        Returns a scalar tensor.
        """
        any_active = any(v > 0.0 for v in self.lambdas.values())
        if not any_active:
            return torch.zeros((), device=self.device)

        penalty = torch.zeros((), device=self.device)
        for key in self.keys:
            lam = self.lambdas.get(key, 0.0)
            if lam <= 0.0:
                continue

            P = self.proj_matrices[key]
            resolved = _resolve_lora_weights(self.model, key)
            if resolved is None:
                continue

            lora_A, lora_B, scaling = resolved

            # Cast to fp32 for stable computation; preserves autograd
            A = lora_A.float()
            B = lora_B.float()
            delta_W = scaling * (B @ A)  # [d_out, d_in], WITH grad
            P_dev = P.to(delta_W.device)
            unsafe_component = delta_W @ P_dev  # [d_out, d_in]
            penalty = penalty + lam * (unsafe_component ** 2).sum()

        return penalty

    # ------------------------------------------------------------------
    # Diagnostic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def measure_alignment_ratio(self) -> dict:
        """
        Returns dict[key -> float] of ||ΔW@P|| / ||ΔW|| per projection point.
        """
        ratios = {}
        for key in self.keys:
            P = self.proj_matrices[key]
            resolved = _resolve_lora_weights(self.model, key)
            if resolved is None:
                continue

            lora_A, lora_B, scaling = resolved
            A = lora_A.detach().float()
            B = lora_B.detach().float()
            delta_W = scaling * (B @ A)
            proj_component = delta_W @ P.to(delta_W.device)
            ratio = proj_component.norm().item() / (delta_W.norm().item() + 1e-9)
            ratios[key] = ratio

        return ratios