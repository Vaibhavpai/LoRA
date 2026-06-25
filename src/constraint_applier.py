"""
constraint_applier.py — Phase 6 (v4 — continuous soft penalty)
=================================================================
Replaces the periodic SVD weight-projection (v2/v3) with a continuous
auxiliary loss term, added to the task loss on EVERY training step.

Why this change: v2/v3 paused every 100 steps and hard-projected ΔW=B@A
away from the unsafe subspace, then let the model train completely
unconstrained for the next 99 steps. That created a sawtooth pattern —
drift, get yanked back, drift again — which is a likely contributor to
noisy refusal_rate across runs.

v4 instead adds a differentiable penalty term every step:

    ΔW_l        = B_l @ A_l                      (this layer's weight update, WITH grad)
    P_l         = U_l @ U_l^T                    (safety-subspace projector, fixed)
    penalty_l   = λ_l * || ΔW_l @ P_l ||_F^2      (squared norm of the unsafe-direction component)
    total_loss  = task_loss + penalty_scale * sum_l(penalty_l)

Backprop through this penalty nudges lora_A/lora_B away from the unsafe
direction continuously, instead of correcting after the fact. No SVD,
no re-factoring, no in-place weight surgery — purely an added loss term.

lambda_l still only changes at eval checkpoints (same cadence as before,
same agent/reflexion loop) — only WHAT lambda multiplies has changed.

Usage:
    from src.constraint_applier import ConstraintApplier

    applier = ConstraintApplier(model, safety_directions, device)
    ...
    # every training step, before loss.backward():
    penalty = applier.compute_penalty()
    loss = task_loss + args.penalty_scale * penalty
    loss.backward()
    ...
    # every eval_every steps, after agent decides:
    applier.set_all_lambdas(new_lambdas)   # no apply_projection() call needed anymore
"""

import logging
import torch

logger = logging.getLogger(__name__)

PROJ_NAMES = ("q_proj", "v_proj")


class ConstraintApplier:
    """
    Holds per-layer lambda + fixed projection matrices, and computes a
    differentiable penalty term every step (no weight surgery).
    """

    def __init__(self, model, safety_directions: dict, device: str, initial_lambda: float = 0.0):
        self.model = model
        self.device = device

        self.layer_indices = [int(k) for k in safety_directions.keys()]
        self.lambdas = {l: initial_lambda for l in self.layer_indices}

        self.proj_matrices = {}
        for layer_idx_raw, U_l in safety_directions.items():
            l = int(layer_idx_raw)
            U = U_l.to(torch.float32).to(device)
            P_l = (U @ U.T).detach().requires_grad_(False)
            self.proj_matrices[l] = P_l

        logger.info(f"ConstraintApplier (v4, continuous penalty) initialized for {len(self.layer_indices)} layers.")

    def set_lambda(self, layer_idx: int, value: float):
        self.lambdas[layer_idx] = max(0.0, min(1.0, value))

    def set_all_lambdas(self, lambda_dict: dict):
        for layer_idx, value in lambda_dict.items():
            l = int(layer_idx)
            if l in self.lambdas:
                self.set_lambda(l, value)
            else:
                logger.warning(f"set_all_lambdas: layer {l} not tracked, ignored.")

    def decay_lambdas(self, rate: float = 0.98):
        """Call once per eval checkpoint, before the agent observes state."""
        for l in self.lambdas:
            self.lambdas[l] = max(0.0, self.lambdas[l] * rate)

    def get_lambdas(self) -> dict:
        return dict(self.lambdas)

    def compute_penalty(self) -> torch.Tensor:
        """
        Differentiable. Call every training step, BEFORE loss.backward().
        Returns a scalar tensor (0.0 tensor if every tracked lambda is 0,
        in which case no layers are even visited — cheap no-op path).
        """
        any_active = any(v > 0.0 for v in self.lambdas.values())
        if not any_active:
            return torch.zeros((), device=self.device)

        penalty = torch.zeros((), device=self.device)
        for layer_idx in self.layer_indices:
            lam = self.lambdas.get(layer_idx, 0.0)
            if lam <= 0.0:
                continue

            P = self.proj_matrices[layer_idx]

            for proj_name in PROJ_NAMES:
                try:
                    layer = self.model.base_model.model.model.layers[layer_idx]
                    proj = getattr(layer.self_attn, proj_name)
                    lora_A = proj.lora_A.default.weight  # [r, d_in], requires_grad=True
                    lora_B = proj.lora_B.default.weight  # [d_out, r], requires_grad=True
                except AttributeError:
                    continue

                # Cast to fp32 for stable penalty computation; cast preserves autograd.
                A = lora_A.float()
                B = lora_B.float()
                delta_W = B @ A                              # [d_out, d_in], WITH grad
                P_dev = P.to(delta_W.device)
                unsafe_component = delta_W @ P_dev            # [d_out, d_in]
                penalty = penalty + lam * (unsafe_component ** 2).sum()

        return penalty

    @torch.no_grad()
    def measure_alignment_ratio(self) -> dict:
        """
        Optional diagnostic, separate from metrics.compute_subspace_alignment
        (kept for parity/debugging — main alignment metric used for the
        agent's observation still comes from src/metrics.py).
        Returns dict[layer_idx -> float] of ||ΔW@P|| / ||ΔW|| per layer.
        """
        ratios = {}
        for layer_idx in self.layer_indices:
            P = self.proj_matrices[layer_idx]
            layer_ratios = []
            for proj_name in PROJ_NAMES:
                try:
                    layer = self.model.base_model.model.model.layers[layer_idx]
                    proj = getattr(layer.self_attn, proj_name)
                    A = proj.lora_A.default.weight.detach().float()
                    B = proj.lora_B.default.weight.detach().float()
                except AttributeError:
                    continue
                delta_W = B @ A
                proj_component = delta_W @ P.to(delta_W.device)
                ratio = proj_component.norm().item() / (delta_W.norm().item() + 1e-9)
                layer_ratios.append(ratio)
            if layer_ratios:
                ratios[layer_idx] = sum(layer_ratios) / len(layer_ratios)
        return ratios