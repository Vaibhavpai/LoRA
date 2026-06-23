"""
constraint_applier.py — Phase 6 (Hybrid v3)
===========================================
Applies soft weight-space projection to ΔW=B@A at each evaluation step.

This replaces the previous gradient hook mechanism. By operating on ΔW directly,
we correctly constrain the exact quantity the subspace alignment metric measures,
preventing drift via lora_B which was previously unconstrained.

Soft projection formula (generalizes baselines.project_lora_layer):
    ΔW_safe = ΔW - lambda_l * (ΔW @ P_l)
    P_l = U_l @ U_l.T

lambda_l = 1.0 reproduces SafeLoRA-B exactly for that layer.
lambda_l = 0.0 leaves that layer fully untouched (vanilla).
"""

import logging
import torch

logger = logging.getLogger(__name__)

PROJ_NAMES = ("q_proj", "v_proj")


class ConstraintApplier:
    """
    Holds per-layer lambda + projection matrices, and applies soft
    weight-space projection to ΔW=B@A on demand (call apply_projection()).
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

        logger.info(f"ConstraintApplier (weight-space) initialized for {len(self.layer_indices)} layers.")

    def set_lambda(self, layer_idx: int, value: float):
        self.lambdas[layer_idx] = max(0.0, min(1.0, value))

    def set_all_lambdas(self, lambda_dict: dict):
        for layer_idx, value in lambda_dict.items():
            l = int(layer_idx)
            if l in self.lambdas:
                self.set_lambda(l, value)
            else:
                logger.warning(f"set_all_lambdas: layer {l} not tracked, ignored.")

    def get_lambdas(self) -> dict:
        return dict(self.lambdas)

    def _project_one_layer(self, layer_idx: int, proj_name: str, P: torch.Tensor, lam: float) -> bool:
        """Soft version of baselines.project_lora_layer — same math, scaled by lam."""
        try:
            layer = self.model.base_model.model.model.layers[layer_idx]
            proj = getattr(layer.self_attn, proj_name)
            lora_A = proj.lora_A.default.weight
            lora_B = proj.lora_B.default.weight
        except AttributeError:
            return False

        if lam <= 0.0:
            return False  # no-op, leave weights untouched

        r = lora_A.shape[0]
        orig_dtype = lora_A.dtype
        dev = lora_A.device
        P_dev = P.to(dev)

        with torch.no_grad():
            A = lora_A.detach().to(torch.float32)
            B = lora_B.detach().to(torch.float32)

            delta_W = B @ A
            delta_W_safe = delta_W - lam * (delta_W @ P_dev)  # SOFT projection, scaled by lambda

            U_svd, S_svd, Vh_svd = torch.linalg.svd(delta_W_safe, full_matrices=False)
            U_r = U_svd[:, :r]
            S_r = S_svd[:r]
            Vh_r = Vh_svd[:r, :]

            sqrt_S = torch.sqrt(S_r.clamp(min=0.0))
            new_B = U_r * sqrt_S.unsqueeze(0)
            new_A = sqrt_S.unsqueeze(1) * Vh_r

            lora_B.data.copy_(new_B.to(orig_dtype))
            lora_A.data.copy_(new_A.to(orig_dtype))

        return True

    def apply_projection(self) -> int:
        """
        Call this AFTER agent.decide() + set_all_lambdas(), every eval_every
        steps. Projects ΔW for every tracked layer using its current lambda.
        Returns number of (layer, proj) pairs actually modified (lam>0).
        """
        n_applied = 0
        for layer_idx in self.layer_indices:
            lam = self.lambdas.get(layer_idx, 0.0)
            P = self.proj_matrices[layer_idx]
            for proj_name in PROJ_NAMES:
                if self._project_one_layer(layer_idx, proj_name, P, lam):
                    n_applied += 1
        logger.info(f"Weight-space projection applied to {n_applied} (layer, proj) pairs.")
        return n_applied