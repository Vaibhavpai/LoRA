"""
simple_controller.py — Phase 5 (v2, per-projection compatible)
================================================================
Rule-based adaptive controller. Adjusts lambda uniformly across all layers.

v2 update: register_gradient_hooks accepts string-keyed safety directions
("{layer}.{proj}") from the v2 subspace extraction.
"""

import torch
import logging

logger = logging.getLogger(__name__)


class SimpleAdaptiveController:
    """
    Rule-Based Training Controller for Phase 5 (Baseline 4).
    Adjusts the constraint lambda uniformly across all layers based on refusal rate.
    """
    def __init__(self, initial_lambda: float = 0.3, delta: float = 0.05,
                 target_min: float = 0.82, target_max: float = 0.92):
        self.current_lambda = initial_lambda
        self.delta = delta
        self.target_min = target_min
        self.target_max = target_max
        self.lambda_history = []

    def update(self, refusal_rate: float) -> float:
        """
        Updates the uniform lambda based on the current refusal rate.
        Returns the new lambda value.
        """
        if refusal_rate < self.target_min:
            new_lambda = self.current_lambda + self.delta
        elif refusal_rate > self.target_max:
            new_lambda = self.current_lambda - self.delta
        else:
            new_lambda = self.current_lambda

        self.current_lambda = max(0.0, min(1.0, new_lambda))
        self.lambda_history.append(self.current_lambda)

        logger.info(f"Controller Update: Refusal={refusal_rate:.3f} | "
                    f"New Lambda={self.current_lambda:.3f}")
        return self.current_lambda


def register_gradient_hooks(model, safety_directions, lambda_state: dict, device: str):
    """
    Registers right-multiply backward hooks on lora_A parameters.

    v2: accepts string-keyed safety directions ("{layer}.{proj}").
    Parses the key to find layer index and projection name.

    Args:
        model: The PEFT model.
        safety_directions: Dict mapping "{layer}.{proj}" -> tensor [d_model, k]
                          OR int layer_idx -> tensor [d_model, k] (legacy).
        lambda_state: Dict mapping int layer_idx -> float. Mutable reference
                      read by hooks on every backward pass.
        device: Device string.
    Returns:
        List of hook handles.
    """
    handles = []

    for key, U_l in safety_directions.items():
        key_str = str(key)

        # Parse key to get layer index and projection name
        if "." in key_str:
            parts = key_str.split(".")
            layer_idx = int(parts[0])
            proj_name = parts[1]
        else:
            layer_idx = int(key_str)
            proj_name = None  # legacy: apply to both q_proj and v_proj

        # Precompute P_l = U_l @ U_l.T
        P_l = (U_l @ U_l.T).detach().requires_grad_(False).to(device)

        def make_hook(l_idx, proj_mat):
            def hook_fn(grad):
                lam = lambda_state.get(l_idx, 0.0)
                return grad - lam * (grad @ proj_mat)
            return hook_fn

        try:
            layer = model.base_model.model.model.layers[layer_idx]

            if proj_name is not None:
                # v2: specific projection
                proj = getattr(layer.self_attn, proj_name)
                param = proj.lora_A.default.weight
                handle = param.register_hook(make_hook(layer_idx, P_l))
                handles.append(handle)
            else:
                # Legacy: apply to both q_proj and v_proj
                for pn in ("q_proj", "v_proj"):
                    proj = getattr(layer.self_attn, pn, None)
                    if proj is not None and hasattr(proj, 'lora_A'):
                        param = proj.lora_A.default.weight
                        handle = param.register_hook(make_hook(layer_idx, P_l))
                        handles.append(handle)

        except Exception as e:
            logger.error(f"Failed to register hook for {key}: {e}")

    logger.info(f"Registered {len(handles)} gradient hooks.")
    return handles
