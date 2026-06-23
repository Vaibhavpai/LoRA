"""
baselines.py — Phase 4: Static Safety Baselines
=================================================
Implements the core projection logic for two SafeLoRA variants:

  B2A — SafeLoRA Post-Hoc   : Project final LoRA weights after training
  B2B — SafeLoRA In-Training : Project LoRA weights every 100 steps during training

Both variants use the safety directions from Phase 3 (safety_directions.pt).

Key formula (SafeLoRA projection):
    ΔW_safe = ΔW - ΔW @ P_l
    where P_l = U_l @ U_l^T  (projection onto safety subspace, input space)

Note on left vs. right multiply:
    The plan states ΔW_safe = ΔW - U U^T ΔW (left-multiply). Left-multiply
    requires d_in == d_out. For Qwen2.5-1.5B with GQA, v_proj has
    d_out=256, d_in=1536, so left-multiply crashes. Right-multiply
    (ΔW - ΔW @ P) is dimensionally safe for all layers and correctly
    operates in the input space where safety directions live.
"""

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Load safety directions from Phase 3
# ===========================================================================

def load_safety_directions(models_dir: Path) -> tuple[dict, dict]:
    """
    Load safety_directions.pt produced by Phase 3.

    Returns:
        directions : dict mapping layer_idx -> tensor [d_model, k]
        metadata   : dict with model_id, k, d_model, etc.
    """
    path = models_dir / "safety_directions.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"safety_directions.pt not found at {path}.\n"
            f"Run Phase 3 first: python src/subspace_extraction.py"
        )
    payload = torch.load(path, map_location="cpu")
    logger.info(
        f"Loaded safety directions: {payload['metadata']['num_layers']} layers, "
        f"d_model={payload['metadata']['d_model']}, k={payload['metadata']['k']}"
    )
    return payload["directions"], payload["metadata"]


# ===========================================================================
# 2. Precompute projection matrices P_l = U_l @ U_l^T
# ===========================================================================

def build_projection_matrices(
    safety_directions: dict[int, torch.Tensor],
    device: str,
) -> dict[int, torch.Tensor]:
    """
    Precomputes P_l = U_l @ U_l^T for every layer l.

    Shape: [d_model, d_model] per layer.
    P_l projects any vector in R^{d_model} onto the safety subspace.

    Computed once before training, stored on the training device.
    Reused by SafeLoRA B (every 100 steps) and Phase 6 gradient hooks.

    Args:
        safety_directions : Layer -> [d_model, k] tensor from Phase 3.
        device            : 'cuda' or 'cpu'.

    Returns:
        dict[int, torch.Tensor] : Layer -> P_l [d_model, d_model], detached.
    """
    projection_matrices = {}
    for layer_idx, directions in safety_directions.items():
        U = directions.to(torch.float32).to(device)  # [d_model, k]
        P = U @ U.T                                   # [d_model, d_model]
        P = P.detach().requires_grad_(False)          # no gradient through P
        projection_matrices[layer_idx] = P
    logger.info(
        f"Built {len(projection_matrices)} projection matrices "
        f"(each [{list(projection_matrices.values())[0].shape}])"
    )
    return projection_matrices


# ===========================================================================
# 3. Core: project LoRA weights at one layer
# ===========================================================================

def project_lora_layer(
    model,
    layer_idx: int,
    proj_name: str,
    P: torch.Tensor,
) -> bool:
    """
    Applies SafeLoRA projection to the LoRA weight update at one layer.

    Steps:
        1. Compute ΔW = B @ A  (full rank-r weight update, [d_out, d_in])
        2. Project:  ΔW_safe = ΔW - ΔW @ P  (removes input-space safety components)
        3. SVD:      best rank-r approximation of ΔW_safe
        4. Factor:   new_B = U_r * sqrt(S_r),  new_A = sqrt(S_r) * Vh_r
        5. Write new_B, new_A back into model weights in-place.

    Args:
        model      : PEFT model with LoRA adapters.
        layer_idx  : Transformer layer index (0-indexed).
        proj_name  : 'q_proj' or 'v_proj'.
        P          : Projection matrix [d_model, d_model] on correct device.

    Returns:
        bool : True if projection was applied, False if layer had no LoRA.
    """
    try:
        layer = model.base_model.model.model.layers[layer_idx]
        proj  = getattr(layer.self_attn, proj_name)
        lora_A = proj.lora_A.default.weight  # [r, d_in]
        lora_B = proj.lora_B.default.weight  # [d_out, r]
    except AttributeError:
        # This layer or projection doesn't have LoRA — skip silently
        return False

    r          = lora_A.shape[0]
    orig_dtype = lora_A.dtype
    dev        = lora_A.device
    P_dev      = P.to(dev)

    with torch.no_grad():
        A = lora_A.detach().to(torch.float32)  # [r, d_in]
        B = lora_B.detach().to(torch.float32)  # [d_out, r]

        # Step 1: full weight update
        delta_W = B @ A  # [d_out, d_in]

        # Step 2: remove safety-subspace components (right-multiply in input space)
        # delta_W @ P removes the portion of delta_W that lies in the safety subspace
        delta_W_safe = delta_W - delta_W @ P_dev  # [d_out, d_in]

        # Step 3: best rank-r SVD of projected weight update
        U_svd, S_svd, Vh_svd = torch.linalg.svd(delta_W_safe, full_matrices=False)
        U_r   = U_svd[:, :r]   # [d_out, r]
        S_r   = S_svd[:r]      # [r]    — could be 0 for nearly-zero rows
        Vh_r  = Vh_svd[:r, :]  # [r, d_in]

        # Step 4: symmetric factorisation: B_new @ A_new = ΔW_safe (rank-r approx)
        sqrt_S = torch.sqrt(S_r.clamp(min=0.0))
        new_B  = U_r * sqrt_S.unsqueeze(0)    # [d_out, r]
        new_A  = sqrt_S.unsqueeze(1) * Vh_r   # [r, d_in]

        # Step 5: write back in original dtype
        lora_B.data.copy_(new_B.to(orig_dtype))
        lora_A.data.copy_(new_A.to(orig_dtype))

    return True


# ===========================================================================
# 4. Apply projection to all LoRA layers
# ===========================================================================

def project_all_lora_layers(
    model,
    projection_matrices: dict[int, torch.Tensor],
    proj_names: tuple[str, ...] = ("q_proj", "v_proj"),
) -> int:
    """
    Applies SafeLoRA projection to ALL LoRA-equipped layers in-place.

    Called once after training for Variant A (post-hoc),
    or every eval_every steps during training for Variant B.

    Args:
        model               : PEFT model.
        projection_matrices : Layer -> P_l [d_model, d_model].
        proj_names          : Which attention projections to apply to.

    Returns:
        int : Number of (layer, proj) pairs actually projected.
    """
    n_projected = 0
    for layer_idx, P in projection_matrices.items():
        for proj_name in proj_names:
            if project_lora_layer(model, layer_idx, proj_name, P):
                n_projected += 1
    logger.info(
        f"SafeLoRA projection applied to {n_projected} "
        f"(layer, proj) pairs across {len(projection_matrices)} layers."
    )
    return n_projected


# ===========================================================================
# 5. Verify projection is working (quick sanity check)
# ===========================================================================

def verify_projection(
    model,
    projection_matrices: dict[int, torch.Tensor],
    tolerance: float = 1e-3,
    num_layers_to_check: int = 3,
) -> bool:
    """
    Sanity check: after projection, the component of ΔW along the safety
    direction should be near zero for each checked layer.

    Computes ||(ΔW @ U_l).F|| / ||ΔW.F|| — the fraction of ΔW energy
    still in the safety subspace. Should be < tolerance after projection.

    Returns:
        bool : True if all checked layers pass the tolerance check.
    """
    layer_ids = sorted(projection_matrices.keys())
    step = max(1, len(layer_ids) // num_layers_to_check)
    check_layers = layer_ids[::step][:num_layers_to_check]

    all_pass = True
    for layer_idx in check_layers:
        P = projection_matrices[layer_idx]
        try:
            layer  = model.base_model.model.model.layers[layer_idx]
            q_proj = layer.self_attn.q_proj
            A = q_proj.lora_A.default.weight.detach().to(torch.float32)
            B = q_proj.lora_B.default.weight.detach().to(torch.float32)
        except AttributeError:
            continue

        delta_W = B @ A
        dev = delta_W.device
        # Energy in safety subspace after projection
        safety_component = delta_W @ P.to(dev)
        ratio = safety_component.norm().item() / (delta_W.norm().item() + 1e-9)

        status = "✅" if ratio < tolerance else "❌"
        logger.info(
            f"Layer {layer_idx:2d} q_proj | safety-subspace energy ratio: "
            f"{ratio:.5f}  {status}"
        )
        if ratio >= tolerance:
            all_pass = False

    return all_pass