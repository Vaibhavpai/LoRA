"""
baselines.py — Phase 4: Static Safety Baselines (v2, per-projection)
=====================================================================
Implements the core projection logic for two SafeLoRA variants:

  B2A — SafeLoRA Post-Hoc   : Project final LoRA weights after training
  B2B — SafeLoRA In-Training : Project LoRA weights every 100 steps during training

v2 changes:
  - String-keyed safety directions: "{layer_idx}.{proj_name}" (56 keys)
  - Scaling-aware projection: accounts for LoRA's scaling = alpha/rank
  - load_safety_directions() auto-detects old int-key vs new string-key format

Key formula (SafeLoRA projection, scaling-aware):
    ΔW_actual = scaling * B @ A
    ΔW_safe   = ΔW_actual - ΔW_actual @ P_l
    Then re-factor ΔW_safe / scaling back into B, A via SVD
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

    Auto-detects format:
      - v2 (per-projection): keys are "{layer_idx}.{proj_name}" strings
      - v1 (legacy): keys are int layer indices → converted to v2 format

    Returns:
        directions : dict mapping string key -> tensor [d_model, k]
        metadata   : dict with model_id, k, d_model, etc.
    """
    path = models_dir / "safety_directions.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"safety_directions.pt not found at {path}.\n"
            f"Run Phase 3 first: python src/subspace_extraction.py"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    raw_directions = payload["directions"]

    # Auto-detect key format
    sample_key = next(iter(raw_directions))
    if isinstance(sample_key, int):
        # v1 legacy format: int keys -> convert to per-projection string keys
        logger.warning(
            "Loaded v1 (int-keyed) safety directions. Converting to v2 per-projection format. "
            "Each layer's directions will be shared between q_proj and v_proj."
        )
        directions = {}
        for layer_idx, tensor in raw_directions.items():
            directions[f"{layer_idx}.q_proj"] = tensor
            directions[f"{layer_idx}.v_proj"] = tensor
    else:
        # v2 format: string keys
        directions = raw_directions

    logger.info(
        f"Loaded safety directions: {len(directions)} projection points, "
        f"d_model={metadata['d_model']}, k={metadata['k']}"
    )
    return directions, metadata


# ===========================================================================
# 2. Precompute projection matrices P = U @ U^T
# ===========================================================================

def build_projection_matrices(
    safety_directions: dict[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    """
    Precomputes P = U @ U^T for every projection point.

    Shape: [d_model, d_model] per key.
    P projects any vector in R^{d_model} onto the safety subspace.

    Args:
        safety_directions : "{layer}.{proj}" -> [d_model, k] tensor.
        device            : 'cuda' or 'cpu'.

    Returns:
        dict[str, torch.Tensor] : key -> P [d_model, d_model], detached.
    """
    projection_matrices = {}
    for key, directions in safety_directions.items():
        U = directions.to(torch.float32).to(device)  # [d_model, k]
        P = U @ U.T                                   # [d_model, d_model]
        P = P.detach().requires_grad_(False)
        projection_matrices[key] = P

    sample_shape = list(projection_matrices.values())[0].shape if projection_matrices else "N/A"
    logger.info(
        f"Built {len(projection_matrices)} projection matrices "
        f"(each {sample_shape})"
    )
    return projection_matrices


# ===========================================================================
# 3. Utility: resolve model module from string key
# ===========================================================================

def _resolve_lora_module(model, key: str):
    """
    Given a key like "24.q_proj", returns the LoRA-wrapped projection module
    and its components (lora_A, lora_B, scaling).

    Returns:
        (proj_module, lora_A_weight, lora_B_weight, scaling) or None if not found.
    """
    parts = key.split(".")
    layer_idx = int(parts[0])
    proj_name = parts[1]

    try:
        layer = model.base_model.model.model.layers[layer_idx]
        proj = getattr(layer.self_attn, proj_name)
        lora_A = proj.lora_A.default.weight  # [r, d_in]
        lora_B = proj.lora_B.default.weight  # [d_out, r]

        # Get LoRA scaling factor (alpha / rank)
        scaling = 1.0
        if hasattr(proj, 'scaling'):
            if isinstance(proj.scaling, dict):
                scaling = proj.scaling.get("default", 1.0)
            elif isinstance(proj.scaling, (int, float)):
                scaling = float(proj.scaling)

        return proj, lora_A, lora_B, scaling
    except AttributeError:
        return None


# ===========================================================================
# 4. Core: project LoRA weights at one projection point (scaling-aware)
# ===========================================================================

def project_lora_layer(
    model,
    key: str,
    P: torch.Tensor,
) -> bool:
    """
    Applies scaling-aware SafeLoRA projection to the LoRA weight update.

    Steps (v2, scaling-aware):
        1. Compute ΔW_actual = scaling * B @ A   (actual contribution to output)
        2. Project:  ΔW_safe = ΔW_actual - ΔW_actual @ P
        3. Un-scale: ΔW_safe_unscaled = ΔW_safe / scaling
        4. SVD:      best rank-r approximation of ΔW_safe_unscaled
        5. Factor:   new_B = U_r * sqrt(S_r),  new_A = sqrt(S_r) * Vh_r
        6. Write new_B, new_A back into model weights in-place.

    Args:
        model : PEFT model with LoRA adapters.
        key   : "{layer_idx}.{proj_name}" string key.
        P     : Projection matrix [d_model, d_model] on correct device.

    Returns:
        bool : True if projection was applied, False if not found.
    """
    resolved = _resolve_lora_module(model, key)
    if resolved is None:
        return False

    proj, lora_A, lora_B, scaling = resolved
    r = lora_A.shape[0]
    orig_dtype = lora_A.dtype
    dev = lora_A.device
    P_dev = P.to(dev)

    with torch.no_grad():
        A = lora_A.detach().to(torch.float32)  # [r, d_in]
        B = lora_B.detach().to(torch.float32)  # [d_out, r]

        # Step 1: actual weight update (scaling-aware)
        delta_W = scaling * (B @ A)  # [d_out, d_in]

        # Step 2: remove safety-subspace components
        delta_W_safe = delta_W - delta_W @ P_dev  # [d_out, d_in]

        # Step 3: un-scale for re-factoring back into B, A
        delta_W_unscaled = delta_W_safe / (scaling if scaling != 0 else 1.0)

        # Step 4: best rank-r SVD
        U_svd, S_svd, Vh_svd = torch.linalg.svd(delta_W_unscaled, full_matrices=False)
        U_r = U_svd[:, :r]
        S_r = S_svd[:r]
        Vh_r = Vh_svd[:r, :]

        # Step 5: symmetric factorisation
        sqrt_S = torch.sqrt(S_r.clamp(min=0.0))
        new_B = U_r * sqrt_S.unsqueeze(0)    # [d_out, r]
        new_A = sqrt_S.unsqueeze(1) * Vh_r   # [r, d_in]

        # Step 6: write back
        lora_B.data.copy_(new_B.to(orig_dtype))
        lora_A.data.copy_(new_A.to(orig_dtype))

    return True


# ===========================================================================
# 5. Apply projection to all LoRA projections
# ===========================================================================

def project_all_lora_layers(
    model,
    projection_matrices: dict[str, torch.Tensor],
) -> int:
    """
    Applies SafeLoRA projection to ALL LoRA-equipped projections in-place.

    Args:
        model               : PEFT model.
        projection_matrices : key -> P [d_model, d_model].

    Returns:
        int : Number of projections actually applied.
    """
    n_projected = 0
    for key, P in projection_matrices.items():
        if project_lora_layer(model, key, P):
            n_projected += 1
    logger.info(
        f"SafeLoRA projection applied to {n_projected} projection points "
        f"across {len(projection_matrices)} keys."
    )
    return n_projected


# ===========================================================================
# 6. Verify projection is working
# ===========================================================================

def verify_projection(
    model,
    projection_matrices: dict[str, torch.Tensor],
    tolerance: float = 1e-3,
    num_to_check: int = 6,
) -> bool:
    """
    Sanity check: after projection, the component of ΔW along the safety
    direction should be near zero for each checked projection point.

    Returns:
        bool : True if all checked points pass the tolerance check.
    """
    keys = sorted(projection_matrices.keys())
    step = max(1, len(keys) // num_to_check)
    check_keys = keys[::step][:num_to_check]

    all_pass = True
    for key in check_keys:
        P = projection_matrices[key]
        resolved = _resolve_lora_module(model, key)
        if resolved is None:
            continue

        _, lora_A, lora_B, scaling = resolved
        A = lora_A.detach().to(torch.float32)
        B = lora_B.detach().to(torch.float32)

        delta_W = scaling * (B @ A)
        dev = delta_W.device
        safety_component = delta_W @ P.to(dev)
        ratio = safety_component.norm().item() / (delta_W.norm().item() + 1e-9)

        status = "✅" if ratio < tolerance else "❌"
        logger.info(
            f"{key:15s} | safety-subspace energy ratio: {ratio:.5f}  {status}"
        )
        if ratio >= tolerance:
            all_pass = False

    return all_pass