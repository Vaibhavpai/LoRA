"""
salora.py — Phase 4, Step 4.2: SaLoRA (Baseline 3)
=====================================================
Paper-accurate implementation of SaLoRA (Safety-Alignment Preserved
Low-Rank Adaptation) following the ICLR 2025 paper by Li et al.

SaLoRA has THREE components (all implemented here):

1. Fixed Safety Module C_SaLoRA = I - U_C @ U_C^T
   - U_C: top-r_s LEFT singular vectors of W·X_h (per-layer)
   - X_h: input activations from ~300 harmful prompts + safe responses
   - Applied as a forward hook: h = W'x + C_SaLoRA(B·A·x)

2. Task-Specific Initialization of Adapters
   - SVD of base weight W = U_bar · S_bar · V_bar^T
   - B_SaLoRA = U_bar[:, :r] · sqrt(S_bar[:r, :r])
   - A_SaLoRA = sqrt(S_bar[:r, :r]) · V_bar[:, :r]^T

3. Weight Re-parameterization
   - W' = W - C_SaLoRA · B_0 · A_0
   - Ensures output is unchanged at init: W'x + C·(B_0·A_0·x) = Wx

Implementation uses PyTorch forward hooks on LoRA-equipped linear
layers — clean, non-invasive, no forked PEFT required.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Extract per-layer safety directions from W·X_h
# ===========================================================================

class _ActivationCapture:
    """Captures input activations at a module and accumulates their uncentered covariance online."""
    def __init__(self):
        self.cov = None
        self.hook = None

    def register(self, module):
        self.hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input_args, output):
        # input_args[0] shape: [batch, seq_len, d_in]
        # Flatten batch and seq_len dimensions: [N, d_in]
        x = input_args[0].detach().float().cpu()
        x_flat = x.reshape(-1, x.shape[-1])
        
        # cov is X_h^T X_h. Accumulate online to save huge amounts of memory.
        # This keeps the memory footprint to just [d_in, d_in] instead of [N, d_in].
        batch_cov = x_flat.T @ x_flat
        
        if self.cov is None:
            self.cov = batch_cov
        else:
            self.cov += batch_cov

    def remove(self):
        if self.hook is not None:
            self.hook.remove()

    def get_covariance(self):
        return self.cov


def extract_per_layer_directions(
    model,
    tokenizer,
    harmful_prompts: list[str],
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    r_s: int = 5,
    device: str = "cuda",
    max_prompts: int = 300,
    max_length: int = 128,
) -> dict[tuple[int, str], torch.Tensor]:
    """
    Extracts per-layer safety directions U_C as described in the SaLoRA paper.

    For each (layer, proj_name):
      1. Collect input activations X_h from harmful prompts ONLY
      2. Compute output features: Y_h = W @ X_h^T  (shape [d_out, N])
      3. SVD of Y_h -> take top-r_s left singular vectors = U_C
      4. Build C = I - U_C @ U_C^T

    Args:
        model           : Base model (NOT PEFT-wrapped yet).
        tokenizer       : Tokenizer for the model.
        harmful_prompts : List of harmful instruction strings.
        target_modules  : Which projection layers to extract for.
        r_s             : Safety rank (number of directions to keep).
        device          : Device string.
        max_prompts     : Max number of prompts to use (paper uses ~300).
        max_length      : Max token length per prompt.

    Returns:
        dict[(layer_idx, proj_name)] -> U_C tensor of shape [d_out, r_s],
        representing the top safety-critical output directions for that layer.
    """
    logger.info(
        f"Extracting per-layer SaLoRA directions (r_s={r_s}, "
        f"max_prompts={max_prompts}) using pure HARMFUL subspace..."
    )

    # Limit to max_prompts
    all_prompts = harmful_prompts[:max_prompts]
    logger.info(f"Using {len(all_prompts)} harmful prompts")

    model.eval()

    # Pre-generate safe responses BEFORE attaching hooks
    # We cache these because generation takes several minutes and is deterministic.
    import os
    from pathlib import Path
    
    # Path handling robust to where the script is run from
    project_root = Path(os.path.abspath(__file__)).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    cache_file = data_dir / f"salora_conversations_cache_{max_prompts}.pt"
    
    if cache_file.exists():
        logger.info(f"Loading cached safe responses from {cache_file}...")
        full_conversations_ids = torch.load(cache_file, map_location=device)
    else:
        logger.info(f"Generating safe responses for {len(all_prompts)} harmful prompts...")
        logger.info("This will take a few minutes but will be cached for future runs.")
        full_conversations_ids = []
        
        with torch.no_grad():
            for i, prompt in enumerate(all_prompts):
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
                inputs = tokenizer(
                    formatted, return_tensors="pt",
                    max_length=max_length, truncation=True,
                ).to(device)
                
                # Generate the response (Qwen is aligned, so it will refuse)
                outputs = model.generate(
                    **inputs, max_new_tokens=64, 
                    pad_token_id=tokenizer.eos_token_id, 
                    do_sample=False
                )
                full_conversations_ids.append(outputs[0])
                
                if (i + 1) % 50 == 0:
                    logger.info(f"  Generated {i + 1}/{len(all_prompts)} responses")
        
        # Save to cache
        logger.info(f"Saving generated responses to cache: {cache_file}")
        torch.save(full_conversations_ids, cache_file)

    # Identify all target layers
    num_layers = model.config.num_hidden_layers
    captures = {}  # (layer_idx, proj_name) -> _ActivationCapture

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for proj_name in target_modules:
            proj_module = getattr(layer.self_attn, proj_name, None)
            if proj_module is not None:
                cap = _ActivationCapture()
                cap.register(proj_module)
                captures[(layer_idx, proj_name)] = cap

    logger.info(f"Registered {len(captures)} activation captures")

    # Run all full conversations through the model to collect activations
    logger.info("Passing full conversations through the model to capture activations...")
    with torch.no_grad():
        for i, input_ids in enumerate(full_conversations_ids):
            # input_ids is 1D, shape it to [1, seq_len]
            inputs = {"input_ids": input_ids.unsqueeze(0).to(device)}

            model(**inputs)

            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(full_conversations_ids)} conversations")

    # Remove hooks
    for cap in captures.values():
        cap.remove()

    # Compute per-layer U_C from SVD of W·X_h
    per_layer_directions = {}

    for (layer_idx, proj_name), cap in captures.items():
        cov = cap.get_covariance()  # [d_in, d_in]
        if cov is None:
            logger.warning(f"No activations for layer {layer_idx} {proj_name}")
            continue

        # Get the base weight W: [d_out, d_in]
        layer = model.model.layers[layer_idx]
        proj_module = getattr(layer.self_attn, proj_name)
        W = proj_module.weight.detach().float().to(device)
        d_out, d_in = W.shape
        
        cov = cov.to(device)

        # We want the left singular vectors of Y_h = W @ X_h^T.
        # These are exactly the eigenvectors of Y_h Y_h^T.
        # Y_h Y_h^T = (W @ X_h^T) @ (X_h @ W^T) = W @ (X_h^T @ X_h) @ W^T = W @ cov @ W^T
        Y_h_cov = W @ cov @ W.T  # [d_out, d_out]

        # Eigen decomposition (since Y_h_cov is symmetric positive semi-definite)
        # eigh returns eigenvalues in ascending order, eigenvectors in columns.
        eigenvalues, eigenvectors = torch.linalg.eigh(Y_h_cov)
        
        # Take the top r_s eigenvectors (the last r_s columns)
        actual_r_s = min(r_s, d_out)
        U_C = eigenvectors[:, -actual_r_s:]  # [d_out, actual_r_s]
        
        # Optionally, flip the order so the largest is first (like SVD output)
        U_C = torch.flip(U_C, dims=[1])

        per_layer_directions[(layer_idx, proj_name)] = U_C.cpu()

        logger.debug(
            f"Layer {layer_idx} {proj_name}: W=[{d_out},{d_in}], "
            f"U_C=[{U_C.shape[0]},{U_C.shape[1]}]"
        )

        # Free memory
        del cov, Y_h_cov, eigenvalues, eigenvectors
        cap.cov = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info(
        f"Extracted {len(per_layer_directions)} per-layer direction sets "
        f"(r_s={r_s})"
    )

    return per_layer_directions


# ===========================================================================
# 2. Task-specific initialization of LoRA adapters (from SVD of base W)
# ===========================================================================

def initialize_salora_adapters(model, device: str = "cuda"):
    """
    Applies SaLoRA's task-specific initialization to all LoRA adapters.

    Paper formula (Eqn 10):
        B_SaLoRA = U_bar[:, :r] @ sqrt(S_bar[:r, :r])   shape [d_out, r]
        A_SaLoRA = sqrt(S_bar[:r, :r]) @ V_bar[:, :r]^T  shape [r, d_in]

    Where U_bar, S_bar, V_bar come from SVD of the base weight W.

    This replaces the default LoRA init (A=Gaussian, B=Zero) with values
    derived from the base weight's most important singular components.
    """
    logger.info("Applying SaLoRA task-specific initialization...")
    n_initialized = 0

    for name, module in model.named_modules():
        # Find LoRA-wrapped linear layers
        if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
            continue
        if not hasattr(module, 'base_layer'):
            continue

        # Get base weight: [d_out, d_in]
        W = module.base_layer.weight.detach().float().to(device)
        d_out, d_in = W.shape

        # Get LoRA rank from the existing adapter shape
        # lora_A['default'].weight has shape [r, d_in]
        for adapter_name in module.lora_A:
            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]
            r = lora_A.weight.shape[0]

            # SVD of base weight
            actual_r = min(r, d_out, d_in)
            U_bar, S_bar, V_bar = torch.svd_lowrank(W, q=actual_r)
            # U_bar: [d_out, r], S_bar: [r], V_bar: [d_in, r]

            sqrt_S = torch.sqrt(S_bar[:actual_r])

            # B_SaLoRA = U_bar[:, :r] @ diag(sqrt(S_bar[:r]))  -> [d_out, r]
            B_init = U_bar[:, :actual_r] * sqrt_S.unsqueeze(0)  # broadcast multiply

            # A_SaLoRA = diag(sqrt(S_bar[:r])) @ V_bar[:, :r]^T  -> [r, d_in]
            A_init = sqrt_S.unsqueeze(1) * V_bar[:, :actual_r].T  # broadcast multiply

            # Apply the initialization
            with torch.no_grad():
                lora_A.weight.copy_(A_init.to(lora_A.weight.dtype))
                lora_B.weight.copy_(B_init.to(lora_B.weight.dtype))

            n_initialized += 1
            logger.debug(
                f"Initialized {name} adapter '{adapter_name}': "
                f"W=[{d_out},{d_in}], r={actual_r}"
            )

            del U_bar, S_bar, V_bar, sqrt_S, B_init, A_init

    logger.info(f"Task-specific initialization applied to {n_initialized} adapters.")
    return n_initialized


# ===========================================================================
# 3. Weight re-parameterization: W' = W - C · B_0 · A_0
# ===========================================================================

def reparameterize_base_weights(
    model,
    C_matrices: dict[tuple[int, str], torch.Tensor],
    device: str = "cuda",
):
    """
    Applies weight re-parameterization so the model output is unchanged
    at initialization.

    Paper formula (Eqn 11):
        W' = W - C_SaLoRA · B_0 · A_0

    At init: W'x + C(B_0·A_0·x) = (W - C·B_0·A_0)x + C·B_0·A_0·x = Wx ✓

    This modifies the base_layer weight in-place.
    """
    logger.info("Applying weight re-parameterization W' = W - C·B_0·A_0...")
    n_reparamed = 0

    for (layer_idx, proj_name), C in C_matrices.items():
        try:
            layer = model.base_model.model.model.layers[layer_idx]
            proj_module = getattr(layer.self_attn, proj_name)

            if not hasattr(proj_module, 'lora_A') or not hasattr(proj_module, 'base_layer'):
                continue

            C_dev = C.to(device).float()

            for adapter_name in proj_module.lora_A:
                A_0 = proj_module.lora_A[adapter_name].weight.detach().float().to(device)  # [r, d_in]
                B_0 = proj_module.lora_B[adapter_name].weight.detach().float().to(device)  # [d_out, r]

                # Get scaling factor
                scaling = proj_module.scaling.get(adapter_name, 1.0)

                # C · B_0 · A_0  -> [d_out, d_in]
                correction = scaling * (C_dev @ B_0 @ A_0)

                # W' = W - correction
                W = proj_module.base_layer.weight.detach().float().to(device)
                W_prime = W - correction

                with torch.no_grad():
                    proj_module.base_layer.weight.copy_(
                        W_prime.to(proj_module.base_layer.weight.dtype)
                    )

                n_reparamed += 1
                logger.debug(
                    f"Re-parameterized layer {layer_idx} {proj_name} "
                    f"(correction norm: {correction.norm().item():.6f})"
                )

                del A_0, B_0, correction, W, W_prime

        except AttributeError as e:
            logger.warning(
                f"Could not reparameterize layer {layer_idx} {proj_name}: {e}"
            )

    logger.info(f"Re-parameterized {n_reparamed} base weights.")
    return n_reparamed


# ===========================================================================
# 4. Build C_SaLoRA matrices from per-layer U_C directions
# ===========================================================================

def build_C_matrices(
    per_layer_directions: dict[tuple[int, str], torch.Tensor],
    device: str = "cuda",
) -> dict[tuple[int, str], torch.Tensor]:
    """
    Builds C = I - U_C @ U_C^T for each (layer, proj_name) pair.

    Args:
        per_layer_directions: dict[(layer_idx, proj_name)] -> U_C [d_out, r_s]

    Returns:
        dict[(layer_idx, proj_name)] -> C matrix [d_out, d_out]
    """
    C_matrices = {}

    for key, U_C in per_layer_directions.items():
        U_C = U_C.to(device).float()
        d_out = U_C.shape[0]
        C = torch.eye(d_out, device=device, dtype=torch.float32) - U_C @ U_C.T
        C = C.detach().requires_grad_(False)
        C_matrices[key] = C

    logger.info(f"Built {len(C_matrices)} C_SaLoRA projection matrices.")
    return C_matrices


# ===========================================================================
# 5. Forward hook: apply C_SaLoRA projection to LoRA output
# ===========================================================================

class SaLoRAHook:
    """
    Registers forward hooks on LoRA-equipped linear layers to apply the
    SaLoRA safety projection during every forward pass.

    The hook intercepts the module output and modifies only the LoRA
    contribution (adapter output), projecting it through C_SaLoRA.

    Standard PEFT LoRA forward:
        output = base_layer(x) + scaling * dropout(lora_B(lora_A(x)))

    With SaLoRA hook:
        base_output = base_layer(x)
        lora_output = scaling * dropout(lora_B(lora_A(x)))
        output = base_output + C_SaLoRA @ lora_output

    We compute:
        1. full_output = base_output + lora_output  (what PEFT gives us)
        2. corrected   = full_output - P @ lora_output
                       = base_output + C @ lora_output   (SaLoRA formula)

    Where P = U_C @ U_C^T = I - C.
    """

    def __init__(
        self,
        model,
        C_matrices: dict[tuple[int, str], torch.Tensor],
    ):
        self.model = model
        self.C_matrices = C_matrices
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on each LoRA-equipped projection layer."""
        for (layer_idx, proj_name), C in self.C_matrices.items():
            try:
                layer = self.model.base_model.model.model.layers[layer_idx]
                proj_module = getattr(layer.self_attn, proj_name)

                # Build the projection-out matrix P = I - C = U_C @ U_C^T
                d_out = C.shape[0]
                P = (torch.eye(d_out, device=C.device, dtype=C.dtype) - C).detach()

                hook = proj_module.register_forward_hook(
                    self._make_hook(proj_module, P, layer_idx, proj_name)
                )
                self.hooks.append(hook)
                logger.debug(
                    f"Registered SaLoRA hook: layer {layer_idx} {proj_name}"
                )
            except AttributeError as e:
                logger.warning(
                    f"Could not register hook for layer {layer_idx} {proj_name}: {e}"
                )

        logger.info(f"Registered {len(self.hooks)} SaLoRA forward hooks.")

    @staticmethod
    def _make_hook(proj_module, P, layer_idx, proj_name):
        """
        Creates a forward hook that applies the SaLoRA correction.

        We isolate the true LoRA contribution by subtracting the base
        layer output from the total output:

            lora_output = output - base_layer(x)          # exact
            corrected   = output - P @ lora_output
                        = base_output + (I - P) @ lora_output
                        = base_output + C @ lora_output   # SaLoRA formula

        Since P = U_C @ U_C^T is symmetric, matmul(lora_output, P.T)
        is equivalent to left-multiplying P @ lora_output on each
        feature vector in the batch.

        Gradients flow correctly: d(corrected)/d(lora_params) is projected
        through C = (I - P), constraining LoRA updates to be orthogonal
        to the safety subspace.
        """
        def hook_fn(module, input_args, output):
            x = input_args[0]

            if not hasattr(module, 'active_adapters') or not module.active_adapters:
                return output

            if module.disable_adapters or module.merged:
                return output

            # Recompute the base layer output to isolate the true LoRA
            # contribution.  base weights are frozen, so no_grad saves memory.
            with torch.no_grad():
                base_out = module.base_layer(x)

            # True LoRA contribution = total output − base output
            lora_output = output - base_out.to(output.dtype)

            # Project out the safety-subspace component
            # P is symmetric (U @ U^T), so P.T == P
            P_dev = P.to(output.device).to(output.dtype)
            correction = torch.matmul(lora_output, P_dev.T)

            return output - correction

        return hook_fn

    def remove_hooks(self):
        """Remove all registered hooks (for cleanup after training)."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        logger.info("Removed all SaLoRA forward hooks.")


# ===========================================================================
# 6. Full SaLoRA setup pipeline
# ===========================================================================

def setup_salora(
    model,
    per_layer_directions: dict[tuple[int, str], torch.Tensor],
    device: str = "cuda",
) -> SaLoRAHook:
    """
    Full paper-accurate SaLoRA setup on a PEFT model.

    Steps:
      1. Build C_SaLoRA matrices from per-layer U_C directions
      2. Apply task-specific initialization of LoRA A, B adapters
      3. Re-parameterize base weights: W' = W - C·B_0·A_0
      4. Register forward hooks for safety projection

    Args:
        model                : PEFT model with LoRA adapters.
        per_layer_directions : From extract_per_layer_directions().
        device               : Training device.

    Returns:
        SaLoRAHook : The hook manager (call .remove_hooks() to clean up).
    """
    logger.info("=" * 60)
    logger.info("Setting up paper-accurate SaLoRA (Li et al., ICLR 2025)")
    logger.info("=" * 60)

    # Step 1: Build C matrices
    C_matrices = build_C_matrices(per_layer_directions, device)

    # Step 2: Task-specific initialization
    n_init = initialize_salora_adapters(model, device)

    # Step 3: Weight re-parameterization
    n_reparam = reparameterize_base_weights(model, C_matrices, device)

    # Step 4: Register forward hooks
    hook_manager = SaLoRAHook(model, C_matrices)

    logger.info(
        f"SaLoRA setup complete: "
        f"{len(C_matrices)} C matrices, "
        f"{n_init} adapters initialized, "
        f"{n_reparam} weights re-parameterized, "
        f"{len(hook_manager.hooks)} hooks registered."
    )
    return hook_manager


# ===========================================================================
# 7. Verify SaLoRA is working (sanity check)
# ===========================================================================

def verify_salora_active(
    model,
    tokenizer,
    hook_manager: SaLoRAHook,
    device: str,
    test_prompt: str = "Hello, how are you?",
) -> bool:
    """
    Quick sanity check that SaLoRA hooks are active and modifying outputs.

    With task-specific initialization, B is NOT zero at init, so the
    hooks should produce a measurable difference right away (unlike
    the old implementation where B=0 meant no LoRA contribution).

    Returns:
        bool : True if outputs differ (hooks are working).
    """
    model.eval()

    messages = [{"role": "user", "content": test_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        # Forward with hooks (SaLoRA active)
        out_with = model(**inputs, output_hidden_states=False)
        logits_with = out_with.logits[:, -1, :].clone()

        # Temporarily remove hooks
        hook_manager.remove_hooks()
        out_without = model(**inputs, output_hidden_states=False)
        logits_without = out_without.logits[:, -1, :].clone()

        # Re-register hooks
        hook_manager._register_hooks()

    diff = (logits_with - logits_without).abs().max().item()
    hooks_working = diff > 1e-6

    logger.info(
        f"SaLoRA verification: max logit difference = {diff:.6f} "
        f"({'✅ hooks active' if hooks_working else '❌ hooks NOT modifying output'})"
    )
    return hooks_working
