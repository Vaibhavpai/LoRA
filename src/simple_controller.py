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
            # Too much drift, tighten constraint
            new_lambda = self.current_lambda + self.delta
        elif refusal_rate > self.target_max:
            # Over-constrained, relax constraint
            new_lambda = self.current_lambda - self.delta
        else:
            # In dead zone, no change
            new_lambda = self.current_lambda

        # Clamp to [0, 1]
        self.current_lambda = max(0.0, min(1.0, new_lambda))
        self.lambda_history.append(self.current_lambda)
        
        logger.info(f"Controller Update: Refusal={refusal_rate:.3f} | "
                    f"New Lambda={self.current_lambda:.3f}")
        return self.current_lambda


def get_lora_A_param(model, layer_idx: int):
    """
    Helper to extract the specific parameter tensor for lora_A from a PEFT model.
    """
    # model.base_model.model.model.layers[layer_idx].self_attn.q_proj.lora_A.default.weight
    # Depending on target_modules, we need to apply to both q_proj and v_proj
    q_proj_A = model.base_model.model.model.layers[layer_idx].self_attn.q_proj.lora_A.default.weight
    v_proj_A = model.base_model.model.model.layers[layer_idx].self_attn.v_proj.lora_A.default.weight
    return [q_proj_A, v_proj_A]


def register_gradient_hooks(model, safety_directions, lambda_state: dict, device: str):
    """
    Registers right-multiply backward hooks on lora_A parameters for all layers.
    
    Args:
        model: The PEFT model.
        safety_directions: Dict mapping layer_idx -> tensor [d_model, k]
        lambda_state: Dict mapping layer_idx -> float. This is a mutable reference 
                      that the hooks will read from on every backward pass.
        device: Device string.
    Returns:
        List of hook handles.
    """
    handles = []
    
    for layer_idx_str, U_l in safety_directions.items():
        layer_idx = int(layer_idx_str)
        
        # Precompute P_l = U_l @ U_l.T outside the hook
        P_l = (U_l @ U_l.T).detach()
        P_l = P_l.requires_grad_(False)
        P_l = P_l.to(device)
        
        # We need a closure factory to capture the layer_idx correctly
        def make_hook(l_idx, proj_mat):
            def hook_fn(grad):
                lam = lambda_state.get(l_idx, 0.0)
                # grad is shape [r, d_in]
                # proj_mat is shape [d_in, d_in]
                # result is [r, d_in]
                return grad - lam * (grad @ proj_mat)
            return hook_fn
        
        try:
            params = get_lora_A_param(model, layer_idx)
            for param in params:
                handle = param.register_hook(make_hook(layer_idx, P_l))
                handles.append(handle)
        except Exception as e:
            logger.error(f"Failed to register hook on layer {layer_idx}: {e}")
            
    return handles
