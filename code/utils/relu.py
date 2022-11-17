from typing import Tuple

import torch

from .utils import compute_bounds



def get_bounds_relu(l_0:        torch.tensor, 
                    u_0:        torch.tensor,
                    parameter:  torch.tensor,
                    l_s_weight: torch.tensor,
                    u_s_weight: torch.tensor,
                    l_s_bias:   torch.tensor,
                    u_s_bias:   torch.tensor) -> Tuple[torch.tensor, 
                                                       torch.tensor, 
                                                       torch.tensor, 
                                                       torch.tensor]:
    """
    Compute symbolic bounds from ReLU layer
    """

    # Compute lower and upper numerical bounds for previous layer
    l, u = compute_bounds(l_0, 
                          u_0, 
                          l_s_weight, 
                          u_s_weight, 
                          l_s_bias, 
                          u_s_bias)

    # Separate case Strictly positive ( 0 <= l, u ) and case Crossing ReLU ( l < 0 < u )
    # (and implicitly the case Strictly negative ( l, u <= 0 ))
    mask_1 = l.ge(0)
    mask_2 = ~mask_1 & u.gt(0)

    alpha = torch.sigmoid(parameter)
    weight_l = mask_1 + mask_2 * alpha
    
    lambda_ = u / (u - l)
    weight_u = mask_1 + mask_2 * lambda_

    # ReLU resolution for weights
    l_s_weight *= weight_l.unsqueeze(1)
    u_s_weight *= weight_u.unsqueeze(1)

    # ReLU resolution for biases
    l_s_bias *= weight_l
    u_s_bias -= mask_2 * l
    u_s_bias *= weight_u

    return l_s_weight, u_s_weight, l_s_bias, u_s_bias
