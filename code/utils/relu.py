from typing import Tuple

import torch


def get_bounds_relu(l:          torch.tensor, 
                    u:          torch.tensor,
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

    # Separate case Strictly positive ( 0 <= l, u ) and case Crossing ReLU ( l < 0 < u )
    # (and implicitly the case Strictly negative ( l, u <= 0 ))
    mask_1 = l.ge(0)
    mask_2 = ~mask_1 & u.gt(0)

    alpha = torch.sigmoid(parameter)
    weight_l = mask_1 + mask_2 * alpha
    
    # lambda_ = u / (u - l)
    lambda_ = torch.where(u != l, u / (u - l), torch.zeros_like(u))
    weight_u = mask_1 + mask_2 * lambda_
 
    # If previous layer was a Convolutional layer
    if len(weight_l.shape) > 2:
        einsum_weight = 'bijkl, bij -> bijkl'
        einsum_bias = 'bij, bij -> bij'
    # If previous layer was a Linear layer
    elif len(weight_l.shape) == 2:
        einsum_weight = 'bij, bi -> bij'
        einsum_bias = 'bi, bi -> bi'
    else:
        assert False

    # ReLU resolution for weights
    l_s_weight = torch.einsum(einsum_weight, l_s_weight, weight_l)
    u_s_weight = torch.einsum(einsum_weight, u_s_weight, weight_u)

    # ReLU resolution for biases
    l_s_bias = torch.einsum(einsum_bias, l_s_bias.clone(), weight_l)
    u_s_bias -= mask_2 * l
    u_s_bias = torch.einsum(einsum_bias, u_s_bias, weight_u)

    return l_s_weight, u_s_weight, l_s_bias, u_s_bias
