from typing import Tuple

import time
import torch


TIME_START = time.time()
TIME_LIMIT = 60



# Initialize symbolic bounds from weight/bias
init_symbolic_bounds = lambda weight, bias: ( weight, weight.clone(), bias, bias.clone() )



def get_pos_neg(weight: torch.tensor) -> Tuple[torch.tensor, 
                                               torch.tensor]:
    """
    Return tensor of positive, and tensor of negative coefficients of weight
    """
    mask = weight > 0
    weight_pos = weight *  mask
    weight_neg = weight * ~mask
    return weight_pos, weight_neg



def backsubstitute_bound(weight_pos:      torch.tensor, 
                         weight_neg:      torch.tensor, 
                         prev_weight:     torch.tensor, 
                         prev_weight_inv: torch.tensor) -> torch.tensor:
    """
    Backsubstitute bound using previous weights
    """
    return torch.matmul(weight_pos, prev_weight) + \
           torch.matmul(weight_neg, prev_weight_inv)



# Compute numerical bound from symbolic one
get_numerical_bound = lambda l, u, weight, bias: backsubstitute_bound(*get_pos_neg(weight), l, u) + bias


def get_numerical_bounds(l_0:      torch.tensor, 
                         u_0:      torch.tensor, 
                         l_weight: torch.tensor, 
                         u_weight: torch.tensor, 
                         l_bias:   torch.tensor, 
                         u_bias:   torch.tensor) -> Tuple[torch.tensor, 
                                                          torch.tensor]:
    """
    Compute numerical bounds from symbolic ones
    """
    l = get_numerical_bound(l_0, u_0, l_weight, l_bias)
    u = get_numerical_bound(u_0, l_0, u_weight, u_bias)
    return l, u



def deep_poly(l:        torch.tensor, 
              u:        torch.tensor,
              param:    torch.tensor,
              l_weight: torch.tensor,
              u_weight: torch.tensor,
              l_bias:   torch.tensor,
              u_bias:   torch.tensor) -> Tuple[torch.tensor, 
                                               torch.tensor, 
                                               torch.tensor, 
                                               torch.tensor]:
    """
    Compute symbolic bounds from ReLU layer
    """

    # Separate cases Strictly positive ( 0 <= l, u ) and Crossing ReLU ( l < 0 < u )
    # (and implicitly Strictly negative ( l, u <= 0 ))
    mask_1 = l.ge(0)
    mask_2 = ~mask_1 & u.gt(0)

    alpha = torch.sigmoid(param)
    mask_l = mask_1 + mask_2 * alpha

    # Add tiny epsilon to circumvent DivBackward error
    lambda_ = torch.where(mask_2, u / (u - l + 1e-12), torch.zeros_like(u))
    mask_u = mask_1 + mask_2 * lambda_

    # ReLU resolution for weights
    l_weight = l_weight * mask_l.unsqueeze(1)
    u_weight = u_weight * mask_u.unsqueeze(1)

    # ReLU resolution for biases
    l_bias = l_bias                * mask_l
    u_bias = (u_bias - mask_2 * l) * mask_u

    return l_weight, u_weight, l_bias, u_bias
