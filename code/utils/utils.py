from typing import Tuple

import time
import torch


TIME_START = time.time()
TIME_LIMIT = 60



# Initialize symbolic bounds from weight/bias
init_symbolic_bounds = lambda weight, bias: ( weight, weight.clone(), bias, bias.clone() )



def backsubstitute_bound(weight:          torch.tensor, 
                         prev_weight:     torch.tensor, 
                         prev_weight_inv: torch.tensor) -> torch.tensor:
    """
    Backsubstitute bound using previous weights
    """
    mask = weight > 0
    return torch.matmul(weight *  mask, prev_weight) + \
           torch.matmul(weight * ~mask, prev_weight_inv)



# Compute numerical bound from symbolic one
get_numerical_bound = lambda l, u, weight, bias: backsubstitute_bound(weight, l, u) + bias


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



def deep_poly(l:     torch.tensor, 
              u:     torch.tensor,
              param: torch.tensor) -> Tuple[torch.tensor, 
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

    lambda_ = torch.where(mask_2, u / (u - l + 1e-12), torch.zeros_like(u))
    mask_u = mask_1 + mask_2 * lambda_

#     assert lambda_[mask_2].ne(0).all()
#     assert ~lambda_.grad[mask_2].isnan().any()
#     assert ~(1 / lambda_[mask_2]**2).isnan().any()

    # ReLU resolution for weights
    l_weight = torch.diag(mask_l)
    u_weight = torch.diag(mask_u)

    # ReLU resolution for biases
    l_bias = torch.zeros_like(l)
    u_bias = -lambda_ * l

    return l_weight, u_weight, l_bias, u_bias
