from typing import Tuple

import time
import torch
import torch.nn.functional as F


TIME_START = time.time()
TIME_LIMIT = 60



def get_pos_neg(t: torch.tensor) -> Tuple[torch.tensor, 
                                          torch.tensor]:
    """
    Get matrices of positive and negative coefs
    """ 
    return F.relu(t), -F.relu(-t)



def backsubstitute_bound(weight:          torch.tensor, 
                         prev_weight:     torch.tensor, 
                         prev_weight_inv: torch.tensor) -> torch.tensor:
    """
    Backsubstitute bound using previous weights
    """
    weight_pos, weight_neg = get_pos_neg(weight)
    return torch.matmul(weight_pos, prev_weight) + \
           torch.matmul(weight_neg, prev_weight_inv)



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
    l = backsubstitute_bound(l_weight, l_0, u_0) + l_bias
    u = backsubstitute_bound(u_weight, u_0, l_0) + u_bias

    # TODO: To remove
    assert (u - l).ge(0).all()

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

    # Separate case Strictly positive ( 0 <= l, u ) and case Crossing ReLU ( l < 0 < u )
    # (and implicitly the case Strictly negative ( l, u <= 0 ))
    mask_1 = l.ge(0)
    mask_2 = ~mask_1 & u.gt(0)

    alpha = torch.sigmoid(param)
    mask_l = mask_1 + mask_2 * alpha
    
    # TODO: To remove
    assert u.ge(l).all()

    lambda_ = u / (u - l)
    mask_u = mask_1 + mask_2 * lambda_

    # ReLU resolution for weights
    l_weight = l_weight * mask_l.unsqueeze(1)
    u_weight = u_weight * mask_u.unsqueeze(1)

    # ReLU resolution for biases
    l_bias = l_bias                * mask_l
    u_bias = (u_bias - mask_2 * l) * mask_u

    return l_weight, u_weight, l_bias, u_bias
