from typing import Tuple

import time
import torch
import torch.nn.functional as F


TIME_START = time.time()
TIME_LIMIT = 60



def get_symbolic_bounds(layer: dict) -> Tuple[torch.tensor, 
                                              torch.tensor, 
                                              torch.tensor, 
                                              torch.tensor]:
    """
    Get symbolic bounds of layer
    """
    weight, bias = layer['weight_bias']
    return weight, weight.clone(), bias, bias.clone()



def get_pos_neg(t: torch.tensor) -> Tuple[torch.tensor, 
                                          torch.tensor]:
    """
    Get matrices of positive and negative coefs
    """ 
    return F.relu(t), -F.relu(-t)



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

    # Get positive and negative weights
    l_weight_pos, l_weight_neg = get_pos_neg(l_weight)
    u_weight_pos, u_weight_neg = get_pos_neg(u_weight)
    
    # Compute lower numerical bound
    l = torch.matmul(l_weight_pos, l_0) + \
        torch.matmul(l_weight_neg, u_0) + l_bias

    # Compute upper numerical bound
    u = torch.matmul(u_weight_pos, u_0) + \
        torch.matmul(u_weight_neg, l_0) + u_bias

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
    assert (u - l).ge(0).all()

    lambda_ = u / (u - l)
    mask_u = mask_1 + mask_2 * lambda_

    # ReLU resolution for weights
    l_weight = l_weight * mask_l.unsqueeze(1)
    u_weight = u_weight * mask_u.unsqueeze(1)

    # ReLU resolution for biases
    l_bias = l_bias                * mask_l
    u_bias = (u_bias - mask_2 * l) * mask_u

    return l_weight, u_weight, l_bias, u_bias
