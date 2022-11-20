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



def get_numerical_bounds(l_0:        torch.tensor, 
                         u_0:        torch.tensor, 
                         l_s_weight: torch.tensor, 
                         u_s_weight: torch.tensor, 
                         l_s_bias:   torch.tensor, 
                         u_s_bias:   torch.tensor) -> Tuple[torch.tensor, 
                                                            torch.tensor]:
    """
    Compute numerical bounds from symbolic ones
    """

    # Get positive and negative weights
    l_s_weight_pos, l_s_weight_neg = get_pos_neg(l_s_weight)
    u_s_weight_pos, u_s_weight_neg = get_pos_neg(u_s_weight)
    
    # Compute lower numerical bound
    l = torch.matmul(l_s_weight_pos, l_0) + \
        torch.matmul(l_s_weight_neg, u_0) + l_s_bias

    # Compute upper numerical bound
    u = torch.matmul(u_s_weight_pos, u_0) + \
        torch.matmul(u_s_weight_neg, l_0) + u_s_bias

    # TODO: To remove
    assert (u - l).ge(0).all()

    return l, u



def deep_poly(l:          torch.tensor, 
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
    
    # TODO: To remove
    assert (u - l).ge(0).all()

    lambda_ = u / (u - l)
    weight_u = mask_1 + mask_2 * lambda_

    # ReLU resolution for weights
    l_s_weight = l_s_weight * weight_l.unsqueeze(1)
    u_s_weight = u_s_weight * weight_u.unsqueeze(1)

    # ReLU resolution for biases
    l_s_bias = l_s_bias                * weight_l
    u_s_bias = (u_s_bias - mask_2 * l) * weight_u

    return l_s_weight, u_s_weight, l_s_bias, u_s_bias
