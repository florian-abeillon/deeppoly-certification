from typing import Tuple

import time
import torch
import torch.nn.functional as F


TIME_START = time.time()
TIME_LIMIT = 60



def get_bound(weight:     torch.tensor, 
              bias:       torch.tensor,
              b_s_weight: torch.tensor,
              b_s_bias:   torch.tensor) -> Tuple[torch.tensor, 
                                                 torch.tensor]:
    """
    Compute symbolic bound
    """
    b_s_weight = torch.matmul(weight, b_s_weight)
    b_s_bias = torch.matmul(weight, b_s_bias) + bias
    return b_s_weight, b_s_bias



def get_bounds(weight:     torch.tensor, 
               bias:       torch.tensor,
               l_s_weight: torch.tensor,
               u_s_weight: torch.tensor,
               l_s_bias:   torch.tensor,
               u_s_bias:   torch.tensor) -> Tuple[torch.tensor, 
                                                  torch.tensor, 
                                                  torch.tensor, 
                                                  torch.tensor]:
    """
    Compute symbolic bounds
    """
    l_s_weight, l_s_bias = get_bound(weight, bias, l_s_weight, l_s_bias)
    u_s_weight, u_s_bias = get_bound(weight, bias, u_s_weight, u_s_bias)
    return l_s_weight, u_s_weight, l_s_bias, u_s_bias



def get_pos_neg(t: torch.tensor) -> Tuple[torch.tensor, 
                                          torch.tensor]:
    """
    Get matrices of positive and negative coefficients of t
    """ 
    return ( F.relu(t), - F.relu(-t) )



def compute_bounds(l_0:        torch.tensor, 
                   u_0:        torch.tensor, 
                   l_s_weight: torch.tensor, 
                   u_s_weight: torch.tensor, 
                   l_s_bias:   torch.tensor, 
                   u_s_bias:   torch.tensor) -> Tuple[torch.tensor, 
                                                      torch.tensor]:
    """
    Compute (non-symbolic) bounds
    """

    # Get positive and negative weights
    l_s_weight_pos, l_s_weight_neg = get_pos_neg(l_s_weight)
    u_s_weight_pos, u_s_weight_neg = get_pos_neg(u_s_weight)
    
    # Compute bounds using lower and upper input bounds (depending on weight signs), and additional bias
    l = torch.matmul(l_s_weight_pos, l_0) + \
        torch.matmul(l_s_weight_neg, u_0) + l_s_bias
    u = torch.matmul(u_s_weight_pos, u_0) + \
        torch.matmul(u_s_weight_neg, l_0) + u_s_bias

    return l, u
