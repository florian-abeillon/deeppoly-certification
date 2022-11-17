from typing import Callable, Tuple

import time
import torch
import torch.nn.functional as F


TIME_START = time.time()
TIME_LIMIT = 60



def get_pos_neg(t: torch.tensor) -> Tuple[torch.tensor, 
                                          torch.tensor]:
    """
    Get matrices of positive and negative coefficients of t
    """ 
    return F.relu(t), - F.relu(-t)


def batch_matmatmul(a: torch.tensor, 
                    b: torch.tensor) -> torch.tensor:
    """
    Matrix multiplication along several dimensions
    """
    return torch.einsum('bij, bjk -> bik', a, b)


def batch_matvecmul(a: torch.tensor, 
                    b: torch.tensor) -> torch.tensor:
    """
    Matrix-vector multiplication along several dimensions
    """
    return torch.einsum('bij, bj -> bi', a, b)


def zero_pad(t: torch.tensor,
             p: int         ) -> torch.tensor:
    """
    Pad matrix with zeros
    """
    padding = (p,) * 4
    return F.pad(t, padding, "constant", 0)


def compute_bounds(l_0:        torch.tensor, 
                   u_0:        torch.tensor, 
                   l_s_weight: torch.tensor, 
                   u_s_weight: torch.tensor, 
                   l_s_bias:   torch.tensor, 
                   u_s_bias:   torch.tensor,
                   operator:   Callable    ) -> Tuple[torch.tensor, 
                                                      torch.tensor]:
    """
    Compute (non-symbolic) bounds
    """

    # Get positive and negative weights
    l_s_weight_pos, l_s_weight_neg = get_pos_neg(l_s_weight)
    u_s_weight_pos, u_s_weight_neg = get_pos_neg(u_s_weight)
    
    # Compute bounds using lower and upper input bounds (depending on weight signs), and additional bias
    l = operator(l_s_weight_pos, l_0) + \
        operator(l_s_weight_neg, u_0) + l_s_bias
    u = operator(u_s_weight_pos, u_0) + \
        operator(u_s_weight_neg, l_0) + u_s_bias

    return l, u
