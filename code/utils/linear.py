from typing import Tuple

import torch

from .utils import (
    batch_matmatmul, batch_matvecmul,
    compute_bounds
)



def compute_bounds_linear(l_0:        torch.tensor, 
                          u_0:        torch.tensor, 
                          l_s_weight: torch.tensor, 
                          u_s_weight: torch.tensor, 
                          l_s_bias:   torch.tensor, 
                          u_s_bias:   torch.tensor) -> Tuple[torch.tensor, 
                                                             torch.tensor]:
    """
    Compute numerical bounds for Linear layer
    """
    return compute_bounds(l_0,
                          u_0,
                          l_s_weight,
                          u_s_weight,
                          l_s_bias,
                          u_s_bias,
                          batch_matvecmul)


def get_bound_linear(weight:     torch.tensor, 
                     bias:       torch.tensor,
                     b_s_weight: torch.tensor,
                     b_s_bias:   torch.tensor) -> Tuple[torch.tensor, 
                                                        torch.tensor]:
    """
    Compute symbolic bound from Linear layer
    """
    # Get weights of output wrt initial input (matrix multiplication)
    b_s_weight = batch_matmatmul(weight, b_s_weight)
    # Add bias of current layer
    b_s_bias = batch_matvecmul(weight, b_s_bias) + bias
    return b_s_weight, b_s_bias



def get_bounds_linear(weight:     torch.tensor, 
                      bias:       torch.tensor,
                      l_s_weight: torch.tensor,
                      u_s_weight: torch.tensor,
                      l_s_bias:   torch.tensor,
                      u_s_bias:   torch.tensor) -> Tuple[torch.tensor, 
                                                         torch.tensor, 
                                                         torch.tensor, 
                                                         torch.tensor]:
    """
    Compute symbolic bounds from Linear layer
    """
    # Get lower symbolic bounds
    l_s_weight, l_s_bias = get_bound_linear(weight, bias, l_s_weight, l_s_bias)
    # Get upper symbolic bounds
    u_s_weight, u_s_bias = get_bound_linear(weight, bias, u_s_weight, u_s_bias)
    return l_s_weight, u_s_weight, l_s_bias, u_s_bias
