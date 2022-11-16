from typing import Tuple

import torch
import torch.nn.functional as F

from utils import batch_matmatmul, batch_matvecmul, get_pos_neg


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
    l = batch_matvecmul(l_s_weight_pos, l_0) + \
        batch_matvecmul(l_s_weight_neg, u_0) + l_s_bias
    u = batch_matvecmul(u_s_weight_pos, u_0) + \
        batch_matvecmul(u_s_weight_neg, l_0) + u_s_bias

    return l, u



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
    
    lambda_ = u / (u - l)
    weight_u = mask_1 + mask_2 * lambda_

    # ReLU resolution for weights
    l_s_weight *= weight_l.unsqueeze(2)
    u_s_weight *= weight_u.unsqueeze(2)

    # ReLU resolution for biases
    l_s_bias *= weight_l
    u_s_bias -= mask_2 * l
    u_s_bias *= weight_u

    return l_s_weight, u_s_weight, l_s_bias, u_s_bias



def get_bound_conv(weight:     torch.tensor, 
                   b_s_weight: torch.tensor, 
                   k:          int,
                   padding:    Tuple[int],
                   n_out:      int         ) -> torch.tensor:
    """
    Compute symbolic bound from Convolutional layer
    """
    
    # Add constant 0-padding around the matrix
    b_s_weight_padded = F.pad(b_s_weight, padding, "constant", 0)

    # Iterate over every submatrix the size of the kernel
    b_s_weight = torch.empty(0)
    for i in range(n_out):

        b_s_weight_row = torch.empty(0)
        for j in range(n_out):

            # Compute every convolution
            b_s_weight_el = b_s_weight_padded[0, 0, i:i+k, j:j+k] * weight
            b_s_weight_row = torch.cat(( b_s_weight_row, b_s_weight_el ), dim=1)     

        b_s_weight = torch.cat(( b_s_weight, b_s_weight_row )) 

    return b_s_weight



def get_bounds_conv(weight:     torch.tensor, 
                    bias:       torch.tensor,
                    s:          torch.tensor,
                    p:          torch.tensor,
                    l_s_weight: torch.tensor,
                    u_s_weight: torch.tensor,
                    l_s_bias:   torch.tensor,
                    u_s_bias:   torch.tensor) -> Tuple[torch.tensor, 
                                                       torch.tensor, 
                                                       torch.tensor, 
                                                       torch.tensor]:
    """
    Compute symbolic bounds from Convolutional layer
    """

    k = weight.shape[2] 
    n_in = l_s_weight.shape[2]
    n_out = (n_in + 2 * p - k) // s + 1
    padding = (p,) * 4

    # Get weights of output wrt initial input
    l_s_weight = get_bound_conv(weight, l_s_weight, k, padding, n_out)
    u_s_weight = get_bound_conv(weight, u_s_weight, k, padding, n_out)

    # Add bias of current layer
    bias = torch.full((n_out, n_out), bias)
    l_s_bias = bias
    u_s_bias = bias

    return l_s_weight, u_s_weight, l_s_bias, u_s_bias
