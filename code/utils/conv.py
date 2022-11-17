from typing import Tuple

import torch
import torch.nn.functional as F

from .utils import compute_bounds, zero_pad



def batch_convolution(b_s_weight: torch.tensor, 
                      b_0:        torch.tensor) -> torch.tensor:
    """
    Convolution along several dimensions
    """
    return torch.einsum('bijkl, bkl -> bij', b_s_weight, b_0)



def compute_bounds_conv(l_0:        torch.tensor, 
                        u_0:        torch.tensor, 
                        l_s_weight: torch.tensor, 
                        u_s_weight: torch.tensor, 
                        l_s_bias:   torch.tensor, 
                        u_s_bias:   torch.tensor) -> Tuple[torch.tensor, 
                                                           torch.tensor]:
    """
    Compute (non-symbolic) bounds for Convolutional layer
    """
    return compute_bounds(l_0,
                          u_0,
                          l_s_weight,
                          u_s_weight,
                          l_s_bias,
                          u_s_bias,
                          batch_convolution)



def get_bound_conv(weight:     torch.tensor, 
                   b_s_weight: torch.tensor, 
                   c_out:      int, 
                   c_in:       int, 
                   k:          int,
                   p:          int,
                   s:          int,
                   n_in:       int,
                   n_out:      int         ) -> torch.tensor:
    """
    Compute symbolic bound from Convolutional layer
    """
    
    # Add constant 0-padding around the matrix
    b_s_weight_padded = zero_pad(b_s_weight, p)
    n_in += 2

    return torch.cat([
        
        # For every out_channel
        torch.cat([

            # Isolate every submatrix of size (k, k)
            torch.cat([

                # Add padding to reconstruct padded input shape
                # Then flatten into a list
                F.pad(

                    # Sum convolutions element-wise on all in_channels
                    torch.sum(

                        torch.cat([

                            # Compute every convolution
                            (
                                b_s_weight_padded[b_in, i*s:i*s+k, j*s:j*s+k] * weight[b_out, b_in]
                            ).unsqueeze(0)

                            for b_in in range(c_in)
                        
                        ]),
                        dim=0

                    ),
                    ( j, n_in - k - j, i, n_in - k - i ), 
                    "constant", 
                    0

                ).unsqueeze(0)
                # ).to_sparse().unsqueeze(0)

                for j in range(n_out)

            ]).unsqueeze(0)  
            for i in range(n_out)

        ]).unsqueeze(0)
        for b_out in range(c_out)

    ])



def compute_n_out(n_in: int,
                  p:    int,
                  k:    int,
                  s:    int) -> int:
    """
    Compute out_dimension of Convolutional layer
    """
    return (n_in + 2 * p - k) // s + 1



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

    c_out, c_in, k, _ = weight.shape
    n_in = l_s_weight.shape[1]
    n_out = compute_n_out(n_in, p, k, s)

    # Get weights of output wrt initial input
    l_s_weight = get_bound_conv(weight, l_s_weight, c_out, c_in, k, p, s, n_in, n_out)
    u_s_weight = get_bound_conv(weight, u_s_weight, c_out, c_in, k, p, s, n_in, n_out)

    # Add bias of current layer
    bias = torch.cat([
        torch.full(( n_out, n_out ), bias[b_out]).unsqueeze(0)
        for b_out in range(c_out)
    ])
    l_s_bias = bias
    u_s_bias = bias

    return l_s_weight, u_s_weight, l_s_bias, u_s_bias



def flatten_bounds(l_s_weight: torch.tensor,
                   u_s_weight: torch.tensor,
                   l_s_bias:   torch.tensor,
                   u_s_bias:   torch.tensor) -> Tuple[torch.tensor,
                                                      torch.tensor,
                                                      torch.tensor,
                                                      torch.tensor]:
    """
    Format symbolic bounds from Convolutional layer
    """

    l_s_weight = l_s_weight.flatten(end_dim=2).flatten(1).unsqueeze(0)
    u_s_weight = u_s_weight.flatten(end_dim=2).flatten(1).unsqueeze(0)

    l_s_bias = l_s_bias.flatten().unsqueeze(0)
    u_s_bias = u_s_bias.flatten().unsqueeze(0)

    return l_s_weight, u_s_weight, l_s_bias, u_s_bias
