from typing import List

import torch
import torch.nn.functional as F



def compute_out_dim(in_dim: int,
                    k:      int,
                    p:      int,
                    s:      int) -> int:
    """
    Compute out_dim of Convolutional layer
    """
    return (in_dim + 2 * p - k) // s + 1
    
    
    
def get_kernel_block(kernel:        torch.tensor,
                     in_dim_padded: int,
                     out_dim:       int,
                     k:             int,
                     s:             int         ) -> List[torch.tensor]:
    """
    Build flattened convolution kernel block
    """
    return torch.cat([
        
        # Pad left- and right-hand sides of kernel to fill until in_dim
        # Then flatten it out
        F.pad(
            kernel, 
            ( 
                i * s, 
                in_dim_padded - k - i * s
            )
        ).flatten().unsqueeze(0)

        for i in range(out_dim)

    ])



def get_conv_block(kernel:        torch.tensor,
                   in_dim_padded: int,
                   out_dim:       int,
                   k:             int,
                   s:             int         ) -> torch.tensor:
    """
    Build block from flattened kernel block
    """

    # Get flattened kernel block
    kernel_block = get_kernel_block(kernel, in_dim_padded, out_dim, k, s)

    return torch.cat([

        # Pad left- and right-hand sides of conv_block to fill until in_dim^2
        F.pad(
            kernel_block, 
            ( 
                in_dim_padded * i * s, 
                in_dim_padded * (in_dim_padded - k - i * s)
            )
        )

        for i in range(out_dim)

    ])



def get_non_padding_cols(in_dim:        int,
                         in_dim_padded: int,
                         in_channels:   int,
                         p:             int) -> torch.tensor:
    """
    Get columns from conv_matrix that do not correspond to paddings
    """

    cols = []

    # For every in_channel
    for i in range(in_channels):

        # Offset to first element of in_channel
        offset = i * in_dim_padded * in_dim_padded
        # Offset to first non-padding row
        offset += p * in_dim_padded 
        # Offset to first non-padding element
        offset += p

        # Get indices of non-padding coefficients, row by row
        for _ in range(in_dim):
            cols += list(range(offset, offset + in_dim))

            # Offset to next row    
            offset += in_dim_padded

    return cols



def get_conv_matrix(weight:  torch.tensor,
                    in_dim:  int,
                    out_dim: int,
                    k:       int,
                    p:       int,
                    s:       int         ) -> torch.tensor:
    """
    Get flattened convolution matrix
    """

    # Get in-/out-channel dimensions
    out_channels = weight.shape[0]
    in_channels  = weight.shape[1]
    in_dim_padded = in_dim + 2 * p

    # Build conv_matrix
    conv_matrix = torch.cat([

        # For every out_channel
        torch.cat(
            [

                # For every in_channel
                # Build block from flattened kernel
                get_conv_block(weight[i, j], in_dim_padded, out_dim, k, s)
                
                for j in range(in_channels)

            ],
            dim=1
        )

        for i in range(out_channels)

    ])

    # Get columns corresponding to paddings
    cols_non_padding = get_non_padding_cols(in_dim, in_dim_padded, in_channels, p)
    # Remove columns corresponding to paddings
    conv_matrix = conv_matrix[:, cols_non_padding]
    
    return conv_matrix
