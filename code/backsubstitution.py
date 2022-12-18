from typing import List, Tuple

import torch

from resnet import BasicBlock
from utils import (
    backsubstitute_bound, deep_poly, 
    get_numerical_bounds, init_symbolic_bounds
)



def backsubstitute_step(l_weight:      torch.tensor, 
                        u_weight:      torch.tensor, 
                        l_bias:        torch.tensor, 
                        u_bias:        torch.tensor,
                        prev_l_weight: torch.tensor,
                        prev_u_weight: torch.tensor,
                        prev_l_bias:   torch.tensor,
                        prev_u_bias:   torch.tensor) -> Tuple[torch.tensor, 
                                                              torch.tensor, 
                                                              torch.tensor, 
                                                              torch.tensor]:
    """
    Backpropagate symbolic bounds using ReLU ones
    """

    # Update lower bounds
    l_weight, l_bias = (
        backsubstitute_bound(l_weight, prev_l_weight, prev_u_weight),
        backsubstitute_bound(l_weight, prev_l_bias,   prev_u_bias) + l_bias
    )

    # Update upper bounds
    u_weight, u_bias = (
        backsubstitute_bound(u_weight, prev_u_weight, prev_l_weight),
        backsubstitute_bound(u_weight, prev_u_bias,   prev_l_bias) + u_bias
    )
    
    return l_weight, u_weight, l_bias, u_bias



def backsubstitute(layers:     List[dict],
                   sym_bounds: tuple      = ()) -> Tuple[torch.tensor, 
                                                         torch.tensor, 
                                                         torch.tensor, 
                                                         torch.tensor]:
    """
    Backsubstitute from last to first layer
    """

    # Iterate over every layer backwards
    for layer in reversed(layers):

        prev_sym_bounds = layer['sym_bounds']

        # Update symbolic bounds
        if sym_bounds:
            sym_bounds = backsubstitute_step(*sym_bounds, *prev_sym_bounds)
        else:
            sym_bounds = prev_sym_bounds


        # If Residual block
        if layer['type'] == BasicBlock.__name__:

            # Backsubstitute through each path independently
            sym_bounds_a = backsubstitute(layer['path_a'], sym_bounds=sym_bounds)
            sym_bounds_b = backsubstitute(layer['path_b'], sym_bounds=sym_bounds)

            # Sum symbolic bounds back together (and remove initial bias, as it was passed to both path)
            sym_bounds = tuple([
                sym_bounds_a[0] + sym_bounds_b[0],
                sym_bounds_a[1] + sym_bounds_b[1],
                sym_bounds_a[2] + sym_bounds_b[2] - sym_bounds[2],
                sym_bounds_a[3] + sym_bounds_b[3] - sym_bounds[3]
            ])
            

    return sym_bounds



def get_symbolic_bounds(layers:      List[dict], 
                        l_0:         torch.tensor, 
                        u_0:         torch.tensor,
                        prev_layers: List[dict]   = []) -> List[dict]:
    """
    Add symbolic bounds to every layer
    """

    if prev_layers:
        prev_layers = prev_layers.copy()


    # Iterate over every layer
    for layer in layers:

        # If Residual block
        if layer['type'] == BasicBlock.__name__:

            # Add symbolic bounds to every element of both paths
            _ = get_symbolic_bounds(layer['path_a'], l_0, u_0, prev_layers=prev_layers)
            _ = get_symbolic_bounds(layer['path_b'], l_0, u_0, prev_layers=prev_layers)


        prev_layers.append(layer)


        # If layer is followed by a ReLU layer
        if 'relu_param' in layer:

            # Re-initialize symbolic bounds using layer's weight/bias
            layer['sym_bounds'] = init_symbolic_bounds(*layer['weight_bias'])

            # Backsubstituting until first layer, to get numerical bounds
            sym_bounds_backsub = backsubstitute(prev_layers)
            num_bounds = get_numerical_bounds(l_0, u_0, *sym_bounds_backsub)

            # Get symbolic bounds using DeepPoly ReLU approximation
            layer['sym_bounds'] = deep_poly(*num_bounds, layer['relu_param'], *layer['sym_bounds'])


    return prev_layers
