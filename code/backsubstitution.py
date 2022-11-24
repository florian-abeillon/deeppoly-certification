from typing import List, Tuple

import torch

from resnet import BasicBlock
from utils import (
    backsubstitute_bound, deep_poly, 
    get_numerical_bounds
)



def backsubstitute_step(prev_l_weight: torch.tensor,
                        prev_u_weight: torch.tensor,
                        prev_l_bias:   torch.tensor,
                        prev_u_bias:   torch.tensor,
                        l_weight:      torch.tensor, 
                        u_weight:      torch.tensor, 
                        l_bias:        torch.tensor, 
                        u_bias:        torch.tensor) -> Tuple[torch.tensor, 
                                                              torch.tensor, 
                                                              torch.tensor, 
                                                              torch.tensor]:
    """
    Backpropagate symbolic bounds using ReLU ones
    """

    # Update lower bounds
    l_weight, l_bias = (
        backsubstitute_bound(l_weight, prev_l_weight, prev_u_weight),
        backsubstitute_bound(l_weight, prev_l_bias, prev_u_bias) + l_bias
    )

    # Update upper bounds
    u_weight, u_bias = (
        backsubstitute_bound(u_weight, prev_u_weight, prev_l_weight),
        backsubstitute_bound(u_weight, prev_u_bias, prev_l_bias) + u_bias
    )
    
    return l_weight, u_weight, l_bias, u_bias



def backsubsitute(layer:           dict,
                  prev_sym_bounds: tuple, 
                  num_bounds:      Tuple[torch.tensor,
                                         torch.tensor]) -> Tuple[torch.tensor, 
                                                                 torch.tensor, 
                                                                 torch.tensor, 
                                                                 torch.tensor]:
    """
    Backsubsitute symbolic bounds using the ones from previous layer
    """

    # If Residual block
    if layer['type'] == BasicBlock.__name__:

        # Get symbolic bounds for each path
        sym_bounds_a = get_bounds(layer['path_a'], *num_bounds, prev_sym_bounds=prev_sym_bounds)
        sym_bounds_b = get_bounds(layer['path_b'], *num_bounds, prev_sym_bounds=prev_sym_bounds)

        # Add up the symbolic bounds of the two paths
        sym_bounds = tuple([
            sym_bound_a + sym_bound_b 
            for sym_bound_a, sym_bound_b in zip(sym_bounds_a, sym_bounds_b) 
        ])
    

    # If another type of layer (encoded with a weight/bias)
    else:

        weight, bias = layer['weight_bias']
        sym_bounds = ( weight, weight.clone(), bias, bias.clone() )
        
        # Backsubsitute using previous layer's backsubstituted bounds
        if prev_sym_bounds:
            sym_bounds = backsubstitute_step(*prev_sym_bounds, *sym_bounds)

        # If layer is followed by a ReLU layer
        if 'relu_param' in layer:

            # Get numerical bounds so far
            num_bounds = get_numerical_bounds(*num_bounds, *sym_bounds)
            param = layer['relu_param']

            # Get symbolic bounds using DeepPoly ReLU approximation
            sym_bounds = deep_poly(*num_bounds, param, *sym_bounds)
        

    return sym_bounds



def get_bounds(layers:          List[dict], 
               l_0:             torch.tensor, 
               u_0:             torch.tensor,
               prev_sym_bounds: tuple        = ()) -> Tuple[torch.tensor, 
                                                            torch.tensor, 
                                                            torch.tensor, 
                                                            torch.tensor]:
    """
    Get symbolic bounds of last layer
    """
    
    # Initialize bounds
    sym_bounds = prev_sym_bounds
    num_bounds = ( l_0, u_0 )

    # Iterate over every layer
    for layer in layers:
        # Get symbolic bounds for each
        sym_bounds = backsubsitute(layer, sym_bounds, num_bounds)

    return sym_bounds
