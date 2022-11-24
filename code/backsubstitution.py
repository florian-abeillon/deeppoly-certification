from typing import List, Tuple

import torch

from utils import (
    backsubstitute_bound, deep_poly, 
    get_numerical_bounds, get_symbolic_bounds
)



def backsubstitution_step(prev_l_weight: torch.tensor,
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



def get_prev_symbolic_bounds(layers: List[dict], 
                             l_0:    torch.tensor, 
                             u_0:    torch.tensor) -> Tuple[torch.tensor, 
                                                            torch.tensor, 
                                                            torch.tensor, 
                                                            torch.tensor]:
    """
    Backsubstitute symbolic bounds on previous layer
    """
    
    last_layer = layers[-1]

    # If Residual block
    if 'path_a' in last_layer:

        # Get symbolic bounds for each path (independently of everything else)
        symbolic_bounds_a = backsubstitute(last_layer['path_a'], l_0, u_0)
        symbolic_bounds_b = backsubstitute(last_layer['path_b'], l_0, u_0)

        # Add up the symbolic bounds of the two paths
        symbolic_bounds = ( 
            symbolic_bound_a + symbolic_bound_b 
            for symbolic_bound_a, symbolic_bound_b in zip(symbolic_bounds_a, symbolic_bounds_b) 
        )

        return symbolic_bounds


    # Get symbolic bounds of current layer
    symbolic_bounds = get_symbolic_bounds(last_layer)
    
    ## If no ReLU layer aftewards
    if not 'relu_param' in last_layer:
        return symbolic_bounds
        
    ## If ReLU layer afterwards
    # Backsubstitute from current layer, to get numerical bounds
    prev_symbolic_bounds = backsubstitute(layers, l_0, u_0)
    numerical_bounds = get_numerical_bounds(l_0, u_0, *prev_symbolic_bounds)
    
    # Update symbolic bounds using DeepPoly
    param = last_layer['relu_param']
    symbolic_bounds = deep_poly(*numerical_bounds, param, *symbolic_bounds)

    return symbolic_bounds



def backsubstitute(layers: List[dict], 
                   l_0:    torch.tensor, 
                   u_0:    torch.tensor) -> Tuple[torch.tensor, 
                                                  torch.tensor, 
                                                  torch.tensor, 
                                                  torch.tensor]:
    """
    Backsubstitute symbolic bounds on every layer
    """
    
    # Initialize symbolic bounds
    last_layer = layers[-1]
    symbolic_bounds = get_symbolic_bounds(last_layer)

    # Iterate over every layer (backwards)
    for i in range(1, len(layers)):
        
        # Get symbolic bounds of layer wrt to previous layer
        prev_layers = layers[:-i]
        prev_symbolic_bounds = get_prev_symbolic_bounds(prev_layers, l_0, u_0)
        
        # Update symbolic bounds with those of layer
        symbolic_bounds = backsubstitution_step(*prev_symbolic_bounds, *symbolic_bounds)

    return symbolic_bounds
