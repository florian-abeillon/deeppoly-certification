from typing import List, Tuple

import torch

from utils import (
    deep_poly, get_numerical_bounds,
    get_pos_neg, get_symbolic_bounds
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

    # Get positive and negative weights
    l_weight_pos, l_weight_neg = get_pos_neg(l_weight)
    u_weight_pos, u_weight_neg = get_pos_neg(u_weight)

    # Update weights
    l_weight = torch.matmul(l_weight_pos, prev_l_weight) + \
               torch.matmul(l_weight_neg, prev_u_weight)
    u_weight = torch.matmul(u_weight_pos, prev_u_weight) + \
               torch.matmul(u_weight_neg, prev_l_weight)

    # Update biases
    l_bias = l_bias + torch.matmul(l_weight_pos, prev_l_bias) + \
                      torch.matmul(l_weight_neg, prev_u_bias)
    u_bias = u_bias + torch.matmul(u_weight_pos, prev_u_bias) + \
                      torch.matmul(u_weight_neg, prev_l_bias)

    return l_weight, u_weight, l_bias, u_bias



def get_prev_symbolic_bounds(layers:  List[dict], 
                             l_0:     torch.tensor, 
                             u_0:     torch.tensor) -> Tuple[torch.tensor, 
                                                             torch.tensor, 
                                                             torch.tensor, 
                                                             torch.tensor]:
    """
    Backsubstitute symbolic bounds on previous layer
    """

    # TODO: To remove
    assert layers
    
    # Get symbolic bounds of current layer
    last_layer = layers[-1]
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



def backsubstitute(layers:  List[dict], 
                   l_0:     torch.tensor, 
                   u_0:     torch.tensor) -> Tuple[torch.tensor, 
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
