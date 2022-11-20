from typing import List, Tuple

import torch

from utils import (
    deep_poly, get_numerical_bounds,
    get_pos_neg, get_symbolic_bounds
)



def backsubstitution_step(l_s_weight_prev: torch.tensor,
                          u_s_weight_prev: torch.tensor,
                          l_s_bias_prev:   torch.tensor,
                          u_s_bias_prev:   torch.tensor,
                          l_s_weight:      torch.tensor, 
                          u_s_weight:      torch.tensor, 
                          l_s_bias:        torch.tensor, 
                          u_s_bias:        torch.tensor) -> Tuple[torch.tensor, 
                                                                  torch.tensor, 
                                                                  torch.tensor, 
                                                                  torch.tensor]:
    """
    Backpropagate symbolic bounds using ReLU ones
    """

    # Get positive and negative weights
    l_s_weight_pos, l_s_weight_neg = get_pos_neg(l_s_weight)
    u_s_weight_pos, u_s_weight_neg = get_pos_neg(u_s_weight)

    # Update weights
    l_s_weight = torch.matmul(l_s_weight_pos, l_s_weight_prev) + \
                 torch.matmul(l_s_weight_neg, u_s_weight_prev)
    u_s_weight = torch.matmul(u_s_weight_pos, u_s_weight_prev) + \
                 torch.matmul(u_s_weight_neg, l_s_weight_prev)

    # Update biases
    l_s_bias = l_s_bias + torch.matmul(l_s_weight_pos, l_s_bias_prev) + \
                          torch.matmul(l_s_weight_neg, u_s_bias_prev)
    u_s_bias = u_s_bias + torch.matmul(u_s_weight_pos, u_s_bias_prev) + \
                          torch.matmul(u_s_weight_neg, l_s_bias_prev)

    return l_s_weight, u_s_weight, l_s_bias, u_s_bias



def get_symbolic_bounds_prev(layers:  List[dict], 
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
    symbolic_bounds_prev = backsubstitute(layers, l_0, u_0)
    numerical_bounds = get_numerical_bounds(l_0, u_0, *symbolic_bounds_prev)
    
    # Update symbolic bounds using DeepPoly
    parameter = last_layer['relu_param']
    symbolic_bounds = deep_poly(*numerical_bounds, parameter, *symbolic_bounds)

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
        symbolic_bounds_prev = get_symbolic_bounds_prev(prev_layers, l_0, u_0)
        
        # Update symbolic bounds with those of layer
        symbolic_bounds = backsubstitution_step(*symbolic_bounds_prev, *symbolic_bounds)

    return symbolic_bounds
