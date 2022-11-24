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



def add_bounds(layer:       dict,
               prev_layers: List[dict], 
               l_0:         torch.tensor, 
               u_0:         torch.tensor) -> None:
    """
    Backsubsitute symbolic bounds using the ones from previous layer
    """

    # If Residual block
    if layer['type'] == BasicBlock.__name__:

        # Add symbolic bounds to each element of both paths
        get_bounds(layer['path_a'], l_0, u_0, prev_layers=prev_layers)
        get_bounds(layer['path_b'], l_0, u_0, prev_layers=prev_layers)
    

    # If another type of layer (encoded with a weight/bias)
    else:

        weight, bias = layer['weight_bias']
        sym_bounds = ( weight, weight.clone(), bias, bias.clone() )

        # If layer is followed by a ReLU layer
        if 'relu_param' in layer:
        
            if prev_layers:
                prev_sym_bounds_backsub = backsubstitute(prev_layers)
                sym_bounds_backsub = backsubstitute_step(*prev_sym_bounds_backsub, *sym_bounds)
            else:
                sym_bounds_backsub = sym_bounds

            # Get numerical bounds so far
            num_bounds = get_numerical_bounds(l_0, u_0, *sym_bounds_backsub)
            param = layer['relu_param']

            # Get symbolic bounds using DeepPoly ReLU approximation
            sym_bounds = deep_poly(*num_bounds, param, *sym_bounds)


        # Add symbolic bound to layer element (to be used later on)
        layer['sym_bounds'] = sym_bounds



def get_bounds(layers:      List[dict], 
               l_0:         torch.tensor, 
               u_0:         torch.tensor,
               prev_layers: List[dict]   = []) -> None:
    """
    Add symbolic bounds for every layer
    """

    prev_layers = prev_layers.copy()

    # Get symbolic bounds of every layer
    for layer in layers:
        add_bounds(layer, prev_layers, l_0, u_0)
        prev_layers.append(layer)



def backsubstitute(layers:     List[dict],
                   sym_bounds: tuple      = ()) -> Tuple[torch.tensor, 
                                                         torch.tensor, 
                                                         torch.tensor, 
                                                         torch.tensor]:
    """
    Backsubstitute from last to first layer
    """

    for i, layer in enumerate(reversed(layers)):
        
        # If Residual block
        if layer['type'] == BasicBlock.__name__:

            # Create two independent paths that go from current layer to input layer
            i_lim = len(layers) - i - 1
            prev_layers = layers[:i_lim]
            path_a = prev_layers + layer['path_a']
            path_b = prev_layers + layer['path_b']

            # Backsubstitute each path independently
            sym_bounds_a = backsubstitute(path_a, sym_bounds=sym_bounds)
            sym_bounds_b = backsubstitute(path_b, sym_bounds=sym_bounds)

            # Add symbolic bounds
            sym_bounds = tuple((
                sym_bound_a + sym_bound_b
                for sym_bound_a, sym_bound_b in zip(sym_bounds_a, sym_bounds_b)
            ))

            # Return already (as the two paths went all the way to the input layer)
            return sym_bounds

            
        # If another type of layer (encoded with sym_bounds)
        # Get symbolic bounds directly
        elif sym_bounds:
            sym_bounds = backsubstitute_step(*layer['sym_bounds'], *sym_bounds)

        else:
            sym_bounds = layer['sym_bounds']


    return sym_bounds
