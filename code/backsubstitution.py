from typing import List, Tuple

import torch
import torch.nn as nn

from resnet import BasicBlock
from utils import (
    backsubstitute_bound, deep_poly, 
    get_numerical_bounds
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
        backsubstitute_bound(l_weight, prev_l_bias, prev_u_bias) + l_bias
    )

    # Update upper bounds
    u_weight, u_bias = (
        backsubstitute_bound(u_weight, prev_u_weight, prev_l_weight),
        backsubstitute_bound(u_weight, prev_u_bias, prev_l_bias) + u_bias
    )
    
    return l_weight, u_weight, l_bias, u_bias




def linearize_resblock(layer: nn.Module) -> List[dict]:
    """
    Linearize a Residual block
    """

    layers = []

    # Add Linear layer that splits data in two
    layers.append(layer['layer_in'])

    # Iterate over every pair of layers
    for layer_a, layer_b in zip(layer['path_a'], layer['path_b']):

        # Get their weight and bias
        l_weight_a, u_weight_a, l_bias_a, u_bias_a = layer_a['sym_bounds']
        l_weight_b, u_weight_b, l_bias_b, u_bias_b = layer_b['sym_bounds']

        # Concatenate weights and biases
        l_weight = torch.block_diag(l_weight_a, l_weight_b)
        u_weight = torch.block_diag(u_weight_a, u_weight_b)
        l_bias = torch.cat([ l_bias_a, l_bias_b ])
        u_bias = torch.cat([ u_bias_a, u_bias_b ])

        # Create Linear layer out of them
        utils = {
            'type': nn.Linear.__name__,
            'sym_bounds': ( l_weight, u_weight, l_bias, u_bias )
        }
        layers.append(utils)

    # Add Linear layer that puts data back together
    layers.append(layer['layer_out'])

    return layers



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

    return sym_bounds



def add_sym_bounds(layers:      List[dict], 
                   l_0:         torch.tensor, 
                   u_0:         torch.tensor,
                   prev_layers: List[dict]   = []) -> List[dict]:
    """
    Add symbolic bounds to every layer
    """

    prev_layers = prev_layers.copy()

    # Iterate over every layer
    for layer in layers:
        
        # If Residual block
        if layer['type'] == BasicBlock.__name__:

            # Add symbolic bounds to every element of both paths
            _ = add_sym_bounds(layer['path_a'], l_0, u_0, prev_layers=prev_layers)
            _ = add_sym_bounds(layer['path_b'], l_0, u_0, prev_layers=prev_layers)

            layers_linearized = linearize_resblock(layer)
            prev_layers.extend(layers_linearized)
        

        # If another type of layer
        else:

            # Initialize symbolic bounds with weight/bias
            weight, bias = layer['weight_bias']
            sym_bounds = ( weight, weight.clone(), bias, bias.clone() )

            # If layer is followed by a ReLU layer
            if 'relu_param' in layer:
            
                # Get numerical bounds so far
                sym_bounds_backsub = backsubstitute(prev_layers, sym_bounds=sym_bounds)
                num_bounds = get_numerical_bounds(l_0, u_0, *sym_bounds_backsub)

                # Get symbolic bounds using DeepPoly ReLU approximation
                param = layer['relu_param']
                sym_bounds = deep_poly(*num_bounds, param, *sym_bounds)

            # Update symbolic bounds
            layer['sym_bounds'] = sym_bounds

            prev_layers.append(layer)

    return prev_layers
