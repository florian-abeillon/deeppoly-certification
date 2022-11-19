from typing import List, Tuple

import torch
import torch.nn as nn

from uuid import uuid4

from networks import Normalization
from resnet import BasicBlock
from utils import (
    TIME_LIMIT, TIME_START,
    backsubstitution_step, compute_out_dim,
    deep_poly, get_conv_matrix, 
    get_numerical_bounds, init_symbolic_bounds
)



def get_layers_utils(net:    nn.Sequential,
                     in_dim: int          ) -> Tuple[List[dict], 
                                                     List[torch.tensor],
                                                     int]:
    """
    Get utils from every layer of net
    """

    layers, parameters = [], {}

    for layer in net.modules():
        
        type_ = type(layer)
        utils = {
            'type': type_.__name__
        }


        # If Linear layer
        if type_ == nn.Linear:

            weight = layer.weight.detach()
            bias = layer.bias.detach()

            utils['weight_bias'] = ( weight, bias )

            in_dim = weight.shape[0]


        # If Convolutional layer
        elif type_ == nn.Conv2d:

            weight = layer.weight.detach()
            bias = layer.bias.detach()
            p = layer.padding[0]
            s = layer.stride[0]

            # Compute out_dim
            k = weight.shape[2]
            out_dim = compute_out_dim(in_dim, k, p, s)

            # Get flattened convolutional matrix, and bias
            weight = get_conv_matrix(weight, in_dim, out_dim, k, p, s)
            bias = bias.repeat_interleave(out_dim * out_dim)

            utils['weight_bias'] = ( weight, bias )

            in_dim = out_dim


        # If ReLU layer
        elif type_ == nn.ReLU:
            
            id = str(uuid4())
            # Initialize alpha parameter as a vector filled with zeros
            parameter = torch.ones(weight.shape[0])
            parameters[id] = parameter

            layers[-1]['relu_param_id'] = id
            continue


        # If Normalization layer
        elif type_ in [ Normalization, nn.BatchNorm2d ]:

            utils['layer'] = layer

            
        # If ResidualBlock
        elif type_ == BasicBlock.__name__:

            # Get utils from both path of ResidualBlock
            path_a, parameters_a, _      = get_layers_utils(layer.path_a, in_dim)
            path_b, parameters_b, in_dim = get_layers_utils(layer.path_b, in_dim)

            # Add their parameters to the list of tracked parameters
            parameters.extend(parameters_a)
            parameters.extend(parameters_b)

            utils['path_a'] = path_a
            utils['path_b'] = path_b


        else:
            continue


        layers.append(utils)
        parameters = nn.ParameterDict(parameters)
        

    return layers, parameters, in_dim



def get_symbolic_bounds(layers: List[dict], 
                        parameters: nn.ParameterDict,
                        l_0:    torch.tensor, 
                        u_0:    torch.tensor, detach = False) -> Tuple[torch.tensor, 
                                                       torch.tensor, 
                                                       torch.tensor, 
                                                       torch.tensor]:
    """
    Backsubstitute symbolic bounds on previous layer
    """
    
    # If first layer (no previous layers)
    if not layers:
        shape = l_0.shape[0]
        return init_symbolic_bounds(shape)
    
    # Get symbolic bounds of layer wrt to previous layer
    last_layer = layers[-1]
    weight, bias = last_layer['weight_bias']
    symbolic_bounds = ( weight, weight.clone(), bias, bias.clone() )
    
    ## If no ReLU layer aftewards
    if not 'relu_param_id' in last_layer:
        return symbolic_bounds
        
    ## If ReLU layer afterwards
    # Backsubstitute from current layer, to get numerical bounds
    symbolic_bounds_backsub = backsubstitute(layers, parameters, l_0, u_0, detach=True)
    numerical_bounds = get_numerical_bounds(l_0, u_0, *symbolic_bounds_backsub)
    
    # Update symbolic bounds using DeepPoly
    parameter_id = last_layer['relu_param_id']
    if detach:
        parameter = parameters[parameter_id].detach()
        parameter.requires_grad = False
    else:
        parameter = parameters[parameter_id]
    symbolic_bounds = deep_poly(*numerical_bounds, parameter, *symbolic_bounds)
    
    return symbolic_bounds



def backsubstitute(layers: List[dict],
                   parameters: nn.ParameterDict,
                   l_0:    torch.tensor, 
                   u_0:    torch.tensor, detach = False) -> Tuple[torch.tensor, 
                                                  torch.tensor, 
                                                  torch.tensor, 
                                                  torch.tensor]:
    """
    Backsubstitute symbolic bounds on every layer
    """
    
    # Initialize symbolic bounds
    last_layer = layers[-1]
    weight, bias = last_layer['weight_bias']
    symbolic_bounds = ( weight, weight.clone(), bias, bias.clone() )

    # Iterate over every layer (backwards)
    for i in range(len(layers)):
        
        # Get symbolic bounds of layer wrt to previous layer
        prev_layers = layers[:-i-1]
        symbolic_bounds_prev = get_symbolic_bounds(prev_layers, parameters, l_0, u_0, detach=detach)
        
        # Update symbolic bounds with those of layer
        symbolic_bounds = backsubstitution_step(*symbolic_bounds_prev, *symbolic_bounds)
        
    return symbolic_bounds
