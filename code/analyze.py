from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from backsubstitution import backsubstitute
from networks import Normalization
from resnet import BasicBlock
from utils import (
    TIME_LIMIT, TIME_START,
    compute_out_dim, get_conv_matrix, 
    get_numerical_bounds
)



def get_layers_utils(net:    nn.Sequential,
                     in_dim: int          ) -> Tuple[List[dict], 
                                                     List[torch.tensor],
                                                     int]:
    """
    Get utils from every layer of net
    """

    layers, parameters = [], []

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
            
            # Initialize alpha parameter as a vector filled with zeros
            parameter = torch.zeros(weight.shape[0], requires_grad=True)
            parameters.append(parameter)

            # Add parameter to previous layer
            layers[-1]['relu_param'] = parameter
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
    

    return layers, parameters, in_dim




def preprocess_bounds(layers: List[dict], 
                      l_0:    torch.tensor, 
                      u_0:    torch.tensor) -> Tuple[List[dict], 
                                                     torch.tensor, 
                                                     torch.tensor]:
    """
    Preprocess lower and upper bounds for every input node
    """

    # If first layer is a Normalization layer
    first_layer = layers[0]

    if first_layer['type'] == Normalization.__name__:

        # Normalize initial bounds
        normalize = first_layer['layer']
        l_0 = normalize(l_0)
        u_0 = normalize(u_0)

        # Remove normalization layer
        layers = layers[1:]
    
    # Flatten initial bounds
    l_0 = l_0.flatten()
    u_0 = u_0.flatten()

    return layers, l_0, u_0



def analyze(net, inputs, eps, true_label) -> bool:

    # Get an overview of layers in net
    in_dim = inputs.shape[2]
    layers, parameters, _ = get_layers_utils(net, in_dim)

    # Initialize lower and upper bounds
    l_0 = (inputs - eps).clamp(0, 1)
    u_0 = (inputs + eps).clamp(0, 1)
    layers, l_0, u_0 = preprocess_bounds(layers, l_0, u_0)

    # Optimization
    optimizer = optim.Adam(parameters, lr=1)

    # TODO: To remove
    i = 0
    while i < 1000:
    # while time.time() - TIME_START < TIME_LIMIT:
        optimizer.zero_grad()

        # Get lower and upper symbolic bounds using DeepPoly
        symbolic_bounds = backsubstitute(layers, l_0, u_0)
        # Using them, compute lower and upper numerical bounds
        l, u = get_numerical_bounds(l_0, u_0, *symbolic_bounds)

        # Get the differences between output upper bounds, and lower bound of true_label
        diffs = l[true_label] - u
        diffs = torch.cat([ diffs[:true_label], diffs[true_label + 1:] ])

        # Errors whenever at least one output upper bound is greater than lower bound of true_label
        errors = diffs[diffs < 0]
        if len(errors) == 0:
            print(i)
            return True

        # Compute loss, and backpropagate to learn alpha parameters
        loss = torch.max(torch.log(-errors))
        # loss = torch.sqrt(torch.sum(torch.square(errors)))
        loss.backward()
        optimizer.step()

        # TODO: To remove
        if i % 10 == 0:
            print(i)
            print(errors.data)
            print(loss.data)
            print()
        i+= 1

    return False
