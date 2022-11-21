from typing import List, Tuple
import time

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

    layers, params = [], []

    for layer in net.modules():
        
        type_ = type(layer)
        utils = {
            'type': type_.__name__
        }


        # If Linear or Convolutional layer
        if type_ in [ nn.Linear, nn.Conv2d ]:

            weight = layer.weight.detach()
            # Networks with batchnorm have no bias
            bias = False
            if layer.bias is not None:
                bias = layer.bias.detach()

            # If Convolutional layer
            if type_ == nn.Conv2d:

                p = layer.padding[0]
                s = layer.stride[0]

                # Compute out_dim
                k = weight.shape[2]
                out_dim = compute_out_dim(in_dim, k, p, s)

                # Get flattened convolutional matrix, and bias
                weight = get_conv_matrix(weight, in_dim, out_dim, k, p, s)
                if bias:
                    bias = bias.repeat_interleave(out_dim * out_dim)
                else:
                    bias = torch.zeros(weight.shape[0])

                in_dim = out_dim

            else:
                
                in_dim = weight.shape[0]

            utils['weight_bias'] = ( weight, bias )


        # If ReLU layer
        elif type_ == nn.ReLU:
            
            # Initialize alpha parameter as a vector filled with zeros
            param = torch.zeros(weight.shape[0], requires_grad=True)
            params.append(param)

            # Add parameter to previous layer
            layers[-1]['relu_param'] = param
            continue


        # If Normalization layer
        elif type_ == Normalization:

            utils['layer'] = layer

        elif type_ == nn.BatchNorm2d:
            mean = layer.running_mean.detach()
            prev_layer = layers[-1]
            prev_weight, prev_bias = prev_layer['weight_bias']
            var = layer.running_var.detach()
            eps = layer.eps
            gamma = layer.weight.detach()
            bias = layer.bias.detach()

            scale = gamma / torch.sqrt(var + eps)
            shift = bias - mean * scale
            scale_extended = scale.repeat_interleave(in_dim * in_dim)
            shift_extended = shift.repeat_interleave(in_dim * in_dim)
            prev_weight *= scale_extended.unsqueeze(1)
            prev_bias += shift_extended

            prev_layer['weight_bias'] = (prev_weight, prev_bias)

            
        # If ResidualBlock
        elif type_ == BasicBlock.__name__:

            # Get utils from both path of ResidualBlock
            path_a, params_a, _      = get_layers_utils(layer.path_a, in_dim)
            path_b, params_b, in_dim = get_layers_utils(layer.path_b, in_dim)

            # Add their parameters to the list of tracked parameters
            params.extend(params_a)
            params.extend(params_b)

            utils['path_a'] = path_a
            utils['path_b'] = path_b

        else:
            continue


        layers.append(utils)


    return layers, params, in_dim




def add_final_layer(layers:     List[dict], 
                    out_dim:    int,
                    true_label: int       ) -> List[dict]:
    """
    Artificially add a final layer, to subtract the true_label output node to every other output node
    """

    # -1 on diagonal, and 1 for true_label
    final_weight = -torch.eye(out_dim)
    final_weight[:, true_label] = 1.
    final_weight = torch.cat([ final_weight[:true_label], final_weight[true_label + 1:] ])

    # No bias
    final_bias = torch.zeros((out_dim - 1,))

    final_layer = {
        'type': nn.Linear.__name__,
        'weight_bias': ( final_weight, final_bias )
    }
    layers.append(final_layer)
    
    return layers



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
    layers, params, out_dim = get_layers_utils(net, in_dim)
    layers = add_final_layer(layers, out_dim, true_label)

    # Initialize lower and upper bounds
    l_0 = (inputs - eps).clamp(0, 1)
    u_0 = (inputs + eps).clamp(0, 1)
    layers, l_0, u_0 = preprocess_bounds(layers, l_0, u_0)

    # Optimization
    optimizer = optim.Adam(params, lr=1)

    # TODO: To remove
    i = 0
    # while i < 1000:
    while time.time() - TIME_START < TIME_LIMIT:
        optimizer.zero_grad()

        # Get lower and upper symbolic bounds using DeepPoly
        symbolic_bounds = backsubstitute(layers, l_0, u_0)
        # Using them, compute lower numerical bound of final_layer
        l, _ = get_numerical_bounds(l_0, u_0, *symbolic_bounds)

        # Errors whenever at least one output upper bound is greater than lower bound of true_label
        if l.gt(0).all():
            print(i)
            return True

        # Compute loss, and backpropagate to learn alpha parameters
        loss = torch.log(torch.max(-l))
        # loss = torch.sqrt(torch.sum(torch.square(errors)))
        loss.backward()
        optimizer.step()

        # TODO: To remove
        if i % 10 == 0:
            print(i)
            print(loss.data)
            print()
        i+= 1

    return False
