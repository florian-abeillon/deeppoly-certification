from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from networks import Normalization
from utils import (
    TIME_LIMIT, TIME_START,
    compute_bounds, compute_out_dim,
    get_conv_matrix, get_bounds, 
    get_bounds_relu
)



def get_layers_utils(net:    nn.Sequential,
                     in_dim: int          ) -> Tuple[List[tuple], 
                                                     List[torch.tensor]]:
    """
    Get utils from every layer of net
    """

    layers, parameters = [], []

    for layer in net.modules():

        type_ = type(layer)


        # If Linear layer
        if type_ == nn.Linear:

            weight = layer.weight.detach()
            bias = layer.bias.detach()

            utils = ( weight, bias )

            in_dim = weight.shape[0]


        # If ReLU layer
        elif type_ == nn.ReLU:

            # Initialize alpha parameter as a vector filled with zeros
            parameter = torch.zeros(weight.shape[0], requires_grad=True)
            parameters.append(parameter)

            utils = parameter


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

            utils = ( weight, bias )

            in_dim = out_dim


        # If Normalization layer
        elif type_ == Normalization:

            utils = layer


        else:
            continue


        layers.append(( type_.__name__, utils ))
        

    return layers, parameters



def preprocess_bounds(layers: List[dict], 
                      l_0:    torch.tensor, 
                      u_0:    torch.tensor) -> Tuple[List[dict], 
                                                     torch.tensor, 
                                                     torch.tensor]:
    """
    Preprocess lower and upper bounds for every input node
    """

    # If first layer is a Normalization layer
    type_, layer = layers[0]
    if type_ == Normalization.__name__:

        # Normalize initial bounds
        l_0 = layer(l_0)
        u_0 = layer(u_0)

        # Remove normalization layer
        layers = layers[1:]
    
    # Flatten initial bounds
    l_0 = l_0.flatten()
    u_0 = u_0.flatten()

    return layers, l_0, u_0



def deep_poly(layers: List[dict], 
              l_0:    torch.tensor, 
              u_0:    torch.tensor) -> Tuple[torch.tensor, 
                                             torch.tensor]:
    """
    Compute lower and upper bounds for every output node
    """

    # Initialize (symbolic) lower and upper bounds
    weight_empty = torch.diag(torch.ones_like(l_0))
    bias_empty = torch.zeros_like(l_0)

    l_s_weight = weight_empty
    u_s_weight = weight_empty
    l_s_bias = bias_empty
    u_s_bias = bias_empty


    # Iterate over every layer
    for type_, utils in layers:

        # If Linear or Convolutional layer
        if type_ in [ nn.Linear.__name__, nn.Conv2d.__name__ ]:

            weight, bias = utils

            l_s_weight, u_s_weight, l_s_bias, u_s_bias = get_bounds(weight, 
                                                                    bias, 
                                                                    l_s_weight, 
                                                                    u_s_weight, 
                                                                    l_s_bias, 
                                                                    u_s_bias)

        
        # If ReLU layer
        elif type_ == nn.ReLU.__name__:

            parameter = utils

            l_s_weight, u_s_weight, l_s_bias, u_s_bias = get_bounds_relu(l_0, 
                                                                         u_0, 
                                                                         parameter, 
                                                                         l_s_weight, 
                                                                         u_s_weight, 
                                                                         l_s_bias, 
                                                                         u_s_bias)


    # Compute lower and upper numerical bounds
    l, u = compute_bounds(l_0, 
                          u_0, 
                          l_s_weight, 
                          u_s_weight, 
                          l_s_bias, 
                          u_s_bias)
    return l, u


def analyze(net, inputs, eps, true_label) -> bool:

    # Get an overview of layers in net
    in_dim = inputs.shape[2]
    layers, parameters = get_layers_utils(net, in_dim)

    # Initialize lower and upper bounds
    l_0 = (inputs - eps).clamp(0, 1)
    u_0 = (inputs + eps).clamp(0, 1)
    layers, l_0, u_0 = preprocess_bounds(layers, l_0, u_0)

    # Optimization
    optimizer = optim.Adam(parameters, lr=.1)

    i = 0
    while i < 1000:
    # while time.time() - TIME_START < TIME_LIMIT:
        optimizer.zero_grad()

        # Compute upper and lower bounds of output using DeepPoly
        l, u = deep_poly(layers, l_0, u_0)

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

        if i % 10 == 0:
            print(i)
            print(errors)
            print(loss)
            print()
        
        i+= 1

    return False
