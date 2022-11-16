from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from get_bounds import (
    compute_bounds, get_bounds_conv,
    get_bounds_linear, get_bounds_relu
)
from networks import Normalization
from utils import TIME_LIMIT, TIME_START



def get_layers_utils(net: nn.Sequential) -> Tuple[List[tuple], 
                                                  List[torch.tensor]]:
    """
    Go through all layers of net, and get useful variables for each
    """

    layers, parameters = [], []

    for layer in net.modules():

        type_ = type(layer)


        # If Linear layer
        if type_ == nn.Linear:

            weight = layer.weight.detach().unsqueeze(0)
            bias = layer.bias.detach().unsqueeze(0)

            utils = ( weight, bias )


        # If ReLU layer
        elif type_ == nn.ReLU:

            # Initialize alpha parameter as a vector filled with zeros
            # Use 'weight' from last Liner layer, to get actual shape
            parameter = torch.zeros(( weight.shape[0], weight.shape[1] ), requires_grad=True)
            parameters.append(parameter)

            utils = parameter


        # If Convolutional layer
        elif type_ == nn.Conv2d:

            weight = layer.weight.detach()
            bias = layer.bias.detach()
            s = layer.stride[0]
            p = layer.padding[0]

            utils = ( weight, bias, s, p )


        # If Flatten or Normalization layer
        elif type_ in [ Normalization, nn.Flatten ]:

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

    # Iterate over layers
    for i, ( type_, layer ) in enumerate(layers):

        # If Normalization or Flatten layer
        if type_ in [ Normalization.__name__, nn.Flatten.__name__ ]:
            l_0 = layer(l_0)
            u_0 = layer(u_0)

        # If other layer, end of preprocessing
        else:
            break

    layers = layers[i:]
    return layers, l_0, u_0



def deep_poly(layers: List[dict], 
              l_0: torch.tensor, 
              u_0: torch.tensor) -> Tuple[torch.tensor, 
                                          torch.tensor]:
    """
    Compute lower and upper bounds for every output node
    """

    # Create identity weight matrices for every channel
    weight_empty = torch.diag(torch.ones(l_0.shape[1])).unsqueeze(0)
    weight_empty = torch.cat((weight_empty,) * l_0.shape[0])
    # Create empty bias vectors for every channel
    bias_empty = torch.zeros_like(l_0)

    # Initialize (symbolic) lower and upper bounds
    l_s_weight = weight_empty
    u_s_weight = weight_empty
    l_s_bias = bias_empty
    u_s_bias = bias_empty

    # Iterate over every layer
    for type_, utils in layers:

        # If Linear layer
        if type_ == nn.Linear.__name__:

            weight, bias = utils
            l_s_weight, u_s_weight, l_s_bias, u_s_bias = get_bounds_linear(weight, 
                                                                           bias, 
                                                                           l_s_weight, 
                                                                           u_s_weight, 
                                                                           l_s_bias, 
                                                                           u_s_bias)

        
        # If ReLU layer
        elif type_ == nn.ReLU.__name__:

            # Compute lower and upper bounds of previous layer
            l, u = compute_bounds(l_0, 
                                  u_0, 
                                  l_s_weight, 
                                  u_s_weight, 
                                  l_s_bias, 
                                  u_s_bias)

            parameter = utils
            l_s_weight, u_s_weight, l_s_bias, u_s_bias = get_bounds_relu(l, 
                                                                         u, 
                                                                         parameter, 
                                                                         l_s_weight, 
                                                                         u_s_weight, 
                                                                         l_s_bias, 
                                                                         u_s_bias)


        # If Conv layer
        elif type_ == nn.Conv2d.__name__:

            weight, bias, p, s = utils

            l_s_weight, u_s_weight, l_s_bias, u_s_bias = get_bounds_conv(weight,
                                                                         bias,
                                                                         p,
                                                                         s, 
                                                                         l_s_weight, 
                                                                         u_s_weight, 
                                                                         l_s_bias, 
                                                                         u_s_bias)


        # # If Normalization or Flatten layer
        # if type_ in [ Normalization.__name__, nn.Flatten.__name__ ]:
        #     layer = utils
        #     l = layer(l_0)
        #     u = layer(u_0)


    # Compute lower and upper bounds from initial bounds
    l, u = compute_bounds(l_0, u_0, l_s_weight, u_s_weight, l_s_bias, u_s_bias)
    return l, u


def analyze(net, inputs, eps, true_label) -> bool:

    # Get an overview of layers in net
    layers, parameters = get_layers_utils(net)

    # Initialize lower and upper bounds
    l_0 = (inputs - eps).clamp(0, 1)
    u_0 = (inputs + eps).clamp(0, 1)
    layers, l_0, u_0 = preprocess_bounds(layers, l_0, u_0)

    # Optimization
    optimizer = optim.Adam(parameters, lr=.1)
    # optimizer = optim.SGD(parameters, lr=10)

    i = 0
    while i < 1000:
    # while time.time() - TIME_START < TIME_LIMIT:
        optimizer.zero_grad()

        # Compute upper and lower bounds of output using DeepPoly
        l, u = deep_poly(layers, l_0, u_0)
        l, u = l[0], u[0]

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

        if i % 100 == 0:
            print(i)
        
        i+= 1

    return False
