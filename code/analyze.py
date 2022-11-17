from typing import List, Tuple

import math
import torch
import torch.nn as nn
import torch.optim as optim

from networks import Normalization
from utils import (
    TIME_LIMIT, TIME_START,
    compute_n_out, compute_bounds_conv,
    compute_bounds_linear, flatten_bounds,
    get_bounds_conv, get_bounds_linear, 
    get_bounds_relu, zero_pad
)



def get_layers_utils(net:   nn.Sequential,
                     shape: Tuple[int]   ) -> Tuple[List[tuple], 
                                                    List[torch.tensor]]:
    """
    Go through all layers of net, and get useful variables for each
    """

    n_in = shape[2]
    layers, parameters = [], []

    for layer in net.modules():

        type_ = type(layer)


        # If Linear layer
        if type_ == nn.Linear:

            weight = layer.weight.detach().unsqueeze(0)
            bias = layer.bias.detach().unsqueeze(0)

            utils = ( weight, bias )

            # Get output shape
            # shape = weight.shape[1]
            shape = ( weight.shape[0], weight.shape[1] )
            n_in = weight.shape[2]


        # If ReLU layer
        elif type_ == nn.ReLU:

            # Initialize alpha parameter as a vector filled with zeros
            # Use 'weight' from last Liner layer, to get actual shape
            parameter = torch.zeros(shape, requires_grad=True)
            parameters.append(parameter)

            utils = parameter


        # If Convolutional layer
        elif type_ == nn.Conv2d:

            weight = layer.weight.detach()
            bias = layer.bias.detach()
            s = layer.stride[0]
            p = layer.padding[0]

            utils = ( weight, bias, s, p )

            # Get output shape
            k = weight.shape[2]
            n_out = compute_n_out(n_in, p, k, s)
            shape = ( weight.shape[0], n_out, n_out )
            n_in = n_out


        # If Flatten layer
        elif type_ in [ Normalization, nn.Flatten ]:

            utils = layer

            if type_ == nn.Flatten:
                n_in = ( 1, math.prod(shape) )


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

    has_flatten = False

    # Iterate over layers
    for i, ( type_, layer ) in enumerate(layers):

        # If Normalization or Flatten layer
        if type_ in [ Normalization.__name__, nn.Flatten.__name__ ]:

            l_0 = layer(l_0)
            u_0 = layer(u_0)

            has_flatten |= type_ == nn.Flatten.__name__

        # If other layer, end of preprocessing
        else:
            break

    # Remove preprocessing layers
    layers = layers[i:]

    # If there was no Flatten layer in the preprocessing
    # (if there was, it mimics a "one channel" behavior)
    if not has_flatten:
        # One-element arrays
        l_0, u_0 = l_0[0], u_0[0]

    return layers, l_0, u_0



def init_weight_bias(shape: Tuple[int]) -> Tuple[torch.tensor,
                                                 torch.tensor]:
    """
    Create empty weight and bias matrices/vectors
    """

    # Create identity weight matrices, for every channel
    # TODO: Doesn't work for conv
    weight_empty = torch.diag(torch.ones(shape[1])).unsqueeze(0)
    weight_empty = torch.cat((weight_empty,) * shape[0])

    # Create empty bias vectors for every channel
    bias_empty = torch.zeros(shape)

    return weight_empty, bias_empty



def deep_poly(layers: List[dict], 
              l_0:    torch.tensor, 
              u_0:    torch.tensor) -> Tuple[torch.tensor, 
                                             torch.tensor]:
    """
    Compute lower and upper bounds for every output node
    """

    # Initialize (symbolic) lower and upper bounds
    weight_empty, bias_empty = init_weight_bias(l_0.shape)

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

            # Compute lower and upper numerical bounds
            l, u = compute_bounds_linear(l_0, 
                                         u_0, 
                                         l_s_weight, 
                                         u_s_weight, 
                                         l_s_bias, 
                                         u_s_bias)

        
        # If ReLU layer
        elif type_ == nn.ReLU.__name__:
            parameter = utils

            l_s_weight, u_s_weight, l_s_bias, u_s_bias = get_bounds_relu(l, 
                                                                         u, 
                                                                         parameter, 
                                                                         l_s_weight, 
                                                                         u_s_weight, 
                                                                         l_s_bias, 
                                                                         u_s_bias)

            a = 0


        # If Conv layer
        elif type_ == nn.Conv2d.__name__:

            weight, bias, s, p = utils
                    
            # Add constant 0-padding around the matrix
            l_0 = zero_pad(l_0, p)
            u_0 = zero_pad(u_0, p)

            l_s_weight, u_s_weight, l_s_bias, u_s_bias = get_bounds_conv(weight,
                                                                         bias,
                                                                         s,
                                                                         p, 
                                                                         l_s_weight, 
                                                                         u_s_weight, 
                                                                         l_s_bias, 
                                                                         u_s_bias)

            # Compute lower and upper numerical bounds
            l, u = compute_bounds_conv(l_0, 
                                       u_0, 
                                       l_s_weight, 
                                       u_s_weight, 
                                       l_s_bias, 
                                       u_s_bias)


        # If Flatten layer after a Convolutional layer
        elif type_ == nn.Flatten.__name__:

            layer = utils
            l_0, u_0 = layer(l_0), layer(u_0)

            l_s_weight, u_s_weight, l_s_bias, u_s_bias = flatten_bounds(l_s_weight, 
                                                                        u_s_weight, 
                                                                        l_s_bias, 
                                                                        u_s_bias)


    return l, u


def analyze(net, inputs, eps, true_label) -> bool:

    # Get an overview of layers in net
    layers, parameters = get_layers_utils(net, inputs.shape)

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

        if i % 20 == 0:
            print(i)
            print(errors)
            print(loss)
            print()
        
        i+= 1

    return False
