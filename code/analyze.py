from typing import List, Tuple

import time
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import Normalization


TIME_START = time.time()
TIME_LIMIT = 60

def get_kernel_matrix(kernel:           torch.tensor,
                      padded_input_dim: int,
                      output_dim:       int,
                      stride:           int):
    kernel_dim = kernel.shape[0]
    kernel_matrix = torch.zeros((output_dim, padded_input_dim * kernel_dim))
    for kernel_row in range(kernel_dim):
        kernel_row_matrix = torch.zeros((output_dim, padded_input_dim))
        col_offset = kernel_row * padded_input_dim
        for row in range(output_dim):
            row_offset = row * stride
            kernel_row_matrix[row, row_offset:row_offset + kernel_dim] = kernel[kernel_row]
        kernel_matrix[:, col_offset:col_offset + padded_input_dim] = kernel_row_matrix
    return kernel_matrix



def get_layers_utils(net: nn.Sequential, dataset: str) -> Tuple[List[dict], List[torch.tensor]]:
    """
    Go through all layers of net, and get useful variables for each
    """

    layers, parameters = [], []
    input_dim = 28 if dataset == 'mnist' else 32

    for layer in net.modules():

        type_ = type(layer)

        # If Linear layer
        if type_ == nn.Linear:

            # Get weights and biases of Linear layer
            weight = layer.weight.detach()
            bias = layer.bias.detach()

            layers.append(
                {
                    'type': type_.__name__,
                    'utils': (weight, bias)
                }
            )

        # If ReLU layer
        elif type_ == nn.ReLU:

            # Initialize alpha parameter as a vector filled with zeros
            # Use 'weight' from last Linear/Conv layer, to get actual shape
            parameter = torch.zeros(weight.shape[0], requires_grad=True)

            parameters.append(parameter)
            layers.append(
                {
                    'type': type_.__name__,
                    'utils': parameter
                }
            )


        # If Conv layer
        elif type_ == nn.Conv2d:
            weight = layer.weight.detach()
            bias   = layer.bias.detach().flatten()

            padding = layer.padding[0]
            stride  = layer.stride[0]
            kernel_size = layer.kernel_size[0]
            padded_input_dim = input_dim + 2 * padding
            in_channels = layer.in_channels
            out_channels = layer.out_channels

            out_dimensions = (padded_input_dim - kernel_size) // stride + 1

            out_matrix = torch.zeros((out_channels * out_dimensions ** 2, padded_input_dim * padded_input_dim * in_channels))

            for out_channel in range(out_channels):
                out_channel_row_offset = out_channel * out_dimensions ** 2
                for in_channel in range(in_channels):
                    in_channel_col_offset = in_channel * padded_input_dim
                    kernel_matrix = get_kernel_matrix(weight[out_channel, in_channel, :, :],
                                                        padded_input_dim,
                                                        out_dimensions,
                                                        stride)
                    for row in range(out_dimensions):
                        row_offset = row * out_dimensions
                        col_offset = row * stride * padded_input_dim

                        total_row_offset = out_channel_row_offset + row_offset
                        total_col_offset = in_channel_col_offset + col_offset
                        assert out_matrix[
                            total_row_offset:total_row_offset + kernel_matrix.shape[0],
                            total_col_offset:total_col_offset + kernel_matrix.shape[1]
                            ].sum() == 0
                        out_matrix[
                            total_row_offset:total_row_offset + kernel_matrix.shape[0],
                            total_col_offset:total_col_offset + kernel_matrix.shape[1]
                            ] = kernel_matrix
            
            # To find out which columns we should omit
            pad = nn.ZeroPad2d(padding)
            to_keep = torch.ones((in_channels, input_dim, input_dim))
            to_keep = pad(to_keep).flatten().bool()

            weight = out_matrix[:, to_keep]
            # We need to repeat each element of the bias sequentially
            # as there is one bias per output channel
            bias = bias.repeat_interleave(out_dimensions ** 2)

            layers.append(
                {
                    'type': type_.__name__,
                    'utils': (weight, bias)
                }
            )

            input_dim = out_dimensions


        # If Flatten or Normalization layer
        elif type_ == Normalization:
            mean = layer.mean.flatten()
            sigma = layer.sigma.flatten()

            layers.append(
                {
                    'type': type_.__name__,
                    'utils': (mean, sigma)
                    # 'utils': layer
                }
            )
            

    return layers, parameters


def preprocess_bounds(layers: List[dict], l_0: torch.tensor, u_0: torch.tensor) -> Tuple[List[dict], torch.tensor, torch.tensor]:
    """
    Preprocess lower and upper bounds for every input node
    """

    # Iterate over layers
    for i, layer in enumerate(layers):

        # If Normalization or Flatten layer
        if layer['type'] in [Normalization.__name__, nn.Flatten.__name__]:
            layer = layer['utils']
            l_0 = layer(l_0)
            u_0 = layer(u_0)

        # If other layer, end of preprocessing
        else:
            break

    layers = layers[i:]
    return layers, l_0, u_0


# To separate positive and negative coefs of a torch.tensor
def get_pos_neg(t): return (F.relu(t), - F.relu(-t))



def compute_bounds(l_0:        torch.tensor, 
                   u_0:        torch.tensor, 
                   l_s_weight: torch.tensor, 
                   u_s_weight: torch.tensor, 
                   l_s_bias:   torch.tensor, 
                   u_s_bias:   torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """
    Compute (non-symbolic) bounds
    """

    # Get positive and negative weights
    l_s_weight_pos, l_s_weight_neg = get_pos_neg(l_s_weight)
    u_s_weight_pos, u_s_weight_neg = get_pos_neg(u_s_weight)

    # Compute bounds using lower and upper input bounds (depending on weight signs), and additional bias
    l = torch.matmul(l_s_weight_pos, l_0) + \
        torch.matmul(l_s_weight_neg, u_0) + l_s_bias
    u = torch.matmul(u_s_weight_pos, u_0) + \
        torch.matmul(u_s_weight_neg, l_0) + u_s_bias
    return l, u


def deep_poly(layers: List[dict], l_0: torch.tensor, u_0: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """
    Compute lower and upper bounds for every output node
    """

    input_dims = l_0.shape

    l_0, u_0 = l_0.flatten(), u_0.flatten()
    weight_empty = torch.diag(torch.ones_like(l_0))
    bias_empty = torch.zeros_like(l_0)

    # Initialize (symbolic) lower and upper bounds
    l_s_weight = weight_empty.clone()
    u_s_weight = weight_empty.clone()
    l_s_bias = bias_empty.clone()
    u_s_bias = bias_empty.clone()

    # Iterate over every layer
    for layer in layers:
        # If Normalization layer
        if layer['type'] == Normalization.__name__:
            mean, sigma = layer['utils']
            # Check if mnist or cifar
            if mean.shape[0] == 1:
                l_s_weight /= sigma
                u_s_weight /= sigma
                l_s_bias -= mean / sigma
                u_s_bias -= mean / sigma

            else:
                sigma_vec = (1 / sigma).repeat_interleave(input_dims[2] * input_dims[3])
                l_s_weight /= sigma_vec.reshape((-1,1))
                u_s_weight /= sigma_vec.reshape((-1,1))
                mean_vec = mean.repeat_interleave(input_dims[2] * input_dims[3])
                l_s_bias -= mean_vec / sigma_vec
                u_s_bias -= mean_vec / sigma_vec


        # If Linear or Conv layer
        if layer['type'] in [nn.Linear.__name__, nn.Conv2d.__name__]:

            weight, bias = layer['utils']

            # Get weights of output wrt initial input
            l_s_weight = torch.matmul(weight, l_s_weight)
            u_s_weight = torch.matmul(weight, u_s_weight)

            # Add bias of current layer
            l_s_bias = torch.matmul(weight, l_s_bias) + bias
            u_s_bias = torch.matmul(weight, u_s_bias) + bias
        
        # If ReLU layer
        elif layer['type'] == nn.ReLU.__name__:

            # Compute lower and upper bounds of previous layer
            l, u = compute_bounds(l_0, u_0, l_s_weight, u_s_weight, l_s_bias, u_s_bias)

            # Separate case Strictly positive ( 0 <= l, u ) and case Crossing ReLU ( l < 0 < u )
            # (and implicitly the case Strictly negative ( l, u <= 0 ))
            mask_1 = l.ge(0)
            mask_2 = ~mask_1 & u.gt(0)

            parameter = layer['utils']
            alpha = torch.sigmoid(parameter)
            weight_l = mask_1 + mask_2 * alpha
            
            lambda_ = u / (u - l)
            weight_u = mask_1 + mask_2 * lambda_

            # Get ReLU resolution for weights
            l_s_weight *= weight_l.unsqueeze(1)
            u_s_weight *= weight_u.unsqueeze(1)

            # Add ReLU resolution for biases
            l_s_bias *= weight_l
            u_s_bias -= mask_2 * l
            u_s_bias *= weight_u



        # If Normalization or Flatten layer
        # if layer['type'] in [ Normalization.__name__, nn.Flatten.__name__ ]:
        #     layer = layer['utils']
        #     l = layer(l_0)
        #     u = layer(u_0)


    # Compute lower and upper bounds from initial bounds
    l, u = compute_bounds(l_0, u_0, l_s_weight, u_s_weight, l_s_bias, u_s_bias)
    return l, u


def analyze(net, inputs, eps, true_label, dataset) -> bool:

    # Get an overview of layers in net
    layers, parameters = get_layers_utils(net, dataset)

    # Initialize lower and upper bounds
    l_0 = (inputs - eps).clamp(0, 1)
    u_0 = (inputs + eps).clamp(0, 1)
    # layers, l_0, u_0 = preprocess_bounds(layers, l_0, u_0)
    if dataset == 'mnist':
        l_0, u_0 = l_0[0], u_0[0]

    # Optimization
    optimizer = optim.Adam(parameters, lr=.1)
    # optimizer = optim.SGD(parameters, lr=10)

    i = 0
    # while i < 10000:
    while time.time() - TIME_START < TIME_LIMIT:
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

        if i % 100 == 0:
            print(i)
            print(loss)
        
        i+= 1

    return False