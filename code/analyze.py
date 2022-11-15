from typing import List, Tuple

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import Normalization


TIME_START = time.time()
TIME_LIMIT = 60

GRADS = {}


def store(grad, parent):
    GRADS[parent] = grad.clone()


def get_layers_utils(net: nn.Sequential) -> Tuple[List[dict], List[torch.tensor]]:
    """
    Go through all layers of net, and get useful variables for each
    """

    layers, parameters = [], []

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
            # Use 'weight' from last Liner layer, to get actual shape
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

            layers.append(
                {
                    'type': type_.__name__
                }
            )

        # If Flatten or Normalization layer
        elif type_ in [Normalization, nn.Flatten]:

            layers.append(
                {
                    'type': type_.__name__,
                    'utils': layer
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


def compute_bounds(l_0:        torch.tensor, u_0:        torch.tensor,
                   l_s_weight: torch.tensor, u_s_weight: torch.tensor,
                   l_s_bias:   torch.tensor, u_s_bias:   torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
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

    weight_empty = torch.diag(torch.ones_like(l_0))
    bias_empty = torch.zeros_like(l_0)

    # Initialize (symbolic) lower and upper bounds
    l_s_weight = weight_empty
    u_s_weight = weight_empty
    l_s_bias = bias_empty
    u_s_bias = bias_empty

    # Iterate over every layer
    for layer in layers:

        # If Linear layer
        if layer['type'] == nn.Linear.__name__:

            weight, bias = layer['utils']

            # Compute symbolic bounds
            # Get weights of output wrt initial input
            l_s_weight = torch.matmul(weight, l_s_weight)
            u_s_weight = torch.matmul(weight, u_s_weight)
            # Add bias of current layer
            l_s_bias = torch.matmul(weight, l_s_bias) + bias
            u_s_bias = torch.matmul(weight, u_s_bias) + bias

            # Compute lower and upper bounds from initial bounds
            l, u = compute_bounds(l_0, u_0, l_s_weight,
                                  u_s_weight, l_s_bias, u_s_bias)

        # If ReLU layer
        elif layer['type'] == nn.ReLU.__name__:

            # Separate case ( l > 0 ) and case ( l < 0 and u > 0 )
            # (and implicitly the case ( l, u > 0 ))
            mask_1 = l.ge(0)
            mask_2 = ~mask_1 & u.ge(0)

            # Utils
            parameter = layer['utils']
            parameter.register_hook(lambda grad: store(grad, '0 parameter'))
            alpha = torch.sigmoid(parameter)
            alpha.register_hook(lambda grad: store(grad, '1 alpha'))
            # If u == l for some, replace with 1
            lambda_ = torch.where(u != l, u / (u - l), torch.ones_like(u))

            # Get ReLU resolution for weights
            weight_l = mask_1 + alpha * mask_2
            weight_l.register_hook(lambda grad: store(grad, '2 weight_l'))
            weight_u = mask_1 + lambda_ * mask_2
            l_s_weight = l_s_weight * weight_l.unsqueeze(1)
            l_s_weight.register_hook(lambda grad: store(grad, '3 l_s_weight'))
            u_s_weight = u_s_weight * weight_u.unsqueeze(1)

            # Add ReLU resolution for biases
            # l_s_bias += torch.zeros_like(l)                       # No bias
            u_s_bias -= l * lambda_ * mask_2

            # Compute lower and upper bounds from initial bounds
            l, u = compute_bounds(l_0, u_0, l_s_weight,
                                  u_s_weight, l_s_bias, u_s_bias)
            l.register_hook(lambda grad: store(grad, '4 l'))

        # TODO
        # If Conv layer
        elif layer['type'] == nn.Conv2d.__name__:
            assert False

        # # If Normalization or Flatten layer
        # if layer['type'] in [ Normalization.__name__, nn.Flatten.__name__ ]:
        #     layer = layer['utils']
        #     l = layer(l_0)
        #     u = layer(u_0)

    l_s_weight.register_hook(lambda grad: store(grad, '5 l_s_weight final'))
    l.register_hook(lambda grad: store(grad, '7 l final'))

    return l, u


def analyze(net, inputs, eps, true_label, dataset) -> bool:

    # Get an overview of layers in net
    layers, parameters = get_layers_utils(net)

    # Initialize lower & upper bounds
    l_0 = (inputs - eps).clamp(0, 1)
    u_0 = (inputs + eps).clamp(0, 1)
    layers, l_0, u_0 = preprocess_bounds(layers, l_0, u_0)
    if dataset == 'mnist':
        l_0, u_0 = l_0[0], u_0[0]

    # Optimization
    optimizer = optim.Adam(parameters, lr=1)
    # optimizer = optim.SGD(parameters, lr=1)

    while time.time() - TIME_START < TIME_LIMIT:
        optimizer.zero_grad()

        # Compute upper and lower bounds of last nodes using DeepPoly
        l, u = deep_poly(layers, l_0, u_0)

        # Get the differences between output upper bounds, and lower bound of true_label
        diffs = l[true_label] - u
        diffs = torch.cat([diffs[:true_label], diffs[true_label + 1:]])
        diffs.register_hook(lambda grad: store(grad, '9 diffs'))

        # Errors whenever at least one output upper bound is greater than lower bound of true_label
        errors = diffs[diffs < 0]
        errors.register_hook(lambda grad: store(grad, '10 errors'))
        if len(errors) == 0:
            return True

        # Compute loss, and backpropagate to learn alpha parameters
        loss = torch.log(-errors).max()
        # loss = nn.MSELoss(-errors)
        loss.register_hook(lambda grad: store(grad, '11 loss'))
        loss.backward()
        optimizer.step()

    return False
