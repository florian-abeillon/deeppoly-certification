from typing import List, Tuple

import argparse
import csv
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import NormalizedResnet, get_net_name, get_network


DEVICE = 'cpu'
DTYPE = torch.float32

TIME_START = time.time()
TIME_LIMIT = 50


def transform_image(pixel_values, input_dim):
    normalized_pixel_values = torch.tensor([float(p) / 255.0 for p in pixel_values])
    if len(input_dim) > 1:
        input_dim_in_hwc = (input_dim[1], input_dim[2], input_dim[0])
        image_in_hwc = normalized_pixel_values.view(input_dim_in_hwc)
        image_in_chw = image_in_hwc.permute(2, 0, 1)
        image = image_in_chw
    else:
        image = normalized_pixel_values

    assert (image >= 0).all()
    assert (image <= 1).all()
    return image

def get_spec(spec, dataset):
    input_dim = [1, 28, 28] if dataset == 'mnist' else [3, 32, 32]
    eps = float(spec[:-4].split('/')[-1].split('_')[-1])
    test_file = open(spec, "r")
    test_instances = csv.reader(test_file, delimiter=",")
    for i, (label, *pixel_values) in enumerate(test_instances):
        inputs = transform_image(pixel_values, input_dim)
        inputs = inputs.to(DEVICE).to(dtype=DTYPE)
        true_label = int(label)
    inputs = inputs.unsqueeze(0)
    # with open(spec, "r") as test_file:
    #     test_instances = csv.reader(test_file, delimiter=",")
    #     label, *pixel_values = next(test_instances)
    # inputs = transform_image(pixel_values, input_dim)
    # inputs = inputs.to(DEVICE).to(dtype=DTYPE)
    # true_label = int(label)
    return inputs, true_label, eps


def get_net(net, net_name):
    net = get_network(DEVICE, net)
    state_dict = torch.load('team-6/nets/%s' % net_name, map_location=torch.device(DEVICE))
    # state_dict = torch.load('../nets/%s' % net_name, map_location=torch.device(DEVICE))
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    net.load_state_dict(state_dict)
    net = net.to(dtype=DTYPE)
    net.eval()
    if 'resnet' in net_name:
        net = NormalizedResnet(DEVICE, net)
    return net



def get_layers_utils(net: nn.Sequential) -> Tuple[List[dict], nn.ParameterList]:
    """
    Go through all layers of net, and get useful variables of each
    """

    layers = []
    parameters = nn.ParameterList()

    for layer in net.modules():

        type_ = type(layer)

        if type_ == nn.Linear:

            # Get weights and biases of Linear layer
            weight = layer.weight.detach()
            bias   = layer.bias.detach()
            
            # Separate positive and negative weights
            weight_pos =  F.relu(weight)
            weight_neg = -F.relu(-weight)

            layers.append(
                {
                    'type': 'linear',
                    'utils': ( weight_pos, weight_neg, bias )
                }
            )

        elif type_ == nn.ReLU:

            # Initialize alpha parameter as a vector filled with zeros
            # Use 'weight' from last Liner layer, to get actual shape
            parameter = torch.nn.Parameter(torch.zeros(weight.shape[0]))

            parameters.append(parameter)
            layers.append(
                {
                    'type': 'relu',
                    'utils': parameter
                }
            )

    return layers, parameters



def compute_bounds(layers: List[dict], l_0: torch.tensor, u_0: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """
    Compute lower and upper bounds for every output node
    """

    # TODO: Remove from loop to be more efficient?
    weight_empty = torch.diag(torch.ones_like(l_0))
    bias_empty   = torch.zeros_like(l_0)

    # Initialize weights and biases
    l_s_weight = weight_empty
    u_s_weight = weight_empty
    l_s_bias   = bias_empty
    u_s_bias   = bias_empty

    # Iterate over every layer
    for layer in layers:

        # If Linear layer
        if layer['type'] == 'linear':

            weight_pos, weight_neg, bias = layer['utils']

            ## Compute symbolic bounds
            # Get weights of output wrt initial input
            l_s_weight, u_s_weight = (
                torch.matmul(weight_pos, l_s_weight) - torch.matmul(weight_neg, u_s_weight),
                torch.matmul(weight_pos, u_s_weight) - torch.matmul(weight_neg, l_s_weight)
            )
            # Add bias of current layer
            l_s_bias = torch.matmul(weight_pos, l_s_bias) + bias
            u_s_bias = torch.matmul(weight_pos, u_s_bias) + bias


        # If ReLU layer
        elif layer['type'] == 'relu':

            # Get lower and upper bounds of previous (Linear) layer
            l = torch.matmul(l_s_weight, l_0) + l_s_bias
            u = torch.matmul(u_s_weight, u_0) + u_s_bias

            # Separate case ( l > 0 ) and case ( l < 0 and u > 0 )
            # (and implicitly the case ( l, u > 0 ))
            case_1 = l.ge(0)
            case_2 = ~case_1 & u.ge(0)

            ## Utils
            parameter = layer['utils']
            alpha = 1 / (1 + torch.exp(parameter))
            # If u == l for some, replace with 1
            lambda_ = torch.where(u != l, u / (u - l), torch.ones_like(u))

            # Get ReLU resolution for weights
            weight_l = case_1 + alpha   * case_2
            weight_u = case_1 + lambda_ * case_2

            l_s_weight = l_s_weight * weight_l.unsqueeze(1)
            u_s_weight = u_s_weight * weight_u.unsqueeze(1)

            # Add ReLU resolution for biases
            # l_s_bias += torch.zeros_like(l)
            u_s_bias += - l * lambda_ * case_2

            # Get lower and upper bounds
            l = torch.matmul(l_s_weight, l_0) + l_s_bias
            u = torch.matmul(u_s_weight, u_0) + u_s_bias


    return l, u


def analyze(net, inputs, eps, true_label) -> bool:

    # Initialize lower & upper bounds
    l_0 = (inputs - eps).clamp(0, 1).flatten()
    u_0 = (inputs + eps).clamp(0, 1).flatten()

    # Get an overview of layers in net
    layers, parameters = get_layers_utils(net)
    
    # Optimization
    opt = optim.Adam(parameters, lr=1)

    while time.time() - TIME_START < TIME_LIMIT:
        opt.zero_grad()

        # Compute upper and lower bounds of last nodes using DeepPoly
        l, u = compute_bounds(layers, l_0, u_0)

        # Get the differences between output upper bounds, and lower bound of true_label
        diffs = l[true_label] - u

        # Errors whenever at least one output upper bound is greater than lower bound of true_label
        errors = diffs[diffs < 0]
        if len(errors) == 0:
            return True

        # Compute loss, and optimize alpha
        loss = torch.log(-errors).max()
        loss.backward()
        opt.step()

    return False


def main():
    # parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    # parser.add_argument('--net', type=str, required=True, help='Neural network architecture to be verified.')
    # parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    # args = parser.parse_args()

    net_name = get_net_name('net1')
    # net_name = get_net_name(args.net)
    dataset = 'mnist' if 'mnist' in net_name else 'cifar10'
    
    inputs, true_label, eps = get_spec('team-6/examples/net1/img1_0.0500.txt', dataset)
    # inputs, true_label, eps = get_spec(args.spec, dataset)
    net = get_net('net1', net_name)
    # net = get_net(args.net, net_name)

    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
