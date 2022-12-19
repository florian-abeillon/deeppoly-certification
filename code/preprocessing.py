from typing import List, Tuple

import torch
import torch.nn as nn

from networks import Normalization
from resnet import BasicBlock
from utils import (
    compute_out_dim, get_conv_matrix,
    init_symbolic_bounds
)



# Get dimension of flattened tensor
get_in_dim_flat = lambda in_chans, in_dim: in_chans * in_dim**2



def get_net_layers(net: nn.Module) -> List[nn.Module]:
    """
    Get list of layers in net
    """
    
    # If ResNet
    if hasattr(net, 'resnet'):
        net_layers = [net.normalization] 

        for layer in net.resnet:
            if type(layer) == nn.Sequential:
                net_layers.extend(list(layer))
            else:
                net_layers.append(layer)

    # If FullyConnected or Conv
    elif hasattr(net, 'layers'):
        net_layers = net.layers

    # If Sequential (from Residual block)
    else:
        net_layers = net

    return net_layers



def linearize_linear(layer: nn.Module) -> Tuple[torch.tensor,
                                                torch.tensor,
                                                int]:
    """
    Get utils for Linear layer
    """
    weight = layer.weight.detach()
    bias = layer.bias.detach()
    out_dim = layer.out_features
    return weight, bias, out_dim



def linearize_conv(layer:  nn.Module,
                   in_dim: int      ) -> Tuple[torch.tensor,
                                               torch.tensor,
                                               int,
                                               int,
                                               int]:
    """
    Get utils for Linear layer
    """

    weight = layer.weight.detach()
    bias = layer.bias.detach() if layer.bias is not None else torch.zeros(weight.shape[0])
    p = layer.padding[0]
    s = layer.stride[0]

    # Compute out_dim
    out_chans, in_chans, k, _ = weight.shape
    out_dim = compute_out_dim(in_dim, k, p, s)

    # Get flattened convolutional matrix, and bias
    weight = get_conv_matrix(weight, in_dim, out_dim, in_chans, out_chans, k, p, s)
    bias = bias.repeat_interleave(out_dim**2)

    out_dim_flat = get_in_dim_flat(out_chans, out_dim)

    return weight, bias, out_dim, out_chans, out_dim_flat



def linearize_norm(layer:  nn.Module, 
                   in_dim: int      ,
                   batch:  bool      = False) -> Tuple[torch.tensor,
                                                       torch.tensor]:
    """
    Get utils for (Batch) Normalization layer
    """

    # If Normalization layer
    if not batch:

        mean = layer.mean.detach()
        sigma = layer.sigma.detach()

        # Build weight and bias
        weight = 1 / sigma
        bias = -mean * weight

    # If Batch normalization layer
    else:

        mean = layer.running_mean.detach()
        var = layer.running_var.detach()
        eps = layer.eps

        # Build weight and bias
        weight = layer.weight.detach()
        weight /= torch.sqrt(var + eps)

        bias = layer.bias.detach()
        bias -= mean * weight


    # Reshape weight and bias
    n = in_dim**2
    weight = weight.repeat_interleave(n)
    bias   = bias.repeat_interleave(n)

    # Turn weight vector into diagonal matrix (for matrix multiplication)
    weight = torch.diag(weight)

    return weight, bias



def linearize_resblock_paths(layer:       nn.Module, 
                             in_dim:      int,
                             in_chans:    int,
                             in_dim_flat: int      ) -> Tuple[dict,
                                                              List[torch.tensor],
                                                              int,
                                                              int,
                                                              int]:
    """
    Get utils for Residual block
    """

    # Get utils from both path of ResidualBlock
    path_a, params_a, *_                            = linearize_layers(layer.path_a, in_dim, in_chans)
    path_b, params_b, in_dim, in_chans, in_dim_flat = linearize_layers(layer.path_b, in_dim, in_chans)

    # Create symbolic bounds that correspond to ReLU layer to come next
    weight = torch.eye(in_dim_flat)
    bias = torch.zeros(in_dim_flat)

    utils = {
        'path_a': path_a,
        'path_b': path_b
    }
    params = params_a + params_b

    return weight, bias, utils, params, in_dim, in_chans, in_dim_flat



def linearize_layers(net:      nn.Sequential,
                     in_dim:   int          ,
                     in_chans: int          ) -> Tuple[List[dict], 
                                                       List[torch.tensor],
                                                       int]:
    """
    Get utils from every layer of net
    """

    net_layers = get_net_layers(net)
    in_dim_flat = get_in_dim_flat(in_chans, in_dim)

    layers, params = [], []

    # Iterate over every layer
    for layer in net_layers:
        
        type_ = type(layer)
        utils = {
            'type': type_.__name__
        }

        
        # If Linear layer
        if type_ == nn.Linear:

            weight, bias, in_dim_flat = linearize_linear(layer)

        
        # If Convolutional layer
        elif type_ == nn.Conv2d:

            weight, bias, in_dim, in_chans, in_dim_flat = linearize_conv(layer, in_dim)


        # If Normalization layer
        elif type_ == Normalization:

            weight, bias = linearize_norm(layer, in_dim, batch=False)


        # If Batch normalization layer
        elif type_ == nn.BatchNorm2d:

            weight, bias = linearize_norm(layer, in_dim, batch=True)

            prev_layer = layers[-1]
            assert 'relu_param' not in prev_layer
                
            # No bias when followed by Batch normalization layer
            weight_prev, _ = prev_layer['weight_bias']

            # Combine with previous layer (as there is no ReLU between them)
            weight = torch.matmul(weight, weight_prev)
            prev_layer['weight_bias'] = ( weight, bias )
            prev_layer['sym_bounds'] = init_symbolic_bounds(*prev_layer['weight_bias'])

            continue


        # If Residual block
        elif type_ == BasicBlock:

            resblock = linearize_resblock_paths(layer, in_dim, in_chans, in_dim_flat)
            weight, bias, utils_resblock, params_resblock, in_dim, in_chans, in_dim_flat = resblock

            utils.update(utils_resblock)
            params.extend(params_resblock)



        else:

            # If ReLU layer
            if type_ == nn.ReLU:
                
                # Initialize alpha parameter as a vector filled with zeros
                param = torch.zeros(in_dim_flat, requires_grad=True)
                params.append(param)

                # Add parameter to previous layer
                layers[-1]['relu_param'] = param


            # If Flatten layer
            elif type_ == nn.Flatten:

                in_chans = 1


            continue


        utils['weight_bias'] = ( weight, bias )
        utils['sym_bounds'] = init_symbolic_bounds(*utils['weight_bias'])

        layers.append(utils)


    return layers, params, in_dim, in_chans, in_dim_flat



def add_final_layer(layers:     List[dict],
                    true_label: int       ) -> None:
    """
    Artificially add a final layer, to subtract the true_label output node to every other output node
    """

    last_layer = layers[-1]

    assert 'relu_param' not in last_layer

    weight, bias = last_layer['weight_bias']

    weight_true_label = weight[true_label]
    weight = torch.cat([ weight[:true_label], weight[true_label + 1:] ])
    weight = weight_true_label - weight

    bias_true_label = bias[true_label]
    bias = torch.cat([ bias[:true_label], bias[true_label + 1:] ])
    bias = bias_true_label - bias

    last_layer['sym_bounds'] = init_symbolic_bounds(weight, bias)



def preprocess_bounds(layers: List[dict], 
                      l_0:    torch.tensor, 
                      u_0:    torch.tensor) -> Tuple[List[dict], 
                                                     torch.tensor, 
                                                     torch.tensor]:
    """
    Preprocess lower and upper bounds for every input node
    """
    
    # Flatten initial bounds
    l_0 = l_0.flatten()
    u_0 = u_0.flatten()

    # If first layer is a Normalization layer
    first_layer = layers[0]

    if first_layer['type'] == Normalization.__name__:

        weight, bias = first_layer['weight_bias']
        l_0 = torch.matmul(weight, l_0) + bias
        u_0 = torch.matmul(weight, u_0) + bias

        # Remove normalization layer
        layers = layers[1:]

    return layers, l_0, u_0
