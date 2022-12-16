from typing import List, Tuple

import torch
import torch.nn as nn

from networks import Normalization
from resnet import BasicBlock
from utils import (
    compute_out_dim, get_conv_matrix,
    get_numerical_bounds
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



def linearize_norm(layer:    nn.Module, 
                   in_dim:   int      ,
                   is_batch: bool      = False) -> Tuple[torch.tensor,
                                                         torch.tensor]:
    """
    Get utils for (Batch) Normalization layer
    """

    # If Normalization layer
    if not is_batch:

        mean = layer.mean.detach()
        sigma = layer.sigma.detach()

        # Build weight and bias
        weight = 1 / sigma
        bias = - mean * weight

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
    bias = bias.repeat_interleave(n)

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

    utils = {}
            
    # Create Linear layer that splits data in two
    weight_in = torch.eye(in_dim_flat)
    weight_in = torch.cat([ weight_in, weight_in ])
    bias_in = torch.zeros(in_dim_flat * 2)
    utils_in = {
        'type': nn.Linear.__name__,
        'sym_bounds': ( weight_in, weight_in.clone(), bias_in, bias_in.clone() )
    }
    utils['layer_in'] = utils_in


    # Get utils from both path of ResidualBlock
    path_a, params_a, *_                            = linearize_layers(layer.path_a, in_dim, in_chans)
    path_b, params_b, in_dim, in_chans, in_dim_flat = linearize_layers(layer.path_b, in_dim, in_chans)

    params = params_a + params_b

    # Artificially add Identity layers to have paths with same length
    weight_identity = torch.eye(in_dim_flat)
    bias_identity = torch.zeros(in_dim_flat)
    utils_identity = {
        'type': nn.Identity.__name__,
        'sym_bounds': ( weight_identity, weight_identity.clone(), bias_identity, bias_identity.clone() )
    }

    diff = len(path_a) - len(path_b)
    if diff > 0:
        path_b += [ utils_identity.copy() for _ in range(diff)  ]
    elif diff < 0:
        path_a += [ utils_identity.copy() for _ in range(-diff) ]

    utils['path_a'] = path_a
    utils['path_b'] = path_b


    # Create Linear layer that puts data back together
    weight_out = torch.eye(in_dim_flat)
    weight_out = torch.cat([ weight_out, weight_out ], dim=1)
    bias_out = torch.zeros(in_dim_flat)
    utils_out = {
        'type': nn.Linear.__name__,
        'weight_bias': ( weight_out, bias_out )
    }
    utils['layer_out'] = utils_out

    return utils, params, in_dim, in_chans, in_dim_flat



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

            utils['weight_bias'] = ( weight, bias )

        
        # If Convolutional layer
        elif type_ == nn.Conv2d:

            weight, bias, in_dim, in_chans, in_dim_flat = linearize_conv(layer, in_dim)

            utils['weight_bias'] = ( weight, bias )


        # If ReLU layer
        elif type_ == nn.ReLU:
            
            # Initialize alpha parameter as a vector filled with zeros
            param = torch.zeros(in_dim_flat, requires_grad=True)
            params.append(param)

            # Add parameter to previous layer
            last_layer = layers[-1]
            if last_layer['type'] == BasicBlock.__name__:
                last_layer = last_layer['layer_out']
            last_layer['relu_param'] = param
            continue


        # If (Batch) Normalization layer
        elif type_ in [ Normalization, nn.BatchNorm2d ]:

            is_batch = type_ == nn.BatchNorm2d
            utils['weight_bias'] = linearize_norm(layer, in_dim, is_batch=is_batch)

            
        # If Residual block
        elif type_ == BasicBlock:

            utils_resblock, params_resblock, in_dim, in_chans, in_dim_flat = linearize_resblock_paths(layer, in_dim, in_chans, in_dim_flat)

            utils.update(utils_resblock)
            params.extend(params_resblock)


        # If Flatten layer
        elif type_ == nn.Flatten:

            in_chans = 1
            continue


        else:
            continue


        layers.append(utils)


    return layers, params, in_dim, in_chans, in_dim_flat



def add_final_layer(layers:     List[dict], 
                    out_dim:    int,
                    true_label: int       ) -> List[dict]:
    """
    Artificially add a final layer, to subtract the true_label output node to every other output node
    """

    # -1 on diagonal, and 1 for every element in the true_label column
    final_weight = -torch.eye(out_dim)
    final_weight[:, true_label] = 1.
    final_weight = torch.cat([ final_weight[:true_label], final_weight[true_label + 1:] ])

    # No bias
    final_bias = torch.zeros(out_dim - 1)

    assert 'weight_bias' in layers[-1] and 'relu_param' not in layers[-1]

    weight, bias = layers[-1]['weight_bias']

    weight = torch.matmul(final_weight, weight)
    bias   = torch.matmul(final_weight, bias)   + final_bias

    layers[-1]['weight_bias'] = ( weight, bias )



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

    for i, layer in enumerate(layers):

        if 'relu_param' in layer:
            break

        weight, bias = layer['weight_bias']
        l_0, u_0 = get_numerical_bounds(l_0, u_0, weight, weight, bias, bias)

    return layers[i:], l_0, u_0
