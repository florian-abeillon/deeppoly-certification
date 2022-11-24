from typing import List, Tuple

import torch
import torch.nn as nn

from networks import Normalization
from resnet import BasicBlock
from utils import compute_out_dim, get_conv_matrix



def get_in_dim_flat(in_chans: int,
                    in_dim:   int) -> int:
    """
    Get dimension of flattened tensor
    """
    return in_chans * in_dim**2




def get_utils_identity(in_dim_flat: int) -> Tuple[torch.tensor,
                                                  torch.tensor]:
    """
    Get utils for Identity layer
    """
    weight = torch.eye(in_dim_flat)
    bias = torch.zeros(in_dim_flat)
    return weight, bias



def get_utils_linear(layer: nn.Module) -> Tuple[torch.tensor,
                                                torch.tensor,
                                                int]:
    """
    Get utils for Linear layer
    """
    weight = layer.weight.detach()
    bias = layer.bias.detach()
    out_dim = layer.out_features
    return weight, bias, out_dim



def get_utils_conv(layer:  nn.Module,
                   in_dim: int      ) -> Tuple[torch.tensor,
                                               torch.tensor,
                                               int,
                                               int]:
    """
    Get utils for Linear layer
    """

    weight = layer.weight.detach()
    bias = layer.bias.detach()
    p = layer.padding[0]
    s = layer.stride[0]

    # Compute out_dim
    out_chans, in_chans, k, _ = weight.shape
    out_dim = compute_out_dim(in_dim, k, p, s)

    # Get flattened convolutional matrix, and bias
    weight = get_conv_matrix(weight, in_dim, out_dim, in_chans, out_chans, k, p, s)
    bias = bias.repeat_interleave(out_dim**2)

    return weight, bias, out_dim, out_chans



def get_utils_norm(layer:    nn.Module, 
                   in_dim:   int      ) -> Tuple[torch.tensor,
                                                 torch.tensor]:
    """
    Get utils for Normalization layer
    """

    mean = layer.mean.detach()
    sigma = layer.sigma.detach()

    # Build weight and bias
    weight = 1 / sigma
    bias = -mean * weight

    # Reshape weight and bias
    n = in_dim**2
    weight = weight.repeat_interleave(n)
    bias = bias.repeat_interleave(n)

    # Turn weight vector into diagonal matrix (for matrix multiplication)
    weight = torch.diag(weight)

    return weight, bias



def get_utils_batch_norm(layer:    nn.Module, 
                         in_dim:   int      ) -> Tuple[torch.tensor,
                                                       torch.tensor]:
    """
    Get utils for Batch normalization layer
    """

    mean = layer.running_mean.detach()
    var = layer.running_var.detach()
    eps = layer.eps
    gamma = layer.weight.detach()

    # Build weight and bias
    weight = gamma / torch.sqrt(var + eps)
    bias = layer.bias.detach()
    bias -= mean * weight

    # Reshape weight and bias
    n = in_dim**2
    weight = weight.repeat_interleave(n)
    bias = bias.repeat_interleave(n)

    # Turn weight vector into diagonal matrix (for matrix multiplication)
    weight = torch.diag(weight)

    return weight, bias



def get_layers_utils(net:      nn.Sequential,
                     in_dim:   int          ,
                     in_chans: int          ) -> Tuple[List[dict], 
                                                       List[torch.tensor],
                                                       int]:
    """
    Get utils from every layer of net
    """

    in_dim_flat = get_in_dim_flat(in_chans, in_dim)

    layers, params = [], []
    to_skip = 0

    for layer in net.modules():
        
        type_ = type(layer)

        # TODO: Find better way?
        if to_skip > 0 and type_ in [ nn.Linear, nn.Conv2d, Normalization, nn.BatchNorm2d, BasicBlock ]:
            to_skip -= 1
            continue

        utils = {
            'type': type_.__name__
        }

        
        # If Linear layer
        if type_ == nn.Linear:

            assert in_chans == 1

            weight, bias, in_dim_flat = get_utils_linear(layer)
            in_dim = None

            assert weight.shape[0] == in_dim_flat

            utils['weight_bias'] = ( weight, bias )

        
        # If Convolutional layer
        elif type_ == nn.Conv2d:

            weight, bias, in_dim, in_chans = get_utils_conv(layer, in_dim)
            in_dim_flat = get_in_dim_flat(in_chans, in_dim)

            assert weight.shape[0] == in_dim_flat

            utils['weight_bias'] = ( weight, bias )


        # If ReLU layer
        elif type_ == nn.ReLU:
            
            # Initialize alpha parameter as a vector filled with zeros
            param = torch.zeros(in_dim_flat, requires_grad=True)
            params.append(param)

            # Add parameter to previous layer
            layers[-1]['relu_param'] = param
            continue


        # If Normalization layer
        elif type_ == Normalization:

            utils['weight_bias'] = get_utils_norm(layer, in_dim)


        # If Batch normalization layer
        elif type_ == nn.BatchNorm2d:

            utils['weight_bias'] = get_utils_batch_norm(layer, in_dim)


        # # If Identity layer
        # elif type_ == nn.Identity:

        #     utils['weight_bias'] = get_utils_identity(in_dim_flat)

            
        # If Residual block
        elif type_ == BasicBlock:

            # Get utils from both path of ResidualBlock
            path_a, params_a, *_                            = get_layers_utils(layer.path_a, in_dim, in_chans)
            path_b, params_b, in_dim, in_chans, in_dim_flat = get_layers_utils(layer.path_b, in_dim, in_chans)

            # Add their parameters to the list of tracked parameters
            params.extend(params_a)
            params.extend(params_b)

            utils['path_a'] = layers + path_a
            utils['path_b'] = layers + path_b

            # Reset layers (previous path already encapsulated in path_a and path_b)
            layers = [utils]

            # Add Identity layer before BasicBlock to have a layer with weight/bias
            utils_identity = {
                'type': nn.Identity.__name__,
                'weight_bias': get_utils_identity(in_dim_flat)
            }
            # layers.append(utils_identity)

            # layers.append(utils)

            # To skip the layers that were already captured in the Residual block
            to_skip = len(path_a) + len(path_b) + 1


        # If Flatten layer
        elif type_ == nn.Flatten:

            in_chans = 1
            in_dim = None
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
    
    # Flatten initial bounds
    l_0 = l_0.flatten()
    u_0 = u_0.flatten()

    # If first layer is a Normalization layer
    first_layer = layers[0]

    if first_layer['type'] == Normalization.__name__:

        weight, bias = first_layer ['weight_bias']
        l_0 = torch.matmul(weight, l_0.flatten()) + bias
        u_0 = torch.matmul(weight, u_0.flatten()) + bias

        # Remove normalization layer
        layers = layers[1:]
    
    # Flatten initial bounds
    l_0 = l_0.flatten()
    u_0 = u_0.flatten()

    return layers, l_0, u_0
