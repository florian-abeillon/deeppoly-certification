from typing import List, Tuple

import torch
import torch.nn as nn

from networks import Normalization
from resnet import BasicBlock
from utils import compute_out_dim, get_conv_matrix



def get_in_dim_flat(in_dim:   int,
                    in_chans: int) -> int:
    """
    Get dimension of flattened tensor
    """
    return in_chans * in_dim**2




def get_utils_identity(in_dim:   int,
                       in_chans: int) -> Tuple[torch.tensor,
                                               torch.tensor]:
    """
    Get utils for Identity layer
    """
    in_dim_flat = get_in_dim_flat(in_dim, in_chans)
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
    in_dim = weight.shape[0]
    return weight, bias, in_dim



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

    layers, params = [], []
    to_skip = 0

    for layer in net.modules():
        
        type_ = type(layer)
        utils = {
            'type': type_.__name__
        }

        
        # If Linear layer
        if type_ == nn.Linear:

            weight, bias, in_dim = get_utils_linear(layer)
            utils['weight_bias'] = ( weight, bias )

        
        # If Convolutional layer
        elif type_ == nn.Conv2d:

            weight, bias, in_dim, in_chans = get_utils_conv(layer, in_dim)
            utils['weight_bias'] = ( weight, bias )


        # If ReLU layer
        elif type_ == nn.ReLU:
            
            # Initialize alpha parameter as a vector filled with zeros
            in_dim_flat = get_in_dim_flat(in_dim, in_chans)
            param = torch.zeros(in_dim_flat, requires_grad=True)
            params.append(param)

            # Add parameter to previous layer
            layers[-1]['relu_param'] = param
            continue


        # If Normalization layer
        elif type_ == Normalization:

            utils['weight_bias'] = get_utils_norm(layer, in_dim)

            # TODO: To remove
            utils['layer'] = layer


        # If Batch normalization layer
        elif type_ == nn.BatchNorm2d:

            utils['weight_bias'] = get_utils_batch_norm(layer, in_dim)


        # If Identity layer
        elif type_ == nn.Identity:

            utils['weight_bias'] = get_utils_identity(in_dim, in_chans)

            
        # If Residual block
        elif type_ == BasicBlock:

            # Add Identity layer before BasicBlock to have a layer with weight/bias
            utils_identity = {
                'type': nn.Identity.__name__,
                'weight_bias': get_utils_identity(in_dim, in_chans)
            }
            layers.append(utils_identity)

            # Get utils from both path of ResidualBlock
            path_a, params_a, *_               = get_layers_utils(layer.path_a, in_dim, in_chans)
            path_b, params_b, in_dim, in_chans = get_layers_utils(layer.path_b, in_dim, in_chans)

            # Add their parameters to the list of tracked parameters
            params.extend(params_a)
            params.extend(params_b)

            utils['prev_path'] = layers.copy()
            utils['path_a'] = path_a
            utils['path_b'] = path_b
            layers.append(utils)

            # To skip the layers that were already captured in the Residual block
            to_skip = len(path_a) + len(path_b) + 1

            
        # If Flatten layer
        elif type_ == nn.Flatten:

            in_dim = in_chans * in_dim**2
            in_chans = 1
            continue


        else:
            continue


        # TODO: Find better way?
        if to_skip > 0:
            to_skip -= 1
        else:
            layers.append(utils)


    return layers, params, in_dim, in_chans




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
    
    # # Flatten initial bounds
    # l_0 = l_0.flatten()
    # u_0 = u_0.flatten()

    # If first layer is a Normalization layer
    first_layer = layers[0]

    if first_layer['type'] == Normalization.__name__:

        # TODO: Make it work
        weight, bias = first_layer ['weight_bias']
        l_1 = torch.matmul(weight, l_0.flatten()) + bias
        u_1 = torch.matmul(weight, u_0.flatten()) + bias

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
