from typing import List, Tuple

import time
import torch
import torch.optim as optim

from backsubstitution import get_layers_utils, backsubstitute
from networks import Normalization
from utils import get_numerical_bounds




def preprocess_bounds(layers: List[dict], 
                      l_0:    torch.tensor, 
                      u_0:    torch.tensor) -> Tuple[List[dict], 
                                                     torch.tensor, 
                                                     torch.tensor]:
    """
    Preprocess lower and upper bounds for every input node
    """

    # If first layer is a Normalization layer
    first_layer = layers[0]

    if first_layer['type'] == Normalization.__name__:

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



def analyze(net, inputs, eps, true_label) -> bool:

    # Get an overview of layers in net
    in_dim = inputs.shape[2]
    layers, parameters, _ = get_layers_utils(net, in_dim)

    # Initialize lower and upper bounds
    l_0 = (inputs - eps).clamp(0, 1)
    u_0 = (inputs + eps).clamp(0, 1)
    layers, l_0, u_0 = preprocess_bounds(layers, l_0, u_0)

    # Optimization
    optimizer = optim.Adam(parameters, lr=1)

    i = 0
    while i < 1000:
    # while time.time() - TIME_START < TIME_LIMIT:
        optimizer.zero_grad()

        # Get lower and upper symbolic bounds using DeepPoly
        symbolic_bounds = backsubstitute(layers, l_0, u_0)
        # Using them, compute lower and upper numerical bounds
        l, u = get_numerical_bounds(l_0, u_0, *symbolic_bounds)

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
            print(errors.data)
            print(loss.data)
            print()
        
        i+= 1

    return False
