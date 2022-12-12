import time
import torch
import torch.optim as optim

from backsubstitution import (
    add_sym_bounds, backsubstitute
)
from preprocessing import (
    add_final_layer, linearize_layers, 
    preprocess_bounds
)
from utils import (
    TIME_LIMIT, TIME_START,
    get_numerical_bounds
)



def analyze(net, inputs, eps, true_label) -> bool:

    # Get an overview of layers in net
    in_chans, in_dim = inputs.shape[1], inputs.shape[2]
    layers, params, _, _, out_dim_flat = linearize_layers(net, in_dim, in_chans)
    layers = add_final_layer(layers, out_dim_flat, true_label)

    # Initialize lower and upper bounds
    l_0 = (inputs - eps).clamp(0, 1)
    u_0 = (inputs + eps).clamp(0, 1)
    layers, l_0, u_0 = preprocess_bounds(layers, l_0, u_0)

    # Optimization
    optimizer = optim.Adam(params, lr=1)
    
    while time.time() - TIME_START < TIME_LIMIT:
        optimizer.zero_grad()

        layers_linearized = add_sym_bounds(layers, l_0, u_0)
        sym_bounds = backsubstitute(layers_linearized)
        l, _ = get_numerical_bounds(l_0, u_0, *sym_bounds)

        # Errors whenever at least one output upper bound is greater than lower bound of true_label
        err = torch.min(l)
        if err > 0:
            return True

        # Compute loss, and backpropagate to learn alpha parameters
        loss = torch.log(-err)
        loss.backward()
        optimizer.step()

    return False
