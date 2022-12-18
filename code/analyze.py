import time
import torch
import torch.optim as optim

from backsubstitution import (
    get_symbolic_bounds, backsubstitute
)
from preprocessing import (
    add_final_layer, linearize_layers, 
    preprocess_bounds
)
from utils import (
    TIME_LIMIT, TIME_START,
    get_numerical_bound
)

torch.autograd.set_detect_anomaly(True)



def analyze(net, inputs, eps, true_label) -> bool:

    # Get an overview of layers in net
    in_chans, in_dim = inputs.shape[1], inputs.shape[2]
    layers, params, _, _, out_dim_flat = linearize_layers(net, in_dim, in_chans)
    add_final_layer(layers, out_dim_flat, true_label)

    # Initialize lower and upper bounds
    l_0 = (inputs - eps).clamp(0, 1)
    u_0 = (inputs + eps).clamp(0, 1)
    layers, l_0, u_0 = preprocess_bounds(layers, l_0, u_0)

    # Optimization
    optimizer = optim.Adam(params, lr=1)
    
    # while time.time() - TIME_START < TIME_LIMIT:
    while True:
        optimizer.zero_grad()

        layers_linearized = get_symbolic_bounds(layers, l_0, u_0, prev_layers=[])
        # layers_linearized = add_sym_bounds(layers, l_0, u_0, prev_layers)
        sym_bounds = backsubstitute(layers_linearized)
        l_weight, l_bias = sym_bounds[0], sym_bounds[2]
        l = get_numerical_bound(l_0, u_0, l_weight, l_bias)

        # Errors whenever at least one output upper bound is greater than lower bound of true_label
        err = torch.min(l)
        if err > 0:
            return True

        # Compute loss, and backpropagate to learn alpha parameters
        loss = torch.log(-err)
        loss.backward()
        optimizer.step()

    return False
