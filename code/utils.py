from typing import Tuple

import time
import torch
import torch.nn.functional as F


TIME_START = time.time()
TIME_LIMIT = 60



def batch_matmatmul(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    """
    Matrix multiplication along several dimensions
    """
    return torch.einsum('bij, bjk -> bik', a, b)


def batch_matvecmul(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    """
    Matrix-vector multiplication along several dimensions
    """
    return torch.einsum('bij, bj -> bi', a, b)


def get_pos_neg(t: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """
    Get matrices of positive and negative coefficients of t
    """ 
    return F.relu(t), - F.relu(-t)
