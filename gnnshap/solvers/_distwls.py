from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from gnnshap.utils import get_logger

from ._base import BaseSolver

import torch.distributed as dist
from time import time

log = get_logger(__name__)

    

class DistWLSSolver(BaseSolver):
    """Solver that uses PyTorch Distributed with allreduce to solve weighted 
    least squares.
    """
    def __init__(self, mask_matrix: Tensor, kernel_weights: Tensor,
                 yhat: Tensor, fnull: Tensor, ffull: Tensor,
                 precision: torch.dtype=torch.float32, device: str='cpu',
                 **kwargs: dict) -> None:
        """Initialization for DistWLSSolver.

        Args:
            mask_matrix (Tensor): mask matrix
            kernel_weights (Tensor): kernel weights
            yhat (Tensor): model predictions
            fnull (float): null model prediction
            ffull (float): full model prediction
            precision (torch.dtype): precision of the tensors. Defaults to
                torch.float32.
            device (str): device to run the solver. Defaults to 'cpu'.
            **kwargs (dict): additional arguments
        """
        super().__init__(mask_matrix, kernel_weights, yhat, fnull, ffull,
                         precision, device, **kwargs)

        self.rank = kwargs.get('rank', 0)
        
        self.world_size = kwargs.get('world_size', 0)

   
    def solve(self) -> Tuple[np.array, dict]:
        r"""Solves WLS problem via PyTorch Distributed.

        Args:

        Returns:
            np.array: shapley_values
            dict: dictionary containing solver statistics
        """

        self.comp_time = 0
        self.comm_time = 0
        
        start = time()

        # no need to add base value as player thanks to this:
        # (base + shap_values) = ffull
        self.yhat -= self.fnull


        # eliminate one variable with the constraint that all features
        # sum to the output
        self.yhat = self.yhat - self.mask_matrix[:, -1] * (
            self.ffull - self.fnull)       
        self.mask_matrix[:, :-1] = self.mask_matrix[:, :-1]  \
            - self.mask_matrix[:,-1].unsqueeze(1)

        MW_t = (self.mask_matrix[:, :-1] * self.kernel_weights.unsqueeze(1)
                ).transpose(0, 1)
        # last column was not used in the above calculation
        self.mask_matrix[:, -1] = self.yhat

        G = torch.mm(MW_t, self.mask_matrix)
        
        torch.cuda.synchronize() # TODO: remove this. It is for accurate timing

        self.comp_time += time() - start

        start = time()
        dist.reduce(G, 0)
        self.comm_time += time() - start

        start = time()

        phi = 0
        if self.rank == 0:
            # solve the linear system
            phi_wls = torch.linalg.solve(G[:,:-1], G[:,-1])

            # no +1 since the last column was not removed like in the
            # non-distributed version
            phi = torch.zeros(self.mask_matrix.size(1), device=self.device)
            phi[:-1] = phi_wls
            phi[-1] = (self.ffull - self.fnull) - torch.sum(phi_wls)
            phi = phi.detach().cpu().numpy()

        self.comp_time += time() - start


        return phi, {'comm_time': self.comm_time, 'comp_time': self.comp_time}