from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from gnnshap.utils import get_logger

from ._base import BaseSolver

import torch.distributed as dist
from time import time

log = get_logger(__name__)

    

class DistCGLSSolver(BaseSolver):
    """Distributed Conjugate Gradient Weighted Solver.
    """
    def __init__(self, mask_matrix: Tensor, kernel_weights: Tensor,
                 yhat: Tensor, fnull: Tensor, ffull: Tensor,
                 precision: torch.dtype=torch.float32, device: str='cpu',
                 **kwargs: dict) -> None:
        """Initialization.

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
        


        self.nplayers = self.mask_matrix.size(1)
        
        self.world_size = kwargs.get('world_size', 0)
        
        self.niters = kwargs.get('niters', 100)
        self.tol = kwargs.get('tol', 1e-6)
   
    def solve(self) -> Tuple[np.array, dict]:
        r"""Solves the regression problem.

        Args:

        Returns:
            np.array: shapley_values
            dict: dictionary containing solver statistics
        """

        self.comm_time = 0
        self.comp_time = 0

        start = time()

        # no need to add base value as player thanks to this:
        # (base + shap_values) = ffull
        self.yhat -= self.fnull


        # eliminate one variable with the constraint that all features
        # sum to the output
        self.yhat = (self.yhat \
                     - self.mask_matrix[:, -1] * (self.ffull - self.fnull))       
        self.mask_matrix[:, :-1] = self.mask_matrix[:, :-1]  \
            - self.mask_matrix[:,-1].unsqueeze(1)

        self.mask_matrix = self.mask_matrix[:, :-1] # remove the last column

        self.kernel_weights = torch.sqrt(self.kernel_weights).unsqueeze(1)

        self.mask_matrix *= self.kernel_weights
        self.yhat = self.yhat.unsqueeze(1) * self.kernel_weights

        self.solution = torch.zeros(self.nplayers-1, 1, dtype=self.precision,
                                    device=self.device)

        iter = 0
        norms_arr = []

        r = (self.mask_matrix @ self.solution) - self.yhat
        s = self.mask_matrix.T @ r
        torch.cuda.synchronize() # TODO: remove this. It is for accurate timing
        self.comp_time += time() - start
        start = time()
        dist.all_reduce(s)
        self.comm_time += time() - start
        start = time()
        p = s
        norms0 = torch.linalg.norm(s)
        gamma = norms0 ** 2
        normx = torch.linalg.norm(self.solution)
        xmax = normx
        flag = 0
        torch.cuda.synchronize() # TODO: remove this. It is for accurate timing
        self.comp_time += time() - start
        while(iter < self.niters and flag == 0):
            start = time()
            iter = iter + 1
            q = self.mask_matrix @ p
            delta = q.T @ q
            torch.cuda.synchronize() # TODO: remove this. It is for accurate timing
            self.comp_time += time() - start
            start = time()
            dist.all_reduce(delta)
            self.comm_time += time() - start
            start = time()
            alpha = gamma / delta
            self.solution = self.solution + alpha * p
            r = r - alpha * q
            s = self.mask_matrix.T @ r
            torch.cuda.synchronize() # TODO: remove this. It is for accurate timing
            self.comp_time += time() - start
            start = time()
            dist.all_reduce(s)
            self.comm_time += time() - start
            start = time()
            norms = torch.linalg.norm(s)
            gamma1 = gamma
            gamma = norms ** 2
            beta = gamma / gamma1
            p = s + beta * p


            normx = torch.linalg.norm(self.solution)
            norms_arr.append(norms.item())
            xmax = max(normx, xmax)
            flag = (norms <= norms0 * self.tol) or (normx * self.tol >= 1)
            torch.cuda.synchronize() # TODO: remove this. It is for accurate timing
            self.comp_time += time() - start

        start = time()
        phi = 0
        if self.rank == 0:
            phi = torch.zeros(self.nplayers, device=self.device,
                                  dtype=self.precision)
            phi[:-1] = -self.solution.squeeze()
            phi[-1] = (self.ffull - self.fnull) - torch.sum(phi[:-1])
            phi = phi.detach().cpu().numpy()
        self.comp_time += time() - start
        return phi, {'comm_time': self.comm_time, 'comp_time':self.comp_time,
                     'niters': iter, 'norms': np.array(norms_arr)}