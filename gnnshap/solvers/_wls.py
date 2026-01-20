from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from time import time

from gnnshap.utils import get_logger

from ._base import BaseSolver

log = get_logger(__name__)


class WLSSolver(BaseSolver):
    """Solver that uses pytorch to solve the weighted least squares problem."""
    def __init__(self, mask_matrix: Tensor, kernel_weights: Tensor,
                 yhat: Tensor, fnull: Tensor, ffull: Tensor,
                 precision: torch.dtype = torch.float32, device: str='cpu',
                 **kwargs: dict) -> None:
        """Initialization for WLSSolver.

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
        self.device = kwargs.get('device', 'cpu')

    
    def solve(self) -> np.array:
        """Solves the weighted least squares problem and learns the shapley
        values.

        Returns:
            np.array: shapley values
        """
        # print("device: ", self.device)
        with torch.no_grad():
            return self._solve()
   
    def _solve(self) -> Tuple[np.array, dict]:
        r"""Solves weighted least squares problem and learns shapley values

        Args:
            mask_matrix (Tensor): coalition matrix
            kernel_weights (Tensor): coalition weight values
            ey (Tensor): coalition predictions

        Returns:
            np.array: shapley values
            dict: dictionary containing solver statistics

        """

        self.comp_time = 0

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
        
        self.mask_matrix = self.mask_matrix[:, :-1]


        MW_t = (self.mask_matrix * self.kernel_weights.unsqueeze(1)
                ).transpose(0, 1)

        G = torch.mm(MW_t, self.mask_matrix)

        w = torch.linalg.solve(G, torch.mm(MW_t, self.yhat.unsqueeze(1)))

        w = w.cpu().flatten()
        phi = torch.zeros(self.mask_matrix.size(1)+1, requires_grad=False, 
                          device='cpu')
        phi[:-1] = w
        phi[-1] = (self.ffull - self.fnull) - torch.sum(w)

        self.comp_time = time() - start

        return phi.numpy(), {'comp_time': self.comp_time}
