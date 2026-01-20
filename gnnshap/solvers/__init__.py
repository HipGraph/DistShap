from torch import Tensor
import torch

from ._base import BaseSolver
from ._wlr import WLRSolver
from ._wls import WLSSolver
from ._distwls import DistWLSSolver
from ._distcgls import DistCGLSSolver
from ._cgls import CGLSSolver


def get_solver(solver_name: str, mask_matrix: Tensor, kernel_weights: Tensor,
               yhat: Tensor, fnull: float, ffull: float,
               precision: torch.dtype = torch.float32, device: str = 'cpu',
                 **kwargs: dict) -> BaseSolver:
    """Returns the instanciated solver based on the name.

    Args:
        solver_name (str): Solver name
        mask_matrix (Tensor): mask matrix
        kernel_weights (Tensor): kernel weights
        yhat (Tensor): model predictions
        fnull (float): null model prediction
        ffull (float): full model prediction
        precision (torch.dtype): precision of the tensors. Defaults to
            torch.float32.
        device (str): device to run the solver. Defaults to 'cpu'.

    Raises:
        KeyError: If solver name is not found

    Returns:
        BaseSolver: Instanciated solver
    """
    solvers = {
        'WLSSolver': WLSSolver,
        'WLRSolver': WLRSolver,
        'DistWLSSolver': DistWLSSolver,
        'DistCGLSSolver': DistCGLSSolver,
        'CGLSSolver': CGLSSolver,
    }

    try:
        return solvers[solver_name](mask_matrix, kernel_weights, yhat, fnull,
                                    ffull, precision=precision, device=device,
                                    **kwargs)
    except KeyError as exc:
        raise KeyError(f"Solver '{solver_name}' not found!") from exc
