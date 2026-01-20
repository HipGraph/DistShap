from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor

from gnnshap.utils import get_logger

log = get_logger(__name__)


class BaseSampler(ABC):
    """A base class for samplers
    """

    def __init__(self, nplayers: int, nsamples: int) -> None:
        assert nplayers > 1, "Number of players should be greater than 1"
        assert nsamples > 1, "Number of samples should be a positive number"
        self.nplayers = nplayers
        self.nsamples = nsamples

        if self.nsamples % 2 != 0: # make sure nsamples is even
            self.nsamples += 1

        self.max_samples = 2 ** 30
        if self.nplayers <= 30:
            self.max_samples = 2 ** self.nplayers - 2
        # don't use more samples than 2 ** 30
        self.nsamples = min(self.nsamples, self.max_samples)

    @abstractmethod
    def sample(self) -> Tuple[Tensor, Tensor]:
        """An abstract method that all samplers should override.

        Returns:
            Tuple[Tensor, Tensor]: 2d booelan mask_matrix and  1d coalition weights.
        """