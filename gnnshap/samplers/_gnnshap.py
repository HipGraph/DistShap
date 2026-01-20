import copy
import os
from typing import Tuple

import numpy as np
import torch
from scipy.special import binom
from torch import Tensor
import math
import gnnshap_cuda_extension as gnnshap_ext
from gnnshap.utils import get_logger

from ._base import BaseSampler

log = get_logger(__name__)

from torch.utils.cpp_extension import load

import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        module="torch.utils.cpp_extension")


class GNNShapSampler(BaseSampler):
    r"""This sampling algorithm is implemented in Cuda to speed up the sampling 
        process. It creates samples in parallel. The number of blocks and 
        threads can be adjusted. It can be also used in distributed settings.
        """

    def __init__(self, nplayers: int, nsamples: int, **kwargs) -> None:
        """number of players and number of samples are required.

        Args:
            nplayers (int): number of players
            nsamples (int): number of samples
            adjacent_sym (bool, optional): whether to sample symmetrically.
                The first published version was not symmetric. Symmetric version
                is more efficient. Non-symmetric version is kept for 
                reproducibility. Non-symmetric's weights are scaled to 100. It 
                can't be changed. It can't also handle self loops. Self loops
                need to be added manually. Defaults to True.
            num_blocks (int, optional): number of blocks for cuda.
                Defaults to 32.
            num_threads (int, optional): number of threads for cuda.
                Defaults to 256.
            weight_scale (int, optional): weight scale. It makes the total sum
                of the kernel weights equal to weight_scale. Defaults to 
                nsamples.
            device (str, optional): device to use. Defaults to 'cuda:0'.
            nselfloops (int, optional): number of self loops. When it is set to
                non-zero, it adds self loops to the mask matrix's columns. No
                need to add self loops manually in the edge_index masking step.
                Defaults to 0.
            world_size (int, optional): number of workers. Defaults to 1.
            rank (int, optional): worker rank. Defaults to 0.
            



        """
        super().__init__(nplayers=nplayers, nsamples=nsamples)
        self.num_blocks = kwargs.get('num_blocks', 32)
        self.num_threads = kwargs.get('num_threads', 256)
        self.adjacent_sym = kwargs.get('adjacent_sym', True)
        self.world_size = kwargs.get('world_size', 1)
        self.rank = kwargs.get('rank', 0)
        
        # default is nplayers
        self.weight_scale = kwargs.get('weight_scale', self.nsamples)
        self.nselfloops = kwargs.get('nselfloops', 0)

        self.device = kwargs.get('device', 'cuda:0')

        assert self.num_blocks > 0, \
            "Number of blocks should be greater than 0"
        assert self.num_threads > 0, \
            "Number of threads should be greater than 0"




    def sample(self) -> Tuple[Tensor, Tensor]:
        
        # original approach. Less efficient. No distributed support.
        if not self.adjacent_sym:

            assert self.nselfloops == 0, \
                ("Self loops are not supported in non-symmetric version",
                 "Please set it to 0 and add self loops manually")
            assert self.weight_scale == 100, \
                "non-symmetric sampling only supports weight scale of 100"

            assert self.world_size == 1, \
                "world_size should be 1 for non-symmetric version"
            
            assert self.rank == 0, \
                "rank should be 0 for non-symmetric version"
            
            mask_matrix = torch.zeros((self.nsamples, self.nplayers),
                                    dtype=torch.bool, requires_grad=False,
                                    device=self.device)
            kernel_weights = torch.zeros((self.nsamples), dtype=torch.float64,
                                        requires_grad=False, device=self.device)
            gnnshap_ext.sample(mask_matrix, kernel_weights, self.nplayers,
                            self.nsamples, self.num_blocks, self.num_threads)
        
        else: # symmetric version. More efficient. Distributed support.
            
            assert self.world_size >= 0, \
                "Number of workers should be greater than or equal to 0"
            assert self.rank >= 0, \
                "Worker rank should be greater than or equal to 0"
            assert self.weight_scale >= 1, \
                "Weight scale should be greater than 0"
            assert self.rank < self.world_size, \
                "Worker rank should be less than number of workers"
            
            nsamples_half = math.ceil(self.nsamples / 2)
            worker_samples = math.ceil(nsamples_half / self.world_size)
            
            worker_offset = self.rank * worker_samples

            if self.rank == self.world_size - 1:
                worker_samples = nsamples_half - (self.world_size - 1
                                                  ) * worker_samples


            partial_samples = worker_samples * 2

            mask_matrix = torch.ones((partial_samples,
                                      self.nplayers + self.nselfloops),
                                      dtype=torch.bool, requires_grad=False,
                                      device=self.device)
            kernel_weights = torch.empty((partial_samples), dtype=torch.float64,
                                         requires_grad=False,
                                         device=self.device)
            gnnshap_ext.partialAdjacentSymSample(mask_matrix, kernel_weights,
                                                 self.nplayers, self.nsamples,
                                                 worker_offset, self.nselfloops,
                                                 self.world_size, self.rank,
                                                 self.weight_scale,
                                                 self.num_blocks,
                                                 self.num_threads)
            
        return mask_matrix, kernel_weights
