import os
import math
import time
from typing import Callable, List, Tuple, Union

import torch
import torch_geometric
from torch import BoolTensor, Tensor
from torch.nn.functional import softmax
from torch_geometric.utils import add_remaining_self_loops
from tqdm import tqdm
import math

from gnnshap.samplers._gnnshap import GNNShapSampler
from gnnshap.solvers import get_solver
from gnnshap.utils import *
from gnnshap.explanation import GNNShapExplanation
import gnnshap_cuda_extension as gnnshap_ext

log = get_logger(__name__)


def default_predict_fn(model: torch.nn.Module,
                       node_features: Tensor,
                       edge_index: Tensor,
                       node_idx: Union[int, List[int]],
                       edge_weight: Tensor = None) -> Tensor:
    r"""Model prediction function for prediction. A custom predict function can 
    be provided for different tasks.

    Args:
        model (torch.nn.Module): a PyG model.
        node_features (Tensor): node feature tensor.
        edge_index (Tensor): edge_index tensor.
        node_idx (Union[int, List[int]]): node index. Can be an integer or list
            (list for batched data).
        edge_weight (Tensor, optional): edge weights. Defaults to None.

    Returns:
        Tensor: model prediction for the node being explained.
    """

    model.eval()

    # [node_idx] will only work for non-batched. [node_idx, :] works for both
    pred = model.forward(node_features, edge_index, edge_weight=edge_weight)

    pred = softmax(pred[node_idx, :], dim=-1)
    return pred


class DistShapExplainer:
    """DistShap explainer. It can be used in both distributed and 
    non-distributed settings.

    Args:
        model (torch.nn.Module): PyG model.
        data (torch_geometric.data.Data): PyG data object.
        nhops (int, optional): Number of hops. If None, it will be set to the
            number of GNN layers. Defaults to None.
        device (Tuple[str, torch.device], optional): Device to use. Defaults to
            'cuda:0'.
        world_size (int, optional): Number of workers. Set to 1 for single
            worker. Defaults to 1.
        rank (int, optional): Worker rank. Set to 0 for single worker. Defaults
            to 0.
        precision (torch.dtype, optional): Result precision. Defaults to 
            torch.float32.
        forward_fn (Callable, optional): Prediction function. Defaults to
            default_predict_fn.
        progress_hide (bool, optional): Hide progress bar. Defaults to False.
        verbose (int, optional): Verbosity level. Defaults to 0.

    """
    def __init__(self, model: torch.nn.Module, data: torch_geometric.data.Data,
                 nhops: int = None, device: Tuple[str, torch.device] = 'cuda:0',
                 world_size: int = 1, rank: int = 0,
                 precision: torch.dtype = torch.float32,
                 forward_fn: Callable = default_predict_fn,
                 progress_hide: bool = False,
                 verbose: int = 0):

        self.model = model
        self.num_hops = nhops if nhops is not None else len(
            get_gnn_layers(self.model))
        self.data = data
        self.forward_fn = forward_fn  # prediction function
        self.progress_hide = progress_hide  # tqdm progress bar show or hide
        self.verbose = verbose  # to show or hide extra info
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.precision = precision
        self.has_self_loops = data.has_self_loops() if hasattr(
            data, 'has_self_loops') else False
        self.sampler = None  # will be set in explain.
        self.preds = None # will be set in compute_model_predictions.

        assert self.num_hops > 0, "Number of hops should be greater than zero."
        self.fnull = None
        self.fx = None

        assert self.world_size > 0, "World size should be greater than zero."
        assert self.rank >= 0, "Rank should be greater than or equal to zero."


    def compute_model_predictions(self, node_features: Tensor,
                                  edge_index: Tensor,
                                  mask_matrix: torch.BoolTensor, node_idx: int,
                                  batch_size: int, target_class: int
                                  ) -> Tuple[Tensor, int]:
        """Computes model predictions and writes results to self.preds variable.

        Args:
            node_features (Tensor): node features.
            edge_index (Tensor): edge index.
            mask_matrix (torch.BoolTensor): boolean 2d mask (coalition) matrix.
            node_idx (int): node index (it should be the relabeled node index 
                in the subgraph).
            batch_size (int): batch size. No batching if set to zero.
            target_class (int): Target class.

        Returns:
            Tuple[Tensor, int]: Returns predictions tensor and number of 
                computed samples.

        """
        assert batch_size >= 0, "Batch size can not be a negative number"

        nplayers = self.sampler.nplayers
        nselfloops = self.nselfloops

        # empty coalition prediction, use only self loop edges
        self.fnull = self.forward_fn(self.model, node_features,
                                    edge_index[:, nplayers:],
                                node_idx)[target_class].item()

        preds = torch.empty((mask_matrix.size(0)),
                                 dtype=self.precision, device=self.device,
                                 requires_grad=False)
        
        

        
        num_batches = math.ceil(mask_matrix.shape[0] / batch_size)
        num_nodes = node_features.size(0)

        # We need to get predictions of the same node for each subgraph.
        # [node0, node1, node2, node3 ... node0, node1, node2]
        node_indices = torch.arange(node_idx, batch_size * num_nodes, num_nodes,
                                    device=self.device)
        current_ind = 0

        edge_size = edge_index.size(1)
        batch_node_features = node_features.repeat(batch_size, 1)
        #create large batched edge indices
        batch_edge_index = torch.empty((2, edge_index.size(1) * batch_size),
                                       device=self.device, dtype=torch.long)
        
        for k, n_ind  in enumerate(range(0, batch_size * edge_size, edge_size)):
            batch_edge_index[:, n_ind:n_ind + edge_size
                             ] = edge_index + k * num_nodes


        # predictions for batches
        total_mask_time = 0
        for i in tqdm(range(num_batches), desc="Batched coalition scores",
                        disable=self.progress_hide, leave=False):
            batch_start = batch_size * i
            batch_end = min(batch_size * (i + 1), mask_matrix.shape[0])

            tmp_batch_size = batch_end - batch_start
            if tmp_batch_size < batch_size: # for the last batch
                node_indices = node_indices[:tmp_batch_size]
                batch_node_features = batch_node_features[
                    :tmp_batch_size * num_nodes]

                batch_edge_index = batch_edge_index[
                    :, :edge_index.size(1) * tmp_batch_size]

            mstart = time.time()


            masked_b_edge_index = batch_edge_index[
                :,mask_matrix[batch_start:batch_end].flatten()]
            
            total_mask_time += time.time() - mstart
            y_hat = self.forward_fn(model=self.model,
                                    node_features=batch_node_features,
                                    edge_index=masked_b_edge_index,
                                    edge_weight=None,
                                    node_idx=node_indices)
            preds[current_ind:current_ind + tmp_batch_size] =  \
                y_hat[:, target_class]
            current_ind += tmp_batch_size


        return preds, {'mask_time': total_mask_time}

    @torch.no_grad()
    def explain(self, node_idx: int, nsamples: int,
                    batch_size: int = 512, target_class: Union[int, None]=None,
                    solver_name:str = "WLSSolver",
                    **kwargs)-> GNNShapExplanation:
        """Explains the node with node_idx.

        Args:
            node_idx (int): Node index to be explained.
            nsamples (int): Number of samples.
            batch_size (int, optional): Batch size. GNNShap uses batched 
                prediction for computational efficiency. It is much faster
                than predicting each sample one by one. Defaults to 512.
            target_class (Union[int, None], optional): Which target class to
                explain. If None, the predicted class will be used. Defaults to
                None.
            solver_name (str, optional): Solver name. Defaults to "WLSSolver".

        Returns:
            GNNShapExplanation: GNNShap explanation object.
        """
        assert nsamples > 2, "Number of samples should be greater than 2."
        assert batch_size >= 1, "Batch size should be greater than zero."

        #TODO: assert solvers based on distributed or not

        device = self.device
        self.model= self.model.to(device)

        # temporarily switch add_self_loops to False if it is enabled.
        # we will add self loops manually.
        use_add_self_loops = has_add_self_loops(self.model)
        add_self_loops_swithced = False
        if use_add_self_loops:
            switch_add_self_loops(self.model)
            add_self_loops_swithced = True

        start_time = start = time.time()
        # we only need k-hop neighbors for explanation
        (subset, sub_edge_index, sub_mapping,
        sub_edge_mask) = pruned_comp_graph(node_idx, self.num_hops,
                                                self.data.edge_index.to(device),
                                                relabel_nodes=True)
        
        # main data can be on cpu on some cases. We just move edge_index to
        # gpu to speed up the computation above. We can move the rest of the
        # data to gpu after filtering the subset below.
        subset = subset.to(self.data.x.device)
        sub_mapping = sub_mapping.to(self.data.x.device)
        sub_edge_mask = sub_edge_mask.to(self.data.x.device)
        
        # new node_idx after relabeling in k hop subgraph.
        new_node_idx = sub_mapping[0].item()

        nplayers = sub_edge_index.size(1)

        sub_edge_index = sub_edge_index.to(device)

        # add remaining self loops if GNN layers' add_self_loops parameter
        # set to True
        sub_edge_index = add_remaining_self_loops(
            edge_index=sub_edge_index
            )[0] if use_add_self_loops else sub_edge_index

        nselfloops = sub_edge_index.size(1) - nplayers
        self.nselfloops = nselfloops

        compgraph_time = time.time() - start
        log.info(f"Computational graph finding time(s):\t",
                 f"{compgraph_time:.4f}")

        nsamples = math.ceil(nsamples / batch_size) * batch_size
        
        start = time.time()
        self.sampler = GNNShapSampler(nplayers=nplayers, nsamples=nsamples,
                                      world_size=self.world_size,
                                      rank=self.rank, device=device,
                                      nselfloops=nselfloops,
                                      adjacent_sym=True, **kwargs)
        

        mask_matrix, kernel_weights = self.sampler.sample()

        sampling_time = time.time() - start

        nsamples = self.sampler.nsamples  # samplers may update nsamples


        log.info(f"Sampling time(s):\t\t{sampling_time:.4f}")


        self.model.eval()

        if self.verbose == 1:
            print(f"Number of samples: {self.sampler.nsamples}, "
                  f"sampler:{self.sampler.__class__.__name__}, "
                  "batch size: {batch_size}")

        
        
        node_features = self.data.x[subset]
        node_features= node_features.to(device)

        start = time.time()
         # use the predicted class if no target class is provided.
        if target_class is None:
            # target_class = self.data.y[node_idx].item() # for ground truth
            pred = self.forward_fn(self.model, node_features,
                                    sub_edge_index, new_node_idx)
            target_class = torch.argmax(pred).item()
            self.fx = pred[target_class].item()
            del pred

        else: # get the prediction for the target class
            self.fx = self.forward_fn(self.model, node_features,
                                      sub_edge_index, new_node_idx
                                      )[target_class].item()
            
        (preds, pred_stats) = self.compute_model_predictions(node_features,
                                                     sub_edge_index, 
                                                     mask_matrix,
                                                     new_node_idx, batch_size,
                                                     target_class)
        pred_time = time.time() - start

        # revert back if add_self_loops are disabled
        if add_self_loops_swithced:
            switch_add_self_loops(self.model)

        
        log.info(f"Model predictions time(s):{pred_time:.4f}")

        # no need for self loops in the mask matrix from now on
        mask_matrix = mask_matrix[:, :nplayers]
        del node_features, sub_edge_index

        start = time.time()
        solver = get_solver(solver_name, mask_matrix, kernel_weights, preds,
                            self.fnull, self.fx, precision=self.precision,
                            device=device, rank=self.rank, **kwargs)
        del mask_matrix, kernel_weights, preds # cleanup some memory
        shap_vals, solver_stats = solver.solve()
        solve_time = time.time() - start

        log.info(f"Solve time(s):\t{solve_time:.4f}")

        # non-relabeled computional edge index
        sub_edge_index = self.data.edge_index[:, sub_edge_mask]

        total_time = time.time() - start_time
        explanation = GNNShapExplanation(node_idx, nplayers, float(self.fnull),
                                         shap_vals, nsamples,
                                         self.fx, target_class, sub_edge_index,
                                         subset, self.data.y[subset],
                                         time_total_comp=total_time,
                                         time_comp_graph=compgraph_time,
                                         time_sampling=sampling_time,
                                         time_predictions=pred_time,
                                         time_solver=solve_time,
                                         pred_stats=pred_stats,
                                         solver_stats=solver_stats,
                                         losses=None)

        return explanation

