import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

def result2dict(node_id: int, scores: np.array, comp_time: float) -> dict:
    """Converts an explanation result to a dictionary

    Args:
        node_id (int): node id
        scores (np.array): importance scores
        comp_time (float): computation time

    Returns:
        dict: result as dictionary
    """
    return {'node_id': node_id, 'scores': scores, 'num_players': len(scores),
            'time': comp_time}



def get_explainer_train_nodes(data: torch_geometric.data,
                              nhops: int, maxnodes:int,
                              nsamples: int) -> list[int]:
    """Gets training nodes for the explainer. If the number of training nodes is
    less than the required number of samples, it returns all training nodes.
    Otherwise, it samples the similar number of random nodes for each class.
    After balanced sampling, if the number of nodes is less than the required
    number of samples, it randomly samples the remaining nodes.

    Args:
        data (torch_geometric.data): PyG Data object
        nsamples (int): number of samples

    Returns:
        list[int]: list of training nodes
    """


    num_train_nodes = data.train_mask.sum().item()
    if num_train_nodes <= nsamples:
        return data.train_mask.nonzero(as_tuple=False
                                       ).cpu().numpy().flatten().tolist()


    num_classes = data.y.max().item() + 1
    per_class_cnt = nsamples // num_classes
    selected_nodes = []
    for c in range(num_classes):
        nodes = torch.nonzero(data.y[data.train_mask] == c).flatten()
        # shuffle the nodes
        nodes = nodes[torch.randperm(len(nodes))]
        # sample the nodes, num_nodes should be less than 30k,
        # otherwise it will run out of memory.
        cnt = 0
        for n in nodes:
            neighbors = k_hop_subgraph(
                n.item(), nhops, data.edge_index, relabel_nodes=True,
                num_nodes=data.num_nodes)[0]
            if len(neighbors) > maxnodes:
                # if the number of neighbors is more than 30k, skip this node
                continue
            else:
                # sample the nodes
                selected_nodes.append(n.item())
                cnt += 1
            if cnt == per_class_cnt:
                break


    if len(selected_nodes) < nsamples:
        remaining_nodes = torch.cat([torch.nonzero(data.y[data.train_mask] == c
                           ).flatten() for c in range(num_classes)])
        
        remaining_nodes = remaining_nodes[~torch.isin(
            remaining_nodes, torch.tensor(selected_nodes, device=remaining_nodes.device))]

        # shuffle the remaining nodes
        remaining_nodes = remaining_nodes[torch.randperm(len(remaining_nodes))]
        # sample the remaining nodes
        # num_samples should be less than 30k, otherwise it will run out of memory.

        rem_cnt = nsamples - len(selected_nodes)

        for n in remaining_nodes:
            neighbors = k_hop_subgraph(
                n.item(), nhops, data.edge_index, relabel_nodes=True,
                num_nodes=data.num_nodes)[0]
            if len(neighbors) > maxnodes:
                # if the number of neighbors is more than 30k, skip this node
                continue
            else:
                # sample the nodes
                selected_nodes.append(n.item())
                rem_cnt -= 1
            if rem_cnt == 0:
                break
    return selected_nodes