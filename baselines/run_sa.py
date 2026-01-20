import argparse
import pickle
import time

import torch
from captum.attr import Saliency
from tqdm.auto import tqdm

from baselines.utils import result2dict
from dataset.utils import get_model_data_config
from gnnshap.utils import pruned_comp_graph
import os

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='Cora', type=str)
parser.add_argument('--repeat', default=1, type=int)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--subset', type=str, default=None)
parser.add_argument('--result_path', type=str, default='./results',
                    help=('Path to save the results.'))

args = parser.parse_args()

dataset_name = args.dataset

result_path = args.result_path
if args.subset is not None:
    result_path = os.path.join(args.result_path, args.subset)
os.makedirs(result_path, exist_ok=True)

device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')


model, data, config = get_model_data_config(args.dataset, load_pretrained=True,
                                            device=device, full_data=True,
                                            explain_subset=args.subset)


model.eval()
test_nodes = config['test_nodes']

def model_forward_node(x, model, edge_index, node_idx):
    out = model(x, edge_index).softmax(dim=-1)
    return out[[node_idx]]

for r in range(args.repeat):
    results = []
    for i, ind in tqdm(enumerate(test_nodes),
                       desc=f"SA explanations - run{r+1}"):
        start_time = time.time()
        model.eval()
        (subset, sub_edge_index, sub_mapping,
         sub_edge_mask) = pruned_comp_graph(ind, config['num_hops'],
                                            data.edge_index,
                                            relabel_nodes=True)
        target = model(data.x[subset].to(device),
                       sub_edge_index).argmax(dim=-1)[sub_mapping.item()]

        explainer = Saliency(model_forward_node)


        x_mask = data.x[subset].clone().requires_grad_(True).to(device)
        saliency_mask = explainer.attribute(
            x_mask, target=target,
            additional_forward_args=(model, sub_edge_index,
                                     sub_mapping.item()), abs=False)
        
        node_importance = saliency_mask.cpu().numpy().sum(axis=1)
        results.append(result2dict(ind, node_importance,
                                   time.time() - start_time))
    
    
    rfile = f'{result_path}/{args.dataset}_SA_run{r+1}.pkl'
    with open(rfile, 'wb') as pkl_file:
        pickle.dump([results, 0], pkl_file)
    print(f"Results saved to {rfile}")
