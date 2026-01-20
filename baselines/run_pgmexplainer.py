import torch
from baselines.methods.pgm_explainer import PGM_Node_Explainer
import numpy as np
import time
from tqdm.auto import tqdm
from baselines.utils import result2dict
import pickle
from gnnshap.utils import pruned_comp_graph
from torch_geometric.utils import k_hop_subgraph
from dataset.utils import get_model_data_config
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='Cora', type=str)
parser.add_argument('--repeat', default=1, type=int)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--subset', type=str, default=None)
parser.add_argument('--result_path', type=str, default='./results',
                    help=('Path to save the results.'))

args = parser.parse_args()
device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

dataset_name = args.dataset

result_path = args.result_path
if args.subset is not None:
    result_path = os.path.join(args.result_path, args.subset)
os.makedirs(result_path, exist_ok=True)


model, data, config = get_model_data_config(args.dataset, load_pretrained=True,
                                            device=device,
                                            log_softmax_return=True,
                                            full_data=True,
                                            explain_subset=args.subset)


model.eval()

# use the predictions as the target
# target = torch.argmax(model(data.x, data.edge_index), dim=-1)
test_nodes = config['test_nodes']

for r in range(args.repeat):
    results = []
    for ind in tqdm(test_nodes, desc=f"PGMExp explanations - run{r+1}"):
        start_time = time.time()

        # out of memory on large datasets. limit with k-hop subgraph
        (subset, sub_edge_index, sub_mapping,
         sub_edge_mask) = k_hop_subgraph(ind, config['num_hops'],
                                            data.edge_index,
                                            relabel_nodes=True)
        
        target = torch.argmax(model(data.x[subset], sub_edge_index),
                              dim=-1)[sub_mapping.item()]
        pgm_explainer = PGM_Node_Explainer(model, sub_edge_index,
                                           None, data.x[subset],
                                           num_layers=config['num_hops'],
                                           device=device, mode=0,
                                           print_result=1)
        explanation = pgm_explainer.explain(sub_mapping.item(),
                                            target=target, num_samples=100,
                                            top_node=None)
        
        results.append(result2dict(ind, np.array(explanation),
                                   time.time() - start_time))
    rfile = f'{result_path}/{dataset_name}_PGMExplainer_run{r+1}.pkl'
    with open(rfile, 'wb') as pkl_file:
        pickle.dump([results, 0], pkl_file)
    print(f"Results saved to {rfile}")
