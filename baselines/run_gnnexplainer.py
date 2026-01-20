import argparse
import pickle
import time

import torch
import os
from torch_geometric.explain import CaptumExplainer, Explainer, GNNExplainer
from tqdm.auto import tqdm

from baselines.utils import result2dict
from dataset.utils import get_model_data_config
from gnnshap.utils import pruned_comp_graph
from torch_geometric.utils import k_hop_subgraph

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='Cora', type=str)
parser.add_argument('--repeat', default=1, type=int)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--subset', type=str, default=None)
parser.add_argument('--result_path', type=str, default='./results',
                    help=('Path to save the results.'))

args = parser.parse_args()

device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

result_path = args.result_path
if args.subset is not None:
    result_path = os.path.join(args.result_path, args.subset)
os.makedirs(result_path, exist_ok=True)

model, data, config = get_model_data_config(args.dataset, load_pretrained=True,
                                            device='cpu', full_data=True,
                                            explain_subset=args.subset)
data.edge_index = data.edge_index.to(device)
model = model.to(device)
model.eval()

algorithm = GNNExplainer(epochs=100)


test_nodes = config['test_nodes']

for r in range(args.repeat):
    explainer = Explainer(
    model=model,
    algorithm=algorithm,
    explanation_type='model',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    ),
    node_mask_type=None,# no node mask
    edge_mask_type='object', # only edge mask
    threshold_config=None,
    )
    results = []
    
    for ind in tqdm(test_nodes,
                    desc=f"GNNExplainer explanations - run{r+1}"):
        try:
            start_time = time.time()

            (subset, sub_edge_index, sub_mapping,
             sub_edge_mask) = k_hop_subgraph(ind, config['num_hops'],
                                                data.edge_index,
                                                relabel_nodes=True)
            x = data.x[subset.cpu()].to(device)
            # target = torch.argmax(model(x, sub_edge_index), dim=-1).to(device)
            # skip nodes with less than 2 edges
            if sub_edge_index.size(1) < 2:
                print(f"Skipping node {ind} with {sub_edge_index.size(1)} edges")
                continue
            target=None
            explanation = explainer(x, sub_edge_index,
                                target=target,
                                index=sub_mapping.item())

            # save in our format: pruned edges
            (_, _, mapping2, mask2) = pruned_comp_graph(sub_mapping, config['num_hops'],
                                                        sub_edge_index, relabel_nodes=False)
            edge_importance = explanation.edge_mask[mask2].detach().cpu().numpy()
            
            results.append(result2dict(ind, edge_importance, time.time() - start_time))

            del explanation, x, sub_edge_index, mapping2, mask2, subset, sub_mapping, sub_edge_mask

        except Exception as e:
            print(f"Node {ind} failed!, {e}")
    rfile = (f"{result_path}/{args.dataset}_GNNExplainer_run{r+1}.pkl")
    with open(rfile, 'wb') as pkl_file:
        pickle.dump([results, 0], pkl_file)
    print(f"Results saved to {rfile}")