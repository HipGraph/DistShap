import argparse
import pickle
import time
import torch
import os

from torch_geometric.explain import Explainer, PGExplainer
from tqdm.auto import tqdm

from dataset.utils import get_model_data_config
from gnnshap.utils import pruned_comp_graph
from baselines.utils import result2dict, get_explainer_train_nodes
from torch_geometric.utils import k_hop_subgraph

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='Coauthor-CS', type=str)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--repeat', default=1, type=int)
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

train_nodes = data.train_mask.nonzero(as_tuple=False
                                      ).cpu().numpy().flatten().tolist()
test_nodes = config['test_nodes']
    
    
num_epochs = 50
for r in range(args.repeat):
    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=num_epochs, lr=0.003, device=device),
        explanation_type='phenomenon', # it only supports this. no model option
        model_config=dict(
            mode='multiclass_classification',
            task_level='node', #node level prediction.
            return_type='raw',),
        node_mask_type=None,# no node mask
        edge_mask_type='object', # only edge mask
        threshold_config=None,)
    
    explainer.algorithm.mlp = explainer.algorithm.mlp.to(device)


    train_start = time.time()
    # Train the explainer.
    for epoch in tqdm(range(num_epochs), desc="PGExplainer Model Training", disable=True):

        

        target = torch.argmax(model(data.x, data.edge_index), dim=-1)
        for index in tqdm(train_nodes, leave=False):
                loss = explainer.algorithm.train(epoch, model, data.x,
                                                data.edge_index,
                                                target=target,
                                                index=int(index))
    train_time = time.time() - train_start

    results = []
    for ind in tqdm(test_nodes, desc=f"PGExplainer explanations - run{r+1}"):
        start_time = time.time()
        
        (subset, sub_edge_index, sub_mapping,
         sub_edge_mask) = k_hop_subgraph(ind, config['num_hops'],
                                            data.edge_index,
                                            relabel_nodes=True)

        target = torch.argmax(model(data.x[subset], sub_edge_index), dim=-1)
        explanation = explainer(data.x[subset], sub_edge_index,
                                index=sub_mapping.item(),
                                edge_weight=data.edge_weight,
                                target=target)
        
        # save in our format: pruned edges
        (_, _, mapping2, mask2) = pruned_comp_graph(sub_mapping, config['num_hops'],
                                                    sub_edge_index, relabel_nodes=False)
        edge_importance = explanation.edge_mask[mask2].detach().cpu().numpy()
        results.append(result2dict(ind, edge_importance, time.time() - start_time))

    rfile = f'{result_path}/{dataset_name}_PGExplainer_run{r+1}.pkl'
    with open(rfile, 'wb') as pkl_file:
        pickle.dump([results, train_time], pkl_file)
    print(f"Results saved to {rfile}")
