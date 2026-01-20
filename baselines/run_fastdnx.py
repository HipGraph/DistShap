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
from baselines.methods.fastdnx import FastDnXSurrogateModel, FastDnX
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='Cora', type=str)
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


    
num_epochs = 1000
for r in range(args.repeat):
    surrogate_model = FastDnXSurrogateModel(in_features=data.x.size(1),
                                            num_classes=data.y.max().item()+1,
                                            nhops=config['num_hops']).to(device)
    optim = torch.optim.Adam(surrogate_model.parameters(), lr=0.01)
    model.eval()
    with torch.no_grad():
        target = model(data.x, data.edge_index).log_softmax(dim=-1)


    train_start = time.time()
    # Train the explainer.
    for epoch in tqdm(range(num_epochs), desc="FastDnX Model Training"):
        surrogate_model.train()
        optim.zero_grad()
        out = surrogate_model(data.x, data.edge_index).log_softmax(dim=-1)
        loss = F.kl_div(out, target, reduction="batchmean", log_target=True)
        loss.backward()
        optim.step()

    surrogate_model.eval()
    explainer = FastDnX(surrogate_model, data.x, 'node', config['num_hops'], data.edge_index)
    explainer.prepare()

    train_time = time.time() - train_start

    results = []
    for ind in tqdm(test_nodes, desc=f"FastDnX explanations - run{r+1}"):
        start_time = time.time()
        
        nodes, node_importance = explainer.explain(ind)
        results.append(result2dict(ind, node_importance.cpu().numpy(), time.time() - start_time))

    rfile = f'{result_path}/{dataset_name}_FastDnX_run{r+1}.pkl'
    with open(rfile, 'wb') as pkl_file:
        pickle.dump([results, train_time], pkl_file)
    print(f"Results saved to {rfile}")
