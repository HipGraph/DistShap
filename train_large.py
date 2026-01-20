# Description: This file is used to train GNN models on large datasets. The
# trained model is saved in the pretrained folder. The model is trained using
# NeighborLoader, which is not supported in train.py.

# The trained model is used for benchmarking explanation methods.


import argparse
import copy
import os
import sys

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from dataset.utils import get_model_data_config
from gnnshap.utils import pruned_comp_graph

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ogbn-products', type=str)
args = parser.parse_args()
dataset_name = args.dataset

def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()

    return total_loss / total_examples, total_correct / total_examples

@torch.no_grad()
def test(loader: NeighborLoader) -> float:
    model.eval()

    correct = 0
    for batch in loader:
        out = model(batch.x.to(device), batch.edge_index.to(device))
        pred = out.argmax(dim=1)
        correct += (pred[:batch.batch_size] == batch.y[:batch.batch_size].to(device)).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # don't load data to GPU, as it will be loaded to GPU during sampling.
    model, data, config = get_model_data_config(dataset_name,
                                                load_pretrained=False,
                                                device='cpu',
                                                full_data=True)

    model = model.to(device)


    pretrained_file = (f"{config['root_path']}/pretrained/{dataset_name}_"
                       "pretrained.pt")

    if os.path.exists(pretrained_file):
        user_input = input('A pretrained file exist. '
                           'Do you want to retrain? (y/n):')
        if user_input.lower() != 'y':
            print("Skipping training!")
            sys.exit(0)


    # Already send node features/labels to GPU for faster access during sampling:
    data = data.to(device, 'x', 'y')
    neig_args = config['nei_sampler_args']
    kwargs = {'batch_size': neig_args['batch_size'], 'num_workers': 6,
              'persistent_workers': True}
    train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                                  num_neighbors=neig_args['sizes'],
                                  shuffle=True, **kwargs)

    val_loader = NeighborLoader(data, input_nodes=data.val_mask,
                                num_neighbors=neig_args['sizes'], **kwargs)
    test_loader = NeighborLoader(data, input_nodes=data.test_mask,
                                 num_neighbors=neig_args['sizes'], **kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val = 0
    best_test = 0

    for epoch in range(1, config['epoch'] + 1):
        loss, train_acc = train(epoch)
        # print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        val_acc = test(val_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f},'
              f' Test: {test_acc:.4f}')
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            torch.save(model.state_dict(), pretrained_file)
    print(f'Best Val: {best_val:.4f}, Test: {best_test:.4f}')
    print(f"Model saved to {pretrained_file}.")



    # Sample explain data and save. This is used for benchmarking explanation
    # methods. This makes sure that the explain data is the same for all
    # methods.

    num_test_nodes = 100
    test_nodes = data.test_mask.nonzero()[:num_test_nodes, 0]
    explain_loader = NeighborLoader(data,
                                    input_nodes=test_nodes,
                                    num_neighbors=neig_args['sizes'],
                                    batch_size=num_test_nodes,
                                    num_workers=24, persistent_workers=True)

    max_size = 0
    max_ind = 0
    avg_size = 0
    batch = next(iter(explain_loader))
    for i in range(batch.batch_size):
        m = pruned_comp_graph(i, 2, batch.edge_index)[1].size(1)
        if m > max_size:
            max_size = m
            max_ind = i
        avg_size += m
    # reduce saved file size in disk. Can be reloaded from the original data.
    del batch.x, batch.y 

    torch.save(batch, f"{config['root_path']}/pretrained/{dataset_name}"
               "_explain_data.pt")
    print(f"Explain data saved to {config['root_path']}/pretrained.")
    print("Maximum size: ", max_size, "max index: ", max_ind, "avg size: ",
          avg_size / num_test_nodes)