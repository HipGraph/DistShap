import argparse
import os
import pickle
import torch
import torch.distributed as dist
from dataset.utils import get_model_data_config
from gnnshap.dist_explainer import DistShapExplainer
from time import time
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")



def run_distributed(dataset_name, num_samples, batch_size, result_path,
              subset, repeat=1, show_progress=False):

    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE",
                    "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    }

    local_rank = int(env_dict["LOCAL_RANK"])

    world_size = int(env_dict["WORLD_SIZE"])
    rank = int(env_dict["RANK"])

    if rank == 0:
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}") 

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")

    # First 100 test nodes if subset is None.
    # don't load the full data to device save memory
    model, data, config = get_model_data_config(dataset_name,
                                                load_pretrained=True,
                                                device='cpu',
                                                full_data=True,
                                                explain_subset=subset)
    model = model.to(device)
    # print(f"rank {rank}: model loaded on device {device}")
    explain_indices = config['test_nodes']
    dist.barrier()

    results = []
    show_progress = True if rank == 0 and show_progress else False
    
    # print("Starting explanation")
    for r in range(repeat):
        start_time = time()


        rfile = (f"{result_path}/{dataset_name}_DistShap_nsamp{num_samples}_"
                 f"world_{world_size}_run{r+1}.pkl")
        for explain_node_idx in tqdm(explain_indices,
                                     disable=not show_progress):
            try:
                shap = DistShapExplainer(model, data,
                                            nhops=config['num_layers'],
                                            device=device,
                                            world_size=world_size, rank=rank,
                                            verbose=0,
                                            precision=torch.float32,
                                            progress_hide=True)
                explanation = shap.explain(explain_node_idx,
                                           nsamples=num_samples,
                                           solver_name='DistCGLSSolver',
                                           batch_size=batch_size)
            except Exception as e:
                print(f"Rank {rank} failed to explain node {explain_node_idx}")
                print(e)
                continue
            dist.barrier()
            if rank == 0:
                results.append(explanation.result2dict())
            del explanation
            dist.barrier()
        
        if rank == 0:
            with open(rfile, 'wb') as pkl_file:
                pickle.dump([results, 0], pkl_file)
            print(f"Results saved to {rfile}")
            print(f"Time taken: {time() - start_time}")
        dist.barrier()
        
    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":

    # these can be used to test on single node & single GPU.
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '45550'
    # os.environ['RANK'] = '0'
    # os.environ['WORLD_SIZE'] = '1'
    # os.environ['LOCAL_RANK'] = '0'
    # os.environ['LOCAL_WORLD_SIZE'] = '1'

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Coauthor-CS',
                        help='Name of the dataset')
    parser.add_argument('--result_path', type=str, default='./results/',
                        help=('Root path of the results'))
    parser.add_argument('--num_samples', type=int, default=60000,
                        help='Number of samples to use for DistShap')
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--subset', type=str, default='dist',
                        help='Subset of nodes to explain')
    parser.add_argument('--show_progress', action='store_true',
                        help='Show progress bar')
    args = parser.parse_args()

    dataset_name = args.dataset
    num_samples = args.num_samples
    batch_size = args.batch_size
    subset = args.subset
    repeat = args.repeat
    show_progress = args.show_progress

    if subset is not None:
        result_path = os.path.join(args.result_path, subset)
        os.makedirs(result_path, exist_ok=True)

    run_distributed(dataset_name, num_samples, batch_size, result_path, subset,
                    repeat, show_progress)
