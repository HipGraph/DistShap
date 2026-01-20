"""GNNShap explanation runner."""

import argparse
import pickle
import time

import torch
from tqdm.auto import tqdm

from dataset.utils import get_model_data_config
from gnnshap.explainer import GNNShapExplainer
from gnnshap.dist_explainer import DistShapExplainer
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Coauthor-CS')
    parser.add_argument('--result_path', type=str, default=None,
                        help=('Path to save the results. It will be saved in '
                              'the config results path if not provided.'))
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to use for GNNShap')
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--sampler', type=str, default='GNNShapSampler',
                        help='Sampler to use for sampling coalitions',
                        choices=['GNNShapSampler', 'SVXSampler', 'SHAPSampler',
                                'SHAPUniqueSampler'])
    parser.add_argument('--solver', type=str, default='WLSSolver',
                        help='Solver to use for solving SVX',
                        choices=['WLSSolver', 'WLRSolver', 'CGLSSolver'])
    parser.add_argument('--subset', type=str, default=None,
                        help='Subset of nodes to explain')
    # SVXSampler maximum size of coalitions to sample from
    parser.add_argument('--size_lim', type=int, default=3)

    args = parser.parse_args()

    dataset_name = args.dataset
    num_samples = args.num_samples
    batch_size = args.batch_size
    sampler_name = args.sampler
    solver_name = args.solver
    subset = args.subset


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data, config = get_model_data_config(dataset_name,
                                                load_pretrained=True,
                                                device=device,
                                                full_data=True,
                                                explain_subset=subset)

    test_nodes = config['test_nodes']

    result_path = args.result_path if args.result_path is not None \
        else config["results_path"]
    
    if subset is not None:
        result_path = os.path.join(result_path, subset)
    os.makedirs(result_path, exist_ok=True)

    if sampler_name == "SVXSampler":
        EXTRA_PARAM_SUFFIX = f"_{args.size_lim}"
    else:
        EXTRA_PARAM_SUFFIX = ""

    for r in range(args.repeat):
        results = []

        # original GNNShap code for replication
        # shap = GNNShapExplainer(model, data, nhops=config['num_hops'],
        #                         verbose=0, device=device,
        #                         progress_hide=True)

        # DistShap can be used for non-distributed settings as well.
        # Only tested with GNNShapSampler. For other samplers, use
        # GNNShapExplainer.
        shap = DistShapExplainer(model, data, nhops=config['num_hops'],
                                verbose=0, device=device,
                                progress_hide=True)

        start_time = time.time()

        failed_indices = []
        for ind in tqdm(test_nodes, desc=f"GNNShap explanations - run{r+1}"):
            try:

                # original GNNShap code for replication. symmetric sample 
                # location in the mask is different from the one in DistShap.
                # explanation = shap.explain(ind, nsamples=num_samples,
                #                            sampler_name=sampler_name,
                #                            batch_size=batch_size,
                #                            solver_name=solver_name,
                #                            size_lim=args.size_lim,
                #                            weight_scale=100,
                #                            nselfloops=0,
                #                            adjacent_sym=False)
                
                # DistShap explanation with a single GPU.
                explanation = shap.explain(ind, nsamples=num_samples,
                                           sampler_name=sampler_name,
                                           batch_size=batch_size,
                                           solver_name=solver_name,
                                           size_lim=args.size_lim)

                results.append(explanation.result2dict())
            except RuntimeError as e:
                failed_indices.append(ind)
                if 'out of memory' in str(e):
                    print(f"Node {ind} has failed: out of memory")
                else:
                    print(f"Node {ind} has failed: {e}")
            except Exception as e: # pylint: disable=broad-exception-caught
                print(f"Node {ind} has failed. General error: {e}")
                failed_indices.append([ind, shap.sampler.nplayers])

        rfile = (f'{result_path}/{dataset_name}_GNNShap_{sampler_name}_'
                 f'{solver_name}_{num_samples}_{batch_size}'
                 f'{EXTRA_PARAM_SUFFIX}_run{r+1}.pkl')
        with open(rfile, 'wb') as pkl_file:
            pickle.dump([results, 0], pkl_file)
        print(f"Total time: {time.time()-start_time}")
        if len(failed_indices) > 0:
            print(f"Failed indices: {failed_indices}")
        print(f"Results saved to {rfile}")