#!/bin/bash

NNODES=$SLURM_NNODES
WORLD_SIZE=$(($NNODES * 4))

# Set environment variables
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=45550

export OMP_NUM_THREADS=32

export NODE_RANK=$SLURM_NODEID

dataset="ogbn-arxiv"
num_samples=600000

batch_size=200

echo torchrun --nnodes=$NNODES --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --node_rank=$NODE_RANK \
    run_distgnnshap.py --dataset $dataset --batch_size $batch_size

torchrun --nnodes=$NNODES --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --node_rank=$NODE_RANK \
    run_distgnnshap.py --dataset $dataset --batch_size $batch_size \
    --num_samples $num_samples
