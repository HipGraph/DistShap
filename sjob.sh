#!/bin/bash
#SBATCH --job-name=explain
#SBATCH --qos=regular
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=120
#SBATCH --output=explain32-%j.out
#SBATCH -A account_name
#SBATCH --gres=gpu:4
#SBATCH -C gpu&hbm80g


srun ./torchrun.sh
