#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=learning_PDE
#SBATCH --nodes=2        #unfortunately 3 is the max on strand at the moment.
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --account=hai_ml_pde
#SBATCH --partition=booster
#SBATCH --exclusive

# module load compilers/cuda/11.0
export CUDA_VISIBLE_DEVICES="0, 1, 2, 3"
srun /p/project/hai_ml_pde/miniconda3/envs/gpu-hackathon/bin/python supervisor_trian_2nodes_save_model_new_loss.py