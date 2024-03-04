#!/bin/bash

#SBATCH --job-name=learn_pde     # Specify job name
#SBATCH --partition=gpu            # Specify partition name
#SBATCH --nodes=1                  # Specify number of nodes
#SBATCH --gpus=4                   # Specify number of GPUs needed for the job
#SBATCH --exclusive                # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
#SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=00:30:00            # Set a limit on the total run time
#SBATCH --mail-type=FAIL           # Notify user by email in case of job failure
#SBATCH --account=gg0028         # Charge resources on this project account
#SBATCH --output=my_job.o%j        # File name for standard output

# module load compilers/cuda/11.0
export CUDA_VISIBLE_DEVICES="0, 1, 2, 3"
srun /home/g/g260202/miniconda3/envs/gpu-hackathon/bin/python supervisor_trian_2nodes_save_model_new_loss.py