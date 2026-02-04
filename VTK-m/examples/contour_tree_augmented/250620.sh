#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=concat_freud3d

#SBATCH --time=15:50:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=2

./ContourTree_Augmented --vtkm-device=openmp --printCT --vtkm-log-level=INFO ../Data/out_hh_256_coarsened_binary.bin
