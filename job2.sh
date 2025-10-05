#!/bin/bash
#PBS -N job2
#PBS -l walltime=01:20:00
#PBS -l select=10:ncpus=36:mpiprocs=1:ompthreads=36:mem=250gb:place=pack
cd $PBS_O_WORKDIR
source /scratch/e0588224/miniconda3/etc/profile.d/conda.sh
python Lyapunov/train.py --export_data true --cpu true --reload_data '' --lyap_pure_polynomial true --env_base_seed -1  --num_workers 72 
conda activate myenv
# run command
Rscript script.R