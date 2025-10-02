#!/bin/bash
#PBS -N job1

#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:mem=250gb
cd $PBS_O_WORKDIR
# using R as an example non-MPI application
source ~/miniconda3/etc/profile.d/conda.sh
# run command
conda activate Lyapunov
python train.py --export_data true --cpu true --reload_data '' --env_base_seed -1  --num_workers 72


