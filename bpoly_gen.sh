#!/bin/bash
#PBS -N BPoly
#PBS -l walltime=02:00:00
#PBS -l select=5:ncpus=36:mpiprocs=1:ompthreads=36:mem=250gb
cd $PBS_O_WORKDIR
source /scratch/e0588224/miniconda3/etc/profile.d/conda.sh
conda activate Lyapunov
# run command
python train.py --num_workers 180 --cpu true --export_data true  --reload_data '' --max_len 0 --max_output_len 0 --min_degree 2 --max_degree 5 --lyap_max_degree 6 --lyap_polynomial_H true --lyap_basic_functions_num true --lyap_pure_polynomial true --lyap_proba_proper_composition 0 --lyap_proba_proper_multiply 0 --lyap_proba_cross_composition 0 --lyap_proba_cross_multiply 0 --max_int 3 --lyap_find_domain false --lyap_SOS_checker true --lyap_SOS_fwd_gen false --lyap_proper_fwd false

