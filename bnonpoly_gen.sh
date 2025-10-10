#!/bin/bash
#PBS -N BNonPoly
#PBS -l walltime=07:00:00
#PBS -l select=5:ncpus=36:mpiprocs=1:ompthreads=36:mem=250gb
cd $PBS_O_WORKDIR
source /scratch/e0588224/miniconda3/etc/profile.d/conda.sh
conda activate Lyapunov
# run command
python train.py --num_workers 180 \
    --cpu true \
    --export_data true  \
    --reload_data '' \
    --max_len 0 \
    --max_output_len 0 \
    --min_degree 2 \
    --max_degree 3 \
    --lyap_max_degree 4 \
    --lyap_polynomial_H true \
    --lyap_basic_functions_num false \
    --lyap_pure_polynomial false \
    --lyap_proba_proper_composition 0.2 \
    --lyap_proba_proper_multiply 0.2 \
    --lyap_proba_cross_composition 0.2 \
    --lyap_proba_cross_multiply 0 \
    --max_int 3 \
    --lyap_find_domain true \
    --lyap_SOS_checker false \
    --lyap_SOS_fwd_gen false \
    --lyap_proper_fwd false

