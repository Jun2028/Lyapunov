#!/bin/bash
#PBS -N eval_bnonpoly_best

#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=72:mpiprocs=1:ompthreads=72:mem=500gb:ngpus=2
cd $PBS_O_WORKDIR
source /scratch/e0588224/miniconda3/etc/profile.d/conda.sh
# run command
conda activate Lyapunov
python /home/svu/e0588224/Lyapunov/train.py \
    --eval_only true \
    --eval_size 200 \
    --reload_model /scratch/e0588224/data/bnonpoly_model/debug/263943_stdct-mgmt-02/best-valid_ode_lyapunov_beam_acc.pth \
    --reload_data "ode_lyapunov,/scratch/e0588224/data/debug/bnonpoly/data.prefix.100.data.prefix.100.cleaned.train,/scratch/e0588224/data/debug/bnonpoly/data.prefix.100.data.prefix.100.cleaned.test.final,/home/svu/e0588224/Lyapunov/benchmarks/FLyap" \
    --eval_verbose 1 \
    --n_enc_layers 8 \
    --n_dec_layers 8 \
    --max_len 1024 \
    --batch_size_eval 10 \
    --beam_eval true \
    --beam_size 50 \
    --num_workers 1 \
    --lyap_SOS_checker true \
    --lyap_pure_polynomial true \
    --validation_metrics "valid_ode_lyapunov_beam_acc"

# qsub -I -l select=2 -l place=pack -l walltime=03:00:00
# qsub -I -l select=2:ngpus=1 -l place=pack -l walltime=12:00:00
# python Lyapunov/train.py --export_data true --cpu true --reload_data '' --env_base_seed -1  --num_workers 72 
# python /home/svu/e0588224/Lyapunov/create_dataset.py
#   --lyap_SOS_checker false \
#   --lyap_find_domain true \
#   --lyap_polynomial_H false \
#   --validation_metrics "valid_ode_lyapunov_beam_acc"