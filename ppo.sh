#!/bin/bash
#PBS -N ppo_bpoly

#PBS -l walltime=60:00:00
#PBS -l select=1:ncpus=72:mpiprocs=1:ompthreads=72:mem=500gb:ngpus=2
cd $PBS_O_WORKDIR
source /scratch/e0588224/miniconda3/etc/profile.d/conda.sh
# run command
cd /home/svu/e0588224/Lyapunov
conda activate Lyapunov
python -m src.rl.rlvr.train \
  --exp_name rlvr_bpoly \
  --num_workers 0 \
  --dump_path /scratch/e0588224/data/reinforcement \
  --reload_model /scratch/e0588224/data/bpoly_model/debug/263838_stdct-mgmt-02/best-valid_ode_lyapunov_beam_acc.pth \
  --reload_data "ode_lyapunov,/scratch/e0588224/data/FLyap/flyap_mixture_data.prefix.cleaned" \
  --n_enc_layers 8 \
  --n_dec_layers 8 \
  --max_len 1024 \
  --lyap_pure_polynomial true \
  --lyap_polynomial_H true \
  --lyap_basic_functions_num true \
  --lyap_proba_proper_composition 0 \
  --lyap_proba_proper_multiply 0 \
  --lyap_proba_cross_composition 0 \
  --lyap_proba_cross_multiply 0 \
  --lyap_SOS_checker true \
  --lyap_find_domain false \
  --min_degree 2 \
  --lyap_max_degree 6 \
  --max_int 3 \
  --rl_lr 5e-5 \
  --rl_reward_scale 0.1 \
  --rl_use_kl_penalty 1 \
  --rl_kl_coef 0.05 \
  --rl_ent_coef 0.01

# use qsub -I -l select=2:ngpus=1 -l place=pack -l walltime=1:00:00 for interactive session

