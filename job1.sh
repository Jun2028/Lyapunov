#!/bin/bash
#PBS -N job1

#PBS -l walltime=50:00:00
#PBS -l select=1:ncpus=72:mpiprocs=1:ompthreads=72:mem=500gb:ngpus=2
cd $PBS_O_WORKDIR
source /scratch/e0588224/miniconda3/etc/profile.d/conda.sh
# run command
conda activate Lyapunov
python /home/svu/e0588224/Lyapunov/train.py --dump_path /scratch/e0588224/data/model1 --batch_size 6 --batch_size_eval 24 --beam_eval true --beam_size 50 --optimizer adam_inverse_sqrt,warmup_updates=10000,lr=0.0001 --epoch_size 1100000 --num_workers 1 --eval_size 200 --lyap_pure_polynomial true --lyap_SOS_checker true --stopping_criterion "valid_ode_lyapunov_beam_acc,100" --validation_metrics "valid_ode_lyapunov_beam_acc" --reload_data "ode_lyapunov,/scratch/e0588224/data/debug/data.prefix.100.data.prefix.100.cleaned.train,/scratch/e0588224/data/debug/data.prefix.100.data.prefix.100.cleaned.valid.final,/home/svu/e0588224/Lyapunov/benchmarks/BPoly,/home/svu/e0588224/Lyapunov/benchmarks/FBarr,/home/svu/e0588224/Lyapunov/benchmarks/FLyap,/home/svu/e0588224/Lyapunov/benchmarks/FSOSTOOL" 

# qsub -I -l select=2 -l place=pack -l walltime=03:00:00
# python Lyapunov/train.py --export_data true --cpu true --reload_data '' --env_base_seed -1  --num_workers 72 
# python /home/svu/e0588224/Lyapunov/create_dataset.py