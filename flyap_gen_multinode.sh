#!/bin/bash
#PBS -P CFP01-SF-009
#PBS -N FLyapArray
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:mem=250gb
# Submit with: qsub -J 0-5 flyap_gen_multinode.sh   (adjust range as needed)

set -euo pipefail

cd "$PBS_O_WORKDIR"
source /scratch/e0588224/miniconda3/etc/profile.d/conda.sh
conda activate Lyapunov

# ---------------- user-tunable knobs ----------------
: "${FLYAP_NUM_WORKERS:=32}"   # DataLoader workers per array task
: "${FLYAP_OUTPUT_ROOT:=/scratch/e0588224/data/FLyap_array}"  # where each task dumps results
: "${FLYAP_EXTRA_ARGS:=}"      # optional extra CLI flags appended to train.py

# ---------------- array metadata -------------------
IDX=${PBS_ARRAY_INDEX:-0}
JOB_TAG=$(echo "${PBS_JOBID:-manual}" | tr -c '[:alnum:]_' '_')
EXP_ID="flyap_${JOB_TAG}_idx${IDX}"
DUMP_PATH="${FLYAP_OUTPUT_ROOT}/idx_${IDX}"

mkdir -p "$DUMP_PATH"
echo "[FLyap array] Task index ${IDX}"
echo "[FLyap array] dump_path=${DUMP_PATH}"
echo "[FLyap array] exp_id=${EXP_ID}"

python train.py \
    --dump_path "${DUMP_PATH}" \
    --exp_id "${EXP_ID}" \
    --num_workers "${FLYAP_NUM_WORKERS}" \
    --cpu true \
    --export_data true \
    --reload_data "" \
    --max_len 0 \
    --max_output_len 0 \
    --min_degree 2 \
    --max_degree 3 \
    --lyap_max_degree 4 \
    --lyap_polynomial_H true \
    --lyap_basic_functions_num true \
    --lyap_pure_polynomial true \
    --lyap_proba_proper_composition 0 \
    --lyap_proba_proper_multiply 0 \
    --lyap_proba_cross_composition 0 \
    --lyap_proba_cross_multiply 0 \
    --max_int 5 \
    --lyap_gen_weight 0.3 \
    --lyap_find_domain false \
    --lyap_SOS_checker true \
    --lyap_SOS_fwd_gen true \
    --lyap_proper_fwd true \
    ${FLYAP_EXTRA_ARGS}

echo "[FLyap array] Task ${IDX} finished. Dataset stored under ${DUMP_PATH}/FLyap_Gen/${EXP_ID}"
