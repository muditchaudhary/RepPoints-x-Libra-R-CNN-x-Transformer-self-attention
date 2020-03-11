#!/usr/bin/env bash

source ../../anaconda3/etc/profile.d/conda.sh
conda activate pytorch12
set -x

PARTITION=gpu_2h
JOB_NAME=reppoints_test
CONFIG=./configs/my_configs/Empirical_attention/reppoints_moment_r50_fpn_2x_BFP_attn_0010_dcn.py
CHECKPOINT=./work_dirs/reppoints_moment_r50_fpn_2x_BFP_attn_0010_dcn/latest.pth
GPUS=${5:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
#PY_ARGS=${@:5}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u ./mmdetection/tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" --out "./work_dirs/reppoints_moment_r50_fpn_2x_BFP_attn_0010_dcn/results.pkl" --eval "bbox"
    #${PY_ARGS}
