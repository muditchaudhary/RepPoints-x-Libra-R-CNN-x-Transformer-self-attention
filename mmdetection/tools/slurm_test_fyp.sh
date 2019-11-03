#!/usr/bin/env bash

set -x

PARTITION=gpu_2h
JOB_NAME=reppoints_test
CONFIG=./configs/reppoints_moment_r50_fpn_2x_mt.py
CHECKPOINT=./reppoints_moment_r50_fpn_2x_mt.pth
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-2}
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
    python -u tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" --out "results.pkl" --eval "bbox"
    #${PY_ARGS}
