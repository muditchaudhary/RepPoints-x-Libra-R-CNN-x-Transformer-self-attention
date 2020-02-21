#!/usr/bin/env bash
source ../../anaconda3/etc/profile.d/conda.sh
conda activate pytorch12 #specify conda environment
set -x

PARTITION=gpu_2h   #specify gpu time 2h, 8h, 24h
JOB_NAME=reppoints_moment_r50_fpn_2x_BLoss_CELoss_BSampler #job name can be anything   
CONFIG=./configs/my_configs/Libra/reppoints_moment_r50_fpn_2x_FLoss_1_initBSampler.py #specify the config file
WORK_DIR=./work_dirs/reppoints_moment_r50_fpn_2x_FLoss_1_initBSampler #where to save the models
GPUS=${5:-1}    #specify number of gpus ${5:-num_gpus}
GPUS_PER_NODE=${GPUS_PER_NODE:-1} #specify number of gpus per node ${:-num_gpu per node}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-"--validate"}
# NOTE: first stage train 12 epoches

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u ./mmdetection/tools/train.py ${CONFIG} --work_dir=${WORK_DIR} --launcher="slurm" 
    #${PY_ARGS}
