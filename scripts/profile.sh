#!/bin/bash

set -e

# Set a single cuda device to run the benchmarks
export CUDA_VISIBLE_DEVICES=0
echo "Running benchmarks on GPU: $CUDA_VISIBLE_DEVICES"

export GPU_MODE=$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=compute_mode --format=csv | awk 'NR==2')

# Collect single-job performance with and without Tally
# This should give insights of the overhead of API forwarding in Tally
echo "======== Collecting single-job performance with and without Tally ... ========"
python3 -u scripts/run_bench.py \
    --save-results \
    --use-tally-naive \
    --runtime 30 \
    --warmup-iters 30

# Profile preemptive kernel metrics for throughput-oriented jobs for priority scheduler
echo "======== Profiling kernel metrics for throughput-oriented jobs for priority scheduler ... ========"
python3 -u scripts/run_bench.py \
    --use-tally-priority \
    --runtime 60 \
    --warmup-iters 1000 \
    --profile-only \
    --save-results