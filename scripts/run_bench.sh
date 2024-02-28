#!/bin/bash

# This script serves as an automated script to launch all benchmarks

# Utility function to measure benchmark duration
time_cmd() {

    # Record the start time
    start_time=$(date +%s.%N)

    # Execute the provided commands
    "$@"

    # Record the end time
    end_time=$(date +%s.%N)

    # Calculate the elapsed time
    elapsed_time=$(echo "$end_time - $start_time" | bc)

    echo "Time elapsed: $elapsed_time seconds."

}

set -e

export RUNTIME=30
export WARMUP_ITERS=30

# Set a single cuda device to run the benchmarks
export CUDA_VISIBLE_DEVICES=0
echo "Running benchmarks on GPU: $CUDA_VISIBLE_DEVICES"

export GPU_MODE=$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=compute_mode --format=csv | awk 'NR==2')

# Collect single-job performance with and without Tally
# This should give insights of the overhead of API forwarding in Tally
echo "======== Collecting single-job performance with and without Tally ... ========"
time_cmd \
    python3 -u scripts/run_bench.py \
        --save-results \
        --use-tally-naive \
        --runtime $RUNTIME \
        --warmup-iters $WARMUP_ITERS

# Profile preemptive kernel metrics for throughput-oriented jobs for priority scheduler
echo "======== Profiling kernel metrics for throughput-oriented jobs for priority scheduler ... ========"
SCHEDULER_POLICY=PRIORITY \
    time_cmd \
    python3 -u scripts/run_bench.py \
        --use-tally-priority \
        --runtime 60 \
        --warmup-iters 100 \
        --profile-only \
        --save-results

# Run priority-related co-located experiments with Tally priority scheduler
if [[ $GPU_MODE == "Exclusive_Process" ]]; then
echo "======== Collecting priority-related pair-wise performance with Tally priority scheduler ... ========"
SCHEDULER_POLICY=PRIORITY \
    time_cmd \
    python3 -u scripts/run_bench.py \
        --save-results \
        --use-tally-priority \
        --runtime $RUNTIME \
        --warmup-iters $WARMUP_ITERS \
        --run-pairwise
else
    echo "Skip collecting pair-wise performance with Tally priority scheduler because GPU_MODE is not Exclusive_Process"
fi

# Run priority-related co-located experiments with MPS
if [[ $GPU_MODE == "Exclusive_Process" ]]; then
echo "======== Collecting priority-related pair-wise performance with MPS ... ========"
time_cmd \
    python3 -u scripts/run_bench.py \
        --save-results \
        --use-mps \
        --runtime $RUNTIME \
        --warmup-iters $WARMUP_ITERS \
        --run-pairwise
else
    echo "Skip collecting pair-wise performance with MPS because GPU_MODE is not EXCLUSIVE"
fi