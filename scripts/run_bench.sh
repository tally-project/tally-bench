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

export RUNTIME=5
export WARMUP_ITERS=10

export GPU_MODE=$(nvidia-smi --query-gpu=compute_mode --format=csv | awk 'NR==2')

# 1. Collect single-job performance with and without Tally
#    This should give insights of the overhead of API forwarding in Tally
if [[ $GPU_MODE == "Default" ]]; then
echo "======== Collecting single-job performance with and without Tally ... ========"
time_cmd \
    python3 -u scripts/run_bench.py \
        --save-results \
        --use-tally \
        --runtime $RUNTIME \
        --warmup-iters $WARMUP_ITERS
else
    echo "Skip collecting single-job performance with and without Tally because GPU_MODE is not DEFAULT"
fi

# 2. Profile kernel metrics for workload-agnostic and priority scheduler
#    Run each job for a long period of time to make sure each kernel is profiled
#    Results won't be collected
if [[ $GPU_MODE == "Default" ]]; then
echo "======== Profiling kernel metrics for workload-agnostic and priority scheduler ... ========"

for policy in WORKLOAD_AGNOSTIC_SHARING PRIORITY
do
    SCHEDULER_POLICY=$policy \
        time_cmd \
        python3 -u scripts/run_bench.py \
            --use-tally \
            --runtime 10 \
            --warmup-iters 1000 \
            --profile-only  \
            --save-results

    SCHEDULER_POLICY=$policy \
        time_cmd \
        python3 -u scripts/run_bench.py \
            --save-results \
            --use-tally \
            --runtime $RUNTIME \
            --warmup-iters $WARMUP_ITERS
done

else
    echo "Skip profiling kernel metrics for workload-agnostic scheduler because GPU_MODE is not DEFAULT"
fi

# 3. Run co-located experiments without MPS nor Tally (Hardware multi-processing)
if [[ $GPU_MODE == "Default" ]]; then
echo "======== Collecting pair-wise performance with and hardware multi-processing ... ========"
time_cmd \
    python3 -u scripts/run_bench.py \
        --save-results \
        --runtime $RUNTIME \
        --warmup-iters $WARMUP_ITERS \
        --run-pairwise
else
    echo "Skip collecting pair-wise performance with and hardware multi-processing because GPU_MODE is not DEFAULT"
fi

# 4. Run co-located experiments with MPS
if [[ $GPU_MODE == "Exclusive_Process" ]]; then
echo "======== Collecting pair-wise performance with MPS ... ========"
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

# 5. Run co-located experiments with Tally workload agnostic
if [[ $GPU_MODE == "Default" ]]; then
echo "======== Collecting pair-wise performance with Tally workload agnostic ... ========"
SCHEDULER_POLICY=WORKLOAD_AGNOSTIC_SHARING \
    time_cmd \
    python3 -u scripts/run_bench.py \
        --save-results \
        --use-tally \
        --runtime $RUNTIME \
        --warmup-iters $WARMUP_ITERS \
        --run-pairwise
else
    echo "Skip collecting pair-wise performance with Tally workload agnostic because GPU_MODE is not DEFAULT"
fi

# 5. Run co-located experiments with Tally priority scheduler
if [[ $GPU_MODE == "Default" ]]; then
echo "======== Collecting pair-wise performance with Tally priority scheduler ... ========"
SCHEDULER_POLICY=PRIORITY \
    time_cmd \
    python3 -u scripts/run_bench.py \
        --save-results \
        --use-tally \
        --runtime $RUNTIME \
        --warmup-iters $WARMUP_ITERS \
        --run-pairwise
else
    echo "Skip collecting pair-wise performance with Tally priority scheduler because GPU_MODE is not DEFAULT"
fi