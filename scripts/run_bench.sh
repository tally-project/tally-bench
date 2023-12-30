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

# Make sure the script is run with sudo permission
# This is to change the nvidia GPU mode between DEFAULT and EXCLUSIVE_PROCESS
if [ "$UID" -ne 0 ]; then
    echo "This script must be run with sudo."
    exit 1
fi

export RUNTIME=60
export WARMUP_ITERS=100

# Set GPU mode to DEFAULT for non-MPS experiments
nvidia-smi -i 0 -c DEFAULT

# 1. Collect single-job performance with and without Tally
#    This should give insights of the overhead of API forwarding in Tally
echo "======== Collecting single-job performance with and without Tally ... ========"
time_cmd sudo -u $SUDO_USER \
    python3 -u scripts/run_bench.py \
        --save-results \
        --use-tally \
        --runtime $RUNTIME \
        --warmup-iters $WARMUP_ITERS

# 2. Profile kernel metrics for workload-agnostic scheduler
#    Run each job for a long period of time to make sure each kernel is profiled
#    Results won't be collected
echo "======== Profiling kernel metrics for workload-agnostic scheduler ... ========"
time_cmd sudo -u $SUDO_USER \
    SCHEDULER_POLICY=WORKLOAD_AGNOSTIC_SHARING \
    python3 -u scripts/run_bench.py \
        --use-tally \
        --runtime 600 \
        --warmup-iters 1000

# Save results now
time_cmd sudo -u $SUDO_USER \
    SCHEDULER_POLICY=WORKLOAD_AGNOSTIC_SHARING \
    python3 -u scripts/run_bench.py \
        --save-results \
        --use-tally \
        --runtime $RUNTIME \
        --warmup-iters $WARMUP_ITERS

# 3. Run co-located experiments without MPS nor Tally (Hardware multi-processing)
echo "======== Collecting pair-wise performance with and hardware multi-processing ... ========"
time_cmd sudo -u $SUDO_USER \
    python3 -u scripts/run_bench.py \
        --save-results \
        --runtime $RUNTIME \
        --warmup-iters $WARMUP_ITERS \
        --run-pairwise

# 4. Run co-located experiments with MPS
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
echo "======== Collecting pair-wise performance with MPS ... ========"
time_cmd sudo -u $SUDO_USER \
    python3 -u scripts/run_bench.py \
        --save-results \
        --use-mps \
        --runtime $RUNTIME \
        --warmup-iters $WARMUP_ITERS \
        --run-pairwise