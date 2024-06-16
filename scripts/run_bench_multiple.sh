#!/bin/bash

set -e

# Set a single cuda device to run the benchmarks
export CUDA_VISIBLE_DEVICES=0
echo "Running benchmarks on GPU: $CUDA_VISIBLE_DEVICES"

export GPU_MODE=$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=compute_mode --format=csv | awk 'NR==2')

if [[ $GPU_MODE == "Exclusive_Process" ]]; then
    for i in {1..10}
    do
        python3 ./scripts/run_bench_multiple.py --num-workloads $i
    done
fi