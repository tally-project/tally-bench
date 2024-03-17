#!/bin/bash

set -e

# Set a single cuda device to run the benchmarks
export CUDA_VISIBLE_DEVICES=0
echo "Running benchmarks on GPU: $CUDA_VISIBLE_DEVICES"

export GPU_MODE=$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=compute_mode --format=csv | awk 'NR==2')

if [[ $GPU_MODE == "Exclusive_Process" ]]; then
    python3 ./scripts/run_azure_bench.py --use-mps
    python3 ./scripts/run_azure_bench.py --use-mps-priority
    python3 ./scripts/run_azure_bench.py --use-tally-priority
fi

if [[ $GPU_MODE == "Default" ]]; then
    python3 ./scripts/run_azure_bench.py
fi