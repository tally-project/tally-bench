#!/bin/bash

export TALLY_HOME=$PWD/tally

TALLY_PRELOAD_LOCAL="./tally/scripts/start_client_local.sh"
TALLY_PRELOAD="./tally/scripts/start_client.sh"

RUN_ORIGINAL=TRUE
RUN_TALLY_LOCAL=TRUE
RUN_TALLY=TRUE

train_pytorch_models=(
    "resnet50"
    "bert"
    "pointnet"
    "pegasus-x-base"
    "whisper-large-v3"
    "gpt2-large"
)

infer_hidet_models=(
    "resnet50"
)

infer_onnxruntime_models=(
    "llama-2-7b"
    "bert"
)

infer_pytorch_models=(
    "yolov6m"
    "gpt-neo-2.7B"
    "stable-diffusion"
)

cleanup() {
    ./tally/scripts/kill_server.sh
    ./tally/scripts/kill_iox.sh
}

run_tally_test() {

    # Launch tally server in the background
    ./tally/scripts/start_server.sh &

    sleep 5

    # Launch client process
    $TALLY_PRELOAD $@

    ./tally/scripts/kill_server.sh
}

# arg_1: framework
# arg_2: model
launch_bench() {

    local is_train=FALSE

    for arg in "$@"; do
        if [[ $arg == *train* ]]; then
            is_train=TRUE
        fi
    done

    launch_cmd="python3 -u ./scripts/launch.py --framework $1 --benchmark $2 --warmup-iters 10 --runtime 5 ${@:3}"

    if [ "$RUN_ORIGINAL" = "TRUE" ]; then
        echo $launch_cmd
        $launch_cmd
        echo
    fi

    if [ "$RUN_TALLY_LOCAL" = "TRUE" ]; then
        echo REPLACE_CUBLAS=TRUE $TALLY_PRELOAD_LOCAL $launch_cmd
        REPLACE_CUBLAS=TRUE $TALLY_PRELOAD_LOCAL $launch_cmd
        echo
    fi

    if [ "$RUN_TALLY" = "TRUE" ]; then
        echo REPLACE_CUBLAS=TRUE $TALLY_PRELOAD $launch_cmd
        REPLACE_CUBLAS=TRUE run_tally_test $launch_cmd
        echo
    fi
}

# Set up
trap cleanup ERR
set -e

if [ "$RUN_TALLY" = "TRUE" ]; then
    ./tally/scripts/start_iox.sh &
    sleep 5
fi

for model in "${infer_hidet_models[@]}"; do
    launch_bench hidet $model --infer --batch-size 1
done

for model in "${infer_pytorch_models[@]}"; do
    launch_bench pytorch $model --infer --batch-size 1
done

for model in "${infer_onnxruntime_models[@]}"; do
    launch_bench onnxruntime $model --infer --batch-size 1
done

for model in "${train_pytorch_models[@]}"; do
    if [ "$model" = "pointnet" ]; then
        batch_size=2
    else
        batch_size=1
    fi
    launch_bench pytorch $model --train --batch-size $batch_size
done

cleanup