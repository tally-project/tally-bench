import os
import random
import numpy as np
import select

import torch

from utils.util import execute_cmd
from utils.mps import shut_down_mps
from utils.tally import (
    shut_down_tally,
    shut_down_iox_roudi
)


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_bench_id(benchmarks: list):
    _str = ""
    for i in range(len(benchmarks)):
        benchmark = benchmarks[i]
        _str += str(benchmark)
        if i != len(benchmarks) - 1:
            _str += "_"
    return _str


def get_pipe_name(idx):
    return f"/tmp/tally_bench_pipe_{idx}"


def get_torch_compile_options():
    compile_options = {
        "epilogue_fusion": True,
        "max_autotune": True,
        "triton.cudagraphs": False
    }

    return compile_options


def get_benchmark_func(framework, model_name):
    bench_func = None

    if framework == "hidet":
        if model_name in ["resnet50"]:
            from workloads.hidet.resnet import run_resnet as hidet_run_resnet
            bench_func = hidet_run_resnet
    
    elif framework == "pytorch":

        if model_name in ["resnet50"]:
            from workloads.pytorch.resnet.train_resnet import train_resnet
            bench_func = train_resnet

        if model_name in ["bert"]:
            from workloads.pytorch.bert.train_bert import train_bert
            bench_func = train_bert

        if model_name in ["VGG", "ShuffleNetV2"]:
            from workloads.pytorch.cifar.train_cifar import train_cifar
            bench_func = train_cifar

        if model_name in ["dcgan"]:
            from workloads.pytorch.dcgan.train_dcgan import train_dcgan
            bench_func = train_dcgan

        if model_name in ["LSTM"]:
            from workloads.pytorch.lstm.train_lstm import train_lstm
            bench_func = train_lstm

        if model_name in ["NeuMF-pre"]:
            from workloads.pytorch.ncf.train_ncf import train_ncf
            bench_func = train_ncf
        
        if model_name in ["pointnet"]:
            from workloads.pytorch.pointnet.train_pointnet import train_pointnet
            bench_func = train_pointnet

        if model_name in ["transformer"]:
            from workloads.pytorch.transformer.train_transformer import train_transformer
            bench_func = train_transformer

        if model_name in ["yolov6n"]:
            from workloads.pytorch.yolov6.train_yolov6 import train_yolov6
            bench_func = train_yolov6
    
        if model_name in ["pegasus-x-base", "pegasus-large"]:
            from workloads.pytorch.pegasus.train_pegasus import train_pegasus
            bench_func = train_pegasus

        if model_name in ["whisper-small"]:
            from workloads.pytorch.whisper.train_whisper import train_whisper
            bench_func = train_whisper

    return bench_func

  
def init_env(use_mps=False, use_tally=False):
    tear_down_env()

    out, err, rc = execute_cmd("nvidia-smi --query-gpu=compute_mode --format=csv", get_output=True)
    mode = out.split("compute_mode")[1].strip()

    required_mode = ""

    if use_mps:
        required_mode = "Exclusive_Process"

    elif use_tally:
        scheduler_policy = os.environ.get("SCHEDULER_POLICY", "NAIVE")

        if scheduler_policy == "WORKLOAD_AGNOSTIC_SHARING":
            required_mode = "Exclusive_Process"
        else:
            required_mode = "Default"
    else:
        return

    if mode != required_mode:
        raise Exception(f"GPU mode is not {required_mode}. Now: {mode}")


def tear_down_env():
    shut_down_tally()
    shut_down_mps()
    shut_down_iox_roudi()


def wait_for_signal(pipe_name):

    with open(pipe_name, 'w') as pipe:
        pipe.write("benchmark is warm\n")

    with open(pipe_name, 'r') as pipe:
        while True:
            readable, _, _ = select.select([pipe], [], [], 1)
            if readable:
                line = pipe.readline()
                if "start" in line:
                    break