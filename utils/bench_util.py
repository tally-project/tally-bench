import os
import random
import numpy as np
import select
import logging

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


def set_all_logging_level(level):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)


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


def get_cuda_device_id():
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    cuda_devices = cuda_devices.split(",")
    return cuda_devices[0]

  
def init_env(use_mps=False, use_tally=False):
    tear_down_env()
    cuda_device_id = get_cuda_device_id()

    out, err, rc = execute_cmd(f"nvidia-smi -i {cuda_device_id} --query-gpu=compute_mode --format=csv", get_output=True)
    mode = out.split("compute_mode")[1].strip().split("\n")[0]

    required_mode = ""

    if use_mps:
        required_mode = "Exclusive_Process"
    elif use_tally:
        required_mode = "Default"
    else:
        required_mode = "Default"

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