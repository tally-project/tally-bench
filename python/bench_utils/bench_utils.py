import os
import random
import numpy as np
import select
import logging
import errno
from datetime import datetime

import torch

from bench_utils.utils import execute_cmd
from bench_utils.mps import shut_down_mps
from bench_utils.tally import (
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

  
def init_env(use_mps=False, use_tally=False, use_tgs=False, run_pairwise=False):
    tear_down_env()

    # does not matter if not running pairwise
    if not run_pairwise:
        return

    cuda_device_id = get_cuda_device_id()

    out, err, rc = execute_cmd(f"nvidia-smi -i {cuda_device_id} --query-gpu=compute_mode --format=csv", get_output=True)
    mode = out.split("compute_mode")[1].strip().split("\n")[0]

    required_mode = ""

    if use_mps:
        required_mode = "Exclusive_Process"
    elif use_tally or use_tgs:
        required_mode = "Exclusive_Process"
    else:
        required_mode = "Default"

    if mode != required_mode:
        raise Exception(f"GPU mode is not {required_mode}. Now: {mode}")


def tear_down_env():
    shut_down_tally()
    shut_down_mps()
    shut_down_iox_roudi()


def wait_for_signal(pipe_name, break_if_not_ready=False):

    if break_if_not_ready:
        try:
            pipe_fd = os.open(pipe_name, os.O_WRONLY | os.O_NONBLOCK)
            with os.fdopen(pipe_fd, 'w') as pipe:
                pipe.write("benchmark is warm\n")

            pipe_fd = os.open(pipe_name, os.O_RDONLY | os.O_NONBLOCK)
            with os.fdopen(pipe_fd, 'r') as pipe:
                while True:
                    readable, _, _ = select.select([pipe], [], [], 0.0000001)

                    if readable and "start" in pipe.readline():
                        return True
                
                    if break_if_not_ready:
                        return False
        except Exception as e:
            return False
    else:
        with open(pipe_name, 'w') as pipe:
            pipe.write("benchmark is warm\n")

        with open(pipe_name, 'r') as pipe:
            while True:
                readable, _, _ = select.select([pipe], [], [], 1)
                if readable and "start" in pipe.readline():
                    return True


def get_backend_name(use_tally=False, use_mps=False, use_mps_priority=False, use_tgs=False, tally_config=None):
    backend = "default"

    if use_mps_priority:
        backend = "mps-priority"
    elif use_mps:
        backend = "mps"
    elif use_tgs:
        backend = "tgs"
    elif use_tally:
        backend = f"tally_{tally_config.scheduler_policy}".lower()

    return backend