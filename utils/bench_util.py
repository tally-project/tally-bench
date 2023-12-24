import sys
import subprocess
import time
import json
import os
import random
import numpy as np
import logging
import select

import torch

from utils.util import execute_cmd
from utils.mps import start_mps, shut_down_mps
from utils.tally import (
    shut_down_tally,
    start_tally,
    tally_client_script,
    tally_client_local_script, 
    start_iox_roudi,
    shut_down_iox_roudi,
    query_tally
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

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

def launch_benchmark(benchmarks: list, use_mps=False, use_tally=False, result=None):

    output_dict = None
    result_key = "default"

    if use_mps:
        result_key = "mps"
    elif use_tally:
        policy = os.environ.get("SCHEDULER_POLICY", "NAIVE")
        result_key = f"tally_{policy}".lower()
    
    if result_key not in result:
        result[result_key] = {}

    bench_id = get_bench_id(benchmarks)
    if bench_id in result[result_key]:
        return
    
    result[result_key][bench_id] = {}
    output_dict = result[result_key][bench_id]

    processes = []
    abort = False

    try:
        if use_mps:
            shut_down_mps()
            start_mps()

        elif use_tally:

            if policy == "WORKLOAD_AGNOSTIC_SHARING":
                shut_down_mps()
                start_mps()
            else:
                shut_down_tally()
                shut_down_iox_roudi
                start_iox_roudi()
                start_tally()

        for idx, benchmark in enumerate(benchmarks):

            pipe_name = f"/tmp/tally_bench_pipe_{idx}"

            launch_cmd = (f"python3 -u scripts/launch.py "
                            f"--framework {benchmark.framework} "
                            f"--benchmark {benchmark.model_name} "
                            f"--batch-size {benchmark.batch_size} "
                            f"--warmup-iters {benchmark.warmup_iters} "
                            f"--runtime {benchmark.runtime} "
                            f"--signal "
                            f"--pipe {pipe_name} ")
            
            if benchmark.total_iters:
                launch_cmd += f"--total-iters {benchmark.total_iters} "

            if benchmark.amp:
                launch_cmd += "--amp "
            
            if use_tally:
                launch_cmd = f"{tally_client_script} {launch_cmd}"
            # else:
            #     launch_cmd = f"{tally_client_local_script} {launch_cmd}"

            print(f"launch_cmd: {launch_cmd}")

            launch_cmd_list = launch_cmd.strip().split(" ")
            process = subprocess.Popen(launch_cmd_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            processes.append(process)

            try:
                os.mkfifo(pipe_name)
            except OSError:
                pass
        
            while True:
                poll = process.poll()
                if poll is not None or (use_tally and policy != "WORKLOAD_AGNOSTIC_SHARING" and query_tally() == 1):
                    abort = True
                    output_dict["error"] = "Encountered error."
                    break

                # Note this will block indefinitely if the client aborts
                with open(pipe_name, 'r') as fifo:
                    readable, _, _ = select.select([fifo], [], [], 1)
                    if readable:
                        line = fifo.readline()
                        if "benchmark is warm" in line:
                            print("benchmark is warm")
                            break

        if abort:
            print("Detect process abort.")
            for process in processes:
                process.kill()
            raise Exception("Detect process abort.")

        # All benchmarks should be warm, signal start
        print("Setting start signals ...")

        for i in range(len(processes)):
            pipe_name = f"/tmp/tally_bench_pipe_{i}"

            with open(pipe_name, 'w') as pipe:
                pipe.write("start\n")
        
        for i in range(len(processes)):
            process = processes[i]
            process.wait()
            output = process.communicate()[0].strip()
            print(output)
            output_lines = output.split("\n")
            result_dict = None
            for line in output_lines:
                try:
                    result_dict = json.loads(line)
                    break
                except:
                    pass
            if not result_dict:
                raise Exception("Cannot parse result dict")
            bench = benchmarks[i]
            output_dict[f"{bench}_{i}"] = result_dict

        print(output_dict)
        print(bench_id)

    except Exception as e:
        print(f"Caught exception when running the benchmark: Error: {e}")
        time.sleep(10)
    finally:
        tear_down_env()