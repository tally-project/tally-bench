import sys
import subprocess
import time
import json
import os
import random
import numpy as np
import selectors

import torch

from utils.util import execute_cmd
from utils.mps import start_mps, shut_down_mps
from utils.tally import shut_down_tally, start_tally, tally_client_script, start_iox_roudi, shut_down_iox_roudi, query_tally

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

def wait_for_signal():
    while True:
        print("benchmark is warm\n", flush=True)
        sys.stdin.flush()
        inp = sys.stdin.readline()
        if "start" in inp:
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

        for benchmark in benchmarks:
            
            launch_cmd = (f"python3 -u launch.py " +
                            f"--framework {benchmark.framework} " +
                            f"--benchmark {benchmark.model_name} " +
                            f"--batch-size {benchmark.batch_size} " +
                            f"--warmup-iters {benchmark.warmup_iters} " +
                            f"--runtime {benchmark.runtime} " +
                            f"--signal ")
            
            print(f"launch_cmd: {launch_cmd}")
            
            if benchmark.total_iters:
                launch_cmd += f"--total-iters {benchmark.total_iters} "

            if benchmark.amp:
                launch_cmd += "--amp "
            
            if use_tally:
                launch_cmd = f"{tally_client_script} {launch_cmd}"

            launch_cmd_list = launch_cmd.strip().split(" ")
            process = subprocess.Popen(launch_cmd_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            processes.append(process)

            sel = selectors.DefaultSelector()
            sel.register(process.stdout, selectors.EVENT_READ)
            sel.register(process.stderr, selectors.EVENT_READ)
            break_loop = False

            while True:
                poll = process.poll()
                if poll is not None or (use_tally and query_tally() == 1):
                    abort = True
                    break
            
                for key, val1 in sel.select(timeout=1):
                    line = key.fileobj.readline()
                    if line:
                        print(line, end="")
                    if not line or "benchmark is warm" in line:
                        break_loop = True
                        break
                if break_loop:
                    break

                time.sleep(0.01)

        if abort:
            print("Detect process abort.")
            for process in processes:
                process.kill()
            raise Exception("Detect process abort.")

        # All benchmarks should be warm, signal start
        print("Setting start signals ...")

        for process in processes:
            process.stdin.write("start\n")
            process.stdin.flush()
        
        for i in range(len(processes)):
            process = processes[i]
            process.wait()
            output = process.communicate()[0].strip()
            try:
                result_dict = json.loads(output.split("\n")[-1])
            except Exception as e:
                print(output.split("\n")[-1])
                raise e

            bench = benchmarks[i]
            output_dict[f"{bench}_{i}"] = result_dict
            print(result_dict)
            
        print(bench_id)
    except Exception as e:
        print(f"Caught exception when running the benchmark: Error: {e}")
        time.sleep(10)
    finally:
        tear_down_env()