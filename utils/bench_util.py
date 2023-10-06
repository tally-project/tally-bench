import sys
import subprocess
import time
import json
import os

from utils.util import execute_cmd
from utils.mps import start_mps, shut_down_mps
from utils.tally import shut_down_tally, start_tally, tally_client_script, start_iox_roudi, shut_down_iox_roudi

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

    out, err = execute_cmd("nvidia-smi --query-gpu=compute_mode --format=csv", get_output=True)
    mode = out.split("compute_mode")[1].strip()

    if use_mps:
        if mode != "Exclusive_Process":
            raise Exception(f"GPU mode is not Exclusive_Process. Now: {mode}")

    elif use_tally:
        if mode != "Default":
            raise Exception(f"GPU mode is not Default. Now: {mode}")
        start_iox_roudi()

def tear_down_env():
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

    if use_tally:
        shut_down_tally()
        start_tally()

    if use_mps:
        shut_down_mps()
        start_mps()

    for benchmark in benchmarks:
        
        launch_cmd = (f"python3 -u launch.py " +
                        f"--framework {benchmark.framework} " +
                        f"--benchmark {benchmark.model_name} " +
                        f"--batch-size {benchmark.batch_size} " +
                        f"--warmup-iters {benchmark.warmup_iters} " +
                        f"--runtime {benchmark.runtime} " +
                        f"--signal ")
        
        if benchmark.total_iters:
            launch_cmd += f"--total-iters {benchmark.total_iters} "

        if benchmark.amp:
            launch_cmd += "--amp "
        
        if use_tally:
            launch_cmd = f"{tally_client_script} {launch_cmd}"

        launch_cmd_list = launch_cmd.strip().split(" ")
        process = subprocess.Popen(launch_cmd_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        processes.append(process)

        while True:
            poll = process.poll()
            if poll is not None:
                abort = True
                break
            process.stdout.flush()
            response = process.stdout.readline().strip()
            if response:
                print(response)
            if "benchmark is warm" in response:
                break
            time.sleep(0.01)

    if abort:
        print("Detect process abort. Terminating ...")
        for process in processes:
            process.terminate()
        std_out = process.stdout.readlines()
        std_err = process.stderr.readlines()
        for line in std_out + std_err:
            print(line.strip())
        shut_down_tally()
        exit(1)

    # All benchmarks should be warm, signal start
    print("Setting start signals ...")

    for process in processes:
        process.stdin.write("start\n")
        process.stdin.flush()
    
    for i in range(len(processes)):
        process = processes[i]
        process.wait()
        output = process.communicate()[0].strip()
        result_dict = json.loads(output.split("\n")[-1])

        bench = benchmarks[i]
        output_dict[f"{bench}_{i}"] = result_dict
        print(result_dict)
    
    if use_tally:
        shut_down_tally()
    if use_mps:
        shut_down_mps()