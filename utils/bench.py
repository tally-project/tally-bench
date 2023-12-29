from typing import List
import subprocess
import time
import json
import os
import select
import pandas as pd
import threading
import selectors
from multiprocessing import Process, Manager

from utils.nvidia_smi import smi_getter, parse_smi_list
from utils.bench_util import get_bench_id, get_pipe_name, tear_down_env
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


class Benchmark:

    def __init__(self, framework, model_name, batch_size, amp, warmup_iters, runtime, total_iters=None):
        self.framework = framework
        self.model_name = model_name
        self.batch_size = batch_size
        self.amp = amp
        self.warmup_iters = warmup_iters
        self.runtime = runtime
        self.total_iters = total_iters
    
    def __str__(self):
        _str = f"{self.framework}_{self.model_name}_{self.batch_size}"
        if self.amp:
            _str += "_amp"
        return _str
    

def launch_benchmark(benchmarks: List[Benchmark], use_mps=False, use_tally=False, result=None):

    output_dict = None
    result_key = "default"
    manager = Manager()
    smi_list = manager.list()

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
            shut_down_tally()
            shut_down_iox_roudi
            start_iox_roudi()
            start_tally()

        for idx, benchmark in enumerate(benchmarks):

            pipe_name = get_pipe_name(idx)

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
            else:
                launch_cmd = f"{tally_client_local_script} {launch_cmd}"

            print(f"launch_cmd: {launch_cmd}")

            launch_cmd_list = launch_cmd.strip().split(" ")
            process = subprocess.Popen(launch_cmd_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True)
            # When debugging, turn on the stderr
            # process = subprocess.Popen(launch_cmd_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            processes.append(process)

            def wait_for_bench_warm():
                # Note this will block indefinitely if the client aborts
                while not abort:
                    with open(pipe_name, 'r') as fifo:
                        readable, _, _ = select.select([fifo], [], [], 1)
                        if readable:
                            line = fifo.readline()
                            if "benchmark is warm" in line:
                                print("benchmark is warm")
                                break

            wait_t = threading.Thread(target=wait_for_bench_warm)
            wait_t.start()

            sel = selectors.DefaultSelector()
            sel.register(process.stdout, selectors.EVENT_READ)
            # sel.register(process.stderr, selectors.EVENT_READ)

            p_stdout = ""

            try:
                os.mkfifo(pipe_name)
            except OSError:
                pass
        
            while True:
                poll = process.poll()
                if poll is not None or (use_tally and query_tally() == 1):
                    abort = True
                    output_dict["error"] = "Encountered error."
                    break

                for key, val in sel.select(timeout=1):
                    line = key.fileobj.readline()
                    if line:
                        p_stdout += line

                if not wait_t.is_alive():
                    break
            
            print(p_stdout.strip())

        if abort:
            print("Detect process abort.")
            for process in processes:
                process.kill()
                stdout, stderr = process.communicate()
                print(stdout.strip())
                if stderr:
                    print(stderr.strip())
            raise Exception("Detect process abort.")

        # All benchmarks should be warm, signal start
        print("Setting start signals ...")

        for i in range(len(processes)):
            pipe_name = get_pipe_name(i)

            with open(pipe_name, 'w') as pipe:
                pipe.write("start\n")
        
        smi_p = Process(target=smi_getter, args=(smi_list, [0]))
        smi_p.start()

        print("waiting for benchmark to finish ...")
        for i in range(len(processes)):
            process = processes[i]
            stdout, stderr = process.communicate()
            if smi_p.is_alive():
                smi_p.terminate()
                
            print(stdout.strip())
            output_lines = stdout.split("\n")

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

        output_dict["metrics"] =  parse_smi_list(smi_list)

        print(output_dict)
        print(bench_id)

    except Exception as e:
        print(f"Caught exception when running the benchmark: Error: {e}")
        time.sleep(10)
    finally:
        tear_down_env()