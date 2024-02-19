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
from utils.util import logger


class Benchmark:

    def __init__(self, framework, model_name, warmup_iters, runtime, is_train,
                 batch_size=1, amp=False, total_iters=None, infer_mode=None,
                 infer_load=None):
        
        if is_train:
            assert(batch_size)
        else:
            assert(infer_mode in ["single-stream", "server"])
        
        self.framework = framework
        self.model_name = model_name
        self.warmup_iters = warmup_iters
        self.runtime = runtime
        self.is_train = is_train
        self.batch_size = batch_size
        self.amp = amp
        self.total_iters = total_iters
        self.infer_mode = infer_mode
        self.infer_load = infer_load
        self.priority = None
        self.replace_cublas = False

        if (
            (not self.is_latency_critical()) or
            any([m in self.model_name for m in ["yolo", "gpt-neo", "stable", "stable-diffusion"]])
        ):
            self.replace_cublas = True
    
    def is_latency_critical(self):
        if not self.is_train:
            if self.infer_mode in ["single-stream", "server"]:
                return True
        
        return False

    def set_priority(self, priority):
        self.priority = priority

    def __str__(self):
        _str = f"{self.framework}_{self.model_name}"

        if self.is_train:
            _str += "_train"
        else:
            _str += f"_infer_{self.infer_mode}"
            if self.infer_mode == "server":
                _str += f"_load_{self.infer_load}"

        _str += f"_{self.batch_size}"

        if self.amp:
            _str += "_amp"

        return _str

    def get_launch_cmd(self, use_tally, pipe_name=None):
        launch_cmd = (f"python3 -u scripts/launch.py "
                            f"--framework {self.framework} "
                            f"--benchmark {self.model_name} "
                            f"--batch-size {self.batch_size} "
                            f"--warmup-iters {self.warmup_iters} "
                            f"--runtime {self.runtime} "
                            f"{'--train ' if self.is_train else '--infer '}"
                            f"{'--amp ' if self.amp else ''}")

        if self.total_iters:
            launch_cmd += f"--total-iters {self.total_iters} "
        
        if pipe_name:
            launch_cmd += f"--signal --pipe {pipe_name} "
        
        if self.infer_mode:
            launch_cmd += f"--infer-type {self.infer_mode} "
        
            if self.infer_load:
                launch_cmd += f"--infer-load {self.infer_load} "
        
        if use_tally:
            launch_cmd = f"{tally_client_script} {launch_cmd}"
        else:
            launch_cmd = f"{tally_client_local_script} {launch_cmd}"
        
        return launch_cmd


def get_train_benchmarks(training_workloads, warmup_iters, runtime):
    train_benchmarks = []

    for framework in training_workloads:
        for model in training_workloads[framework]:

            bench_config = training_workloads[framework][model]

            for batch_size in bench_config["batch-sizes"]:
                for amp in bench_config["amp"]:

                    bench = Benchmark(framework, model, warmup_iters, runtime, is_train=True,
                                      batch_size=batch_size, amp=amp)
                    train_benchmarks.append(bench)

    return train_benchmarks

def get_infer_benchmarks(inference_workloads, inference_load_factors, warmup_iters, runtime):
    infer_benchmarks = []

    for framework in inference_workloads:
        for model in inference_workloads[framework]:
            
            single_stream_bench = Benchmark(framework, model, warmup_iters, runtime, is_train=False,
                                            batch_size=1, infer_mode="single-stream")
            infer_benchmarks.append(single_stream_bench)
        
            for load in inference_load_factors:
                server_bench = Benchmark(framework, model, warmup_iters, runtime, is_train=False, 
                                        batch_size=1, infer_mode="server", infer_load=load)
                infer_benchmarks.append(server_bench)

    return infer_benchmarks

def launch_benchmark(benchmarks: List[Benchmark], use_mps=False, use_tally=False, result=None,
                     profile_only=False, preemption_limit=None):

    output_dict = None
    result_key = "default"
    manager = Manager()
    smi_list = manager.list()

    # easier for json
    if preemption_limit is not None:
        preemption_limit = str(preemption_limit)

    if use_mps:
        result_key = "mps"
    elif use_tally:
        policy = os.environ.get("SCHEDULER_POLICY", "NAIVE")
        result_key = f"tally_{policy}".lower()
    
    if result_key not in result:
        result[result_key] = {}

    bench_id = get_bench_id(benchmarks)
    if bench_id in result[result_key]:

        bench_res = result[result_key][bench_id]

        if use_tally and policy == "PRIORITY" and preemption_limit is not None:
            if preemption_limit in bench_res:
                preemption_limit_res = bench_res[preemption_limit]
                if len(preemption_limit_res) == 1 and list(preemption_limit_res.keys())[0] == "profiled":
                    if profile_only:
                        return False
                else:
                    return False
        elif len(bench_res) == 1 and list(bench_res.keys())[0] == "profiled":
            if profile_only:
                return False
        else:
            return False

    if use_tally and policy == "PRIORITY" and preemption_limit is not None:

        if bench_id not in result[result_key]:
            result[result_key][bench_id] = {}
        if preemption_limit not in result[result_key][bench_id]:
            result[result_key][bench_id][preemption_limit] = {}
        output_dict = result[result_key][bench_id][preemption_limit]
        
    else:
        result[result_key][bench_id] = {}
        output_dict = result[result_key][bench_id]

    processes = []
    abort = False
    smi_p = None

    try:
        if use_mps:
            shut_down_mps()
            start_mps()

        elif use_tally:
            shut_down_tally()
            shut_down_iox_roudi
            start_iox_roudi()
            start_tally(preemption_limit)

        for idx, benchmark in enumerate(benchmarks):

            pipe_name = get_pipe_name(idx)

            try:
                os.mkfifo(pipe_name)
            except OSError:
                pass

            launch_cmd = benchmark.get_launch_cmd(pipe_name=pipe_name, use_tally=use_tally)
            logger.info(f"bench {idx} launch_cmd: {launch_cmd}")

            process_env = os.environ.copy()
            if benchmark.priority:
                process_env["PRIORITY"] = str(benchmark.priority)
            if benchmark.replace_cublas:
                process_env["REPLACE_CUBLAS"] = "TRUE"

            launch_cmd_list = launch_cmd.strip().split(" ")
            process = subprocess.Popen(launch_cmd_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, universal_newlines=True, env=process_env)
            # When debugging, turn on the stderr
            # process = subprocess.Popen(launch_cmd_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            #                            stderr=subprocess.PIPE, universal_newlines=True, env=process_env)
            processes.append(process)

            def wait_for_bench_warm():
                
                pipe_fd = os.open(pipe_name, os.O_RDONLY | os.O_NONBLOCK)

                while not abort:
                    should_exit = False
                    readable, _, _ = select.select([pipe_fd], [], [], 1)
                    if readable:
                        chunk = os.read(pipe_fd, 4096)
                        lines = chunk.decode('utf-8').split('\n')
                        for line in lines:
                            if "benchmark is warm" in line:
                                logger.info(f"benchmark {idx} is warm")
                                should_exit = True
                        if should_exit:
                            break
                            
            wait_t = threading.Thread(target=wait_for_bench_warm)
            wait_t.start()

            sel = selectors.DefaultSelector()
            sel.register(process.stdout, selectors.EVENT_READ)
            # sel.register(process.stderr, selectors.EVENT_READ)

            p_stdout = ""
        
            while True:
                poll = process.poll()
                if poll is not None or (use_tally and query_tally() == 1):
                    abort = True
                    wait_t.join()
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
            logger.info("Detect process abort.")
            for process in processes:
                process.kill()
            raise Exception("Detect process abort.")

        # All benchmarks should be warm, signal start
        logger.info("Setting start signals ...")

        for i in range(len(processes)):
            pipe_name = get_pipe_name(i)

            with open(pipe_name, 'w') as pipe:
                pipe.write("start\n")
        
        smi_p = Process(target=smi_getter, args=(smi_list,))
        smi_p.start()

        logger.info("waiting for benchmark to finish ...")
        
        abort_timeout = benchmarks[0].runtime * 2

        for i in range(len(processes)):
            process = processes[i]
            stdout, stderr = process.communicate(timeout=abort_timeout)
            if smi_p.is_alive():
                smi_p.terminate()
                
            print(stdout.strip())
            output_lines = stdout.split("\n")

            result_dict = None
            for line in output_lines:
                try:
                    parsed = json.loads(line)
                    if "time_elapsed" in parsed and "iters" in parsed:
                        result_dict = parsed
                        break
                except:
                    pass
            if not result_dict:
                raise Exception("Cannot parse result dict")
            
            bench = benchmarks[i]
            if bench.priority:
                result_dict["priority"] = bench.priority

            if not profile_only:
                output_dict[f"{bench}_{i}"] = result_dict

        if not profile_only:
            output_dict["metrics"] =  parse_smi_list(smi_list)
        else:
            output_dict["profiled"] = True

        logger.info(f"bench_id: {bench_id}")
        logger.info(output_dict)

    except Exception as e:
        output_dict["error"] = str(e)
        logger.warning(f"Caught exception when running the benchmark: Error: {e}")
        time.sleep(10)
    finally:
        if smi_p and smi_p.is_alive():
            smi_p.terminate()
        for process in processes:
            process.kill()
        tear_down_env()

    return True