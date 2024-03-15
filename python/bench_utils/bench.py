from typing import List
import subprocess
import time
import json
import os
import select
import pandas as pd
import threading
import selectors
import copy
from multiprocessing import Process, Manager

from bench_utils.utils import load_json_from_file, write_json_to_file, logger, get_possion_arrival_trace
from bench_utils.bench_utils import init_env, tear_down_env, get_bench_id, get_pipe_name, get_backend_name
from bench_utils.nvidia_smi import smi_getter, parse_smi_list, get_cuda_mem
from bench_utils.mps import start_mps, shut_down_mps
from bench_utils.tally import (
    shut_down_tally,
    start_tally,
    tally_client_script,
    tally_client_local_script, 
    start_iox_roudi,
    shut_down_iox_roudi,
    query_tally,
    TallyConfig
)

from configs.train_config import training_workloads
from configs.infer_config import inference_workloads, inference_load_factors


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
        self.trace_file = None

        if (
            (not self.is_latency_critical()) or
            any([m in self.model_name for m in ["bert", "yolo", "gpt-neo", "stable", "stable-diffusion"]])
        ):
            self.replace_cublas = True
    
    def is_latency_critical(self):
        return not self.is_train

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
        
        if self.infer_mode:
            launch_cmd += f"--infer-type {self.infer_mode} "
        
            if self.trace_file:
                launch_cmd += f"--infer-trace {self.trace_file} "
            elif self.infer_load:
                launch_cmd += f"--infer-load {self.infer_load} "
        
        if pipe_name:
            launch_cmd += f"--signal --pipe {pipe_name} "
        
        if use_tally:
            launch_cmd = f"{tally_client_script} {launch_cmd}"
        else:
            launch_cmd = f"{tally_client_local_script} {launch_cmd}"
        
        return launch_cmd


def get_infer_benchmark_trace(benchmark, result, trace_path):
    if os.path.exists(trace_path):
        trace = load_json_from_file(trace_path)
        return trace

    single_stream_bench = copy.deepcopy(benchmark)
    single_stream_bench.infer_mode = "single-stream"
    single_stream_bench_id = get_bench_id((single_stream_bench,))

    # get benchmark per-request latency
    single_stream_latencies = result["default"][single_stream_bench_id]["measurements"][0][f"{single_stream_bench_id}_0"]["latencies"]
    avg_latency = sum(single_stream_latencies) / len(single_stream_latencies)
    avg_latency /= 1000
    trace = get_possion_arrival_trace(avg_latency, benchmark.infer_load, benchmark.runtime)

    write_json_to_file(trace, trace_path)
    return trace


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


def get_smallest_max_allowed_latency():
    smallest_max_allowed_latency = 1
    for framework in inference_workloads:
        for model in inference_workloads[framework]:
            configs = inference_workloads[framework][model]
            for config in configs:
                smallest_max_allowed_latency = min(smallest_max_allowed_latency, config.max_allowed_latency)
    return smallest_max_allowed_latency


def get_tally_configs(benchmark):
    framework = benchmark.framework
    model = benchmark.model_name

    configs = inference_workloads[framework][model]
    return configs


def launch_benchmark(benchmarks: List[Benchmark], use_mps=False, use_mps_priority=False,
                     use_tally=False, result=None, profile_only=False, tally_config=None):

    output_dict = {}
    backend = get_backend_name(use_tally, use_mps, use_mps_priority=use_mps_priority, tally_config=tally_config)
    bench_id = get_bench_id(benchmarks)
    
    if backend not in result:
        result[backend] = {}
    backend_res = result[backend]

    if bench_id not in backend_res:
        backend_res[bench_id] = {}
        backend_res[bench_id]["measurements"] = []

    bench_res = backend_res[bench_id]
    measurements = bench_res["measurements"]

    if use_tally and backend == "tally_priority":
        if profile_only:
            assert(len(benchmarks) == 1)
            if benchmarks[0].is_latency_critical():
                return False
            if "profiled" in bench_res:
                return False
        else:
            for measurement in measurements:
                config = measurement["tally_config"]
                if config == tally_config.to_dict():
                    return False
    else:
        if len(measurements) > 0:
            return False
    
    measurements.append(output_dict)
    if tally_config:
        output_dict["tally_config"] = tally_config.to_dict()

    use_mps = use_mps or use_mps_priority
    processes = []
    abort = False
    smi_p = None
    should_exit = False

    try:
        if use_mps:
            shut_down_mps()
            start_mps()

        elif use_tally:
            shut_down_tally()
            shut_down_iox_roudi
            start_iox_roudi()
            start_tally(tally_config)

        for idx, benchmark in enumerate(benchmarks):

            pipe_name = get_pipe_name(idx)

            if os.path.exists(pipe_name):
                os.remove(pipe_name)
            os.mkfifo(pipe_name)

            launch_cmd = benchmark.get_launch_cmd(pipe_name=pipe_name, use_tally=use_tally)
            logger.info(f"bench {idx} launch_cmd: {launch_cmd}")

            process_env = os.environ.copy()
            if benchmark.priority:
                process_env["PRIORITY"] = str(benchmark.priority)
                if use_mps_priority:
                    process_env["CUDA_MPS_CLIENT_PRIORITY"] = str(benchmark.priority)
            if benchmark.replace_cublas:
                process_env["REPLACE_CUBLAS"] = "TRUE"
            if use_mps:
                process_env["CUDA_VISIBLE_DEVICES"] = "0"

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
            timeout = None
            if not profile_only:
                timeout = 1800
            start_t = time.time()
        
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
            
                curr_t = time.time()
                if timeout and curr_t - start_t >= timeout:
                    raise Exception("Timeout while waiting for benchmark to warm up")
            
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
        
        smi_list = Manager().list()
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
            bench_res["profiled"] = True

        logger.info(f"bench_id: {bench_id}")
        logger.info(output_dict)

    except ValueError as e:
        logger.warning(f"Caught exception when running the benchmark: Error: {e}")
        logger.warning("Exiting experiments ...")
        should_exit = True
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
        if should_exit:
            exit(1)

    return True


def run_benchmark_suite(
        use_mps=False,
        use_mps_priority=False,
        use_tally_naive=False,
        use_tally_priority=False,
        run_pairwise=False,
        runtime=10,
        warmup_iters=10,
        save_results=False,
        profile_only=False,
):
    tally_bench_result_dir = "tally_bench_results"
    if not os.path.exists(tally_bench_result_dir):
        os.makedirs(tally_bench_result_dir)

    result_file = f"{tally_bench_result_dir}/result.json"
    result_backup_file = f"{tally_bench_result_dir}/result_backup.json"

    def save_results_to_file(result, updated, save_results):
        if updated and save_results:
            write_json_to_file(result, result_file)
            write_json_to_file(result, result_backup_file)

    # cuda memory capacity
    cuda_mem_cap = get_cuda_mem()

    curr_dir = os.getcwd()
    os.environ["TALLY_HOME"] = f"{curr_dir}/tally"

    result = load_json_from_file(result_file)

    train_benchmarks = get_train_benchmarks(training_workloads, warmup_iters, runtime)
    infer_benchmarks = get_infer_benchmarks(inference_workloads, inference_load_factors, warmup_iters, runtime)
    
    # reduce warmup iters for long-pipeline inference benchmarks
    for bench in infer_benchmarks:
        if bench.model_name in ["stable-diffusion", "gpt-neo-2.7B", "llama-2-7b"]:
            bench.warmup_iters = 10

    use_tally = use_tally_naive or use_tally_priority
    use_mps = use_mps or use_mps_priority
    init_env(use_mps, use_tally, run_pairwise)

    single_job_benchmarks = train_benchmarks + infer_benchmarks
    if use_tally_priority:
        if profile_only:
            single_job_benchmarks = [b for b in single_job_benchmarks if not b.is_latency_critical()]
        else:
            single_job_benchmarks = []

    # Run single-job benchmark
    for idx, benchmark in enumerate(single_job_benchmarks):

        bench_id = get_bench_id([benchmark])
        logger.info(f"Running {idx + 1} out of {len(single_job_benchmarks)} single-job benchmarks: {bench_id} ...")

        if benchmark.infer_mode == "server":
            trace_path = f"infer_trace/{bench_id}.json"
            trace = get_infer_benchmark_trace(benchmark, result, trace_path)
            benchmark.trace_file = trace_path

        updated = launch_benchmark([benchmark], result=result)

        if use_tally:
            if use_tally_naive:
                scheduler_policy = "naive"
            elif use_tally_priority:
                scheduler_policy = "priority"
            tally_config = TallyConfig(scheduler_policy, max_allowed_latency=get_smallest_max_allowed_latency())
            updated |= launch_benchmark((benchmark,), result=result, use_tally=use_tally,
                                        profile_only=profile_only, tally_config=tally_config)
        save_results_to_file(result, updated, save_results)

    # Run pairwise training benchmark
    if run_pairwise:

        assert(not profile_only)

        # Prepare pairwise benchmarks
        pair_wise_benchmarks = []

        for j in range(len(infer_benchmarks)):
            for i in range(len(train_benchmarks)):
                pair = (copy.copy(train_benchmarks[i]), copy.copy(infer_benchmarks[j]))
                pair_wise_benchmarks.append(pair)

        for idx, pair in enumerate(pair_wise_benchmarks):

            bench_1, bench_2 = pair
            bench_id = get_bench_id(pair)

            if bench_2.infer_mode == "server":
                bench_2_id = get_bench_id([bench_2])
                trace_path = f"infer_trace/{bench_2_id}.json"
                trace = get_infer_benchmark_trace(bench_2, result, trace_path)
                bench_2.trace_file = trace_path

            logger.info(f"Running {idx + 1} out of {len(pair_wise_benchmarks)} pairwise benchmarks: {bench_id} ...")

            bench_1_mem = result["tally_naive"][str(bench_1)]["measurements"][0]["metrics"]["gmem"]
            bench_2_mem = result["tally_naive"][str(bench_2)]["measurements"][0]["metrics"]["gmem"]
            sum_mem = bench_1_mem + bench_2_mem

            if sum_mem > 0.98 * cuda_mem_cap:
                logger.info(f"Skipping {bench_id} as required memory of {sum_mem} MB exceeds system limit of {cuda_mem_cap} MB")
                continue

            assert(not bench_1.is_latency_critical())
 
            bench_1.set_priority(1)
            bench_2.set_priority(2)

            updated = False

            if use_tally_priority:

                tally_configs = get_tally_configs(bench_2)

                for config in tally_configs:
                    updated |= launch_benchmark(pair, use_mps=use_mps, use_mps_priority=use_mps_priority, use_tally=use_tally, result=result, tally_config=config)
            else:
                updated = launch_benchmark(pair, use_mps=use_mps, use_mps_priority=use_mps_priority, use_tally=use_tally, result=result)
            
            save_results_to_file(result, updated, save_results)

    tear_down_env()