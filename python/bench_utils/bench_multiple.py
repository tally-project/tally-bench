import os
import math
import copy
import random

from bench_utils.utils import write_json_to_file, load_json_from_file
from bench_utils.bench_utils import init_env, tear_down_env, get_bench_id
from bench_utils.bench import Benchmark, launch_benchmark, get_train_benchmarks, get_infer_benchmark_trace, get_infer_benchmark_latency
from bench_utils.tally_config import default_tally_config
from configs.train_config import training_workloads

def bench_multiple_workloads(
    num_workloads=3,
    result_file="result_multiple.json",
):
    tally_bench_result_dir = "tally_bench_results"
    if not os.path.exists(tally_bench_result_dir):
        os.makedirs(tally_bench_result_dir)

    base_result_file = f"{tally_bench_result_dir}/result.json"
    base_result = load_json_from_file(base_result_file)

    result_file = f"{tally_bench_result_dir}/{result_file}"
    result = load_json_from_file(result_file)

    curr_dir = os.getcwd()
    os.environ["TALLY_HOME"] = f"{curr_dir}/tally"

    init_env(use_tally=True, run_pairwise=True)

    benchmark = Benchmark("hidet", "resnet50", warmup_iters=100, runtime=10, is_train=False,
                          batch_size=1, infer_mode="server", infer_load=0.1)

    runtime = 60
    bench_2_id = get_bench_id([benchmark])
    trace_path = f"infer_trace/{bench_2_id}_multiple_{runtime}.json"
    if os.path.exists(trace_path):
        trace = load_json_from_file(trace_path)
    else:
        latency = get_infer_benchmark_latency(benchmark, base_result)
        num_events = int(runtime / latency * benchmark.infer_load)
        trace = sorted([random.uniform(0, 60) for _ in range(num_events)])
        write_json_to_file(trace, trace_path)

    benchmark.trace_file = trace_path
    benchmark.set_priority(2)
    benchmark.runtime = runtime

    best_effort_benchmarks = []
    for i in range(num_workloads - 1):

        be_benchmark = copy.copy(benchmark)
        be_benchmark.set_priority(1)
        best_effort_benchmarks.append(be_benchmark)
    
    benchmarks = [benchmark] + best_effort_benchmarks
        
    updated = False

    # Profile multi-job performance
    updated |= launch_benchmark(benchmarks, use_tally=True, result=result, tally_config=default_tally_config, truncate_result=False, keep_trace=True)

    if updated:
        write_json_to_file(result, result_file)

    tear_down_env()
