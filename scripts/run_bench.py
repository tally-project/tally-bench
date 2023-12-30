import sys
import os
import random
import argparse

sys.path.append('.')

from utils.util import load_json_from_file, write_json_to_file, execute_cmd
from utils.bench import Benchmark, launch_benchmark
from utils.bench_util import init_env, tear_down_env, get_bench_id
from utils.parse import parse_result
from utils.nvidia_smi import get_cuda_mem

parser = argparse.ArgumentParser(prog="benchmark suite launcher", description="Launch benchmark suite")

parser.add_argument("--save-results", action="store_true", default=False)
parser.add_argument("--use-mps", action="store_true", default=False)
parser.add_argument("--use-tally", action="store_true", default=False)
parser.add_argument("--run-pairwise", action="store_true", default=False)
parser.add_argument("--runtime", type=int, default=60)
parser.add_argument("--warmup-iters", type=int, default=100)

args = parser.parse_args()

benchmark_list = {
    "hidet": {
        "resnet50": [64]
    },
    "pytorch": {
        "resnet50": [64],
        "bert": [16],
        "dcgan": [64],
        "LSTM": [64],
        "NeuMF-pre": [64],
        "pointnet": [64],
        "transformer": [8],
        "yolov6n": [8],
        "pegasus-x-base": [1]
    }
}

# Manually specify which pair has tuned all kernel pair configs
prepared_workload_aware = []

# Benchmark options
save_results = args.save_results
use_mps = args.use_mps
use_tally = args.use_tally
run_pairwise = args.run_pairwise
assert(not (use_mps and use_tally))

runtime = args.runtime
warmup_iters = args.warmup_iters

if __name__ == "__main__":

    curr_dir = os.getcwd()
    os.environ["TALLY_HOME"] = f"{curr_dir}/tally"

    if use_tally:
        scheduler_policy = os.environ.get("SCHEDULER_POLICY", "NAIVE")
        print(f"Benchmarking tally with SCHEDULER_POLICY: {scheduler_policy}")

    result = load_json_from_file("result.json")
    # single_job_result, co_locate_result = parse_result("result.json")

    benchmarks = []

    for framework in benchmark_list:
        for model in benchmark_list[framework]:
            for batch_size in benchmark_list[framework][model]:
                for amp in [True, False]:

                    if model in ["transformer"] and amp:
                        continue
                
                    if model in ["yolov6n", "pegasus-x-base"] and not amp:
                        continue

                    bench = Benchmark(framework, model, batch_size, amp, warmup_iters=warmup_iters, runtime=runtime)
                    benchmarks.append(bench)

    init_env(use_mps, use_tally)

    for benchmark in benchmarks:
        launch_benchmark([benchmark], result=result)
        if use_tally:
            launch_benchmark([benchmark], result=result, use_tally=use_tally)
        if save_results:
            write_json_to_file(result, "result.json")
            execute_cmd("cp result.json result_copy.json")

    benchmark_pairs = []

    for i in range(len(benchmarks)):
        for j in range(i, len(benchmarks)):

            bench_1 = benchmarks[i]
            bench_2 = benchmarks[j]

            benchmark_pairs.append([bench_1, bench_2])

    random.shuffle(benchmark_pairs)

    cuda_mem_cap = get_cuda_mem()

    if run_pairwise:
        for pair in benchmark_pairs:

            bench_1, bench_2 = pair
            bench_1_mem = result["tally_naive"][str(bench_1)]["metrics"]["gmem"]
            bench_2_mem = result["tally_naive"][str(bench_2)]["metrics"]["gmem"]
            sum_mem = bench_1_mem + bench_2_mem

            if sum_mem > 0.95 * cuda_mem_cap:
                bench_id = get_bench_id(pair)
                print(f"Skipping {bench_id} as required memory of {sum_mem} MB exceeds system limit of {cuda_mem_cap} MB")
                continue

            launch_benchmark(pair, use_mps=use_mps, use_tally=use_tally, result=result)
            
            if save_results:
                write_json_to_file(result, "result.json")
                execute_cmd("cp result.json result_copy.json")

    tear_down_env()