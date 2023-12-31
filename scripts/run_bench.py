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
from workloads.configs.train_config import training_list

parser = argparse.ArgumentParser(prog="benchmark suite launcher", description="Launch benchmark suite")

parser.add_argument("--save-results", action="store_true", default=False)
parser.add_argument("--use-mps", action="store_true", default=False)
parser.add_argument("--use-tally", action="store_true", default=False)
parser.add_argument("--run-pairwise", action="store_true", default=False)
parser.add_argument("--runtime", type=int, default=60)
parser.add_argument("--warmup-iters", type=int, default=100)

args = parser.parse_args()

# Benchmark options
save_results = args.save_results
use_mps = args.use_mps
use_tally = args.use_tally
run_pairwise = args.run_pairwise
assert(not (use_mps and use_tally))

runtime = args.runtime
warmup_iters = args.warmup_iters

tally_bench_result_dir = "tally-bench-results"
if not os.path.exists(tally_bench_result_dir):
    os.makedirs(tally_bench_result_dir)

train_result_file = f"{tally_bench_result_dir}/train_result.json"
infer_result_file = f"{tally_bench_result_dir}/infer_result.json"

train_result_backup_file = f"{tally_bench_result_dir}/train_result_backup.json"
infer_result_backup_file = f"{tally_bench_result_dir}/infer_result_backup.json"

if __name__ == "__main__":

    # cuda memory capacity
    cuda_mem_cap = get_cuda_mem()

    curr_dir = os.getcwd()
    os.environ["TALLY_HOME"] = f"{curr_dir}/tally"

    if use_tally:
        scheduler_policy = os.environ.get("SCHEDULER_POLICY", "NAIVE")
        print(f"Benchmarking tally with SCHEDULER_POLICY: {scheduler_policy}")

    train_result = load_json_from_file(train_result_file)
    infer_result = load_json_from_file(infer_result_file)

    single_job_result, co_locate_result = parse_result(train_result_file)

    train_benchmarks = []
    benchmark_pairs = []

    for framework in training_list:
        for model in training_list[framework]:

            bench_config = training_list[framework][model]

            for batch_size in bench_config["batch-sizes"]:
                for amp in bench_config["amp"]:

                    bench = Benchmark(framework, model, batch_size, amp, warmup_iters=warmup_iters, runtime=runtime)
                    train_benchmarks.append(bench)

    for i in range(len(train_benchmarks)):
        for j in range(i, len(train_benchmarks)):

            bench_1 = train_benchmarks[i]
            bench_2 = train_benchmarks[j]

            benchmark_pairs.append([bench_1, bench_2])

    # random.shuffle(benchmark_pairs)

    init_env(use_mps, use_tally)

    # Run single-job training benchmark
    for benchmark in train_benchmarks:
        launch_benchmark([benchmark], result=train_result)
        if use_tally:
            launch_benchmark([benchmark], result=train_result, use_tally=use_tally)
        if save_results:
            write_json_to_file(train_result, train_result_file)
            write_json_to_file(train_result, train_result_backup_file)

    # Run pairwise training benchmark
    if run_pairwise:
        for idx, pair in enumerate(benchmark_pairs):

            print(f"Running {idx + 1} out of {len(benchmark_pairs)} train_benchmarks ...")

            bench_1, bench_2 = pair
            bench_1_mem = train_result["tally_naive"][str(bench_1)]["metrics"]["gmem"]
            bench_2_mem = train_result["tally_naive"][str(bench_2)]["metrics"]["gmem"]
            sum_mem = bench_1_mem + bench_2_mem

            if sum_mem > 0.95 * cuda_mem_cap:
                bench_id = get_bench_id(pair)
                print(f"Skipping {bench_id} as required memory of {sum_mem} MB exceeds system limit of {cuda_mem_cap} MB")
                continue

            launch_benchmark(pair, use_mps=use_mps, use_tally=use_tally, result=train_result)
            
            if save_results:
                write_json_to_file(train_result, train_result_file)
                write_json_to_file(train_result, train_result_backup_file)

    tear_down_env()