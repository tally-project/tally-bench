import sys
import os
import random
import argparse
import copy

sys.path.append('.')

from utils.util import load_json_from_file, write_json_to_file, execute_cmd, logger
from utils.bench import launch_benchmark, get_train_benchmarks, get_infer_benchmarks
from utils.bench_util import init_env, tear_down_env, get_bench_id
from utils.parse import parse_result
from utils.nvidia_smi import get_cuda_mem
from workloads.configs.train_config import training_workloads
from workloads.configs.infer_config import inference_workloads, inference_load_factors

parser = argparse.ArgumentParser(prog="benchmark suite launcher", description="Launch benchmark suite")

parser.add_argument("--save-results", action="store_true", default=False)
parser.add_argument("--use-mps", action="store_true", default=False)
parser.add_argument("--use-tally", action="store_true", default=False)
parser.add_argument("--run-pairwise", action="store_true", default=False)
parser.add_argument("--runtime", type=int, default=10)
parser.add_argument("--warmup-iters", type=int, default=100)
parser.add_argument("--profile-only", action="store_true", default=False)
parser.add_argument("--run-priority", action="store_true", default=False)
parser.add_argument("--run-throughput", action="store_true", default=False)

args = parser.parse_args()

# Benchmark options
save_results = args.save_results
use_mps = args.use_mps
use_tally = args.use_tally
run_pairwise = args.run_pairwise
run_priority = args.run_priority
run_throughput = args.run_throughput
assert(not (use_mps and use_tally))

runtime = args.runtime
warmup_iters = args.warmup_iters

tally_bench_result_dir = "tally-bench-results"
if not os.path.exists(tally_bench_result_dir):
    os.makedirs(tally_bench_result_dir)

result_file = f"{tally_bench_result_dir}/result.json"
result_backup_file = f"{tally_bench_result_dir}/result_backup.json"

tally_priority_preemption_limits = [0.01, 0.1, 1.0]

def save_results(result, updated, save_results):
    if updated and save_results:
        write_json_to_file(result, result_file)
        write_json_to_file(result, result_backup_file)

if __name__ == "__main__":

    # cuda memory capacity
    cuda_mem_cap = get_cuda_mem()

    curr_dir = os.getcwd()
    os.environ["TALLY_HOME"] = f"{curr_dir}/tally"
    scheduler_policy = None

    if use_tally:
        scheduler_policy = os.environ.get("SCHEDULER_POLICY", "NAIVE")
        logger.info(f"Benchmarking tally with SCHEDULER_POLICY: {scheduler_policy}")

    result = load_json_from_file(result_file)

    train_benchmarks = get_train_benchmarks(training_workloads, warmup_iters, runtime)
    infer_benchmarks = get_infer_benchmarks(inference_workloads, inference_load_factors, warmup_iters, runtime)
    
    init_env(use_mps, use_tally)

    # Prepare single job benchmarks
    single_job_benchmarks = []
    for idx, benchmark in enumerate(train_benchmarks + infer_benchmarks):
        
        if scheduler_policy == "WORKLOAD_AGNOSTIC_SHARING":
            if benchmark.is_latency_critical():
                continue

        if scheduler_policy == "PRIORITY":

            # no need to measure throughput-oriented jobs
            if not args.profile_only and not benchmark.is_latency_critical():
                continue
                
            # can skip profiling server because it is the same kernels as single-stream
            if args.profile_only and benchmark.infer_mode == "server":
                continue

            if args.profile_only and benchmark.is_latency_critical():
                continue
        
        single_job_benchmarks.append(benchmark)

    # Run single-job benchmark
    for idx, benchmark in enumerate(single_job_benchmarks):

        bench_id = get_bench_id((benchmark,))
        logger.info(f"Running {idx + 1} out of {len(single_job_benchmarks)} single-job benchmarks: {bench_id} ...")

        updated = launch_benchmark([benchmark], result=result)
        if use_tally:
            updated |= launch_benchmark((benchmark,), result=result, use_tally=use_tally,
                                        profile_only=args.profile_only, preemption_limit=min(tally_priority_preemption_limits))
        save_results(result, updated, save_results)

    # Prepare pairwise benchmarks
    pair_wise_benchmarks = []

    # if run_priority - run pairwise experiments between inference and training pairs
    #                   such that inference is high-priority and training is best-effort
    #                   Potentially we can also run priority on training and training pairs
    #                   Given that training generally consumes a significant amount of memory,
    #                   we won't consider that for now
    if run_priority:
        assert(not (use_tally and scheduler_policy == "WORKLOAD_AGNOSTIC_SHARING"))
        for j in range(len(infer_benchmarks)):
            for i in range(len(train_benchmarks)):
                pair_wise_benchmarks.append((copy.copy(train_benchmarks[i]), copy.copy(infer_benchmarks[j])))

    # if throughput - run pairwise experiments between training and training pairs
    if run_throughput:
        assert(not (use_tally and scheduler_policy == "PRIORITY"))
        for i in range(len(train_benchmarks)):
            for j in range(i, len(train_benchmarks)):
                pair_wise_benchmarks.append((copy.copy(train_benchmarks[i]), copy.copy(train_benchmarks[j])))

    # Run pairwise training benchmark
    if run_pairwise:
        for idx, pair in enumerate(pair_wise_benchmarks):

            bench_1, bench_2 = pair
            bench_id = get_bench_id(pair)

            logger.info(f"Running {idx + 1} out of {len(pair_wise_benchmarks)} pairwise benchmarks: {bench_id} ...")

            bench_1_mem = result["tally_naive"][str(bench_1)]["metrics"]["gmem"]
            bench_2_mem = result["tally_naive"][str(bench_2)]["metrics"]["gmem"]
            sum_mem = bench_1_mem + bench_2_mem

            if sum_mem > 0.95 * cuda_mem_cap:
                logger.info(f"Skipping {bench_id} as required memory of {sum_mem} MB exceeds system limit of {cuda_mem_cap} MB")
                continue

            assert(not bench_1.is_latency_critical())
 
            # Do not run train_infer pairs under workload-agnostic-sharing
            if use_tally and scheduler_policy == "PRIORITY":

                    # let bench 2 be high-priority job
                    bench_1.set_priority(1)
                    bench_2.set_priority(2)

                    updated = False
                    for limit in tally_priority_preemption_limits:
                        updated |= launch_benchmark(pair, use_mps=use_mps, use_tally=use_tally, result=result, preemption_limit=limit)

                    save_results(result, updated, save_results)
                    continue
            
            updated = launch_benchmark(pair, use_mps=use_mps, use_tally=use_tally, result=result)
            save_results(result, updated, save_results)

    tear_down_env()