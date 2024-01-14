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
from workloads.configs.infer_config import inference_workloads

parser = argparse.ArgumentParser(prog="benchmark suite launcher", description="Launch benchmark suite")

parser.add_argument("--save-results", action="store_true", default=False)
parser.add_argument("--use-mps", action="store_true", default=False)
parser.add_argument("--use-tally", action="store_true", default=False)
parser.add_argument("--run-pairwise", action="store_true", default=False)
parser.add_argument("--runtime", type=int, default=10)
parser.add_argument("--warmup-iters", type=int, default=100)
parser.add_argument("--profile-only", action="store_true", default=False)

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

result_file = f"{tally_bench_result_dir}/result.json"
result_backup_file = f"{tally_bench_result_dir}/result_backup.json"


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
    # single_job_result, co_locate_result = parse_result(result_file)

    train_benchmarks = get_train_benchmarks(training_workloads, warmup_iters, runtime)
    infer_benchmarks = get_infer_benchmarks(inference_workloads, warmup_iters, runtime)
    all_benchmarks = train_benchmarks + infer_benchmarks

    train_pairs = []
    train_infer_pairs = []

    for i in range(len(train_benchmarks)):

        for j in range(i, len(train_benchmarks)):
            train_pairs.append((copy.copy(train_benchmarks[i]), copy.copy(train_benchmarks[j])))
        
        for j in range(len(infer_benchmarks)):
            train_infer_pairs.append((copy.copy(train_benchmarks[i]), copy.copy(infer_benchmarks[j])))

    # random.shuffle(train_pairs)

    init_env(use_mps, use_tally)

    # Run single-job benchmark
    for idx, benchmark in enumerate(all_benchmarks):

        bench_id = get_bench_id((benchmark,))
        logger.info(f"Running {idx + 1} out of {len(all_benchmarks)} single-job benchmarks: {bench_id} ...")

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

        updated = launch_benchmark([benchmark], result=result)
        if use_tally:
            updated |= launch_benchmark((benchmark, ), result=result, use_tally=use_tally, profile_only=args.profile_only)
        if updated and save_results:
            write_json_to_file(result, result_file)
            write_json_to_file(result, result_backup_file)

    # Run pairwise training benchmark
    if run_pairwise:

        all_pairs = train_pairs + train_infer_pairs

        for idx, pair in enumerate(all_pairs):

            bench_1, bench_2 = pair
            bench_id = get_bench_id(pair)

            logger.info(f"Running {idx + 1} out of {len(all_pairs)} pairwise benchmarks: {bench_id} ...")

            bench_1_mem = result["tally_naive"][str(bench_1)]["metrics"]["gmem"]
            bench_2_mem = result["tally_naive"][str(bench_2)]["metrics"]["gmem"]
            sum_mem = bench_1_mem + bench_2_mem

            if sum_mem > 0.95 * cuda_mem_cap:
                logger.info(f"Skipping {bench_id} as required memory of {sum_mem} MB exceeds system limit of {cuda_mem_cap} MB")
                continue

            assert(not bench_1.is_latency_critical())
            is_latency_critical = bench_2.is_latency_critical()

            if not use_tally and is_latency_critical:
                logger.info(f"Skipping {bench_id} for latency-critical tasks")
                continue

            # Do not run train_infer pairs under workload-agnostic-sharing
            if use_tally:

                if scheduler_policy == "WORKLOAD_AGNOSTIC_SHARING":
                    if is_latency_critical:
                        logger.info(f"Skipping {bench_id} as workload-agnostic-sharing scheduler does not apply to latency-critical tasks")
                        continue

                if scheduler_policy == "PRIORITY":
                    if not is_latency_critical:
                        logger.info(f"Skipping {bench_id} as we only run priority scheduler on LC/BE pair for now")
                        continue

                    assert(bench_1.is_train)

                    # let bench 2 be high-priority job
                    bench_1.set_priority(1)
                    bench_2.set_priority(2)

                    # launch_benchmark(pair, use_mps=use_mps, use_tally=use_tally, result=result)

                    # # if both are training jobs, let bench 1 be high-priority job as well
                    # if pair in train_pairs:
                    #     bench_1.set_priority(2)
                    #     bench_2.set_priority(1)

                    #     reverse_pair = (bench_2, bench_1)
                    #     launch_benchmark(reverse_pair, use_mps=use_mps, use_tally=use_tally, result=result)

                    # continue
            
            updated = launch_benchmark(pair, use_mps=use_mps, use_tally=use_tally, result=result)
            
            if updated and save_results:
                write_json_to_file(result, result_file)
                write_json_to_file(result, result_backup_file)

    tear_down_env()