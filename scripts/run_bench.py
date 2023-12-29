import sys
import os
import random

sys.path.append('.')

from utils.util import load_json_from_file, write_json_to_file, execute_cmd
from utils.bench import Benchmark, launch_benchmark
from utils.bench_util import init_env, tear_down_env
from utils.parse import parse_result

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
save_results = False
use_mps = False
use_tally = True
assert(not (use_mps and use_tally))

runtime = 60
warmup_iters = 100

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

    # benchmark_pairs = []

    # for i in range(len(benchmarks)):
    #     for j in range(i, len(benchmarks)):

    #         bench_1 = benchmarks[i]
    #         bench_2 = benchmarks[j]

    #         benchmark_pairs.append([bench_1, bench_2])

    # random.shuffle(benchmark_pairs)

    # if use_tally or use_mps:

    #     for pair in benchmark_pairs:

    #         # Only run workload aware scheduler if the tally overhead is small (i.e. GPU bounded jobs)
    #         if use_tally and scheduler_policy == "WORKLOAD_AWARE_SHARING":

    #             model_1_norm_speed, model_2_norm_speed = None, None

    #             for res in single_job_result:
    #                 if res["model"] == str(pair[0]):
    #                     model_1_norm_speed = res["tally_workload_aware"]
    #                 if res["model"] == str(pair[1]):
    #                     model_2_norm_speed = res["tally_workload_aware"]

    #             assert(model_1_norm_speed and model_2_norm_speed)

    #             if model_1_norm_speed < 0.7 or model_2_norm_speed < 0.7:
    #                 print(f"Skipping workload-aware benchmark for {pair[0]} and {pair[1]}")
    #                 print(f"Norm speeds: {pair[0]}: {model_1_norm_speed} {pair[1]}: {model_2_norm_speed}")
    #                 continue

    #             bench_id = get_bench_id(pair)

    #             # Skip already prepared model pairs
    #             if not save_results:
    #                 if bench_id in prepared_workload_aware:
    #                     print(f"Skipping workload-aware benchmark for {pair[0]} and {pair[1]} as all kernel pairs have been tuned")
    #                     continue
    #             else:
    #                 if bench_id not in prepared_workload_aware:
    #                     print(f"Skipping workload-aware benchmark for {pair[0]} and {pair[1]} as not all kernel pairs have been tuned")
    #                     continue

    #         launch_benchmark(pair, use_mps=use_mps, use_tally=use_tally, result=result)
            
    #         if save_results:
    #             write_json_to_file(result, "result.json")
    #             execute_cmd("cp result.json result_copy.json")

    tear_down_env()