import sys
import os

sys.path.append('utils')

from utils.util import execute_cmd, write_json_to_file, load_json_from_file
from utils.bench import Benchmark
from utils.bench_util import launch_benchmark, init_env, tear_down_env

benchmark_list = {
    "hidet": {
        "resnet50": [64]
    },
    "pytorch": {
        "resnet50": [64],
        "bert": [16],
        "VGG": [64],
        "dcgan": [64],
        "LSTM": [64],
        "NeuMF-pre": [64],
        "pointnet": [64],
        # "transformer": [8]
    }
}

use_mps = False
use_tally = True
assert(not (use_mps and use_tally))

if __name__ == "__main__":

    curr_dir = os.getcwd()
    os.environ["TALLY_HOME"] = f"{curr_dir}/tally"

    if use_tally:
        scheduler_policy = os.environ.get("SCHEDULER_POLICY", "NAIVE")
        print(f"Benchmarking tally with SCHEDULER_POLICY: {scheduler_policy}")

    result = load_json_from_file("result.json")
    
    benchmarks = []

    for framework in benchmark_list:
        for model in benchmark_list[framework]:
            for batch_size in benchmark_list[framework][model]:
                for amp in [True, False]:

                    if model == "transformer" and amp:
                        continue

                    bench = Benchmark(framework, model, batch_size, amp, warmup_iters=10, runtime=10)
                    benchmarks.append(bench)

    init_env(use_mps, use_tally)

    for benchmark in benchmarks:
        launch_benchmark([benchmark], result=result)
        if use_tally:
            launch_benchmark([benchmark], result=result, use_tally=use_tally)
        write_json_to_file(result, "result.json")
        execute_cmd("cp result.json result_copy.json")

    for i in range(len(benchmarks)):
        for j in range(i, len(benchmarks)):

            bench_1 = benchmarks[i]
            bench_2 = benchmarks[j]

            launch_benchmark([bench_1, bench_2], use_mps=use_mps, use_tally=use_tally, result=result)
            
            write_json_to_file(result, "result.json")
            execute_cmd("cp result.json result_copy.json")

    tear_down_env()