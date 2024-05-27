import os
import math

from bench_utils.utils import write_json_to_file, load_json_from_file
from bench_utils.bench_utils import init_env, tear_down_env
from bench_utils.bench import Benchmark, launch_benchmark, get_train_benchmarks
from bench_utils.tally_config import default_tally_config
from configs.train_config import training_workloads

def bench_azure_trace(
    trace_path="infer_trace/azure_trace_for_bert_one_day.json",
    result_file="azure_result_one_day.json",
    use_mps=False,
    use_mps_priority=False,
    use_tally_priority=False,
):
    tally_bench_result_dir = "tally_bench_results"
    if not os.path.exists(tally_bench_result_dir):
        os.makedirs(tally_bench_result_dir)

    result_file = f"{tally_bench_result_dir}/{result_file}"
    result = load_json_from_file(result_file)

    curr_dir = os.getcwd()
    os.environ["TALLY_HOME"] = f"{curr_dir}/tally"

    use_tally = use_tally_priority
    use_mps = use_mps or use_mps_priority
    init_env(use_mps, use_tally, run_pairwise=True)

    azure_trace = load_json_from_file(trace_path)
    last_ts = azure_trace[-1]
    runtime = int(last_ts)

    train_benchmarks = get_train_benchmarks(training_workloads, warmup_iters=100, runtime=runtime)

    bert_infer_bench = Benchmark("onnxruntime", "bert", warmup_iters=100, runtime=runtime, is_train=False,
                                 batch_size=1, infer_mode="server", infer_load=None)
    bert_infer_bench.trace_file = trace_path
    bert_infer_bench.set_priority(2)

    for train_benchmark in train_benchmarks:

        if "two_week" in trace_path:
            if train_benchmark.model_name != "bert":
                continue

        train_benchmark.set_priority(1)

        benchmarks = [train_benchmark, bert_infer_bench]
        
        updated = False

        # Profile single-job performance
        for benchmark in benchmarks:
            updated |= launch_benchmark([benchmark], result=result, truncate_result=False, keep_trace=True)

        if use_tally_priority:
            updated |= launch_benchmark(benchmarks, use_mps, use_mps_priority, use_tally=use_tally_priority,
                                    result=result, tally_config=default_tally_config, truncate_result=False, keep_trace=True)
        else:
            updated |= launch_benchmark(benchmarks, use_mps, use_mps_priority, use_tally=use_tally_priority,
                                        result=result, truncate_result=False, keep_trace=True)
        if updated:
            write_json_to_file(result, result_file)

    tear_down_env()
