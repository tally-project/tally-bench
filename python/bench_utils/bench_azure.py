import os
import math

from bench_utils.utils import write_json_to_file, load_json_from_file
from bench_utils.bench_utils import init_env, tear_down_env
from bench_utils.bench import Benchmark, launch_benchmark
from bench_utils.tally import TallyConfig

def bench_azure_trace(
    use_mps=False,
    use_mps_priority=False,
    use_tally_priority=False,
):
    tally_bench_result_dir = "tally_bench_results"
    if not os.path.exists(tally_bench_result_dir):
        os.makedirs(tally_bench_result_dir)

    result_file = f"{tally_bench_result_dir}/azure_result.json"
    result = load_json_from_file(result_file)

    curr_dir = os.getcwd()
    os.environ["TALLY_HOME"] = f"{curr_dir}/tally"

    use_tally = use_tally_priority
    use_mps = use_mps or use_mps_priority
    init_env(use_mps, use_tally, run_pairwise=True)

    trace_path = f"infer_trace/azure_trace_for_bert.json"
    azure_trace = load_json_from_file(trace_path)
    last_ts = azure_trace[-1]
    runtime = math.ceil(last_ts)

    bert_infer_bench = Benchmark("onnxruntime", "bert", warmup_iters=30, runtime=runtime, is_train=False,
                                 batch_size=1, infer_mode="server", infer_load=None)
    bert_infer_bench.trace_file = trace_path
    bert_infer_bench.set_priority(2)

    training_bench = Benchmark("pytorch", "bert", warmup_iters=30, runtime=runtime, is_train=True,
                               batch_size=32, amp=False)
    training_bench.set_priority(1)

    benchmarks = [bert_infer_bench, training_bench]
    
    updated = False

    # Profile single-job performance
    for benchmark in benchmarks:
        updated |= launch_benchmark([benchmark], result=result, truncate_result=False, keep_trace=True)

    if use_tally_priority:
        tally_configs = [
            TallyConfig(scheduler_policy="priority", max_allowed_latency=0.1),
            TallyConfig(scheduler_policy="priority", max_allowed_latency=0.1, wait_time_to_use_original=100)
        ]

        for tally_config in tally_configs:
            updated |= launch_benchmark(benchmarks, use_mps, use_mps_priority, use_tally=use_tally_priority,
                                       result=result, tally_config=tally_config, truncate_result=False, keep_trace=True)
    else:
        updated |= launch_benchmark(benchmarks, use_mps, use_mps_priority, use_tally=use_tally_priority,
                                    result=result, truncate_result=False, keep_trace=True)
    
    if updated:
        write_json_to_file(result, result_file)

    tear_down_env()
