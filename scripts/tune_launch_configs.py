import sys
import math
import matplotlib.pyplot as plt

sys.path.append('python')

from bench_utils.tally import TallyConfig
from bench_utils.utils import load_json_from_file, write_json_to_file, compute_percentile, mkdir_if_not_exists
from bench_utils.bench import Benchmark, launch_benchmark, get_train_benchmarks
from configs.train_config import training_workloads
from bench_utils.bench_utils import get_bench_id
from bench_utils.bench import Benchmark, launch_benchmark, get_train_benchmarks, get_infer_benchmark_trace, get_infer_benchmark_latency


def tune_launch_config(
    result_file, infer_framework, infer_model, tally_configs, parameter,
    warmup_iters=30, runtime=30, plot_save_dir="tally_results/plots/", x_axis_label=""
):

    result = load_json_from_file(result_file)

    train_benchmarks = get_train_benchmarks(training_workloads, warmup_iters, runtime)
    infer_benchmark = Benchmark(framework=infer_framework, model_name=infer_model, warmup_iters=30,
                                runtime=runtime, is_train=False, batch_size=1, infer_mode="server",
                                infer_load=0.5)
    
    bench_id = get_bench_id([infer_benchmark])
    trace_path = f"infer_trace/{bench_id}.json"

    trace = get_infer_benchmark_trace(infer_benchmark, result, trace_path)
    infer_benchmark.trace_file = trace_path

    trace_last_ts = trace[-1]
    runtime = math.ceil(trace_last_ts)

    tune_res = {}

    for train_benchmark in train_benchmarks:

        train_bench_id = get_bench_id([train_benchmark])
        tune_res[train_bench_id] = {}

        train_benchmark.set_priority(1)
        infer_benchmark.set_priority(2)

        train_benchmark.runtime = runtime
        infer_benchmark.runtime = runtime

        benchmarks = [train_benchmark, infer_benchmark]

        updated = False
        
        for config in tally_configs:
            updated |= launch_benchmark(benchmarks, use_tally=True,
                                        result=result, tally_config=config, truncate_result=True)
        
        if updated:
            write_json_to_file(result, result_file)
        
        benchmarks = [train_benchmark, infer_benchmark]

        bench_id = get_bench_id(benchmarks)
        train_bench_id = get_bench_id((train_benchmark,))
        infer_bench_id = get_bench_id((infer_benchmark,))
        measurements = result["tally_priority"][bench_id]["measurements"]

        train_baseline_res = result["default"][train_bench_id]["measurements"][0][f"{train_bench_id}_0"]
        train_baseline_throughput = train_baseline_res["iters"] / train_baseline_res["time_elapsed"]

        infer_99th_latency_list = []
        train_throughput_list = []
        parameter_list = []

        for tally_config in tally_configs:
            for measurement in measurements:
                config = measurement["tally_config"]
                if config == tally_config.to_dict():
                    break

            train_res = measurement[f"{train_bench_id}_0"]
            infer_res = measurement[f"{infer_bench_id}_1"]

            infer_99th_latency = compute_percentile(infer_res["latencies"], 99)
            train_throughput = (train_res["iters"] / train_res["time_elapsed"]) / train_baseline_throughput

            parameter_val = getattr(tally_config, parameter)

            infer_99th_latency_list.append(infer_99th_latency)
            train_throughput_list.append(train_throughput)
            parameter_list.append(parameter_val)
    
        tune_res[train_bench_id]["latencies"] = infer_99th_latency_list
        tune_res[train_bench_id]["throughputs"] = train_throughput_list

        plt.clf()
        plt.figure(figsize=(10, 6))

        for best_effort_job in tune_res:

            latencies = tune_res[best_effort_job]["latencies"]
            plt.plot(parameter_list, latencies, label=best_effort_job, marker='o')

        plt.xscale('log')
        plt.xlabel(x_axis_label)
        plt.ylabel('99th Percentile Latency (ms)')

        plt.legend()
        # plt.grid(True)

        plt.savefig(f"{plot_save_dir}/{infer_bench_id}_latency.png")

        plt.clf()
        plt.figure(figsize=(10, 6))

        for best_effort_job in tune_res:

            throughputs = tune_res[best_effort_job]["throughputs"]
            plt.plot(parameter_list, throughputs, label=best_effort_job, marker='o')

        plt.xscale('log')
        plt.xlabel(x_axis_label)
        plt.ylabel('Normalized Throughput')

        plt.legend()
        # plt.grid(True)

        plt.savefig(f"{plot_save_dir}/{infer_bench_id}_throughput.png")

        # fig, ax1 = plt.subplots()

        # line1, = ax1.plot(parameter_list, infer_99th_latency_list, color='tab:blue', linestyle='dotted', marker="o", label='Latency')
        # ax1.set_xlabel(x_axis_label)
        # ax1.set_ylabel('99th Percentile Latency (ms)')
        # ax1.tick_params(axis='y')
        # ax1.set_xscale('log')

        # ax2 = ax1.twinx()
        # line2, = ax2.plot(parameter_list, train_throughput_list, color='tab:orange', linestyle='dashed', marker="s", label='Throughput')
        # ax2.set_ylabel('Normalized Throughput')
        # ax2.tick_params(axis='y')

        # fig.tight_layout()

        # lines = [line1, line2]
        # labels = [line.get_label() for line in lines]
        # plt.legend(lines, labels, loc='upper left')

        # plt.savefig(f"{plot_save_dir}/{bench_id}.png")


if __name__ == "__main__":

    result_file = f"tally_results/result.json"
    warmup_iters = 30
    runtime = 30

    # 1. tune max_allowed_latency for bert inference
    plot_save_dir = "tally_results/plots/preemption_latency_tuning"
    mkdir_if_not_exists(plot_save_dir)

    parameter = "max_allowed_latency"

    tally_configs = [
        TallyConfig("priority", max_allowed_latency=0.01),    # 10^-2
        TallyConfig("priority", max_allowed_latency=0.0316),  # 10^-2 * sqrt(10)
        TallyConfig("priority", max_allowed_latency=0.1),     # 10^-1
        TallyConfig("priority", max_allowed_latency=0.316),   # 10^-1 * sqrt(10)
        TallyConfig("priority", max_allowed_latency=1),       # 10^0
        TallyConfig("priority", max_allowed_latency=3.16),    # 10^0  * sqrt(10)
    ]

    infer_benchmarks = [
        ("onnxruntime", "bert"),
        ("hidet", "resnet50"),
        ("pytorch", "yolov6m"),
    ]

    for framework, model in infer_benchmarks:
        tune_launch_config(
            result_file, framework, model, tally_configs, parameter, warmup_iters=warmup_iters,
            runtime=runtime, plot_save_dir=plot_save_dir, x_axis_label='Preemption Latency (ms)'
        )