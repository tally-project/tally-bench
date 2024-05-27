import sys
import matplotlib.pyplot as plt

sys.path.append('python')

from bench_utils.tally import TallyConfig
from bench_utils.utils import load_json_from_file, write_json_to_file, compute_percentile, mkdir_if_not_exists
from bench_utils.bench import Benchmark, launch_benchmark, get_train_benchmarks
from configs.train_config import training_workloads
from bench_utils.bench_utils import get_bench_id


def tune_launch_config(
    result_file, infer_frameowork, infer_model, tally_configs, parameter,
    warmup_iters=30, runtime=30, plot_save_dir="tally_bench_results/plots/", x_axis_label=""
):

    result = load_json_from_file(result_file)

    train_benchmarks = get_train_benchmarks(training_workloads, warmup_iters, runtime)
    infer_benchmark = Benchmark(framework=infer_frameowork, model_name=infer_model, warmup_iters=30,
                                runtime=runtime, is_train=False, batch_size=1, infer_mode="server",
                                infer_load=0.5)
    
    for train_benchmark in train_benchmarks:

        train_benchmark.set_priority(1)
        infer_benchmark.set_priority(2)

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

        infer_95th_latency_list = []
        train_throughput_list = []
        parameter_list = []

        for tally_config in tally_configs:
            for measurement in measurements:
                config = measurement["tally_config"]
                if config == tally_config.to_dict():
                    break

            train_res = measurement[f"{train_bench_id}_0"]
            infer_res = measurement[f"{infer_bench_id}_1"]

            infer_95th_latency = compute_percentile(infer_res["latencies"], 95)
            train_throughput = (train_res["iters"] / train_res["time_elapsed"]) / train_baseline_throughput

            parameter_val = getattr(tally_config, parameter)
            
            tally_config.max_allowed_latency

            infer_95th_latency_list.append(infer_95th_latency)
            train_throughput_list.append(train_throughput)
            parameter_list.append(parameter_val)
        
        fig, ax1 = plt.subplots()

        line1, = ax1.plot(parameter_list, infer_95th_latency_list, color='tab:blue', linestyle='dotted', marker="o", label='Latency')
        ax1.set_xlabel(x_axis_label)
        ax1.set_ylabel('95th Percentile Latency (ms)')
        ax1.tick_params(axis='y')
        ax1.set_xscale('log')

        ax2 = ax1.twinx()
        line2, = ax2.plot(parameter_list, train_throughput_list, color='tab:orange', linestyle='dashed', marker="s", label='Throughput')
        ax2.set_ylabel('Normalized Throughput')
        ax2.tick_params(axis='y')

        fig.tight_layout()

        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        plt.legend(lines, labels, loc='upper left')

        plt.savefig(f"{plot_save_dir}/{bench_id}.png")


if __name__ == "__main__":

    result_file = f"tally_bench_results/result.json"
    warmup_iters = 30
    runtime = 30

    # 1. tune max_allowed_latency for bert inference
    plot_save_dir = "tally_bench_results/plots/preemption_latency_tuning"
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

    tune_launch_config(
        result_file, "onnxruntime", "bert", tally_configs, parameter, warmup_iters=warmup_iters,
        runtime=runtime, plot_save_dir=plot_save_dir, x_axis_label='Preemption Latency (ms)'
    )

    # 2. tune min_wait_time for llama inference
    plot_save_dir = "tally_bench_results/plots/wait_time_tuning"
    mkdir_if_not_exists(plot_save_dir)
    
    parameter = "min_wait_time"

    tally_configs = [
        TallyConfig("priority", use_original_configs=True, min_wait_time=0.1),
        TallyConfig("priority", use_original_configs=True, min_wait_time=0.316),
        TallyConfig("priority", use_original_configs=True, min_wait_time=1.0),
        TallyConfig("priority", use_original_configs=True, min_wait_time=3.16),
        TallyConfig("priority", use_original_configs=True, min_wait_time=10),
        TallyConfig("priority", use_original_configs=True, min_wait_time=31.6),
        TallyConfig("priority", use_original_configs=True, min_wait_time=100),
    ]

    tune_launch_config(
        result_file, "onnxruntime", "llama-2-7b", tally_configs, parameter, warmup_iters=warmup_iters,
        runtime=runtime, plot_save_dir=plot_save_dir, x_axis_label='Min Wait Time (ms)'
    )