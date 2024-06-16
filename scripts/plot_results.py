import pandas as pd
import sys

sys.path.append('python')

from bench_utils.utils import mkdir_if_not_exists
from bench_utils.plot import (
    plot_motivation_latency_comparison,
    plot_tally_slo_achievable_throughput,
    plot_slo_comparison_seperate_throughput,
    plot_slo_comparison_system_throughput,
    plot_slo_comparison_tally_sensitivity,
    plot_slo_comparison_system_throughput_all,
    plot_throughput_vs_load,
    plot_latency_vs_load,
    plot_varying_load
)


def main():
    plot_directory = "tally_bench_results/plots"
    mkdir_if_not_exists(plot_directory)

    priority_df = pd.read_csv("tally_bench_results/priority-aware-perf.csv")
    high_priority_jobs = priority_df["high_priority_job"].unique()
    high_priority_jobs = [high_priority_job for high_priority_job in high_priority_jobs if "server" in high_priority_job]
    best_effort_jobs = priority_df["best_effort_job"].unique()

    # metrics = ["avg", "90th", "95th", "99th"]
    metrics = ["99th"]
    # tolerance_levels = [0.10]
    varying_loads = [0.1, 0.3, 0.5, 0.7, 0.9]

    # # plot Baseline, MPS, Time-sliced latency comparison
    # for high_priority_job in high_priority_jobs:
    #     for metric in metrics:
    #         plot_motivation_latency_comparison(priority_df, high_priority_job, best_effort_jobs, metric=metric)

    # # plot Tally Achievable Throughput under a certain SLO
    # for metric in metrics:
    #     for tolerance_level in tolerance_levels:
    #         plot_tally_slo_achievable_throughput(priority_df, high_priority_jobs, best_effort_jobs, tolerance_level=tolerance_level, metric=metric)

    # for metric in metrics:
    #     plot_slo_comparison_system_throughput_all(priority_df, high_priority_jobs, best_effort_jobs, metric=metric)

    # plot Baseline, MPS, Time-sliced, Tally latency and Throughput comparison under a certain SLO
    for high_priority_job in high_priority_jobs:

        if "load_0.5" not in high_priority_job:
            continue
        
        for metric in metrics:
            # plot_slo_comparison_seperate_throughput(priority_df, high_priority_job, best_effort_jobs, metric=metric)
            plot_slo_comparison_system_throughput(priority_df, high_priority_job, best_effort_jobs, metric=metric)
            plot_slo_comparison_tally_sensitivity(priority_df, high_priority_job, best_effort_jobs, metric=metric)

    plot_throughput_vs_load(priority_df, "onnxruntime_bert", best_effort_jobs, varying_loads)
    plot_throughput_vs_load(priority_df, "onnxruntime_llama-2-7b", best_effort_jobs, varying_loads)

    plot_latency_vs_load(priority_df, "onnxruntime_bert", best_effort_jobs, varying_loads)
    plot_latency_vs_load(priority_df, "onnxruntime_llama-2-7b", best_effort_jobs, varying_loads)

    # plot_varying_load(priority_df, "onnxruntime_bert", best_effort_jobs, varying_loads)
    # plot_varying_load(priority_df, "onnxruntime_llama-2-7b", best_effort_jobs, varying_loads)


if __name__ == "__main__":
    main()