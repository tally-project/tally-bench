import pandas as pd
import sys

sys.path.append('python')

from bench_utils.utils import mkdir_if_not_exists
from bench_utils.plot import (
    plot_motivation_latency_comparison,
    plot_tally_slo_achievable_throughput,
    plot_slo_comparison_seperate_throughput,
    plot_slo_comparison_system_throughput
)


def main():
    plot_directory = "tally_bench_results/plots"
    mkdir_if_not_exists(plot_directory)

    priority_df = pd.read_csv("tally_bench_results/priority-aware-perf.csv")
    high_priority_jobs = priority_df["high_priority_job"].unique()
    best_effort_jobs = priority_df["best_effort_job"].unique()

    metrics = ["avg", "90th", "95th", "99th"]
    tolerance_levels = [0.1]

    # plot Baseline, MPS, Time-sliced latency comparison
    for high_priority_job in high_priority_jobs:
        for metric in metrics:
            plot_motivation_latency_comparison(priority_df, high_priority_job, best_effort_jobs, metric=metric)

    # plot Tally Achievable Throughput under a certain SLO
    for metric in metrics:
        for tolerance_level in tolerance_levels:
            plot_tally_slo_achievable_throughput(priority_df, high_priority_jobs, best_effort_jobs, tolerance_level=tolerance_level, metric=metric)

    # plot Baseline, MPS, Time-sliced, Tally latency and Throughput comparison under a certain SLO
    for high_priority_job in high_priority_jobs:
        for metric in metrics:
            for tolerance_level in tolerance_levels:
                plot_slo_comparison_seperate_throughput(priority_df, high_priority_job, best_effort_jobs, metric=metric, tolerance_level=tolerance_level)
                plot_slo_comparison_system_throughput(priority_df, high_priority_job, best_effort_jobs, metric=metric, tolerance_level=tolerance_level)


if __name__ == "__main__":
    main()