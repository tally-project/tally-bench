import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import re

sys.path.append('python')

from bench_utils.utils import mkdir_if_not_exists
from bench_utils.parse import get_slo_comparison_data
from bench_utils.plot import (
    colors,
    get_high_priority_job_label_custom,
    get_best_effort_job_label_custom,
    tally_default_config,
    plot_slo_comparison_tally_sensitivity
)

def plot_slo_comparison_system_throughput_combined(priority_df, high_priority_jobs, best_effort_jobs, metric="avg", out_directory="tally_bench_results/plots"):

    savefig_dir = f"{out_directory}/slo_comparison_system_throughput_combined/{metric}"
    mkdir_if_not_exists(savefig_dir)

    systems = ["time_sliced", "mps", "mps_priority", "tgs", "tally"]
    
    data = {}
    used_best_effort_jobs = None

    for high_priority_job in high_priority_jobs:
        data[high_priority_job] = get_slo_comparison_data(priority_df, high_priority_job, best_effort_jobs, tally_default_config, metric)
        job_data = data[high_priority_job]
        used_best_effort_jobs = job_data["used_best_effort_jobs"]

        for system in systems:
            data[high_priority_job][f"{system}_system_throughputs"] = [x + y for x, y in zip(job_data[f"priority_{system}_throughputs"], job_data[f"{system}_throughputs"])]

    assert(len(high_priority_jobs) == 2)

    # plotting
    plt.clf()
    fig, axs = plt.subplots(2, 2, figsize=(30, 20), sharex=True)

    # Width of a bar
    width = 0.13
    
    pos = np.arange(len(used_best_effort_jobs))

    for i in range(2):

        ax1 = axs[0, i]
        ax2 = axs[1, i]
        idx = i
        high_priority_job = high_priority_jobs[idx]

        job_data = data[high_priority_jobs[idx]]
        baseline_latencies = job_data["baseline_latencies"]

        # plot latency
        ax1.bar(pos, baseline_latencies, width, label=f"Baseline", color=colors[0], alpha=0.7, edgecolor='black')
        for k, system in enumerate(systems):
            ax1.bar(pos + (k+1) * width, job_data[f"{system}_latencies"], width, label=system, color=colors[k+1], alpha=0.7, edgecolor='black')
        
        upper_title = get_high_priority_job_label_custom(high_priority_job)
        ax1.set_title(upper_title, fontsize=30, pad=15)
        ax1.set_ylabel(f"99th-Percentile Latency (ms)", fontsize=15)
        plt.setp(ax1.get_yticklabels(), fontsize=15)

        # plot system throughput
        for k, system in enumerate(systems):
            ax2.bar(pos + (k+1) * width, job_data[f"{system}_system_throughputs"], width, label=system, color=colors[k+1], alpha=0.7, edgecolor='black')
        
        ax2.set_ylabel(f"System Throughput", fontsize=15)

        num_targets = 5
        ax2.set_xticks(pos + ((num_targets - 1) / 2) * width)
        plt.setp(ax2.get_yticklabels(), fontsize=15)
        ax2.set_xticklabels([get_best_effort_job_label_custom(best_effort_job) for best_effort_job in used_best_effort_jobs], fontsize=20)
        ax2.tick_params(axis='x', which='major', pad=15)

    # Add a legend at the top of the figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=len(systems) + 1, fontsize=25)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    plt.savefig(f"{savefig_dir}/end-to-end-results.png")


def main():
    plot_directory = "tally_bench_results/plots"
    mkdir_if_not_exists(plot_directory)

    priority_df = pd.read_csv("tally_bench_results/priority-aware-perf.csv")
    high_priority_jobs = priority_df["high_priority_job"].unique()
    high_priority_jobs = [high_priority_job for high_priority_job in high_priority_jobs if ("bert" in high_priority_job) or ("llama" in high_priority_job)]
    high_priority_jobs = [high_priority_job for high_priority_job in high_priority_jobs if "server" in high_priority_job]
    high_priority_jobs = [high_priority_job for high_priority_job in high_priority_jobs if "load_0.5" in high_priority_job]
    best_effort_jobs = priority_df["best_effort_job"].unique()

    metrics = ["99th"]

    plot_slo_comparison_system_throughput_combined(priority_df, high_priority_jobs, best_effort_jobs, metric=metrics[0])

    for high_priority_job in high_priority_jobs:
        for metric in metrics:
            plot_slo_comparison_tally_sensitivity(priority_df, high_priority_job, best_effort_jobs, metric=metric)


if __name__ == "__main__":
    main()