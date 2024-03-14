import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List
import os

def mkdir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

plot_directory = "tally_bench_results/plots"
mkdir_if_not_exists(plot_directory)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'xkcd:sky blue', 'xkcd:light green', 'xkcd:light red', 'xkcd:light purple', 'xkcd:light brown', 'xkcd:light pink', 'xkcd:light gray', 'xkcd:light olive', 'xkcd:light cyan']
markers = ['o', '^', 's', 'p', '*', '+', 'x', 'd', 'v', '<', '>', 'h', 'H', 'D', 'P', 'X']

small_batch_jobs =  [
    'pytorch_resnet50_train_64',
    'pytorch_resnet50_train_64_amp',
    'pytorch_whisper-large-v3_train_8',
    'pytorch_whisper-large-v3_train_8_amp',
    'pytorch_pointnet_train_64',
    'pytorch_pointnet_train_64_amp',
    'pytorch_pegasus-x-base_train_4',
    'pytorch_pegasus-x-base_train_4_amp',
    'pytorch_bert_train_16',
    'pytorch_bert_train_16_amp',
]

def get_best_effort_job_label(best_effort_job, break_lines=False):
    for word in ["pytorch_", "train_"]:
        best_effort_job = best_effort_job.replace(word, "")
    if break_lines:
        best_effort_job = best_effort_job.replace("_", " ")
        best_effort_job = best_effort_job.replace("-", " ")
        best_effort_job = best_effort_job.replace(" ", "\n")
    return best_effort_job


def get_high_priority_job_label(high_priority_job, break_lines=False):
    for word in ["infer_", "_1"]:
        high_priority_job = high_priority_job.replace(word, "")
    if break_lines:
        high_priority_job = high_priority_job.replace("_", " ")
        high_priority_job = high_priority_job.replace("-", " ")
        high_priority_job = high_priority_job.replace(" ", "\n")
    return high_priority_job


def get_metric_label(metric):
    if metric == "avg":
        return metric
    elif metric in ["90th", "95th", "99th"]:
        return f"{metric}_percentile"


def plot_tally_slo_achievable_throughput(priority_df, high_priority_jobs, best_effort_jobs, tolerance_level=0.1, metric="avg"):

    savefig_dir = f"{plot_directory}/tally_achievable_throughputs/{metric}"
    mkdir_if_not_exists(savefig_dir)

    all_jobs_throughputs = []

    for best_effort_job in best_effort_jobs:

        best_effort_job_throughputs = []

        for high_priority_job in high_priority_jobs:
            
            measurements = priority_df[
                (priority_df["high_priority_job"] == high_priority_job) &
                (priority_df["best_effort_job"] == best_effort_job)
            ]

            if measurements.empty:
                best_effort_job_throughputs.append(0.)
                continue

            baseline_latency = measurements[f"high_priority_orig_{metric}_latency"]
            acceptable_latency_bound = (1 + tolerance_level) * baseline_latency

            tally_acceptable_df = measurements[measurements[f"high_priority_tally_{metric}_latency"] <= acceptable_latency_bound]
            if tally_acceptable_df.empty:
                best_effort_job_throughputs.append(0.)
                continue

            max_throughput = tally_acceptable_df[f"best_effort_tally_throughput"].max()
            best_effort_job_throughputs.append(max_throughput)
    
        all_jobs_throughputs.append(best_effort_job_throughputs)

    # plotting
    plt.clf()
    fig, ax = plt.subplots(figsize=(len(high_priority_jobs) * 5, 8))

    # Width of a bar
    width = 0.07
    
    pos = np.arange(len(high_priority_jobs))

    # for i, metric in enumerate(metrics):
    for idx, best_effort_job in enumerate(best_effort_jobs):
        ax.bar(pos + idx * width, all_jobs_throughputs[idx], width, label=f"{get_best_effort_job_label(best_effort_job)}", color=colors[idx], alpha=0.8, edgecolor='black')

    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xticks(pos + ((len(best_effort_jobs) - 1) / 2) * width)
    ax.set_xticklabels([get_high_priority_job_label(high_priority_job) for high_priority_job in high_priority_jobs], fontsize=12)
    ax.set_xlabel("Inference Jobs")
    ax.set_ylabel("Best-Effort Job Throughput")
    ax.set_title(f"Achievable Throughput for {metric} latency with tolerance level {tolerance_level}")

    ax.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=20)
    plt.subplots_adjust(right=0.80)
    plt.savefig(f"{savefig_dir}/tolerance_level_{tolerance_level}.png")


def plot_motivation_latency_comparison(priority_df, high_priority_job, best_effort_jobs, metric="avg"):
    
    savefig_dir = f"{plot_directory}/latency_comparison/{metric}"
    mkdir_if_not_exists(savefig_dir)
    
    high_priority_job_df = priority_df[priority_df["high_priority_job"] == high_priority_job]
    baseline_latencies, time_sliced_latencies, mps_latencies, mps_priority_latencies = [], [], [], []
    plotted_best_effort_jobs = []

    for best_effort_job in best_effort_jobs:
        best_effort_job_df = high_priority_job_df[high_priority_job_df["best_effort_job"] == best_effort_job]

        if best_effort_job_df.empty:
            continue
    
        plotted_best_effort_jobs.append(best_effort_job)
        
        baseline_latency = best_effort_job_df[f"high_priority_orig_{metric}_latency"].values[0]
        time_sliced_latency = best_effort_job_df[f"high_priority_hardware_mp_{metric}_latency"].values[0]
        mps_latency = best_effort_job_df[f"high_priority_mps_{metric}_latency"].values[0]
        mps_priority_latency = best_effort_job_df[f"high_priority_mps_priority_{metric}_latency"].values[0]

        baseline_latencies.append(baseline_latency)
        time_sliced_latencies.append(time_sliced_latency)
        mps_latencies.append(mps_latency)
        mps_priority_latencies.append(mps_priority_latency)

    # plotting
    plt.clf()
    fig, ax = plt.subplots(figsize=(len(plotted_best_effort_jobs) * 2, 8))

    # Width of a bar
    width = 0.2
    
    pos = np.arange(len(plotted_best_effort_jobs))

    ax.bar(pos, baseline_latencies, width, label=f"Baseline", color=colors[0], alpha=0.8, edgecolor='black')
    ax.bar(pos + 1 * width, time_sliced_latencies, width, label=f"Time-sliced", color=colors[1], alpha=0.8, edgecolor='black')
    ax.bar(pos + 2 * width, mps_latencies, width, label=f"MPS", color=colors[2], alpha=0.8, edgecolor='black')
    ax.bar(pos + 3 * width, mps_priority_latencies, width, label=f"MPS-Priority", color=colors[3], alpha=0.8, edgecolor='black')
    
    num_targets = 4
    ax.set_xticks(pos + ((num_targets - 1) / 2) * width)
    ax.set_xticklabels([get_best_effort_job_label(best_effort_job, break_lines=True) for best_effort_job in plotted_best_effort_jobs])
    ax.set_xlabel("Best-Effort Jobs")
    ax.set_ylabel(f"Latency (ms)")
    
    ax.legend()
    plt.savefig(f"{savefig_dir}/{high_priority_job}.png")


def plot_slo_comparison(priority_df, high_priority_job, best_effort_jobs, metric="avg", tolerance_level=0.1):

    savefig_dir = f"{plot_directory}/slo_comparison/{metric}/{tolerance_level}"
    mkdir_if_not_exists(savefig_dir)
    
    high_priority_job_df = priority_df[priority_df["high_priority_job"] == high_priority_job]
    baseline_latencies, time_sliced_latencies, mps_latencies, mps_priority_latencies, tally_latencies, = [], [], [], [], []
    priority_time_sliced_throughputs, priority_mps_throughputs, priority_mps_priority_throughputs, priority_tally_throughputs = [], [], [], []
    time_sliced_throughputs, mps_throughputs, mps_priority_throughputs, tally_throughputs = [], [], [], []
    plotted_best_effort_jobs = []

    for best_effort_job in best_effort_jobs:
        best_effort_job_df = high_priority_job_df[high_priority_job_df["best_effort_job"] == best_effort_job]

        if best_effort_job_df.empty:
            continue

        baseline_latency = best_effort_job_df[f"high_priority_orig_{metric}_latency"].values[0]
        time_sliced_latency = best_effort_job_df[f"high_priority_hardware_mp_{metric}_latency"].values[0]
        priority_time_sliced_throughput = best_effort_job_df[f"high_priority_hardware_mp_throughput"].values[0]
        time_sliced_throughput = best_effort_job_df[f"best_effort_hardware_mp_throughput"].values[0]
        mps_latency = best_effort_job_df[f"high_priority_mps_{metric}_latency"].values[0]
        priority_mps_throughput = best_effort_job_df[f"high_priority_mps_throughput"].values[0]
        mps_throughput = best_effort_job_df[f"best_effort_mps_throughput"].values[0]
        mps_priority_latency = best_effort_job_df[f"high_priority_mps_priority_{metric}_latency"].values[0]
        priority_mps_priority_throughput = best_effort_job_df[f"high_priority_mps_priority_throughput"].values[0]
        mps_priority_throughput = best_effort_job_df[f"best_effort_mps_priority_throughput"].values[0]

        acceptable_latency_bound = (1 + tolerance_level) * baseline_latency
        tally_acceptable_df = best_effort_job_df[best_effort_job_df[f"high_priority_tally_{metric}_latency"] <= acceptable_latency_bound]

        if tally_acceptable_df.empty:
            tally_latency = 0.
            tally_throughput = 0.
            priority_tally_throughput = 0.
        else:
            best_achievable_throughput = tally_acceptable_df[f"best_effort_tally_throughput"].max()
            best_measurement = tally_acceptable_df[tally_acceptable_df[f"best_effort_tally_throughput"] == best_achievable_throughput]
            tally_latency = best_measurement[f"high_priority_tally_{metric}_latency"].values[0]
            tally_throughput = best_measurement[f"best_effort_tally_throughput"].values[0]
            priority_tally_throughput = best_measurement[f"high_priority_tally_throughput"].values[0]

        baseline_latencies.append(baseline_latency)
        time_sliced_latencies.append(time_sliced_latency)
        mps_latencies.append(mps_latency)
        mps_priority_latencies.append(mps_priority_latency)
        tally_latencies.append(tally_latency)

        priority_time_sliced_throughputs.append(priority_time_sliced_throughput)
        priority_mps_throughputs.append(priority_mps_throughput)
        priority_mps_priority_throughputs.append(priority_mps_priority_throughput)
        priority_tally_throughputs.append(priority_tally_throughput)

        time_sliced_throughputs.append(time_sliced_throughput)
        mps_throughputs.append(mps_throughput)
        mps_priority_throughputs.append(mps_priority_throughput)
        tally_throughputs.append(tally_throughput)

        plotted_best_effort_jobs.append(best_effort_job)
    
    # plotting
    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(len(plotted_best_effort_jobs) * 1, 8), sharex=True)

    # Width of a bar
    width = 0.15
    
    pos = np.arange(len(plotted_best_effort_jobs))

    # plot latency
    ax1.bar(pos, baseline_latencies, width, label=f"Baseline", color=colors[0], alpha=0.8, edgecolor='black')
    ax1.bar(pos + 1 * width, time_sliced_latencies, width, label=f"Time-Sliced", color=colors[1], alpha=0.8, edgecolor='black')
    ax1.bar(pos + 2 * width, mps_latencies, width, label=f"MPS", color=colors[2], alpha=0.8, edgecolor='black')
    ax1.bar(pos + 3 * width, mps_priority_latencies, width, label=f"MPS-Priority", color=colors[3], alpha=0.8, edgecolor='black')
    ax1.bar(pos + 4 * width, tally_latencies, width, label=f"Tally", color=colors[4], alpha=0.8, edgecolor='black')

    ax1.set_title("High-Priority Latency Comparison")
    ax1.set_ylabel(f"Latency (ms)")
    ax1.legend()

    # plot high-priority throughput
    ax2.bar(pos + 1 * width, priority_time_sliced_throughputs, width, label=f"Time-sliced", color=colors[1], alpha=0.8, edgecolor='black')
    ax2.bar(pos + 2 * width, priority_mps_throughputs, width, label=f"MPS", color=colors[2], alpha=0.8, edgecolor='black')
    ax2.bar(pos + 3 * width, priority_mps_priority_throughputs, width, label=f"MPS-Priority", color=colors[3], alpha=0.8, edgecolor='black')
    ax2.bar(pos + 4 * width, priority_tally_throughputs, width, label=f"Tally", color=colors[4], alpha=0.8, edgecolor='black')

    ax2.set_title("High-Priority Throughput Comparison")
    ax2.set_ylabel(f"Normalized Throughput")
    # ax2.legend()

    # plot best-effort throughput
    ax3.bar(pos + 1 * width, time_sliced_throughputs, width, label=f"Time-sliced", color=colors[1], alpha=0.8, edgecolor='black')
    ax3.bar(pos + 2 * width, mps_throughputs, width, label=f"MPS", color=colors[2], alpha=0.8, edgecolor='black')
    ax3.bar(pos + 3 * width, mps_priority_throughputs, width, label=f"MPS-Priority", color=colors[3], alpha=0.8, edgecolor='black')
    ax3.bar(pos + 4 * width, tally_throughputs, width, label=f"Tally", color=colors[4], alpha=0.8, edgecolor='black')

    ax3.set_title("Best-effort Throughput Comparison")
    ax3.set_ylabel(f"Normalized Throughput")
    # ax3.legend()

    num_targets = 5
    ax3.set_xticks(pos + ((num_targets - 1) / 2) * width)
    ax3.set_xticklabels([get_best_effort_job_label(best_effort_job, break_lines=True) for best_effort_job in plotted_best_effort_jobs])
    ax3.set_xlabel("Best-Effort Jobs")

    plt.savefig(f"{savefig_dir}/{high_priority_job}.png")


def main():
    priority_df = pd.read_csv("tally_bench_results/priority-aware-perf.csv")
    high_priority_jobs = priority_df["high_priority_job"].unique()
    best_effort_jobs = priority_df["best_effort_job"].unique()
    best_effort_jobs = [job for job in best_effort_jobs if job not in small_batch_jobs]

    metrics = ["avg", "90th", "95th", "99th"]

    # plot Baseline, MPS, Time-sliced latency comparison
    for high_priority_job in high_priority_jobs:
        for metric in metrics:
            plot_motivation_latency_comparison(priority_df, high_priority_job, best_effort_jobs, metric=metric)

    # plot Tally Achievable Throughput under a certain SLO
    for metric in metrics:
        for tolerance_level in [0.1, 0.2, 0.3]:
            plot_tally_slo_achievable_throughput(priority_df, high_priority_jobs, best_effort_jobs, tolerance_level=tolerance_level, metric=metric)

    # plot Baseline, MPS, Time-sliced, Tally latency and Throughput comparison under a certain SLO
    for high_priority_job in high_priority_jobs:
        for metric in metrics:
            for tolerance_level in [0.1, 0.2, 0.3]:
                plot_slo_comparison(priority_df, high_priority_job, best_effort_jobs, metric=metric, tolerance_level=tolerance_level)
                

if __name__ == "__main__":
    main()