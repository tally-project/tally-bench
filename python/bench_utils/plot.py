import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

from bench_utils.utils import mkdir_if_not_exists
from bench_utils.parse import get_slo_comparison_data

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:olive', 'xkcd:light purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:purple', 'tab:cyan', 'xkcd:sky blue', 'xkcd:light green', 'xkcd:light red', 'xkcd:light purple', 'xkcd:light brown', 'xkcd:light pink', 'xkcd:light gray', 'xkcd:light olive', 'xkcd:light cyan']
markers = ['o', '^', 's', 'p', '*', '+', 'x', 'd', 'v', '<', '>', 'h', 'H', 'D', 'P', 'X']

tally_default_config = {
    "preemption_latency_limit": 0.1,
    "min_wait_time": "Default",
    "use_original_configs": False,
    "use_space_share": False,
    "disable_transformation": False,
    "wait_time_to_use_original": "Default",
}


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


def get_metric_str(metric):
    if metric == "avg":
        return "Average"
    elif metric in ["90th", "95th", "99th"]:
        return f"{metric}-percentile"


def plot_tally_slo_achievable_throughput(priority_df, high_priority_jobs, best_effort_jobs, tolerance_level=0.1, metric="avg", out_directory="tally_bench_results/plots"):

    savefig_dir = f"{out_directory}/tally_achievable_throughputs/{metric}"
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
    fig, ax = plt.subplots(figsize=(len(high_priority_jobs) * 6, 8))

    # Width of a bar
    width = 0.07
    
    pos = np.arange(len(high_priority_jobs))

    # for i, metric in enumerate(metrics):
    for idx, best_effort_job in enumerate(best_effort_jobs):
        ax.bar(pos + idx * width, all_jobs_throughputs[idx], width, label=f"{get_best_effort_job_label(best_effort_job)}", color=colors[idx], alpha=0.7, edgecolor='black')

    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xticks(pos + ((len(best_effort_jobs) - 1) / 2) * width)
    ax.set_xticklabels([get_high_priority_job_label(high_priority_job) for high_priority_job in high_priority_jobs], fontsize=12)
    ax.set_xlabel("Inference Jobs")
    ax.set_ylabel("Best-Effort Job Throughput")
    ax.set_title(f"Achievable Throughput for {metric} latency with tolerance level {tolerance_level}")

    ax.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=20)
    plt.subplots_adjust(right=0.80)
    plt.savefig(f"{savefig_dir}/tolerance_level_{tolerance_level}.png")


def plot_motivation_latency_comparison(priority_df, high_priority_job, best_effort_jobs, metric="avg",
                                       out_directory="tally_bench_results/plots", out_filename=None, remove_amp=False):
    
    if not out_filename:
        out_filename = high_priority_job

    savefig_dir = f"{out_directory}/motivation_latency_comparison/{metric}"
    mkdir_if_not_exists(savefig_dir)
    
    high_priority_job_df = priority_df[priority_df["high_priority_job"] == high_priority_job]
    baseline_latencies, time_sliced_latencies, mps_latencies, mps_priority_latencies = [], [], [], []
    used_best_effort_jobs = []

    for best_effort_job in best_effort_jobs:
        best_effort_job_df = high_priority_job_df[high_priority_job_df["best_effort_job"] == best_effort_job]

        if best_effort_job_df.empty:
            continue
    
        used_best_effort_jobs.append(best_effort_job)
        
        baseline_latency = best_effort_job_df[f"high_priority_orig_{metric}_latency"].values[0]
        time_sliced_latency = best_effort_job_df[f"high_priority_time_slicing_{metric}_latency"].values[0]
        mps_latency = best_effort_job_df[f"high_priority_mps_{metric}_latency"].values[0]
        mps_priority_latency = best_effort_job_df[f"high_priority_mps_priority_{metric}_latency"].values[0]

        baseline_latencies.append(baseline_latency)
        time_sliced_latencies.append(time_sliced_latency)
        mps_latencies.append(mps_latency)
        mps_priority_latencies.append(mps_priority_latency)

    # plotting
    plt.clf()
    fig, ax = plt.subplots(figsize=(len(used_best_effort_jobs) * 2, 8))

    # Width of a bar
    width = 0.2
    
    pos = np.arange(len(used_best_effort_jobs))

    ax.bar(pos, baseline_latencies, width, label=f"Baseline", color=colors[0], alpha=0.7, edgecolor='black')
    ax.bar(pos + 1 * width, time_sliced_latencies, width, label=f"Time-sliced", color=colors[1], alpha=0.7, edgecolor='black')
    ax.bar(pos + 2 * width, mps_latencies, width, label=f"MPS", color=colors[2], alpha=0.7, edgecolor='black')
    ax.bar(pos + 3 * width, mps_priority_latencies, width, label=f"MPS-Priority", color=colors[3], alpha=0.7, edgecolor='black')
    
    num_targets = 4
    ax.set_xticks(pos + ((num_targets - 1) / 2) * width)
    ax.set_xticklabels([get_best_effort_job_label(best_effort_job, break_lines=True) for best_effort_job in used_best_effort_jobs])
    ax.set_xlabel("Best-Effort Jobs")
    ax.set_ylabel(f"Latency (ms)")
    
    ax.legend()
    plt.savefig(f"{savefig_dir}/{out_filename}.png")


def plot_slo_comparison_seperate_throughput(priority_df, high_priority_job, best_effort_jobs, metric="avg", out_directory="tally_bench_results/plots"):

    savefig_dir = f"{out_directory}/slo_comparison_seperate_throughput/{metric}"
    mkdir_if_not_exists(savefig_dir)
    
    data = get_slo_comparison_data(priority_df, high_priority_job, best_effort_jobs, tally_default_config, metric)

    baseline_latencies = data["baseline_latencies"]
    time_sliced_latencies = data["time_sliced_latencies"]
    mps_latencies = data["mps_latencies"]
    mps_priority_latencies = data["mps_priority_latencies"]
    tally_latencies = data["tally_latencies"]
    tgs_latencies = data["tgs_latencies"]
    priority_time_sliced_throughputs = data["priority_time_sliced_throughputs"]
    priority_mps_throughputs = data["priority_mps_throughputs"]
    priority_mps_priority_throughputs = data["priority_mps_priority_throughputs"]
    priority_tally_throughputs = data["priority_tally_throughputs"]
    priority_tgs_throughputs = data["priority_tgs_throughputs"]
    time_sliced_throughputs = data["time_sliced_throughputs"]
    mps_throughputs = data["mps_throughputs"]
    mps_priority_throughputs = data["mps_priority_throughputs"]
    tally_throughputs = data["tally_throughputs"]
    tgs_throughputs = data["tgs_throughputs"]
    used_best_effort_jobs = data["used_best_effort_jobs"]

    # plotting
    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(len(used_best_effort_jobs) * 1, 5), 8), sharex=True)

    # Width of a bar
    width = 0.15
    
    pos = np.arange(len(used_best_effort_jobs))

    # plot latency
    ax1.bar(pos, baseline_latencies, width, label=f"Baseline", color=colors[0], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 1 * width, time_sliced_latencies, width, label=f"Time-Sliced", color=colors[1], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 2 * width, mps_latencies, width, label=f"MPS", color=colors[2], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 3 * width, mps_priority_latencies, width, label=f"MPS-Priority", color=colors[3], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 4 * width, tgs_latencies, width, label=f"TGS", color=colors[4], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 5 * width, tally_latencies, width, label=f"Tally", color=colors[5], alpha=0.7, edgecolor='black')

    ax1.set_title(f"High-Priority {get_metric_str(metric)} Latency Comparison")
    ax1.set_ylabel(f"Latency (ms)")
    ax1.legend()

    # plot high-priority throughput
    ax2.bar(pos + 1 * width, priority_time_sliced_throughputs, width, label=f"Time-sliced", color=colors[1], alpha=0.7, edgecolor='black')
    ax2.bar(pos + 2 * width, priority_mps_throughputs, width, label=f"MPS", color=colors[2], alpha=0.7, edgecolor='black')
    ax2.bar(pos + 3 * width, priority_mps_priority_throughputs, width, label=f"MPS-Priority", color=colors[3], alpha=0.7, edgecolor='black')
    ax2.bar(pos + 4 * width, priority_tgs_throughputs, width, label=f"TGS", color=colors[4], alpha=0.7, edgecolor='black')
    ax2.bar(pos + 5 * width, priority_tally_throughputs, width, label=f"Tally", color=colors[5], alpha=0.7, edgecolor='black')

    ax2.set_title("High-Priority Throughput Comparison")
    ax2.set_ylabel(f"Normalized Throughput")
    # ax2.legend()

    # plot best-effort throughput
    ax3.bar(pos + 1 * width, time_sliced_throughputs, width, label=f"Time-sliced", color=colors[1], alpha=0.7, edgecolor='black')
    ax3.bar(pos + 2 * width, mps_throughputs, width, label=f"MPS", color=colors[2], alpha=0.7, edgecolor='black')
    ax3.bar(pos + 3 * width, mps_priority_throughputs, width, label=f"MPS-Priority", color=colors[3], alpha=0.7, edgecolor='black')
    ax3.bar(pos + 4 * width, tgs_throughputs, width, label=f"TGS", color=colors[4], alpha=0.7, edgecolor='black')
    ax3.bar(pos + 5 * width, tally_throughputs, width, label=f"Tally", color=colors[5], alpha=0.7, edgecolor='black')


    ax3.set_title("Best-effort Throughput Comparison")
    ax3.set_ylabel(f"Normalized Throughput")
    # ax3.legend()

    num_targets = 5
    ax3.set_xticks(pos + ((num_targets - 1) / 2) * width)
    ax3.set_xticklabels([get_best_effort_job_label(best_effort_job, break_lines=True) for best_effort_job in used_best_effort_jobs])
    ax3.set_xlabel("Best-Effort Jobs")

    plt.savefig(f"{savefig_dir}/{high_priority_job}.png")


def plot_slo_comparison_system_throughput(priority_df, high_priority_job, best_effort_jobs, metric="avg", out_directory="tally_bench_results/plots"):

    savefig_dir = f"{out_directory}/slo_comparison_system_throughput/{metric}"
    mkdir_if_not_exists(savefig_dir)
    
    data = get_slo_comparison_data(priority_df, high_priority_job, best_effort_jobs, tally_default_config, metric)

    baseline_latencies = data["baseline_latencies"]
    time_sliced_latencies = data["time_sliced_latencies"]
    mps_latencies = data["mps_latencies"]
    mps_priority_latencies = data["mps_priority_latencies"]
    tally_latencies = data["tally_latencies"]
    tgs_latencies = data["tgs_latencies"]
    priority_time_sliced_throughputs = data["priority_time_sliced_throughputs"]
    priority_mps_throughputs = data["priority_mps_throughputs"]
    priority_mps_priority_throughputs = data["priority_mps_priority_throughputs"]
    priority_tally_throughputs = data["priority_tally_throughputs"]
    priority_tgs_throughputs = data["priority_tgs_throughputs"]
    time_sliced_throughputs = data["time_sliced_throughputs"]
    mps_throughputs = data["mps_throughputs"]
    mps_priority_throughputs = data["mps_priority_throughputs"]
    tally_throughputs = data["tally_throughputs"]
    tgs_throughputs = data["tgs_throughputs"]
    used_best_effort_jobs = data["used_best_effort_jobs"]

    time_sliced_system_throughputs = [x + y for x, y in zip(priority_time_sliced_throughputs, time_sliced_throughputs)]
    mps_system_throughputs = [x + y for x, y in zip(priority_mps_throughputs, mps_throughputs)]
    mps_priority_system_throughputs = [x + y for x, y in zip(priority_mps_priority_throughputs, mps_priority_throughputs)]
    tally_system_throughputs = [x + y for x, y in zip(priority_tally_throughputs, tally_throughputs)]
    tgs_system_throughputs = [x + y for x, y in zip(priority_tgs_throughputs, tgs_throughputs)]

    # plotting
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(len(used_best_effort_jobs) * 1, 5), 8), sharex=True)

    # Width of a bar
    width = 0.15
    
    pos = np.arange(len(used_best_effort_jobs))

    # plot latency
    ax1.bar(pos, baseline_latencies, width, label=f"Baseline", color=colors[0], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 1 * width, time_sliced_latencies, width, label=f"Time-Sliced", color=colors[1], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 2 * width, mps_latencies, width, label=f"MPS", color=colors[2], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 3 * width, mps_priority_latencies, width, label=f"MPS-Priority", color=colors[3], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 4 * width, tgs_latencies, width, label=f"TGS", color=colors[4], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 5 * width, tally_latencies, width, label=f"Tally", color=colors[5], alpha=0.7, edgecolor='black')

    ax1.set_title(f"High-Priority {get_metric_str(metric)} Latency Comparison")
    ax1.set_ylabel(f"Latency (ms)")
    ax1.legend()

    # plot system throughput
    ax2.bar(pos + 1 * width, time_sliced_system_throughputs, width,  label=f"Time-sliced", color=colors[1], alpha=0.7, edgecolor='black')
    ax2.bar(pos + 2 * width, mps_system_throughputs, width,  label=f"MPS", color=colors[2], alpha=0.7, edgecolor='black')
    ax2.bar(pos + 3 * width, mps_priority_system_throughputs, width, label=f"MPS-Priority", color=colors[3], alpha=0.7, edgecolor='black')
    ax2.bar(pos + 4 * width, tgs_system_throughputs, width,  label=f"TGS", color=colors[4], alpha=0.7, edgecolor='black')
    ax2.bar(pos + 5 * width, tally_system_throughputs, width,  label=f"Tally", color=colors[5], alpha=0.7, edgecolor='black')

    ax2.set_title("System Throughput Comparison")
    ax2.set_ylabel(f"System Throughput")

    num_targets = 5
    ax2.set_xticks(pos + ((num_targets - 1) / 2) * width)
    ax2.set_xticklabels([get_best_effort_job_label(best_effort_job, break_lines=True) for best_effort_job in used_best_effort_jobs])
    ax2.set_xlabel("Best-Effort Jobs")

    plt.savefig(f"{savefig_dir}/{high_priority_job}.png")


def plot_slo_comparison_system_throughput_all(priority_df, high_priority_jobs, best_effort_jobs, metric="avg", out_directory="tally_bench_results/plots"):

    savefig_dir = f"{out_directory}/slo_comparison_system_throughput_all"
    mkdir_if_not_exists(savefig_dir)

    load_levels = ["0.75", "0.5", "0.25"]

    unique_high_priority_jobs = []
    for high_priority_job in high_priority_jobs:
        job_name = high_priority_job.split("_infer_server")[0]
        if job_name not in unique_high_priority_jobs:
            unique_high_priority_jobs.append(job_name)

    all_jobs_throughputs = {}
    for load_level in load_levels:
        all_jobs_throughputs[load_level] = [[0. for _ in range(len(unique_high_priority_jobs))] for _ in range(len(best_effort_jobs))]
    all_jobs_latencies = [[0. for _ in range(len(unique_high_priority_jobs))] for _ in range(len(best_effort_jobs))]

    for i, best_effort_job in enumerate(best_effort_jobs):

        for j, unique_high_priority_job in enumerate(unique_high_priority_jobs):
            for load_level in load_levels:

                high_priority_job = f"{unique_high_priority_job}_infer_server_load_{load_level}_1"

                measurements = priority_df[
                    (priority_df["high_priority_job"] == high_priority_job) &
                    (priority_df["best_effort_job"] == best_effort_job)
                ]

                for param in tally_default_config:
                    val = tally_default_config[param]
                    measurements = measurements[measurements[param] == val]

                if measurements.empty:
                    continue

                baseline_latency = measurements[f"high_priority_orig_{metric}_latency"].values[0]

                best_effort_job_throughput = measurements[f"best_effort_tally_throughput"].values[0]
                high_priority_job_latency = measurements[f"high_priority_tally_{metric}_latency"].values[0]
                high_priority_job_latency_slowdown = high_priority_job_latency / baseline_latency

                all_jobs_throughputs[load_level][i][j] = best_effort_job_throughput
                all_jobs_latencies[i][j] += (high_priority_job_latency_slowdown)
    
    # avg
    all_jobs_latencies = np.array(all_jobs_latencies) / len(load_levels)

    # plotting
    plt.clf()
    fig, axes = plt.subplots(1 + len(load_levels), 1, figsize=(len(unique_high_priority_jobs) * 3, 8), sharex=True)

    # Width of a bar
    width = 0.1
    
    pos = np.arange(len(unique_high_priority_jobs))

    for idx, best_effort_job in enumerate(best_effort_jobs):

        ax1 = axes[0]
        ax1.bar(pos + idx * width, all_jobs_latencies[idx], width, label=f"{get_best_effort_job_label(best_effort_job)}", color=colors[idx], alpha=0.7, edgecolor='black')
        ax1.set_yticks(np.linspace(0, 1, 5))
        ax1.set_title(f"High Priority {get_metric_str(metric)} Slowdown")

        for j, load_level in enumerate(load_levels):
            ax = axes[j + 1]
            ax.bar(pos + idx * width, all_jobs_throughputs[load_level][idx], width, label=f"{get_best_effort_job_label(best_effort_job)}", color=colors[idx], alpha=0.7, edgecolor='black')
            ax.set_yticks(np.linspace(0, 1, 5))
            ax.set_title(f"Best-effort Normalized Throughput w/ Load={load_level}")

    last_ax = axes[-1]
    last_ax.set_xticks(pos + ((len(best_effort_jobs) - 1) / 2) * width)
    last_ax.set_xticklabels([get_high_priority_job_label(high_priority_job) for high_priority_job in unique_high_priority_jobs], fontsize=12)
    last_ax.set_xlabel("Inference Jobs")

    axes[0].legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=20)
    plt.subplots_adjust(right=0.80)
    plt.savefig(f"{savefig_dir}/{metric}.png")


def plot_slo_comparison_tally_sensitivity(priority_df, high_priority_job, best_effort_jobs, metric="avg", out_directory="tally_bench_results/plots"):
    savefig_dir = f"{out_directory}/slo_comparison_tally_sensitivity/{metric}"
    mkdir_if_not_exists(savefig_dir)
    
    data = get_slo_comparison_data(priority_df, high_priority_job, best_effort_jobs, tally_default_config, metric)

    baseline_latencies = data["baseline_latencies"]
    tally_latencies = data["tally_latencies"]
    tally_space_share_latencies = data["tally_space_share_latencies"]
    tally_no_transform_latencies = data["tally_no_transform_latencies"]

    used_best_effort_jobs = data["used_best_effort_jobs"]

    # plotting
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(max(len(used_best_effort_jobs) * 1, 5), 8), sharex=True)

    # Width of a bar
    width = 0.15
    
    pos = np.arange(len(used_best_effort_jobs))

    # plot latency
    ax1.bar(pos, baseline_latencies, width, label=f"Baseline", color=colors[0], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 1 * width, tally_space_share_latencies, width, label=f"No Scheduling", color=colors[1], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 2 * width, tally_no_transform_latencies, width, label=f"Scheduling w/o Transformation", color=colors[2], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 3 * width, tally_latencies, width, label=f"Scheduling + Transformation (Tally)", color=colors[3], alpha=0.7, edgecolor='black')

    ax1.set_yscale("log")
    ax1.set_title(f"High-Priority {get_metric_str(metric)} Latency Comparison")
    ax1.set_ylabel(f"Latency (ms)")
    ax1.legend()

    num_targets = 4
    ax1.set_xticks(pos + ((num_targets - 1) / 2) * width)
    ax1.set_xticklabels([get_best_effort_job_label(best_effort_job, break_lines=True) for best_effort_job in used_best_effort_jobs])
    ax1.set_xlabel("Best-Effort Jobs")

    plt.savefig(f"{savefig_dir}/{high_priority_job}.png")


def plot_throughput_vs_load(priority_df, high_priority_job, best_effort_jobs, load_levels, out_directory="tally_bench_results/plots"):

    savefig_dir = f"{out_directory}/throughput_vs_load"
    mkdir_if_not_exists(savefig_dir)
    all_throughputs = {}

    for best_effort_job in best_effort_jobs:

        all_throughputs[best_effort_job] = []

        for load_level in load_levels:

            high_priority_job_key = f"{high_priority_job}_infer_server_load_{load_level}_1"

            measurements = priority_df[
                (priority_df["high_priority_job"] == high_priority_job_key) &
                (priority_df["best_effort_job"] == best_effort_job)
            ]

            for param in tally_default_config:
                val = tally_default_config[param]
                measurements = measurements[measurements[param] == val]

            if measurements.empty:
                continue

            best_effort_job_throughput = measurements[f"best_effort_tally_throughput"].values[0]
            all_throughputs[best_effort_job].append(best_effort_job_throughput)
    
    # plotting
    plt.clf()
    plt.figure(figsize=(10, 6))

    idle_percentage = [1 - x for x in load_levels]

    for best_effort_job in best_effort_jobs:
        plt.plot(idle_percentage, all_throughputs[best_effort_job], label=get_best_effort_job_label(best_effort_job), marker='o')

    plt.xlabel('Idle Percentage (%)')
    plt.ylabel('Normalized Throughput')

    plt.legend()
    plt.grid(True)

    plt.savefig(f"{savefig_dir}/{high_priority_job}.png")