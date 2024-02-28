import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List
import os

plot_directory = "tally_bench_results/plots"
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

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

def get_best_effort_job_label(best_effort_job):
    for word in ["pytorch_", "train_"]:
        best_effort_job = best_effort_job.replace(word, "")
    return best_effort_job


def get_high_priority_job_label(high_priority_job):
    for word in ["infer_", "_1"]:
        high_priority_job = high_priority_job.replace(word, "")
    return high_priority_job


def get_metric_label(metric):
    if metric == "avg":
        return metric
    elif metric in ["90th", "95th", "99th"]:
        return f"{metric}_percentile"


def plot_latency_vs_throughput(metric, tolerance_level, best_effort_job_throughputs):

    high_priority_jobs = list(best_effort_job_throughputs.keys())
    best_effort_jobs = []
    for high_priority_job in high_priority_jobs:
        best_effort_jobs.extend(list(best_effort_job_throughputs[high_priority_job].keys()))
    best_effort_jobs = list(set(best_effort_jobs))
    best_effort_jobs.sort()

    plt.clf()

    # Setting up the figure and axes
    fig, ax = plt.subplots(figsize=(len(high_priority_jobs) * 10, 8))

    # Width of a bar
    width = 0.05
    
    pos = np.arange(len(high_priority_jobs))

    job_throughputs = []
    for best_effort_job in best_effort_jobs:

        throughputs = []
        for high_priority_job in high_priority_jobs:
            if high_priority_job in best_effort_job_throughputs and best_effort_job in best_effort_job_throughputs[high_priority_job]:
                throughputs.append(best_effort_job_throughputs[high_priority_job][best_effort_job]["tally"])
            else:
                throughputs.append(0.)

        job_throughputs.append(throughputs)

    # for i, metric in enumerate(metrics):
    for j, best_effort_job in enumerate(best_effort_jobs):
        ax.bar(pos + j * width, job_throughputs[j], width, label=f"{get_best_effort_job_label(best_effort_job)}", color=colors[j], alpha=0.8, edgecolor='black')

    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xticks(pos + ((len(best_effort_jobs) - 1) / 2) * width)
    ax.set_xticklabels([get_high_priority_job_label(high_priority_job) for high_priority_job in high_priority_jobs])
    ax.set_xlabel("Inference Jobs")
    ax.set_ylabel("Best-Effort Job Throughput")
    ax.set_title(f"Achievable Throughput for {metric} latency with tolerance level {tolerance_level}")

    ax.legend()
    plt.savefig(f"{plot_directory}/metric_{get_metric_label(metric)}_threshold_{tolerance_level}.png")


def plot_latency(high_priority_job, mps_latencies):
    
    best_effort_jobs = list(mps_latencies.keys())
    best_effort_jobs.sort()
    metrics = ["avg", "90th", "95th", "99th"]

    plt.clf()

    # Setting up the figure and axes
    fig, ax = plt.subplots(figsize=(len(best_effort_jobs) * 4, 8))

    # Width of a bar
    width = 0.2
    
    pos = np.arange(len(best_effort_jobs))

    metric_latencies = []
    for metric in metrics:
        latencies = []
        for best_effort_job in best_effort_jobs:
            if best_effort_job in mps_latencies and metric in mps_latencies[best_effort_job]:
                latencies.append(mps_latencies[best_effort_job][metric])
            else:
                latencies.append(0.)
        metric_latencies.append(latencies)

    for j, metric in enumerate(metrics):
        ax.bar(pos + j * width, metric_latencies[j], width, label=f"{get_metric_label(metric)}", color=colors[j], alpha=0.8, edgecolor='black')
    
    ax.set_xticks(pos + ((len(metrics) - 1) / 2) * width)
    ax.set_xticklabels([best_effort_job for best_effort_job in best_effort_jobs])
    ax.set_xlabel("Best-Effort Jobs")
    ax.set_ylabel("MPS Latency Ratio")
    
    ax.legend()
    plt.savefig(f"{plot_directory}/{high_priority_job}_mps_latencies.png")


def plot_mps_latency(priority_df):

    high_priority_jobs = priority_df["high_priority_job"].unique()
    metrics = ["avg", "90th", "95th", "99th"]

    for high_priority_job in high_priority_jobs:

        mps_latencies = {}

        if "single-stream" in high_priority_job:
                continue
        
        high_priority_job_df = priority_df[priority_df["high_priority_job"] == high_priority_job]
        best_effort_jobs = high_priority_job_df["best_effort_job"].unique()
        
        for best_effort_job in best_effort_jobs:
            
            # if "amp" not in best_effort_job:
            #     continue

            if best_effort_job in small_batch_jobs:
                continue

            best_effort_job_df = high_priority_job_df[
                                    (high_priority_job_df["best_effort_job"] == best_effort_job) &
                                    (high_priority_job_df["best_effort_mps_throughput"] != "")
                                ]
            if best_effort_job_df.empty:
                continue

            mps_latencies[best_effort_job] = {}

            for metric in metrics:

                original_latency = best_effort_job_df[f"high_priority_orig_{metric}_latency"].values[0]
                mps_latency = best_effort_job_df[f"high_priority_mps_{metric}_latency"].values[0]

                mps_latencies[best_effort_job][metric] = mps_latency / original_latency

        plot_latency(high_priority_job, mps_latencies)

def plot_achievable_throughput(priority_df):

    tolerance_levels = [0.10, 0.20, 0.30]
    metrics = ["avg", "90th", "95th", "99th"]
    high_priority_jobs = priority_df["high_priority_job"].unique()

    for metric in metrics:

        for tolerance_level in tolerance_levels:

            best_effort_job_throughputs = {}

            for high_priority_job in high_priority_jobs:

                if "single-stream" in high_priority_job:
                    continue

                best_effort_job_throughputs[high_priority_job] = {}

                high_priority_job_df = priority_df[priority_df["high_priority_job"] == high_priority_job]
                best_effort_jobs = high_priority_job_df["best_effort_job"].unique()

                for best_effort_job in best_effort_jobs:

                    # if "amp" not in best_effort_job:
                    #     continue

                    if best_effort_job in small_batch_jobs:
                        continue

                    best_effort_job_df = high_priority_job_df[
                                            (high_priority_job_df["best_effort_job"] == best_effort_job) &
                                            (high_priority_job_df["best_effort_tally_throughput"] != "")
                                        ]
                    if best_effort_job_df.empty:
                        continue

                    best_effort_job_throughputs[high_priority_job][best_effort_job] = {}
                    achievable_throughput = best_effort_job_throughputs[high_priority_job][best_effort_job]

                    original_latency = best_effort_job_df[f"high_priority_orig_{metric}_latency"].values[0]
                    acceptable_latency = original_latency * (1 + tolerance_level)

                    for backend in ["mps", "tally"]:

                        acceptable_df = best_effort_job_df[best_effort_job_df[f"high_priority_{backend}_{metric}_latency"] <= acceptable_latency]
                        if not acceptable_df.empty:
                            achievable_throughput[backend] = acceptable_df[f"best_effort_{backend}_throughput"].max()
                        else:
                            achievable_throughput[backend] = 0.
                
            plot_latency_vs_throughput(metric, tolerance_level, best_effort_job_throughputs)


if __name__ == "__main__":
    priority_df = pd.read_csv("tally_bench_results/priority-aware-perf.csv")
    plot_achievable_throughput(priority_df)
    plot_mps_latency(priority_df)