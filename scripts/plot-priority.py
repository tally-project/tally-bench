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


def get_best_effort_job_label(best_effort_job):
    for word in ["pytorch_", "_amp", "train_"]:
        best_effort_job = best_effort_job.replace(word, "")
    return best_effort_job


def get_high_priority_job_label(high_priority_job):
    for word in ["infer_", "_1"]:
        high_priority_job = high_priority_job.replace(word, "")
    return high_priority_job


def plot_latency_vs_throughput(high_priority_job, metric, best_effort_throughputs):

    plt.clf()
    high_priority_label = get_high_priority_job_label(high_priority_job)

    sla_thresholds = list(list(best_effort_throughputs.values())[0].keys())
    sla_threshold_percentages = [x * 100 for x in sla_thresholds]

    plt.figure(figsize=(10, 6))
    idx = 0
    for best_effort_job, throughputs in best_effort_throughputs.items():

        achievable_throughput = [throughputs[sla_threshold]["tally"] for sla_threshold in sla_thresholds]
        plt.plot(
            [sla_threshold_percentages[idx] for idx in range(len(sla_threshold_percentages)) if achievable_throughput[idx] > 0],
            [achievable_throughput[idx] for idx in range(len(sla_threshold_percentages)) if achievable_throughput[idx] > 0],
            label=f"{get_best_effort_job_label(best_effort_job)}",
            markersize=5,
            marker=markers[idx],
            color=colors[idx]
        )
        idx += 1

    plt.xticks(sla_threshold_percentages)
    plt.legend()

    plt.grid(True, linestyle=':', alpha=0.5)
    plt.title(f"{high_priority_label}")
    plt.xlabel(f"{metric} SLA Attainment (%)")
    plt.ylabel('Best-Effort Throughput')
    plt.savefig(f"{plot_directory}/{high_priority_label}_{metric}.png")


def plot_priority_data(data_file_name):

    sla_thresholds = [0.05, 0.10, 0.15, 0.20]

    priority_df = pd.read_csv(data_file_name)
    high_priority_jobs = priority_df["high_priority_job"].unique()

    for high_priority_job in high_priority_jobs:

        high_priority_job_df = priority_df[priority_df["high_priority_job"] == high_priority_job]
        best_effort_jobs = high_priority_job_df["best_effort_job"].unique()

        for metric in ["avg", "90th", "95th", "99th", ]:

            best_effort_throughputs = {}

            for best_effort_job in best_effort_jobs:

                if "amp" not in best_effort_job:
                    continue

                best_effort_throughputs[best_effort_job] = {}
                
                best_effort_job_df = high_priority_job_df[high_priority_job_df["best_effort_job"] == best_effort_job]
                if best_effort_job_df.empty:
                    continue
                    
                original_latency = best_effort_job_df[f"high_priority_orig_{metric}_latency"].head()
                
                for sla_threshold in sla_thresholds:

                    best_effort_throughputs[best_effort_job][sla_threshold] = {}
                    achievable_throughput = best_effort_throughputs[best_effort_job][sla_threshold]

                    acceptable_latency = original_latency * (1 + sla_threshold)

                    for backend in ["mps", "tally"]:

                        acceptable_df = best_effort_job_df[best_effort_job_df[f"high_priority_{backend}_{metric}_latency"] <= acceptable_latency]
                        if not acceptable_df.empty:
                            achievable_throughput[backend] = acceptable_df[f"best_effort_{backend}_throughput"].max()
                        else:
                            achievable_throughput[backend] = 0.
            
            plot_latency_vs_throughput(high_priority_job, metric, best_effort_throughputs)   


if __name__ == "__main__":
    plot_priority_data("tally_bench_results/priority-aware-perf.csv")