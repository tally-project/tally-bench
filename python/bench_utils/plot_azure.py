import math
import numpy as np
import matplotlib.pyplot as plt

from bench_utils.utils import compute_avg, compute_percentile

tally_bench_result_dir = "tally_bench_results"
plot_directory = f"{tally_bench_result_dir}/plots"

bench_id = "onnxruntime_bert_infer_server_1_pytorch_bert_train_32"
high_priority_key = "onnxruntime_bert_infer_server_1_0"
best_effort_key = "pytorch_bert_train_32_1"

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'xkcd:light purple', 'tab:olive', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:purple', 'tab:cyan', 'xkcd:sky blue', 'xkcd:light green', 'xkcd:light red', 'xkcd:light purple', 'xkcd:light brown', 'xkcd:light pink', 'xkcd:light gray', 'xkcd:light olive', 'xkcd:light cyan']
markers = ['o', '^', 's', 'p', '*', '+', 'x', 'd', 'v', '<', '>', 'h', 'H', 'D', 'P', 'X']


def plot_latency_over_time(result, interval=0.5, metric="95th"):
    mps_res = result["mps"]
    mps_priority_res = result["mps-priority"]
    tally_priority_res = result["tally_priority"]

    def get_timestamps_latencies_list(res):
        lst = []
        for measurement in res[bench_id]["measurements"]:
            end_timestamps = measurement[high_priority_key]["end_timestamps"][1:]
            latencies = measurement[high_priority_key]["latencies"][1:]
            lst.append((end_timestamps, latencies))
        return lst

    mps_end_timestamps, mps_latencies = get_timestamps_latencies_list(mps_res)[0]
    mps_priority_end_timestamps, mps_priority_latencies = get_timestamps_latencies_list(mps_priority_res)[0]
    tally_timestamps_latencies_list = get_timestamps_latencies_list(tally_priority_res)
    
    last_ts = mps_end_timestamps[-1]
    num_intervals = math.floor(last_ts / interval)

    def get_interval_statistics(end_timestamps, latencies, metric):
        interval_counts = [[] for _ in range(num_intervals)]
        for i in range(len(end_timestamps)):
            ts = end_timestamps[i]
            latency = latencies[i]

            interval_idx = int(ts / interval)
            if interval_idx >= num_intervals:
                break
            interval_counts[interval_idx].append(latency)
        
        interval_statistics = []
        for i in range(num_intervals):
            if len(interval_counts[i]) > 0:
                if metric == "avg":
                    interval_statistics.append(compute_avg(interval_counts[i]))
                elif "th" in metric:
                    percentile = int(metric.split("th")[0])
                    interval_statistics.append(compute_percentile(interval_counts[i], percentile))
            else:
                interval_statistics.append(0)

        interval_statistics = np.array(interval_statistics)
        interval_statistics[interval_statistics == 0] = np.nan
        return interval_statistics

    interval_timestamps = [(i + 0.5) * interval for i in range(num_intervals)]

    mps_interval_statistics = get_interval_statistics(mps_end_timestamps, mps_latencies, metric)
    mps_priority_interval_statistics = get_interval_statistics(mps_priority_end_timestamps, mps_priority_latencies, metric)
    tally_interval_statistics_list = [get_interval_statistics(tally_end_timestamps, tally_latencies, metric) for tally_end_timestamps, tally_latencies in tally_timestamps_latencies_list]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(interval_timestamps, mps_interval_statistics, linestyle='-', color=colors[0], label="mps")
    plt.plot(interval_timestamps, mps_priority_interval_statistics, linestyle='-', color=colors[1], label="mps-priority")
    for idx, tally_interval_statistics in enumerate(tally_interval_statistics_list):
        plt.plot(interval_timestamps, tally_interval_statistics, linestyle='-', color=colors[2 + idx], label=f"tally-{idx}")
    
    plt.title(f'{metric} Latency over time')
    plt.xlabel('Timestamp')
    plt.ylabel('Latency (ms)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{plot_directory}/azure_latency_comparison.png")


def plot_throughput_over_time(result, interval=10):
    mps_res = result["mps"]
    mps_priority_res = result["mps-priority"]
    tally_priority_res = result["tally_priority"]

    def get_timestamps_list(res):
        lst = []
        for measurement in res[bench_id]["measurements"]:
            end_timestamps = measurement[best_effort_key]["end_timestamps"][1:]
            lst.append(end_timestamps)
        return lst

    mps_end_timestamps = get_timestamps_list(mps_res)[0]
    mps_priority_end_timestamps = get_timestamps_list(mps_priority_res)[0]
    tally_end_timestamps_list = get_timestamps_list(tally_priority_res)
    
    last_ts = mps_end_timestamps[-1]
    num_intervals = math.floor(last_ts / interval)

    def get_interval_throughputs(end_timestamps):
        interval_throughputs = [0 for _ in range(num_intervals)]
        for i in range(len(end_timestamps)):
            ts = end_timestamps[i]

            interval_idx = int(ts / interval)
            if interval_idx >= num_intervals:
                break
            interval_throughputs[interval_idx] += 1

        interval_throughputs = np.array(interval_throughputs)
        interval_throughputs = interval_throughputs / interval
        interval_throughputs[interval_throughputs == 0] = np.nan
        return interval_throughputs

    interval_timestamps = [(i + 0.5) * interval for i in range(num_intervals)]

    mps_interval_throughputs = get_interval_throughputs(mps_end_timestamps)
    mps_priority_interval_throughputs = get_interval_throughputs(mps_priority_end_timestamps)
    tally_interval_throughputs_list = [get_interval_throughputs(tally_end_timestamps) for tally_end_timestamps in tally_end_timestamps_list]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(interval_timestamps, mps_interval_throughputs, linestyle='-', color=colors[0], label="mps")
    plt.plot(interval_timestamps, mps_priority_interval_throughputs, linestyle='-', color=colors[1], label="mps-priority")
    for idx, tally_interval_throughputs in enumerate(tally_interval_throughputs_list):
        plt.plot(interval_timestamps, tally_interval_throughputs, linestyle='-', color=colors[2 + idx], label=f"tally-{idx}")
    
    plt.title(f'Best-effort Throughput over time')
    plt.xlabel('Timestamp')
    plt.ylabel('Iterations/s')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{plot_directory}/azure_throughput_comparison.png")