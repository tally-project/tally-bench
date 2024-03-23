import math
import numpy as np
import matplotlib.pyplot as plt

from bench_utils.utils import compute_avg, compute_percentile
from bench_utils.plot import get_metric_str

infer_bench_id = "onnxruntime_bert_infer_server_1"
train_bench_id = "pytorch_bert_train_32"

pair_bench_id = "pytorch_bert_train_32_onnxruntime_bert_infer_server_1"
best_effort_key = "pytorch_bert_train_32_0"
high_priority_key = "onnxruntime_bert_infer_server_1_1"

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'xkcd:light purple', 'tab:olive', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:purple', 'tab:cyan', 'xkcd:sky blue', 'xkcd:light green', 'xkcd:light red', 'xkcd:light purple', 'xkcd:light brown', 'xkcd:light pink', 'xkcd:light gray', 'xkcd:light olive', 'xkcd:light cyan']


def plot_azure_trace_simulation(
    trace_timestamps,
    result,
    trace_interval=1,
    latency_interval=2,
    throughput_interval=10,
    server_throughput=None,
    metric="95th",
    out_directory="."
):
    default_res = result["default"]
    mps_res = result["mps"]
    mps_priority_res = result["mps-priority"]
    tally_priority_res = result["tally_priority"]

    last_ts = trace_timestamps[-1]

    # ============ 1. parse trace traffic over time ================

    trace_bins = np.arange(0, math.ceil(last_ts), trace_interval)
    trace_counts, trace_edges = np.histogram(trace_timestamps, bins=trace_bins)

    trace_edges = trace_edges[:-1] + (trace_interval / 2)
    trace_counts = trace_counts / trace_interval
    trace_counts[trace_counts == 0] = np.nan

    # ============ 2. parse high-priority latencies over time ================

    def get_infer_timestamps_latencies_list(res, bench_id, job_key):
        lst = []
        for measurement in res[bench_id]["measurements"]:
            end_timestamps = measurement[job_key]["end_timestamps"][1:]
            latencies = measurement[job_key]["latencies"][1:]
            lst.append((end_timestamps, latencies))
        return lst

    baseline_infer_timestamps, baseline_latencies = get_infer_timestamps_latencies_list(default_res, infer_bench_id, f"{infer_bench_id}_0")[0]
    time_sliced_infer_timestamps, time_sliced_latencies = get_infer_timestamps_latencies_list(default_res, pair_bench_id, high_priority_key)[0]
    mps_infer_timestamps, mps_latencies = get_infer_timestamps_latencies_list(mps_res, pair_bench_id, high_priority_key)[0]
    mps_priority_infer_timestamps, mps_priority_latencies = get_infer_timestamps_latencies_list(mps_priority_res, pair_bench_id, high_priority_key)[0]
    tally_infer_timestamps_latencies_list = get_infer_timestamps_latencies_list(tally_priority_res, pair_bench_id, high_priority_key)

    latency_num_intervals = math.floor(last_ts / latency_interval)

    def get_interval_statistics(end_timestamps, latencies, metric):
        interval_counts = [[] for _ in range(latency_num_intervals)]
        for i in range(len(end_timestamps)):
            ts = end_timestamps[i]
            latency = latencies[i]

            interval_idx = int(ts / latency_interval)
            if interval_idx >= latency_num_intervals:
                break
            interval_counts[interval_idx].append(latency)
        
        interval_statistics = []
        for i in range(latency_num_intervals):
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

    baseline_infer_interval_statistics = get_interval_statistics(baseline_infer_timestamps, baseline_latencies, metric)
    time_sliced_infer_interval_statistics = get_interval_statistics(time_sliced_infer_timestamps, time_sliced_latencies, metric)
    mps_infer_interval_statistics = get_interval_statistics(mps_infer_timestamps, mps_latencies, metric)
    mps_priority_infer_interval_statistics = get_interval_statistics(mps_priority_infer_timestamps, mps_priority_latencies, metric)
    tally_infer_interval_statistics_list = [
        get_interval_statistics(tally_infer_timestamps, tally_latencies, metric)
        for tally_infer_timestamps, tally_latencies in tally_infer_timestamps_latencies_list
    ]

    latency_interval_timestamps = [(i + 0.5) * latency_interval for i in range(latency_num_intervals)]

    # ============ 3. parse best-effort throughputs over time ================

    def get_train_timestamps_list(res, bench_id, job_key):
        lst = []
        for measurement in res[bench_id]["measurements"]:
            end_timestamps = measurement[job_key]["end_timestamps"][1:]
            lst.append(end_timestamps)
        return lst

    baseline_train_timestamps = get_train_timestamps_list(default_res, train_bench_id, f"{train_bench_id}_0")[0]
    time_sliced_train_timestamps = get_train_timestamps_list(default_res, pair_bench_id, best_effort_key)[0]
    mps_train_timestamps = get_train_timestamps_list(mps_res, pair_bench_id, best_effort_key)[0]
    mps_priority_train_timestamps = get_train_timestamps_list(mps_priority_res, pair_bench_id, best_effort_key)[0]
    tally_train_timestamps_list = get_train_timestamps_list(tally_priority_res, pair_bench_id, best_effort_key)
    
    throughput_num_intervals = math.floor(last_ts / throughput_interval)

    def get_interval_throughputs(end_timestamps):
        interval_throughputs = [0 for _ in range(throughput_num_intervals)]
        for i in range(len(end_timestamps)):
            ts = end_timestamps[i]

            interval_idx = int(ts / throughput_interval)
            if interval_idx >= throughput_num_intervals:
                break
            interval_throughputs[interval_idx] += 1

        interval_throughputs = np.array(interval_throughputs)
        interval_throughputs = interval_throughputs / throughput_interval
        interval_throughputs[interval_throughputs == 0] = np.nan
        return interval_throughputs

    throughput_interval_timestamps = [(i + 0.5) * throughput_interval for i in range(throughput_num_intervals)]

    baseline_interval_throughputs = get_interval_throughputs(baseline_train_timestamps)
    time_sliced_interval_throughputs = get_interval_throughputs(time_sliced_train_timestamps)
    mps_interval_throughputs = get_interval_throughputs(mps_train_timestamps)
    mps_priority_interval_throughputs = get_interval_throughputs(mps_priority_train_timestamps)
    tally_interval_throughputs_list = [
        get_interval_throughputs(tally_train_timestamps)
        for tally_train_timestamps in tally_train_timestamps_list
    ]

    # ============= plotting the data together in one figure =================

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 8), sharex=True)
    
    ax1.plot(trace_edges, trace_counts, linestyle='-', color='g', alpha=0.7)
    ax1.set_ylabel('Request Count')
    ax1.set_title(f'Request Count Over Time')

    if server_throughput:
        ax1.axhline(y=server_throughput, color='r', linestyle='--', label='Server Peak Throughput')
    ax1.legend()

    ax2.plot(latency_interval_timestamps, baseline_infer_interval_statistics, linestyle='-', color=colors[0], label="Baseline")
    ax2.plot(latency_interval_timestamps, time_sliced_infer_interval_statistics, linestyle='-', color=colors[1], label="Time-sliced")
    ax2.plot(latency_interval_timestamps, mps_infer_interval_statistics, linestyle='-', color=colors[2], label="MPS")
    ax2.plot(latency_interval_timestamps, mps_priority_infer_interval_statistics, linestyle='-', color=colors[3], label="MPS-Priority")
    for idx, tally_interval_statistics in enumerate(tally_infer_interval_statistics_list):
        ax2.plot(latency_interval_timestamps, tally_interval_statistics, linestyle='-', color=colors[4 + idx], label=f"Tally-Config-{idx}")

    ax2.set_ylabel(f'{get_metric_str(metric)} Latency (ms)')
    ax2.set_title(f'High-priority {get_metric_str(metric)} Latency Over Time')
    ax2.legend()

    ax3.plot(throughput_interval_timestamps, baseline_interval_throughputs, linestyle='-', color=colors[0], label="Baseline")
    # ax3.plot(throughput_interval_timestamps, time_sliced_interval_throughputs, linestyle='-', color=colors[1], label="Time-sliced")
    # ax3.plot(throughput_interval_timestamps, mps_interval_throughputs, linestyle='-', color=colors[2], label="MPS")
    # ax3.plot(throughput_interval_timestamps, mps_priority_interval_throughputs, linestyle='-', color=colors[3], label="MPS-Priority")
    for idx, tally_interval_throughputs in enumerate(tally_interval_throughputs_list):
        ax3.plot(throughput_interval_timestamps, tally_interval_throughputs, linestyle='-', color=colors[4 + idx], label=f"Tally-Config-{idx}")
    
    ax3.set_ylabel(f'Throughput (Iterations/sec)')
    ax3.set_title(f'Best-effort Throughput Over Time')
    ax3.legend()

    ax3.set_xlabel("Timestamp (s)")

    fig.savefig(f"{out_directory}/azure_simulation_{metric}.png")