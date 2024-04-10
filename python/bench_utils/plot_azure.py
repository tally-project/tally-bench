import math
import numpy as np
import matplotlib.pyplot as plt

from bench_utils.utils import compute_avg, compute_percentile, mkdir_if_not_exists
from bench_utils.plot import get_metric_str, get_best_effort_job_label

# infer_bench_id = "onnxruntime_bert_infer_server_1"
# train_bench_id = "pytorch_bert_train_32"

# pair_bench_id = "pytorch_bert_train_32_onnxruntime_bert_infer_server_1"
# best_effort_key = "pytorch_bert_train_32_0"
# high_priority_key = "onnxruntime_bert_infer_server_1_1"

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'xkcd:light purple', 'tab:olive', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:purple', 'tab:cyan', 'xkcd:sky blue', 'xkcd:light green', 'xkcd:light red', 'xkcd:light purple', 'xkcd:light brown', 'xkcd:light pink', 'xkcd:light gray', 'xkcd:light olive', 'xkcd:light cyan']


def plot_azure_trace_simulation(
    trace_timestamps,
    result,
    train_bench_id,
    infer_bench_id,
    trace_interval=1,
    latency_interval=2,
    throughput_interval=5,
    server_throughput=None,
    metric="95th",
    out_directory=".",
    out_filename=None,
):
    save_dir = f"{out_directory}/{metric}"
    mkdir_if_not_exists(save_dir)

    pair_bench_id = f"{train_bench_id}_{infer_bench_id}"
    best_effort_key = f"{train_bench_id}_0"
    high_priority_key = f"{infer_bench_id}_1"

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

    print(f"Metric: {metric}")
    print(f"avg baseline_latencies: {compute_avg(baseline_latencies)}")
    print(f"avg time_sliced_latencies: {compute_avg(time_sliced_latencies)}")
    print(f"avg mps_latencies: {compute_avg(mps_latencies)}")
    print(f"avg mps_priority_latencies: {compute_avg(mps_priority_latencies)}")
    print(f"avg tally_latencies: {compute_avg(tally_infer_timestamps_latencies_list[0][1])}")

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
            if job_key not in measurement:
                print(measurement.keys())
            end_timestamps = measurement[job_key]["end_timestamps"][1:]
            lst.append(end_timestamps)
        return lst

    baseline_train_timestamps = get_train_timestamps_list(default_res, train_bench_id, f"{train_bench_id}_0")[0]
    time_sliced_train_timestamps = get_train_timestamps_list(default_res, pair_bench_id, best_effort_key)[0]
    mps_train_timestamps = get_train_timestamps_list(mps_res, pair_bench_id, best_effort_key)[0]
    mps_priority_train_timestamps = get_train_timestamps_list(mps_priority_res, pair_bench_id, best_effort_key)[0]
    tally_train_timestamps_list = get_train_timestamps_list(tally_priority_res, pair_bench_id, best_effort_key)
    
    print(f"len(baseline_train_timestamps): {len(baseline_train_timestamps)}")
    print(f"len(time_sliced_train_timestamps): {len(time_sliced_train_timestamps)}")
    print(f"len(mps_train_timestamps): {len(mps_train_timestamps)}")
    print(f"len(mps_priority_train_timestamps): {len(mps_priority_train_timestamps)}")
    print(f"len(tally_train_timestamps_list[0]): {len(tally_train_timestamps_list[0])}")


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
        ax2.plot(latency_interval_timestamps, tally_interval_statistics, linestyle='-', color=colors[4 + idx], label=f"Tally")
        break

    ax2.set_ylabel(f'{get_metric_str(metric)} Latency (ms)')
    ax2.set_title(f'High-priority {get_metric_str(metric)} Latency Over Time')
    ax2.legend()

    ax3.plot(throughput_interval_timestamps, baseline_interval_throughputs, linestyle='-', color=colors[0], label="Baseline")
    # ax3.plot(throughput_interval_timestamps, time_sliced_interval_throughputs, linestyle='-', color=colors[1], label="Time-sliced")
    # ax3.plot(throughput_interval_timestamps, mps_interval_throughputs, linestyle='-', color=colors[2], label="MPS")
    # ax3.plot(throughput_interval_timestamps, mps_priority_interval_throughputs, linestyle='-', color=colors[3], label="MPS-Priority")
    for idx, tally_interval_throughputs in enumerate(tally_interval_throughputs_list):
        ax3.plot(throughput_interval_timestamps, tally_interval_throughputs, linestyle='-', color=colors[4 + idx], label=f"Tally")
        break

    ax3.set_ylabel(f'Throughput (Iterations/sec)')
    ax3.set_title(f'Best-effort Throughput Over Time')
    ax3.legend()

    ax3.set_xlabel("Timestamp (s)")

    if out_filename is None:
        out_filename = train_bench_id

    fig.savefig(f"{save_dir}/{out_filename}.png")

def plot_azure_slo_comparison_system_throughput(
    result,
    train_bench_ids,
    infer_bench_id,
    metric="95th",
    out_directory=".",
    out_filename=None
):
    
    save_dir = f"{out_directory}/{metric}"
    mkdir_if_not_exists(save_dir)

    baseline_latencies = []
    time_sliced_latencies = []
    mps_latencies = []
    mps_priority_latencies = []
    tally_latencies = []
    priority_time_sliced_throughputs = []
    priority_mps_throughputs = []
    priority_mps_priority_throughputs = []
    priority_tally_throughputs = []
    time_sliced_throughputs = []
    mps_throughputs = []
    mps_priority_throughputs = []
    tally_throughputs = []

    for train_bench_id in train_bench_ids:

        pair_bench_id = f"{train_bench_id}_{infer_bench_id}"
        best_effort_key = f"{train_bench_id}_0"
        high_priority_key = f"{infer_bench_id}_1"

        default_res = result["default"]
        mps_res = result["mps"]
        mps_priority_res = result["mps-priority"]
        tally_priority_res = result["tally_priority"]

        def compute_metric_latency(latencies):
            if "avg" in metric:
                metric_latency = compute_avg(latencies)
            elif "th" in metric:
                percentile = int(metric.split("th")[0])
                metric_latency = compute_percentile(latencies, percentile)
            return metric_latency

        baseline_high_priority_measurement = default_res[infer_bench_id]["measurements"][0][f"{infer_bench_id}_0"]
        baseline_latency = compute_metric_latency(baseline_high_priority_measurement["latencies"][1:])
        baseline_high_priority_throughput = baseline_high_priority_measurement["iters"] / baseline_high_priority_measurement["time_elapsed"]

        baseline_best_effort_measurement = default_res[train_bench_id]["measurements"][0][f"{train_bench_id}_0"]
        baseline_best_effort_throughput = baseline_best_effort_measurement["iters"] / baseline_best_effort_measurement["time_elapsed"]

        def parse_res(res):
            measurement = res[pair_bench_id]["measurements"][0]
            latencies = measurement[high_priority_key]["latencies"][1:]
            metric_latency = compute_metric_latency(latencies)
            high_priority_throughput = (measurement[high_priority_key]["iters"] / measurement[high_priority_key]["time_elapsed"]) / baseline_high_priority_throughput
            best_effort_throughput = (measurement[best_effort_key]["iters"] / measurement[best_effort_key]["time_elapsed"]) / baseline_best_effort_throughput
            return metric_latency, high_priority_throughput, best_effort_throughput
        
        time_sliced_latency, time_sliced_high_priority_throughput, time_sliced_best_effort_throughput = parse_res(default_res)
        mps_latency, mps_high_priority_throughput, mps_best_effort_throughput = parse_res(mps_res)
        mps_priority_latency, mps_priority_high_priority_throughput, mps_priority_best_effort_throughput = parse_res(mps_priority_res)
        tally_latency, tally_high_priority_throughput, tally_best_effort_throughput = parse_res(tally_priority_res)
    
        baseline_latencies.append(baseline_latency)
        time_sliced_latencies.append(time_sliced_latency)
        mps_latencies.append(mps_latency)
        mps_priority_latencies.append(mps_priority_latency)
        tally_latencies.append(tally_latency)
        priority_time_sliced_throughputs.append(time_sliced_high_priority_throughput)
        priority_mps_throughputs.append(mps_high_priority_throughput)
        priority_mps_priority_throughputs.append(mps_priority_high_priority_throughput)
        priority_tally_throughputs.append(tally_high_priority_throughput)
        time_sliced_throughputs.append(time_sliced_best_effort_throughput)
        mps_throughputs.append(mps_best_effort_throughput)
        mps_priority_throughputs.append(mps_priority_best_effort_throughput)
        tally_throughputs.append(tally_best_effort_throughput)

    time_sliced_system_throughputs = [x + y for x, y in zip(priority_time_sliced_throughputs, time_sliced_throughputs)]
    mps_system_throughputs = [x + y for x, y in zip(priority_mps_throughputs, mps_throughputs)]
    mps_priority_system_throughputs = [x + y for x, y in zip(priority_mps_priority_throughputs, mps_priority_throughputs)]
    tally_system_throughputs = [x + y for x, y in zip(priority_tally_throughputs, tally_throughputs)]

    def compute_avg_slowdown(lst1, lst2):
        slowdowns = [b / a for a, b in zip(lst1, lst2)]
        avg_slowdown = compute_avg(slowdowns)
        return avg_slowdown

    print(f"Metric: {metric}")
    print(f"avg time_sliced slowdown: {compute_avg_slowdown(baseline_latencies, time_sliced_latencies)}")
    print(f"avg mps slowdown: {compute_avg_slowdown(baseline_latencies, mps_latencies)}")
    print(f"avg mps_priority slowdown: {compute_avg_slowdown(baseline_latencies, mps_priority_latencies)}")
    print(f"avg tally slowdown: {compute_avg_slowdown(baseline_latencies, tally_latencies)}")

    print(f"avg time_sliced_system_throughputs: {compute_avg(time_sliced_system_throughputs)}")
    print(f"avg mps_system_throughputs: {compute_avg(mps_system_throughputs)}")
    print(f"avg mps_priority_system_throughputs: {compute_avg(mps_priority_system_throughputs)}")
    print(f"avg tally_system_throughputs: {compute_avg(tally_system_throughputs)}")

    # plotting
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(len(train_bench_ids) * 1, 5), 8), sharex=True)

    # Width of a bar
    width = 0.15
    
    pos = np.arange(len(train_bench_ids))

    # plot latency
    ax1.bar(pos, baseline_latencies, width, label=f"Baseline", color=colors[0], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 1 * width, time_sliced_latencies, width, label=f"Time-Sliced", color=colors[1], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 2 * width, mps_latencies, width, label=f"MPS", color=colors[2], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 3 * width, mps_priority_latencies, width, label=f"MPS-Priority", color=colors[3], alpha=0.7, edgecolor='black')
    ax1.bar(pos + 4 * width, tally_latencies, width, label=f"Tally", color=colors[4], alpha=0.7, edgecolor='black')

    ax1.set_title(f"High-Priority {get_metric_str(metric)} Latency Comparison")
    ax1.set_ylabel(f"Latency (ms)")
    ax1.legend()

    # plot system throughput
    ax2.bar(pos + 1 * width, time_sliced_system_throughputs, width,  label=f"Time-sliced", color=colors[1], alpha=0.7, edgecolor='black')
    ax2.bar(pos + 2 * width, mps_system_throughputs, width,  label=f"MPS", color=colors[2], alpha=0.7, edgecolor='black')
    ax2.bar(pos + 3 * width, mps_priority_system_throughputs, width, label=f"MPS-Priority", color=colors[3], alpha=0.7, edgecolor='black')
    ax2.bar(pos + 4 * width, tally_system_throughputs, width,  label=f"Tally", color=colors[4], alpha=0.7, edgecolor='black')

    ax2.set_title("System Throughput Comparison")
    ax2.set_ylabel(f"System Throughput")

    num_targets = 5
    ax2.set_xticks(pos + ((num_targets - 1) / 2) * width)
    ax2.set_xticklabels([get_best_effort_job_label(best_effort_job, break_lines=True) for best_effort_job in train_bench_ids])
    ax2.set_xlabel("Best-Effort Jobs")

    if out_filename is None:
        out_filename = infer_bench_id

    plt.savefig(f"{save_dir}/{out_filename}.png")