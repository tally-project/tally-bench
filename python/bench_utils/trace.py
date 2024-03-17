import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


def resample_azure_trace(trace_file_name, start_day=3, end_day=4, scaled_dur_secs=600):

    trace_df = pd.read_csv(trace_file_name)

    # use the most frequent fn as the trace target
    fn_count = trace_df['func'].value_counts(ascending=False)
    most_frequent_fn = fn_count.keys()[0]
    target_fn = most_frequent_fn

    # parse fn arrival timestamps
    target_fn_df = trace_df[trace_df["func"] == target_fn]
    target_fn_arrival_ts = target_fn_df["end_timestamp"] - target_fn_df["duration"]
    target_fn_arrival_ts = target_fn_arrival_ts.to_numpy()
    target_fn_arrival_ts.sort()

    # filter timestamps based on days
    start_ts = start_day * 3600 * 24
    end_ts = end_day * 3600 * 24
    target_fn_arrival_ts = target_fn_arrival_ts[
                                (target_fn_arrival_ts >= start_ts) &
                                (target_fn_arrival_ts < end_ts)]

    first_ts = target_fn_arrival_ts[0]
    last_ts = target_fn_arrival_ts[-1]
    trace_dur = last_ts - first_ts

    # let trace start from zero
    target_fn_arrival_ts = target_fn_arrival_ts - first_ts

    # scale the trace to have duration `scaled_dur_secs`
    normalized_arrival_ts = target_fn_arrival_ts * (scaled_dur_secs / trace_dur)

    arrivial_ts_list = normalized_arrival_ts.tolist()[1:]
    return arrivial_ts_list


def plot_server_trace(timestamps, interval=5, output_file="azure_trace.png", server_throughput=None):

    last_ts = timestamps[-1]
    bins = np.arange(0, math.ceil(last_ts), interval)

    # Use histogram to count the number of values within each window
    counts, edges = np.histogram(timestamps, bins=bins)

    timestamps = edges[:-1] + (interval / 2)
    request_counts = counts / interval
    request_counts[request_counts == 0] = np.nan

    plt.figure(figsize=(20, 6))
    plt.plot(timestamps, request_counts, linestyle='-', color='g', alpha=0.7)
    plt.xlabel('Timestamp(s)')
    plt.ylabel('Request count')
    plt.title(f'Request count over time')

    if server_throughput:
        plt.axhline(y=server_throughput, color='r', linestyle='--', label='Server Throughput')

    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.savefig(output_file)