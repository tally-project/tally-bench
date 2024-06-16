import sys
import pandas as pd

sys.path.append("python")

from bench_utils.utils import write_json_to_file
from bench_utils.parse import get_slo_comparison_data
from bench_utils.plot import tally_default_config

if __name__ == "__main__":
    priority_df = pd.read_csv("tally_bench_results/priority-aware-perf.csv")
    high_priority_jobs = priority_df["high_priority_job"].unique()
    high_priority_jobs = [high_priority_job for high_priority_job in high_priority_jobs if "server" in high_priority_job]
    best_effort_jobs = priority_df["best_effort_job"].unique()

    metric = "99th"

    count_pairs = 0

    tally_avg_latency_slowdown = 0.
    tally_avg_sys_throughput = 0.

    for high_priority_job in high_priority_jobs:

        if "load_0.5" not in high_priority_job or "inception" in high_priority_job:
            continue

        data = get_slo_comparison_data(priority_df, high_priority_job, best_effort_jobs, tally_default_config, metric=metric)
        
        baseline_latencies = data["baseline_latencies"]
        tally_latencies = data["tally_latencies"]
        priority_tally_throughputs = data["priority_tally_throughputs"]
        tally_throughputs = data["tally_throughputs"]
        
        tally_latency_slowdowns = [x / y for x, y in zip(tally_latencies, baseline_latencies)]
        tally_system_throughputs = [x + y for x, y in zip(priority_tally_throughputs, tally_throughputs)]

        tally_avg_latency_slowdown += sum(tally_latency_slowdowns)
        tally_avg_sys_throughput += sum(tally_system_throughputs)

        count_pairs += len(tally_latency_slowdowns)
    
    tally_avg_latency_slowdown /= count_pairs
    tally_avg_sys_throughput /= count_pairs

    print(f"tally_avg_latency_slowdown: {tally_avg_latency_slowdown}")
    print(f"tally_avg_sys_throughput: {tally_avg_sys_throughput}")
        