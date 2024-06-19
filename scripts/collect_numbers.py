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
    avg_latency_slowdown_dict = {}
    avg_sys_throughput_dict = {}
    max_latency_slowdown_dict = {}
    min_latency_slowdown_dict = {}

    systems = ["time_sliced", "mps", "mps_priority", "tgs", "tally"]

    for high_priority_job in high_priority_jobs:

        if "load_0.5" not in high_priority_job or "inception" in high_priority_job:
            continue

        data = get_slo_comparison_data(priority_df, high_priority_job, best_effort_jobs, tally_default_config, metric=metric)
        baseline_latencies = data["baseline_latencies"]
        
        for system in systems:
            latencies = data[f"{system}_latencies"]
            priority_throughputs = data[f"priority_{system}_throughputs"]
            throughputs = data[f"{system}_throughputs"]
            
            latency_slowdowns = [x / y for x, y in zip(latencies, baseline_latencies)]
            system_throughputs = [x + y for x, y in zip(throughputs, priority_throughputs)]

            if system not in avg_latency_slowdown_dict:
                avg_latency_slowdown_dict[system] = 0
                avg_sys_throughput_dict[system] = 0
                max_latency_slowdown_dict[system] = 0
                min_latency_slowdown_dict[system] = 1000

            if system == "tally":
                print(high_priority_job)
                print(latency_slowdowns)

            avg_latency_slowdown_dict[system] += sum(latency_slowdowns)
            avg_sys_throughput_dict[system] += sum(system_throughputs)
            max_latency_slowdown_dict[system] = max(max(latency_slowdowns), max_latency_slowdown_dict[system])
            min_latency_slowdown_dict[system] = min(min(latency_slowdowns), min_latency_slowdown_dict[system])

        count_pairs += len(latency_slowdowns)
    
    for system in systems:
        avg_latency_slowdown_dict[system] /= count_pairs
        avg_sys_throughput_dict[system] /= count_pairs

        print(f"System: {system}")
        print(f"avg_latency_slowdown: {avg_latency_slowdown_dict[system]}")
        print(f"avg_sys_throughput: {avg_sys_throughput_dict[system]}")
        print(f"max_latency_slowdown: {max_latency_slowdown_dict[system]}")
        print(f"min_latency_slowdown: {min_latency_slowdown_dict[system]}")
        print()
        