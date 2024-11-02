import sys
import pandas as pd

sys.path.append("python")

from bench_utils.utils import write_json_to_file
from bench_utils.parse import get_slo_comparison_data
from bench_utils.plot import tally_default_config

def collect_end_to_end():
    priority_df = pd.read_csv("tally_results/priority-aware-perf.csv")
    high_priority_jobs = priority_df["high_priority_job"].unique()
    high_priority_jobs = [high_priority_job for high_priority_job in high_priority_jobs if "server" in high_priority_job]
    best_effort_jobs = priority_df["best_effort_job"].unique()

    metric = "99th"

    latency_slowdown_dict = {}
    sys_throughput_dict = {}

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

            if system not in latency_slowdown_dict:
                latency_slowdown_dict[system] = []
                sys_throughput_dict[system] = []

            latency_slowdown_dict[system].extend(latency_slowdowns)
            sys_throughput_dict[system].extend(system_throughputs)

    for system in systems:

        latency_slowdowns = latency_slowdown_dict[system]
        sys_throughputs = sys_throughput_dict[system]
        
        avg_latency_slowdown = sum(latency_slowdowns) / len(latency_slowdowns)
        max_latency_slowdown = max(latency_slowdowns)
        min_latency_slowdown = min(latency_slowdowns)
        avg_sys_throughput = sum(sys_throughputs) / len(sys_throughputs)

        print(f"System: {system}")
        print(f"avg_latency_slowdown: {avg_latency_slowdown}")
        print(f"avg_sys_throughput: {avg_sys_throughput}")
        print(f"max_latency_slowdown: {max_latency_slowdown}")
        print(f"min_latency_slowdown: {min_latency_slowdown}")
        print()
        

def collect_azure():
    priority_df = pd.read_csv("tally_results/azure_priority-aware-perf.csv")
    high_priority_jobs = priority_df["high_priority_job"].unique()
    high_priority_jobs = [high_priority_job for high_priority_job in high_priority_jobs if "server" in high_priority_job]
    best_effort_jobs = priority_df["best_effort_job"].unique()

    metric = "99th"

    latency_slowdown_dict = {}
    sys_throughput_dict = {}

    systems = ["time_sliced", "mps", "mps_priority", "tgs", "tally"]

    for high_priority_job in high_priority_jobs:

        data = get_slo_comparison_data(priority_df, high_priority_job, best_effort_jobs, tally_default_config, metric=metric)
        baseline_latencies = data["baseline_latencies"]
        
        for system in systems:
            latencies = data[f"{system}_latencies"]
            priority_throughputs = data[f"priority_{system}_throughputs"]
            throughputs = data[f"{system}_throughputs"]

            latency_slowdowns = [x / y for x, y in zip(latencies, baseline_latencies)]
            system_throughputs = [x + y for x, y in zip(throughputs, priority_throughputs)]

            if system not in latency_slowdown_dict:
                latency_slowdown_dict[system] = []
                sys_throughput_dict[system] = []

            latency_slowdown_dict[system].extend(latency_slowdowns)
            sys_throughput_dict[system].extend(system_throughputs)

    for system in systems:

        latency_slowdowns = latency_slowdown_dict[system]
        sys_throughputs = sys_throughput_dict[system]
        
        avg_latency_slowdown = sum(latency_slowdowns) / len(latency_slowdowns)
        max_latency_slowdown = max(latency_slowdowns)
        min_latency_slowdown = min(latency_slowdowns)
        avg_sys_throughput = sum(sys_throughputs) / len(sys_throughputs)

        print(f"System: {system}")
        print(f"avg_latency_slowdown: {avg_latency_slowdown}")
        print(f"avg_sys_throughput: {avg_sys_throughput}")
        print(f"max_latency_slowdown: {max_latency_slowdown}")
        print(f"min_latency_slowdown: {min_latency_slowdown}")
        print()


if __name__ == "__main__":
    collect_azure()