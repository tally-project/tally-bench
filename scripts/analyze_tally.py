import sys
import pandas as pd

sys.path.append("python")

from bench_utils.utils import write_json_to_file

tally_config_best_counts = []


def mark_best_tally_config(tally_config, bench_id):
    found = False
    for idx, (config, _, _) in enumerate(tally_config_best_counts):
        if tally_config == config:
            tally_config_best_counts[idx][1] += 1

            if bench_id not in tally_config_best_counts[idx][2]:
                tally_config_best_counts[idx][2].append(bench_id)

            found = True
            break
    if not found:
        tally_config_best_counts.append([tally_config, 1, [bench_id]])


def analyze_tally_slo_performance(priority_df, high_priority_job, best_effort_jobs, metric="avg", tolerance_level=0.1):
    
    positive_res = {}
    negative_res = {}
    high_priority_job_df = priority_df[priority_df["high_priority_job"] == high_priority_job]

    def get_tally_config(measurement):
        config = {}
        for parameter in [
            "preemption_latency_limit",
            "min_wait_time",
            "use_original_configs",
            "use_space_share",
            "wait_time_to_use_original",
        ]:
            config[parameter] = measurement.get(parameter, "default")
        return config

    for best_effort_job in best_effort_jobs:
        best_effort_job_df = high_priority_job_df[high_priority_job_df["best_effort_job"] == best_effort_job]

        if best_effort_job_df.empty:
            continue

        baseline_latency = best_effort_job_df[f"high_priority_orig_{metric}_latency"].values[0]
        acceptable_latency_bound = (1 + tolerance_level) * baseline_latency

        best_effort_job_df = best_effort_job_df.sort_values(by=f"high_priority_tally_{metric}_latency")
        tally_acceptable_df = best_effort_job_df[best_effort_job_df[f"high_priority_tally_{metric}_latency"] <= acceptable_latency_bound]

        bench_id = f"{best_effort_job}_{high_priority_job}"

        if tally_acceptable_df.empty:
            closest_measurement = best_effort_job_df.iloc[0].to_dict()

            tally_latency = closest_measurement[f"high_priority_tally_{metric}_latency"]
            tally_throughput = closest_measurement[f"best_effort_tally_throughput"]
            priority_tally_throughput = closest_measurement[f"high_priority_tally_throughput"]
            best_achievable_throughput = closest_measurement[f"best_effort_tally_throughput"]

            negative_res[bench_id] = {
                "baseline_latency": baseline_latency,
                "tally_latency": tally_latency,
                "tally_throughput": tally_throughput,
                "priority_tally_throughput": priority_tally_throughput,
                "tally_config": get_tally_config(closest_measurement)
            }
        else:
            best_achievable_throughput = tally_acceptable_df[f"best_effort_tally_throughput"].max()
            best_measurement = tally_acceptable_df[tally_acceptable_df[f"best_effort_tally_throughput"] == best_achievable_throughput].iloc[0].to_dict()
            tally_latency = best_measurement[f"high_priority_tally_{metric}_latency"]
            tally_throughput = best_measurement[f"best_effort_tally_throughput"]
            priority_tally_throughput = best_measurement[f"high_priority_tally_throughput"]
            tally_config = get_tally_config(best_measurement)
            mark_best_tally_config(tally_config, bench_id)

            positive_res[bench_id] = {
                "baseline_latency": baseline_latency,
                "tally_latency": tally_latency,
                "tally_throughput": tally_throughput,
                "priority_tally_throughput": priority_tally_throughput,
                "tally_config": tally_config
            }

    return positive_res, negative_res

if __name__ == "__main__":
    priority_df = pd.read_csv("tally_bench_results/priority-aware-perf.csv")
    high_priority_jobs = priority_df["high_priority_job"].unique()
    high_priority_jobs = [high_priority_job for high_priority_job in high_priority_jobs if "server" in high_priority_job]
    best_effort_jobs = priority_df["best_effort_job"].unique()

    metrics = ["avg", "90th", "95th", "99th"]
    tolerance_levels = [0.1]

    positive_res = {}
    negative_res = {}

    for high_priority_job in high_priority_jobs:
        for metric in metrics:

            if metric not in positive_res:
                positive_res[metric] = {}
                negative_res[metric] = {}

            for tolerance_level in tolerance_levels:

                if tolerance_level not in positive_res[metric]:
                    positive_res[metric][tolerance_level] = {}
                    negative_res[metric][tolerance_level] = {}

                pos, neg = analyze_tally_slo_performance(priority_df, high_priority_job, best_effort_jobs, metric, tolerance_level)
      
                positive_res[metric][tolerance_level].update(pos)
                negative_res[metric][tolerance_level].update(neg)

    write_json_to_file(positive_res, "tally_bench_results/postive_results.json")
    write_json_to_file(negative_res, "tally_bench_results/negative_results.json")

    write_json_to_file(tally_config_best_counts, "tally_bench_results/tally_config_best_counts.json")