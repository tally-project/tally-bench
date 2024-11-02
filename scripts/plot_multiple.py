import os
import sys
import matplotlib.pyplot as plt

sys.path.append("python")

from bench_utils.utils import load_json_from_file, compute_avg, compute_percentile, mkdir_if_not_exists

tally_bench_result_dir = "tally_results"
result_file = f"{tally_bench_result_dir}/result_multiple.json"
result = load_json_from_file(result_file)["tally_priority"]

multi_res = {}

for exp_key in result.keys():

    measurement = result[exp_key]["measurements"][0]

    jobs = list(measurement.keys())
    jobs.remove("tally_config")
    jobs.remove("metrics")
    num_jobs = len(jobs)
    multi_res[num_jobs] = {"high_priority_99th_latency": 0, "throughputs": []}

    for job in jobs:

        job_res = measurement[job]
        priority = job_res["priority"]

        if priority > 1:
            multi_res[num_jobs]["high_priority_99th_latency"] = compute_percentile(job_res["latencies"], 99)
        multi_res[num_jobs]["throughputs"].append(job_res["iters"])

        multi_res[num_jobs]["throughputs"].sort(reverse=True)

num_jobs_lst = list(range(1, 11))
latencies = []
throughputs = []

for i in num_jobs_lst:
    latency = multi_res[i]["high_priority_99th_latency"]
    throughput = sum(multi_res[i]["throughputs"])

    latencies.append(latency)
    throughputs.append(throughput)

# Create a figure and axis
fig, ax1 = plt.subplots()

# Plot the first line with the primary y-axis
ax1.plot(num_jobs_lst, latencies, 'b-', marker="o", label="High Priority Latency")
ax1.set_xlabel('# Workloads')
ax1.set_ylabel('Latency (ms)')
ax1.tick_params(axis='y')
yticks = [0, 1, 2, 3, 4, 5]
ax1.set_yticks(yticks)
ax1.set_yticklabels([f'{tick:.1f}' for tick in yticks])

# Create a secondary y-axis and plot the second line
ax2 = ax1.twinx()
ax2.plot(num_jobs_lst, throughputs, 'r-', marker=".", label='System Throughput')
ax2.set_ylabel('# Requests / min')
ax2.tick_params(axis='y')

plt.legend()
fig.tight_layout()

mkdir_if_not_exists(f"tally_results/multiple_workloads")
plt.savefig(f"tally_results/multiple_workloads/multiple.png")