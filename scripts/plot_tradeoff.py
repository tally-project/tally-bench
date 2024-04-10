import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.path.append('python')

from bench_utils.utils import mkdir_if_not_exists


def plot_latency_throughput_tradeoff(priority_df, exp_key, metric="avg", out_directory="tally_bench_results/plots"):

    savefig_dir = f"{out_directory}/latency_throughput_tradeoff/{metric}"
    mkdir_if_not_exists(savefig_dir)

    exp_key_df = priority_df[priority_df["exp_key"] == exp_key]
    # exp_key_df = exp_key_df[(exp_key_df["use_space_share"] == True) & (exp_key_df["space_share_max_sm_perc"] > 0)]

    if exp_key_df.empty:
        return

    # exp_key_df = exp_key_df.sort_values(by="space_share_max_sm_perc")

    high_priority_latencies = []
    high_priority_throughputs = []
    best_effort_throughputs = []
    space_share_max_sm_perc_lst = []

    for index, row in exp_key_df.iterrows():

        tally_high_priority_latency_norm = row[f"high_priority_tally_{metric}_latency"] / row[f"high_priority_orig_{metric}_latency"]
        tally_high_priority_throughput = row[f"high_priority_tally_throughput"]
        tally_best_effort_throughput = row[f"best_effort_tally_throughput"]
        space_share_max_sm_perc = row[f"space_share_max_sm_perc"]

        high_priority_latencies.append(tally_high_priority_latency_norm)
        high_priority_throughputs.append(tally_high_priority_throughput)
        best_effort_throughputs.append(tally_best_effort_throughput)
        space_share_max_sm_perc_lst.append(space_share_max_sm_perc)

    def keep_frontier_only(x_values, y_values):
        # Combine the x and y values into a list of tuples
        tuples = list(zip(x_values, y_values))
        
        # Filter tuples
        filtered_tuples = [t for t in tuples if not any(u[0] > t[0] and u[1] > t[1] for u in tuples)]
        left_x_values = [tuple[0] for tuple in  filtered_tuples]
        left_y_values = [tuple[1] for tuple in  filtered_tuples]

        return left_x_values, left_y_values

    best_effort_throughputs, high_priority_throughputs = keep_frontier_only(best_effort_throughputs, high_priority_throughputs)

    zipped_list = zip(best_effort_throughputs, high_priority_throughputs)
    zipped_list = sorted(zipped_list)
    best_effort_throughputs, high_priority_throughputs = zip(*zipped_list)

    # plotting
    plt.clf()

    plt.figure(figsize=(10, 6))
    # plt.plot(space_share_max_sm_perc_lst, high_priority_latencies, marker="o", label='latency')
    # plt.plot(space_share_max_sm_perc_lst, high_priority_throughputs, marker="s", label='high-priority throughput')
    # plt.plot(space_share_max_sm_perc_lst, best_effort_throughputs, marker="^", label='best-effort throughput', linestyle='--')

    plt.plot(best_effort_throughputs, high_priority_throughputs, marker="^", linestyle='--')

    plt.ylabel('High-Priority Throughput')
    plt.xlabel('Best-Effort Throughput')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{savefig_dir}/{exp_key}.png")


def main():
    plot_directory = "tally_bench_results/plots"
    mkdir_if_not_exists(plot_directory)

    priority_df = pd.read_csv("tally_bench_results/priority-aware-perf.csv")
    exp_keys = priority_df["exp_key"].unique()
    exp_keys = [exp_key for exp_key in exp_keys if "single-stream" in exp_key]
    exp_keys = [exp_key for exp_key in exp_keys if "onnxruntime_bert" in exp_key]
    
    metrics = ["avg"]

    for exp_key in exp_keys:
        for metric in metrics:
            plot_latency_throughput_tradeoff(priority_df, exp_key, metric)


if __name__ == "__main__":
    main()