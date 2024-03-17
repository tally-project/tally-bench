import sys

sys.path.append('python')

from bench_utils.utils import load_json_from_file, mkdir_if_not_exists
from bench_utils.plot_azure import plot_latency_over_time, plot_throughput_over_time
from bench_utils.trace import plot_server_trace

def main():
    tally_bench_result_dir = "tally_bench_results"
    plot_directory = f"{tally_bench_result_dir}/plots"
    mkdir_if_not_exists(plot_directory)

    result_file = f"{tally_bench_result_dir}/azure_result.json"
    result = load_json_from_file(result_file)

    plot_latency_over_time(result)
    plot_throughput_over_time(result)

    azure_trace = load_json_from_file("infer_trace/azure_trace_for_bert.json")
    plot_server_trace(azure_trace, interval=1, server_throughput=250, output_file="tally_bench_results/plots/azure_trace.png")


if __name__ == "__main__":
    main()