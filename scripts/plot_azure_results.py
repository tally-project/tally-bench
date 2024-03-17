import sys

sys.path.append('python')

from bench_utils.utils import load_json_from_file, mkdir_if_not_exists
from bench_utils.plot_azure import plot_azure_trace_simulation

def main():
    tally_bench_result_dir = "tally_bench_results"
    plot_directory = f"{tally_bench_result_dir}/plots"
    mkdir_if_not_exists(plot_directory)

    result_file = f"{tally_bench_result_dir}/azure_result.json"
    result = load_json_from_file(result_file)

    azure_trace = load_json_from_file("infer_trace/azure_trace_for_bert.json")

    plot_azure_trace_simulation(
        azure_trace,
        result,
        server_throughput=250,
        out_directory=plot_directory
    )

if __name__ == "__main__":
    main()