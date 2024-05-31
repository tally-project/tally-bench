import sys

sys.path.append('python')

from bench_utils.utils import load_json_from_file, mkdir_if_not_exists
from bench_utils.plot_azure import plot_azure_trace_simulation, plot_azure_slo_comparison_system_throughput

def main():
    tally_bench_result_dir = "tally_bench_results"
    plot_directory = f"{tally_bench_result_dir}/plots"
    mkdir_if_not_exists(plot_directory)

    plot_directory = f"{plot_directory}/azure_simulation"
    mkdir_if_not_exists(plot_directory)

    result_file_two_week = f"{tally_bench_result_dir}/azure_result_two_week.json"
    result_two_week = load_json_from_file(result_file_two_week)
    azure_trace_two_week = load_json_from_file("infer_trace/azure_trace_for_bert_two_week.json")

    metrics = ["99th"]
    server_throughput=250

    for metric in metrics:

        plot_azure_trace_simulation(
            azure_trace_two_week,
            result_two_week,
            "pytorch_bert_train_32",
            "onnxruntime_bert_infer_server_1",
            server_throughput=server_throughput,
            out_directory=plot_directory,
            metric=metric,
            trace_interval=3,
            latency_interval=10,
            throughput_interval=30,
            out_filename=f"pytorch_bert_train_32_two_week"
        )

if __name__ == "__main__":
    main()