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

    result_file_one_day = f"{tally_bench_result_dir}/azure_result_one_day.json"
    result_file_two_week = f"{tally_bench_result_dir}/azure_result_two_week.json"

    result_one_day = load_json_from_file(result_file_one_day)
    result_two_week = load_json_from_file(result_file_two_week)

    azure_trace_one_day = load_json_from_file("infer_trace/azure_trace_for_bert_one_day.json")
    azure_trace_two_week = load_json_from_file("infer_trace/azure_trace_for_bert_two_week.json")

    metrics = ["90th", "95th", "99th", "avg"]

    infer_bench_id = "onnxruntime_bert_infer_server_1"
    train_bench_ids = {
        "pytorch_resnet50_train_128_amp": {},
        "pytorch_resnet50_train_128": {},
        "pytorch_pointnet_train_128_amp": {},
        "pytorch_pointnet_train_128": {},
        "pytorch_bert_train_32_amp": {},
        "pytorch_bert_train_32": {
            "latency_interval": 0.5,
            "throughput_interval": 12
        },
        "pytorch_pegasus-x-base_train_8_amp": {},
        "pytorch_pegasus-x-base_train_8": {},
        "pytorch_whisper-large-v3_train_16_amp": {},
        "pytorch_whisper-large-v3_train_16": {},
        "pytorch_gpt2-large_train_1_amp": {},
        "pytorch_gpt2-large_train_1": {},
    }

    trace_interval = 1
    latency_interval = 2
    throughput_interval = 5

    server_throughput=250

    for metric in metrics:
        for train_bench_id in train_bench_ids:

            bench_latency_interval = train_bench_ids[train_bench_id].get("latency_interval", latency_interval) 
            bench_throughput_interval = train_bench_ids[train_bench_id].get("throughput_interval", throughput_interval)

            plot_azure_trace_simulation(
                azure_trace_one_day,
                result_one_day,
                train_bench_id,
                infer_bench_id,
                server_throughput=server_throughput,
                out_directory=plot_directory,
                metric=metric,
                trace_interval=trace_interval,
                latency_interval=bench_latency_interval,
                throughput_interval=bench_throughput_interval,
                out_filename=f"{train_bench_id}_one_day"
            )

        plot_azure_trace_simulation(
            azure_trace_two_week,
            result_two_week,
            "pytorch_bert_train_32",
            infer_bench_id,
            server_throughput=server_throughput,
            out_directory=plot_directory,
            metric=metric,
            trace_interval=3,
            latency_interval=5,
            throughput_interval=30,
            out_filename=f"pytorch_bert_train_32_two_week"
        )

        plot_azure_slo_comparison_system_throughput(
            result_one_day,
            train_bench_ids,
            infer_bench_id,
            metric=metric,
            out_directory=f"{tally_bench_result_dir}/plots/azure_slo_comparison",
            out_filename=infer_bench_id
        )

if __name__ == "__main__":
    main()