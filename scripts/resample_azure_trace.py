import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

sys.path.append('python')

from bench_utils.utils import write_json_to_file
from bench_utils.trace import scale_azure_trace, generate_azure_trace_with_load, plot_server_trace

if __name__ == "__main__":
    trace_file_name = "infer_trace/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
    
    # trace should be at most 600 seconds
    max_trace_span = 600

    target_loads = [0.25, 0.5, 0.75]

    # ========== For trace analysis ===============

    # # generate 1 day trace
    # generated_trace = scale_azure_trace(trace_file_name, start_day=3, end_day=4, scaled_dur_secs=600)
    # plot_server_trace(generated_trace, interval=1, server_throughput=250, output_file="azure_trace_one_day.png")
    # write_json_to_file(generated_trace, "infer_trace/azure_trace_for_bert_one_day.json")

    # generate 2 week trace
    generated_trace = scale_azure_trace(trace_file_name, start_day=0, end_day=14, scaled_dur_secs=600 * 14)
    plot_server_trace(generated_trace, interval=1, server_throughput=250, output_file="azure_trace_two_week.png")
    write_json_to_file(generated_trace, "infer_trace/azure_trace_for_bert_two_week.json")

    # ========== For traffic load analysis ===============
    
    # single_job_df = pd.read_csv("tally_results/single-job-perf.csv")
    # single_job_df = single_job_df[single_job_df["workload_type"] == "inference-single-stream"]
    # inference_jobs = single_job_df["exp_key"].unique()
    # for inference_job in inference_jobs:

    #     inference_job_name = inference_job.replace("_infer_single-stream_1", "")
    #     req_latency_ms = single_job_df[single_job_df["exp_key"] == inference_job]["original_avg_latency"].values[0]
    #     req_latency_s = req_latency_ms / 1000

    #     for load in target_loads:
    #         generated_trace = generate_azure_trace_with_load(trace_file_name, req_latency_s, max_trace_span, start_day=3, end_day=4, target_load=load)
    #         write_json_to_file(generated_trace, f"infer_trace/azure_trace_{inference_job_name}_load_{load}.json")
