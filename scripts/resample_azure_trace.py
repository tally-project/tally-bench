import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

sys.path.append('python')

from bench_utils.utils import write_json_to_file
from bench_utils.trace import resample_azure_trace, plot_server_trace

if __name__ == "__main__":
    trace_file_name = "infer_trace/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"

    # generate 1 day trace
    generated_trace = resample_azure_trace(trace_file_name, start_day=3, end_day=4, scaled_dur_secs=600)
    plot_server_trace(generated_trace, interval=1, server_throughput=250, output_file="azure_trace_one_day.png")
    write_json_to_file(generated_trace, "infer_trace/azure_trace_for_bert_one_day.json")

    # generate 2 week trace
    generated_trace = resample_azure_trace(trace_file_name, start_day=0, end_day=14, scaled_dur_secs=600 * 14)
    plot_server_trace(generated_trace, interval=1, server_throughput=250, output_file="azure_trace_two_week.png")
    write_json_to_file(generated_trace, "infer_trace/azure_trace_for_bert_two_week.json")
