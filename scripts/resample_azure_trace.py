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
    generated_trace = resample_azure_trace(trace_file_name)
    plot_server_trace(generated_trace, interval=1, server_throughput=250, output_file="tally_bench_results/plots/azure_trace.png")
    write_json_to_file(generated_trace, "infer_trace/azure_trace_for_bert.json")
