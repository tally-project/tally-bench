import sys

sys.path.append(".")

from utils.parse import parse_result

if __name__ == "__main__":
    parse_result(
        "tally-bench-results/result.json",
        single_job_result_out="tally-bench-results/single-job-perf.csv",
        throughput_result_out="tally-bench-results/throughput-oriented-perf.csv",
        priority_result_out="tally-bench-results/priority-aware-perf.csv"
    )