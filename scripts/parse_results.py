import sys

sys.path.append("python")

from bench_utils.parse import parse_result

if __name__ == "__main__":
    parse_result(
        "tally_bench_results/result.json",
        single_job_result_out="tally_bench_results/single-job-perf.csv",
        priority_result_out="tally_bench_results/priority-aware-perf.csv"
    )

    # parse_result(
    #     "tally_bench_results/azure_result_two_week.json",
    #     single_job_result_out="tally_bench_results/azure_single-job-perf.csv",
    #     priority_result_out="tally_bench_results/azure_priority-aware-perf.csv"
    # )