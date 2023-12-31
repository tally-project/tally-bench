import sys

sys.path.append(".")

from utils.parse import parse_result

if __name__ == "__main__":
    parse_result(
        "tally-bench-results/train_result.json",
        single_job_result_out="tally-bench-results/single.csv",
        co_locate_result_out="tally-bench-results/pair.csv"
    )