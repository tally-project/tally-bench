import sys

sys.path.append("utils")

from utils.parse import parse_result

if __name__ == "__main__":
    parse_result("result.json", single_job_result_out="single.csv", co_locate_result_out="pair.csv")