import sys
import argparse

sys.path.append('python')

from bench_utils.bench_azure import bench_azure_trace

parser = argparse.ArgumentParser(prog="benchmark suite launcher", description="Launch benchmark suite")

parser.add_argument("--use-mps", action="store_true", default=False)
parser.add_argument("--use-mps-priority", action="store_true", default=False)
parser.add_argument("--use-tally-priority", action="store_true", default=False)
parser.add_argument("--use-full-trace", action="store_true", default=False)

args = parser.parse_args()
assert(sum([args.use_mps, args.use_mps_priority, args.use_tally_priority]) <= 1)


if __name__ == "__main__":

    trace_file = "one_day"
    if args.use_full_trace:
        trace_file = "two_week"

    bench_azure_trace(
        trace_path=f"infer_trace/azure_trace_for_bert_{trace_file}.json",
        result_file=f"azure_result_{trace_file}.json",
        use_mps=args.use_mps,
        use_mps_priority=args.use_mps_priority,
        use_tally_priority=args.use_tally_priority
    )