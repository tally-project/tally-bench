import sys
import argparse

sys.path.append('python')

from bench_utils.bench_azure import bench_azure_trace

parser = argparse.ArgumentParser(prog="benchmark suite launcher", description="Launch benchmark suite")

parser.add_argument("--use-mps", action="store_true", default=False)
parser.add_argument("--use-mps-priority", action="store_true", default=False)
parser.add_argument("--use-tally-priority", action="store_true", default=False)
parser.add_argument("--use-tgs", action="store_true", default=False)

args = parser.parse_args()
assert(sum([args.use_mps, args.use_mps_priority, args.use_tally_priority, args.use_tgs]) <= 1)


if __name__ == "__main__":

    bench_azure_trace(
        trace_path=f"infer_trace/azure_trace_for_bert_two_week.json",
        result_file=f"azure_result_two_week.json",
        use_mps=args.use_mps,
        use_mps_priority=args.use_mps_priority,
        use_tally_priority=args.use_tally_priority,
        use_tgs=args.use_tgs
    )