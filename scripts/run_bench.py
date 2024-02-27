import sys
import argparse

sys.path.append('python')

from bench_utils.bench import run_benchmark_suite

parser = argparse.ArgumentParser(prog="benchmark suite launcher", description="Launch benchmark suite")

parser.add_argument("--save-results", action="store_true", default=False)
parser.add_argument("--use-mps", action="store_true", default=False)
parser.add_argument("--use-tally-naive", action="store_true", default=False)
parser.add_argument("--use-tally-priority", action="store_true", default=False)
parser.add_argument("--run-pairwise", action="store_true", default=False)
parser.add_argument("--runtime", type=int, default=10)
parser.add_argument("--warmup-iters", type=int, default=100)
parser.add_argument("--profile-only", action="store_true", default=False)

args = parser.parse_args()
assert(not (args.use_mps and args.use_tally_naive and args.use_tally_priority))


if __name__ == "__main__":

    run_benchmark_suite(
        use_mps=args.use_mps,
        use_tally_naive=args.use_tally_naive,
        use_tally_priority=args.use_tally_priority,
        run_pairwise=args.run_pairwise,
        runtime=args.runtime,
        warmup_iters=args.warmup_iters,
        save_results=args.save_results,
        profile_only=args.profile_only
    )