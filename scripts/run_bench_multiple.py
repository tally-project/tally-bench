import sys
import argparse

sys.path.append('python')

from bench_utils.bench_multiple import bench_multiple_workloads

parser = argparse.ArgumentParser(prog="benchmark suite launcher", description="Launch benchmark suite")
parser.add_argument("--num-workloads", type=int, default=3)

args = parser.parse_args()


if __name__ == "__main__":
    bench_multiple_workloads(args.num_workloads)