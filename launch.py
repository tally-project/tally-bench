from multiprocessing import Process, Manager, Value
import sys
import time
import random
import argparse
import json

random.seed(0)

sys.path.append('workloads')

from workloads.hidet.resnet import run_resnet as hidet_run_resnet

parser = argparse.ArgumentParser(prog="benchmark launcher", description="Launch a benchmark")

parser.add_argument("--framework", type=str, default="hidet")
parser.add_argument("--benchmark", type=str, default="resnet50")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--amp", action="store_true", default=False)
parser.add_argument("--warmup-iters", type=int, default=10)
parser.add_argument("--total-iters", type=int, default=0)
parser.add_argument("--runtime", type=int, default=10)
parser.add_argument("--signal", action="store_true", default=False)

args = parser.parse_args()

benchmark_list = {
    "hidet": {
        "resnet50": hidet_run_resnet
    }
}

if __name__ == "__main__":

    # Retrieve benchmark function
    benchmark_func = benchmark_list[args.framework][args.benchmark]

    total_iters = args.total_iters if args.total_iters else None
    result_dict = {}

    benchmark_func(args.benchmark, args.batch_size, args.amp, args.warmup_iters,
                   args.runtime, total_iters, result_dict, args.signal)
        
    print(f"Benchmark: {args.benchmark} Time: {result_dict['time_elapsed']} Iterations: {result_dict['iters']}")

    # Print json format result
    print(json.dumps(dict(result_dict)))