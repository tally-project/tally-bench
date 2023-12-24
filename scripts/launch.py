import sys
import argparse
import json
import os

sys.path.append('.')

from utils.bench_util import set_deterministic, get_benchmark_func

set_deterministic()

parser = argparse.ArgumentParser(prog="benchmark launcher", description="Launch a benchmark")

parser.add_argument("--framework", type=str, default="hidet")
parser.add_argument("--benchmark", type=str, default="resnet50")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--amp", action="store_true", default=False)
parser.add_argument("--warmup-iters", type=int, default=10)
parser.add_argument("--total-iters", type=int, default=0)
parser.add_argument("--runtime", type=int, default=10)
parser.add_argument("--signal", action="store_true", default=False)
parser.add_argument("--pipe", type=str, default="")

args = parser.parse_args()

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if args.signal:
        assert(args.pipe)

    # Retrieve benchmark function
    benchmark_func = get_benchmark_func(args.framework, args.benchmark)

    total_iters = args.total_iters if args.total_iters else None
    result_dict = {}

    print(f"Running framework: {args.framework} benchmark: {args.benchmark} Batch size: {args.batch_size} amp: {args.amp}")

    benchmark_func(args.benchmark, args.batch_size, args.amp, args.warmup_iters,
                   args.runtime, total_iters, result_dict, args.signal, args.pipe)
    
    print(f"Benchmark: {args.benchmark} Time: {result_dict['time_elapsed']} Iterations: {result_dict['iters']}")

    # Print json format result
    print(json.dumps(dict(result_dict)))