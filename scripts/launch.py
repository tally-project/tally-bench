import sys
import argparse
import json

sys.path.append('.')
# sys.path.append('workloads')
# sys.path.append('utils')

from utils.bench_util import set_deterministic

set_deterministic()

from workloads.hidet.resnet import run_resnet as hidet_run_resnet

from workloads.pytorch.imagenet.profile_imagenet import benchmark_imagenet
from workloads.pytorch.bert.profile_bert import benchmark_bert
from workloads.pytorch.cifar.profile_cifar import benchmark_cifar
from workloads.pytorch.dcgan.profile_dcgan import benchmark_dcgan
from workloads.pytorch.lstm.profile_lstm import benchmark_lstm
from workloads.pytorch.ncf.profile_ncf import benchmark_ncf
from workloads.pytorch.pointnet.profile_pointnet import benchmark_pointnet
from workloads.pytorch.translation.profile_transformer import benchmark_transformer

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

benchmark_list = {
    "hidet": {
        "resnet50": hidet_run_resnet
    },
    "pytorch": {
        "resnet50": benchmark_imagenet,
        "bert": benchmark_bert,
        "VGG": benchmark_cifar,
        "dcgan": benchmark_dcgan,
        "LSTM": benchmark_lstm,
        "NeuMF-pre": benchmark_ncf,
        "pointnet": benchmark_pointnet,
        "transformer": benchmark_transformer,
    }
}

def get_benchmark_func(framework, benchmark):

    # Lazy loading yolo benchmark
    if framework == "pytorch":
        if benchmark in ["yolov6n"]:
            from workloads.pytorch.yolov6.profile_yolov6 import benchmark_yolov6
            return benchmark_yolov6

    return benchmark_list[framework][benchmark]

if __name__ == "__main__":

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