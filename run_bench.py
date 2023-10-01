import sys
import os

sys.path.append('utils')

from utils.bench import Benchmark
from utils.bench_util import launch_benchmark

benchmark_list = {
    "hidet": {
        "resnet50": 64
    },
    "pytorch": {
        "resnet50": 64,
        "bert": 16,
        "VGG": 64,
        "dcgan": 64,
        "LSTM": 64,
        "NeuMF-pre": 64,
        "pointnet": 64,
        "transformer": 8
    }
}

if __name__ == "__main__":

    curr_dir = os.getcwd()
    os.environ["TALLY_HOME"] = f"{curr_dir}/tally"
    os.environ["PYTHONUNBUFFERED"] = "true"

    bench_1 = Benchmark("hidet", "resnet50", 64, False, 10, 10)
    bench_2 = Benchmark("pytorch", "resnet50", 64, False, 10, 10)
    bench_3 = Benchmark("pytorch", "bert", 16, False, 10, 10)
    bench_4 = Benchmark("pytorch", "VGG", 64, False, 10, 10)
    # bench_5 = Benchmark("pytorch", "dcgan", 64, False, 10, 10)
    # bench_6 = Benchmark("pytorch", "LSTM", 64, False, 10, 10)
    # bench_7 = Benchmark("pytorch", "NeuMF-pre", 64, False, 10, 10)
    # bench_8 = Benchmark("pytorch", "pointnet", 64, False, 10, 10)
    # bench_9 = Benchmark("pytorch", "transformer", 8, False, 10, 10)

    for bench in [bench_1, bench_2, bench_3, bench_4]:
        launch_benchmark([bench], use_tally=True)