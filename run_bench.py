import sys
import os

sys.path.append('utils')

from utils.bench import Benchmark
from utils.bench_util import launch_benchmark
from utils.tally import start_iox_roudi, shut_down_iox_roudi

benchmark_list = {
    "hidet": {
        "resnet50": [64]
    },
    "pytorch": {
        "resnet50": [64],
        "bert": [16],
        "VGG": [64],
        "dcgan": [64],
        "LSTM": [64],
        "NeuMF-pre": [64],
        "pointnet": [64],
        "transformer": [8]
    }
}

if __name__ == "__main__":

    curr_dir = os.getcwd()
    os.environ["TALLY_HOME"] = f"{curr_dir}/tally"
    os.environ["PYTHONUNBUFFERED"] = "true"

    benchmarks = []

    for framework in benchmark_list:
        for model in benchmark_list[framework]:
            for batch_size in benchmark_list[framework][model]:
                for amp in [True, False]:
                    bench = Benchmark(framework, model, batch_size, amp, warmup_iters=10, runtime=10)
                    benchmarks.append(bench)
    
    shut_down_iox_roudi()
    start_iox_roudi()

    for bench in benchmarks:
        launch_benchmark([bench], use_tally=True)
    
    shut_down_iox_roudi()