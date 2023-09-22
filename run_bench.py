import subprocess
import time
import json
import sys

sys.path.append('utils')

from utils.mps import shut_down_mps, start_mps

class Benchmark:

    def __init__(self, framework, model_name, batch_size, amp, warmup_iters, runtime):
        self.framework = framework
        self.model_name = model_name
        self.batch_size = batch_size
        self.amp = amp
        self.warmup_iters = warmup_iters
        self.runtime = runtime

def launch_benchmark(benchmarks: list, use_mps=False, preload=""):

    shut_down_mps()
    if use_mps:
        start_mps()
        assert(preload == "")

    processes = []

    for benchmark in benchmarks:
        
        launch_cmd = (f"{preload} python3 launch.py " +
                        f"--framework {benchmark.framework} " +
                        f"--benchmark {benchmark.model_name} " +
                        f"--batch-size {benchmark.batch_size} " +
                        f"--warmup-iters {benchmark.warmup_iters} " +
                        f"--runtime {benchmark.runtime} " +
                        f"--signal ")
        if benchmark.amp:
            launch_cmd += "--amp "

        process = subprocess.Popen(launch_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
        processes.append(process)

        while True:
            response = process.stdout.readline()
            print(response.strip())
            if "benchmark is warm" in response:
                break
            time.sleep(0.01)
    
    # All benchmarks should be warm, signal start
    print("Setting start signals ...")

    for process in processes:
        process.stdin.write("start\n")
        process.stdin.flush()
    
    for process in processes:
        process.wait()
        output = process.communicate()[0].strip()
        result_dict = json.loads(output.split("\n")[-1])

        print(result_dict)


if __name__ == "__main__":

    bench_1 = Benchmark("hidet", "resnet50", 64, True, 10, 10)
    bench_2 = Benchmark("hidet", "resnet50", 64, False, 10, 10)

    # bench_1 = Benchmark("pytorch", "resnet50", 64, True, 10, 10)
    # bench_2 = Benchmark("pytorch", "resnet50", 64, False, 10, 10)

    preload = "LD_PRELOAD=~/tally/build/libtally_client.so"
    # preload = ""

    # launch_benchmark([bench_1], preload=preload)

    # launch_benchmark([bench_1, bench_2], preload=preload)
    launch_benchmark([bench_1, bench_2], use_mps=True)