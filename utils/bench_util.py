import sys
import subprocess
import time
import json

from utils.mps import shut_down_mps, start_mps
from utils.tally import shut_down_tally, start_tally, tally_client_script

def wait_for_signal():
    print("benchmark is warm", flush=True)

    while True:
        sys.stdin.flush()
        inp = sys.stdin.readline()
        if "start" in inp:
            break

def launch_benchmark(benchmarks: list, use_mps=False, use_tally=False):

    shut_down_mps()
    shut_down_tally()

    if use_mps:
        start_mps()
        assert(not use_tally)

    if use_tally:
        start_tally()

    processes = []
    abort = False

    for benchmark in benchmarks:
        
        launch_cmd = (f"python3 launch.py " +
                        f"--framework {benchmark.framework} " +
                        f"--benchmark {benchmark.model_name} " +
                        f"--batch-size {benchmark.batch_size} " +
                        f"--warmup-iters {benchmark.warmup_iters} " +
                        f"--runtime {benchmark.runtime} " +
                        f"--signal ")
        
        if benchmark.total_iters:
            launch_cmd += f"--total-iters {benchmark.total_iters} "

        if benchmark.amp:
            launch_cmd += "--amp "
        
        if use_tally:
            launch_cmd = f"{tally_client_script} {launch_cmd}"
        
        print(f"launch_cmd: {launch_cmd}")

        process = subprocess.Popen(launch_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
        processes.append(process)

        while True:
            poll = process.poll()
            if poll is not None:
                abort = True
                break
            process.stdout.flush()
            response = process.stdout.readline().strip()
            if response:
                print(response)
            if "benchmark is warm" in response:
                break
            time.sleep(0.01)

    if abort:
        print("Detect process abort. Terminating ...")
        for process in processes:
            process.terminate()
        shut_down_tally()
        exit(1)

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
    
    shut_down_tally()