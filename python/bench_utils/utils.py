import subprocess
import json
import os
import logging
import time
import random
import numpy as np

logger = logging.getLogger("tally-bench")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(format)

def execute_cmd(cmd, get_output=False):
    if get_output:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        process.wait()
        out, err = process.communicate()
        rc = process.returncode
        return out, err, rc
    else:
        process = subprocess.Popen(cmd, shell=True, universal_newlines=True)
        process.wait()

def write_json_to_file(_dict, f_name):

    dir = os.path.dirname(f_name)
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    with open(f_name, 'w') as f:
        json.dump(_dict, f, indent=4, sort_keys=True)

def load_json_from_file(f_name):
    result = {}
    if os.path.isfile(f_name):
        with open(f_name) as f:
            result = json.load(f)
    return result

def busy_sleep(seconds):
    start_time = time.time()
    while (time.time() < start_time + seconds):
        pass

def get_possion_arrival_trace(latency, load, total_time):
    num_events = total_time / latency * load
    lambda_rate = 1 / latency * load
    arrival_intervals = np.random.exponential(1 / lambda_rate, int(num_events * 2))
    trace = np.cumsum(arrival_intervals).tolist()
    return trace