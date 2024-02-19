import time
import numpy as np
import timeit
import torch

from utils.bench_util import wait_for_signal
from utils.util import busy_sleep


class InferMonitor:

    def __init__(self, warmup_iters, total_time, result_dict, signal, pipe):
        self.start_time = None
        self.num_iters = 0
        self.warm_iters = 0
        self.warm = False
        self.warmup_iters = warmup_iters
        self.result_dict = result_dict
        self.signal = signal
        self.total_time = total_time
        self.finished = False
        self.time_elapsed = None
        self.pipe = pipe
    
    def on_step_end(self):

        should_stop = False
        
        self.num_iters += 1
        if self.warm:
            self.warm_iters += 1
 
            # break if time is up
            curr_time = timeit.default_timer()
            if curr_time - self.start_time >= self.total_time:
                should_stop = True

        if self.num_iters == self.warmup_iters:
            self.warm = True

            if self.signal:
                wait_for_signal(self.pipe)

            self.start_time = timeit.default_timer()
            print("Measurement starts ...")
        
        if should_stop:
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            self.time_elapsed = end_time - self.start_time
        
        return should_stop


class SingleStreamInferMonitor(InferMonitor):

    def __init__(self, warmup_iters, total_time, result_dict, signal, pipe):
        super().__init__(warmup_iters, total_time, result_dict, signal, pipe)

        self.step_begin_time = None
        self.step_end_time = None
        self.latencies = []

    def on_step_begin(self):
        self.step_begin_time = timeit.default_timer()

    def on_step_end(self):

        self.step_end_time = timeit.default_timer()

        if self.warm:
            elapsed_time_ms = (self.step_end_time - self.step_begin_time) * 1000
            self.latencies.append(elapsed_time_ms)
        
        return super().on_step_end()

    def write_to_result(self):

        # remove first 10 latency measurement
        if len(self.latencies) > 10:
            self.latencies = self.latencies[min(10, len(self.latencies) // 2):]

        if self.result_dict is not None:
            self.result_dict["time_elapsed"] = self.time_elapsed
            self.result_dict["latencies"] = self.latencies
            self.result_dict["iters"] = self.warm_iters


class ServerInferMonitor(InferMonitor):

    def __init__(self, warmup_iters, total_time, result_dict, signal, pipe, load):
        super().__init__(warmup_iters, total_time, result_dict, signal, pipe)

        self.step_begin_time = None
        self.step_end_time = None
        self.latencies = []
        self.load = load
        self.query_latency = None
        self.poisson_lambda = None

    def on_step_begin(self):
        self.step_begin_time = timeit.default_timer()

    def on_step_end(self):
        self.step_end_time = timeit.default_timer()

        if not self.query_latency and self.num_iters >= 5:
            elapsed_time_seconds = self.step_end_time - self.step_begin_time
            self.query_latency = elapsed_time_seconds
            self.poisson_lambda = (1 / self.query_latency) * self.load
            print(f"Poisson lambda rate: {self.poisson_lambda}")

        if self.query_latency:
            elapsed_time_ms = (self.step_end_time - self.step_begin_time) * 1000

            if self.warm:
                self.latencies.append(elapsed_time_ms)

            # wait time to simulate arrival rate of poisson distribution
            assert(self.poisson_lambda)
            interval = np.random.exponential(1 / self.poisson_lambda)
            busy_sleep(interval)
            
        return super().on_step_end()
    
    def write_to_result(self):

        # remove first 10 latency measurement
        if len(self.latencies) > 10:
            self.latencies = self.latencies[min(10, len(self.latencies) // 2):]
            
        if self.result_dict is not None:
            self.result_dict["time_elapsed"] = self.time_elapsed
            self.result_dict["latencies"] = self.latencies
            self.result_dict["iters"] = self.warm_iters


def get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe, load=None):

    if mode == "single-stream":
        return SingleStreamInferMonitor(warmup_iters, total_time, result_dict, signal, pipe)
    elif mode == "server":
        assert(load)
        return ServerInferMonitor(warmup_iters, total_time, result_dict, signal, pipe, load)
    else:
        raise Exception("unknown mode")
