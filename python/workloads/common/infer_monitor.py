import time
import numpy as np
import timeit
import torch
import random

from bench_utils.bench_utils import wait_for_signal
from bench_utils.utils import busy_sleep, get_possion_arrival_trace, load_json_from_file


class InferMonitor:

    def __init__(self, warmup_iters, total_time, result_dict, signal=None, pipe=None,
                 no_waiting=False):
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
        self.should_stop = False
        self.no_waiting = no_waiting
    
    def on_step_end(self):

        self.num_iters += 1
        if self.warm:
            self.warm_iters += 1
 
            # break if time is up
            curr_time = timeit.default_timer()
            if curr_time - self.start_time >= self.total_time:
                self.should_stop = True

        if not self.warm and self.num_iters >= self.warmup_iters:

            if self.signal:
                if wait_for_signal(self.pipe, break_if_not_ready=self.no_waiting):
                    pass
                else:
                    return self.should_stop
            
            self.warm = True

            self.start_time = timeit.default_timer()
            print("Measurement starts ...")
        
        if self.should_stop:
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            self.time_elapsed = end_time - self.start_time
        
        return self.should_stop


class SingleStreamInferMonitor(InferMonitor):

    def __init__(self, warmup_iters, total_time, result_dict, signal, pipe, no_waiting=False):
        super().__init__(warmup_iters, total_time, result_dict, signal, pipe, no_waiting)

        self.step_begin_time = None
        self.step_end_time = None
        self.latencies = []
        self.end_timestamps = []

    def on_step_begin(self):
        self.step_begin_time = timeit.default_timer()

    def on_step_end(self):

        self.step_end_time = timeit.default_timer()

        if self.warm:
            elapsed_time_ms = (self.step_end_time - self.step_begin_time) * 1000
            self.latencies.append(elapsed_time_ms)

            elapsed_from_start = self.step_end_time - self.start_time
            self.end_timestamps.append(elapsed_from_start)
        
        return super().on_step_end()

    def write_to_result(self):

        if self.result_dict is not None:
            self.result_dict["time_elapsed"] = self.time_elapsed
            self.result_dict["iters"] = self.warm_iters
            self.result_dict["latencies"] = self.latencies
            self.result_dict["end_timestamps"] = self.end_timestamps


class ServerInferMonitor(InferMonitor):

    def __init__(self, warmup_iters, total_time, result_dict, signal, pipe,
                 load=None, trace_file=None, no_waiting=False):
        super().__init__(warmup_iters, total_time, result_dict, signal, pipe, no_waiting)

        self.step_begin_time = None
        self.step_end_time = None
        self.latencies = []
        self.end_timestamps = []
        self.load = load
        if trace_file:
            self.trace = load_json_from_file(trace_file)
        else:
            self.trace = None
        self.query_latency = float('inf')

    def on_step_begin(self):
        self.step_begin_time = timeit.default_timer()

    def on_step_end(self):
        self.step_end_time = timeit.default_timer()
        elapsed_time_seconds = self.step_end_time - self.step_begin_time

        if not self.trace:

            self.query_latency = min(self.query_latency, elapsed_time_seconds)
            if self.num_iters + 1 == self.warmup_iters:
                # simulate arrivial timestamps
                self.trace = get_possion_arrival_trace(self.query_latency, self.load, self.total_time)

        if self.warm:
            assert(self.trace)

            elapsed_from_start = self.step_end_time - self.start_time
    
            if len(self.latencies) >= len(self.trace):
                self.should_stop = True
            else:
                next_arrival_ts = self.trace[len(self.latencies)]

                if next_arrival_ts >= self.total_time:
                    self.should_stop = True

                elif elapsed_from_start < next_arrival_ts:
                    wait_time = next_arrival_ts - elapsed_from_start
                    busy_sleep(wait_time)
            
            elapsed_time_ms = elapsed_time_seconds * 1000
            self.latencies.append(elapsed_time_ms)

            self.end_timestamps.append(elapsed_from_start)
            
        return super().on_step_end()
    
    def write_to_result(self):
            
        if self.result_dict is not None:
            self.result_dict["time_elapsed"] = self.time_elapsed
            self.result_dict["iters"] = self.warm_iters
            self.result_dict["latencies"] = self.latencies
            self.result_dict["end_timestamps"] = self.end_timestamps


def get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe,
                      load=None, trace_file=None, no_waiting=False):

    if mode == "single-stream":
        return SingleStreamInferMonitor(warmup_iters, total_time, result_dict, signal, pipe, no_waiting)
    elif mode == "server":
        assert(load)
        return ServerInferMonitor(warmup_iters, total_time, result_dict, signal, pipe, load, trace_file, no_waiting)
    else:
        raise Exception("unknown mode")
