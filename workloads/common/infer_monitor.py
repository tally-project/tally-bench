import time

from utils.bench_util import wait_for_signal


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
            curr_time = time.time()
            if curr_time - self.start_time >= self.total_time:
                should_stop = True

        if self.num_iters == self.warmup_iters:
            self.warm = True

            if self.signal:
                wait_for_signal(self.pipe)

            self.start_time = time.time()
            print("Measurement starts ...")
        
        if should_stop:
            end_time = time.time()
            self.time_elapsed = end_time - self.start_time
        
        return should_stop


class SingleStreamInferMonitor(InferMonitor):

    def __init__(self, warmup_iters, total_time, result_dict, signal, pipe):
        super().__init__(warmup_iters, total_time, result_dict, signal, pipe)

        self.step_begin_time = None
        self.step_end_time = None
        self.latencies = []

    def on_step_begin(self):
        if self.warm:
            self.step_begin_time = time.time()

    def on_step_end(self):

        if self.warm:
            self.step_end_time = time.time()
            elapsed_time_ms = (self.step_end_time - self.step_begin_time) * 1000
            self.latencies.append(elapsed_time_ms)
        
        return super().on_step_end()

    def write_to_result(self):
        if self.result_dict is not None:
            self.result_dict["time_elapsed"] = self.time_elapsed
            self.result_dict["latencies"] = self.latencies


class ServerInferMonitor(InferMonitor):

    def __init__(self, warmup_iters, total_time, result_dict, signal, pipe):
        super().__init__(warmup_iters, total_time, result_dict, signal, pipe)

        self.step_begin_time = None
        self.step_end_time = None
        self.latencies = []

    def on_step_begin(self):
        if self.warm:
            self.step_begin_time = time.time()

    def on_step_end(self):

        if self.warm:
            self.step_end_time = time.time()
            elapsed_time_ms = (self.step_end_time - self.step_begin_time) * 1000
            self.latencies.append(elapsed_time_ms)

            # add wait time
        
        return super().on_step_end()
    
    def write_to_result(self):
        if self.result_dict is not None:
            self.result_dict["time_elapsed"] = self.time_elapsed
            self.result_dict["latencies"] = self.latencies


class OfflineInferMonitor(InferMonitor):

    def __init__(self, warmup_iters, total_time, result_dict, signal, pipe):
        super().__init__(warmup_iters, total_time, result_dict, signal, pipe)

    def on_step_begin(self):
        pass

    def on_step_end(self):
        return super().on_step_end()

    def write_to_result(self):
        if self.result_dict is not None:
            self.result_dict["time_elapsed"] = self.time_elapsed
            self.result_dict["iters"] = self.warm_iters