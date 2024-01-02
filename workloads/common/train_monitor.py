import time
import torch

from utils.bench_util import wait_for_signal


class TrainMonitor:

    def __init__(self, warmup_iters, total_time, total_iters, result_dict, signal, pipe):
        self.start_time = None
        self.num_iters = 0
        self.warm_iters = 0
        self.warm = False
        self.warmup_iters = warmup_iters
        self.result_dict = result_dict
        self.signal = signal
        self.total_time = total_time
        self.total_iters = total_iters
        self.finished = False
        self.time_elapsed = None
        self.pipe = pipe

    def on_step_end(self):

        torch.cuda.synchronize()

        should_training_stop = False
        
        self.num_iters += 1
        if self.warm:
            self.warm_iters += 1

            # Break if reaching total iterations
            if self.warm_iters == self.total_iters:
                should_training_stop = True
                
            # Or break if time is up
            curr_time = time.time()
            if curr_time - self.start_time >= self.total_time:
                should_training_stop = True

        if self.num_iters == self.warmup_iters:
            self.warm = True

            if self.signal:
                wait_for_signal(self.pipe)

            self.start_time = time.time()
            print("Measurement starts ...")
        
        if should_training_stop:
            end_time = time.time()
            self.time_elapsed = end_time - self.start_time

            self.write_to_result()
        
        return should_training_stop

    def write_to_result(self):
        if self.result_dict is not None:
            self.result_dict["time_elapsed"] = self.time_elapsed
            self.result_dict["iters"] = self.warm_iters