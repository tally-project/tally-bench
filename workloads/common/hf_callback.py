from transformers import TrainerCallback

import time
import timeit

from utils.bench_util import wait_for_signal


class BenchCallback(TrainerCallback):

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

    def on_step_end(self, args, state, control, **kwargs):
        
        self.num_iters += 1
        if self.warm:
            self.warm_iters += 1

            # Break if reaching total iterations
            if self.warm_iters == self.total_iters:
                control.should_training_stop = True
                
            # Or break if time is up
            curr_time = timeit.default_timer()
            if curr_time - self.start_time >= self.total_time:
                control.should_training_stop = True

        if self.num_iters == self.warmup_iters:
            self.warm = True

            if self.signal:
                wait_for_signal(self.pipe)

            self.start_time = timeit.default_timer()
            print("Measurement starts ...")
        
        if control.should_training_stop:
            end_time = timeit.default_timer()
            self.time_elapsed = end_time - self.start_time

