import time
import torch
import timeit

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

    # optionally pass a loss value to be printed at warm and at end
    # to verify the training process is ok
    def on_step_end(self, loss=None):

        torch.cuda.synchronize()

        should_training_stop = False
        
        self.num_iters += 1
        if self.warm:
            self.warm_iters += 1

            # Break if reaching total iterations
            if self.warm_iters == self.total_iters:
                should_training_stop = True
                
            # Or break if time is up
            curr_time = timeit.default_timer()
            if curr_time - self.start_time >= self.total_time:
                should_training_stop = True

        if self.num_iters == self.warmup_iters:
            self.warm = True

            if loss:
                print(f"loss: {loss}")

            if self.signal:
                wait_for_signal(self.pipe)

            self.start_time = timeit.default_timer()
            print("Measurement starts ...")
        
        if should_training_stop:
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            self.time_elapsed = end_time - self.start_time

            if loss:
                print(f"loss: {loss}")

            self.write_to_result()
        
        return should_training_stop

    def write_to_result(self):
        if self.result_dict is not None:
            self.result_dict["time_elapsed"] = self.time_elapsed
            self.result_dict["iters"] = self.warm_iters