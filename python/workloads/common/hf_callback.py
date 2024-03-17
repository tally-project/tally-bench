from transformers import TrainerCallback

from workloads.common.train_monitor import TrainMonitor


class BenchCallback(TrainerCallback):

    def __init__(self, warmup_iters, total_time, total_iters, result_dict, signal, pipe):
        self.monitor = TrainMonitor(warmup_iters, total_time, total_iters, result_dict, signal, pipe)

    def on_step_end(self, args, state, control, **kwargs):
        control.should_training_stop = self.monitor.on_step_end()