class Benchmark:

    def __init__(self, framework, model_name, batch_size, amp, warmup_iters, runtime, total_iters=None):
        self.framework = framework
        self.model_name = model_name
        self.batch_size = batch_size
        self.amp = amp
        self.warmup_iters = warmup_iters
        self.runtime = runtime
        self.total_iters = total_iters
    
    def __str__(self):
        _str = f"{self.framework}_{self.model_name}_{self.batch_size}"
        if self.amp:
            _str += "_amp"
        return _str