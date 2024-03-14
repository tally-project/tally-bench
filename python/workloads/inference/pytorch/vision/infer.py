import torch
import torch.utils.data.distributed
import torchvision

from workloads.common.util import get_torch_compile_options
from workloads.common.infer_monitor import get_infer_monitor

def vision_infer(model_name, mode, batch_size, warmup_iters, total_time,
                 load=0.5, trace_file=None, result_dict=None, signal=False, pipe=None):
    
    model = getattr(torchvision.models, model_name)()
    model = model.cuda().eval()

    torch.set_float32_matmul_precision("high")

    compile_options = get_torch_compile_options()
    model = torch.compile(model, backend="inductor", options=compile_options)

    if mode in ["single-stream", "server"]:
        assert(batch_size == 1)

    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe, load, trace_file)
    data = torch.randn(batch_size, 3, 224, 224).cuda()

    while True:

        monitor.on_step_begin()

        y = model(data)
        
        torch.cuda.synchronize()

        should_stop = monitor.on_step_end()
        if should_stop:
            monitor.write_to_result()
            break