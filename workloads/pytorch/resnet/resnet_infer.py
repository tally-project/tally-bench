import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import time
import torchvision

from workloads.common.infer_monitor import get_infer_monitor

def resnet_infer(model_name, mode, batch_size, amp, warmup_iters, total_time,
                 load=0.5, result_dict=None, signal=False, pipe=None):
    
    if mode in ["single-stream", "server"]:
        batch_size = 1

    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe, load)
    data = torch.randn(batch_size, 3, 224, 224).cuda()

    model = getattr(torchvision.models, model_name)()
    model = model.cuda().eval()

    while True:

        monitor.on_step_begin()

        if amp:
            with torch.cuda.amp.autocast():
                y = model(data)
        else:
            y = model(data)
        
        torch.cuda.synchronize()

        should_stop = monitor.on_step_end()
        if should_stop:
            monitor.write_to_result()
            break

