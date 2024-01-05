import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torchvision
import time
from torchvision import transforms

from workloads.pytorch.cifar.models import *
from workloads.common.infer_monitor import get_infer_monitor

# Training
def cifar_infer(model_name, mode, batch_size, amp, warmup_iters, total_time,
                 load=0.5, result_dict=None, signal=False, pipe=None):
    
    if mode in ["single-stream", "server"]:
        batch_size = 1

    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe, load)
    data = torch.randn(batch_size, 3, 32, 32).cuda()

    if model_name == 'VGG':
        model = VGG('VGG11')
    elif model_name == 'ShuffleNetV2': 
        model = ShuffleNetV2(net_size=0.5)
    else:
        model = eval(model_name)()

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