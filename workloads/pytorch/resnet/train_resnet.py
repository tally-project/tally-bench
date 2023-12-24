import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import time
import torchvision

from utils.bench_util import wait_for_signal, get_torch_compile_options
from workloads.pytorch.common.train_monitor import TrainMonitor

# Training
def train_resnet(model_name, batch_size, amp, warmup_iters, total_time,
                        total_iters=None, result_dict=None, signal=False, pipe=None):

    train_monitor = TrainMonitor(warmup_iters, total_time, total_iters, result_dict, signal, pipe)

    model = getattr(torchvision.models, model_name)()
    model = model.cuda()

    # compile_options = get_torch_compile_options()
    # model = torch.compile(model, backend='inductor', options=compile_options)

    data = torch.randn(batch_size, 3, 224, 224)
    target = torch.LongTensor(batch_size).random_() % 1000
    data, target = data.cuda(), target.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    if amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    model.train()

    while True:
        optimizer.zero_grad()

        if amp:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = F.cross_entropy(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        
        should_training_stop = train_monitor.on_step_end()
        if should_training_stop:
            break

    if result_dict is not None:
        result_dict["time_elapsed"] = train_monitor.time_elapsed
        result_dict["iters"] = train_monitor.warm_iters

    return train_monitor.time_elapsed, train_monitor.warm_iters