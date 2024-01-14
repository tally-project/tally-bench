import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torchvision
import time
from torchvision import transforms

from workloads.pytorch.cifar.models import *
from utils.bench_util import wait_for_signal, get_torch_compile_options
from workloads.common.train_monitor import TrainMonitor

# Training
def train_cifar(model_name, batch_size, amp, warmup_iters, total_time,
                        total_iters=None, result_dict=None, signal=False, pipe=None):
    
    train_monitor = TrainMonitor(warmup_iters, total_time, total_iters, result_dict, signal, pipe)

    device = 'cuda'

    if model_name == 'VGG':
        model = VGG('VGG11')
    elif model_name == 'ShuffleNetV2': 
        model = ShuffleNetV2(net_size=0.5)
    else:
        model = eval(model_name)()

    model = model.cuda()

    # compile_options = get_torch_compile_options()
    # model = torch.compile(model, backend='inductor', options=compile_options)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    if amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=2)

    model.train()

    while True:
        should_training_stop = False

        for inputs, targets in trainloader:
            optimizer.zero_grad()

            if amp:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            should_training_stop = train_monitor.on_step_end(loss)
            if should_training_stop:
                break
        
        if should_training_stop:
            break