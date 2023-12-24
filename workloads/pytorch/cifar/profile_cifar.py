import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torchvision
import time
from torchvision import transforms

from workloads.pytorch.cifar.models import *
from utils.bench_util import wait_for_signal

# Training
def benchmark_cifar(model_name, batch_size, amp, warmup_iters, total_time,
                        total_iters=None, result_dict=None, signal=False, pipe=None):
    device = 'cuda'

    if model_name == 'VGG':
        model = VGG('VGG11')
    elif model_name == 'ShuffleNetV2': 
        model = ShuffleNetV2(net_size=0.5)
    else:
        model = eval(model_name)()

    model = model.cuda()

    # compile_options = {
    #     "epilogue_fusion": True,
    #     "max_autotune": True,
    #     "triton.cudagraphs": False,
    # }
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

    start_time = None
    num_iters = 0
    warm_iters = 0
    warm = False
    model.train()

    while True:
        stop = False

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
            
            # Increment iterations
            num_iters += 1
            if warm:
                warm_iters += 1

                # Break if reaching total iterations
                if warm_iters == total_iters:
                    stop = True
                    break

                # Or break if time is up
                curr_time = time.time()
                if curr_time - start_time >= total_time:
                    stop = True
                    break

            if num_iters == warmup_iters:
                warm = True

                if signal:
                    wait_for_signal(pipe)

                start_time = time.time()
                print("Measurement starts ...")
        
        if stop:
            break

    end_time = time.time()
    time_elapsed = end_time - start_time
    
    if result_dict is not None:
        result_dict["time_elapsed"] = time_elapsed
        result_dict["iters"] = warm_iters

    return time_elapsed, warm_iters