import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed

import time

from workloads.pytorch.pointnet.dataset import ShapeNetDataset
from workloads.pytorch.pointnet.pointnet import PointNetCls, feature_transform_regularizer

from utils.bench_util import wait_for_signal

def build_dataset(data_dir, num_points):
    # Dataset: shapenet
    trainset = ShapeNetDataset(root=data_dir, classification=True, npoints=num_points,)
    return trainset

def build_dataloader(trainset, batch_size):
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size, shuffle=True, num_workers=2, drop_last=True,
    )
    return trainloader

def loss_fn(output, label, trans_feat, feature_transform):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, label)
    if feature_transform:
        loss += feature_transform_regularizer(trans_feat) * 0.001
    return loss

# Training
def benchmark_pointnet(model_name, batch_size, amp, warmup_iters, total_time,
                    total_iters=None, result_dict=None, signal=False,
                    data_dir="./data/shapenetcore", num_points=2500,
                    feature_transform=True):
    device = 'cuda'

    trainset = build_dataset(data_dir, num_points)
    trainloader = build_dataloader(trainset, batch_size)
    num_classes = len(trainset.classes)

    model = PointNetCls(k=num_classes, feature_transform=feature_transform)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    compile_options = {
        "epilogue_fusion": True,
        "max_autotune": True,
        "triton.cudagraphs": False,
    }
    model = torch.compile(model, backend='inductor', options=compile_options)

    start_time = None
    num_iters = 0
    warm_iters = 0
    warm = False
    model.train()

    while True:
        stop = False

        for inputs, targets in trainloader:
            optimizer.zero_grad()
            targets = targets[:, 0]
            inputs = inputs.transpose(2, 1)
            if amp:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.cuda.amp.autocast():
                    pred, trans, trans_feat = model(inputs)
                    loss = loss_fn(pred, targets, trans_feat, feature_transform)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                inputs, targets = inputs.to(device), targets.to(device)
                pred, trans, trans_feat = model(inputs)
                loss = loss_fn(pred, targets, trans_feat, feature_transform)
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
                    wait_for_signal()

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