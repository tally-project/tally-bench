import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed

import time

from workloads.pytorch.pointnet.dataset import ShapeNetDataset
from workloads.pytorch.pointnet.pointnet import PointNetCls, feature_transform_regularizer
from workloads.common.train_monitor import TrainMonitor

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
def train_pointnet(model_name, batch_size, amp, warmup_iters, total_time,
                    total_iters=None, result_dict=None, signal=False, pipe=None,
                    data_dir="./data/shapenetcore", num_points=2500,
                    feature_transform=True):
    device = 'cuda'
    train_monitor = TrainMonitor(warmup_iters, total_time, total_iters, result_dict, signal, pipe)

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

    # compile_options = {
    #     "epilogue_fusion": True,
    #     "max_autotune": True,
    #     "triton.cudagraphs": False,
    # }
    # model = torch.compile(model, backend='inductor', options=compile_options)

    model.train()

    while True:
        should_training_stop = False

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
            
            should_training_stop = train_monitor.on_step_end(loss)
            if should_training_stop:
                break
        
        if should_training_stop:
            break