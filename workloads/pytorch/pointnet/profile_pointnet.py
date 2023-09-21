from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import numpy as np
import os
import pandas as pd
import torchvision
import time

from torch.nn import DataParallel
from torchvision import transforms
from models.pointnet.dataset import ShapeNetDataset
from models.pointnet.pointnet import PointNetCls, feature_transform_regularizer

from models.util import get_benchmark_str

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

def benchmark_pointnet(model_name, batch_size, mixed_precision, gpu_id, warmup_epoch, total_time,
                       warm_signal=None, start_signal=None, bench_id="", result_dict=None, total_iters=None,
                       data_dir="./data/shapenetcore", num_points=2500, feature_transform=True):

    if len(gpu_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)

    cudnn.benchmark = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # specify dataset
    # print('==> Preparing data..')
    trainset = build_dataset(data_dir, num_points)
    trainloader = build_dataloader(trainset, batch_size)
    num_classes = len(trainset.classes)
    # print("classes", num_classes)

    # Model
    # print('==> Building model..')
    model = PointNetCls(k=num_classes, feature_transform=feature_transform)
    model = model.to(device)

 
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    if len(gpu_id) > 1:
        model = DataParallel(model)
    
     # Train
    def benchmark_step():
        t_start = time.time()
        iter_num = 0
        iter_warm = 0
        warm = False
        exit_flag = False
        model.train()
        # Prevent total batch number < warmup+benchmark situation
        while True:
            for inputs, targets in trainloader:
                # Warm-up: previous 10 iters
                if iter_num == warmup_epoch - 1:
                    warm = True
                    if warm_signal is not None:
                        warm_signal.value = 1
                    if start_signal is not None:
                        while start_signal.value != 1:
                            time.sleep(0.1)
                    print("Measurement starts...")
                    torch.cuda.cudart().cudaProfilerStart()
                    t_warmend = time.time()
                # Reach timeout: exit benchmark
                if (warm and (time.time() - t_warmend >= total_time)) or (total_iters is not None and iter_warm == total_iters):
                    t_end = time.time()
                    t_pass = t_end - t_warmend
                    exit_flag = True
                    torch.cuda.cudart().cudaProfilerStop()
                    break
                optimizer.zero_grad()
                targets = targets[:, 0]
                inputs = inputs.transpose(2, 1)
                if mixed_precision:
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
                iter_num += 1
                if warm:
                    iter_warm += 1
            if exit_flag:
                break
        return t_pass, iter_warm


    benchmark_id = get_benchmark_str(model_name, batch_size, mixed_precision, bench_id)

    try:
        print(f'==> Training {model_name} model with {batch_size} batch size, {mixed_precision} mp..')
        t_pass, iter_warm = benchmark_step()
        print(f"Time: {t_pass}, Iterations: {iter_warm}")
        
        if result_dict is not None:
            result_dict[benchmark_id] = {
                "Time": t_pass,
                "Iterations": iter_warm
            }
    except Exception as e:
        if result_dict is not None:
            result_dict[benchmark_id] = {
                "Error": str(e)
            }
        else:
            raise e