from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import os
import pandas as pd
import time

from torchvision import transforms
from torch.nn import DataParallel
import models.ncf.models as models
import models.ncf.config as config
import models.ncf.data_utils as data_utils

from models.util import get_benchmark_str

def benchmark_ncf(model_name, batch_size, mixed_precision, gpu_id, warmup_epoch, total_time,
                  warm_signal=None, start_signal=None, bench_id="", result_dict=None, total_iters=None,
                  num_ng=4, factor_num=32, num_layers=3, dropout=0.0, lr=0.001):
    if len(gpu_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)

    cudnn.benchmark = True 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ############################## PREPARE DATASET ##########################
    train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()

    # construct the train and test datasets
    train_dataset = data_utils.NCFData(
        train_data, item_num, train_mat, num_ng, True)
    train_loader = data.DataLoader(train_dataset,
        batch_size, shuffle=True, num_workers=2)

    ########################### CREATE MODEL #################################
    if model_name == 'NeuMF-end':
        assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
        assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None

    model = models.NCF(user_num, item_num, factor_num, num_layers, dropout, config.model, GMF_model, MLP_model)
    model.cuda()

    loss_function = nn.BCEWithLogitsLoss()

    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    if len(gpu_id) > 1:
        model = DataParallel(model)
    ########################### TRAINING #####################################

    def benchmark_step():
        iter_num = 0
        iter_warm = 0
        warm = False
        exit_flag = False
        model.train()
        train_loader.dataset.ng_sample()
        t_start = time.time()
        while True:
            for idx, (user, item, label) in enumerate(train_loader):
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
                if (warm and (time.time() - t_warmend >= total_time)) or (total_iters is not None and iter_warm == total_iters):
                    t_end = time.time()
                    t_pass = t_end - t_warmend
                    exit_flag = True
                    torch.cuda.cudart().cudaProfilerStop()
                    break
                user = user.cuda()
                item = item.cuda()
                label = label.float().cuda()
                optimizer.zero_grad()
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        prediction = model(user, item)
                        loss = loss_function(prediction, label)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    prediction = model(user, item)
                    loss = loss_function(prediction, label)
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