from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import sys
import numpy as np
import os
import pandas as pd
import torchvision
import time
import models.lstm.data as data
import models.lstm.models as models

from torch.nn import DataParallel
from torchvision import transforms

from models.util import get_benchmark_str

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# def get_batch(source, i, bptt):
#     seq_len = min(bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].view(-1)
#     return data, target


def benchmark_lstm(model_name, batch_size, mixed_precision, gpu_id, warmup_epoch, total_time,
                   warm_signal=None, start_signal=None, bench_id="", result_dict=None, total_iters=None,
                   data_dir='./data/wikitext-2', bptt=35, emsize=200,
                   nhead=2, nhid=200, nlayers=2, dropout=0.2, tied=False):

    if len(gpu_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)

    cudnn.benchmark = True 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset
    # print('==> Preparing data..')
    corpus = data.Corpus(data_dir)


    class CorpusDataset(torch.utils.data.Dataset):
        def __init__(self, data, batch_size, bptt):
            self._data = data.narrow(0, 0, (data.size(0) // batch_size) * batch_size)
            # Evenly divide the data across the bsz batches.
            self._data = self._data.view(batch_size, -1).t().contiguous().to(device)
            self._data_length = data.size(0)
            self._batch_size = batch_size
            self._bptt = bptt
        
        def get_input(self, row_idx, col_idx):
            row_idx = row_idx % len(self._data)
            seq_len = min(self._bptt, len(self._data) - 1 - row_idx)
            data = self._data[row_idx: row_idx+seq_len, col_idx]
            target = self._data[row_idx+1: row_idx+1+seq_len, col_idx].view(data.size())
            data = torch.cat([data, data.new_zeros(self._bptt - data.size(0))])
            target = torch.cat([target, target.new_zeros(self._bptt - target.size(0))])
            return data, target

        def __len__(self):
            return self._data_length // self._bptt

        def __getitem__(self, idx):
            return self.get_input((idx // self._batch_size) * self._bptt,
                                idx % self._batch_size)


    trainset = CorpusDataset(corpus.train,
                              batch_size,
                              bptt)

    trainloader = torch.utils.data.DataLoader(trainset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           sampler=None,
                                           drop_last=True)

    # Model
    # print('==> Building model..')
    ntokens = len(corpus.dictionary)
    if model_name == 'Transformer':
        model = models.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    else:
        model = models.RNNModel(model_name, ntokens, emsize, nhid, nlayers, dropout, tied).to(device)


    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    if len(gpu_id) > 1:
        model = DataParallel(model)
    
    t_start = time.time()
    
    # Train
    def benchmark_step():
        iter_num = 0
        iter_warm = 0
        warm = False
        exit_flag = False
        model.train()
        ntokens = len(corpus.dictionary)
        if model_name != 'Transformer':
            hidden = model.init_hidden(batch_size)
        # Prevent total batch number < warmup+benchmark situation
        while True:
            for data, targets in trainloader:
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
                data = data.t()
                targets = targets.t()
                optimizer.zero_grad()
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        if model_name == 'Transformer':
                            outputs = model(data)
                            outputs = outputs.view(-1, ntokens)
                        else:
                            hidden = repackage_hidden(hidden)
                            outputs, hidden = model(data, hidden)
                        loss = criterion(outputs, targets.flatten())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if model_name == 'Transformer':
                        outputs = model(data)
                        outputs = outputs.view(-1, ntokens)
                    else:
                        hidden = repackage_hidden(hidden)
                        outputs, hidden = model(data, hidden)
                    loss = criterion(outputs, targets.flatten())
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