from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed

import numpy as np
import os
import pandas as pd
import time
from models.translation.transformer import Constants
from tqdm import tqdm 

from models.translation.dataset import TranslationDataset, paired_collate_fn
from models.translation.transformer.Models import Transformer
from models.translation.transformer.Optim import ScheduledOptim

from models.util import get_benchmark_str

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss

def prepare_dataloaders(data, distributed, batch_size):
    # ========= Preparing DataLoader =========#
    train_dataset = TranslationDataset(
        src_word2idx=data['dict']['src'],
        tgt_word2idx=data['dict']['tgt'],
        src_insts=data['train']['src'],
        tgt_insts=data['train']['tgt'])
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=paired_collate_fn,
        shuffle=train_sampler is None,
        sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=2,
        batch_size=batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader


def benchmark_transformer(model_name, batch_size, mixed_precision, gpu_id, warmup_epoch, total_time, warm_signal=None, start_signal=None,
                          bench_id="", result_dict=None, total_iters=None, data='./data/multi30k/multi30k.atok.low.pt',
                          master_addr=None, embs_share_weight=False, proj_share_weight=True, d_k=64, d_v=64, d_model=512,
                          d_word_vec=512, d_inner_hid=2048, n_layers=6, n_head=8, dropout=0.1,
                          n_warmup_steps=4000, label_smoothing=True):
    if len(gpu_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)


    cudnn.benchmark = True 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #========= Loading Dataset =========#
    data = torch.load(data)
    max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, master_addr is not None, batch_size)

    src_vocab_size = training_data.dataset.src_vocab_size
    tgt_vocab_size = training_data.dataset.tgt_vocab_size

    #========= Preparing Model =========#
    if embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        max_token_seq_len,
        tgt_emb_prj_weight_sharing=proj_share_weight,
        emb_src_tgt_weight_sharing=embs_share_weight,
        d_k=d_k,
        d_v=d_v,
        d_model=d_model,
        d_word_vec=d_word_vec,
        d_inner=d_inner_hid,
        n_layers=n_layers,
        n_head=n_head,
        dropout=dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
            d_model, n_warmup_steps)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    # Train
    def benchmark_step():
        iter_num = 0
        iter_warm = 0
        warm = False
        exit_flag = False
        model.train()
        t_start = time.time()
        while True:
            for batch in tqdm(
                training_data, mininterval=2,
                desc='  - (Training)   ', leave=False):
                if iter_num == warmup_epoch - 1:
                    warm = True
                    if warm_signal is not None:
                        warm_signal.value = 1
                    if start_signal is not None:
                        while start_signal.value != 1:
                            time.sleep(0.1)
                    print("Measurement starts...")
                    torch.cuda.cudart().cudaProfilerStart()
                    t_start = time.time()
                if (warm and (time.time() - t_start >= total_time)) or (total_iters is not None and iter_warm == total_iters):
                    t_end = time.time()
                    t_pass = t_end - t_start
                    exit_flag = True
                    torch.cuda.cudart().cudaProfilerStop()
                    break
                # prepare data
                src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
                gold = tgt_seq[:, 1:]
                optimizer.zero_grad()
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
                        loss, n_correct = cal_performance(pred, gold, smoothing=label_smoothing)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
                    # backward
                    loss, n_correct = cal_performance(pred, gold, smoothing=label_smoothing)
                    loss.backward()
                    # update parameters
                    optimizer.step_and_update_lr()
                iter_num += 1
                if warm:
                    iter_warm += 1
            
            if exit_flag:
                break
            
        return t_pass, iter_warm

    benchmark_id = get_benchmark_str(model_name, batch_size, False, bench_id)

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