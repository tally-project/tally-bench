import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed

import os
import time
from tqdm import tqdm 

from workloads.pytorch.translation.transformer import Constants
from workloads.pytorch.translation.dataset import TranslationDataset, paired_collate_fn
from workloads.pytorch.translation.transformer.Models import Transformer
from workloads.pytorch.translation.transformer.Optim import ScheduledOptim

from utils.bench_util import wait_for_signal

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


# Training
def benchmark_transformer(model_name, batch_size, amp, warmup_iters, total_time, total_iters=None, result_dict=None, signal=False, pipe=None,
                    data='./data/multi30k/multi30k.atok.low.pt', master_addr=None, embs_share_weight=False, proj_share_weight=True,
                    d_k=64, d_v=64, d_model=512, d_word_vec=512, d_inner_hid=2048, n_layers=6, n_head=8, dropout=0.1,
                    n_warmup_steps=4000, label_smoothing=True):
    device = 'cuda'

    data = torch.load(data)
    max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, master_addr is not None, batch_size)

    src_vocab_size = training_data.dataset.src_vocab_size
    tgt_vocab_size = training_data.dataset.tgt_vocab_size

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

        for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]
            optimizer.zero_grad()
            if amp:
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