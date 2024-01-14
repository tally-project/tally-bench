import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed

import time
import workloads.pytorch.lstm.data as lstm_data
import workloads.pytorch.lstm.models as models

from utils.bench_util import wait_for_signal
from workloads.common.train_monitor import TrainMonitor

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

# Training
def train_lstm(model_name, batch_size, amp, warmup_iters, total_time,
                    total_iters=None, result_dict=None, signal=False, pipe=None,
                    data_dir='./data/wikitext-2', bptt=35, emsize=200,
                    nhead=2, nhid=200, nlayers=2, dropout=0.2, tied=False):
    train_monitor = TrainMonitor(warmup_iters, total_time, total_iters, result_dict, signal, pipe)
    device = 'cuda'

    corpus = lstm_data.Corpus(data_dir)

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

    ntokens = len(corpus.dictionary)
    if model_name == 'Transformer':
        model = models.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    else:
        model = models.RNNModel(model_name, ntokens, emsize, nhid, nlayers, dropout, tied).to(device)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

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

    ntokens = len(corpus.dictionary)
    if model_name != 'Transformer':
        hidden = model.init_hidden(batch_size)

    model.train()

    while True:
        should_training_stop = False

        for data, targets in trainloader:
            data = data.t()
            targets = targets.t()
            optimizer.zero_grad()
            if amp:
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
            
            should_training_stop = train_monitor.on_step_end(loss)
            if should_training_stop:
                break
        
        if should_training_stop:
            break