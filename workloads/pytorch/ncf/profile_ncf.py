import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import os
import time

import workloads.pytorch.ncf.models as models
import workloads.pytorch.ncf.config as config
import workloads.pytorch.ncf.data_utils as data_utils

from utils.bench_util import wait_for_signal

# Training
def benchmark_ncf(model_name, batch_size, amp, warmup_iters, total_time,
                    total_iters=None, result_dict=None, signal=False,
                    num_ng=4, factor_num=32, num_layers=3, dropout=0.0, lr=0.001):
    
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
    train_loader.dataset.ng_sample()

    while True:
        stop = False

        for idx, (user, item, label) in enumerate(train_loader):
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()
            optimizer.zero_grad()
            if amp:
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