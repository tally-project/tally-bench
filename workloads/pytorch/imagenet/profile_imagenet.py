from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import time
import os
import torchvision

from torch.nn import DataParallel

from models.util import get_benchmark_str

# Training
def benchmark_imagenet(model_name, batch_size, mixed_precision, gpu_id, warmup_epoch,
                       total_time, warm_signal=None, start_signal=None, bench_id="", result_dict=None, total_iters=None):
    if len(gpu_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)

    cudnn.benchmark = True

    model = getattr(torchvision.models, model_name)()
    model = model.cuda()

    data = torch.randn(batch_size, 3, 224, 224)
    target = torch.LongTensor(batch_size).random_() % 1000
    data, target = data.cuda(), target.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None
    
    if len(gpu_id) > 1:
        model = DataParallel(model)
    
    def benchmark_step():
        t_start = time.time()
        iter_num = 0
        iter_warm = 0
        warm = False
        while True:
            optimizer.zero_grad()
            if iter_num == warmup_epoch - 1:
                t_warmend = time.time()
                warm = True
                if warm_signal is not None:
                    warm_signal.value = 1
                if start_signal is not None:
                    while start_signal.value != 1:
                        time.sleep(0.1)
                print("Measurement starts...")
                torch.cuda.cudart().cudaProfilerStart()
                t_warmend = time.time()
            # Reach timeout: exit profiling
            if (warm and (time.time() - t_warmend >= total_time)) or (total_iters is not None and iter_warm == total_iters):
                t_end = time.time()
                t_pass = t_end - t_warmend
                torch.cuda.cudart().cudaProfilerStop()
                break
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            iter_num += 1
            if warm:
                iter_warm += 1
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