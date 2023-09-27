import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import time
import torchvision

from utils.bench_util import wait_for_signal

# Training
def benchmark_imagenet(model_name, batch_size, amp, warmup_iters, total_time,
                        total_iters=None, result_dict=None, signal=False):

    model = getattr(torchvision.models, model_name)()
    model = model.cuda()

    compile_options = {
        "epilogue_fusion": True,
        "max_autotune": True,
        "triton.cudagraphs": False,
    }
    model = torch.compile(model, backend='inductor', options=compile_options)

    data = torch.randn(batch_size, 3, 224, 224)
    target = torch.LongTensor(batch_size).random_() % 1000
    data, target = data.cuda(), target.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    if amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    start_time = None
    num_iters = 0
    warm_iters = 0
    warm = False
    model.train()

    while True:
        optimizer.zero_grad()

        if amp:
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
        
        # Increment iterations
        num_iters += 1
        if warm:
            warm_iters += 1

            # Break if reaching total iterations
            if warm_iters == total_iters:
                break

            # Or break if time is up
            curr_time = time.time()
            if curr_time - start_time >= total_time:
                break

        if num_iters == warmup_iters:
            warm = True

            if signal:
                wait_for_signal()

            start_time = time.time()
            print("Measurement starts ...")

    end_time = time.time()
    time_elapsed = end_time - start_time
    
    if result_dict is not None:
        result_dict["time_elapsed"] = time_elapsed
        result_dict["iters"] = warm_iters

    return time_elapsed, warm_iters