import time

import hidet
import torch

from utils.bench_util import wait_for_signal

def run_resnet(model_name, batch_size, amp, warmup_iters, total_time,
               total_iters=None, result_dict=None, signal=False):

    hidet.torch.dynamo_config.use_cuda_graph(False)

    if amp:
        hidet.torch.dynamo_config.use_tensor_core(True)
        hidet.torch.dynamo_config.use_fp16(flag=True)
    else:
        hidet.torch.dynamo_config.use_tensor_core(False)
        hidet.torch.dynamo_config.use_fp16(flag=False)

    x = torch.randn(batch_size, 3, 224, 224).cuda()
    model = torch.hub.load(
        'pytorch/vision:v0.9.0', model_name, pretrained=True, verbose=False
    )
    model = model.cuda().eval()

    # optimize the model with 'hidet' backend
    model_opt = torch.compile(model, backend='hidet')

    start_time = None
    num_iters = 0
    warm_iters = 0
    warm = False

    while True:
        
        y = model_opt(x)
        torch.cuda.synchronize()

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