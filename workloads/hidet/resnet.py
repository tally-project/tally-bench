import hidet
import torch
import torchvision

from workloads.common.infer_monitor import get_infer_monitor

def resnet_infer(model_name, mode, batch_size, amp, warmup_iters, total_time,
                 load=0.5, result_dict=None, signal=False, pipe=None):

    if mode in ["single-stream", "server"]:
        batch_size = 1
        
    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe, load)

    hidet.torch.dynamo_config.use_cuda_graph(False)

    if amp:
        hidet.torch.dynamo_config.use_tensor_core(True)
        hidet.torch.dynamo_config.use_fp16(flag=True)
    else:
        hidet.torch.dynamo_config.use_tensor_core(False)
        hidet.torch.dynamo_config.use_fp16(flag=False)

    model = getattr(torchvision.models, model_name)()
    model = model.cuda().eval()

    # optimize the model with 'hidet' backend
    model_opt = torch.compile(model, backend='hidet')
    x = torch.randn(batch_size, 3, 224, 224).cuda()

    while True:

        monitor.on_step_begin()

        y = model_opt(x)
        torch.cuda.synchronize()

        should_stop = monitor.on_step_end()
        if should_stop:
            monitor.write_to_result()
            break