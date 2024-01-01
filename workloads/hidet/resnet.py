import hidet
import torch

from workloads.common.infer_monitor import SingleStreamInferMonitor, ServerInferMonitor, OfflineInferMonitor

def resnet_infer(model_name, mode, batch_size, amp, warmup_iters, total_time,
                 bustiness=0.5, result_dict=None, signal=False, pipe=None):

    hidet.torch.dynamo_config.use_cuda_graph(False)

    if amp:
        hidet.torch.dynamo_config.use_tensor_core(True)
        hidet.torch.dynamo_config.use_fp16(flag=True)
    else:
        hidet.torch.dynamo_config.use_tensor_core(False)
        hidet.torch.dynamo_config.use_fp16(flag=False)

    model = torch.hub.load(
        'pytorch/vision:v0.9.0', model_name, pretrained=True, verbose=False
    )
    model = model.cuda().eval()

    # optimize the model with 'hidet' backend
    model_opt = torch.compile(model, backend='hidet')

    if mode == "single-stream":
        x = torch.randn(1, 3, 224, 224).cuda()
        monitor = SingleStreamInferMonitor(warmup_iters, total_time, result_dict, signal, pipe)
    elif mode == "server":
        x = torch.randn(1, 3, 224, 224).cuda()
        monitor = ServerInferMonitor(warmup_iters, total_time, result_dict, signal, pipe, bustiness)
    elif mode == "offline":
        x = torch.randn(batch_size, 3, 224, 224).cuda()
        monitor = OfflineInferMonitor(warmup_iters, total_time, result_dict, signal, pipe)
    else:
        raise Exception("unknown mode")

    while True:

        monitor.on_step_begin()

        y = model_opt(x)
        torch.cuda.synchronize()

        should_stop = monitor.on_step_end()
        if should_stop:
            monitor.write_to_result()
            break