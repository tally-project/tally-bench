import hidet
import torch
import torchvision

from workloads.common.infer_monitor import get_infer_monitor

def vision_infer(model_name, mode, batch_size, warmup_iters, total_time,
                 load=0.5, trace_file=None, result_dict=None, signal=False, pipe=None):

    model = getattr(torchvision.models, model_name)()
    model = model.cuda().eval()

    if mode in ["single-stream", "server"]:
        assert(batch_size == 1)
        
    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe, load, trace_file)

    hidet.torch.dynamo_config.search_space(2)
    hidet.torch.dynamo_config.use_cuda_graph(False)

    hidet.torch.dynamo_config.use_tensor_core(True)
    hidet.torch.dynamo_config.use_fp16(True)
    hidet.torch.dynamo_config.use_fp16_reduction(True)

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