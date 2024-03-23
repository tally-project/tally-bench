import torch
import tvm

from workloads.common.infer_monitor import get_infer_monitor
from workloads.common.utils import export_torchvision_model_as_onnx
from workloads.common.tvm_utils import tvm_graph_module_from_onnx

def vision_infer(model_name, mode, batch_size, warmup_iters, total_time,
                 load=0.5, trace_file=None, result_dict=None, signal=False, pipe=None):

    if mode in ["single-stream", "server"]:
        assert(batch_size == 1)
        
    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe, load, trace_file)

    model_path = f"./data/{model_name}.onnx"
    export_torchvision_model_as_onnx(model_name, model_path)
    
    if model_name == 'resnet50':
        input_names = ['data']
        input_tensors = [torch.randn(batch_size, 3, 224, 224)]
    elif model_name == 'inception_v3':
        input_names = ['data']
        input_tensors = [torch.randn(batch_size, 3, 299, 299)]

    tune_engine = 'ansor'
    tune_trials = 800

    gmod = tvm_graph_module_from_onnx(
        onnx_model_path=model_path,
        input_shapes={
            name: tensor.shape for name, tensor in zip(input_names, input_tensors)
        },
        tune_autotvm=(tune_engine == 'autotvm'),
        tune_ansor=(tune_engine == 'ansor'),
        tune_trial_per_task=tune_trials
    )
    inputs = {name: tensor for name, tensor in zip(input_names, input_tensors)}

    for name, tensor in inputs.items():
        gmod.set_input(name, value=tvm.nd.array(tensor.cpu().numpy()))

    while True:

        monitor.on_step_begin()

        gmod.run()
        outputs = []
        for i in range(gmod.get_num_outputs()):
            output: tvm.nd.NDArray = gmod.get_output(i)
            outputs.append(output)

        torch.cuda.synchronize()

        should_stop = monitor.on_step_end()
        
        if should_stop:
            monitor.write_to_result()
            break