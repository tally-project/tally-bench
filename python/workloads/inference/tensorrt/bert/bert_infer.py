import torch
import torch_tensorrt
from transformers import BertModel

from workloads.common.infer_monitor import get_infer_monitor

def bert_infer(model_name, mode, batch_size, warmup_iters, total_time,
                 load=0.5, result_dict=None, signal=False, pipe=None):

    # Initialize model with float precision and sample inputs
    model = BertModel.from_pretrained("bert-base-uncased").eval().to("cuda")
    inputs = [
        torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
        torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
    ]

    # Enabled precision for TensorRT optimization
    enabled_precisions = {torch.float}

    # Workspace size for TensorRT
    workspace_size = 20 << 30

    # Maximum number of TRT Engines
    # (Lower value allows more graph segmentation)
    min_block_size = 7

    # Operations to Run in Torch, regardless of converter support
    torch_executed_ops = {}

    # Define backend compilation keyword arguments
    compilation_kwargs = {
        "debug": True,
        "enabled_precisions": enabled_precisions,
        "workspace_size": workspace_size,
        "min_block_size": min_block_size,
        "torch_executed_ops": torch_executed_ops,
    }

    # Build and compile the model with torch.compile, using Torch-TensorRT backend
    optimized_model = torch.compile(
        model,
        backend="torch_tensorrt",
        options=compilation_kwargs,
    )

    if mode in ["single-stream", "server"]:
        assert(batch_size == 1)
        
    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe, load)

    while True:

        monitor.on_step_begin()

        y = optimized_model(*inputs)
        torch.cuda.synchronize()

        should_stop = monitor.on_step_end()
        
        if should_stop:
            monitor.write_to_result()
            break