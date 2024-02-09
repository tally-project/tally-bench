import torch
from transformers import BertModel, AutoTokenizer

from workloads.common.util import get_torch_compile_options
from workloads.common.infer_monitor import get_infer_monitor

def bert_infer(model_name, mode, batch_size, warmup_iters, total_time,
                 load=0.5, result_dict=None, signal=False, pipe=None):

    # Initialize model with float precision and sample inputs
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").eval().to("cuda")

    torch.set_float32_matmul_precision("high")

    compile_options = get_torch_compile_options()
    model = torch.compile(model, backend="inductor", options=compile_options)

    if mode in ["single-stream", "server"]:
        assert(batch_size == 1)

    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe, load)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda")

    while True:

        monitor.on_step_begin()

        y = model(**inputs)
        torch.cuda.synchronize()

        should_stop = monitor.on_step_end()
        
        if should_stop:
            monitor.write_to_result()
            break