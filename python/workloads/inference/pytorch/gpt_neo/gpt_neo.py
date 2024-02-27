from transformers import AutoTokenizer
import transformers
import torch

from workloads.common.infer_monitor import get_infer_monitor

def gpt_neo_infer(model_name, mode, batch_size, warmup_iters, total_time,
                 load=0.5, result_dict=None, signal=False, pipe=None):
    
    if mode in ["single-stream", "server"]:
        assert(batch_size == 1)

    generator = transformers.pipeline(
        'text-generation',
        model='EleutherAI/gpt-neo-2.7B',
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prompt = "EleutherAI has"

    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe, load)

    while True:

        monitor.on_step_begin()

        with torch.no_grad():
            sequences = generator(
                prompt,
                do_sample=False,
                max_length=128,
                num_return_sequences=1
            )

            # for seq in sequences:
            #     print(f"Result: {seq['generated_text']}")
            
            torch.cuda.synchronize()

        should_stop = monitor.on_step_end()
        if should_stop:
            monitor.write_to_result()
            break