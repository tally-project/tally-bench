from transformers import AutoTokenizer
import transformers
import torch

from workloads.common.infer_monitor import get_infer_monitor

def llama2_infer(model_name, mode, batch_size, warmup_iters, total_time,
                 load=0.5, trace_file=None, result_dict=None, signal=False,
                 pipe=None, no_waiting=False):
    
    model_path = "./data/llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prompt = """I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n"""

    if mode in ["single-stream", "server"]:
        assert(batch_size == 1)

    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict,
                                signal, pipe, load, trace_file, no_waiting)

    while True:

        monitor.on_step_begin()

        sequences = pipeline(
            prompt,
            do_sample=False,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=128,
        )

        torch.cuda.synchronize()

        should_stop = monitor.on_step_end()
        if should_stop:
            monitor.write_to_result()
            break

    for seq in sequences:
        print(f"Generated text: {seq['generated_text']}")