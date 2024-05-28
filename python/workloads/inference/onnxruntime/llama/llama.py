
import torch
from transformers import LlamaConfig, LlamaTokenizer
import numpy as np
import onnxruntime as ort

from bench_utils.nvidia_smi import get_gpu_model
from workloads.common.infer_monitor import get_infer_monitor

pt_to_np = {
    "torch.int64": np.int64,
    "torch.float32": np.float32,
    "torch.float16": np.float16
}

def get_initial_inputs_and_outputs(config, tokenizer, prompt, device, use_fp16, use_buffer_share):
    tokenizer.pad_token = "[PAD]"
    encodings_dict = tokenizer.batch_encode_plus(prompt, padding=True)
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    input_ids = torch.tensor(encodings_dict["input_ids"], device=device, dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], device=device, dtype=torch.int64)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    
    inputs = {
        "input_ids": input_ids.contiguous(),
        "attention_mask": attention_mask.contiguous(),
        "position_ids": position_ids.contiguous(),
    }

    batch_size, sequence_length = input_ids.shape
    max_sequence_length = config.max_position_embeddings
    num_heads, head_size = config.num_key_value_heads, config.hidden_size // config.num_attention_heads
    for i in range(config.num_hidden_layers):
        past_key = torch.zeros(batch_size, num_heads, max_sequence_length if use_buffer_share else 0, head_size, device=device, dtype=torch_dtype)
        past_value = torch.zeros(batch_size, num_heads, max_sequence_length if use_buffer_share else 0, head_size, device=device, dtype=torch_dtype)
        inputs.update({
            f"past_key_values.{i}.key": past_key.contiguous(),
            f"past_key_values.{i}.value": past_value.contiguous()
        })

    logits = torch.zeros(batch_size, sequence_length, config.vocab_size, device=device, dtype=torch_dtype)
    outputs = {
        "logits": logits.contiguous()
    }
    if not use_buffer_share:
        for i in range(config.num_hidden_layers):
            present_key = torch.zeros(batch_size, num_heads, sequence_length, head_size, device=device, dtype=torch_dtype)
            present_value = torch.zeros(batch_size, num_heads, sequence_length, head_size, device=device, dtype=torch_dtype)
            outputs.update({
                f"present.{i}.key": present_key.contiguous(),
                f"present.{i}.value": present_value.contiguous()
            })

    return inputs, outputs


def apply_io_binding(model, inputs, outputs, use_fp16, use_buffer_share):
    # Check that all model inputs will be provided
    model_inputs = set(map(lambda model_input: model_input.name, model.get_inputs()))
    user_inputs = set(inputs.keys())
    missing_inputs = model_inputs - user_inputs
    if len(missing_inputs):
        print(f"The following model inputs are missing: {missing_inputs}")
        raise Exception("There are missing inputs to the model. Please add them and try again.")

    # Remove unnecessary inputs from model inputs
    unnecessary_inputs = user_inputs - model_inputs
    if len(unnecessary_inputs):
        for unnecessary_input in unnecessary_inputs:
            print(f"Removing unnecessary input '{unnecessary_input}' from user provided inputs")
            del inputs[unnecessary_input]

    # Bind inputs/outputs to IO binding
    io_binding = model.io_binding()
    device = None

    for k, v in inputs.items():
        io_binding.bind_input(
            name=k,
            device_type=v.device.type,
            device_id=0 if v.device.type == "cpu" else v.device.index,
            element_type=pt_to_np[repr(v.dtype)],
            shape=tuple(v.shape),
            buffer_ptr=v.data_ptr()
        )
        device = v.device

    for output in model.get_outputs():
        name = output.name
        if use_buffer_share and "present" in name:
            # Bind KV cache outputs to KV cache inputs
            v = inputs[name.replace("present", "past_key_values")]
            io_binding.bind_output(
                name=name,
                device_type=v.device.type,
                device_id=v.device.index,
                element_type=np.float16,
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr()
            )
        else:
            v = outputs[name]
            io_binding.bind_output(
                name=name,
                device_type=device.type,
                device_id=0 if device.type == "cpu" else device.index,
                element_type=(np.float16 if use_fp16 else np.float32),
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr()
            )

    return io_binding

def llama2_infer(model_name, mode, batch_size, warmup_iters, total_time,
                 load=0.5, trace_file=None, result_dict=None, signal=False,
                 pipe=None, no_waiting=False):

    if mode in ["single-stream", "server"]:
        assert(batch_size == 1)

    assert 'CUDAExecutionProvider' in ort.get_available_providers()

    # User settings
    model_name = "./data/llama-2-7b-hf"
    onnx_model_path = "./data/llama2-7b-fp16-gqa/rank_0_llama-2-7b-hf_decoder_merged_model_fp16.onnx"
    use_fp16 = True  # True when KV cache inputs/outputs are in float16
    use_buffer_share = True  # True when --use_gqa was passed during export

    if "H100" in get_gpu_model():
        onnx_model_path = "./data/llama2-7b-fp16/rank_0_llama-2-7b-hf_decoder_merged_model_fp16.onnx"
        use_buffer_share = False

    prompt = ["""I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n"""]
    max_length = 128  # max(prompt length + generation length)

    device_id = 0
    device = torch.device(f"cuda:{device_id}")  # Change to torch.device("cpu") if running on CPU

    config = LlamaConfig.from_pretrained(model_name)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    sess_options = ort.SessionOptions()
    ep = ("CUDAExecutionProvider", {"device_id": device_id})  # change to ep = "CPUExecutionProvider" for CPU
    model = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=[ep])

    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe,
                                load, trace_file, no_waiting)

    while True:

        monitor.on_step_begin()

        # Get model and its initial inputs/outputs
        inputs, outputs = get_initial_inputs_and_outputs(config, tokenizer, prompt, device, use_fp16, use_buffer_share)

        all_token_ids = inputs["input_ids"].clone()
        batch_size, sequence_length = all_token_ids.shape
        num_heads, head_size = config.num_key_value_heads, config.hidden_size // config.num_attention_heads

        current_length = sequence_length
        has_eos = torch.zeros(batch_size, device=device, dtype=torch.bool)

        while current_length <= max_length:
            # Run inference
            io_binding = apply_io_binding(model, inputs, outputs, use_fp16, use_buffer_share)
            io_binding.synchronize_inputs()
            model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # Sample with argmax (greedy search)
            if outputs["logits"].shape[1] > 1:
                prompt_end_indices = inputs["attention_mask"].sum(1) - 1
                idxs = prompt_end_indices.unsqueeze(dim=1).repeat(1, config.vocab_size).view(batch_size, 1, config.vocab_size)
                next_token_logits = torch.gather(outputs["logits"], 1, idxs).squeeze()
            else:
                next_token_logits = outputs["logits"][:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Check if we previously reached EOS token id or if generated token id is EOS token id
            has_eos = has_eos | next_tokens == tokenizer.eos_token_id

            # Determine which new tokens to add to list of all token ids
            # Add EOS token ids for batch entries that ended early (ragged batching scenario where some batch entries ended early and some haven't)
            tokens_to_add = next_tokens.masked_fill(has_eos, tokenizer.eos_token_id).reshape([batch_size, 1])
            all_token_ids = torch.cat([all_token_ids, tokens_to_add], dim=-1)

            # Return early if all batch entries have reached EOS token id
            current_length += 1
            if torch.all(has_eos) or current_length > max_length:
                break

            # Update inputs for next inference run
            inputs["input_ids"] = tokens_to_add
            inputs["position_ids"] = torch.max(inputs["position_ids"], dim=1)[0].reshape(batch_size, 1) + 1
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], (~has_eos).to(torch.int64).reshape(batch_size, 1)], 1)

            # Set logits to zeros for next inference run and re-use memory buffer
            if outputs["logits"].shape[1] != 1:
                outputs["logits"] = outputs["logits"][:, :1, :].contiguous()
            outputs["logits"].zero_()

            if not use_buffer_share:
                for i in range(config.num_hidden_layers):
                    inputs[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
                    inputs[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

                new_sequence_length = inputs["attention_mask"].shape[1]
                for i in range(config.num_hidden_layers):
                    present_key = torch.zeros(batch_size, num_heads, new_sequence_length, head_size, device=device, dtype=torch_dtype)
                    present_value = torch.zeros(batch_size, num_heads, new_sequence_length, head_size, device=device, dtype=torch_dtype)
                    outputs.update({
                        f"present.{i}.key": present_key.contiguous(),
                        f"present.{i}.value": present_value.contiguous()
                    })

        torch.cuda.synchronize()

        generated_text = tokenizer.batch_decode(all_token_ids, skip_special_tokens=True)

        should_stop = monitor.on_step_end()
        if should_stop:
            monitor.write_to_result()
            break
    
    # Batch decoding at end of generation
    for text in generated_text:
        print(f"Generated text: {text}")
