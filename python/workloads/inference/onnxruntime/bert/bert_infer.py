import psutil
import os
import onnxruntime
import numpy
import torch

import os
import torch
from transformers import BertTokenizer
from transformers.data.processors.squad import SquadV1Processor
from transformers import squad_convert_examples_to_features

from workloads.common.infer_monitor import get_infer_monitor

def bert_infer(model_name, mode, batch_size, warmup_iters, total_time,
                load=0.5, result_dict=None, signal=False, pipe=None):
    
    if mode in ["single-stream", "server"]:
        assert(batch_size == 1)

    assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()

    # fine-tuned model from https://huggingface.co/models?search=squad
    model_name_or_path = "bert-large-uncased-whole-word-masking-finetuned-squad"
    max_seq_length = 128
    doc_stride = 128
    max_query_length = 64

    # The following code is adapted from HuggingFace transformers
    # https://github.com/huggingface/transformers/blob/master/examples/run_squad.py

    # Load pretrained tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)

    # load data
    predict_file = "./data/SQUAD_DIR/dev-v1.1.json"
    total_samples = 1
    processor = SquadV1Processor()
    examples = processor.get_dev_examples(None, filename=predict_file)

    features, dataset = squad_convert_examples_to_features( 
                            examples=examples[:total_samples], # convert enough examples for this notebook
                            tokenizer=tokenizer,
                            max_seq_length=max_seq_length,
                            doc_stride=doc_stride,
                            max_query_length=max_query_length,
                            is_training=False,
                            return_dataset='pt'
                        )
    data = dataset[0]

    # Set up onnx inference environment
    sess_options = onnxruntime.SessionOptions()

    # Please change the value according to best setting in Performance Test Tool result.
    sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)

    model_path = "./data/bert-base-cased-squad_opt_gpu_fp16.onnx"
    session = onnxruntime.InferenceSession(model_path, sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe, load)

    while True:

        monitor.on_step_begin()

        with torch.no_grad():
            ort_inputs = {
                'input_ids':  data[0].cpu().reshape(1, max_seq_length).numpy(),
                'input_mask': data[1].cpu().reshape(1, max_seq_length).numpy(),
                'segment_ids': data[2].cpu().reshape(1, max_seq_length).numpy()
            }

            ort_outputs = session.run(None, ort_inputs)
            torch.cuda.synchronize()

            start_logits = torch.from_numpy(ort_outputs[0])
            end_logits = torch.from_numpy(ort_outputs[1])

            answer_start_index = start_logits.argmax()
            answer_end_index = end_logits.argmax()

            predict_answer_tokens = ort_inputs["input_ids"][0, answer_start_index : answer_end_index + 1]
            res = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

        should_stop = monitor.on_step_end()
        
        if should_stop:
            monitor.write_to_result()
            break

    print(f"Bert predicted answer: {res}")