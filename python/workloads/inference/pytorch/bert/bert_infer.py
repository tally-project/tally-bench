import torch
from transformers import (BertConfig, BertForQuestionAnswering, BertTokenizer)
from transformers.data.processors.squad import SquadV1Processor
from transformers import squad_convert_examples_to_features

from workloads.common.utils import get_torch_compile_options
from workloads.common.infer_monitor import get_infer_monitor

def bert_infer(model_name, mode, batch_size, warmup_iters, total_time,
                 load=0.5, trace_file=None, result_dict=None, signal=False, pipe=None):

    if mode in ["single-stream", "server"]:
        assert(batch_size == 1)

    # fine-tuned model from https://huggingface.co/models?search=squad
    model_name_or_path = "bert-large-uncased-whole-word-masking-finetuned-squad"
    max_seq_length = 128
    doc_stride = 128
    max_query_length = 64

    # The following code is adapted from HuggingFace transformers
    # https://github.com/huggingface/transformers/blob/master/examples/run_squad.py

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = (BertConfig, BertForQuestionAnswering, BertTokenizer)
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
    model = model_class.from_pretrained(model_name_or_path, from_tf=False, config=config)
    model = model.cuda().eval()
    
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

    # Compile model with pytorch inductor
    torch.set_float32_matmul_precision("high")

    compile_options = get_torch_compile_options()
    model = torch.compile(model, backend="inductor", options=compile_options)

    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe, load, trace_file)

    while True:

        monitor.on_step_begin()

        inputs = {
            'input_ids':      data[0].to("cuda").reshape(1, max_seq_length),
            'attention_mask': data[1].to("cuda").reshape(1, max_seq_length),
            'token_type_ids': data[2].to("cuda").reshape(1, max_seq_length)
        }

        outputs = model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = inputs["input_ids"][0, answer_start_index : answer_end_index + 1]
        res = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

        torch.cuda.synchronize()

        should_stop = monitor.on_step_end()
        
        if should_stop:
            monitor.write_to_result()
            break
    
    print(f"Bert predicted answer: {res}")