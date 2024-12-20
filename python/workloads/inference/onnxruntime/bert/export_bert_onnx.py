import os
import torch
from transformers import (BertConfig, BertForQuestionAnswering, BertTokenizer)

cache_dir = "./bert-logs"

predict_file = "./data/SQUAD_DIR/dev-v1.1.json"

# Whether allow overwriting existing ONNX model and download the latest script from GitHub
enable_overwrite = True

# Total samples to inference, so that we can get average latency
total_samples = 1

# ONNX opset version
opset_version=11

# fine-tuned model from https://huggingface.co/models?search=squad
model_name_or_path = "bert-large-uncased-whole-word-masking-finetuned-squad"
max_seq_length = 128
doc_stride = 128
max_query_length = 64

# The following code is adapted from HuggingFace transformers
# https://github.com/huggingface/transformers/blob/master/examples/run_squad.py

# Load pretrained model and tokenizer
config_class, model_class, tokenizer_class = (BertConfig, BertForQuestionAnswering, BertTokenizer)
config = config_class.from_pretrained(model_name_or_path, cache_dir=cache_dir)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True, cache_dir=cache_dir)
model = model_class.from_pretrained(model_name_or_path,
                                    from_tf=False,
                                    config=config,
                                    cache_dir=cache_dir)
# load some examples
from transformers.data.processors.squad import SquadV1Processor

processor = SquadV1Processor()
examples = processor.get_dev_examples(None, filename=predict_file)

from transformers import squad_convert_examples_to_features
features, dataset = squad_convert_examples_to_features( 
                        examples=examples[:total_samples], # convert enough examples for this notebook
                        tokenizer=tokenizer,
                        max_seq_length=max_seq_length,
                        doc_stride=doc_stride,
                        max_query_length=max_query_length,
                        is_training=False,
                        return_dataset='pt'
                    )

output_dir = os.path.join(".", "onnx_models")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)   
export_model_path = os.path.join(output_dir, 'bert-base-cased-squad_opset{}.onnx'.format(opset_version))

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# Get the first example data to run the model and export it to ONNX
data = dataset[0]
inputs = {
    'input_ids':      data[0].to(device).reshape(1, max_seq_length),
    'attention_mask': data[1].to(device).reshape(1, max_seq_length),
    'token_type_ids': data[2].to(device).reshape(1, max_seq_length)
}

# Set model to inference mode, which is required before exporting the model because some operators behave differently in 
# inference and training mode.
model.eval()
model.to(device)

if enable_overwrite or not os.path.exists(export_model_path):
    with torch.no_grad():
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(model,                                            # model being run
                          args=tuple(inputs.values()),                      # model input (or a tuple for multiple inputs)
                          f=export_model_path,                              # where to save the model (can be a file or file-like object)
                          opset_version=opset_version,                      # the ONNX version to export the model to
                          do_constant_folding=True,                         # whether to execute constant folding for optimization
                          input_names=['input_ids',                         # the model's input names
                                       'input_mask', 
                                       'segment_ids'],
                          output_names=['start', 'end'],                    # the model's output names
                          dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                        'input_mask' : symbolic_names,
                                        'segment_ids' : symbolic_names,
                                        'start' : symbolic_names,
                                        'end' : symbolic_names})
        print("Model exported at ", export_model_path)

# optimize with
# python3 -m onnxruntime.transformers.optimizer --input "./onnx_models/bert-base-cased-squad_opset11.onnx" --output "./data/bert-base-cased-squad_opt_gpu_fp16.onnx" --float16 --use_gpu