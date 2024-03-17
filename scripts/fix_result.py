import json
import os

def load_json_from_file(f_name):
    result = {}
    if os.path.isfile(f_name):
        with open(f_name) as f:
            result = json.load(f)
    return result

result = load_json_from_file("tally_bench_results/result.json")

small_batch_jobs =  [
    'pytorch_resnet50_train_64',
    'pytorch_resnet50_train_64_amp',
    'pytorch_whisper-large-v3_train_8',
    'pytorch_whisper-large-v3_train_8_amp',
    'pytorch_pointnet_train_64',
    'pytorch_pointnet_train_64_amp',
    'pytorch_pegasus-x-base_train_4',
    'pytorch_pegasus-x-base_train_4_amp',
    'pytorch_bert_train_16',
    'pytorch_bert_train_16_amp',
]

def fix_result(result):
    if isinstance(result, dict):

        keys = list(result.keys())
        for key in keys:

            found = False
            for small_batch_job in small_batch_jobs:
                if small_batch_job in key:
                    del result[key]
                    found = True
                    break
            
            if not found:
                val = result[key]
                fix_result(val)
    elif isinstance(result, list):
        for item in result:
            fix_result(item)


fix_result(result)

def write_json_to_file(_dict, f_name):
    with open(f_name, 'w') as f:
        json.dump(_dict, f, indent=4, sort_keys=True)

write_json_to_file(result, "tally_bench_results/result_test.json")

