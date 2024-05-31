import json
import os

def count_jobs(key):
    count = 0
    count += key.count("hidet")
    count += key.count("pytorch")
    count += key.count("onnxruntime")
    return count

def load_json_from_file(f_name):
    result = {}
    if os.path.isfile(f_name):
        with open(f_name) as f:
            result = json.load(f)
    return result

result = load_json_from_file("tally_bench_results/result.json")

def fix_result(result):
    if isinstance(result, dict):
        keys = list(result.keys())
        for key in keys:

            if "inception" in key:
                del result[key]
            else:
                val = result[key]
                fix_result(val)

    elif isinstance(result, list):
        for item in result:
            fix_result(item)


fix_result(result)

def write_json_to_file(_dict, f_name):
    with open(f_name, 'w') as f:
        json.dump(_dict, f, indent=4, sort_keys=True)

write_json_to_file(result, "tally_bench_results/result.json")

