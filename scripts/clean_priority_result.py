import json
import os

def load_json_from_file(f_name):
    result = {}
    if os.path.isfile(f_name):
        with open(f_name) as f:
            result = json.load(f)
    return result

result = load_json_from_file("tally-bench-results/result.json")

tally_priority = result["tally_priority"]

keys = list(tally_priority.keys())

for key in keys:
    key_res = tally_priority[key]
    inner_keys = list(key_res.keys())

    for inner_key in inner_keys:
        if inner_key != "0.01" or len(key_res["0.01"]) != 1:
            del key_res[inner_key]
    
    if tally_priority[key] == {}:
        del tally_priority[key]


def write_json_to_file(_dict, f_name):
    with open(f_name, 'w') as f:
        json.dump(_dict, f, indent=4, sort_keys=True)

write_json_to_file(result, "tally-bench-results/result.json")

