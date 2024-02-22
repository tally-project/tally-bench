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
        inner_res = key_res[inner_key]
    
        if "error" in inner_res:
            del key_res[inner_key]
        
    if key_res == {}:
        del tally_priority[key]        

def write_json_to_file(_dict, f_name):
    with open(f_name, 'w') as f:
        json.dump(_dict, f, indent=4, sort_keys=True)

write_json_to_file(result, "tally-bench-results/result.json")

