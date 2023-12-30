import subprocess
import json
import os

def execute_cmd(cmd, get_output=False):
    if get_output:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        process.wait()
        out, err = process.communicate()
        rc = process.returncode
        return out, err, rc
    else:
        process = subprocess.Popen(cmd, shell=True, universal_newlines=True)
        process.wait()

def write_json_to_file(_dict, f_name):
    with open(f_name, 'w') as f:
        json.dump(_dict, f, indent=4, sort_keys=True)

def load_json_from_file(f_name):
    result = {}
    if os.path.isfile(f_name):
        with open(f_name) as f:
            result = json.load(f)
    return result