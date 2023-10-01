import subprocess

def execute_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True, universal_newlines=True)
    process.wait()