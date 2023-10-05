from utils.util import execute_cmd

def start_mps():
    print("Starting MPS ...")
    # execute_cmd("sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS")
    execute_cmd("nvidia-cuda-mps-control -d")

def shut_down_mps():
    print("Shutting down MPS ...")
    execute_cmd("echo quit | nvidia-cuda-mps-control")
    # execute_cmd("sudo nvidia-smi -i 0 -c DEFAULT")