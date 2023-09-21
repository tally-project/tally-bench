import subprocess

def shut_down_mps():
    print("Shutting down MPS ...")
    process = subprocess.Popen(f"echo quit | nvidia-cuda-mps-control", shell=True, universal_newlines=True)
    process.wait()

def start_mps():
    print("Starting MPS ...")
    process = subprocess.Popen(f"CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d", shell=True, universal_newlines=True)
    process.wait()