import time
from bench_utils.utils import execute_cmd, logger

def start_mps():
    logger.info("Starting MPS ...")
    execute_cmd("nvidia-cuda-mps-control -d")
    time.sleep(10)

def shut_down_mps():
    logger.info("Shutting down MPS ...")
    execute_cmd("echo quit | nvidia-cuda-mps-control")