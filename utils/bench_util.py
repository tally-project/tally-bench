import sys

def wait_for_signal():
    print("benchmark is warm", flush=True)

    while True:
        inp = sys.stdin.readline()
        if "start" in inp:
            break