import sys

def wait_for_signal():
    print("benchmark is warm", flush=True)

    while True:
        sys.stdin.flush()
        inp = sys.stdin.readline()
        if "start" in inp:
            break