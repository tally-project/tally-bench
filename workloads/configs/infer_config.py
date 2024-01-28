# Inference benchmark modes:
#   1. single-stream (back-to-back inference with batch-size 1)
#   2. server (arrival rate of possion distribution)

inference_workloads = {
    "hidet": {
        "resnet50": {
            "load": [0.2, 0.5, 0.8],
            "amp": [True, False]
        }
    },
    "pytorch": {
        "resnet50": {
            "load": [0.2, 0.5, 0.8],
            "amp": [True, False]
        }
    }
}