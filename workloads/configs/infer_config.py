# Inference benchmark modes:
#   1. single-stream (back-to-back inference with batch-size 1)
#   2. server (arrival rate of possion distribution)

inference_load_factors = [0.5]

inference_workloads = {
    "pytorch": [
        "bert",
        "yolov6m",
        "llama-2-7b",
        "gpt-neo-2.7B",
        "stable-diffusion"
    ],
    "hidet": [
        "resnet50",
        "efficientnet_b0"
    ]
}