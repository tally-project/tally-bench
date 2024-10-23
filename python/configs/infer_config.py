inference_load_factors = [0.5]

inference_workloads = {
    "onnxruntime": [
        "bert",
        "llama-2-7b"
    ],
    "pytorch": [
        "yolov6m",
        "gpt-neo-2.7B",
        "stable-diffusion",
    ],
    "hidet": [
        "resnet50",
    ]
}

inference_workloads_small = {
    "onnxruntime": [
        "bert",
        "llama-2-7b"
    ],
}