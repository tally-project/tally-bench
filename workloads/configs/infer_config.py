
inference_workloads = {
    "hidet": {
        "resnet50": {
            "batch-sizes": [64, 128],
            "load": [0.2, 0.5, 0.8],
            "amp": [True, False]
        }
    }
}