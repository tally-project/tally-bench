training_workloads = {
    "pytorch": {
        "resnet50": {
            "batch-sizes": [128],
            "amp": [True, False]
        },
        # "yolov6l": {
        #     "batch-sizes": [32, 64],
        #     "amp": [True]
        # },
        "pointnet": {
            "batch-sizes": [128],
            "amp": [True, False]
        },
        "bert": {
            "batch-sizes": [32],
            "amp": [True, False]
        },
        "pegasus-x-base": {
            "batch-sizes": [8],
            "amp": [True, False]
        },
        "whisper-large-v3": {
            "batch-sizes": [16],
            "amp": [True, False]
        },
        "gpt2-large": {
            "batch-sizes": [1],
            "amp": [True, False]
        }
    }
}