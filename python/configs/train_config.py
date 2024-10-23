training_workloads_large = {
    "pytorch": {
        "whisper-large-v3": {
            "batch-sizes": [16],
            "amp": [False]
        },
        "bert": {
            "batch-sizes": [32],
            "amp": [False]
        },
        "pegasus-x-base": {
            "batch-sizes": [4],
            "amp": [False]
        },
        "resnet50": {
            "batch-sizes": [128],
            "amp": [False]
        },
        "pointnet": {
            "batch-sizes": [128],
            "amp": [False]
        },
        "gpt2-large": {
            "batch-sizes": [1],
            "amp": [False]
        }
    }
}