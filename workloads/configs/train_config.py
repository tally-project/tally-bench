training_workloads = {
    "pytorch": {
        "resnet50": {
            "batch-sizes": [64, 128],
            "amp": [True, False]
        },
        "EfficientNetB0": {
            "batch-sizes": [64, 128],
            "amp": [True, False]
        },
        "pointnet": {
            "batch-sizes": [64, 128],
            "amp": [True, False]
        },
        "bert": {
            "batch-sizes": [16, 32],
            "amp": [True, False]
        },
        "yolov6m": {
            "batch-sizes": [32, 64],
            "amp": [True]
        },
        "pegasus-x-base": {
            "batch-sizes": [1, 2],
            "amp": [True, False]
        }
        # "dcgan": {
        #     "batch-sizes": [64],
        #     "amp": [True, False]
        # },
        # "LSTM": {
        #     "batch-sizes": [64],
        #     "amp": [True, False]
        # },
        # "NeuMF-pre": {
        #     "batch-sizes": [64],
        #     "amp": [True, False]
        # },
        # "transformer": {
        #     "batch-sizes": [8],
        #     "amp": [False]
        # },
    }
}