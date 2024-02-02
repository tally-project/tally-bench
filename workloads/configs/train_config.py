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
        "pegasus-x-base": {
            "batch-sizes": [2, 4],
            "amp": [True, False]
        },
        "whisper-small": {
            "batch-sizes": [8, 16],
            "amp": [True, False]
        },
        "yolov6m": {
            "batch-sizes": [16, 32],
            "amp": [True]
        },
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