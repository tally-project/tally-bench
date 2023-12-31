
# All inference benchmark contains three mode:
#   1. single-stream (back-to-back inference with batch-size 1)
#   2. server (arrival rate of possion distribution)
#   3. offline processing with large batch size

inference_list = {
    "hidet": {
        "resnet50": {
            "batch-sizes": [64],
            "amp": [True, False]
        }
    },
    "pytorch": {
        "resnet50": {
            "batch-sizes": [64],
            "amp": [True, False]
        }
    }
}