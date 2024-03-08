# Inference benchmark modes:
#   1. single-stream (back-to-back inference with batch-size 1)
#   2. server (arrival rate of possion distribution)

from bench_utils.tally import TallyConfig

inference_load_factors = [0.5]

default_configs = [
    TallyConfig("priority", max_allowed_latency=0.01),
    TallyConfig("priority", max_allowed_latency=0.05),
    TallyConfig("priority", max_allowed_latency=0.1),
    TallyConfig("priority", max_allowed_latency=0.5),
    TallyConfig("priority", max_allowed_latency=1.0),
    TallyConfig("priority", max_allowed_latency=5.0),
    TallyConfig("priority", use_original_configs=True),
    TallyConfig("priority", use_original_configs=True, min_wait_time=1.0),
    TallyConfig("priority", use_space_share=True),
]

inference_workloads = {
    # "onnxruntime": {
    #     "bert": [],
    #     "llama-2-7b": [
    #         TallyConfig("priority", use_original_configs=True, min_wait_time=5.0),
    #         TallyConfig("priority", use_original_configs=True, min_wait_time=10.0),
    #     ],
    # },
    # "pytorch": {
    #     "yolov6m": [],
    #     "gpt-neo-2.7B": [],
    #     "stable-diffusion": [],
    # },
    # "hidet": {
    #     "resnet50": [],
    #     "inception_v3": [],
    # }
}

for framework in inference_workloads:
    for model in inference_workloads[framework]:
        inference_workloads[framework][model].extend(default_configs)