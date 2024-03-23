# Inference benchmark modes:
#   1. single-stream (back-to-back inference with batch-size 1)
#   2. server (arrival rate of possion distribution)

from bench_utils.tally import TallyConfig

inference_load_factors = [0.5]

default_configs = [
    TallyConfig("priority", max_allowed_latency=0.1),
    TallyConfig("priority", max_allowed_latency=0.1, min_wait_time=1),
    TallyConfig("priority", max_allowed_latency=0.5),
    TallyConfig("priority", max_allowed_latency=0.5, min_wait_time=1),
    TallyConfig("priority", max_allowed_latency=0.5, wait_time_to_use_original=100),
    TallyConfig("priority", max_allowed_latency=0.5, min_wait_time=1, wait_time_to_use_original=100),
    TallyConfig("priority", max_allowed_latency=1.0),
    TallyConfig("priority", use_original_configs=True),
    TallyConfig("priority", use_original_configs=True, min_wait_time=1.0),
    TallyConfig("priority", use_space_share=True),
]

inference_workloads = {
    "onnxruntime": {
        "bert": [
            TallyConfig("priority", max_allowed_latency=0.5, use_original_kernel_latency_threshold=0.5),
            TallyConfig("priority", max_allowed_latency=0.5, use_original_kernel_latency_threshold=0.5),
            TallyConfig("priority", max_allowed_latency=0.5, use_original_kernel_latency_threshold=3),
            TallyConfig("priority", max_allowed_latency=0.5, use_original_kernel_latency_threshold=10),
            TallyConfig("priority", use_space_share=True, space_share_max_sm_perc=0.2),
            TallyConfig("priority", use_space_share=True, space_share_max_sm_perc=0.5),
            TallyConfig("priority", use_space_share=True, space_share_max_sm_perc=0.8),
            TallyConfig("priority", use_space_share=True, space_share_max_sm_perc=1),
        ],
        "llama-2-7b": [
            TallyConfig("priority", use_original_configs=True, min_wait_time=5.0),
            TallyConfig("priority", use_original_configs=True, min_wait_time=10.0),
        ],
    },
    "pytorch": {
        "yolov6m": [
            TallyConfig("priority", max_allowed_latency=0.5, min_wait_time=3),
        ],
        "gpt-neo-2.7B": [],
        "stable-diffusion": [
            TallyConfig("priority", use_original_configs=True, min_wait_time=5.0),
        ],
    },
    "hidet": {
        "resnet50": [
            TallyConfig("priority", max_allowed_latency=0.05, min_wait_time=0.1),
            TallyConfig("priority", max_allowed_latency=0.01, min_wait_time=1),
        ],
        "inception_v3": [
            TallyConfig("priority", max_allowed_latency=0.01, min_wait_time=0.1),
        ],
    },
    "tvm": {
        "resnet50": [
            # TallyConfig("priority", max_allowed_latency=0.05, min_wait_time=0.1),
            TallyConfig("priority", max_allowed_latency=0.01, min_wait_time=0.1),
        ]
    }
}


for framework in inference_workloads:
    for model in inference_workloads[framework]:
        inference_workloads[framework][model].extend(default_configs)


def get_tally_configs(framework, model_name, mode):
    configs = inference_workloads[framework][model_name]

    if "single-stream" in mode:
        configs = [config for config in configs if config.wait_time_to_use_original is None]

    return configs