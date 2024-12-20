import torch
import time
from diffusers import DiffusionPipeline

import torch_tensorrt

from workloads.common.infer_monitor import get_infer_monitor

def stable_diffusion_infer(model_name, mode, batch_size, warmup_iters, total_time,
                 load=0.5, trace_file=None, result_dict=None, signal=False, pipe=None,
                 no_waiting=False):

    if mode in ["single-stream", "server"]:
        assert(batch_size == 1)
        
    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict,
                                signal, pipe, load, trace_file, no_waiting)

    model_id = "CompVis/stable-diffusion-v1-4"

    # Instantiate Stable Diffusion Pipeline with FP16 weights
    pipe = DiffusionPipeline.from_pretrained(
        model_id, revision="fp16", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # Optimize the UNet portion with Torch-TensorRT
    pipe.unet = torch.compile(
        pipe.unet,
        backend="torch_tensorrt",
        options={
            "truncate_long_and_double": True,
            "precision": torch.float16,
        },
        dynamic=False,
    )

    prompt = "a majestic castle in the clouds"

    while True:

        monitor.on_step_begin()

        image = pipe(prompt).images[0]
        torch.cuda.synchronize()

        should_stop = monitor.on_step_end()
        
        if should_stop:
            monitor.write_to_result()
            break

    # image.save("majestic_castle.png")