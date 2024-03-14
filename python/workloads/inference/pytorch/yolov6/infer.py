#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import pathlib

import torch

from workloads.common.util import get_torch_compile_options
from workloads.common.infer_monitor import get_infer_monitor

curr_dir = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(curr_dir / '../../../common/YOLOv6'))

from yolov6.layers.common import DetectBackend


def yolov6_infer(model_name, mode, batch_size, warmup_iters, total_time,
                 load=0.5, trace_file=None, result_dict=None, signal=False, pipe=None):
    device = torch.device("cuda")
    model = DetectBackend(f"./data/weights/{model_name}.pt", device=device).eval()

    torch.set_float32_matmul_precision("high")

    compile_options = get_torch_compile_options()
    model = torch.compile(model, backend="inductor", options=compile_options)

    if mode in ["single-stream", "server"]:
        assert(batch_size == 1)

    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe, load, trace_file)
    data = torch.randn(batch_size, 3, 640, 640).cuda()

    while True:

        monitor.on_step_begin()

        y = model(data)
        
        torch.cuda.synchronize()

        should_stop = monitor.on_step_end()
        if should_stop:
            monitor.write_to_result()
            break