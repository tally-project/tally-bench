from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset

from workloads.common.infer_monitor import get_infer_monitor

def vit_infer(model_name, mode, batch_size, warmup_iters, total_time,
                 load=0.5, trace_file=None, result_dict=None, signal=False, pipe=None):
    
    if mode in ["single-stream", "server"]:
        assert(batch_size == 1)

    monitor = get_infer_monitor(mode, warmup_iters, total_time, result_dict, signal, pipe, load, trace_file)
    
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").cuda()

    inputs = image_processor(image, return_tensors="pt")
    print(inputs)

    while True:

        monitor.on_step_begin()

        with torch.no_grad():
            logits = model(inputs['pixel_values'].cuda()).logits
        
        torch.cuda.synchronize()

        should_stop = monitor.on_step_end()
        if should_stop:
            monitor.write_to_result()
            break
    
    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])