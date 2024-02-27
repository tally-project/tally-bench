def get_torch_compile_options():
    compile_options = {
        "epilogue_fusion": True,
        "max_autotune": True,
        "triton.cudagraphs": False
    }

    return compile_options


def get_benchmark_func(framework, model_name, run_training=True):
    bench_func = None
    training = run_training
    inference = not run_training

    torchvision_models = ["resnet50", "efficientnet_b0", "inception_v3"]

    if framework == "onnxruntime":

        if inference:

            if model_name == "llama-2-7b":
                from workloads.inference.onnxruntime.llama.llama import llama2_infer
                bench_func = llama2_infer

            if model_name == "bert":
                from workloads.inference.onnxruntime.bert.bert_infer import bert_infer
                bench_func = bert_infer

    if framework == "hidet":

        if inference:

            if model_name in torchvision_models:
                from workloads.inference.hidet.vision.infer import vision_infer
                bench_func = vision_infer

    if framework == "tensorrt":

        if inference:

            if model_name == "bert":
                from workloads.inference.tensorrt.bert.bert_infer import bert_infer
                bench_func = bert_infer
            
            if model_name == "stable-diffusion":
                from workloads.inference.tensorrt.stable_diffusion.stable_diffusion import stable_diffusion_infer
                bench_func = stable_diffusion_infer

    if framework == "pytorch":
        
        if inference:

            if model_name in torchvision_models:
                from workloads.inference.pytorch.vision.infer import vision_infer
                bench_func = vision_infer

            if model_name in ["yolov6n", "yolov6m", "yolov6l"]:
                from workloads.inference.pytorch.yolov6.infer import yolov6_infer
                bench_func = yolov6_infer

            if model_name == "llama-2-7b":
                from workloads.inference.pytorch.llama.llama import llama2_infer
                bench_func = llama2_infer

            if model_name == "gpt-neo-2.7B":
                from workloads.inference.pytorch.gpt_neo.gpt_neo import gpt_neo_infer
                bench_func = gpt_neo_infer
            
            if model_name == "bert":
                from workloads.inference.pytorch.bert.bert_infer import bert_infer
                bench_func = bert_infer
            
            if model_name == "stable-diffusion":
                from workloads.inference.pytorch.stable_diffusion.stable_diffusion import stable_diffusion_infer
                bench_func = stable_diffusion_infer

        if training:

            if model_name in ["resnet50"]:
                from workloads.training.pytorch.resnet.train_resnet import train_resnet
                bench_func = train_resnet
            
            if model_name in ["VGG", "EfficientNetB0", "ShuffleNetV2", ]:
                from workloads.training.pytorch.cifar.train_cifar import train_cifar
                bench_func = train_cifar

            if model_name in ["bert"]:
                from workloads.training.pytorch.bert.train_bert import train_bert
                bench_func = train_bert

            if model_name in ["dcgan"]:
                from workloads.training.pytorch.dcgan.train_dcgan import train_dcgan
                bench_func = train_dcgan

            if model_name in ["LSTM"]:
                from workloads.training.pytorch.lstm.train_lstm import train_lstm
                bench_func = train_lstm

            if model_name in ["NeuMF-pre"]:
                from workloads.training.pytorch.ncf.train_ncf import train_ncf
                bench_func = train_ncf
            
            if model_name in ["pointnet"]:
                from workloads.training.pytorch.pointnet.train_pointnet import train_pointnet
                bench_func = train_pointnet

            if model_name in ["transformer"]:
                from workloads.training.pytorch.transformer.train_transformer import train_transformer
                bench_func = train_transformer

            if model_name in ["yolov6n", "yolov6m", "yolov6l"]:
                from workloads.training.pytorch.yolov6.train_yolov6 import train_yolov6
                bench_func = train_yolov6
        
            if model_name in ["pegasus-x-base", "pegasus-large"]:
                from workloads.training.pytorch.pegasus.train_pegasus import train_pegasus
                bench_func = train_pegasus

            if model_name in ["whisper-small", "whisper-large-v3"]:
                from workloads.training.pytorch.whisper.train_whisper import train_whisper
                bench_func = train_whisper
            
            if model_name in ["gpt2-xl", "gpt2-large"]:
                from workloads.training.pytorch.gpt2.train_gpt2 import train_gpt2
                bench_func = train_gpt2

    if not bench_func:
        raise Exception("Cannot find benchmark function")

    return bench_func