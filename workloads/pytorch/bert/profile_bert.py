from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import os
import pickle
import time
from torch.utils.data import DataLoader, RandomSampler

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    squad_convert_examples_to_features,
)

from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor

from models.util import get_benchmark_str

def load_and_cache_examples(tokenizer, version_2_with_negative, data_dir=None,
                            train_file="./data/SQUAD_DIR/train-v1.1.json",
                            max_seq_length=384, doc_stride=128, max_query_length=64, threads=4,
                            evaluate=False, output_examples=False):
    processor = SquadV2Processor() if version_2_with_negative else SquadV1Processor()
    examples = processor.get_train_examples(data_dir, filename=train_file)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=threads,
    )

    if output_examples:
        return dataset, examples, features
    return dataset


def benchmark_bert(model_name, batch_size, mixed_precision, warmup_epoch, total_time,
                   warm_signal=None, start_signal=None, bench_id="", result_dict=None, total_iters=None, model_type='bert', config_name="",
                   model_name_or_path='bert-base-uncased', cache_dir="./data", tokenizer_name="",
                   do_lower_case=True, weight_decay=0.0, learning_rate=5e-5, adam_epsilon=1e-8,
                   version_2_with_negative=True, lang_id=0):

    cudnn.benchmark = True 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_type = model_type.lower()
    config = AutoConfig.from_pretrained(
        config_name if config_name else model_name_or_path,
        cache_dir=cache_dir if cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        do_lower_case=do_lower_case,
        cache_dir=cache_dir if cache_dir else None,
        use_fast=False
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir if cache_dir else None,
    )

    model.to(device)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    cached_dataset_name = f"{cache_dir}/{model_name_or_path}_dataset.pkl"
    if os.path.exists(cached_dataset_name):
        with open(cached_dataset_name, 'rb') as f:
            train_dataset = pickle.load(f)
    else:
        train_dataset = load_and_cache_examples(tokenizer, version_2_with_negative)
        with open(cached_dataset_name, 'wb') as f:
            pickle.dump(train_dataset, f)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    # Train
    def benchmark_step():
        t_start = time.time()
        iter_num = 0
        iter_warm = 0
        warm = False
        exit_flag = False
        model.train()
        while True:
            for step, batch in enumerate(train_dataloader):
                if iter_num == warmup_epoch - 1:
                    warm = True
                    if warm_signal is not None:
                        warm_signal.value = 1
                    if start_signal is not None:
                        while start_signal.value != 1:
                            time.sleep(0.1)
                    print("Measurement starts...")
                    torch.cuda.cudart().cudaProfilerStart()
                    t_warmend = time.time()
                if (warm and (time.time() - t_warmend >= total_time)) or (total_iters is not None and iter_warm == total_iters):
                    t_end = time.time()
                    t_pass = t_end - t_warmend
                    exit_flag = True
                    torch.cuda.cudart().cudaProfilerStop()
                    break
                optimizer.zero_grad()
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }

                if model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                    del inputs["token_type_ids"]

                if model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                    if version_2_with_negative:
                        inputs.update({"is_impossible": batch[7]})
                    if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                        inputs.update(
                            {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * lang_id).to(device)}
                        )
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(**inputs)
                        loss = outputs[0]
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(**inputs)
                    loss = outputs[0]
                    loss.backward()
                    optimizer.step()
                iter_num += 1
                if warm:
                    iter_warm += 1
            if exit_flag:
                break
        return t_pass, iter_warm

    benchmark_id = get_benchmark_str(model_name, batch_size, mixed_precision, bench_id)

    try:
        print(f'==> Training {model_name} model with {batch_size} batch size, {mixed_precision} mp..')
        t_pass, iter_warm = benchmark_step()
        print(f"Time: {t_pass}, Iterations: {iter_warm}")
    
        if result_dict is not None:
            result_dict[benchmark_id] = {
                "Time": t_pass,
                "Iterations": iter_warm
            }
    except Exception as e:
        if result_dict is not None:
            result_dict[benchmark_id] = {
                "Error": str(e)
            }
        if warm_signal is not None:
            warm_signal.value = 1
        else:
            raise e