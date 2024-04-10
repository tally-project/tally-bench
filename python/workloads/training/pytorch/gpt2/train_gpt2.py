import os
import pickle
import time

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from datasets import load_dataset, Dataset
from workloads.common.hf_callback import BenchCallback

# Training
def train_gpt2(model_name, batch_size, amp, warmup_iters, total_time, total_iters=None,
                result_dict=None, signal=False, pipe=None):
    
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))

    datasets = load_dataset("bookcorpus")
    datasets["train"] = datasets["train"].select(range(1000000))
    column_names = datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True
    )

    block_size = 512
  
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        load_from_cache_file=True
    )
    train_dataset = lm_datasets["train"]

    bench_callback = BenchCallback(warmup_iters, total_time, total_iters, result_dict, signal, pipe)
    training_args = TrainingArguments(
        output_dir="./logs/gpt2-log",
        fp16=amp,
        num_train_epochs=1,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=batch_size,
        save_steps=4000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        callbacks=[bench_callback]
    )

    trainer.train()