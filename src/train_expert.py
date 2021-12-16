# !/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import sys
from dataclasses import dataclass, field

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock


from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import is_main_process
from transformers.utils import check_min_version
from train_utils import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class AdditionalArguments:
    training_file: str = field(
        default='data/xsum-train.csv',
        metadata={"help": "Training data CSV File path."},
    )
    validation_file: str = field(
        default='data/xsum-val.csv',
        metadata={"help": "Validation data CSV File path."},
    )
    model_checkpoint_name: str = field(
        default="facebook/bart-large-xsum",
        metadata={"help": "Pre-trained model name."},
    )
    cache_dir_path: str = field(
        default="pre-trained-models/bart-large-xsum",
        metadata={"help": "Cache directory to load save/load pretrained model."},
    )
    kl_alpha: float = field(
        default=0.5,
        metadata={"help": "KL diveregence loss component multiplier."},
    )
    regularization_model: bool = field(
        default=True,
        metadata={"help": "True for model-based expert, False for reference-based expert."},
    )
    reward_metric: str = field(
        default="dae",
        metadata={"help": "value = dae, ner-p and ner-r for DAE, NER precision and NER recall metrics respectively"},
    )
    dae_model_dir: str = field(
        default="/export/home/code/dae-factuality-datasets/DAE_model_xsum",
        metadata={"help": "DAE Model dir"},
    )
    default_max_length: bool = field(
        default=True,
        metadata={"help": "To use default maximum sample length when training expert. Set to False if you want to "
                          "use reference's length as the maximum sample length"},
    )


def main():

    parser = HfArgumentParser((AdditionalArguments, Seq2SeqTrainingArguments))
    additional_args, training_args = parser.parse_args_into_dataclasses()
    model_checkpoint = additional_args.model_checkpoint_name  # "facebook/bart-large-xsum"
    cache_dir = additional_args.cache_dir_path  # "/export/home/pre-trained-models/bart-large-xsum"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    train_datasets = get_train_dataset(additional_args.training_file)
    val_datasets = get_val_dataset(additional_args.validation_file)

    config = AutoConfig.from_pretrained(
        model_checkpoint,
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        cache_dir=cache_dir
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_checkpoint,
        config=config,
        cache_dir=cache_dir
    )
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_checkpoint,
        config=config,
        cache_dir=cache_dir
    )

    def freeze_params(model):
        for par in model.parameters():
            par.requires_grad = False

    freeze_params(model.get_encoder())

    # model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Tokenizer length {len(tokenizer)}")

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Temporarily set max_target_length for training.
    max_target_length = 128
    padding = False

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=config.max_position_embeddings,
            padding=padding,
            truncation=True
        )

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=max_target_length, padding=padding, truncation=True)

        labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        if "ner" in examples:
            model_inputs["ner"] = tokenizer(examples["ner"])['input_ids'] # 0 and 2 always included
        return model_inputs

    def preprocess_eval_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=config.max_position_embeddings,
            padding=padding,
            truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=max_target_length, padding=padding, truncation=True)

        labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    column_names = train_datasets["train"].column_names

    if training_args.do_train:
        if "train" not in train_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = train_datasets["train"]
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=None,
            remove_columns=column_names,
            load_from_cache_file=True,
        )

    column_names = val_datasets["validation"].column_names
    if training_args.do_eval:
        if "validation" not in val_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = val_datasets["validation"]
        eval_dataset = eval_dataset.map(
            preprocess_eval_function,
            batched=True,
            num_proc=None,
            remove_columns=column_names,
            load_from_cache_file=True,
        )

    # Data collator
    label_pad_token_id = -100
    data_collator = CustomDataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        base_model=base_model,
        kl_alpha=additional_args.kl_alpha,
        dae_model_dir=additional_args.dae_model_dir,
        reward_metric=additional_args.reward_metric,
        regularization_model=additional_args.regularization_model,
        default_max_length=additional_args.default_max_length,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=config.max_length, num_beams=config.num_beams, metric_key_prefix="eval"
        )
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

