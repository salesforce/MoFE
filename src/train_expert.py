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
from typing import Optional, Union, List, Any, Dict, Tuple

import torch
import torch.nn.functional as F
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers.integrations import is_deepspeed_zero3_enabled
from filelock import FileLock

from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from reward_utils import DAEReward, NERRReward, NERPReward

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

@dataclass
class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, all_features):
        features = [{key: all_fs[key] for key in all_fs if key != 'ner'} for all_fs in all_features]
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        if 'ner' in all_features[0]:
            features['ner'] = [all_features[i]["ner"] for i in range(len(all_features))]

        return features


def get_train_dataset(fname):
    return load_dataset('csv',
                        data_files={
                            'train': [fname]
                        }
                        )


def get_val_dataset(fname):
    return load_dataset('csv',
                        data_files={
                            'validation': [fname]
                        }
                        )


class CustomSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(
            self,
            model,
            args,
            train_dataset,
            eval_dataset,
            tokenizer,
            data_collator,
            compute_metrics,
            base_model,
            kl_alpha,
            dae_model_dir,
            reward_metric='dae',
            regularization_model=True,
            default_max_length=True,
    ):
        super(Seq2SeqTrainer, self).__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        if regularization_model:
            self.base_model = base_model

        if reward_metric == 'dae':
            self.reward_model = DAEReward(dae_model_dir, device='cuda')
        elif reward_metric == 'ner-r':
            self.reward_model = NERRReward()
        elif reward_metric == 'ner-p':
            self.reward_model = NERPReward()
        else:
            raise NotImplementedError(f'Reward type not defined, select one of the dae, ner-p and ner-r')

        self.kl_alpha = kl_alpha
        self.reward_metric = reward_metric
        self.regularization_model = regularization_model
        self.default_max_length = default_max_length

    def compute_loss(self, model, inputs, return_outputs=False):
        # TODO: Reverify decoder_input_ids, but in my best understanding, generate() method starts with
        #  start_token_id and I don't need to worry about this.
        model_ips = {elem: inputs[elem] for elem in inputs if elem != 'ner'}
        if not self.regularization_model:
            outputs = self.model(**model_ips)
            loss = self.kl_alpha * outputs["loss"].mean()
        if self.default_max_length:
            max_sample_length = self.model.config.max_length
        else:
            max_sample_length = model_ips["labels"].size(1)

        _ = model_ips.pop("decoder_input_ids")  # Just for safety, remove "decoder_input_ids"
        gen_kwargs = {
            "max_length": max_sample_length,
            "do_sample": True,
            "min_length": self.model.config.min_length,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }
        sampled_text = self.model.generate(
            model_ips["input_ids"],
            attention_mask=model_ips["attention_mask"],
            **gen_kwargs,
        )  # Don't need to worry about decoder_input_ids, generate() doesn't use that and starts with start_token_id
        gen_kwargs = {
            "max_length": max_sample_length,
            "do_sample": False,
            "min_length":  self.model.config.min_length,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }
        with torch.no_grad():
            argmax_text = self.model.generate(
                model_ips["input_ids"],
                attention_mask=model_ips["attention_mask"],
                **gen_kwargs,
            )  # Don't need to worry about decoder_input_ids, generate() doesn't use that and starts with start_token_id

        if self.reward_metric == 'dae':
            decoded_sample = self.tokenizer.batch_decode(sampled_text, skip_special_tokens=True)
            decoded_argmax = self.tokenizer.batch_decode(argmax_text, skip_special_tokens=True)
            decoded_source = self.tokenizer.batch_decode(model_ips["input_ids"], skip_special_tokens=True)

            reward = self.reward_model.calculate_reward(decoded_source,
                                                        decoded_argmax,
                                                        decoded_sample,
                                                        sampled_text.get_device())
        else:
            actor_predicts = [set(x) for x in sampled_text.tolist()]  # set(sampled_text.tolist())
            argmax_predicts = [set(x) for x in argmax_text.tolist()]  # set(argmax_text.tolist())
            source = [set(x) for x in inputs['ner']]
            reward = self.reward_model.calculate_reward(actor_predicts,
                                                        argmax_predicts,
                                                        source,
                                                        sampled_text.get_device())

        sampled_text[sampled_text == self.tokenizer.pad_token_id] = -100
        model_ips["labels"] = sampled_text
        model_ips["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(labels=sampled_text)
        outputs = model(**model_ips)
        if self.regularization_model:
            self.base_model.to(sampled_text.get_device())
            with torch.no_grad():
                base_outputs = self.base_model(**model_ips)  # KL
            loss_fct = torch.nn.KLDivLoss(reduce=False)
            loss = loss_fct(F.log_softmax(outputs.logits.view(-1, self.model.config.vocab_size)),
                            F.softmax(base_outputs.logits.view(-1, self.model.config.vocab_size))).sum(dim=-1)
            loss = self.kl_alpha * loss.mean()

        reward = reward.view(-1, 1)
        reward = reward.repeat(1, sampled_text.size(1))
        reward = reward.to(outputs.logits.get_device())

        loss_fct = torch.nn.CrossEntropyLoss(reduce=False)
        loss += (1. - self.kl_alpha) * torch.mean(
            loss_fct(outputs.logits.view(-1, self.model.config.vocab_size), sampled_text.view(-1)) * reward.view(-1))

        return (loss, outputs) if return_outputs else loss


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

