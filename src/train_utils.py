"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_metric

import transformers
from transformers.integrations import is_deepspeed_zero3_enabled

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
