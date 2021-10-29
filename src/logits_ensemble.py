#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import logging
from typing import Optional, Dict, Any, Tuple
import warnings
from torch.nn import functional as F
from copy import deepcopy

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from transformers.generation_beam_search import BeamSearchScorer, BeamScorer
from transformers.generation_logits_process import LogitsProcessorList, NoRepeatNGramLogitsProcessor, \
    MinLengthLogitsProcessor, ForcedEOSTokenLogitsProcessor
from transformers.generation_stopping_criteria import StoppingCriteriaList, MaxLengthCriteria, \
    validate_stopping_criteria
from transformers.file_utils import ModelOutput


logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def parse_args():
    parser = argparse.ArgumentParser(description="MoFE: Logits Ensembling")
    parser.add_argument(
        "--baseline_checkpoint_name",
        type=str,
        default="facebook/bart-large-xsum",
        help="Pre-trained model name.",
    )
    parser.add_argument(
        "--baseline_cache_dir_path",
        type=str,
        default="pre-trained-models/bart-large-xsum",
        help="Cahce directory to load save/load pretrained model.",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default="data/xsum-val.csv",
        help="Evaluation File.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/predicts/output.txt",
        help="Output File.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default='',
        help='Coefficient for Experts separated by comma e.g. "Expert alpha 1,Expert alpha 2,..."',
    )
    parser.add_argument(
        "--experts",
        type=str,
        default='',
        help='Paths for Experts separated by comma e.g. "Expert path 1,Expert path 2,..."',
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=6,
        help="Number of beams to use for evaluation. This argument will be "
             "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=6,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    args = parser.parse_args()

    return args


def get_dataset(fname):
    return load_dataset('csv',
                        data_files={
                            'validation': [fname]
                        }
                        )


class NewGenerator:
    def __init__(
            self,
            base_model,
            config,
            experts_models,
            alpha
    ):
        self.base_model = base_model
        self.config = config
        self.experts = experts_models
        self.alpha = alpha
        self.baseline_alpha = 1.-sum(alpha)

    @torch.no_grad()
    def generate(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
            **model_kwargs,
    ) -> torch.LongTensor:
        # set init values
        max_length = max_length if max_length is not None else self.config.max_length
        num_beams = num_beams if num_beams is not None else self.config.num_beams

        pad_token_id = self.config.pad_token_id
        eos_token_id = self.config.eos_token_id
        experts_model_kwargs = []
        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            base_model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs,
                                                                                    self.base_model)
            for model in self.experts:
                experts_model_kwargs.append(
                    self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs, model))
            # Prafulla Call for all experts

            input_ids = self._prepare_decoder_input_ids_for_generation(
                input_ids,
                decoder_start_token_id=self.config.decoder_start_token_id
            )

        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # get distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            max_length=max_length,
            eos_token_id=eos_token_id,
            no_repeat_ngram_size=3,
        )

        stopping_criteria = self._get_stopping_criteria(max_length=max_length)

        batch_size = input_ids.shape[0]
        length_penalty = self.config.length_penalty
        early_stopping = self.config.early_stopping

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            # num_beam_hyps_to_keep=num_beams,
            device=self.base_model.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
        )

        for index in range(len(experts_model_kwargs)):
            experts_model_kwargs[index] = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                is_expert=True,
                **experts_model_kwargs[index]
            )[1]
        input_ids, base_model_kwargs = self._expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **base_model_kwargs
        )

        return self.beam_search(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            experts_model_kwargs=experts_model_kwargs,
            **base_model_kwargs,
        )

    def _get_stopping_criteria(
            self,
            max_length: Optional[int],
    ) -> StoppingCriteriaList:
        stopping_criteria = StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
        return stopping_criteria

    def _prepare_decoder_input_ids_for_generation(
            self, input_ids: torch.LongTensor, decoder_start_token_id: int = None
    ) -> torch.LongTensor:
        # decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        decoder_input_ids = (
                torch.ones((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device) * decoder_start_token_id
        )
        return decoder_input_ids

    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            attention_mask: torch.LongTensor = None,
            encoder_outputs: ModelOutput = None,
            is_expert=False,
            **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        if not is_expert:
            input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, input_ids: torch.LongTensor, mkwargs, model
    ) -> Dict[str, Any]:
        model_kwargs = deepcopy(mkwargs)
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = model.get_encoder()
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
            }
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
        return model_kwargs

    @staticmethod
    def _update_model_kwargs_for_generation(
            outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return model_kwargs

    def _get_logits_processor(
            self,
            max_length: int,
            eos_token_id: int,
            no_repeat_ngram_size: int,
    ) -> LogitsProcessorList:
        """
        This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
        """
        processors = LogitsProcessorList()

        # init warp parameters
        no_repeat_ngram_size = (
            self.config.no_repeat_ngram_size
        )
        min_length = self.config.min_length
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        forced_eos_token_id = (
            self.config.forced_eos_token_id
        )

        # instantiate processors list
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))

        return processors

    def beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            experts_model_kwargs=[],
            **base_model_kwargs,
    ) -> torch.LongTensor:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        assert (
                num_beams * batch_size == batch_beam_size
        ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:

            model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **base_model_kwargs)

            outputs = self.base_model(
                **model_inputs,
                return_dict=True,
            )
            summ_next_token_logits = outputs.logits[:, -1, :]
            summ_next_token_logits = self.base_model.adjust_logits_during_generation(summ_next_token_logits,
                                                                                     cur_len=cur_len)

            experts_outputs = []
            expert_logits = []
            for index in range(len(self.experts)):
                model_inputs = self.experts[index].prepare_inputs_for_generation(input_ids,
                                                                                       **experts_model_kwargs[
                                                                                           index])
                experts_outputs.append(self.experts[index](
                    **model_inputs,
                    return_dict=True, ))
                expert_next_token_logits = experts_outputs[index].logits[:, -1, :]
                # expert also includes <mask> token (id: 50264)

                expert_logits.append(
                    self.experts[index].adjust_logits_during_generation(expert_next_token_logits,
                                                                              cur_len=cur_len))
            corrected_logits = self.baseline_alpha * summ_next_token_logits

            for index, hal_logits in enumerate(expert_logits):
                corrected_logits += (self.alpha[index]) * hal_logits  # [:, :-1]

            next_token_scores = F.log_softmax(corrected_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            base_model_kwargs = self._update_model_kwargs_for_generation(
                outputs, base_model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            for index in range(len(experts_model_kwargs)):
                experts_model_kwargs[index] = self._update_model_kwargs_for_generation(
                    experts_outputs[index], experts_model_kwargs[index],
                    is_encoder_decoder=self.config.is_encoder_decoder
                )

            if base_model_kwargs["past"] is not None:
                base_model_kwargs["past"] = self.base_model._reorder_cache(base_model_kwargs["past"], beam_idx)
            for index in range(len(experts_model_kwargs)):
                if experts_model_kwargs[index]["past"] is not None:
                    experts_model_kwargs[index]["past"] = self.experts[index]._reorder_cache(
                        experts_model_kwargs[index]["past"], beam_idx
                    )
                    # Prafulla, probably call for experts

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, None):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        return sequence_outputs["sequences"]


def main():
    args = parse_args()
    base_model_checkpoint = args.baseline_checkpoint_name
    base_cache_dir = args.baseline_cache_dir_path
    experts_model_dirs = args.experts.split(',')
    experts_model_alphas = [float(elem) for elem in args.alphas.split(',')]
    print(experts_model_alphas, experts_model_dirs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    common_config = AutoConfig.from_pretrained(base_model_checkpoint, cache_dir=base_cache_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    raw_datasets = get_dataset(args.eval_data_path)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_checkpoint, cache_dir=base_cache_dir)
    base_model = base_model.to(device)

    common_tokenizer = AutoTokenizer.from_pretrained(base_model_checkpoint, cache_dir=base_cache_dir)  
    experts_models = []
    for exp_path in experts_model_dirs:
        expert_model = AutoModelForSeq2SeqLM.from_pretrained(exp_path)
        expert_model = expert_model.to(device)
        experts_models.append(expert_model)

    padding = False

    generator = NewGenerator(base_model, config=common_config, experts_models=experts_models,
                             alpha=experts_model_alphas)

    def preprocess_function(examples):
        model_inputs = common_tokenizer(examples["text"], max_length=common_config.max_position_embeddings, padding=padding,
                                      truncation=True)

        # Setup the tokenizer for targets
        with common_tokenizer.as_target_tokenizer():
            labels = common_tokenizer(examples["summary"], max_length=common_config.max_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != common_tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    column_names = raw_datasets["validation"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    eval_dataset = processed_datasets["validation"]

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        common_tokenizer,
        label_pad_token_id=label_pad_token_id
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Metric
    metric = load_metric("rouge")

    logger.info("***** Running Evaluation *****")

    gen_kwargs = {
        "max_length": common_config.max_length,
        "num_beams": args.num_beams,
    }
    dump = open(args.output_path, 'w')

    for step, batch in tqdm(enumerate(eval_dataloader)):
        with torch.no_grad():
            for named_args in batch:
                batch[named_args] = batch[named_args].to(device)
            generated_tokens = generator.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = generated_tokens.cpu().numpy()

            labels = batch['labels'].cpu().numpy()

            labels = np.where(labels != -100, labels, common_tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = common_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = common_tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            for pred in decoded_preds:
                dump.write(' '.join(pred.strip().split('\n')) + '\n')

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    result = metric.compute(use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    result = {k: round(v, 4) for k, v in result.items()}

    logger.info(result)


if __name__ == "__main__":
    main()
