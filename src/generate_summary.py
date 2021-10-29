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


import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import argparse

import logging
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the summarization model")
    parser.add_argument(
        "--eval_data_path",
        type=str, default=None,
        help="",
    )
    parser.add_argument(
        "--dump_file",
        type=str, default=None,
        help="",
    )
    parser.add_argument(
        "--model_path",
        type=str, default=None,
        help="",
    )
    parser.add_argument(
        "--batch_size",
        type=int, default=8,
        help="",
    )
    args = parser.parse_args()
    return args


def get_dataset(fname):
    return load_dataset('csv',
                        data_files={
                            'test': [fname]
                        }
                        )

def main():
    args = parse_args()

    dump = open(args.dump_file, 'w')
    model_dir = args.model_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = AutoConfig.from_pretrained(model_dir)

    raw_datasets = get_dataset(args.test_file)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    model = model.to(device)
    print(config.max_position_embeddings)
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=config.max_position_embeddings,
            padding=False,
            truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], padding=False)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Metric
    metric = load_metric("rouge")

    logger.info("***** Running Evaluation *****")
    logger.info(f"Model Path = {args.model_path}")
    logger.info(f"Output Path: {args.dump_file}")

    column_names = raw_datasets["test"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    test_dataset = processed_datasets["test"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model
    )

    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    gen_kwargs = {
        "max_length": config.max_length,
        "num_beams": config.num_beams,
    }

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    for step,batch in tqdm(enumerate(test_dataloader)):

        with torch.no_grad():
            for named_args in batch:
                batch[named_args] = batch[named_args].to(device)
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = generated_tokens.cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

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


