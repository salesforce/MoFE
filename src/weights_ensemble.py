"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import collections
import shutil
import os
import argparse
from transformers import (
    AutoModelForSeq2SeqLM,
)
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="MoFE: Weights Ensembling")
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
        "--mofe_weight_ensemble_path",
        type=str,
        default="mofe-weight-model",
        help="Cahce directory to load save/load pretrained model.",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.baseline_checkpoint_name,
                                                       cache_dir=args.baseline_cache_dir_path)
    experts_models = []
    for expert_path in args.experts.split(','):
        experts_models.append(
            torch.load(os.path.join(expert_path, "pytorch_model.bin"), map_location=torch.device('cpu')))
    experts_alphas = [float(elem) for elem in args.alphas.split(',')]

    assert len(experts_alphas) == len(experts_models)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_params = base_model.state_dict()
    base_params_keys = list(base_params.keys())
    for expert in experts_models:
        assert set(expert.keys()) == set(base_params_keys)

    averaged_params = collections.OrderedDict()
    base_alpha = 1. - sum(experts_alphas)
    print(base_alpha, experts_alphas)

    for key in base_params_keys:
        averaged_params[key] = base_alpha * base_params[key]
        for ind in range(len(experts_alphas)):
            averaged_params[key] += experts_alphas[ind] * experts_models[ind][key]

    expert_path = args.experts.split(',')[0]
    files = os.listdir(expert_path)
    os.makedirs(args.mofe_weight_ensemble_path, exist_ok=True)
    for fname in files:
        if fname.endswith(('.txt', '.bin', '.json')):
            shutil.copy(os.path.join(expert_path, fname), args.mofe_weight_ensemble_path)
    with open(os.path.join(args.mofe_weight_ensemble_path, 'pytorch_model.bin'), 'wb') as fp:
        torch.save(averaged_params, fp)
