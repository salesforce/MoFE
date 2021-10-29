from torch.utils.data import DataLoader, SequentialSampler
import torch
import numpy as np
from sklearn.utils.extmath import softmax
from dae_train_utils import get_single_features, ElectraDependencyModel
from pycorenlp import StanfordCoreNLP
from transformers import (
    ElectraConfig,
    ElectraTokenizer,
)

class DAEReward:

    def __init__(self, dae_model_dir, device='cuda'):
        config_class, model_class, tokenizer_class = ElectraConfig, ElectraDependencyModel, ElectraTokenizer
        self.dae_tokenizer = tokenizer_class.from_pretrained(dae_model_dir)
        self.dae_model = model_class.from_pretrained(dae_model_dir)
        self.dae_model.to(device)
        self.dae_model.eval()
        self.nlp = StanfordCoreNLP('http://localhost:9000')

    def calculate_reward_util(self, article_datas, summaries, device):
        rewards = []
        for summary, article_data in zip(summaries, article_datas):
            try:
                eval_dataset = get_single_features(summary, article_data, self.dae_tokenizer, self.nlp)
                eval_sampler = SequentialSampler(eval_dataset)
                eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)
                batch = [t for t in eval_dataloader][0]
                batch = tuple(t.to(device) for t in batch)

                with torch.no_grad():
                    input_ids, attention, child, head = batch[0], batch[1], batch[2], batch[3]
                    mask_entail, mask_cont, num_dependency, arcs = batch[4], batch[5], batch[6], batch[7]
                    sent_labels = batch[8]

                    inputs = {'input_ids': input_ids, 'attention': attention, 'child': child,
                              'head': head, 'mask_entail': mask_entail, 'mask_cont': mask_cont,
                              'num_dependency': num_dependency, 'sent_label': sent_labels, 'device': device}

                    outputs = self.dae_model(**inputs)
                    dep_outputs = outputs[1].detach()
                    dep_outputs = dep_outputs.squeeze(0)

                    dep_outputs = dep_outputs[:num_dependency, :].cpu().numpy()
                    num_positive = 0.
                    for j, arc in enumerate(arcs[0]):
                        arc_text = self.dae_tokenizer.decode(arc)
                        arc_text = arc_text.replace(self.dae_tokenizer.pad_token, '').strip()
                        if arc_text == '':  # for bert
                            break

                        softmax_probs = softmax([dep_outputs[j]])
                        pred = np.argmax(softmax_probs[0])
                        if pred == 1:
                            num_positive += 1.
                    rewards.append(num_positive / num_dependency)
            except:
                rewards.append(0.)
        return torch.tensor(rewards, device=device)

    def calculate_reward(self, decoded_source, decoded_argmax, decoded_sample, device):
        return self.calculate_reward_util(decoded_source, decoded_sample, device) - \
               self.calculate_reward_util(decoded_source, decoded_argmax, device)


class NERPReward:

    def calculate_reward(self, actor_predicts, argmax_predicts, source, device):
        precision_sample = [float(len(actor_predicts[idx].intersection(x))) / (len(actor_predicts[idx]) + 0.01)
                            for idx, x in enumerate(source)]
        precision_argmax = [float(len(argmax_predicts[idx].intersection(x))) / (len(argmax_predicts[idx]) + 0.01)
                            for idx, x in enumerate(source)]
        reward = [(r1 - r2) for (r1, r2) in zip(precision_sample, precision_argmax)]
        return torch.tensor([reward], device=device)


class NERRReward:

    def calculate_reward(self, actor_predicts, argmax_predicts, source, device):
        recall_sample = [float(len(actor_predicts[idx].intersection(x)))/(len(x)+0.001)
                               for idx,x in enumerate(source)]
        recall_argmax = [float(len(argmax_predicts[idx].intersection(x)))/(len(x)+0.001)
                               for idx,x in enumerate(source)]
        reward = [(r1-r2) for (r1, r2) in zip(recall_sample, recall_argmax)]
        return torch.tensor([reward], device=device)
