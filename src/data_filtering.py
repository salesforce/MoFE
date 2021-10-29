import csv
import torch
import numpy as np
import spacy
import nltk
from tqdm.auto import tqdm
import argparse
from sklearn.utils.extmath import softmax
from pycorenlp import StanfordCoreNLP
from torch.utils.data import DataLoader, SequentialSampler
from dae_train_utils import ElectraDependencyModel, get_single_features
from transformers import (
    ElectraConfig,
    ElectraTokenizer,
)


def evaluate_dae(article_data, summary, tokenizer, model, nlp, device):
    try:
        eval_dataset = get_single_features(summary, article_data, tokenizer, nlp)
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

            outputs = model(**inputs)
            dep_outputs = outputs[1].detach()
            dep_outputs = dep_outputs.squeeze(0)

            dep_outputs = dep_outputs[:num_dependency, :].cpu().numpy()
            num_negative = 0.
            for j, arc in enumerate(arcs[0]):
                arc_text = tokenizer.decode(arc)
                arc_text = arc_text.replace(tokenizer.pad_token, '').strip()
                if arc_text == '':  # for bert
                    break

                softmax_probs = softmax([dep_outputs[j]])
                pred = np.argmax(softmax_probs[0])
                if pred == 0:
                    num_negative += 1.

            return num_negative
    except:
        return 1.


def process_file(fname):
    return [elem.strip() for elem in open(fname, 'r')]


def parse_args():
    parser = argparse.ArgumentParser(description="Data Filtering")
    parser.add_argument("--metric", default="dae", help="dae/ ner-p/ ner-r")
    parser.add_argument("--source_doc", default="data/xsum/train.source", help="Source Articles")
    parser.add_argument("--target_summary", default="data/xsum/train.target", help="Target Summaries")
    parser.add_argument("--dae_model", default="dae-factuality-datasets/DAE_model_cnn_dm/ENT-C_dae", help="DAE Model")
    parser.add_argument("--dump_csv", default="xsum-train-dae.csv", help="output CSV file")
    args = parser.parse_args()
    return args


def dae_filtering(source_docs, target_summs, args, writer, device='cuda'):
    writer.writerow(['source', 'summary'])
    config_class, model_class, tokenizer_class = ElectraConfig, ElectraDependencyModel, ElectraTokenizer
    dae_tokenizer = tokenizer_class.from_pretrained(args.dae_model)
    dae_model = model_class.from_pretrained(args.dae_model)
    dae_model.to('cuda')
    dae_model.eval()
    nlp = StanfordCoreNLP('http://localhost:9000')
    for summary, source in tqdm(zip(target_summs, source_docs)):
        num_neg = evaluate_dae(source, summary, dae_tokenizer, dae_model, nlp, device)
        if num_neg == 0:
            writer.writerow([source.strip(), summary.strip()])


def ner_utils(source, target):
    nlp = spacy.load("en_core_web_lg")
    nltk.download('stopwords')
    sws = set(nltk.corpus.stopwords.words('english'))
    source_docs = nlp.pipe(source,
                           disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    summs = nlp.pipe(target,
                     disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    return source_docs, summs, sws


def ner_precision_filtering(source_text, target_summs, args, writer):
    writer.writerow(['text', 'summary', 'ner'])
    source_docs, summs, sws = ner_utils(source_text, target_summs)
    for id, (src_ent, sm_ent) in enumerate(zip(source_docs, summs)):
        source_ent = set([x.text for x in src_ent if x.ent_type_ != '' and x.text.lower() not in sws])
        summ_ent = set([x.text for x in sm_ent if x.ent_type_ != '' and x.text.lower() not in sws])
        if len(source_ent) > 0 and len(source_ent.intersection(summ_ent)) == len(summ_ent):
            writer.writerow([source_text[id], target_summs[id], ' '.join(source_ent)])


def ner_recall_filtering(source_text, target_summs, args, writer):
    writer.writerow(['text', 'summary', 'ner'])
    source_docs, summs, sws = ner_utils(source_text, target_summs)
    for id, (src_ent, sm_ent)  in enumerate(zip(source_docs, summs)):
        source_ent = set([x.text for x in src_ent if x.ent_type_ != '' and x.text.lower() not in sws])
        summ_ent = set([x.text for x in sm_ent if x.ent_type_ != '' and x.text.lower() not in sws])
        if len(summ_ent) > 0 and len(source_ent.intersection(summ_ent)) == len(summ_ent):
            writer.writerow([source_text[id], target_summs[id], ' '.join(summ_ent)])


if __name__ == "__main__":
    args = parse_args()
    source_docs = process_file(args.source_doc)
    target_summs = process_file(args.target_summary)
    assert len(source_docs) == len(target_summs)

    dump_file = open(args.dump_csv, 'w')
    writer = csv.writer(dump_file)

    if args.metric == 'dae':
        dae_filtering(source_docs, target_summs, args, writer)
    elif args.metric == 'ner-p':
        ner_precision_filtering(source_docs, target_summs, args, writer)
    elif args.metric == 'ner-r':
        ner_recall_filtering(source_docs, target_summs, args, writer)
    else:
        "Metric not defined"
