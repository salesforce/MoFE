import csv
import spacy
import nltk
import argparse


def process_file(fname):
    return [elem.strip() for elem in open(fname, 'r')]


def parse_args():
    parser = argparse.ArgumentParser(description="Data Filtering")
    parser.add_argument("--metric", default="dae", help="dae/ ner-p/ ner-r")
    parser.add_argument("--source_doc", default="data/xsum/train.source", help="Source Articles")
    parser.add_argument("--target_summary", default="data/xsum/train.target", help="Target Summaries")
    parser.add_argument("--dump_csv", default="xsum-train-dae.csv", help="output CSV file")
    args = parser.parse_args()
    return args


def dae(source_docs, target_summs, writer):
    writer.writerow(['source', 'summary'])
    for summary, source in zip(target_summs, source_docs):
        writer.writerow([source.strip(), summary.strip()])


def ner_precision(source_text, target_summs, args, writer):
    writer.writerow(['text', 'summary', 'ner'])
    nlp = spacy.load("en_core_web_lg")
    nltk.download('stopwords')
    sws = set(nltk.corpus.stopwords.words('english'))
    source_docs = nlp.pipe(source_text,
                           disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    for id, src_ent in enumerate(source_docs):
        source_ent = set([x.text for x in src_ent if x.ent_type_ != '' and x.text.lower() not in sws])
        writer.writerow([source_text[id], target_summs[id], ' '.join(source_ent)])


def ner_recall(source_text, target_summs, args, writer):
    writer.writerow(['text', 'summary', 'ner'])
    nlp = spacy.load("en_core_web_lg")
    nltk.download('stopwords')
    sws = set(nltk.corpus.stopwords.words('english'))
    summs = nlp.pipe(target_summs,
                     disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    for id, sm_ent  in enumerate(summs):
        summ_ent = set([x.text for x in sm_ent if x.ent_type_ != '' and x.text.lower() not in sws])
        writer.writerow([source_text[id], target_summs[id], ' '.join(summ_ent)])


if __name__ == "__main__":
    args = parse_args()
    source_docs = process_file(args.source_doc)
    target_summs = process_file(args.target_summary)
    assert len(source_docs) == len(target_summs)

    dump_file = open(args.dump_csv, 'w')
    writer = csv.writer(dump_file)

    if args.metric == 'dae':
        dae(source_docs, target_summs, args, writer)
    elif args.metric == 'ner-p':
        ner_precision(source_docs, target_summs, args, writer)
    elif args.metric == 'ner-r':
        ner_recall(source_docs, target_summs, args, writer)
    else:
        "Metric not defined"
