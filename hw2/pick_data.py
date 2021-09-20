#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
from simalign import simalign
from tqdm import tqdm


optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

# sys.stderr.write("Loading Sim Aligner...")
# model = simalign.SentenceAligner(model="bert")

sys.stderr.write("Loading Data...")

f_corpus = []
e_corpus = []

for (f, e) in tqdm(zip(open(f_data), open(e_data))):
    f_corpus.append(f)
    e_corpus.append(e)

with open('split_data/fold_1/hansards.e', 'w', encoding='utf-8') as f_e:
    with open('split_data/fold_1/hansards.f', 'w', encoding='utf-8') as f_f:
        for e_instance in e_corpus[30000: 47500]:
            f_e.write(e_instance)
        for f_instance in f_corpus[30000: 47500]:
            f_f.write(f_instance)

with open('split_data/fold_2/hansards.e', 'w', encoding='utf-8') as f_e:
    with open('split_data/fold_2/hansards.f', 'w', encoding='utf-8') as f_f:
        for e_instance in e_corpus[47500: 65000]:
            f_e.write(e_instance)
        for f_instance in f_corpus[47500: 65000]:
            f_f.write(f_instance)


with open('split_data/fold_3/hansards.e', 'w', encoding='utf-8') as f_e:
    with open('split_data/fold_3/hansards.f', 'w', encoding='utf-8') as f_f:
        for e_instance in e_corpus[65000: 82500]:
            f_e.write(e_instance)
        for f_instance in f_corpus[65000: 82500]:
            f_f.write(f_instance)


with open('split_data/fold_4/hansards.e', 'w', encoding='utf-8') as f_e:
    with open('split_data/fold_4/hansards.f', 'w', encoding='utf-8') as f_f:
        for e_instance in e_corpus[82500:]:
            f_e.write(e_instance)
        for f_instance in f_corpus[82500:]:
            f_f.write(f_instance)







