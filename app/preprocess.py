#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def zhwiki2chars(in_file, out_file):
    reg = re.compile(r'^[a-zA-Z]+$')

    def _isalpha(string):
        return reg.match(string) is not None

    i = 0
    out = open(out_file, 'w')
    wiki = WikiCorpus(in_file, lemmatize=False, dictionary={})
    for article in wiki.get_texts():
        tokens = []
        for token in article:
            token = token.decode("utf-8").strip()
            if _isalpha(token):
                continue
            tokens.append(" ".join(token))  # divided by character
        out.write(" ".join(tokens) + "\n")
        i += 1
        if i % 10000 == 0:
            print("process %d articles" % i)
    out.close()


def gen_embeddings(in_file, out_file, size=100):
    corpus = LineSentence(in_file)
    model = Word2Vec(
        sentences=corpus, size=size, alpha=0.025, window=5, min_count=5,
        max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
        sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
        trim_rule=None, sorted_vocab=1
    )
    model.save_word2vec_format(out_file, binary=False)
