#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
from subprocess import call
import sys

from app.preprocess import zhwiki2chars, gen_embeddings


def main(argv):
    basedir = os.path.dirname(os.path.realpath(__file__))
    wiki = basedir + "/data/zhwiki-20161001-pages-articles.xml.bz2"
    corpus = basedir + "/data/zhwiki-corpus.txt"
    embeds = basedir + "/data/zhwiki-embeddings-100.txt"
    download_sh = basedir + "/data/download.sh"

    # download
    call([download_sh])

    # create corpus
    if not os.path.isfile(corpus):
        print("create corpus from %s" % wiki)
        zhwiki2chars(wiki, out_file=corpus)

    # create embeddings
    if not os.path.isfile(embeds):
        print("create embeddings from %s" % corpus)
        gen_embeddings(corpus, out_file=embeds, size=100)


if __name__ == "__main__":
    main(sys.argv)
