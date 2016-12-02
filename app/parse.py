#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import serializers
import numpy as np

from model import BLSTM, BLSTMCRF
from libs.base import App as AbstractApp, Logger as Log
from libs.tools import Preprocessor


_use_crf = False


def parse(model_file, embed_file):

    # Load files
    Log.i('initialize preprocessor with %s' % embed_file)
    processor = Preprocessor(embed_file)

    Log.v('')
    Log.v("initialize ...")
    Log.v('')

    with np.load(model_file) as f:
        embeddings = np.zeros(f['embed/W'].shape, dtype=np.float32)

    # Set up a neural network
    cls = BLSTMCRF if _use_crf else BLSTM
    model = cls(
        embeddings=embeddings,
        n_labels=4,
        dropout=0.2,
        train=False,
    )
    Log.i("loading a model from %s ..." % model_file)
    serializers.load_npz(model_file, model)

    LABELS = ['B', 'M', 'E', 'S']

    def _process(raw_text):
        if not raw_text:
            return
        xs = [processor.transform_one([c for c in raw_text])]
        ys = model.parse(xs)
        labels = [LABELS[y] for y in ys[0]]
        print(' '.join(labels))
        seq = []
        for c, label in zip(raw_text, labels):
            seq.append(c)
            if label == 'E' or label == 'S':
                seq.append(' ')
        print(''.join(seq))
        print('-')

    print("Input a Chinese sentence! (use 'q' to exit)")
    while True:
        x = input()
        if x == 'q':
            break
        _process(x)


class App(AbstractApp):

    def _initialize(self):
        self._logdir = self._basedir + '/../logs'

    def main(self):
        datadir = self._basedir + '/../data'

        Log.i("*** [START] ***")
        parse(
            model_file=self._basedir + "/../output/cws.model",
            embed_file=datadir + "/zhwiki-embeddings-100.txt"
        )
        Log.i("*** [DONE] ***")


if __name__ == "__main__":
    App.exec()
