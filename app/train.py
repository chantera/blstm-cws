#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from chainer import cuda, optimizers, serializers
from chainer.optimizer import WeightDecay
from progressbar import ProgressBar

from model import BLSTM, BLSTMCRF
from libs.base import App as AbstractApp, Logger as Log
from libs.tools import Preprocessor
from data import CorpusReader


_use_crf = False


def train(
        train_file,
        test_file,
        embed_file,
        n_epoch=20,
        batch_size=20,
        gpu=-1,
        save=None):

    # Load files
    Log.i('initialize preprocessor with %s' % embed_file)
    processor = Preprocessor(embed_file)
    reader = CorpusReader(processor)
    Log.i('load train dataset from %s' % str(train_file))
    train_dataset = reader.load(train_file, train=True)
    Log.i('load test dataset from %s' % str(test_file))
    test_dataset = reader.load(test_file, train=False)

    hparams = {
        'dropout_ratio': 0.2,
        'adagrad_lr': 0.2,
        'weight_decay': 0.0001,
    }

    Log.v('')
    Log.v("initialize ...")
    Log.v('--------------------------------')
    Log.i('# Minibatch-size: %d' % batch_size)
    Log.i('# epoch: %d' % n_epoch)
    Log.i('# gpu: %d' % gpu)
    Log.i('# hyper-parameters: %s' % str(hparams))
    Log.v('--------------------------------')
    Log.v('')

    # Set up a neural network
    cls = BLSTMCRF if _use_crf else BLSTM
    model = cls(
        embeddings=processor.embeddings,
        n_labels=4,
        dropout=hparams['dropout_ratio'],
        train=True,
    )
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()
    eval_model = model.copy()
    eval_model.train = False

    # Setup an optimizer
    optimizer = optimizers.AdaGrad(lr=hparams['adagrad_lr'])
    optimizer.setup(model)
    optimizer.add_hook(WeightDecay(hparams['weight_decay']))

    def _update(optimizer, loss):
        optimizer.target.zerograds()
        loss.backward()
        optimizer.update()

    def _process(dataset, model):
        size = len(dataset)
        batch_count = 0
        loss = 0.0
        accuracy = 0.0

        p = ProgressBar(min_value=0, max_value=size, fd=sys.stderr).start()
        for i, (xs, ys) in enumerate(dataset.batch(batch_size, colwise=True, shuffle=model.train)):
            p.update((batch_size * i) + 1)
            batch_count += 1
            batch_loss, batch_accuracy = model(xs, ys)
            loss += batch_loss.data
            accuracy += batch_accuracy
            if model.train:
                _update(optimizer, batch_loss)

        p.finish()
        Log.i("[%s] epoch %d - #samples: %d, loss: %f, accuracy: %f"
              % ('training' if model.train else 'evaluation', epoch + 1, size,
                 loss / batch_count, accuracy / batch_count))

    for epoch in range(n_epoch):
        _process(train_dataset, model)
        _process(test_dataset, eval_model)
        Log.v('-')

    if save is not None:
        Log.i("saving the model to %s ..." % save)
        serializers.save_npz(save, model)


class App(AbstractApp):

    def _initialize(self):
        self._logdir = self._basedir + '/../logs'
        self._def_arg('--batchsize', '-b', type=int, default=20,
                      help='Number of examples in each mini-batch')
        self._def_arg('--epoch', '-e', type=int, default=20,
                      help='Number of sweeps over the dataset to train')
        self._def_arg('--gpu', '-g', type=int, default=-1,
                      help='GPU ID (negative value indicates CPU)')
        self._def_arg('--save', action='store_true', default=False,
                      help='Save the NN model')

    def main(self):
        corpus = "pku"
        datadir = self._basedir + '/../data'

        Log.i("*** [START] ***")
        train(
            train_file=datadir + "/icwb2-data/training/%s_training.utf8" % corpus,
            test_file=datadir + "/icwb2-data/gold/%s_test_gold.utf8" % corpus,
            embed_file=datadir + "/zhwiki-embeddings-100.txt",
            n_epoch=self._args.epoch,
            batch_size=self._args.batchsize,
            gpu=self._args.gpu,
            save=self._basedir + "/../output/cws.model" if self._args.save else None
        )
        Log.i("*** [DONE] ***")


if __name__ == "__main__":
    App.exec()
