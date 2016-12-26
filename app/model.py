#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Chain, cuda, Variable
import chainer.functions as F
import chainer.links as L
from chainer.links.connection.n_step_lstm import argsort_list_descent, permutate_list
import numpy as np


class LSTM(L.NStepLSTM):

    def __init__(self, in_size, out_size, dropout=0.5, use_cudnn=True):
        n_layers = 1
        super(LSTM, self).__init__(n_layers, in_size, out_size, dropout, use_cudnn)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(LSTM, self).to_cpu()
        if self.cx is not None:
            self.cx.to_cpu()
        if self.hx is not None:
            self.hx.to_cpu()

    def to_gpu(self, device=None):
        super(LSTM, self).to_gpu(device)
        if self.cx is not None:
            self.cx.to_gpu(device)
        if self.hx is not None:
            self.hx.to_gpu(device)

    def set_state(self, cx, hx):
        assert isinstance(cx, Variable)
        assert isinstance(hx, Variable)
        cx_ = cx
        hx_ = hx
        if self.xp == np:
            cx_.to_cpu()
            hx_.to_cpu()
        else:
            cx_.to_gpu()
            hx_.to_gpu()
        self.cx = cx_
        self.hx = hx_

    def reset_state(self):
        self.cx = self.hx = None

    def __call__(self, xs, train=True):
        batch = len(xs)
        if self.hx is None:
            xp = self.xp
            self.hx = Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype),
                volatile='auto')
        if self.cx is None:
            xp = self.xp
            self.cx = Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype),
                volatile='auto')

        hy, cy, ys = super(LSTM, self).__call__(self.hx, self.cx, xs, train)
        self.hx, self.cx = hy, cy
        return ys


class CRF(L.CRF1d):

    def __init__(self, n_label):
        super(CRF, self).__init__(n_label)

    def __call__(self, xs, ys):
        xs = permutate_list(xs, argsort_list_descent(xs), inv=False)
        xs = F.transpose_sequence(xs)
        ys = permutate_list(ys, argsort_list_descent(ys), inv=False)
        ys = F.transpose_sequence(ys)
        return super(CRF, self).__call__(xs, ys)

    def argmax(self, xs):
        xs = permutate_list(xs, argsort_list_descent(xs), inv=False)
        xs = F.transpose_sequence(xs)
        score, path = super(CRF, self).argmax(xs)
        path = F.transpose_sequence(path)
        return score, path


class SequentialBase(Chain):

    def __init__(self, **links):
        super(SequentialBase, self).__init__(**links)

    def _sequential_var(self, xs):
        if self._cpu:
            xs = [Variable(cuda.to_cpu(x), volatile='auto') for x in xs]
        else:
            xs = [Variable(cuda.to_gpu(x), volatile='auto') for x in xs]
        return xs

    def _accuracy(self, ys, ts):
        ys = permutate_list(ys, argsort_list_descent(ys), inv=False)
        ts = permutate_list(ts, argsort_list_descent(ts), inv=False)
        correct = 0
        total = 0
        exact = 0
        for _y, _t in zip(ys, ts):
            y = _y.data
            t = _t.data
            _correct = (y == t).sum()
            _total = t.size
            if _correct == _total:
                exact += 1
            correct += _correct
            total += _total
        accuracy = correct / total
        self._eval = {'accuracy': accuracy, 'correct': correct, 'total': total, 'exact': exact}
        return accuracy


class BLSTMBase(SequentialBase):

    def __init__(self, embeddings, n_labels, dropout=0.5, train=True):
        vocab_size, embed_size = embeddings.shape
        feature_size = embed_size
        super(BLSTMBase, self).__init__(
            embed=L.EmbedID(
                in_size=vocab_size,
                out_size=embed_size,
                initialW=embeddings,
            ),
            f_lstm=LSTM(feature_size, feature_size, dropout),
            b_lstm=LSTM(feature_size, feature_size, dropout),
            linear=L.Linear(feature_size * 2, n_labels),
        )
        self._dropout = dropout
        self._n_labels = n_labels
        self.train = train

    def reset_state(self):
        self.f_lstm.reset_state()
        self.b_lstm.reset_state()

    def __call__(self, xs):
        self.reset_state()
        xs_f = []
        xs_b = []
        for x in xs:
            _x = self.embed(self.xp.array(x))
            xs_f.append(_x)
            xs_b.append(_x[::-1])
        hs_f = self.f_lstm(xs_f, self.train)
        hs_b = self.b_lstm(xs_b, self.train)
        ys = [self.linear(F.dropout(F.concat([h_f, h_b[::-1]]), ratio=self._dropout, train=self.train)) for h_f, h_b in zip(hs_f, hs_b)]
        return ys


class BLSTM(BLSTMBase):

    def __init__(self, embeddings, n_labels, dropout=0.5, train=True):
        super(BLSTM, self).__init__(embeddings, n_labels, dropout, train)

    def __call__(self, xs, ts):
        ts = self._sequential_var(ts)
        hs = super(BLSTM, self).__call__(xs)

        loss = 0
        ys = []
        for h, t in zip(hs, ts):
            loss += F.softmax_cross_entropy(h, t)
            ys.append(F.reshape(F.argmax(h, axis=1), t.shape))

        accuracy = self._accuracy(ys, ts)
        return loss, accuracy

    def parse(self, xs):
        hs = super(BLSTM, self).__call__(xs)
        ys = []
        for h in hs:
            ys.append(np.argmax(cuda.to_cpu(h.data), axis=1))
        return ys


class BLSTMCRF(BLSTMBase):

    def __init__(self, embeddings, n_labels, dropout=0.5, train=True):
        super(BLSTMCRF, self).__init__(embeddings, n_labels, dropout, train)
        self.add_link('crf', CRF(n_labels))

    def __call__(self, xs, ts):
        ts = self._sequential_var(ts)
        hs = super(BLSTMCRF, self).__call__(xs)

        loss = self.crf(hs, ts)
        _, ys = self.crf.argmax(hs)

        accuracy = self._accuracy(ys, ts)
        return loss, accuracy

    def parse(self, xs):
        hs = super(BLSTMCRF, self).__call__(xs)
        _, ys = self.crf.argmax(hs)
        ys = [y.data for y in ys]
        return ys
