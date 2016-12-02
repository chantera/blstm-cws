#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from libs.tools import Dataset


class CorpusReader:
    DELIMITER = '  '
    LABELS = {'B': 0, 'M': 1, 'E': 2, 'S': 3}

    def __init__(self, converter):
        self.converter = converter

    def load(self, path, train=True):
        X_raw = []
        Y_raw = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chars, labels = self._parse_line(line)
                X_raw.append(chars)
                Y_raw.append(labels)
        X, Y = self._transform(X_raw, Y_raw, train)
        return Dataset(X, Y)

    @classmethod
    def _parse_line(cls, line):
        segments = line.split(cls.DELIMITER)
        chars = []
        labels = []
        for segment in segments:
            s_len = len(segment)
            if s_len == 0:
                continue
            elif s_len == 1:
                labels.append('S')
            else:
                labels.append('B')
                for j in range(s_len - 2):
                    labels.append('M')
                labels.append('E')
            [chars.append(char) for char in segment]
        assert len(chars) == len(labels)
        return chars, labels

    def _transform(self, X_chars, Y_labels, train=True):
        if train:
            X = self.converter.fit_transform(X_chars)
        else:
            X = self.converter.transform(X_chars)
        Y = [None] * len(X)
        for i, labels in enumerate(Y_labels):
            y = np.full(len(labels), -1, dtype=np.int32)
            for j, label in enumerate(labels):
                y[j] = self.LABELS[label]
            Y[i] = y
        return X, Y
