#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from operator import itemgetter
import re

import numpy as np


class Tokenizer(metaclass=ABCMeta):

    @abstractmethod
    def tokenize(self, document):
        raise NotImplementedError()


class SimpleTokenizer(Tokenizer):

    def tokenize(self, document):
        return document.split()


""" Tokenizer example

from nltk.tokenize import word_tokenize

class NltkTokenizer(Tokenizer)

    def tokenize(self, document):
        return word_tokenize(document)
"""


class Preprocessor:

    def __init__(self,
                 embed_file=None,
                 embed_size=50,
                 unknown="<UNK>",
                 tokenizer=None):
        self._init_embeddings(embed_file, embed_size)
        self._unknown_id = self._add_vocabulary(unknown, random=False)
        self._pad_id = -1
        if tokenizer:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = SimpleTokenizer()

    def _init_embeddings(self, embed_file, embed_size):
        if embed_file is not None:
            vocab_file = None
            if isinstance(embed_file, (list, tuple)):
                embed_file, vocab_file = embed_file
            vocabulary, embeddings = self._load_embeddings(embed_file, vocab_file)
            embed_size = embeddings.shape[1]
        elif embed_size is not None:
            if embed_size <= 0 or type(embed_size) is not int:
                raise ValueError("embed_size must be a positive integer value")
            vocabulary, embeddings = {}, np.zeros((0, embed_size), np.float32)
        else:
            raise ValueError("embed_file os embed_size must be specified")

        self._vocabulary = vocabulary
        self._embeddings = embeddings
        self._new_embeddings = []
        self._embed_size = embed_size

    @staticmethod
    def _load_embeddings(embed_file, vocab_file=None):
        vocabulary = {}
        embeddings = []
        if vocab_file:
            with open(embed_file) as ef, open(vocab_file) as vf:
                for line1, line2 in zip(ef, vf):
                    word = line2.strip()
                    vector = line1.strip().split(" ")
                    if word not in vocabulary:
                        vocabulary[word] = len(vocabulary)
                        embeddings.append(np.array(vector, dtype=np.float32))
        else:
            with open(embed_file) as f:
                lines = f.readlines()
                index = 0
                if len(lines[0].strip().split(" ")) <= 2:
                    index = 1  # skip header
                for line in lines[index:]:
                    cols = line.strip().split(" ")
                    word = cols[0]
                    if word not in vocabulary:
                        vocabulary[word] = len(vocabulary)
                        embeddings.append(np.array(cols[1:], dtype=np.float32))
        return vocabulary, np.array(embeddings)

    def _add_vocabulary(self, word, random=True):
        if word in self._vocabulary:
            return self._vocabulary[word]
        index = len(self._vocabulary)
        self._vocabulary[word] = index
        if random:
            word_vector = np.random.uniform(-1, 1, self._embed_size)  # generate a random embedding for an unknown word
        else:
            word_vector = np.zeros(self._embed_size, dtype=np.float32)
        self._new_embeddings.append(word_vector)
        return index

    def fit(self, raw_documents):
        for document in raw_documents:
            self._fit_each(document)
        return self

    def _fit_each(self, raw_document, preprocess=True):
        tokens = self._extract_tokens(raw_document, preprocess)
        for token in tokens:
            if token not in self._vocabulary:
                self._add_vocabulary(token, random=True)
        return self

    def fit_one(self, raw_document, preprocess=True):
        return self._fit_each(raw_document, preprocess)

    def transform(self, raw_documents, length=None):
        samples = []
        for document in raw_documents:
            samples.append(self._transform_each(document, length))
        if length:
            samples = np.array(samples, dtype=np.int32)
        return samples

    def _transform_each(self, raw_document, length=None, preprocess=True):
        tokens = self._extract_tokens(raw_document, preprocess)
        if length is not None:
            if len(tokens) > length:
                raise ValueError("Token length exceeds the specified length value")
            word_ids = np.full(length, self._pad_id, dtype=np.int32)
        else:
            word_ids = np.zeros(len(tokens), dtype=np.int32)
        for i, token in enumerate(tokens):
            word_ids[i] = self._vocabulary.get(token, self._unknown_id)
        return word_ids

    def transform_one(self, raw_document, length=None, preprocess=True):
        return self._transform_each(raw_document, length, preprocess)

    def _extract_tokens(self, raw_document, preprocess=True):
        if type(raw_document) == list or type(raw_document) == tuple:
            tokens = raw_document
        else:
            tokens = self._tokenizer.tokenize(raw_document)
        if preprocess:
            tokens = [self._preprocess_token(token) for token in tokens]
        return tokens

    @staticmethod
    def _preprocess_token(token):
        return re.sub(r'^\d+(,\d+)*(\.\d+)?$', '<NUM>', token.lower())

    def fit_transform(self, raw_documents, length=None):
        return self.fit(raw_documents).transform(raw_documents, length)

    def fit_transform_one(self, raw_document, length=None, preprocess=True):
        return self._fit_each(raw_document, preprocess)._transform_each(raw_document, length, preprocess)

    def pad(self, tokens, length):
        assert type(tokens) == np.ndarray
        pad_size = length - tokens.size
        if pad_size < 0:
            raise ValueError("Token length exceeds the specified length value")
        return np.pad(tokens, (0, pad_size), mode="constant", constant_values=self._pad_id)

    def get_embeddings(self):
        if len(self._new_embeddings) > 0:
            self._embeddings = np.r_[self._embeddings, self._new_embeddings]
            self._new_embeddings = []
        return self._embeddings

    def get_vocabulary_id(self, word):
        return self._vocabulary.get(word, -1)

    @property
    def embeddings(self):
        return self.get_embeddings()

    @property
    def unknown_id(self):
        return self._unknown_id

    @property
    def pad_id(self):
        return self._pad_id


class Dataset(Sequence):
    """Immutable Class"""

    def __init__(self, *samples):
        self._n_cols = len(samples)
        if self._n_cols == 0:
            self._samples = []
            self._n_cols = 1
            self._dtype = list
        elif self._n_cols == 1:
            self._dtype = type(samples[0])
            if self._dtype is np.ndarray:
                self._samples = samples[0]
                self._n_cols = self._samples.shape[0]
            else:
                self._samples = list(samples[0])
                if len(self._samples) > 0 and \
                        (type(self._samples[0]) is tuple or type(self._samples[0]) is list):
                    self._n_cols = len(self._samples[0])
                    self._dtype = type(self._samples[0])
        elif self._n_cols > 1:
            self._dtype = type(samples[0])
            if self._dtype is np.ndarray:
                self._samples = [_samples for _samples in zip(*samples)]
            else:
                self._samples = [self._dtype(_samples) for _samples in zip(*samples)]
        self._len = len(self._samples)
        self._indexes = np.arange(self._len)

    def batch(self, size, shuffle=False, colwise=False):
        if shuffle:
            np.random.shuffle(self._indexes)
        return _DatasetBatchIterator(Dataset(self.take(self._indexes)), size, colwise)

    def __len__(self):
        return self._len

    def __getitem__(self, key):
        if self._dtype == tuple:
            return tuple(self._samples)[key]
        return self._samples[key]

    def take(self, indices):
        if self._dtype is np.ndarray:
            return self._samples.take(indices, axis=0)
        elif isinstance(indices, Iterable):
            return self._dtype(itemgetter(*indices)(self._samples))
        return self._samples[indices]

    def __repr__(self):
        return repr(self._samples)

    def __str__(self):
        return str(self._samples)

    def cols(self):
        if self._dtype is np.ndarray:
            if self._n_cols > 1:
                return np.swapaxes(self._samples, 0, 1)
            return self._samples
        else:
            if self._n_cols > 1:
                return tuple(self._dtype(col) for col in zip(*self._samples))
            return self._dtype(self._samples),

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self):
        return self.__len__()

    @property
    def n_cols(self):
        return self._n_cols


class _DatasetBatchIterator(Iterator):

    def __init__(self, dataset, batch_size, colwise=False):
        self._dataset = dataset
        self._batch_size = batch_size
        self._colwise = colwise

    def __iter__(self):
        dataset = self._dataset
        dtype = dataset.dtype
        if self._colwise:
            if dtype is np.ndarray:
                def _take(dataset, offset, batch_size):
                    return dataset.cols()[:, offset:offset + batch_size]
            elif dataset.n_cols == 1:
                def _take(dataset, offset, batch_size):
                    return dataset[offset:offset + batch_size],
            else:
                def _take(dataset, offset, batch_size):
                    return tuple(dtype(col) for col in zip(*dataset._samples[offset:offset + batch_size]))
        else:
            def _take(dataset, offset, batch_size):
                return dataset[offset:offset + batch_size]
        size = dataset.size
        offset = 0
        while True:
            if offset >= size:
                raise StopIteration()
            yield _take(dataset, offset, self._batch_size)
            offset += self._batch_size

    def __next__(self):
        self.__iter__()


if __name__ == "__main__":
    """Sample Code"""

    import pandas as pd

    train_samples = [
        ("Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .", 1),
        ("Mr. Vinken is chairman of Elsevier N.V. , the Dutch publishing group .", 0),
        ("Rudolph Agnew , 55 years old and former chairman of Consolidated Gold Fields PLC , was named a nonexecutive director of this British industrial conglomerate .", 1),
        ("A form of asbestos once used to make Kent cigarette filters has caused a high percentage of cancer deaths among a group of workers exposed to it more than 30 years ago , researchers reported .", 1),
        ("The asbestos fiber , crocidolite , is unusually resilient once it enters the lungs , with even brief exposures to it causing symptoms that show up decades later , researchers said .", 1),
    ]
    test_samples = [("Lorillard Inc.  , the unit of New York-based Loews Corp. that makes Kent cigarettes , stopped using crocidolite in its Micronite cigarette filters in 1956 .", 0)]

    train_df = pd.DataFrame(train_samples)
    train_df.columns = ['X', 'y']
    test_df = pd.DataFrame(test_samples)
    test_df.columns = ['X', 'y']

    tokenizer = SimpleTokenizer()
    processor = Preprocessor(embed_size=50, tokenizer=tokenizer)
    train_X, train_y = processor.fit_transform(train_df['X'].values.tolist(), length=40), train_df['y'].as_matrix()
    test_X, test_y = processor.transform(test_df['X'].values.tolist(), length=40), test_df['y'].as_matrix()

    train_dataset = Dataset(train_X, train_y)
    print(train_dataset)
    print('---\n')

    test_dataset = Dataset(test_X, test_y)
    print(test_dataset)
    print('---\n')

    def forward_batch(samples):
        loss = 80.0
        accuracy = 0.6
        return loss, accuracy

    def update(loss):
        pass

    print('start training ...\n')

    batch_size = 2
    epoch = 10
    n_train = len(train_dataset)
    n_test = len(test_dataset)
    for i in range(1, epoch + 1):
        # Training
        for batch in train_dataset.batch(batch_size, shuffle=True):
            batch_loss, batch_accuracy = forward_batch(batch)
            update(batch_loss)
        print("[training] epoch %d - #samples: %d, loss: %f, accuracy: %f" % (i, n_train, 80.0, 0.6))

        # Evaluation
        loss = 0.0
        accuracy = 0.0
        for batch in test_dataset.batch(batch_size):
            batch_loss, batch_accuracy = forward_batch(batch)
        print("[evaluation] epoch %d - #samples: %d, loss: %f, accuracy: %f" % (i, n_test, 80.0, 0.6))

    print('--')
