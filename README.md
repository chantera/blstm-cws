# blstm-cws : Bi-directional LSTM for Chinese Word Segmentation

blstm-cws is preliminary implementation for Chinese Word Segmentation.

## Installation

blstm-cws works on Python3 and requires chainer, gensim, numpy and progressbar2.

```sh
$ git clone https://github.com/chantera/blstm-cws
$ cd blstm-cws
$ python setup.py  # this will download large text data and produce embeddings.
```

Then you can try blstm-cws using the following command:

```sh
$ python app/train.py
```

## Usage

```sh
usage: train.py [-h] [--batchsize BATCHSIZE] [--epoch EPOCH] [--gpu GPU]
                [--save] [--debug DEBUG] [--logdir LOGDIR] [--silent]

optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE, -b BATCHSIZE
                        Number of examples in each mini-batch
  --epoch EPOCH, -e EPOCH
                        Number of sweeps over the dataset to train
  --gpu GPU, -g GPU     GPU ID (negative value indicates CPU)
  --save                Save the NN model
  --debug DEBUG         Enable debug mode
  --logdir LOGDIR       Log directory
  --silent, --quiet     Silent execution: does not print any message
```

## Performance

A brief report is available here <http://qiita.com/chantera/items/d8104012c80e3ea96df7> . (Written in Japanese)


## References

  - Chen, X., Qiu, X., Zhu, C., Liu, P. and Huang, X., 2015. Long short-term memory neural networks for chinese word segmentation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (pp. 1385-1394).  <http://aclweb.org/anthology/D15-1141.pdf>
  - Huang, Z., Xu, W., Yu, K., 2015. Bidirectional LSTM-CRF models for sequence tagging. arXiv preprint arXiv:1508.01991. <https://arxiv.org/abs/1508.01991>
  - Yao, Y., Huang, Z., 2016. Bi-directional LSTM Recurrent Neural Network for Chinese Word Segmentation. arXiv preprint arXiv:1602.04874. <https://arxiv.org/abs/1602.04874>

License
----
MIT License

&copy; Copyright 2016 Teranishi Hiroki

