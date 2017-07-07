# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.
RoIDataLayer implements a Caffe Python layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from model.config import cfg
import numpy as np
import time

import matplotlib.pyplot as plt


class Mnist(object):
    """Fast R-CNN data layer used for training."""

    def __init__(self, random=False):
        """Set the roidb to be used by this layer during training."""
        self._images = np.expand_dims(np.array([i.reshape(28, 28) for i in mnist.train.images]), 3)
        self._labels = mnist.train.labels.astype(np.int32)

        self._num_train_images = 55000
        self._num_labels = 100
        self._num_classes = 10
        # data split
        self._images_labeled, self._images_unlabeled, self._labels = self._split_data()
        # Also set a random flag
        self._random = random
        self._shuffle_inds()

    def _split_data(self):
        counts = np.zeros(self._num_classes)
        labeled_indices = list()
        num_per_class = int(self._num_labels / self._num_classes)
        for i, l in enumerate(self._labels):
            index = np.nonzero(l)[0][0]
            if counts[index] < num_per_class:
                counts[index] += 1
                labeled_indices.append(i)
            elif counts.sum() == self._num_labels:
                break
            else:
                continue
        all_indices = set(range(self._num_train_images))
        unlabeled_indices = list(all_indices - set(labeled_indices))
        images_labeled = self._images[labeled_indices]
        images_unlabeled = self._images[unlabeled_indices]
        labels = self._labels[labeled_indices]
        return images_labeled, images_unlabeled, labels

    def _shuffle_inds(self):
        """Randomly permute the training roidb."""
        # If the random flag is set,
        # then the database is shuffled according to system time
        # Useful for the validation set
        if self._random:
            st0 = np.random.get_state()
            millis = int(round(time.time() * 1000)) % 4294967295
            np.random.seed(millis)

        self._perm = np.random.permutation(np.arange(len(self._images_unlabeled)))
        # Restore the random state
        if self._random:
            np.random.set_state(st0)

        self._cur = 0

    def _get_next_minibatches_inds(self):
        """Return the roidb indices for the next minibatch."""

        if self._cur + cfg.TRAIN.BATCH_SIZE >= len(self._images_unlabeled):
            self._shuffle_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.BATCH_SIZE]
        self._cur += cfg.TRAIN.BATCH_SIZE

        return db_inds

    def _get_next_minibatches(self):
        """Return the blobs to be used for the next minibatch.
        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatches_inds()
        minibatch_db = np.array([self._images_unlabeled[i] for i in db_inds])
        return minibatch_db

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        unlabeled_images = self._get_next_minibatches()
        return self._images_labeled, unlabeled_images, self._labels