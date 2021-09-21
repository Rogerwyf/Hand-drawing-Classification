#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Roger Wang
# Created Date: Apr 4 2018
# Data/image handling and preprocessing
# =============================================================================

import os

import matplotlib.pyplot as plt
import numpy as np
import torchwordemb

DATA_DIR = "data"
IMG_DIM = 28


def load_images(filename, cnn_style):
    """
    Loads data from `filename` under the data directory.
    Args:
        filename: string with name of the file under the data directory
                  (see DATA_DIR)
        cnn_style: flag for whether to output images as 1 x rows x cols
        or vectorized
    Returns:
        if `cnn_style` is False,
        [n, d] ndarray where n is the number of images and d is the size of
        each (vectorized) image.
        if `cnn_style` is True,
        [n, 1, rows, cols] array.
    """

    images = np.load(os.path.join(DATA_DIR, filename))
    if cnn_style:
        images = images.reshape(-1, 1, IMG_DIM, IMG_DIM)

    return images


def show_image(img):
    """
    Shows one image on the screen. This assumes the resolution is a square,
    that is, that width = height.
    Args:
        img: [d] ndarray where d is the size of the (vectorized) image.
    """

    assert (len(img.shape) == 1)
    d = img.shape[0]
    wh = int(d ** .5)
    assert (wh ** 2 == d)

    plt.imshow(img.reshape(wh, wh), cmap='gray')
    plt.show()


def make_dataset(categories, validation, test, cnn_style=False, word2vec=False,
                 train_frac=None):
    """
    Makes a dataset out of the given categories and validation and test
    proportions.
    Args:
        categories: list of strings corresponding to files, e.g. 'dog' for
        file 'dog.npy'.
        validation: float in (0, 1), proportion of validation examples.
        test: float in (0, 1), proportion of test examples.

        cnn_style: if True, output shape will be [batch, channel, row, col]
        word2vec: if given, also return an [n, e] array where n is the number
        of classes and e is the dimension of the embedding

        train_frac: if provided, only take this fraction of training examples.
    """

    X_train = []
    y_train = []
    X_valid = []
    y_valid = []
    X_test = []
    y_test = []
    for i, cat in enumerate(categories):
        images = load_images(cat + ".npy", cnn_style=cnn_style)

        num_test = int(test * len(images))
        num_valid = int(validation * len(images))
        num_train = len(images) - num_valid - num_test

        X_train.extend(images[:num_train])
        X_valid.extend(images[num_train: (num_train + num_valid)])
        X_test.extend(images[(num_train + num_valid):])

        y_train.extend([i] * num_train)
        y_valid.extend([i] * num_valid)
        y_test.extend([i] * num_test)

    def shuffled_arrays(X, y):
        X = np.array(X)
        y = np.array(y)
        assert (len(X) == len(y))
        order = np.arange(len(X))
        np.random.shuffle(order)
        return X[order], y[order]

    X_train, y_train = shuffled_arrays(X_train, y_train)
    X_valid, y_valid = shuffled_arrays(X_valid, y_valid)
    X_test, y_test = shuffled_arrays(X_test, y_test)

    mean = X_train.mean()
    std = X_train.std()

    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std
    X_test = (X_test - mean) / std

    if train_frac:
        num_train = int(train_frac * len(X_train))
        X_train = X_train[:num_train]
        y_train = y_train[:num_train]

    if word2vec:
        vocab, vec = torchwordemb.load_glove_text("glove.6B.100d.txt")
        emb = np.zeros((len(categories), 100))
        for i, cat in enumerate(categories):
            pos = vocab[cat]
            emb[i] = vec[pos]
        return X_train, y_train, X_valid, y_valid, X_test, y_test, emb
    else:
        return X_train, y_train, X_valid, y_valid, X_test, y_test