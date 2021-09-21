#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Roger Wang
# Created Date: Apr 6 2018
# Learner function
# =============================================================================
import numpy as np
import torch
from torch import autograd
from torch import nn


class Learner:
    """
    Generic learner module that takes a model and training settings, and
    provides an interface for training, evaluation and prediction.
    Arguments:
        model: a Torch object that generates output when applied to an
        input.
        solver: either 'adam' or a Torch optimizer.
        batch_size: integer
        learning_rate: float
        epochs: integer
        loss: either 'crossentropy', 'mse' or a Torch loss object
        emb: if not None, embedding in which label words are assumed to live
        and against which output will be compared (requires loss = 'mse')
        use_cuda: boolean
    """

    def __init__(self, model, solver='adam', batch_size=256, learning_rate=.01, epochs=3,
                 loss='crossentropy', emb=None, use_cuda=True):
        self.model = model
        self.solver = solver
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss = loss
        self.use_cuda = use_cuda

        if emb is not None:
            self.emb = autograd.Variable(self._maybe_cuda(
                torch.Tensor(emb)))
        else:
            self.emb = None

    def _maybe_cuda(self, x):
        return x.cuda() if self.use_cuda else x

    def _optimizer(self):
        if self.solver == 'adam':
            return torch.optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)
        else:
            raise NotImplementedError

    def _loss(self):
        if self.loss == 'crossentropy':
            return nn.CrossEntropyLoss()
        elif self.loss == 'mse':
            return nn.MSELoss()
        else:
            raise NotImplementedError

    def _shuffle(self, X, y):
        n = len(X)
        assert (n == len(y))
        order = np.arange(n)
        np.random.shuffle(order)
        return X[order], y[order]

    def _input_variable(self, X):
        return self._maybe_cuda(autograd.Variable(torch.Tensor(X)))

    def _label_variable(self, y):
        y = self._maybe_cuda(autograd.Variable(torch.LongTensor(y)))
        if self.emb is None:
            return y
        else:
            return self.emb[y.data]

    def _num_batches(self, n):
        return int(np.ceil(n / self.batch_size))

    def fit(self, X, y):
        """
        Trains model on inputs X and outputs y.
        Arguments:
            X: [batch_size, ...] input numpy array
            y: [batch_size, ...] labels numpy array
        """

        self.model = self._maybe_cuda(self.model)
        self.model.train()

        optimizer = self._optimizer()
        loss_fn = self._loss()

        for epoch in range(self.epochs):
            print("Epoch {} of {}".format(epoch, self.epochs))

            X, y = self._shuffle(X, y)
            num_batches = self._num_batches(len(X))
            avg_loss = -1

            for batch in range(num_batches):
                if batch % 100 == 0:
                    print("Batch %d of %d, loss %.4f" % (
                        batch, num_batches, avg_loss))

                begin = self.batch_size * batch
                end = self.batch_size * (batch + 1)

                input = self._input_variable(X[begin:end])
                label = self._label_variable(y[begin:end])
                output = self.model(input)
                loss = loss_fn(output, label)

                if avg_loss < 0:
                    avg_loss = loss.data[0]
                else:
                    avg_loss = .1 * avg_loss + .9 * loss.data[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def prepare_for_transfer_learning(self):
        if self.emb is None:  # only if learning one-hot encoding, not embedding
            self.model.reset_last_layer()

    def _predict_batch(self, X, k=1):
        self.model = self._maybe_cuda(self.model)
        self.model.eval()

        X = self._input_variable(X)
        output = self.model(X)

        if self.emb is not None:
            diffs = []
            for i in range(len(self.emb)):
                diffs.append(torch.norm(output - self.emb[i], p=2, dim=1,
                                        keepdim=True))
            diffs = torch.cat(diffs, dim=1)
            sorted, indices = torch.sort(diffs, dim=1)
            pred = indices[:, :k].data.cpu().numpy()
            # _, pred = torch.min(diffs, dim=1)
            # pred = pred.data.cpu().numpy()
        else:
            sorted, indices = torch.sort(output, dim=1, descending=True)
            pred = indices[:, :k].data.cpu().numpy()
            # _, pred = torch.max(output, dim=1)
            # pred = pred.data.cpu().numpy()

        assert (len(pred) == len(X))
        return pred

    def predict(self, X, k=1):
        num_batches = self._num_batches(len(X))
        output = np.zeros((len(X), k))

        for batch in range(num_batches):
            begin = self.batch_size * batch
            end = self.batch_size * (batch + 1)
            output[begin:end, :] = self._predict_batch(X[begin:end], k=k)

        return output

    def score(self, X, y, k=1):
        output = self.predict(X, k=k)
        assert (len(output) == len(y))
        correct = np.zeros(len(y))
        for i in range(k):
            correct += (output[:, i] == y)
        return correct.mean()
        # assert(output.shape == y.shape)
        # return (output == y).mean()
