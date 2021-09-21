#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Roger Wang
# Created Date: Apr 3 2018
# Convolutional Neural Networks
# =============================================================================

from torch import nn


# -1 for number of channels means max pool,
# otherwise conv + relu


class CNN(nn.Module):
    def __init__(self, channel_nums, mlp_sizes, kernel_size=9, activation="relu", img_dim=28,
                 num_classes=10):
        super().__init__()

        padding = (kernel_size - 1) // 2

        if activation == "relu":
            activation_module = nn.ReLU()
        elif activation == "leaky_relu":
            activation_module = nn.LeakyReLU(.1)
        else:
            raise NotImplementedError

        layers = []
        prev_channel_num = 1
        filter_size = img_dim
        for channel_num in channel_nums:
            if channel_num == -1:
                layers.append(nn.MaxPool2d(kernel_size=2))
                filter_size = filter_size // 2
            else:
                layers.append(nn.Conv2d(
                    in_channels=prev_channel_num,
                    out_channels=channel_num,
                    kernel_size=kernel_size,
                    padding=padding))
                layers.append(activation_module)
                prev_channel_num = channel_num

        self.cnn = nn.Sequential(*layers)

        layers = []
        prev_features = prev_channel_num * filter_size * filter_size
        for mlp_size in mlp_sizes:
            layers.append(nn.Linear(prev_features, mlp_size))
            layers.append(activation_module)
            # layers.append(nn.Dropout(.5))
            prev_features = mlp_size

        self.mlp = nn.Sequential(*layers)
        self.last_hidden_size = prev_features
        self.last = nn.Linear(self.last_hidden_size, num_classes)
        self.num_classes = num_classes

    def reset_last_layer(self):
        self.last = nn.Linear(self.last_hidden_size, self.num_classes)

    def forward(self, X):
        batch_size = len(X)
        input = X
        conved = self.cnn(input)
        features = conved.view(batch_size, -1)
        return self.last(self.mlp(features))
