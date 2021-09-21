#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Roger Wang
# Created Date: Apr 12 2018
# Main training pipeline
# =============================================================================

import cnn
import data_processing as data
import learner

categories_similar = [
        "apple", "basketball", "circle", "compass", "cookie", "donut", "potato",
        "soccer ball", "watermelon", "wheel", ]

categories_different = [
        "airplane", "axe", "bed", "bicycle", "butterfly", "envelope", "knife",
        "square", "star", "donut", ]

categories_20 = [
        "airplane", "axe", "bed", "bicycle", "butterfly", "cannon", "dog",
        "donut", "envelope", "firetruck", "garden", "hourglass", "knife",
        "laptop", "monkey", "oven", "pillow", "spreadsheet", "square", "star", ]

categories_30 = [
        "airplane", "axe", "bed", "bicycle", "butterfly", "cannon", "dog",
        "donut", "envelope", "firetruck", "garden", "hourglass", "knife",
        "laptop", "monkey", "oven", "pillow", "spreadsheet", "square", "star",
        "speedboat", "sun", "triangle", "zebra", "microphone", "mailbox",
        "hospital", "foot", "cow", "boomerang", ]

categories_split_1 = [
        "butterfly", "cat", "horse", "whale", "duck", "apple", "pear",
        "strawberry", "bed", "table", "basketball", "cup", "fork",
        "car", "bus", "laptop",
        ]

categories_split_2 = [
        "bird", "dog", "fish", "zebra", "monkey", "banana", "grapes",
        "blueberry", "couch", "bench", "baseball", "mug", "knife",
        "truck", "van", "computer",
        ]

categories = categories_split_1
categories_small = categories_split_2

num_classes = len(categories)
assert(num_classes == len(categories_small))

X_train, y_train, X_valid, y_valid, X_test, y_test, emb = data.make_dataset(
        categories, validation=.1, test=.2, cnn_style=True,
        word2vec=True)

img_dim = X_train.shape[1]

model = learner.Learner(model=cnn.CNN(num_classes=num_classes,
                                      channel_nums=[32, -1, 64, -1, 128, -1],
                                      kernel_size=3,
                                      mlp_sizes=[128]),
                        learning_rate=.001,
                        epochs=5)

model.fit(X_train, y_train)

topk = 3

print("Training accuracy:\t%.4f" % model.score(X_train, y_train, k=topk))
print("Validation accuracy:\t%.4f" % model.score(X_valid, y_valid, k=topk))
print("Test accuracy:\t%.4f" % model.score(X_test, y_test, k=topk))

X_train, y_train, X_valid, y_valid, X_test, y_test, emb = data.make_dataset(
    categories_small, validation=.1, test=.2, cnn_style=True,
    word2vec=True, train_frac=.01)

print("Retraining on smaller dataset...")

model.prepare_for_transfer_learning()
model.fit(X_train, y_train)

print("Training accuracy:\t%.4f" % model.score(X_train, y_train, k=topk))
print("Validation accuracy:\t%.4f" % model.score(X_valid, y_valid, k=topk))
print("Test accuracy:\t%.4f" % model.score(X_test, y_test, k=topk))

# Optional models
# model = learner.Learner(model=mlp.MLP(input_size=img_dim,
#                                       hidden_layers=[1024, 512, 256],
#                                       output_size=len(categories),
#                                       dropout=.2),
#                         learning_rate=.001,
#                         epochs=5)
#
# model = learner.Learner(model=resnet.ResNet(), learning_rate=.001, epochs=1)