#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Roger Wang
# Created Date: Apr 4 2018
# Expects that the files "dog.npy" and "cat.npy" have been downloaded to the
# "data/" directory.
# =============================================================================

import random

import data_processing as data

dog = data.load_images("dog.npy")
cat = data.load_images("cat.npy")

data.show_image(random.choice(dog))
data.show_image(random.choice(cat))