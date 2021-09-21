#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Roger Wang
# Created Date: Apr 5 2018
# Resnet Model by PyTorch
# =============================================================================

from torch import nn


class ResNet(nn.Module):
    def __init__(self, num_stages=4, blocks_per_stage=4, block_size=2,
            fc_size=512, init_num_filters=32, img_dim=28, num_classes=10):
        super().__init__()

        kernel_sizes = [11, 5, 3]
        def consume_kernel_size():
            nonlocal kernel_sizes
            kernel_size = kernel_sizes[0]
            if len(kernel_sizes) > 1:
                del kernel_sizes[0]
            return kernel_size

        paddings = [5, 2, 1]
        def consume_padding():
            nonlocal paddings
            padding = paddings[0]
            if len(paddings) > 1:
                del paddings[0]
            return padding

        curr_num_filters = init_num_filters

        self.initial = nn.ModuleList()
        self.initial.append(nn.Conv2d(
            in_channels  = 1,
            out_channels = curr_num_filters,
            kernel_size  = consume_kernel_size(),
            padding      = consume_padding()))

        prev_num_filters = curr_num_filters
        curr_filter_size = img_dim

        self.blocks = nn.ModuleList()
        self.relu   = nn.ReLU()

        self.maxpool = nn.MaxPool2d(2)

        self.projections = nn.ModuleList()

        for _ in range(num_stages):
            for _ in range(blocks_per_stage):
                block_modules = []

                for _ in range(block_size):
                    if curr_num_filters == prev_num_filters:
                        stride = 1
                    else:
                        stride = 2
                        curr_filter_size = (curr_filter_size+1) // 2

                    block_modules.append(nn.Conv2d(
                        in_channels  = prev_num_filters,
                        out_channels = curr_num_filters,
                        kernel_size  = consume_kernel_size(),
                        padding      = consume_padding(),
                        stride       = stride))
                    block_modules.append(nn.BatchNorm2d(curr_num_filters))
                    prev_num_filters = curr_num_filters

                self.blocks.append(nn.Sequential(*block_modules))

            curr_num_filters = curr_num_filters * 2
            next_filter_size = (curr_filter_size+1) // 2
            self.projections.append(nn.Linear(
                prev_num_filters * curr_filter_size * curr_filter_size,
                curr_num_filters * next_filter_size * next_filter_size))

        self.avg_pool = nn.AvgPool2d(kernel_size = curr_filter_size)

        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(prev_num_filters, fc_size))
        self.fc.append(nn.ReLU())
        self.fc.append(nn.Linear(fc_size, num_classes))

    def forward(self, X):
        batch_size = len(X)
        curr = X
        for m in self.initial:
            curr = m(curr)

        past_input = curr
        curr_proj  = 0
        for m in self.blocks:
            curr = m(curr)

            past_channels = past_input.shape[1]
            if curr.shape[1] == past_channels:
                curr += past_input
            else:
                past_input = past_input.view(batch_size, -1)
                past_input = self.projections[curr_proj](past_input)
                past_input = past_input.view(curr.shape)
                curr      += past_input
                curr_proj += 1

            curr = self.relu(curr)
            past_input = curr

        curr = self.avg_pool(curr)
        curr = curr.view(batch_size, -1) # remove 1x1 filter size

        for m in self.fc:
            curr = m(curr)

        return curr