#!/usr/bin/python
# -*- coding: utf-8 -*-


import torch.nn as nn
import torch.nn.functional as F


class TCNNConfig(object):
    """CNN Parameters"""
    embedding_dim = 64      # embedding vector size
    seq_length = 200        # maximum length of sequence
    vocab_size = 50000      # most common words

    num_filters = 64        # number of convolution filters
    kernel_size = 5         # kernel size

    hidden_dim = 128        # hidden size of fully connected layer

    dropout_prob = 0.2      # how much probability to be dropped
    learning_rate = 1e-3    # learning rate
    batch_size = 128        # batch size for training
    num_epochs = 10         # total number of epochs

    print_per_batch = 100   # print out the intermediate status every n batches

    num_classes = 10        # number of classes


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()

        self.dropout_p = config.dropout_prob

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)  # embedding layer
        self.conv = nn.Conv1d(config.embedding_dim, config.num_filters, config.kernel_size)  # conv1d layer

        self.fc1 = nn.Linear(config.num_filters, config.hidden_dim)   # fully connected layer
        self.fc2 = nn.Linear(config.hidden_dim, config.num_classes)   # classification layer

    def forward(self, inputs):
        embedded = self.embedding(inputs).permute(0, 2, 1)    # conv1d takes in (batch, channels, seq_len)

        conv1d = self.conv(embedded).permute(0, 2, 1)         # permute bach to (batch, seq_len, channels)
        gmp_1d = conv1d.max(1)[0]                             # global max pooling

        x = F.dropout(gmp_1d, p=self.dropout_p)               # dropout, disabled when evaluating
        x = F.relu(self.fc1(x))                               # add a relu layer to the first fully connected layer
        x = self.fc2(x)                                       # last fully connected layer
        return x
