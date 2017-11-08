#!/usr/bin/python
# -*- coding: utf-8 -*-


import torch.nn as nn
import torch.nn.functional as F


class TCNNConfig(object):
    """
    CNN Parameters
    """
    embedding_dim = 64      # embedding vector size
    seq_length = 600        # maximum length of sequence
    vocab_size = 5000      # most common words

    num_filters = 256        # number of convolution filters
    kernel_size = 5         # kernel size

    hidden_dim = 128        # hidden size of fully connected layer

    dropout_prob = 0.2      # how much probability to be dropped
    learning_rate = 1e-3    # learning rate
    batch_size = 128        # batch size for training
    num_epochs = 10         # total number of epochs

    print_per_batch = 100   # print out the intermediate status every n batches

    num_classes = 10        # number of classes


class TextCNN(nn.Module):
    """
    CNN model for text classification.
    """
    def __init__(self, config):
        super(TextCNN, self).__init__()

        self.dropout_p = config.dropout_prob

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)  # embedding layer
        self.conv1 = nn.Conv1d(config.embedding_dim, config.num_filters, config.kernel_size)  # conv1d layer
        self.conv2 = nn.Conv1d(config.num_filters, 128, 5)

        self.fc1 = nn.Linear(128, config.hidden_dim)   # fully connected layer
        self.fc2 = nn.Linear(config.hidden_dim, config.num_classes)   # classification layer

    def forward(self, inputs):
        embedded = self.embedding(inputs).permute(0, 2, 1)    # conv1d takes in (batch, channels, seq_len)

        conv1d = self.conv1(embedded)         # permute bach to (batch, seq_len, channels)
        # print(conv1d.size())
        conv1d = self.conv2(conv1d).permute(0, 2, 1)
        # print(conv1d.size())
        gmp_1d = conv1d.max(1)[0]                             # global max pooling

        x = F.dropout(self.fc1(gmp_1d), p=self.dropout_p)     # dropout, disabled when evaluating
        x = self.fc2(F.relu(x))                               # last fully connected layer
        return x
