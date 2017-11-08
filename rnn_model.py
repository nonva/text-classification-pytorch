#!/usr/bin/python
# -*- coding: utf-8 -*-


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TRNNConfig(object):
    """
    RNN Parameters
    """
    embedding_dim = 64  # embedding vector size
    seq_length = 600  # maximum length of sequence
    vocab_size = 5000  # most common words

    num_layers = 2  # number of rnn layers
    hidden_dim = 128  # hidden size of rnn and fully connected layer
    rnn_type = 'LSTM'  # LSTM or GRU

    dropout_prob = 0.2  # how much probability to be dropped
    learning_rate = 1e-3  # learning rate
    batch_size = 128  # batch size for training
    num_epochs = 10  # total number of epochs

    print_per_batch = 100  # print out the intermediate status every n batches

    num_classes = 10  # number of classes


class TextRNN(nn.Module):
    """
    RNN model for text classification.
    """

    def __init__(self, config):
        super(TextRNN, self).__init__()

        self.dropout_p = config.dropout_prob
        self.rnn_type = config.rnn_type
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)  # embedding layer
        self.rnn = getattr(nn, self.rnn_type)(config.embedding_dim,
                                              self.hidden_dim,
                                              self.num_layers,
                                              dropout=self.dropout_p)

        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim)  # fully connected layer
        self.fc2 = nn.Linear(config.hidden_dim, config.num_classes)  # classification layer

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs).permute(1, 0, 2)  # rnn takes in (seq_len, batch, embedding_dim)

        outputs, hidden = self.rnn(embedded, hidden)  # rnn layers
        last = outputs[-1]  # get the last output of each batch

        x = F.dropout(self.fc1(last), p=self.dropout_p)  # dropout, disabled when evaluating
        x = self.fc2(F.relu(x))  # last fully connected layer
        return x

    def init_hidden(self, batch_size):
        """
        Initial hidden states.
        For LSTM return (h0, c0).
        For GRU return h0.
        """
        weight = next(self.parameters()).data
        h = Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        if self.rnn_type == 'LSTM':  # (h0, c0)
            return h, h
        return h  # only h0
