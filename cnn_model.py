#!/usr/bin/python
# -*- coding: utf-8 -*-


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from configuration import *

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()

        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.num_filters = config.num_filters
        self.kernel_size = config.kernel_size
        self.seq_length = config.seq_length
        self.hidden_dim = config.hidden_dim
        self.num_classes = config.num_classes

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.conv = nn.Conv1d(self.embedding_dim, self.num_filters, self.kernel_size)

        self.fc1 = nn.Linear(self.num_filters, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, input):
        embedding = self.embedding(input).permute(0, 2, 1)
        conv1d = self.conv(embedding).permute(0, 2, 1)
        global_max_pooling_1d = conv1d.max(1)[0]
        x = F.dropout(global_max_pooling_1d)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
