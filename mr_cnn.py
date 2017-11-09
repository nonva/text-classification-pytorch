#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This example demonstrates the use of Conv1d for CNN text classification.
Original paper could be found at: https://arxiv.org/abs/1408.5882

This is the baseline model: CNN-rand.

We didn't implement cross validation,
but simply run `python mr_cnn.py` for multiple times,
the average accuracy is close to 76%.

It tooks about 2 minutes for training 20 epochs on a GTX 970 GPU.
"""

__author__ = 'gaussic'

import os
import time
from datetime import timedelta

import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from data_helper.mr_loader import Corpus, read_vocab, process_text

use_cuda = torch.cuda.is_available()

base_dir = 'data/mr'
pos_file = os.path.join(base_dir, 'rt-polarity.pos.txt')
neg_file = os.path.join(base_dir, 'rt-polarity.neg.txt')
vocab_file = os.path.join(base_dir, 'rt-polarity.vocab.txt')

save_path = 'models'  # model save path
if not os.path.exists(save_path):
    os.mkdir(save_path)
model_file = os.path.join(save_path, 'mr_cnn.pt')


class TCNNConfig(object):
    """
    CNN Parameters
    """
    embedding_dim = 128  # embedding vector size
    seq_length = 50  # maximum length of sequence
    vocab_size = 8000  # most common words

    num_filters = 100  # number of the convolution filters (feature maps)
    kernel_sizes = [3, 4, 5]   # three kind of kernels (windows)

    hidden_dim = 128  # hidden size of fully connected layer

    dropout_prob = 0.5  # how much probability to be dropped
    learning_rate = 1e-3  # learning rate
    batch_size = 50  # batch size for training
    num_epochs = 20  # total number of epochs

    print_per_batch = 100  # print out the intermediate status every n batches

    num_classes = 2  # number of classes

    dev_split = 0.1  # percentage of dev data


class TextCNN(nn.Module):
    """
    CNN model for text classification.
    """

    def __init__(self, config):
        super(TextCNN, self).__init__()

        V = config.vocab_size
        E = config.embedding_dim
        Nf = config.num_filters
        Ks = config.kernel_sizes
        C = config.num_classes
        drop = config.dropout_prob

        self.embedding = nn.Embedding(V, E)  # embedding layer

        # three convolution layers
        self.convs = nn.ModuleList([nn.Conv1d(E, Nf, k) for k in Ks])
        self.fc1 = nn.Linear(3 * Nf, C)  # fully connected layer
        self.dropout = nn.Dropout(drop)

    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward(self, inputs):
        # conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        embedded = self.embedding(inputs).permute(0, 2, 1)

        # convolution and global max pooling
        x = [self.conv_and_max_pool(embedded, k) for k in self.convs]
        x = torch.cat((x), 1)  # concatenation

        x = self.dropout(x)  # dropout, disabled when evaluating
        x = self.fc1(x)  # last fully connected layer
        return x


def get_time_dif(start_time):
    """
    Return the time used since start_time.
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(model, data):
    """
    Evaluation on a given data.
    """
    model.eval()  # set mode to evaluation to disable dropout
    data_loader = DataLoader(data, batch_size=256)

    data_len = len(data)
    total_loss = 0.0
    total_acc = 0.0
    total_pred = []
    for x_batch, y_batch in data_loader:
        inputs, targets = Variable(x_batch, volatile=True), Variable(y_batch, volatile=True)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets, size_average=False)
        total_loss += loss.data[0]

        _, pred = torch.max(outputs.data, 1)
        total_pred.extend(pred.cpu().numpy().tolist())

        total_acc += (pred == targets.data).sum()

    model.train()  # set mode back to train
    return total_loss / data_len, total_acc / data_len, np.array(total_pred)


def train():
    """
    Train and evaluate the model with training and validation data.
    """
    print('Loading data...')
    start_time = time.time()
    config = TCNNConfig()
    corpus = Corpus(pos_file, neg_file, vocab_file, config.dev_split, config.seq_length, config.vocab_size)
    print(corpus)
    config.vocab_size = len(corpus.words)

    train_data = TensorDataset(torch.LongTensor(corpus.x_train), torch.LongTensor(corpus.y_train))
    val_data = TensorDataset(torch.LongTensor(corpus.x_test), torch.LongTensor(corpus.y_test))

    print('Configuring CNN model...')
    model = TextCNN(config)
    print(model)

    if use_cuda:
        model.cuda()

    # optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # set the mode to train
    model.train()

    total_batch = 0
    total_loss = 0.0
    best_acc_val = 0.0
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        # load the training data in batch
        train_loader = DataLoader(train_data, batch_size=config.batch_size)
        for x_batch, y_batch in train_loader:
            inputs, targets = Variable(x_batch), Variable(y_batch)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)  # forward computation
            loss = criterion(outputs, targets)
            total_loss += loss.data[0]
            total_batch += 1

            if total_batch % config.print_per_batch == 0:
                # print out intermediate status
                avg_loss = total_loss / config.print_per_batch
                total_loss = 0.0

                _, pred_train = torch.max(outputs.data, 1)
                corrects = (pred_train == targets.data).sum()
                acc_train = corrects / len(x_batch)
                loss_val, acc_val, _ = evaluate(model, val_data)  # evaluate on val data

                if acc_val > best_acc_val:
                    # store the best validation result
                    best_acc_val = acc_val
                    improved_str = '*'
                    torch.save(model.state_dict(), model_file)
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, avg_loss, acc_train, loss_val, acc_val, time_dif, improved_str))

            # back propagation
            loss.backward()
            optimizer.step()

    test(model, val_data)


def test(model, test_data):
    """
    Test the model on test dataset.
    """
    start_time = time.time()
    model.load_state_dict(torch.load(model_file))  # restore the parameters

    print("Testing...")
    loss_test, acc_test, total_pred = evaluate(model, test_data)
    print('Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'.format(loss_test, acc_test))

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(test_data.target_tensor.numpy(), total_pred, target_names=['POS', 'NEG']))

    print('Confusion Matrix...')
    cm = metrics.confusion_matrix(test_data.target_tensor.numpy(), total_pred)
    print(cm)

    print("Time usage:", get_time_dif(start_time))


def predict(text):
    # load config and vocabulary
    config = TCNNConfig()
    _, word_to_id = read_vocab(vocab_file)
    labels = ['POS', 'NEG']

    # load model
    model = TextCNN(config)
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
    # if use_cuda:
    #     model.cuda()

    # process text
    text = process_text(text, word_to_id, config.seq_length)
    text = Variable(torch.LongTensor([text]), volatile=True)
    if use_cuda:
        text = text.cuda()

    # predict
    model.eval()   # very important
    output = model(text)
    _, pred = torch.max(output, 1)

    return labels[pred.data[0]]


if __name__ == '__main__':
    # train()
    print(predict('this film is good'))
    print(predict('this film is bad'))
