#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from cnn_model import TextCNN
from data.mr_loader import Corpus

use_cuda = torch.cuda.is_available()  # if True, use GPU

pos_file = 'data/mr/rt-polarity.pos.txt'
neg_file = 'data/mr/rt-polarity.neg.txt'

save_path = 'models'  # model save path
if not os.path.exists(save_path):
    os.mkdir(save_path)
model_file = os.path.join(save_path, 'cnn_mr.pt')


class TCNNConfig(object):
    """
    CNN Parameters
    """
    embedding_dim = 64  # embedding vector size
    seq_length = 50  # maximum length of sequence
    vocab_size = 8000  # most common words

    num_filters = 256  # number of convolution filters
    kernel_size = 5  # kernel size

    hidden_dim = 128  # hidden size of fully connected layer

    dropout_prob = 0.2  # how much probability to be dropped
    learning_rate = 1e-3  # learning rate
    batch_size = 32  # batch size for training
    num_epochs = 10  # total number of epochs

    print_per_batch = 100  # print out the intermediate status every n batches

    num_classes = 2  # number of classes


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
    model.eval()  # set mode to evaluation
    data_loader = DataLoader(data, batch_size=256)

    data_len = len(data)
    total_loss = 0.0
    total_acc = 0.0
    total_pred = []
    for x_batch, y_batch in data_loader:
        inputs, targets = Variable(x_batch), Variable(y_batch)
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
    corpus = Corpus(pos_file, neg_file, 0.1, config.seq_length, config.vocab_size)
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
    last_improved = 0
    require_improvement = 10000
    flag = False  # if True, stop training
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
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
                    last_improved = total_batch
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

            if total_batch - last_improved > require_improvement:
                # early stopping
                print('No optimization for a long time, auto-stopping...')
                flag = True
                break
        if flag:
            # early stopping
            break

    test(model, val_data)


def test(model, test_data):
    """
    Test the model on test dataset.
    """
    start_time = time.time()
    model.load_state_dict(torch.load(model_file))

    print("Testing...")
    loss_test, acc_test, total_pred = evaluate(model, test_data)  # evaluate on test data
    print('Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'.format(loss_test, acc_test))

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(test_data.target_tensor.numpy(), total_pred, target_names=['POS', 'NEG']))

    print('Confusion Matrix...')
    cm = metrics.confusion_matrix(test_data.target_tensor.numpy(), total_pred)
    print(cm)

    print("Time usage:", get_time_dif(start_time))


if __name__ == '__main__':
    train()