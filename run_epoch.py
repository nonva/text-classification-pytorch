#!/usr/bin/python
# -*- coding: utf-8 -*-

from cnn_model import *
from configuration import *
import time
from datetime import timedelta
from data.cnews_loader import *

import torch.optim as optim

use_cuda = torch.cuda.is_available()

def train(model, learning_rate, batch_train, val_data):

    if use_cuda:
        model.cuda()

    # 优化器与损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练与验证
    print('Training and evaluating...')
    model.train()
    start_time = time.time()
    running_loss = 0.0
    for i, batch in enumerate(batch_train):
        x_batch, y_batch = batch
        inputs = Variable(torch.LongTensor(x_batch))
        labels = Variable(torch.LongTensor(y_batch))

        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % 100 == 99:
            avg_loss = running_loss / 100
            running_loss = 0.0

            pred_train = torch.max(outputs.data, 1)[1]
            corrects = (pred_train == labels.data).sum()
            acc_train = corrects / len(x_batch)

            val_loss, val_acc = evaluate(model, val_data)

            # 时间
            end_time = time.time()
            time_dif = end_time - start_time
            time_dif = timedelta(seconds=int(round(time_dif)))

            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
                + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5}'
            print(msg.format(i + 1, avg_loss, acc_train, val_loss, val_acc, time_dif))


def evaluate(model, eval_data):

    batch_eval = batch_iter(list(zip(eval_data[0], eval_data[1])), 128, 1)

    model.eval()
    corrects, avg_loss, batch_num = 0, 0, 0
    for batch in batch_eval:
        x_batch, y_batch = batch
        inputs = Variable(torch.LongTensor(x_batch))
        labels = Variable(torch.LongTensor(y_batch))

        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels, size_average=False)

        avg_loss += loss.data[0]
        pred_train = torch.max(outputs.data, 1)[1]
        corrects += (pred_train == labels.data).sum()
        batch_num += len(x_batch)

    avg_loss /= batch_num
    corrects /= batch_num
    model.train()
    return avg_loss, corrects


def run_epoch(cnn=True):
    print('Loading data...')
    start_time = time.time()

    if not os.path.exists('data/cnews/vocab_cnews.txt'):
        _build_vocab('data/cnews/cnews.train.txt', vocab_size=5000)
    train_data, test_data, val_data, words = preocess_file()

    if cnn:
        print('Using CNN model...')
        config = TCNNConfig()
        config.vocab_size = len(words)
        model = TextCNN(config)
    else:
        print('Using RNN model...')
        config = TRNNConfig()
        config.vocab_size = len(words)
        model = TextRNN(config)

    end_time = time.time()
    time_dif = end_time - start_time
    time_dif = timedelta(seconds=int(round(time_dif)))
    print('Time usage:', time_dif)

    # 生成批次数据
    print('Generating batch...')
    batch_train = batch_iter(list(zip(train_data[0], train_data[1])),
        config.batch_size, config.num_epochs)

    train(model, config.learning_rate, batch_train, val_data)

    loss_test, acc_test = evaluate(model, test_data)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))



if __name__ == '__main__':
    run_epoch(cnn=True)
