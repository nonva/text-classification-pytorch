#!/usr/bin/python
# -*- coding: utf-8 -*-

from cnn_model import *
from configuration import *
import time
from datetime import timedelta
from data.cnews_loader import *

import torch.optim as optim

def run_epoch(cnn=True):
    # 载入数据
    print('Loading data...')
    start_time = time.time()

    if not os.path.exists('data/cnews/vocab_cnews.txt'):
        _build_vocab('data/cnews/cnews.train.txt', vocab_size=5000)
    x_train, y_train, x_test, y_test, x_val, y_val, words = preocess_file()

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    batch_train = batch_iter(list(zip(x_train, y_train)),
        config.batch_size, config.num_epochs)

    # 训练与验证
    print('Training and evaluating...')
    start_time = time.time()
    running_loss = 0
    for i, batch in enumerate(batch_train):
        x_batch, y_batch = zip(*batch)
        inputs = Variable(torch.LongTensor(np.array(x_batch).tolist()))
        labels = list(map(lambda x: x, np.argmax(np.array(y_batch), 1).tolist()))

        labels = Variable(torch.LongTensor(labels))

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if i % 20 == 19:
            print('[%5d] loss: %.3f' %
                  (i + 1, running_loss / 20))
            running_loss = 0.0

    #
    # # 最后在测试集上进行评估
    # print('Evaluating on test set...')
    # loss_test, acc_test = evaluate(x_test, y_test)
    # msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    # print(msg.format(loss_test, acc_test))
    #
    # session.close()

if __name__ == '__main__':
    run_epoch(cnn=True)
