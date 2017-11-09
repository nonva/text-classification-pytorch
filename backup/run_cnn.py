import os
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from backup.cnn_model import TCNNConfig, TextCNN
from backup.data_loader import Dictionary, TextDataset, build_vocab

use_cuda = torch.cuda.is_available()  # if True, use GPU

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
class_dir = os.path.join(base_dir, 'cnews.classes.txt')

save_path = 'models'  # model save path
if not os.path.exists(save_path):
    os.mkdir(save_path)
model_file = os.path.join(save_path, 'textcnn.pt')


def get_time_dif(start_time):
    """
    Return the time used since start_time.
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(data):
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
    print('Loading training and validation data...')
    start_time = time.time()
    train_data = TextDataset(train_dir, dictionary, max_length=config.seq_length)
    val_data = TextDataset(val_dir, dictionary, max_length=config.seq_length)
    print('Time usage:', get_time_dif(start_time))

    # optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # set the mode to train
    model.train()

    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 1000
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

            if total_batch % config.print_per_batch == 0:
                # print out intermediate status
                _, pred_train = torch.max(outputs.data, 1)
                corrects = (pred_train == targets.data).sum()
                acc_train = corrects / len(x_batch)
                loss_val, acc_val, _ = evaluate(val_data)  # evaluate on validation data

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
                print(msg.format(total_batch, loss.data[0], acc_train, loss_val, acc_val, time_dif, improved_str))

            # back propagation
            loss.backward()
            optimizer.step()
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # early stopping
                print('No optimization for a long time, auto-stopping...')
                flag = True
                break
        if flag:
            # early stopping
            break


def test():
    """
    Test the model on test dataset.
    """
    start_time = time.time()
    model.load_state_dict(torch.load(model_file))  # restore the parameters

    print("Loading test data...")
    test_data = TextDataset(test_dir, dictionary, max_length=config.seq_length)

    print("Testing...")
    loss_test, acc_test, total_pred = evaluate(test_data)  # evaluate on test data
    print('Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'.format(loss_test, acc_test))

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(test_data.y_data, total_pred, target_names=dictionary.categories))

    print('Confusion Matrix...')
    cm = metrics.confusion_matrix(test_data.y_data, total_pred)
    print(cm)

    print("Time usage:", get_time_dif(start_time))


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        print("""Usage: python run_cnn.py [train / test]""")
        exit(1)

    print('Configuring CNN model...')
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir, vocab_dir, config.vocab_size)

    dictionary = Dictionary(vocab_dir, class_dir)
    config.vocab_size = len(dictionary)

    model = TextCNN(config)
    print(model)

    if use_cuda:
        model.cuda()

    if sys.argv[1] == 'train':
        train()
    else:
        test()
