#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import numpy as np
import re
import csv
import os


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def open_file(filename, mode='r'):
    """
    Commonly used file reader and writer, change this to switch between python2 and python3.
    :param filename: filename
    :param mode: 'r' and 'w' for read and write respectively
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    """
    Read data from csv file.
    """
    data, labels = [], []
    with open_file(filename) as csv_file:
        fieldnames = ['category', 'title', 'content']
        reader = csv.DictReader(csv_file, fieldnames=fieldnames)
        for row in reader:
            data.append(clean_str(row['content'].strip()).split())
            labels.append(int(row['category']) - 1)
    return data, labels


def build_vocab(train_dir, vocab_dir, vocab_size=50000):
    """
    Build vocabulary file from training data.
    """
    print('Building vocabulary...')
    data, _ = read_file(train_dir)  # read training data

    all_data = []  # group all data
    for content in data:
        all_data.extend(content)

    counter = Counter(all_data)  # count and get the most common words
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))

    words = ['<PAD>'] + list(words)  # add a padding with id 0 to pad the sentence to same length
    open_file(vocab_dir, 'w').write('\n'.join(words) + '\n')


class Dictionary(object):
    """
    Used for processing vocabulary and category.
    """

    def __init__(self, vocab_dir, class_dir):
        if not os.path.exists(vocab_dir):
            raise FileNotFoundError("Use build_vocab() to generate vocabulary file.")

        self.words = open_file(vocab_dir).read().strip().split('\n')  # vocabularies
        self.word_to_id = dict(zip(self.words, range(len(self.words))))

        self.categories = open_file(class_dir).read().strip().split('\n')  # categories
        self.cat_to_id = dict(zip(self.categories, range(len(self.categories))))

    def __len__(self):
        return len(self.words)

    def __str__(self):
        return 'Vocabulary: %d' % len(self.words)

    def to_words(self, sequence):
        """
        Read words from given sequence of ids
        :param sequence: sequence of ids, i.e. [1, 3, 29, 6, 24]
        """
        return ' '.join(self.words[x] for x in sequence)




class Corpus(object):
    def __init__(self, filename, dictionary, max_length=150, dev_split=0.1, training=True):
        self.dictionary = dictionary
        self.training = training

        x_data, y_data = read_file(filename)  # loading data

        w2x = self.dictionary.word_to_id
        for i in range(len(x_data)):  # tokenizing
            content = [w2x[x] for x in x_data[i] if x in w2x]
            if len(content) < max_length:
                content = [0] * (max_length - len(content)) + content
            x_data[i] = content[:max_length]

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        indices = np.random.permutation(np.arange(len(x_data)))  # shuffle
        x_data = x_data[indices]
        y_data = y_data[indices]

        if training:
            num_train = int((1 - dev_split) * len(x_data))
            self.x_train = x_data[:num_train]
            self.y_train = y_data[:num_train]
            self.x_val = x_data[num_train:]
            self.y_val = y_data[num_train:]
        else:
            self.x_test = x_data
            self.y_test = y_data

    def __str__(self):
        if self.training:
            return 'Training: {}, Validation: {}'.format(len(self.x_train), len(self.x_val))
        return 'Testing: {}'.format(len(self.x_test))
