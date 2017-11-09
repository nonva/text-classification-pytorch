#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import numpy as np
import re


def open_file(filename, mode='r'):
    """
    Commonly used file reader and writer, change this to switch between python2 and python3.
    :param filename: filename
    :param mode: 'r' and 'w' for read and write respectively
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


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


class Corpus(object):
    def __init__(self, pos_file, neg_file, dev_split=0.1, max_length=50, vocab_size=5000):
        # loading data
        pos_examples = [clean_str(s.strip()).split() for s in open_file(pos_file)]
        neg_examples = [clean_str(s.strip()).split() for s in open_file(neg_file)]
        x_data = pos_examples + neg_examples
        y_data = [0] * len(pos_examples) + [1] * len(neg_examples)

        # vocabulary
        all_data = []  # group all data
        for content in x_data:
            all_data.extend(content)

        counter = Counter(all_data)  # count and get the most common words
        count_pairs = counter.most_common(vocab_size - 1)
        words, _ = list(zip(*count_pairs))

        self.words = ['<PAD>'] + list(words)  # add a padding with id 0 to pad the sentence to same length
        self.word_to_id = dict(zip(words, range(len(words))))

        for i in range(len(x_data)):  # tokenizing and padding
            content = [self.word_to_id[x] for x in x_data[i] if x in self.word_to_id]
            if len(content) < max_length:
                content = [0] * (max_length - len(content)) + content
            x_data[i] = content[:max_length]

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        indices = np.random.permutation(np.arange(len(x_data)))  # shuffle
        x_data = x_data[indices]
        y_data = y_data[indices]

        num_train = int((1 - dev_split) * len(x_data)) # train/dev split
        self.x_train = x_data[:num_train]
        self.y_train = y_data[:num_train]
        self.x_test = x_data[num_train:]
        self.y_test = y_data[num_train:]

    def __str__(self):
        return 'Training: {}, Testing: {}, Vocabulary: {}'.format(len(self.x_train), len(self.x_test), len(self.words))
