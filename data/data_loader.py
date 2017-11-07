#!/usr/bin/python
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
from collections import Counter
import numpy as np
import os


def open_file(filename, mode='r'):
    """
    Commonly used file reader and writer, change this to switch between python2 and python3.
    :param filename: filename
    :param mode: 'r' and 'w' for read and write respectively
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    """
    Read data and labels from file.
    """
    data, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                data.append(list(content))
                labels.append(label)
            except ValueError:
                pass
    return data, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """
    Build vocabulary file from training data.
    """
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

    def to_words(self, sequence):
        """
        Read words from given sequence of ids
        :param sequence: sequence of ids, i.e. [1, 3, 29, 6, 24]
        """
        return ' '.join(self.words[x] for x in sequence)


class TextDataset(Dataset):
    """
    Used for loading text data.
    This is further used by torch.utils.data.DataLoader as data iterator.
    """

    def __init__(self, filename, dictionary, max_length=200):
        """
        Load data from text file, and tokenize with given vocabulary list.
        """
        self.dictionary = dictionary
        self.x_data, self.y_data = self.corpus(filename, max_length)

    def corpus(self, filename, max_length):
        """
        Tokenize the data to sequence of ids, and pad to fixed length.
        """
        x_data, y_data = read_file(filename)

        word_to_id = self.dictionary.word_to_id
        for i in range(len(x_data)):
            content = [word_to_id[x] for x in x_data[i] if x in word_to_id]
            if len(content) < max_length:
                content = [0] * (max_length - len(content)) + content

            x_data[i] = content[:max_length]
            y_data[i] = self.dictionary.cat_to_id[y_data[i]]

        x_data, y_data = np.array(x_data), np.array(y_data)   # data needs to be shuffled
        indices = np.random.permutation(np.arange(len(x_data)))
        x_data, y_data = x_data[indices], y_data[indices]

        return x_data, y_data

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.x_data.shape[0]
