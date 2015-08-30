import cPickle
import gzip
import os
import sys
import re

import numpy
import theano

import pandas
from sklearn.feature_extraction.text import CountVectorizer
import pdb

# def build_from_csv(csv_path):
#     """
#     Builds the pkl file from csv_path

#     CSV format:
#     sentence, label
#     """
#     with open(csv_path, 'rb'):

datapath = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'),'data'),'my_data')

def prepare_data(seqs, labels, maxlen=None):
    """
    Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # seqs: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def split_sentence(sent):
    """
    Splits sentence into words and punctuations
    """
    return re.findall(r"[\w']+|[.,!?;]", sent)

def check_dir_files(dataset_path):
    """
    Check if directory exists or train.txt + test.txt files present
    """
    # check if the directory is present
    if not os.path.exists(dataset_path):
        print "The dataset path does not exist"
        sys.exit(1)
    # check if all files are available
    filenames = ['train.txt', 'test.txt']
    filepaths = []
    for f in filenames:
        filepath = os.path.join(dataset_path,f)
        if not os.path.isfile(filepath):
            print "%s does not exist in the dataset path given" % f
            sys.exit(1)
        filepaths.append(filepath)
    return filepaths

def build_dict(dataset_path):
    """
    If 'dictionary.pkl' not present,
    Build and return dictionary with all words + indices
    else,
    Return dictionary from 'dictionary.pkl'
    """
    dict_path = os.path.join(dataset_path, 'dictionary.pkl')
    filepaths = check_dir_files(dataset_path)
    if os.path.isfile(dict_path):
        dictionary = cPickle.load(dict_path)
        return dictionary

    print "dictionary.pkl file not found - building dictionary..."
    all_sents = []
    for f in filepaths:
        df = pandas.read_csv(f)
        sentences = list(df.ix[:,0])
        all_sents = all_sents + sentences

    vectorizer = CountVectorizer().fit(all_sents)
    dictionary = vectorizer.vocabulary_
    dictionary_series = pandas.Series(dictionary.values(), index=dictionary.keys()) + 2
    dictionary_series.sort(axis=1, ascending=False)
    dictionary = list(dictionary_series.index)

    # write dictionary to pkl file
    dictionary_path = os.path.join(dataset_path, 'dictionary.pkl')
    cPickle.dump(dictionary, open(dictionary_path, 'wb'))

    # return dictionary
    return dictionary

def get_dataset_file(csv_path_dir):
    """
    Build the proper dataset pkl file from CSV

    csv format:
    sentence, label
    """
    filepaths = check_dir_files(csv_path_dir)

    dictionary = build_dict(csv_path_dir)
    result = []
    for fidx, f in enumerate(filepaths):
        # Read csv and write to pkl file
        df = pandas.read_csv(f, sep=',')
        sentences = list(df['sentence'])
        labels = list(df['label'])
        for idx, sent in enumerate(sentences):
            sent_vect = []
            for wrd in split_sentence(sent):
                sent_vect.append(dictionary.index(wrd))
            sentences[idx] = sent_vect

        result.append((sentences, labels))

    output_path = os.path.join(csv_path_dir, 'my_data.pkl')
    output_file = open(output_path, 'wb')
    for r in result:
        cPickle.dump(r, output_file)

    return output_path


def load_data(path=os.path.join(datapath, 'my_data.pkl'), n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    '''
    Loads the dataset

    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

    # raise File not found error if not present
    if not os.path.isfile(path):
        # build the .pkl file from the csv file
        data_dir, data_file = os.path.split(path)
        get_dataset_file(data_dir)

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set = cPickle.load(f)
    test_set = cPickle.load(f)
    f.close()
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test