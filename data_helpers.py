# -*- coding: utf-8 -*-

import numpy as np
import re
import gensim


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

def load_text_and_label(file_pos_file, file_neg_file):
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # load data from file
    file_pos = open(file=file_pos_file, mode='r', encoding='utf-8')
    file_neg = open(file=file_neg_file, mode='r', encoding='utf-8')
    pos_examples = list(file_pos.readlines())
    neg_examples = list(file_neg.readlines())
    pos_examples = [s.strip() for s in pos_examples]
    neg_examples = [s.strip() for s in neg_examples]
    # splite by word
    x_text = pos_examples + neg_examples
    x_text = np.array([clean_str(sentence) for sentence in x_text])
    # generate label (y)
    pos_labels = [[0,1] for _ in pos_examples]
    neg_labels = [[1,0] for _ in neg_examples]
    y = np.array(pos_labels + neg_labels)
    return [x_text, y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    '''
    Generates a batch iterator for a dataset.
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size)+1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num*batch_size
            end_index = min((batch_num+1)*batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_word_vectors(vocab_dict, word_vectors_filename, is_binary=True):
    try:
        # vocab_size: 71291
        word_vec_model = gensim.models.KeyedVectors.load_word2vec_format(word_vectors_filename, binary=is_binary)
        vocab = word_vec_model.vocab
        embedding_size = word_vec_model.vectors.shape[1]
        embedding_matrix = np.zeros(shape=(len(vocab_dict), embedding_size))    # shape: [vocab_size, embedding_size] for example: [71291, 200]
        not_include_count = 0
        for word, i in vocab_dict.items():
            try:
                word_vector = word_vec_model.word_vec(word)
            except:
                word_vector = np.random.randn(embedding_size)
                not_include_count += 1
            embedding_matrix[i] = word_vector
    except:
        return None, None
    prop_include_in_vocab = 1.0 - float(not_include_count) / len(vocab_dict.items())
    embedding_matrix = embedding_matrix.astype(np.float32)
    return embedding_matrix, prop_include_in_vocab
