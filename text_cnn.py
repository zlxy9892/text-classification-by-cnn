# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class TextCNN(object):
    '''
    A cnn for text classification.
    Structure: embedding layer -> convolutional layer -> max-pooling layer -> softmax layer.
    '''

    def __init__(
        self, num_classes, sequence_length, vocab_size,
        embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,
        pre_trained_embedding_matrix=None, device_name='/cpu:0'):

        # placeholder for the input, output and dropout
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

        # keep track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.device(device_name):

            with tf.name_scope('embedding'):
                # embedding layer
                # define W [vocab_size, embedding_size], W can put the vocabulary map to embedding (high dimension -> low dimension)
                if pre_trained_embedding_matrix is not None:
                    self.W = tf.Variable(initial_value=pre_trained_embedding_matrix, name='W')
                else:
                    self.W = tf.Variable(initial_value=tf.random_uniform(
                        shape=[vocab_size, embedding_size], minval=-1.0, maxval=1.0), name='W')
                self.embedded_words = tf.nn.embedding_lookup(params=self.W, ids=self.input_x)    # shape: [None, sequence_length, embedding_size]
                self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)    # add one dimension of channel. the final shape is 4 dimension: [None, sequence_length, embedding_size, 1]

            # create a convolutional + max-pooling layer for each filter size
            pooled_output = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope('cov-maxpool-%s' % filter_size):
                    # convolutional layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(shape=filter_shape, mean=0.0, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(value=0.1, shape=[num_filters]), name='b')
                    conv = tf.nn.conv2d(
                        input=self.embedded_words_expanded,
                        filter=W,
                        strides=[1,1,1,1],
                        padding='VALID',
                        name='conv')
                    # apply nonlinearity by activation function: Relu
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    # max-pooling over outputs
                    pooled = tf.nn.max_pool(
                        value=h,
                        ksize=[1,sequence_length-filter_size+1,1,1],
                        strides=[1,1,1,1],
                        padding='VALID',
                        name='pool')
                    pooled_output.append(pooled)    # the shape of the pooled output: [batch_size, 1, 1, num_filters]

            # combine all the pooled features
            num_filter_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(values=pooled_output, axis=3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filter_total])  # shape: [batch_size, num_filter_total]

            # add dropout
            with tf.name_scope('dropout'):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)

            # final output
            with tf.name_scope('output'):
                W = tf.Variable(tf.truncated_normal(shape=[num_filter_total, num_classes], mean=0.0, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(x=self.h_drop, weights=W, biases=b, name='scores_unnormalized')
                self.predictions = tf.argmax(input=self.scores, axis=1, name='predictions')

            # calculate mean cross-entropy loss
            with tf.name_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # calculate accuracy
            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
