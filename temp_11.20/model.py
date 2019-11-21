#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: ruixi L
# @Date  : 2019/11/20


import tensorflow as tf
import numpy as np
from PIL import Image
from util import conv2d, pooling, fully_connected


class Cnn_a(object):
    def __init__(self, num_classes, learning_rate=0.001, train_epochs=100, batch_size=20):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.batch_size = batch_size

    def farword(self, inputs_image):
        inputs = tf.reshape(inputs_image, [-1, 58, 58, 1])
        with tf.variable_scope('conv1'):
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 16]), name='w1')
            net = conv2d(inputs, W1)
            net = pooling(net)
        with tf.variable_scope('conv2'):
            W2 = tf.Variable(tf.random_normal([3, 3, 16, 16]), name='w2')
            net = conv2d(net, W2)
            net = pooling(net)
        with tf.variable_scope('fc'):
            logits = fully_connected(net)
        return logits

    def loss(self, logits, Y):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
        return cost

    def data_scalar(self, data_list):
        img_list = []
        label_list = []
        for i in data_list:
            i = i[:-1]
            if i[16] == '0':
                label_list.append([1, 0, 0])
            elif i[16] == '2':
                label_list.append([0, 1, 0])
            elif i[16] == '5':
                label_list.append([0, 0, 1])
            img = Image.open(i)
            img = np.array(img)
            img_list.append(img)
        img_list = np.squeeze(img_list)
        label_list = np.squeeze(label_list)
        return img_list, label_list

    def shuffle(self, data):
        np.random.seed(0)
        order = np.random.permutation(len(data))
        return data[order]
