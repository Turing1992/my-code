#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : util.py
# @Author: ruixi L
# @Date  : 2019/11/20


import tensorflow as tf


def conv2d(inputs, filter, strides=1, padding='SAME'):
    net = tf.nn.conv2d(inputs, filter=filter, strides=[1, strides, strides, 1], padding=padding)
    net = tf.nn.relu(net)
    return net


def pooling(inputs, ksize=2, strides=2, padding='SAME'):
    pool = tf.nn.max_pool(inputs, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding=padding)
    return pool


def fully_connected(inputs):
    if len(inputs.get_shape()) > 2:
        dim = inputs.get_shape()[1].value * inputs.get_shape()[2].value * inputs.get_shape()[3].value
        x_flat = tf.reshape(inputs, [-1, dim])
        w = tf.Variable(tf.random_normal([dim, 3]), name='w3')
        b = tf.Variable(tf.random_normal([3]), name='b')
        lo = tf.matmul(x_flat, w)
        logits = tf.add(lo, b, name='logits')
    else:
        dim = inputs.get_shape()[1].value
        x_flat = tf.reshape(inputs, [-1, dim])
        w = tf.Variable(tf.random_normal([dim, 3]), name='w3')
        b = tf.Variable(tf.random_normal([3]), name='b')
        lo = tf.matmul(x_flat, w)
        logits = tf.add(lo, b, name='logits')
    return logits


def next_batch(img, label, g_b, size):
    x = img[g_b:g_b + size]
    y = label[g_b:g_b + size]
    return x, y


def load_weight(var_list, weight_path):
    with tf.Session() as sess:
        with tf.gfile.FastGFile(weight_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            name_list = [j.name for j in var_list]

            values = tf.import_graph_def(graph_def, return_elements=name_list, name='')
            print(sess.run(values))
            print(len(values))
    return values
