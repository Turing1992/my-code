#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: ruixi L
# @Date  : 2019/11/26

import tensorflow as tf
# 2.在temp下新建文件model.py，定义类cifar_cnn（共30分）
class cifar_cnn(object):
    # (1)定义构造函数，参数为 学习率、训练周期、训练每批样本数，应设置缺省值（5分）
    def __init__(self,num_classes,learning_rate=0.001,training_epochs=3,batch_size=100):
        self.num_classes=num_classes
        self.learning_rate=learning_rate
        self.training_epochs=training_epochs
        self.batch_size=batch_size

    # (2)定义前向传播函数，里面封装模型计算图，该函数输入为X (placehoder类型)，
    # 里面包含2层卷积网络，一层全连接网络，返回logits（共20分，函数定义正确（包含有返回）5分，每层网络5分）
    def faword(self,inputs):
        W1 = tf.Variable(tf.random_normal([3, 3, 3, 32]), name='w1')
        L1 = tf.nn.conv2d(inputs, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01), name='w2')
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        dim = L2.get_shape()[1].value * L2.get_shape()[2].value * L2.get_shape()[3].value
        L2_flat = tf.reshape(L2, [-1, dim])
        W3 = tf.get_variable("w3", shape=[dim, 10], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.random_normal([10]), name='b')
        logits = tf.matmul(L2_flat, W3)
        logits = tf.add(logits, b, name='logits')
        return logits

    # (3)定义损失函数，输入参数为预测值和真实值（5分）
    def loss(self,logits,Y):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        return cost