#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: ruixi L
# @Date  : 2019/11/20


import tensorflow as tf
from util import next_batch
from model import Cnn_a
from tensorflow.python.framework import graph_util

tf.set_random_seed(666)
train_path = r'./a/my_hand/nametrain.txt'
test_path = r'./a/my_hand/nametest.txt'

model = Cnn_a(num_classes=3)
with open(train_path, 'r') as f:
    train_list = f.readlines()
with open(test_path, 'r') as f:
    test_list = f.readlines()

# 数据处理
train_img, train_label = model.data_scalar(train_list)
test_img, test_label = model.data_scalar(test_list)

# 数据洗牌
train_img = model.shuffle(train_img)
train_label = model.shuffle(train_label)
test_img = model.shuffle(test_img)
test_label = model.shuffle(test_label)

X = tf.placeholder(tf.float32, shape=[None, 58, 58])
Y = tf.placeholder(tf.int32, shape=[None, 3])
logits = model.farword(X)

cost = model.loss(logits, Y)
optimizer = tf.train.AdamOptimizer(model.learning_rate).minimize(cost)
predict = tf.argmax(logits, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(Y, 1)), dtype=tf.float32))
var_list = tf.trainable_variables()
name_list = [t.name.split(':')[0] for t in var_list]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(model.train_epochs):
        avg_cost = 0
        total_batch = int(train_img.shape[0] / model.batch_size)
        g_b = 0
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(train_img, train_label, g_b, model.batch_size)
            cost_val, acc, _ = sess.run([cost, accuracy, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / total_batch
        print('周期{} 代价值为{} 训练集准确率{}'.format(step + 1, avg_cost, acc))

    print('测试集准确率为', sess.run(accuracy, feed_dict={X: test_img, Y: test_label}))
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, name_list)
    with tf.gfile.FastGFile('./cnnmodel.pb', 'wb') as f:
        f.write(constant_graph.SerializeToString())
