#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : re_train.py
# @Author: ruixi L
# @Date  : 2019/11/20


import tensorflow as tf
from model import Cnn_a
from util import next_batch

tf.set_random_seed(666)

#载入数据
model=Cnn_a(num_classes=3)
train_path = r'./a/my_hand/nametrain.txt'
test_path = r'./a/my_hand/nametest.txt'

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

#恢复图
Y=tf.placeholder(tf.int32,shape=[None,3])
saver=tf.train.import_meta_graph('./checkpoint/mymodel-200.meta')
graph=tf.get_default_graph()
X=graph.get_tensor_by_name('x:0')
logits=graph.get_tensor_by_name('fc/logits:0')
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,axis=1),tf.argmax(Y,axis=1)),dtype=tf.float32))

with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('./checkpoint'))
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))
    optimizer1=tf.train.AdamOptimizer(0.001).minimize(cost)
    b=tf.trainable_variables()
    c=tf.global_variables()
    var=[]
    for i in c:
        if i in b:
            continue
        else:
            var.append(i)
    sess.run(tf.variables_initializer(var))
    for step in range(3):
        avg_cost=0
        total_batch=int(len(train_img)/model.batch_size)
        g_b = 0
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(train_img, train_label, g_b, model.batch_size)
            cost_val, acc, _ = sess.run([cost, accuracy, optimizer1], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / total_batch
        print('周期{} 代价值为{} 训练集准确率{}'.format(step + 1, avg_cost, acc))
