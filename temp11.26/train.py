#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: ruixi L
# @Date  : 2019/11/26


# 5.在temp下新建文件train.py,冻结两个卷积层继续训练，
# 利用2中定义的类实现训练功能，相比于cifar.py中的训练过程，应有如下不同：（25分）
# (1)引入model.py中的类依赖包（5分）
import tensorflow as tf
from model import cifar_cnn
import numpy as np
import pickle

tf.set_random_seed(777)  # 设置随机种子
# 获取数据集
fo = open(r'data', 'rb')
dict = pickle.load(fo, encoding='bytes')
fo.close()
imgArr = dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255
total = imgArr.shape[0]
Y_one_hot = np.eye(10)[dict[b'labels']]
# 训练集的比例
train_test = 0.9

g_b = 0


# 自己实现next_batch函数，每次返回一批数据
def next_batch(size):
    global g_b
    xb = imgArr[g_b:g_b + size]
    yb = Y_one_hot[g_b:g_b + size]
    g_b = g_b + size
    return xb, yb


# (2)利用model.py中定义的类定义计算图(5分)
model = cifar_cnn(num_classes=10)
Y = tf.placeholder(tf.float32, [None, 10], name='y')
saver = tf.train.import_meta_graph('./checkpoint/mymodel.meta')
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('x:0')
logits = graph.get_tensor_by_name('logits:0')
# 准确率
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # (3)采用指定变量列表的方式冻结两个卷积层（5分）
    # (4)利用前面saver存的模型数据恢复模型（5分）
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    # 可训练变量
    train_var = tf.trainable_variables()
    no_frezen = [t for t in train_var if not t.name.startswith('w1')]
    optimizer1 = tf.train.AdamOptimizer(model.learning_rate)
    train_op = optimizer1.compute_gradients(cost, no_frezen)
    train_op = optimizer1.apply_gradients(train_op)
    # optimizer1=tf.train.AdamOptimizer(model.learning_rate).minimize(cost)
    # 全局变量
    var = []
    global_var = tf.global_variables()
    for i in global_var:
        if i in train_var:
            continue
        else:
            var.append(i)
    sess.run(tf.initialize_variables(var_list=var))

    print('开始继续学习')
    # 继续训练2个epoch,并输出损失和准确率（5）
    for epoch in range(2):
        avg_cost = 0
        total_batch = int(total * train_test / model.batch_size)  # 计算总批次
        g_b = 0
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(model.batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, train_op], feed_dict=feed_dict)
            avg_cost += c / total_batch
        acc = sess.run(accuracy,
                       feed_dict={X: imgArr[int(total * train_test):], Y: Y_one_hot[int(total * train_test):]})
        print('Epoch:', (epoch + 1), 'cost =', avg_cost, 'acc=', acc)
    print('学习完成')
