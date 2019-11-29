#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : convert_weight.py
# @Author: ruixi L
# @Date  : 2019/11/26


# 3.在temp下新建文件convert_weights.py,将1中存的pb文件数据导入，并采用checkpoint方式保存模型数据（共20分）
import tensorflow as tf
from model import cifar_cnn
from util import load_weight

# (1)定义变量并设置导入pb文件（采用绝对路径）（2.5分）
weight_path = r'./model.pb'
# (2)定义变量并设置存储模型文件（采用绝对路径）（2.5分）
save_path = r'./checkpoint/mymodel'
# (3)利用model.py中的类建立计算图（5分）
model = cifar_cnn(num_classes=10)
X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x')
logits = model.faword(X)
name_list = tf.trainable_variables()
with tf.Session() as sess:
    # (4)定义一种操作load_op,调用函数load_weight，给变量赋权重，参数为变量名列表和pb文件（2.5分）
    load_op = load_weight(name_list, weight_path)
    for i in range(len(load_op)):
        # (5)执行赋值操作（2.5分）
        sess.run(tf.assign(name_list[i], load_op[i]))
    # (6)利用checkpoint形式定义saver，并保存数据和模型，保存路径为当前目录的子目录mymodel下（5分）
    tf.train.Saver().save(sess, save_path)
    print('权重转换完成')
