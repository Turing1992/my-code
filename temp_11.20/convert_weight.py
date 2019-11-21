#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : convert_weight.py
# @Author: ruixi L
# @Date  : 2019/11/20


import tensorflow as tf
from model import Cnn_a
from util import load_weight


weight_path=r'./cnnmodel.pb'
X=tf.placeholder(tf.float32,shape=[None,58,58],name='x')
model=Cnn_a(num_classes=3)
logits=model.farword(X)
name_list=tf.trainable_variables()
print(len(name_list))
with tf.Session() as sess:
    load_op=load_weight(name_list,weight_path)
    for i in range(len(load_op)):
        sess.run(tf.assign(name_list[i],load_op[i]))
    tf.train.Saver().save(sess,'./checkpoint/mymodel',global_step=200)
    print('权重转换完成')