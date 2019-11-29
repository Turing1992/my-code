#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : util.py
# @Author: ruixi L
# @Date  : 2019/11/26


# 4.在temp下新建文件utils.py,在该文件下定义函数load_weight,参数为变量名列表和pb文件 （共15分）
# (1)导入依赖包（2.5分）
import tensorflow as tf
from tensorflow.python.platform import gfile


# (2)从pb文件中读入权重（2.5分）
def load_weight(var_list, weight_path):
    with tf.Session() as sess:
        with gfile.FastGFile(weight_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            # (3)利用变量名列表获取其对应的权重张量（2.5分）
            name_list = [t.name for t in var_list]
            values = tf.import_graph_def(graph_def, return_elements=name_list, name='')
    return values
    # (4)建立列表，用于存放权重赋值操作（2.5分）
    # (5)采用for循环的方式，将变量列表中的每个变量赋权重，并增加进上边列表中（2.5分）
    # 返回赋值操作列表（2.5分）
