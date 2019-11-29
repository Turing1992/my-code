# 卷积神经网络，从cifar10的文件中获取数据
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle

#(1)增加pb文件存储的依赖包（2.5分）
from tensorflow.python.framework import graph_util

tf.set_random_seed(777) #设置随机种子
# 获取数据集
fo = open(r'data', 'rb')
dict = pickle.load(fo, encoding='bytes')
fo.close()
imgArr = dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255
total = imgArr.shape[0]
Y_one_hot = np.eye(10)[dict[b'labels']]
# 训练集的比例
train_test = 0.9

g_b=0
# 自己实现next_batch函数，每次返回一批数据
def next_batch(size):
    global g_b
    xb = imgArr[g_b:g_b+size]
    yb = Y_one_hot[g_b:g_b+size]
    g_b = g_b + size
    return xb,yb

# 参数
learning_rate = 0.001 # 学习率
training_epochs = 2  # 训练总周期
batch_size = 100 # 训练每批样本数

#定义占位符
X = tf.placeholder(tf.float32, [None, 32, 32, 3],name='x')
Y = tf.placeholder(tf.float32, [None, 10],name='y')  # 独热编码

# 第1层卷积，输入图片数据(?, 32, 32, 3)
W1 = tf.Variable(tf.random_normal([3, 3, 3, 32]),name='w1')  #卷积核3x3，输入通道3，输出通道32
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') #卷积输出 （?, 32, 32, 32)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') #池化输出 (?, 16, 16, 32)

# 第2层卷积，输入图片数据(?, 16, 16, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01),name='w2') #卷积核3x3，输入通道32，输出通道64
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME') #卷积输出  (?, 16, 16, 64)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #池化输出 (?, 8, 8, 64)

dim = L2.get_shape()[1].value * L2.get_shape()[2].value * L2.get_shape()[3].value
L2_flat = tf.reshape(L2,[-1, dim])

# 全连接
W3 = tf.get_variable("w3", shape=[ dim, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]),name='b')
logits = tf.matmul(L2_flat, W3)
logits=tf.add(logits,b,name='logits')


#代价或损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # 优化器

# 测试模型检查准确率
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 创建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #全局变量初始化
# 迭代训练
print('开始学习...')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(total * train_test / batch_size)  # 计算总批次
    g_b = 0
    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    acc = sess.run(accuracy, feed_dict={X: imgArr[int(total * train_test):], Y: Y_one_hot[int(total * train_test):]})
    print('Epoch:', (epoch + 1), 'cost =', avg_cost, 'acc=', acc)
print('学习完成')

# 测试模型检查准确率
print('Accuracy:', sess.run(accuracy, feed_dict={X: imgArr[int(total * train_test):], Y: Y_one_hot[int(total * train_test):]}))

# 增加转化变量到常量语句，写出转化函数和指定变量列表两个点（每点2.5分，共5分）
constant_graph=graph_util.convert_variables_to_constants(sess,sess.graph_def,['w1','w2','w3','b','logits'])
# 增加pb文件存储语句，文件名为model.pb,并运行程序，实施数据保存。（2.5分）
with tf.gfile.FastGFile('./model.pb','wb') as f:
    f.write(constant_graph.SerializeToString())

# 在测试集中随机抽一个样本进行测试
# r = random.randint(total * train_test, total - 1)
# print("Label: ", sess.run(tf.argmax(Y_one_hot[r:r + 1], 1)))
# print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: imgArr[r:r + 1]}))
# plt.imshow(imgArr[r:r+1].reshape(32,32,3), interpolation='nearest')
# plt.show()
# while True:
#     str = input()
#     try:
#         if str == 'q':
#             break
#         r = random.randint(total * train_test, total - 1)
#         print("Label: ", sess.run(tf.argmax(Y_one_hot[r:r + 1], 1)))
#         print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: imgArr[r:r + 1]}))
#         plt.imshow(imgArr[r:r + 1].reshape(32, 32, 3), interpolation='nearest')
#         plt.show()
#     except:
#         continue
'''
cifar10 数据可视化
label:
      0 airplane
      1 automobile
      2 bird
      3 cat
      4 deer
      5 dog
      6 frog
      7 horse
      8 ship
      9 truck
'''
