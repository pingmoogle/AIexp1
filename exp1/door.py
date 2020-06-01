import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 下载并加载数据
mnist = input_data.read_data_sets("./mnist_dataset", one_hot=True)

# 数据与标签的占位
x = tf.placeholder(tf.float32, shape=[None, 784])
y_actual = tf.placeholder(tf.float32, shape=[None, 10])

# 初始化权重和偏置
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# softmax回归，得到预测概率
y_predict = tf.nn.softmax(tf.matmul(x, W) + b)
# 求交叉熵得到残差
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_predict), reduction_indices=1))
# 梯度下降法使得残差最小
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 测试阶段，测试准确度计算
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))  # 多个批次的准确度均值

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # 训练，迭代1000次
    for i in range(50000):
        batch_xs, batch_ys = mnist.train.next_batch(100)  # 按批次训练，每批100行数据
        sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})  # 执行训练
        if i % 100 == 0:  # 每训练100次，测试一次
            print("accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))
