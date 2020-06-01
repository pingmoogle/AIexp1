# -*- coding: UTF-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from time import time
import os

save_step = 5
ckpt_dir = ".\modelsave"


def plot_image(img):
    plt.imshow(img.reshape(28, 28), cmap="binary")
    plt.show()


def plot_images_labels_prediction(images, labels, prediction,
                                  index, num=10):
    fig = plt.gcf()
    fig.set_size_inches(10, 12)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(np.reshape(images[index], (28, 28)), cmap='binary')
        title = "label=" + str(np.argmax(labels[index]))
        if len(prediction) > 0:
            title += ",predict=" + str(prediction[index])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        index += 1
    plt.show()


def fcn_layer(inputs, input_dim, output_dim, activation=None):
    W = tf.Variable(tf.truncated_normal(
        [input_dim, output_dim], stddev=0.1
    ))
    b = tf.Variable(tf.zeros([output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs


def study1():
    print(
        "训练集 train 数量：", mnist.train.num_examples,
        "\n验证集 validation 数量：", mnist.validation.num_examples,
        "\n测试集 test 数量：", mnist.test.num_examples
    )
    print(
        "\ntrain image shape: ", mnist.train.images.shape,
        "\nlabels shape: ", mnist.train.labels.shape
    )
    print(
        len(mnist.train.images[0]), '\n',
        mnist.train.images[0].shape, '\n',
        mnist.train.images[0],
    )
    print(mnist.train.images[0].reshape(28, 28))


def study2():
    plot_image(mnist.train.images[20000])


def study3():
    batch_images_xs, batch_labels_ys = mnist.train.next_batch(5)
    print(batch_images_xs)
    print(batch_labels_ys)


def study4():
    # 定义占位符
    x = tf.placeholder(tf.float32, [None, 784], name="X")
    y = tf.placeholder(tf.float32, [None, 10], name="Y")
    # 定义变量
    W = tf.Variable(tf.random_normal([784, 10]), name="W")
    b = tf.Variable(tf.zeros([10]), name="b")
    forward = tf.matmul(x, W) + b  # 前向计算
    pred = tf.nn.softmax(forward)  # Softmax 分类

    # 训练参数
    train_epochs = 200
    batch_size = 200
    total_batch = int(mnist.train.num_examples / batch_size)
    display_step = 1
    learning_rate = 0.01

    # 定义损失函数
    loss_function = tf.reduce_mean(-tf.reduce_sum(
        y * tf.log(pred), reduction_indices=1
    ))

    # 选择优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

    # 定义准确率
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # sess.close()
    for epoch in range(train_epochs):
        for batch in range(batch_size):
            xs, ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: xs, y: ys})
        loss, acc = sess.run([loss_function, accuracy],
                             feed_dict={
                                 x: mnist.validation.images,
                                 y: mnist.validation.labels
                             })
        if (epoch + 1) % display_step == 0:
            print(
                "Train Epoch: ", "%02d" % (epoch + 1),
                " Loss = ", "{:.9f}".format(loss),
                " Accuracy = ", "{:.4f}".format(acc)
            )
    print("Train Finished.")

    accu_test = sess.run(
        accuracy, feed_dict={x: mnist.test.images,
                             y: mnist.test.labels}
    )
    print("Test Accuracy: ", accu_test)

    prediction_result = sess.run(tf.argmax(pred, 1),
                                 feed_dict={x: mnist.test.images})
    print(prediction_result[:10])
    plot_images_labels_prediction(mnist.test.images, mnist.test.labels,
                                  prediction_result, 10, 25)


def study5():
    norm = tf.random_normal([100])
    # TODO: Something wrong here.
    with tf.Session as sess:
        norm_data = norm.eval()
    plt.hist(norm_data)
    plt.show()


def study6():
    x = np.array([[-3.1, 1.8, 9.7, -2.5]])
    pred = tf.nn.softmax(x)
    sess = tf.Session()
    v = sess.run(pred)
    print(v)
    sess.close()


def study7():
    # 定义占位符
    x = tf.placeholder(tf.float32, [None, 784], name="X")
    y = tf.placeholder(tf.float32, [None, 10], name="Y")

    # 隐藏层
    H1_NN = 128
    H2_NN = 128
    H3_NN = 128
    W1 = tf.Variable(tf.random_normal([784, H1_NN]))
    b1 = tf.Variable(tf.zeros([H1_NN]))
    W2 = tf.Variable(tf.random_normal([H1_NN, H2_NN]))
    b2 = tf.Variable(tf.zeros([H2_NN]))
    W3 = tf.Variable(tf.random_normal([H2_NN, H3_NN]))
    b3 = tf.Variable(tf.zeros([H3_NN]))
    Y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
    Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)

    # 输出层
    W4 = tf.Variable(tf.random_normal([H3_NN, 10]))
    b4 = tf.Variable(tf.zeros([10]))
    forward = tf.matmul(Y3, W4) + b4
    pred = tf.nn.softmax(forward)

    # 交叉熵
    loss_function = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=forward, labels=y)
    )

    # 训练参数
    train_epochs = 300
    batch_size = 150
    total_batch = int(mnist.train.num_examples / batch_size)
    display_step = 1
    learning_rate = 0.01

    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

    # 准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    startTime = time()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(train_epochs):
        for batch in range(batch_size):
            xs, ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                x: xs, y: ys
            })
        loss, acc = sess.run([loss_function, accuracy],
                             feed_dict={
                                 x: mnist.validation.images,
                                 y: mnist.validation.labels
                             })
        if (epoch + 1) % display_step == 0:
            print(
                "Train Epoch: ", "%02d" % (epoch + 1),
                " Loss = ", "{:.9f}".format(loss),
                " Accuracy = ", "{:.4f}".format(acc)
            )
        if (epoch+1)%save_step == 0:
            saver.save(sess, os.path.join(
                ckpt_dir, 'mnist_h256_model_{:06d}.ckpt'.format(epoch+1)
            ))

    duration = time() - startTime
    print("Train Finished, Time: ", "{:.2f}".format(duration))
    saver.save(sess, os.path.join(
        ckpt_dir, 'mnist_h256_model.ckpt'
    ))
    accu_test = sess.run(accuracy, feed_dict={
        x: mnist.test.images, y: mnist.test.labels
    })
    print("Test Accuracy: ", accu_test)


if __name__ == '__main__':
    mnist = input_data.read_data_sets("./mnist_dataset", one_hot=True)
    study7()
