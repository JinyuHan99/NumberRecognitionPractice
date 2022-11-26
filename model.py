import tensorflow as tf


# convolutional and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 网络
class Network:
    def __init__(self, learning_rate):
        # learning rate，0.00001 - 0.5
        self.learning_rate = learning_rate

        # input tensor
        self.x = tf.compat.v1.placeholder(tf.float32, [None, 784])

        # if number is 8，the corresponding label is [0,0,0,0,0,0,0,0,1,0]
        self.label = tf.compat.v1.placeholder(tf.float32, [None, 10])

        # output value
        self.y = self.net(self.x)

        # loss function
        self.loss = -tf.reduce_sum(self.label * tf.compat.v1.log(self.y + 1e-10))

        # Back propagation: adjust w and b by gradient descent to minimize loss
        self.train = tf.compat.v1.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        # validate the accuracy
        predict = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.y, 1))

        # predict -> [true, true, true, false, false, true]
        self.accuracy = tf.reduce_mean(tf.cast(predict, 'float'))

    def net(self, input):
        x_img = tf.reshape(input, [-1, 28, 28, 1])

        # the first convolution layer and pooling layer
        w_conv1 = tf.Variable(
            tf.compat.v1.random.truncated_normal([3, 3, 1, 32], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
        h_conv1 = tf.nn.relu(conv2d(x_img, w_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # the second convolution layer and pooling layer
        w_conv2 = tf.Variable(tf.compat.v1.random.truncated_normal([3, 3, 32, 50], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[50]))
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # the first full connected layer
        w_fc1 = tf.Variable(tf.compat.v1.random.truncated_normal([7 * 7 * 50, 1024], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 50])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        # dropout（随机权重失活）
        # keep_prob = tf.compat.v1.placeholder(tf.float32)
        # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # the second full connected layer
        w_fc2 = tf.Variable(tf.compat.v1.random.truncated_normal([1024, 10], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
        z3 = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

        return z3
