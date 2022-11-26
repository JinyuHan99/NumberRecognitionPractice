import tensorflow as tf
import numpy as np
import struct
from model import Network


class Train:
    def __init__(self, train_images_, label_tensor_, test_images_, test_label_tensor_, batch_size, learning_rate,
                 train_step):
        self.net = Network(learning_rate)

        # build data generator
        # create placeholder
        input_data = tf.compat.v1.placeholder(tf.float32, [None, 784])
        label = tf.compat.v1.placeholder(tf.float32, [None, 10])
        dataset = tf.compat.v1.data.Dataset.from_tensor_slices((input_data, label))
        dataset = dataset.shuffle(buffer_size=20).batch(batch_size).repeat()
        iterator = dataset.make_initializable_iterator()
        self.data_element = iterator.get_next()

        # initialize session
        # Network() only create a framework for computing，real computing will be put into session
        self.sess = tf.compat.v1.Session()
        # initialize the iterator and feed data to placeholder
        self.sess.run(iterator.initializer, feed_dict={input_data: train_images_, label: label_tensor_})
        # initialize all variables
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.test_images = test_images_
        self.test_label_tensor = test_label_tensor_
        self.train_step = train_step

    def train(self):
        # function to start training
        for i in range(self.train_step):
            # obtain input and label from dataset
            x, label = self.sess.run(self.data_element)
            _, loss = self.sess.run([self.net.train, self.net.loss], feed_dict={self.net.x: x, self.net.label: label})
            # print loss，loss will gradually decrease during training
            # as training continues，the ability of the network to recognize the numbers will improve
            # because this is a small network, the loss will fluctuate near the end of the training
            if (i + 1) % 10 == 0:
                print('step%5d, loss：%.2f' % (i + 1, loss))

    def calculate_accuracy(self):
        # note：did not compute self.net.train in this function
        # only compute accuracy tensor, so the network will not be updated
        accuracy = self.sess.run(self.net.accuracy,
                                 feed_dict={self.net.x: self.test_images, self.net.label: self.test_label_tensor})
        print("accuracy: %.2f，%dgraphs tested " % (accuracy, len(self.test_label_tensor)))


# extract data for training and testing
def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    train_images = decode_idx3_ubyte(idx3_ubyte_file='data/train-images.idx3-ubyte')
    train_images = train_images / 256.0
    train_images = train_images.reshape(60000, -1)
    train_labels = decode_idx1_ubyte(idx1_ubyte_file='data/train-labels.idx1-ubyte')
    label_tensor = np.zeros((60000, 10))

    for i, j in enumerate(train_labels):
        label_tensor[i][int(j)] = 1
    test_images = decode_idx3_ubyte(idx3_ubyte_file='data/t10k-images.idx3-ubyte')
    test_images = test_images.reshape(10000, -1) / 256.0
    test_labels = decode_idx1_ubyte(idx1_ubyte_file='data/t10k-labels.idx1-ubyte')
    test_labels_tensor = np.zeros((10000, 10))
    for i, j in enumerate(test_labels):
        test_labels_tensor[i][int(j)] = 1
    print('Finish data extraction')
    # start training
    app = Train(train_images, label_tensor, test_images, test_labels_tensor, batch_size=64, learning_rate=0.001,
                train_step=500)
    app.train()
    app.calculate_accuracy()
