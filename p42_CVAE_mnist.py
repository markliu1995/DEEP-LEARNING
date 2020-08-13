import p39_framework as myf
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import numpy as np
import cv2
import os


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.sample_path = '..{sep}samples{sep}MNIST_data'.format(sep=os.sep)
        self.vector_size = 4
        self.momentum = 0.99
        self.cols = 20
        self.img_path = '../imgs/{name}/test.jpg'.format(name=self.get_name())

        self.batch_size = 10
        # self.epoches = 4

    def get_name(self):
        return 'p42'

    def get_tensors(self):
        return MyTensors(self)


class MyTensors:
    def __init__(self, config: MyConfig):
        self.config = config
        with tf.device('/gpu:0'):
            x = tf.placeholder(tf.float32, [None, 784], 'x')
            lr = tf.placeholder(tf.float32, [], 'lr')
            label = tf.placeholder(tf.int32, [None], 'label')
            self.inputs = [x, lr, label]

            x = tf.reshape(x, [-1, 28, 28, 1])
            self.vec = self.encode(x, config.vector_size)    # [-1, 4]
            self.process_normal(self.vec, label)
            self.y = self.decode(self.vec, label)   # [-1, 28, 28, 1]

            loss = tf.reduce_mean(tf.square(self.y - x))
            opt = tf.train.AdamOptimizer(lr)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = opt.minimize(loss)
            self.summary = tf.summary.scalar('loss', tf.sqrt(loss))
            self.y = tf.reshape(self.y, [-1, 28, 28])


    def process_normal(self, vec, label):
        # vec: [-1, vector_size]
        # label: [-1]
        mean = tf.reduce_mean(vec, axis=0)  # [vector_size]
        msd = tf.reduce_mean(tf.square(vec), axis=0)    # mean square difference

        vector_size = vec.shape[1].value
        self.final_mean = tf.get_variable('mean', [vector_size], tf.float32, tf.initializers.zeros, trainable=False)
        self.final_msd = tf.get_variable('msd', [vector_size], tf.float32, tf.initializers.zeros, trainable=False)

        mom = self.config.momentum
        assign = tf.assign(self.final_mean, self.final_mean * mom + mean * (1 - mom))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign)
        assign = tf.assign(self.final_msd, self.final_msd * mom + msd * (1 - mom))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign)


    def encode(self, x, vec_size):
        """
        Encode the x to a vector which size is vec_size
        :param x: the input tensor, which shape is [-1, 28, 28, 1]
        :param vec_size: the size of the semantics vector
        :return: the semantics vectors which shape is [-1, vec_size]
        """
        filters = 16
        x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv1')  # [-1, 28, 28, 16]
        for i in range(2):
            filters *= 2
            x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv2_%d' % i)
            x = tf.layers.max_pooling2d(x, 2, 2)
        # x: [-1, 7, 7, 64]
        x = tf.layers.conv2d(x, vec_size, 7, 1, name='conv3')   # [-1, 1, 1, vec_size]
        return tf.reshape(x, [-1, vec_size])

    def decode(self, vec, label):
        """
        Decode the semantics vector.
        :param vec: [-1, vec_size]
        :param label: [-1]
        :return: [-1, 28, 28, 1]
        """
        y = tf.layers.dense(vec, 7*7*64, tf.nn.relu, name='dense1')
        label = tf.one_hot(label, 10)
        l = tf.layers.dense(label, 7*7*64, name='dense_l_1')
        y += l

        y = tf.reshape(y, [-1, 7, 7, 64])
        size = 7
        filters = 64
        for i in range(2):
            filters //= 2
            size *= 2
            y = tf.layers.conv2d_transpose(y, filters, 3, 2, 'same', name='deconv1_%d' % i, activation=tf.nn.relu)
            l = tf.layers.dense(label, size * size * filters, name='deconv_l_1_%d' % i)
            l = tf.reshape(l, [-1, size, size, filters])
            y += l
        # y: [-1, 28, 28, 16]
        y = tf.layers.conv2d_transpose(y, 1, 3, 1, 'same', name='deconv2')  # [-1, 28, 28, 1]
        return y


class MyDS:
    def __init__(self, ds, config):
        self.ds = ds
        self.lr = config.lr
        self.num_examples = ds.num_examples

    def next_batch(self, batch_size):
        xs, labels = self.ds.next_batch(batch_size)
        return xs, self.lr, labels


def predict(app, samples, path, cols):
    mean = app.session.run(app.ts.final_mean)
    print(mean)
    msd = app.session.run(app.ts.final_msd)
    std = np.sqrt(msd - mean ** 2)
    print(std)

    vec = np.random.normal(mean, std, [samples, len(std)])
    label = [e % 10 for e in range(samples)]
    imgs = app.session.run(app.ts.y, {app.ts.vec: vec, app.ts.inputs[-1]: label})   # [-1, 28, 28]

    imgs = np.reshape(imgs, [-1, cols, 28, 28])
    imgs = np.transpose(imgs, [0, 2, 1, 3])  # [-1, 28, 20, 28]
    # imgs = np.reshape(imgs, [-1, 28, cols*28])
    # imgs = np.transpose(imgs, [1, 0, 2])  # [28, -1, 20*28]
    imgs = np.reshape(imgs, [-1, cols*28])

    # imgs = np.transpose(imgs, [1, 0, 2])  # [28, -1, 28]
    # imgs = np.reshape(imgs, [28, -1, cols * 28])  # [28, -1, 20*28]
    # imgs = np.transpose(imgs, [1, 0, 2])   # [-1, 28, 20*28]
    # imgs = np.reshape(imgs, [-1, cols * 28])  # [-1, 20*28

    myf.make_dirs(path)
    cv2.imwrite(path, imgs * 255)
    print('Write image into', path)


if __name__ == '__main__':
    cfg = MyConfig()
    cfg.from_cmd()
    print('-' * 100)
    print(cfg)

    dss = read_data_sets(cfg.sample_path)
    app = myf.App(cfg)
    with app:
        # app.train(MyDS(dss.train, cfg), MyDS(dss.validation, cfg))
        predict(app, 500, cfg.img_path, cfg.cols)



