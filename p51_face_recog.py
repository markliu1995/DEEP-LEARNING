import p50_framework as myf
import tensorflow as tf
import numpy as np

from p45_celeba import CelebA
from p48_BufferDS import BufferDS
import time


# img_size: 218, 178
class MyConfig(myf.Config):
    def __init__(self, persons):
        super(MyConfig, self).__init__()
        self.lr = 0.0001
        self.epoches = 2000
        self.buffer_size = 10
        self.batch_size = 50

        self.img_size = 32 * 4   # The target size of the images
        self.convs = 5
        self.base_filters = 32
        self.persons = persons
        self.ds = None

    def get_name(self):
        return 'p51'

    def get_sub_tensors(self, gpu_index):
        return MySubTensors(self)

    def get_app(self):
        return MyApp(self)

    def get_ds_train(self):
        return self.ds

    def get_ds_test(self):
        return self.ds


class MySubTensors:
    def __init__(self, config:MyConfig):
        self.config = config
        x = tf.placeholder(tf.float32, [None, None, None, 3], 'x')
        y = tf.placeholder(tf.int32, [None], 'y')
        self.inputs = [x, y]
        with tf.variable_scope('encode'):
            self.vector = self.encode(x)   # [-1, persons]

        logits = self.get_logits(self.vector)
        y2 = tf.one_hot(y, cfg.persons)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y2, logits=logits)
        loss = tf.reduce_mean(loss)
        self.losses = [loss]

        y_predict = tf.argmax(logits, axis=1, output_type=tf.int32)  # [-1]
        precise = tf.cast(tf.equal(y, y_predict), tf.float32)
        self.precise = tf.reduce_mean(precise)

    def get_logits(self, vector):
        logits = tf.layers.dense(vector, cfg.persons)
        return logits

    def encode(self, x):
        cfg = self.config
        x = tf.image.resize_images(x, (cfg.img_size, cfg.img_size))
        # x: [-1, img_size, img_size, 3]

        x = tf.layers.conv2d(x, cfg.base_filters, 3, 1, 'same', activation=tf.nn.relu, name='conv1')
        filters = cfg.base_filters
        size = cfg.img_size
        for i in range(cfg.convs):
            filters *= 2
            size //= 2
            x = tf.layers.conv2d(x, filters, 3, 2, 'same', activation=tf.nn.relu, name='conv2_%d' % i)

        # x: [-1, 4, 4, 1024]
        return tf.layers.flatten(x)


def predict(app):
    pass


class MyApp(myf.App):
    # def train(self, ds_train, ds_valid):
    #     self.ds_valid = ds_valid
    #     super(MyApp, self).train(ds_train, ds_valid)

    def after_epoch(self, epoch):
        super(MyApp, self).after_epoch(epoch)
        # feed_dict = self.get_feed_dict(self.ds_valid)
        feed_dict = self.get_feed_dict(self.config.get_ds_validation())
        precise = [ts.precise for ts in self.ts.sub_ts]
        ps = self.session.run(precise, feed_dict)
        print('precise:', np.mean(ps))

    def test(self, ds_test):
        print('in MyApp.test()', flush=True)


if __name__ == '__main__':
    path_img = '../samples/celeba/Img/img_align_celeba.zip'
    path_ann = '../samples/celeba/Anno/identity_CelebA.txt'
    path_bbox = '../samples/celeba/Anno/list_bbox_celeba.txt'
    celeba = CelebA(path_img, path_ann, path_bbox)
    cfg = MyConfig(celeba.persons)
    ds = BufferDS(cfg.buffer_size, celeba, cfg.batch_size)
    cfg.ds = ds

    cfg.from_cmd()
