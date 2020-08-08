import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import p74_framework as myf
import numpy as np
import cv2


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.img_path = '../imgs/{name}/test.jpg'.format(name=self.get_name())
        self.sample_path = '../samples/MNIST_data'
        self.lr = 1e-6
        self.batch_size = 500
        self.keep_prob = 0.65

        self._ds = None

        self.base_filters = 16
        self.hidden_units = 2000
        self.vec_size = 4
        self.c = 0.01

    def get_name(self):
        return 'p82'

    @property
    def ds(self):
        if self._ds is None:
            self._ds = MyDS(read_data_sets(self.sample_path).train, self.vec_size)
        return self._ds

    def get_ds_train(self):
        return self.ds

    def get_ds_test(self):
        return self.ds

    def get_sub_tensors(self, gpu_index):
        return MySubTensors(self)

    def get_tensors(self):
        return MyTensors(self)

    def get_app(self):
        return MyApp(self)

    def test(self):
        self.keep_prob = 1.0
        super(MyConfig, self).test()


class MyDS:
    def __init__(self, ds, vec_size):
        self.num_examples = ds.num_examples
        self.ds = ds
        self.vec_size = vec_size

    def next_batch(self, batch_size):
        xs, lxs = self.ds.next_batch(batch_size)
        ls = np.random.randint(0, 10, size=[batch_size])
        return xs, lxs, ls


class MyTensors(myf.Tensors):
    def __init__(self, conifg):
        super(MyTensors, self).__init__(conifg)
        self.make_clip()

    def compute_grads(self, opt):
        vars = tf.trainable_variables()
        vars_disc = [var for var in vars if 'disc' in var.name]
        vars_gene = [var for var in vars if 'gene' in var.name]
        vars = [vars_disc, vars_gene, vars_disc, vars_gene, vars_gene]
        grads = []
        for gpu_id, ts in enumerate(self.sub_ts):
            with tf.device("/gpu: %d" % gpu_id):
                grad = [opt.compute_gradients(loss, vs) for vs, loss in zip(vars, ts.losses)]
            grads.append(grad)
        return [self.get_grads_mean(grads, i) for i in range(len(grads[0]))]

    def get_loss_for_summary(self, loss):
        return tf.sqrt(loss) if loss == self.losses[-1] else loss

    def make_clip(self):
        with tf.device("/gpu:0"):
            result = []
            vars = [var for var in tf.trainable_variables() if 'disc' in var.name]
            c = self.config.c
            clips = []
            for var in vars:
                clip = tf.clip_by_value(var, -c, c)
                clip = tf.assign(var, clip)
                clips.append(clip)
            result.append(clips)
            self.clip = result



class MySubTensors:
    def __init__(self, cfg: MyConfig):
        self.cfg = cfg

        x = tf.placeholder(tf.float64, [None, 784], 'x')
        lx = tf.placeholder(tf.int64, [None], 'lx')
        l = tf.placeholder(tf.int64, [None], 'l')
        self.x = x
        self.l = l
        self.inputs = [x, lx, l]
        lx = tf.one_hot(lx, 10, dtype=tf.float64)  # [-1, 10]
        l = tf.one_hot(l, 10, dtype=tf.float64)  # [-1, 10]

        x = tf.reshape(x, [-1, 28, 28, 1])
        with tf.variable_scope('gene') as scope:
            x2 = self.gene(x, l)  # [-1, 28, 28, 1]
            self.x2 = tf.reshape(x2, [-1, 28, 28])
            scope.reuse_variables()
            x3 = self.gene(x2, lx)  # [-1, 28, 28, 1]

        with tf.variable_scope('disc') as scope:
            x2_y = self.disc(x2, l)  # 假样本为真的概率, [-1, 1]
            x = tf.reshape(x, [-1, 28, 28, 1])
            scope.reuse_variables()
            x_y = self.disc(x, lx)  # 真样本为真的概率, [-1, 1]

        loss1 = tf.reduce_mean(-x_y)
        loss2 = tf.reduce_mean(x2_y)
        loss3 = tf.reduce_mean(-x2_y)
        loss4 = tf.reduce_mean(tf.square(x - x3))
        self.losses = [loss1, loss3, loss2, loss3, loss4]

    def disc(self, x, l):
        x = self.encode(x, l, 1)
        return x

    def encode(self, x, l, logits: int):
        # x: [-1, 28, 28, 1]
        # l: [-1, 10]

        filters = self.cfg.base_filters
        size = (x.shape[1].value, x.shape[2].value)
        x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv1')

        for i in range(2):
            filters *= 2
            size = (size[0] // 2, size[1] // 2)
            x = tf.layers.conv2d(x, filters, 3, 2, 'same', activation=tf.nn.relu, name='conv2_%d' % i)

            t = tf.layers.dense(l, size[0] * size[1] * filters, name='dense1_%d' % i)
            t = tf.reshape(t, [-1, size[0], size[1], filters])
            x += t

        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, self.cfg.hidden_units, activation=tf.nn.relu, name='dense1')
        x = tf.nn.dropout(x, self.cfg.keep_prob)
        return tf.layers.dense(x, logits, name='dense2')

    def gene(self, x, l):
        # x: [-1, 28, 28, 1]
        # l: [-1, 10]

        v = self.encode(x, l, self.cfg.vec_size)  # [-1, vec_size]
        filters = self.cfg.base_filters * 4
        size = 7
        v = tf.layers.dense(v, 7 * 7 * filters, activation=tf.nn.relu, name='dense')
        v = tf.reshape(v, [-1, 7, 7, filters])

        for i in range(2):
            filters //= 2
            size *= 2
            v = tf.layers.conv2d_transpose(v, filters, 3, 2, 'same', activation=tf.nn.relu, name='deconv1_%d' % i)

            t = tf.layers.dense(l, size * size * filters, name='dense2_%d' % i)
            t = tf.reshape(t, [-1, size, size, filters])
            v += t

        # v: [-1, 28, 28, filters]
        v = tf.layers.conv2d_transpose(v, 1, 3, 1, 'same', name='deconv2')
        return v  # [-1, 28, 28, 1]


class MyApp(myf.App):
    def after_batch(self, epoch, batch):
        self.session.run(self.ts.clip)

    def test(self, ds):
        cfg = self.config
        ls = [e % 10 for e in range(cfg.batch_size)]
        xs, _, _ = ds.next_batch(cfg.batch_size)
        ts = self.ts.sub_ts[-1]
        imgs = self.session.run(ts.x2, {ts.x: xs, ts.l: ls}) * 255  # [-1, 28, 28]

        xs = np.reshape(xs, [-1, 28, 28]) * 255
        imgs = (xs, imgs)  # [2, -1, 28, 28]
        imgs = np.transpose(imgs, [1, 0, 2, 3])  # [-1, 2, 28, 28]
        imgs = np.reshape(imgs, [-1, 28, 28])  # [-1, 28, 28]

        imgs = np.reshape(imgs, [-1, 20, 28, 28])
        imgs = np.transpose(imgs, [0, 2, 1, 3])  # [-1, 28, 10, 28]
        imgs = np.reshape(imgs, [-1, 20 * 28])

        myf.make_dirs(self.config.img_path)
        cv2.imwrite(self.config.img_path, imgs)
        print('Write image into', self.config.img_path, flush=True)


if __name__ == '__main__':
    MyConfig().from_cmd()
