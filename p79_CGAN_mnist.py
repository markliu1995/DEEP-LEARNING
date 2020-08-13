import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import p74_framework as myf
import numpy as np
import cv2


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.base_filter = 16
        self.sample_path = '../samples/MNIST_data'
        self.imgs_path = './imgs/{name}/test.jpg'.format(name=self.get_name())
        self.lr = 5e-6
        self.batch_size = 200
        self._ds = None
        self.vec_size = 4
        self.keep_prob = 0.65

    @property
    def ds(self):
        if self._ds is None:
            self._ds = MyDS(read_data_sets(self.sample_path).train, self.vec_size)
        return self._ds

    def get_sub_tensors(self, gpu_index):
        return MySubTensor(self)

    def get_ds_train(self):
        return self._ds

    def get_ds_test(self):
        return self.ds

    def get_name(self):
        return "p79"

    def get_tensors(self):
        return MyTensors(self)

    def get_app(self):
        return MyApp(self)

    def random_seed(self):
        np.random.seed(2389123)

    def test(self):
        self.keep_prob = 1.0
        super(MyConfig, self).test()


class MySubTensor:
    def __init__(self, cfg: MyConfig):
        self.cfg = cfg
        x = tf.placeholder(tf.float64, [None, 784], "x")
        lx = tf.placeholder(tf.int64, [None], "lx")  # labels
        v = tf.placeholder(tf.float64, [None, self.cfg.vec_size], "y")
        self.inputs = [x, lx, v]
        self.lv = lx
        lx = tf.one_hot(lx, 10, dtype=tf.float64)

        with tf.variable_scope("gene"):
            x2 = self.gene(v, lx)  # [-1, 28, 28, 1]
            self.v = v
            self.x2 = tf.reshape(x2, [-1, 28, 28])

        with tf.variable_scope("disc") as scope:
            x2_y = self.disc(x2, lx)  # 假样本为真的概率
            scope.reuse_variables()
            x = tf.reshape(x, [-1, 28, 28, 1])
            x_y = self.disc(x, lx)  # 真样本为真的概率

        loss1 = -tf.reduce_mean(tf.log(x_y))   # 训练disc  标签 1
        loss2 = -tf.reduce_mean(tf.log(1 - x2_y)) # 训练disc 假样本为假的概率  标签 0
        loss3 = -tf.reduce_mean(tf.log(x2_y))  # 训练gene   标签 1
        self.losses = [loss1, loss2, loss3]

    def disc(self, x, l):
        # [-1, 28, 28, 1]
        # l：[-1,10]
        filter = self.cfg.base_filter
        size = (x.shape[1].value, x.shape[2].value)
        x = tf.layers.conv2d(x, filter, 3, 1, "same", activation=tf.nn.relu, name="conv1")

        for i in range(2):
            size = (size[0] // 2, size[1] // 2)
            filter *= 2
            x = tf.layers.conv2d(x, filter, 3, 2, "same", activation=tf.nn.relu, name="conv2%d" % i)
            t = tf.layers.dense(l, size[0] * size[1] * filter, name="dense1%d" % i)
            t = tf.reshape(t, [-1, size[0], size[1], filter])
            x += t
        x = tf.layers.flatten(x)
        x = tf.nn.dropout(x, self.cfg.keep_prob)
        x = tf.layers.dense(x, 1, name="dense")
        return tf.nn.sigmoid(x)

    def gene(self, v, l):
        # v shape [-1, 4]
        filters = self.cfg.base_filter * 4
        v = tf.layers.dense(v, 7 * 7 * self.cfg.base_filter, name="dense", activation=tf.nn.relu)
        v = tf.reshape(v, [-1, 7, 7, filters])
        for i in range(2):
            filters //= 2
            v = tf.layers.conv2d_transpose(v, filters, 3, 2, "same", activation=tf.nn.relu, name="deconv_%d" % i)
        # v [-1, 28, 28, filters]
        v = tf.layers.conv2d_transpose(v, 1, 3, 1, "same", name='deconv2')
        return v


class MyTensors(myf.Tensors):
    def compute_grads(self, opt):
        vars = tf.trainable_variables()
        var_disc = [var for var in vars if "disc" in var.name]
        var_gene = [var for var in vars if "gene" in var.name]
        vars = [var_disc, var_disc, var_gene]
        grads = [[opt.compute_gradients(loss, vs) for vs, loss in zip(vars, ts.losses)] for ts in self.sub_ts]  # [gpus, losses]
        return [self.get_grads_mean(grads, i) for i in range(len(grads[0]))]



class MyDS:
    def __init__(self, ds, vec_size):
        self.vec_size = vec_size
        self.num_examples = ds.num_examples
        self.ds = ds

    def next_batch(self, batch_size):
        xs, ls = self.ds.next_batch(batch_size)
        vs = np.random.normal(size=[batch_size, self.vec_size])
        return xs, ls, vs


class MyApp(myf.App):
    def before_epoch(self, epoch):
        self.config.random_seed()

    def test(self, ds):
        vs = np.random.normal(size=[200, self.config.vec_size])
        ls = [e % 10 for e in range(np.shape(vs)[0])]
        ts = self.ts.sub_ts[-1]
        imgs = self.session.run(ts.x2, {ts.v: vs, ts.lv: ls}) # [-1, 28, 28]
        imgs = np.reshape(imgs, [-1, 10, 28, 28])
        imgs = np.transpose(imgs,[0, 2, 1, 3])
        imgs = np.reshape(imgs, [-1, 10 * 28])
        c = cv2.imwrite(self.config.imgs_path, imgs)

        print("save....", c)


if __name__ == "__main__":
    MyConfig().from_cmd()
