import tensorflow as tf
import p74_framework as myf
import numpy as np
import cv2
from p61_UNet import unet
from p56_ResNet import ResNet, RESNET50


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.img_path = '../imgs/{name}/test.jpg'.format(name=self.get_name())
        self.lr = 1e-6
        self.batch_size = 2

        self._ds = None
        self.training = True
        self.tensors = None

        self.img_size = 32 * 4
        self.base_filters = 16

    def get_name(self):
        return 'p80'

    @property
    def ds(self):
        if self._ds is None:
            self._ds = MyDS(self.img_size)
        return self._ds

    def get_ds_train(self):
        return self.ds

    def get_ds_test(self):
        return self.ds

    def get_sub_tensors(self, gpu_index):
        return MySubTensors(self)

    def get_tensors(self):
        return MyTensors(self)

    def random_seed(self):
        MyDS.reset()

    def get_app(self):
        return MyApp(self)

    def test(self):
        self.keep_prob = 1.0
        self.training = False
        super(MyConfig, self).test()


class MyDS:
    def __init__(self, img_size):
        self.img_size = img_size
        self.num_examples = 1000

    def next_batch(self, batch_size):
        simples = np.random.normal(size=[batch_size, self.img_size, self.img_size, 3])
        imgs = np.random.normal(size=[batch_size, self.img_size, self.img_size, 3])
        return simples, imgs

    @staticmethod
    def reset():
        np.random.seed(2938545)


class MyTensors(myf.Tensors):
    def __init__(self, cfg):
        cfg.tensors = self
        with tf.device('/gpu:0'):
            self.training = tf.placeholder(tf.bool, [], 'training')
        super(MyTensors, self).__init__(cfg)

    def compute_grads(self, opt):
        vars = tf.trainable_variables()
        vars_disc = [var for var in vars if 'disc' in var.name]
        vars_gene = [var for var in vars if 'gene' in var.name]
        vars = [vars_disc, vars_gene, vars_disc, vars_gene]

        grads = [[opt.compute_gradients(loss, vs) for vs, loss in zip(vars, ts.losses)] for ts in self.sub_ts]   # [gpus, losses]
        return [self.get_grads_mean(grads, i) for i in range(len(grads[0]))]


class MySubTensors:
    def __init__(self, cfg: MyConfig):
        self.cfg = cfg
        simple = tf.placeholder(tf.float64, [None, cfg.img_size, cfg.img_size, 3], 'simple')
        img = tf.placeholder(tf.float64, [None, cfg.img_size, cfg.img_size, 3], 'img')
        self.inputs = [simple, img]

        with tf.variable_scope('gene'):
            img2 = self.gene(simple)  # [-1, img_size, img_size, 3]

        with tf.variable_scope('disc') as scope:
            img2_p = self.disc(simple, img2)  # 假样本为真的概率. [-1, 1]
            scope.reuse_variables()
            img_p = self.disc(simple, img)    # 真样本为真的概率, [-1, 1]

        loss1 = -tf.reduce_mean(tf.log(img_p))
        loss2 = -tf.reduce_mean(tf.log(1 - img2_p))
        loss3 = -tf.reduce_mean(tf.log(img2_p))
        self.losses = [loss1, loss3, loss2, loss3]

    def disc(self, simple, img):
        # simple: [-1, img_size, img_size, 3]
        # img: [-1, img_size, img_size, 3]

        cfg = self.cfg
        x = tf.concat((simple, img), axis=-1)   # [-1, img_size, img_size, 6]
        resnet = ResNet(RESNET50)
        logits = resnet(x, 1, cfg.tensors.training, 'disc_resnet')  # [-1, 1]
        return tf.sigmoid(logits)

    def gene(self, simple):
        # simple: [-1, img_size, img_size, 3]
        # return: [-1, img_size, img_size, 3

        return unet(simple, 5, self.cfg.base_filters, name='gene_unet')


class MyApp(myf.App):
    def get_feed_dict(self, ds):
        result = super(MyApp, self).get_feed_dict(ds)
        result[self.ts.training] = self.config.training
        return result

    def before_epoch(self, epoch):
        self.config.random_seed()


if __name__ == '__main__':
    MyConfig().from_cmd()