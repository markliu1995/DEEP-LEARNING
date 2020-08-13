import p50_framework as myf
import tensorflow as tf
import numpy as np
import cv2
from p45_celeba import CelebA
from p48_BufferDS import BufferDS


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.buffer_size = 100
        self.batch_size = 50
        self.epoches = 2000
        self.lr = 0.0001

        self.path_img = '../samples/celeba/Img/img_align_celeba.zip'
        self.path_ann = '../samples/celeba/Anno/identity_CelebA.txt'
        self.path_bbox = '../samples/celeba/Anno/list_bbox_celeba.txt'
        self.celeba = None
        self.ds_train = None
        
        self.img_size = 32 * 4   # The target size of the images
        self.vec_size = 100
        self.momentum = 0.99

        self.cols = 5
        self.img_path = '../imgs/{name}/test.jpg'.format(name=self.get_name())

    def test(self):
        with self.get_app() as app:
            app.transfer()

    def get_name(self):
        return 'p55'

    def get_tensors(self):
        return MyTensors(self)

    def get_sub_tensors(self, gpu_index):
        return MySubTensors(self)

    def get_app(self):
        return MyApp(self)

    def get_ds_test(self):
        return None

    def get_ds_train(self):
        if self.ds_train is None:
            self.celeba = CelebA(self.path_img, self.path_ann, self.path_bbox)
            self.persons = self.celeba.persons
            self.ds_train = BufferDS(self.buffer_size, self.celeba, self.batch_size)
        return self.ds_train


class MyTensors(myf.Tensors):
    def get_loss_for_summary(self, loss):
        return tf.sqrt(loss)


class MySubTensors:
    def __init__(self, config: MyConfig):
        self.config = config
        x = tf.placeholder(tf.float32, [None, None, None, 3], 'x')
        self.inputs = [x]

        x = tf.image.resize_images(x, (config.img_size, config.img_size)) / 255  # [-1, img_size, img_size, 3]  Color models: HSV, RGB
        self.vec = self.encode(x, config.vec_size)    # [-1, vec_size]
        self.process_normal(self.vec)
        self.y = self.decode(self.vec)   # [-1, img_size, img_size, 3]

        loss = self.get_loss(x)
        self.losses = [loss]

    def get_loss(self, x):
        loss = tf.reduce_mean(tf.square(self.y - x))
        return loss

    def process_normal(self, vec):
        # vec: [-1, vector_size]
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
        :param x: the input tensor, which shape is [-1, img_size, img_size, 3]
        :param vec_size: the size of the semantics vector
        :return: the semantics vectors which shape is [-1, vec_size]
        """
        filters = 32
        x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv1')  # [-1, img_size, img_size, 32]
        for i in range(5):
            filters *= 2
            x = tf.layers.conv2d(x, filters, 3, 2, 'same', activation=tf.nn.relu, name='conv2_%d' % i)
        # x: [-1, 4, 4, 1024]
        size = self.config.img_size // (2 ** 5)   # 4
        x = tf.layers.conv2d(x, vec_size, size, 1, name='conv3')   # [-1, 1, 1, vec_size]
        return tf.reshape(x, [-1, vec_size])

    def decode(self, vec):
        """
        Decode the semantics vector.
        :param vec: [-1, vec_size]
        :return: [-1, img_size, img_size, 3]
        """
        size = self.config.img_size // 2 ** 5   # 4
        filters = 1024
        y = tf.layers.dense(vec, size * size * filters, tf.nn.relu, name='dense1')
        y = tf.reshape(y, [-1, size, size, filters])
        for i in range(5):
            filters //= 2
            y = tf.layers.conv2d_transpose(y, filters, 3, 2, 'same', name='deconv1_%d' % i, activation=tf.nn.relu)
        # y: [-1, img_size, img_size, 32]
        y = tf.layers.conv2d_transpose(y, 3, 3, 1, 'same', name='deconv2')  # [-1, img_size, img_size, 3]
        return y


class MyDS:
    def __init__(self, ds, config):
        self.ds = ds
        self.lr = config.lr
        self.num_examples = ds.num_examples

    def next_batch(self, batch_size):
        xs, _ = self.ds.next_batch(batch_size)
        return xs, self.lr


class MyApp(myf.App):
    def test(self, ds_test):
        cfg = self.config
        mean = self.session.run(self.ts.sub_ts[0].final_mean)
        print(mean)
        msd = self.session.run(self.ts.sub_ts[0].final_msd)
        std = np.sqrt(msd - mean ** 2)
        print(std)
    
        vec = np.random.normal(mean, std, [cfg.batch_size, len(std)])
        imgs = self.session.run(self.ts.sub_ts[0].y, {self.ts.sub_ts[0].vec: vec})   # [-1, img_size, img_size, 3]

        imgs = np.reshape(imgs, [-1, cfg.cols, cfg.img_size, cfg.img_size, 3])
        imgs = np.transpose(imgs, [0, 2, 1, 3, 4])  # [-1, img_size, 5, img_size, 3]
        imgs = np.reshape(imgs, [-1, cfg.cols * cfg.img_size, 3])
    
        cv2.imwrite(cfg.img_path, imgs * 255)

    def transfer(self):
        path = ['../samples/famous/wlh.jpg', '../samples/famous/ljx.jpg']
        size = self.config.img_size
        imgs = [cv2.resize(cv2.imread(p), (size, size)) for p in path]
        ts = self.ts.sub_ts[0]
        vecs = self.session.run(ts.vec, {ts.inputs[0]: imgs})
        vecs = get_middle_vectors(vecs[0], vecs[1])
        pics = self.session.run(ts.y, {ts.vec: vecs}) * 255   # [-1, img_size, img_size, 3]
        pics = [imgs[1]] + list(pics) + [imgs[0]]
        pics = np.reshape(pics, [-1, self.config.img_size, 3])
        cv2.imwrite(self.config.img_path, pics)
        print('Write image into', self.config.img_path)


def get_middle_vectors(src, dst):
    num = 10
    delta = 1 / (num + 1)
    alpha = delta
    result = []
    for i in range(num):
        vec = src * alpha + dst * (1 - alpha)
        result.append(vec)
        alpha += delta
    return result


if __name__ == '__main__':

    cfg = MyConfig()
    cfg.from_cmd()
