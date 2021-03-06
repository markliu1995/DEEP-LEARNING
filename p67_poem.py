import p50_framework as myf
from p48_BufferDS import BufferDS
from p66_read_qts import QTS
import tensorflow as tf
import numpy as np


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.qts_path = '../samples/qts_7X4.txt'
        self._ds = None

        self.num_units = 200
        self.num_steps = 32
        self.batch_size = 100

    def ds(self):
        self.make_ds()
        return self._ds

    def make_ds(self):
        if self._ds is None:
            qts = QTS(self.qts_path)
            self._ds = BufferDS(1000, qts, self.batch_size)
            self._num_chinese_chars = qts.get_num_chars()

    def num_chinese_chars(self):
        self.make_ds()
        return self._num_chinese_chars

    def get_ds_train(self):
        return self.ds()

    def get_ds_test(self):
        return self.get_ds_train()

    def get_name(self):
        return 'p67'

    def get_sub_tensors(self, gpu_id):
        return MySubTensors(self)


class MySubTensors:
    def __init__(self, cfg: MyConfig):
        self.config = cfg

        x = tf.placeholder(tf.int64, [None, cfg.num_steps], 'x')
        self.inputs = [x]

        x = tf.one_hot(x, cfg.num_chinese_chars())  # [-1, 32, 4340]
        y = tf.layers.dense(x, cfg.num_units, name='dense1')  # [-1, 32, 200]

        cell = tf.nn.rnn_cell.LSTMCell(cfg.num_units, name='cell')
        state = cell.zero_state(tf.shape(y)[0], y.dtype)
        y_predict = []
        losses = []
        with tf.variable_scope('for') as scope:
            for i in range(cfg.num_steps):
                yi, state = cell(y[:, i, :], state)  # [-1, 200]

                logits = tf.layers.dense(yi, cfg.num_chinese_chars(), name='dense2')
                y_predict.append(logits)

                if i < cfg.num_steps - 1:
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=x[:, i+1, :], logits=logits)
                    losses.append(loss)
                scope.reuse_variables()

        self.losses = [tf.reduce_mean(losses)]


if __name__ == '__main__':
    cfg = MyConfig()
    cfg.from_cmd()
