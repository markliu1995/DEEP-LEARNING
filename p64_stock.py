import p50_framework as myf
import tensorflow as tf
import numpy as np


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.num_steps = 10
        self.stocks = 8
        self.batch_size = 1
        self.days = 300
        self.state_size = 4

    def get_name(self):
        return 'p64'

    def get_ds_train(self):
        return MyDS(self.stocks, self.days, self.num_steps)

    def get_ds_test(self):
        return MyDS(self.stocks, self.days, self.num_steps)

    def get_sub_tensors(self, gpu_id):
        return MySubTensors(self)


class MyDS:
    def __init__(self, stocks, days, steps):
        self.data = np.random.normal(size=[stocks, days])
        self.num_examples = days - steps
        self.pos = np.random.randint(0, self.num_examples)
        self.steps = steps

    def next_batch(self, batch_size):
        next = self.pos + self.steps  # steps represent the quantity of data for each batch
        x = self.data[:, self.pos: next]  # [stocks, steps]
        y = self.data[:, next]  # [stocks]
        self.pos += 1
        if self.pos >= self.num_examples:
            self.pos = 0
        return x, y


class MySubTensors:
    def __init__(self, config: MyConfig):
        self.config = config
        x = tf.placeholder(tf.float32, [None, config.num_steps])
        y = tf.placeholder(tf.float32, [None])
        self.inputs = [x, y]

        cell = Cell(config.state_size)
        state = cell.zero_state(tf.shape(x)[0], x.dtype)
        with tf.variable_scope('my_cell') as scope:
            for i in range(config.num_steps):
                _, state = cell(x[:, i], state)  # _ the result of output for each loop which can be symbolized as
                # short-term memory,
                scope.reuse_variables()

        # state: [-1, state_size]

        y_predict = tf.layers.dense(state, 1, name='dense1')  # [-1, 1]
        y_predict = tf.reshape(y_predict, [-1])  # [-1]
        loss = tf.reduce_mean(tf.square(y_predict - y))
        self.losses = [loss]


class Cell:
    def __init__(self, num_units):
        self.num_units = num_units

    def __call__(self, xi, statei):
        # xi: [-1]
        # statei: [-1, state_size]
        xi = tf.reshape(xi, [-1, 1])
        x = tf.concat((xi, statei), axis=1)  # [-1, state_size + 1]
        x = tf.layers.dense(x, 400, name='dense', activation=tf.nn.relu)
        state = tf.layers.dense(x, statei.shape[-1].value, name='dense2')
        return None, state

    def zero_state(self, batch_size, dtype):
        return tf.zeros([batch_size, self.num_units], dtype)


if __name__ == '__main__':
    cfg = MyConfig()
    cfg.from_cmd()
