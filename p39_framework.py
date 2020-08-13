import argparse
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import os


def get_gpus():
    value = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    value = value.split(',')
    return len(value)


def make_dirs(path:str):
    pos = path.rfind(os.sep)
    if pos < 0:
        raise Exception('Can not find the directory from the path', path)
    path = path[:pos]
    os.makedirs(path, exist_ok=True)


class Config:
    def __init__(self):
        self.lr = 0.001
        self.epoches = 2000
        self.batch_size = 10
        self.save_path = '../models/{name}/{name}'.format(name=self.get_name())
        self.sample_path = None
        self.logdir = '../logs/{name}'.format(name=self.get_name())
        self.new_model = False
        self.gpus = get_gpus()

    def get_name(self):
        raise Exception('get_name() is not re-defined.')

    def __repr__(self):
        attrs = self.get_attrs()
        result = ['%s=%s' % (key, attrs[key]) for key in attrs]
        return ', '.join(result)

    def get_attrs(self):
        result = {}
        for attr in dir(self):
            if attr.startswith('_'):
                continue
            value = getattr(self, attr)
            if value is None or type(value) in (int, float, bool, str):
                result[attr] = value
        return result

    def from_cmd(self):
        parser = argparse.ArgumentParser()
        attrs = self.get_attrs()
        for attr in attrs:
            value = attrs[attr]
            t = type(value)
            if t == bool:
                parser.add_argument('--' + attr, default=value, help='Default to %s' % value,
                                    action='store_%s' % ('false' if value else 'true'))
            else:
                parser.add_argument('--' + attr, type=t, default=value, help='Default to %s' % value)
        a = parser.parse_args()
        for attr in attrs:
            setattr(self, attr, getattr(a, attr))

    def get_tensors(self):
        return Tensors()

    def get_app(self):
        return App(self)


class Tensors:
    def __init__(self):
        with tf.device('/gpu:0'):
            pass


class App:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            self.ts = config.get_tensors()
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True
            self.session = tf.Session(config=cfg, graph=graph)
            self.saver = tf.train.Saver()
            if config.new_model:
                self.session.run(tf.global_variables_initializer())
                print('Use a new empty model')
            else:
                try:
                    self.saver.restore(self.session, config.save_path)
                    print('Restore model from %s successfully!' % config.save_path)
                except:
                    print('Fail to restore model from %s, use a new empty model instead!!!!!!' % config.save_path)
                    self.session.run(tf.global_variables_initializer())

    def close(self):
        self.session.close()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def train(self, ds_train, ds_validation):
        cfg = self.config
        ts  = self.ts
        writer = tf.summary.FileWriter(cfg.logdir, self.session.graph)
        batches = ds_train.num_examples // cfg.batch_size

        for epoch in range(cfg.epoches):
            for batch in range(batches):
                _, summary = self.session.run([ts.train_op, ts.summary], self.get_feed_dict(ds_train))
                writer.add_summary(summary, epoch * batches + batch)
            print('Epoch:', epoch, flush=True)
            self.save()

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('Save model into', self.config.save_path, flush=True)

    def test(self, ds_test):
        pass

    def get_feed_dict(self, ds):
        values = ds.next_batch(self.config.batch_size)
        return {tensor: value for tensor, value in zip(self.ts.inputs, values)}


if __name__ == '__main__':
    cfg = Config()
    cfg.from_cmd()
    print('-' * 100)
    print(cfg)

    dss = read_data_sets(cfg.sample_path)

    app = cfg.get_app()
    with app:
        app.train(dss.train, dss.validation)
        app.test(dss.test)
